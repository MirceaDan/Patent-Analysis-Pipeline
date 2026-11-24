#!/usr/bin/env python3
"""Resumable patent-figure retrieval and VLM description pipeline.

Independent of the master-dataset builder and of the classifier. Reads
`master_dataset/` (never modifies it) and writes only to
`data/figure_enrichment/`.

Pipeline per family (annotated families only by default)
  1. order the family's publications: the master publication first (column
     `is_master_publication == "True"` in patent_publications.csv, which
     patent_families_master.csv's `Master Patent Number` defines), then others
     by a fixed country preference, then date, then number;
  2. for each publication build Google-Patents document ids from
     (number, kind code) -> fetch the document page -> VERIFY that the page's
     own publication number (and publication date, when both exist) equal the
     requested publication -> list the figure images that belong to it;
  3. stop at the first publication that has figures; every attempted
     publication is logged in document_resolution.csv;
  4. (not in --dry-run) download each figure into memory, checksum it, send it
     to an OpenAI-compatible vision endpoint, validate the structured JSON,
     store the description.  Images are never written to disk.

Output files (data/figure_enrichment/)
  figure_enrichments.csv   one row per discovered figure (status column)
  document_resolution.csv  one row per (family, publication) attempt
  family_summary.csv       one row per selected family
  figures.jsonl / resolutions.jsonl   append-only journals used for resume
  run.log

Resume = re-run the same command: terminal results are read from the
journals; transient failures are retried up to --max-attempts.

Status vocabularies
  document: resolved_with_figures | resolved_no_figures | document_not_found |
            document_inaccessible | identifier_mismatch | unsupported_identifier
  figure:   pending | described | image_inaccessible | vlm_failed |
            skipped_limit
  family:   not_yet_processed | document_not_found | document_inaccessible |
            no_figures_found | figures_inaccessible | vlm_failed |
            figures_discovered_pending_vlm | described
None of these statuses means "the patent lacks a capability".

VLM configuration (environment; keys are never logged or written):
  PATENT_VLM_BASE_URL   e.g. https://host/v1   (OpenAI-compatible)
  PATENT_VLM_MODEL      model / deployment name
  PATENT_VLM_API_KEY    bearer token (variable name overridable: --api-key-env)
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import logging
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

PIPELINE_VERSION = "1.0"
PROMPT_VERSION = "fig-desc-v1"
ROOT = Path(__file__).resolve().parent.parent
USER_AGENT = ("patent-figure-enrichment/1.0 (academic research; low-rate, "
              "sequential requests)")
log = logging.getLogger("figure_enrichment")

DOC_TERMINAL = {"resolved_with_figures", "resolved_no_figures",
                "document_not_found", "identifier_mismatch",
                "unsupported_identifier"}
FIG_FAILED = {"image_inaccessible", "vlm_failed"}

FIGURE_COLUMNS = [  # first 14 equal master_dataset/figure_enrichments.csv
    "family_number", "publication_number", "kind_code", "figure_id",
    "figure_description", "image_source_url", "source_document_url",
    "source_document_id", "vlm_model", "vlm_model_version", "prompt_version",
    "processing_status", "error_details", "processed_at_utc",
    # extensions
    "image_sha256", "image_bytes", "figure_callout_labels",
    "publication_role", "match_verification", "vlm_config_hash",
    "vlm_prompt_tokens", "vlm_completion_tokens", "attempts",
    "reused_from", "description_json",
]
RESOLUTION_COLUMNS = [
    "family_number", "publication_number", "kind_code", "publication_role",
    "candidate_document_ids", "document_id", "document_url",
    "resolution_status", "http_status", "match_verification",
    "n_figures_listed", "document_pdf_url", "attempts", "error_message",
    "resolved_at_utc", "figures_json",
]
SUMMARY_COLUMNS = [
    "family_number", "family_status", "n_publications_examined",
    "publications_examined", "publication_used", "n_figures_discovered",
    "n_figures_described", "n_figures_failed", "n_figures_pending",
    "n_figures_skipped_limit", "has_usable_description", "failure_reason",
]


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------ identifier handling
US_KIND_MAP = {"AA": ["A1"], "AB": ["A2"], "AC": ["A9"],
               "BA": ["B2", "B1"], "BB": ["B2", "B1"]}
KIND_ALIASES = {"JP": {"B": ["B2", "B1", "B"]}, "KR": {"B": ["B1"]}}
COUNTRY_RANK = {c: i for i, c in enumerate(
    ["US", "EP", "WO", "DE", "KR", "JP", "GB", "FR", "CA", "AU", "IN"])}
MAX_CANDIDATES = 3


def normalize_publication(country: str, number: str, kind: str):
    """Return (google_number, [kind candidates]) or (None, []).

    `number` is the PatBase form including the country prefix (e.g.
    "US2020334401", "WO23247172", "CN118709918"). The mapping is
    deterministic; correctness is established by page verification, never
    assumed from the mapping.
    """
    number = re.sub(r"\s+", "", number or "").upper()
    kind = re.sub(r"\s+", "", kind or "").upper()
    country = (country or number[:2]).upper()
    if not number.startswith(country) or len(number) <= len(country) or not kind:
        return None, []
    digits = number[len(country):]
    if country == "US":
        m = re.fullmatch(r"(\d{4})(\d{6})", digits)
        if m:                                     # application publication
            number = f"US{m.group(1)}0{m.group(2)}"
    elif country == "WO":
        m = re.fullmatch(r"(\d{2})(\d{6})", digits)
        if m:                                     # WOyy nnnnnn -> WOyyyynnnnnn
            yy = int(m.group(1))
            number = f"WO{2000 + yy if yy <= 70 else 1900 + yy}{m.group(2)}"
    if country == "US" and kind in US_KIND_MAP:
        kinds = list(US_KIND_MAP[kind])
    elif kind in KIND_ALIASES.get(country, {}):
        kinds = list(KIND_ALIASES[country][kind])
    elif country == "US" and not re.fullmatch(r"[AB]\d", kind):
        kinds = [kind, "A1", "B2", "B1"]
    else:
        kinds = [kind]
    kinds = list(dict.fromkeys(kinds))[:MAX_CANDIDATES]
    return number, kinds


def candidate_document_ids(country, number, kind):
    n, kinds = normalize_publication(country, number, kind)
    return [f"{n}{k}" for k in kinds] if n else []


def document_url(doc_id: str) -> str:
    return f"https://patents.google.com/patent/{doc_id}/en"


def order_publications(pubs):
    """Master publication first; then country preference, date, number."""
    def key(p):
        return (p["is_master_publication"] != "True",
                COUNTRY_RANK.get(p["country_code"], 99),
                p["publication_date"], p["publication_number"],
                p["kind_code"])
    return sorted(pubs, key=key)


# ------------------------------------------------------------ page parsing
def parse_document_page(html: str) -> dict | None:
    """Parse a Google Patents document page.

    Returns None when the page does not look like a patent document (e.g.
    an interstitial / consent / CAPTCHA page).  Figures are the
    `li[itemprop=images]` entries whose full-size URL file name starts with
    the page's own document id (so images of other documents are ignored).
    """
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    # applications use ..._publication_number, granted patents ..._number
    meta = (soup.find("meta", attrs={"name": "citation_patent_publication_number"})
            or soup.find("meta", attrs={"name": "citation_patent_number"}))
    if not meta or not meta.get("content"):
        return None
    parts = meta["content"].split(":")
    if len(parts) == 3:
        doc_id = "".join(parts)
    elif len(parts) == 2:  # some grants omit the kind: take it from canonical URL
        canon = soup.find("link", attrs={"rel": "canonical"})
        m = re.search(r"/patent/([A-Z0-9]+)/", canon.get("href", "")) if canon else None
        if not m or not m.group(1).startswith("".join(parts)):
            return None
        doc_id = m.group(1)
    else:
        return None
    date = None
    for m in soup.find_all("meta", attrs={"name": "DC.date"}):
        if m.get("scheme") in (None, "", "issue"):  # publication / grant date
            date = m.get("content")
    app = soup.find("meta", attrs={"name": "citation_patent_application_number"})
    figures = []
    for li in soup.find_all("li", attrs={"itemprop": "images"}):
        full = li.find("meta", attrs={"itemprop": "full"})
        if not full or not full.get("content"):
            continue
        url = full["content"]
        fname = url.rsplit("/", 1)[-1]
        # file names carry the document id with or without the kind code
        if not fname.startswith((doc_id, "".join(parts[:2]) + "-")):
            continue
        m = re.search(r"-(D\d{5})\.", fname)
        fig_id = m.group(1) if m else re.sub(r"\.[A-Za-z0-9]+$", "", fname)
        labels = []
        for c in li.find_all("li", attrs={"itemprop": "callouts"}):
            lab = c.find("meta", attrs={"itemprop": "label"})
            cid = c.find("meta", attrs={"itemprop": "id"})
            if lab and lab.get("content"):
                labels.append(f"{cid['content'] if cid else ''}:{lab['content']}")
        figures.append({"figure_id": fig_id, "url": url,
                        "callout_labels": labels})
    pdf = soup.find("meta", attrs={"name": "citation_pdf_url"})
    return {"document_id": doc_id, "publication_date": date,
            "pdf_url": pdf["content"] if pdf and pdf.get("content") else "",
            "application_number": app["content"] if app else None,
            "figures": figures}


def verify_document(page: dict, requested_id: str, dataset_pub_date: str):
    """Return (ok, verification_label)."""
    if page["document_id"] != requested_id:
        return False, f"page_id_{page['document_id']}_!=_{requested_id}"
    page_date = (page.get("publication_date") or "").replace("-", "")
    if page_date and dataset_pub_date:
        if page_date != dataset_pub_date:
            return False, f"pub_date_{page_date}_!=_{dataset_pub_date}"
        return True, "pub_number+pub_date"
    return True, "pub_number_only"


# --------------------------------------------------------------- HTTP layer
class TransientHttpError(Exception):
    pass


class RateLimiter:
    """Minimum delay between requests per host (thread-safe)."""

    def __init__(self, delays: dict, default=1.0, sleep=time.sleep,
                 clock=time.monotonic):
        self.delays, self.default = delays, default
        self._next, self._lock = {}, threading.Lock()
        self._sleep, self._clock = sleep, clock

    def wait(self, url):
        host = urlparse(url).netloc
        d = self.delays.get(host, self.default)
        with self._lock:
            now = self._clock()
            start = max(now, self._next.get(host, 0.0))
            self._next[host] = start + d
        if start > now:
            self._sleep(start - now)


def robots_allows(robots_text: str, path: str, agent="patent-figure-enrichment"):
    """RFC 9309 matching: wildcard '*' and '$', longest matching rule wins,
    Allow wins ties. ( mishandles 'Disallow: /*'.)"""
    groups, cur, in_rules = {}, [], False
    for line in robots_text.splitlines():
        line = line.split("#", 1)[0].strip()
        if ":" not in line:
            continue
        k, v = (x.strip() for x in line.split(":", 1))
        k = k.lower()
        if k == "user-agent":
            if in_rules:
                cur, in_rules = [], False
            cur.append(v.lower())
            groups.setdefault(v.lower(), [])
        elif k in ("allow", "disallow"):
            in_rules = True
            for ua in cur:
                groups[ua].append((k, v))
    rules = groups.get(agent.lower(), groups.get("*", []))
    best_len, allowed = -1, True
    for kind, pat in rules:
        if not pat:
            continue
        rx = re.escape(pat).replace(r"\*", ".*")
        if rx.endswith(r"\$"):
            rx = rx[:-2] + "$"
        if re.match(rx, path) and (len(pat) > best_len or (
                len(pat) == best_len and kind == "allow")):
            best_len, allowed = len(pat), kind == "allow"
    return allowed


class RequestsHttp:
    """GET with timeout, size cap and robots check (no auth, no cookies)."""

    def __init__(self, limiter, timeout=30, max_bytes=8_000_000):
        import requests
        self.requests = requests
        self.s = requests.Session()
        self.s.headers["User-Agent"] = USER_AGENT
        self.limiter, self.timeout, self.max_bytes = limiter, timeout, max_bytes
        self._robots, self._lock = {}, threading.Lock()

    def allowed(self, url):
        p = urlparse(url)
        base = f"{p.scheme}://{p.netloc}"
        with self._lock:
            if base not in self._robots:
                try:
                    r = self.s.get(base + "/robots.txt", timeout=self.timeout)
                    # unreadable/absent robots.txt: no restriction can be read
                    self._robots[base] = r.text if r.status_code == 200 else None
                except self.requests.RequestException:
                    self._robots[base] = None
            txt = self._robots[base]
        return True if txt is None else robots_allows(
            txt, p.path + (f"?{p.query}" if p.query else ""))

    def get(self, url, head=False):
        if not self.allowed(url):
            return 403, {"x-robots": "disallowed"}, b""
        self.limiter.wait(url)
        try:
            if head:
                r = self.s.head(url, timeout=self.timeout, allow_redirects=True)
                return r.status_code, {k.lower(): v for k, v in r.headers.items()}, b""
            r = self.s.get(url, timeout=self.timeout, stream=True)
            body = b""
            for chunk in r.iter_content(65536):
                body += chunk
                if len(body) > self.max_bytes:
                    r.close()
                    return 413, {}, b""
            return r.status_code, {k.lower(): v for k, v in r.headers.items()}, body
        except self.requests.RequestException as e:
            raise TransientHttpError(type(e).__name__) from None


def get_with_retry(http, url, max_retries=3, backoff=2.0, sleep=time.sleep,
                   head=False):
    """Bounded retries on timeouts/429/5xx. Returns (status, headers, body)
    or raises TransientHttpError after the last attempt."""
    last = None
    for attempt in range(max_retries + 1):
        try:
            status, headers, body = http.get(url, head=head)
            if status not in (429, 500, 502, 503, 504):
                return status, headers, body
            last = TransientHttpError(f"HTTP {status}")
            ra = headers.get("retry-after", "")
            wait = min(float(ra), 60) if ra.isdigit() else backoff * 2 ** attempt
        except TransientHttpError as e:
            last, wait = e, backoff * 2 ** attempt
        if attempt < max_retries:
            sleep(wait)
    raise last


# ------------------------------------------------------------------- VLM
SYSTEM_PROMPT = (
    "You describe technical patent drawings. Report only what is visible in "
    "the image. Separate visible evidence from inference. Do not invent "
    "components, labels or functions; if something is illegible or cannot be "
    "determined from the image alone, say so in `uncertainties`. Do not "
    "classify the patent and do not judge what the invention is for beyond "
    "what the figure itself shows. Reply with a single JSON object only."
)
USER_PROMPT = """Describe this patent figure (id: {figure_id}).
Return JSON with exactly these keys:
{{
 "figure_type": "e.g. block diagram, flowchart, schematic, plot, screenshot, photograph, other",
 "purpose": "what the figure appears to illustrate, one sentence",
 "components": [{{"label": "reference numeral or text label, or null", "name": "component/entity as shown", "evidence": "visible" or "inferred"}}],
 "relations": [{{"source": "...", "target": "...", "relation": "connection, data flow, arrow direction, spatial relation", "evidence": "visible" or "inferred"}}],
 "process_or_mechanism": "process, control loop or technical relationship shown, or null",
 "text_in_figure": ["legible labels, axis titles, legend entries"],
 "uncertainties": ["illegible labels, ambiguous arrows, anything not inferable from the image"],
 "description": "concise technical description (2-6 sentences) of the visible content"
}}"""

EVIDENCE_VALUES = {"visible", "inferred"}


class VlmError(Exception):
    def __init__(self, msg, permanent=False):
        super().__init__(msg)
        self.permanent = permanent


def validate_description(obj) -> dict:
    """Strict schema check. Raises ValueError on any violation."""
    if not isinstance(obj, dict):
        raise ValueError("output is not a JSON object")
    req_str = ["figure_type", "purpose", "description"]
    for k in req_str:
        if not isinstance(obj.get(k), str) or not obj[k].strip():
            raise ValueError(f"missing or empty string field {k!r}")
    if len(obj["description"]) > 4000:
        raise ValueError("description too long")
    if obj.get("process_or_mechanism") is not None and not isinstance(
            obj["process_or_mechanism"], str):
        raise ValueError("process_or_mechanism must be string or null")
    for k in ("components", "relations", "text_in_figure", "uncertainties"):
        if not isinstance(obj.get(k), list):
            raise ValueError(f"{k!r} must be a list")
    for c in obj["components"]:
        if not isinstance(c, dict) or not isinstance(c.get("name"), str):
            raise ValueError("component needs a string 'name'")
        if c.get("evidence") not in EVIDENCE_VALUES:
            raise ValueError("component evidence must be visible|inferred")
        if c.get("label") is not None and not isinstance(c["label"], str):
            raise ValueError("component label must be string or null")
    for r in obj["relations"]:
        if not isinstance(r, dict) or not all(
                isinstance(r.get(k), str) for k in ("source", "target", "relation")):
            raise ValueError("relation needs string source/target/relation")
        if r.get("evidence") not in EVIDENCE_VALUES:
            raise ValueError("relation evidence must be visible|inferred")
    for k in ("text_in_figure", "uncertainties"):
        if not all(isinstance(x, str) for x in obj[k]):
            raise ValueError(f"{k!r} must contain strings")
    forbidden = {"mpri", "m/p/r/i", "category", "classification", "label"}
    if forbidden & {k.lower() for k in obj}:
        raise ValueError("output contains classification-like keys")
    return obj


def parse_vlm_text(text: str) -> dict:
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return validate_description(json.loads(t))
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON: {e.msg}") from None


class OpenAICompatibleVLM:
    """Chat-completions client with image input. Config from environment."""

    def __init__(self, base_url, model, api_key, timeout=120, max_tokens=1200,
                 json_mode=True, max_retries=3, backoff=4.0,
                 sleep=time.sleep, session=None):
        import requests
        self.base_url, self.model = base_url.rstrip("/"), model
        self.api_key, self.timeout = api_key, timeout
        self.max_tokens, self.json_mode = max_tokens, json_mode
        self.max_retries, self.backoff, self.sleep = max_retries, backoff, sleep
        self.requests = requests
        self.session = session or requests.Session()

    @classmethod
    def from_env(cls, api_key_env="PATENT_VLM_API_KEY", **kw):
        base, model = os.environ.get("PATENT_VLM_BASE_URL"), os.environ.get(
            "PATENT_VLM_MODEL")
        key = os.environ.get(api_key_env)
        missing = [n for n, v in (("PATENT_VLM_BASE_URL", base),
                                  ("PATENT_VLM_MODEL", model),
                                  (api_key_env, key)) if not v]
        if missing:
            raise VlmError("VLM not configured; missing environment "
                           f"variables: {', '.join(missing)}", permanent=True)
        return cls(base, model, key, **kw)

    @property
    def config_hash(self):
        cfg = {"model": self.model, "prompt": PROMPT_VERSION,
               "max_tokens": self.max_tokens, "json_mode": self.json_mode,
               "temperature": 0, "image_prep": IMAGE_PREP_VERSION}
        return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()
                              ).hexdigest()[:16]

    def describe(self, image_bytes: bytes, mime: str, figure_id: str):
        """Return (validated_dict, usage_dict). Retries transient errors and
        malformed outputs up to max_retries; raises VlmError afterwards."""
        body = {
            "model": self.model, "temperature": 0, "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": USER_PROMPT.format(figure_id=figure_id)},
                    {"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (
                        mime, base64.b64encode(image_bytes).decode())}}]}],
        }
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        last = "unknown"
        for attempt in range(self.max_retries + 1):
            wait = self.backoff * 2 ** attempt
            try:
                r = self.session.post(
                    self.base_url + "/chat/completions", json=body,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=self.timeout)
                if r.status_code in (429, 500, 502, 503, 504):
                    last = f"HTTP {r.status_code}"
                    ra = r.headers.get("Retry-After", "")
                    wait = min(float(ra), 60) if ra.isdigit() else wait
                elif r.status_code != 200:
                    raise VlmError(f"HTTP {r.status_code}", permanent=True)
                else:
                    try:
                        data = r.json()
                        text = data["choices"][0]["message"]["content"]
                        usage = data.get("usage") or {}
                        return parse_vlm_text(text), usage
                    except (ValueError, KeyError, IndexError, TypeError) as e:
                        last = f"invalid_output: {e}"
            except self.requests.RequestException as e:
                last = type(e).__name__
            if attempt < self.max_retries:
                self.sleep(wait)
        raise VlmError(f"gave up after {self.max_retries + 1} attempts: {last}")


# ------------------------------------------------------------------- images
IMAGE_PREP_VERSION = "p1-max2000-png"


def prepare_image(raw: bytes, max_side=2000):
    """Validate and normalise an image; returns (mime, bytes, width, height).
    Raises ValueError for unreadable images."""
    from PIL import Image
    try:
        im = Image.open(io.BytesIO(raw))
        im.load()
    except Exception as e:  # PIL raises many types
        raise ValueError(f"unreadable image: {type(e).__name__}") from None
    w, h = im.size
    if im.mode not in ("L", "RGB"):
        im = im.convert("RGB" if im.mode in ("RGBA", "P", "CMYK") else "L")
    if max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))))
    out = io.BytesIO()
    im.save(out, "PNG")
    return "image/png", out.getvalue(), w, h


# --------------------------------------------------------------------- state
class State:
    """Journal-backed state. Latest journal line per key wins; torn last
    lines (interrupted writes) are ignored."""

    def __init__(self, out_dir: Path):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        for tmp in self.dir.glob("*.tmp"):   # abandoned temp files
            tmp.unlink()
        self.lock = threading.RLock()
        self.res_path = self.dir / "resolutions.jsonl"
        self.fig_path = self.dir / "figures.jsonl"
        self.resolutions = self._load(self.res_path,
                                      lambda r: (r["family_number"], r["publication_number"], r["kind_code"]))
        self.figures = self._load(self.fig_path,
                                  lambda r: (r["family_number"], r["source_document_id"], r["figure_id"]))
        self._dirty = 0

    @staticmethod
    def _load(path, keyfn):
        out = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                    out[keyfn(r)] = r
                except (json.JSONDecodeError, KeyError):
                    continue
        return out

    def _append(self, path, rec):
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def put_resolution(self, r):
        with self.lock:
            self.resolutions[(r["family_number"], r["publication_number"],
                              r["kind_code"])] = r
            self._append(self.res_path, r)

    def put_figure(self, r):
        with self.lock:
            self.figures[(r["family_number"], r["source_document_id"],
                          r["figure_id"])] = r
            self._append(self.fig_path, r)
            self._dirty += 1

    def export_csvs(self, families=None, summaries=None):
        with self.lock:
            self._write(self.dir / "figure_enrichments.csv", FIGURE_COLUMNS,
                        sorted(self.figures.values(), key=lambda r: (
                            r["family_number"], r["source_document_id"], r["figure_id"])))
            self._write(self.dir / "document_resolution.csv", RESOLUTION_COLUMNS,
                        sorted(self.resolutions.values(), key=lambda r: (
                            r["family_number"], r["publication_role"] != "master",
                            r["publication_number"], r["kind_code"])))
            if summaries is not None:
                self._write(self.dir / "family_summary.csv", SUMMARY_COLUMNS,
                            summaries)

    def _write(self, path, cols, rows):
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n",
                               extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in cols})
        os.replace(tmp, path)


# ----------------------------------------------------------------- pipeline
class Pipeline:
    def __init__(self, state, http, vlm=None, dry_run=False,
                 max_publications=4, max_figures=10, max_attempts=3,
                 max_image_bytes=8_000_000, max_retries=3, backoff=2.0,
                 sleep=time.sleep, abort_after_vlm_failures=5):
        self.state, self.http, self.vlm = state, http, vlm
        self.dry_run = dry_run
        self.max_publications, self.max_figures = max_publications, max_figures
        self.max_attempts, self.max_image_bytes = max_attempts, max_image_bytes
        self.max_retries, self.backoff, self.sleep = max_retries, backoff, sleep
        self.abort_after = abort_after_vlm_failures
        self._consec_vlm_fail = 0
        self.stop = threading.Event()
        self.counts = {k: 0 for k in (
            "families", "described", "skipped_resume", "failed",
            "figures_discovered", "documents_resolved")}
        self.cl = threading.Lock()

    def _inc(self, k, n=1):
        with self.cl:
            self.counts[k] += n

    # -- document resolution
    def resolve_publication(self, family, pub, role):
        key = (family, pub["publication_number"], pub["kind_code"])
        prev = self.state.resolutions.get(key)
        if prev:
            if prev["resolution_status"] in DOC_TERMINAL:
                return prev
            if int(prev.get("attempts") or 0) >= self.max_attempts:
                return prev
        attempts = int(prev["attempts"]) + 1 if prev else 1
        base = {"family_number": family, "publication_number": pub["publication_number"],
                "kind_code": pub["kind_code"], "publication_role": role,
                "candidate_document_ids": "", "document_id": "",
                "document_url": "", "http_status": "", "match_verification": "",
                "n_figures_listed": 0, "document_pdf_url": "",
                "attempts": attempts,
                "error_message": "", "resolved_at_utc": utcnow(),
                "figures_json": ""}
        cands = candidate_document_ids(pub["country_code"],
                                       pub["publication_number"], pub["kind_code"])
        base["candidate_document_ids"] = "; ".join(cands)
        if not cands:
            base["resolution_status"] = "unsupported_identifier"
            base["error_message"] = "cannot derive a document id"
            self.state.put_resolution(base)
            return base
        notfound, mismatches, errors = 0, [], []
        for cid in cands:
            url = document_url(cid)
            try:
                status, _, body = get_with_retry(
                    self.http, url, self.max_retries, self.backoff, self.sleep)
            except TransientHttpError as e:
                errors.append(f"{cid}: {e}")
                continue
            base["http_status"] = status
            if status == 404:
                notfound += 1
                continue
            if status != 200:
                errors.append(f"{cid}: HTTP {status}")
                continue
            page = parse_document_page(body.decode("utf-8", "replace"))
            if page is None:
                errors.append(f"{cid}: not a patent document page")
                continue
            ok, ver = verify_document(page, cid, pub["publication_date"])
            if not ok:
                mismatches.append(f"{cid}: {ver}")
                continue
            base.update(document_id=cid, document_url=url, match_verification=ver,
                        n_figures_listed=len(page["figures"]),
                        document_pdf_url=page["pdf_url"],
                        figures_json=json.dumps(page["figures"], ensure_ascii=False),
                        resolution_status=("resolved_with_figures" if page["figures"]
                                           else "resolved_no_figures"))
            self.state.put_resolution(base)
            self._inc("documents_resolved")
            return base
        if errors:
            base["resolution_status"] = "document_inaccessible"
            base["error_message"] = "; ".join(errors + mismatches)[:500]
        elif mismatches:
            base["resolution_status"] = "identifier_mismatch"
            base["error_message"] = "; ".join(mismatches)[:500]
        else:
            base["resolution_status"] = "document_not_found"
            base["error_message"] = f"HTTP 404 for {notfound} candidate id(s)"
        self.state.put_resolution(base)
        return base

    # -- figures
    def _figure_row(self, family, pub, res, fig, status, **kw):
        row = {
            "family_number": family, "publication_number": pub["publication_number"],
            "kind_code": pub["kind_code"], "figure_id": fig["figure_id"],
            "figure_description": "", "image_source_url": fig["url"],
            "source_document_url": res["document_url"],
            "source_document_id": res["document_id"],
            "vlm_model": "", "vlm_model_version": "", "prompt_version": "",
            "processing_status": status, "error_details": "",
            "processed_at_utc": utcnow(), "image_sha256": "", "image_bytes": "",
            "figure_callout_labels": " | ".join(fig.get("callout_labels", [])),
            "publication_role": res["publication_role"],
            "match_verification": res["match_verification"],
            "vlm_config_hash": "", "vlm_prompt_tokens": "",
            "vlm_completion_tokens": "", "attempts": 0, "reused_from": "",
            "description_json": ""}
        row.update(kw)
        return row

    def _describe_figure(self, family, pub, res, fig, probe=False):
        key = (family, res["document_id"], fig["figure_id"])
        prev = self.state.figures.get(key)
        cfg = self.vlm.config_hash if self.vlm else ""
        if prev and prev["processing_status"] == "described" and \
                prev["vlm_config_hash"] == cfg and \
                prev["image_source_url"] == fig["url"]:
            self._inc("skipped_resume")
            return prev
        attempts = int(prev["attempts"]) if prev else 0
        if prev and prev["processing_status"] in FIG_FAILED and \
                attempts >= self.max_attempts:
            return prev
        if self.dry_run:
            if prev:
                return prev
            row = self._figure_row(family, pub, res, fig, "pending")
            if probe:  # accessibility check of the first figure (HEAD only)
                try:
                    st, hd, _ = get_with_retry(
                        self.http, fig["url"], self.max_retries, self.backoff,
                        self.sleep, head=True)
                    row["error_details"] = f"dry_run_head_status={st}"
                    if st != 200:
                        row["processing_status"] = "image_inaccessible"
                except TransientHttpError as e:
                    row.update(processing_status="image_inaccessible",
                               error_details=f"dry_run_head_error={e}")
            self.state.put_figure(row)
            return row
        attempts += 1
        row = self._figure_row(family, pub, res, fig, "image_inaccessible",
                               attempts=attempts)
        try:
            status, _, raw = get_with_retry(
                self.http, fig["url"], self.max_retries, self.backoff, self.sleep)
            if status != 200 or not raw:
                raise ValueError(f"HTTP {status}")
            if len(raw) > self.max_image_bytes:
                raise ValueError("image too large")
            mime, data, w, h = prepare_image(raw)
        except (TransientHttpError, ValueError) as e:
            row["error_details"] = str(e)[:300]
            self.state.put_figure(row)
            self._inc("failed")
            return row
        sha = hashlib.sha256(raw).hexdigest()
        row.update(image_sha256=sha, image_bytes=len(raw))
        # identical image (checksum) already described with this config?
        with self.state.lock:
            twin = next((r for r in self.state.figures.values()
                         if r["image_sha256"] == sha and r["processing_status"] == "described"
                         and r["vlm_config_hash"] == cfg), None)
        if twin:
            row.update({k: twin[k] for k in (
                "figure_description", "vlm_model", "vlm_model_version",
                "prompt_version", "vlm_config_hash", "description_json")})
            row.update(processing_status="described", error_details="",
                       reused_from=f"{twin['family_number']}/{twin['source_document_id']}/{twin['figure_id']}")
            self.state.put_figure(row)
            self._inc("described")
            return row
        try:
            desc, usage = self.vlm.describe(data, mime, fig["figure_id"])
        except VlmError as e:
            row.update(processing_status="vlm_failed", error_details=str(e)[:300],
                       vlm_model=self.vlm.model, prompt_version=PROMPT_VERSION,
                       vlm_config_hash=cfg)
            self.state.put_figure(row)
            self._inc("failed")
            with self.cl:
                self._consec_vlm_fail += 1
                if e.permanent or self._consec_vlm_fail >= self.abort_after:
                    self.stop.set()
            return row
        with self.cl:
            self._consec_vlm_fail = 0
        row.update(processing_status="described",
                   figure_description=desc["description"],
                   description_json=json.dumps(desc, ensure_ascii=False, sort_keys=True),
                   vlm_model=self.vlm.model, vlm_model_version=getattr(self.vlm, "model_version", ""),
                   prompt_version=PROMPT_VERSION, vlm_config_hash=cfg,
                   vlm_prompt_tokens=usage.get("prompt_tokens", ""),
                   vlm_completion_tokens=usage.get("completion_tokens", ""))
        self.state.put_figure(row)
        self._inc("described")
        return row

    def process_family(self, family, pubs):
        if self.stop.is_set():
            return
        self._inc("families")
        ordered = order_publications(pubs)[: self.max_publications]
        for i, pub in enumerate(ordered):
            role = "master" if pub["is_master_publication"] == "True" else "fallback"
            res = self.resolve_publication(family, pub, role)
            if res["resolution_status"] != "resolved_with_figures":
                continue
            figs = json.loads(res["figures_json"])
            self._inc("figures_discovered", len(figs))
            for j, fig in enumerate(figs):
                if self.stop.is_set():
                    return
                if j >= self.max_figures:
                    key = (family, res["document_id"], fig["figure_id"])
                    if key not in self.state.figures:
                        self.state.put_figure(self._figure_row(
                            family, pub, res, fig, "skipped_limit",
                            error_details=f"max_figures={self.max_figures}"))
                    continue
                try:
                    self._describe_figure(family, pub, res, fig, probe=(j == 0))
                except Exception as e:  # per-figure isolation
                    log.exception("figure error %s %s", family, fig["figure_id"])
                    self.state.put_figure(self._figure_row(
                        family, pub, res, fig, "vlm_failed",
                        error_details=f"unexpected: {type(e).__name__}"))
            return  # first publication with figures is used

    # -- reporting
    def summarize(self, families_pubs):
        rows = []
        for fam, pubs in families_pubs.items():
            ordered = order_publications(pubs)[: self.max_publications]
            res = [self.state.resolutions.get((fam, p["publication_number"], p["kind_code"]))
                   for p in ordered]
            res = [r for r in res if r]
            figs = [f for f in self.state.figures.values() if f["family_number"] == fam]
            n_desc = sum(f["processing_status"] == "described" for f in figs)
            n_fail = sum(f["processing_status"] in FIG_FAILED for f in figs)
            n_pend = sum(f["processing_status"] == "pending" for f in figs)
            n_skip = sum(f["processing_status"] == "skipped_limit" for f in figs)
            used = next((r["publication_number"] + " " + r["kind_code"] for r in res
                         if r["resolution_status"] == "resolved_with_figures"), "")
            statuses = [r["resolution_status"] for r in res]
            reason = ""
            if not res:
                st = "not_yet_processed"
            elif n_desc:
                st = "described"
            elif figs and n_fail and any(f["processing_status"] == "vlm_failed" for f in figs):
                st, reason = "vlm_failed", next(f["error_details"] for f in figs
                                                if f["processing_status"] == "vlm_failed")
            elif figs and n_fail:
                st, reason = "figures_inaccessible", next(
                    f["error_details"] for f in figs if f["processing_status"] == "image_inaccessible")
            elif n_pend:
                st = "figures_discovered_pending_vlm"
            elif "resolved_with_figures" in statuses:
                st = "not_yet_processed"
            elif "resolved_no_figures" in statuses:
                st, reason = "no_figures_found", "document found but lists no figure images"
            elif "document_inaccessible" in statuses:
                st = "document_inaccessible"
                reason = next(r["error_message"] for r in res
                              if r["resolution_status"] == "document_inaccessible")
            else:
                st = "document_not_found"
                reason = "; ".join(sorted(set(statuses)))
            rows.append({
                "family_number": fam, "family_status": st,
                "n_publications_examined": len(res),
                "publications_examined": "; ".join(
                    f"{r['publication_number']} {r['kind_code']}={r['resolution_status']}"
                    for r in res),
                "publication_used": used, "n_figures_discovered": len(figs),
                "n_figures_described": n_desc, "n_figures_failed": n_fail,
                "n_figures_pending": n_pend, "n_figures_skipped_limit": n_skip,
                "has_usable_description": str(n_desc > 0), "failure_reason": reason})
        return rows


# ---------------------------------------------------------------------- CLI
def load_publications(master_dir: Path, annotated_only=True):
    import pandas as pd
    pubs = pd.read_csv(master_dir / "patent_publications.csv", dtype=str,
                       keep_default_na=False)
    if annotated_only:
        ann = pd.read_csv(master_dir / "patent_families_annotated.csv", dtype=str,
                          keep_default_na=False,
                          usecols=["Family Number", "is_annotated"])
        fams = list(ann.loc[ann.is_annotated == "True", "Family Number"])
    else:
        fams = list(dict.fromkeys(pubs["family_number"]))
    grouped = {f: [] for f in fams}
    for rec in pubs.to_dict("records"):
        if rec["family_number"] in grouped:
            grouped[rec["family_number"]].append(rec)
    return grouped


def select_families(grouped, family_ids=None, sample=None, seed=0,
                    max_families=None):
    fams = list(grouped)
    if family_ids:
        fams = [f for f in fams if f in set(family_ids)]
    if sample:
        fams = sorted(random.Random(seed).sample(sorted(fams), min(sample, len(fams))))
    if max_families:
        fams = fams[:max_families]
    return {f: grouped[f] for f in fams}


def make_vlm(a):
    if a.vlm_backend == "api":
        return OpenAICompatibleVLM.from_env(a.api_key_env,
                                            json_mode=not a.no_json_mode)
    if not a.local_model:
        raise VlmError("local VLM not configured: set --local-model or "
                       "PATENT_VLM_LOCAL_MODEL", permanent=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from vlm_local import LocalTransformersVLM
    return LocalTransformersVLM(
        a.local_model, SYSTEM_PROMPT, USER_PROMPT, parse_vlm_text, VlmError,
        PROMPT_VERSION, IMAGE_PREP_VERSION, revision=a.local_revision,
        dtype=a.local_dtype, device=a.local_device, load_in_4bit=a.local_4bit)


def vlm_check(vlm):
    """Send a tiny synthetic drawing and validate the reply. Returns report."""
    from PIL import Image, ImageDraw
    im = Image.new("L", (400, 200), 255)
    d = ImageDraw.Draw(im)
    d.rectangle([20, 60, 140, 140], outline=0, width=3)
    d.rectangle([260, 60, 380, 140], outline=0, width=3)
    d.line([140, 100, 258, 100], fill=0, width=3)
    d.polygon([(258, 100), (246, 94), (246, 106)], fill=0)
    d.text((60, 95), "A", fill=0)
    d.text((300, 95), "B", fill=0)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    try:
        desc, usage = vlm.describe(buf.getvalue(), "image/png", "CHECK")
        return {"ok": True, "model": vlm.model, "usage": usage,
                "description": desc["description"]}
    except VlmError as e:
        return {"ok": False, "model": vlm.model, "error": str(e)}


def build_parser():
    p = argparse.ArgumentParser(
        description="Resumable patent figure retrieval + VLM description")
    p.add_argument("--master-dir", default=str(ROOT / "master_dataset"))
    p.add_argument("--out-dir", default=str(ROOT / "data" / "figure_enrichment"))
    p.add_argument("--all-families", action="store_true",
                   help="all families (default: annotated families only)")
    p.add_argument("--family-ids", help="comma-separated Family Numbers")
    p.add_argument("--sample", type=int, help="deterministic random sample size")
    p.add_argument("--seed", type=int, default=20261123)
    p.add_argument("--max-families", type=int)
    p.add_argument("--max-publications-per-family", type=int, default=4)
    p.add_argument("--max-figures-per-family", type=int, default=10)
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--page-delay", type=float, default=2.0,
                   help="seconds between requests to patents.google.com")
    p.add_argument("--image-delay", type=float, default=0.5,
                   help="seconds between requests to the image host")
    p.add_argument("--workers", type=int, default=1, help="1-4 families at once")
    p.add_argument("--timeout", type=float, default=30)
    p.add_argument("--dry-run", action="store_true",
                   help="discover documents/figures only; no image download, no VLM")
    p.add_argument("--summarize-only", action="store_true")
    p.add_argument("--vlm-check", action="store_true",
                   help="test the configured VLM on a synthetic image and exit")
    p.add_argument("--api-key-env", default="PATENT_VLM_API_KEY")
    p.add_argument("--no-json-mode", action="store_true")
    p.add_argument("--vlm-backend", choices=("api", "local"), default="api",
                   help="api: OpenAI-compatible endpoint; local: transformers "
                        "model via scripts/vlm_local.py")
    p.add_argument("--local-model", default=os.environ.get("PATENT_VLM_LOCAL_MODEL"),
                   help="HF model id for --vlm-backend local, e.g. Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--local-revision", default=os.environ.get("PATENT_VLM_LOCAL_REVISION"),
                   help="pin an HF commit/tag for reproducibility")
    p.add_argument("--local-dtype", default="bfloat16")
    p.add_argument("--local-device", default="auto")
    p.add_argument("--local-4bit", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if a.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(out / "run.log"), logging.StreamHandler()])
    vlm = None
    if a.vlm_check or not (a.dry_run or a.summarize_only):
        try:
            vlm = make_vlm(a)
        except VlmError as e:
            log.error("%s", e)
            print(f"BLOCKER: {e}", file=sys.stderr)
            return 2
    if a.vlm_check:
        rep = vlm_check(vlm)
        print(json.dumps(rep, indent=2, ensure_ascii=False))
        return 0 if rep["ok"] else 3

    grouped = load_publications(Path(a.master_dir), not a.all_families)
    fams = select_families(
        grouped, a.family_ids.split(",") if a.family_ids else None,
        a.sample, a.seed, a.max_families)
    state = State(out)
    limiter = RateLimiter({"patents.google.com": a.page_delay,
                           "patentimages.storage.googleapis.com": a.image_delay})
    http = RequestsHttp(limiter, timeout=a.timeout)
    pipe = Pipeline(state, http, vlm, dry_run=a.dry_run,
                    max_publications=a.max_publications_per_family,
                    max_figures=a.max_figures_per_family,
                    max_attempts=a.max_attempts, max_retries=a.max_retries)
    log.info("start families=%d dry_run=%s workers=%d", len(fams), a.dry_run,
             a.workers)
    try:
        if not a.summarize_only:
            with ThreadPoolExecutor(max_workers=max(1, min(a.workers, 4))) as ex:
                try:
                    futs = [ex.submit(pipe.process_family, fam, pubs)
                            for fam, pubs in fams.items()]
                    for n, f in enumerate(futs, 1):
                        f.result()
                        if n % 10 == 0:
                            log.info("progress %d/%d %s", n, len(fams), pipe.counts)
                            state.export_csvs(summaries=pipe.summarize(fams))
                except KeyboardInterrupt:
                    pipe.stop.set()   # queued families return immediately
                    raise
    except KeyboardInterrupt:
        pipe.stop.set()
        log.warning("interrupted; state is saved, rerun the same command to resume")
    finally:
        summ = pipe.summarize(fams)
        state.export_csvs(summaries=summ)
    from collections import Counter
    print(json.dumps({"counts": pipe.counts,
                      "family_status": dict(Counter(r["family_status"] for r in summ)),
                      "stopped_early": pipe.stop.is_set()}, indent=2))
    return 4 if pipe.stop.is_set() else 0


if __name__ == "__main__":
    sys.exit(main())
