"""Standalone client for the documented PatBase REST API (v2.0).

Reference: https://www.patbase.com/rest/2.0/PatBaseRestAPI.pdf
Only operations in that document are used:
  GET /rest/api.php?method=version | getweek     (no authorisation)
  GET /rest/api.php?method=login&userid=&password=   -> Set-Cookie SessionFarm_GUID
  GET /rest/api.php?method=getrecord&family=<PatBase family number>
        -> Family[].Publications[] with PN, KD, CC, PD, AN, AD, IMG (= image pages)
  GET /phpmosaic/getone.php?pn=<PN>&kd=<KD>&pg=<page>   (page image)
Authentication is either login (user id + password) or an API key. An e-mail
address alone does NOT authenticate. Credentials come from the environment or
an interactive prompt, never from source code or a CLI password argument:
  PATBASE_EMAIL (user id), PATBASE_PASSWORD, PATBASE_API_KEY.
Secrets (password, API key, session cookie) are never logged or printed; the
login URL carries the password in its query string (as documented), so request
URLs are never logged and exceptions are reduced to their class name.

Identifier note: the master dataset is a PatBase export, so `Family Number` is
the PatBase family number and `publication_number` + `kind_code` are PatBase
PN/KD. Dataset ids are therefore used unchanged.

Usage:
  PATBASE_EMAIL=... python scripts/patbase_client.py --check-access
  python scripts/patbase_client.py --family 123456 --fetch-figure --out-dir data/patbase_test
Exit codes: 0 authorised access, 1 not authorised / other failure, 2 usage.
"""
from __future__ import annotations

import argparse
import csv
import getpass
import io
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

BASE = "https://www.patbase.com"
API = BASE + "/rest/api.php"
IMG_URL = BASE + "/phpmosaic/getone.php"
ROOT = Path(__file__).resolve().parents[1]

# Access-check outcomes
OK = "authorized"
AUTH_REQUIRED = "authentication_required"
BAD_CREDENTIALS = "wrong_credentials"
API_NOT_ENABLED = "api_not_enabled_for_account"
FORBIDDEN = "insufficient_permissions"
NETWORK = "network_error"
TLS = "tls_error"
UNEXPECTED = "unexpected_or_undocumented_response"
NO_CREDENTIALS = "no_credentials_supplied"

# Error strings documented for the login method
_LOGIN_ERRORS = {
    "wrong userid and or password": BAD_CREDENTIALS,
    "missing userid and or password": NO_CREDENTIALS,
    "access to the api is not allowed for your account": API_NOT_ENABLED,
}


@dataclass
class Result:
    status: str
    detail: str = ""
    data: dict = field(default_factory=dict)

    @property
    def ok(self):
        return self.status == OK


class PatBaseError(Exception):
    def __init__(self, status, detail=""):
        super().__init__(f"{status}: {detail}")
        self.status, self.detail = status, detail


def classify_json(obj) -> str | None:
    """Map a documented {"ERROR": "..."} body to a status; None if no error."""
    if isinstance(obj, dict) and "ERROR" in obj:
        msg = str(obj["ERROR"]).strip().lower()
        for k, v in _LOGIN_ERRORS.items():
            if k in msg:
                return v
        if "login" in msg or "authori" in msg or "session" in msg:
            return AUTH_REQUIRED
        if "permission" in msg or "not allowed" in msg or "subscription" in msg:
            return FORBIDDEN
        return UNEXPECTED
    return None


class PatBaseClient:
    def __init__(self, email=None, password=None, api_key=None, session=None,
                 timeout=30, min_interval=1.0, ca_bundle=None, sleep=time.sleep):
        import requests
        self.requests = requests
        self._email, self._password, self._api_key = email, password, api_key
        self.session = session or requests.Session()
        if ca_bundle and session is None:
            self.session.verify = ca_bundle
        self.timeout, self.min_interval, self._sleep = timeout, min_interval, sleep
        self._last = 0.0
        self.logged_in = False

    # -- transport (single attempt; callers decide about retrying) --------
    def _get(self, url, params):
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            self._sleep(wait)
        self._last = time.monotonic()
        if self._api_key:
            params = dict(params, apikey=self._api_key)
        try:
            return self.session.get(url, params=params, timeout=self.timeout)
        except self.requests.exceptions.SSLError as e:
            raise PatBaseError(TLS, type(e).__name__) from None
        except self.requests.exceptions.RequestException as e:
            raise PatBaseError(NETWORK, type(e).__name__) from None

    def _json(self, resp):
        if resp.status_code in (401, 407):
            raise PatBaseError(AUTH_REQUIRED, f"HTTP {resp.status_code}")
        if resp.status_code == 403:
            raise PatBaseError(FORBIDDEN, "HTTP 403")
        if resp.status_code == 404:
            raise PatBaseError(UNEXPECTED, "HTTP 404 (operation not found)")
        if resp.status_code != 200:
            raise PatBaseError(UNEXPECTED, f"HTTP {resp.status_code}")
        try:
            obj = json.loads(resp.content.decode("utf-8-sig"))
        except ValueError:
            ctype = resp.headers.get("Content-Type", "")
            raise PatBaseError(UNEXPECTED, f"non-JSON response ({ctype.split(';')[0]})") from None
        err = classify_json(obj)
        if err:
            raise PatBaseError(err, "API returned an ERROR object")
        return obj

    # -- documented operations -------------------------------------------
    def version(self):
        return self._json(self._get(API, {"method": "version"}))

    def getweek(self):
        return self._json(self._get(API, {"method": "getweek"}))

    def login(self):
        if self._api_key:          # API-key access needs no login call
            return
        if not (self._email and self._password):
            raise PatBaseError(NO_CREDENTIALS,
                               "need PATBASE_EMAIL+PATBASE_PASSWORD or PATBASE_API_KEY")
        obj = self._json(self._get(API, {"method": "login", "userid": self._email,
                                         "password": self._password}))
        if obj.get("LOGIN_TO_API") != "OK":
            raise PatBaseError(UNEXPECTED, "login response lacks LOGIN_TO_API=OK")
        if not any(c.name == "SessionFarm_GUID" for c in self.session.cookies):
            raise PatBaseError(UNEXPECTED, "login OK but no SessionFarm_GUID cookie")
        self.logged_in = True

    def get_record(self, family_number):
        """Bibliographic family record: Family, PatentNumber, Publications[...]."""
        if not self.logged_in and not self._api_key:
            self.login()
        return self._json(self._get(API, {"method": "getrecord",
                                          "family": str(family_number)}))

    @staticmethod
    def publications(record) -> list[dict]:
        """Flatten getrecord output into publication dicts (PN, KD, CC, PD, AN,
        AD, IMG). Tolerates the documented list-of-families layout."""
        fams = record.get("Family") or record.get("Families") or []
        out = []
        for f in fams if isinstance(fams, list) else [fams]:
            for p in f.get("Publications", []):
                out.append({"family_number": str(f.get("Family", "")),
                            "publication_number": p.get("PN", ""),
                            "kind_code": p.get("KD", ""), "country": p.get("CC", ""),
                            "publication_date": p.get("PD", ""),
                            "application_number": p.get("AN", ""),
                            "image_pages": _to_int(p.get("IMG"))})
        return out

    def fetch_page_image(self, pn, kd, page):
        """Return (bytes, content_type). Raises PatBaseError; validates the image."""
        r = self._get(IMG_URL, {"pn": pn, "kd": kd, "pg": page})
        if r.status_code in (401, 403):
            raise PatBaseError(AUTH_REQUIRED if r.status_code == 401 else FORBIDDEN,
                               f"HTTP {r.status_code}")
        if r.status_code != 200:
            raise PatBaseError(UNEXPECTED, f"HTTP {r.status_code}")
        ctype = r.headers.get("Content-Type", "").split(";")[0].strip().lower()
        data = r.content
        if not validate_image(data):
            raise PatBaseError(UNEXPECTED, f"response is not a valid image (content-type {ctype or 'none'})")
        return data, ctype

    def fetch_figures(self, family_number, pn=None, kd=None, max_pages=1):
        """Structured per-page results for one family (preferring pn/kd, else the
        publications with the most image pages, grouped by IMG>0)."""
        rec = self.get_record(family_number)
        pubs = self.publications(rec)
        if not pubs:
            return [FigureResult(str(family_number), "", "", 0, "document_not_found",
                                 "family record has no publications")]
        want = [p for p in pubs if (not pn or p["publication_number"] == pn)
                and (not kd or p["kind_code"] == kd)]
        with_img = [p for p in want if p["image_pages"]]
        if not with_img:
            return [FigureResult(str(family_number), p["publication_number"],
                                 p["kind_code"], 0, "no_figures_found",
                                 "IMG=0 or absent") for p in want[:1]] or [
                FigureResult(str(family_number), pn or "", kd or "", 0,
                             "document_not_found", "requested publication not in family")]
        p = with_img[0]
        out = []
        for pg in range(1, min(p["image_pages"], max_pages) + 1):
            try:
                data, ctype = self.fetch_page_image(p["publication_number"], p["kind_code"], pg)
                out.append(FigureResult(str(family_number), p["publication_number"],
                                        p["kind_code"], pg, "retrieved", "", ctype,
                                        len(data), data))
            except PatBaseError as e:
                out.append(FigureResult(str(family_number), p["publication_number"],
                                        p["kind_code"], pg, e.status, e.detail))
        return out


@dataclass
class FigureResult:
    family_number: str
    publication_number: str
    kind_code: str
    page: int
    status: str            # retrieved | no_figures_found | document_not_found | <error status>
    error: str = ""
    content_type: str = ""
    n_bytes: int = 0
    data: bytes = field(default=b"", repr=False)

    def public(self):
        d = asdict(self)
        d.pop("data")
        return d


def _to_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def validate_image(data: bytes) -> bool:
    from PIL import Image
    try:
        Image.open(io.BytesIO(data)).verify()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------- diagnostics
def check_access(client: PatBaseClient, probe_family=None) -> Result:
    """Staged, single-attempt diagnostic. Stops at the first failing stage."""
    stages = {}
    try:
        stages["version"] = client.version()
        stages["getweek"] = client.getweek()
    except PatBaseError as e:
        return Result(e.status, f"unauthenticated reachability check failed: {e.detail}", stages)
    if not (client._api_key or (client._email and client._password)):
        return Result(NO_CREDENTIALS,
                      "API reachable (documented unauthenticated methods work), but no "
                      "credentials supplied. The API needs a PatBase API login (user id + "
                      "password with API access enabled) or an API key; an e-mail alone "
                      "cannot authenticate.", stages)
    try:
        client.login()
        stages["login"] = "OK"
        if probe_family:
            rec = client.get_record(probe_family)
            pubs = client.publications(rec)
            stages["getrecord_publications"] = len(pubs)
            if not pubs:
                return Result(UNEXPECTED, "authorised but getrecord returned no publications", stages)
    except PatBaseError as e:
        return Result(e.status, e.detail, stages)
    return Result(OK, "authenticated and authorised", stages)


def pick_probe(master_dir: Path, country="CN"):
    """First dataset family whose master publication is `country`."""
    with open(master_dir / "patent_publications.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["is_master_publication"] == "True" and r["country_code"] == country:
                return r["family_number"], r["publication_number"], r["kind_code"]
    return None


def credentials(args):
    email = args.email or os.environ.get("PATBASE_EMAIL")
    key = os.environ.get("PATBASE_API_KEY")
    pw = os.environ.get("PATBASE_PASSWORD")
    if email and not pw and not key and sys.stdin.isatty():
        pw = getpass.getpass("PatBase API password (not echoed): ") or None
    return email, pw, key


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--email", help="PatBase user id (or env PATBASE_EMAIL)")
    ap.add_argument("--check-access", action="store_true")
    ap.add_argument("--family", help="PatBase family number (default: first CN master family)")
    ap.add_argument("--fetch-figure", action="store_true")
    ap.add_argument("--max-pages", type=int, default=1)
    ap.add_argument("--master-dir", default=str(ROOT / "master_dataset"))
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "patbase_test"))
    ap.add_argument("--ca-bundle", default=os.environ.get("PATBASE_CA_BUNDLE")
                    or os.environ.get("REQUESTS_CA_BUNDLE"))
    a = ap.parse_args(argv)
    if not (a.check_access or a.fetch_figure):
        ap.print_usage()
        return 2
    email, pw, key = credentials(a)
    c = PatBaseClient(email, pw, key, ca_bundle=a.ca_bundle)
    probe = pick_probe(Path(a.master_dir))
    fam = a.family or (probe[0] if probe else None)
    res = check_access(c, fam)
    print(json.dumps({"status": res.status, "detail": res.detail, "stages": res.data},
                     indent=2, default=str))
    if not res.ok or not a.fetch_figure:
        return 0 if res.ok else 1
    figs = c.fetch_figures(fam, max_pages=a.max_pages)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in figs:
        row = f.public()
        if f.status == "retrieved":
            name = f"{f.publication_number}_{f.kind_code}_p{f.page}.img"
            (out / name).write_bytes(f.data)
            row["artifact"] = name
        rows.append(row)
    (out / "figure_test.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))
    return 0 if any(r["status"] == "retrieved" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
