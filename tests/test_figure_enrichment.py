"""Unit tests for scripts/enrich_patent_figures.py.

All fixtures are synthetic. No network access and no real VLM calls.
"""
import importlib.util
import io
import json
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "enrich_patent_figures", ROOT / "scripts" / "enrich_patent_figures.py")
efp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(efp)

NOSLEEP = lambda s: None  # noqa: E731


# ---------------------------------------------------------------- fixtures
def png(seed: int) -> bytes:
    im = Image.new("L", (60, 40), 255)
    im.putpixel((seed % 60, seed % 40), 0)
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


def page_html(doc_id, date="2024-05-01", n_figs=2, parts=3, extra_img=True,
              scheme=None):
    import re
    cc, num, kind = re.match(r"^([A-Z]{2})(\d+)([A-Z]\d?)$", doc_id).groups()
    content = f"{cc}:{num}:{kind}" if parts == 3 else f"{cc}:{num}"
    name = ("citation_patent_publication_number" if parts == 3
            else "citation_patent_number")
    figs = ""
    for i in range(n_figs):
        figs += (
            '<li itemprop="images" itemscope repeat>'
            f'<img itemprop="thumbnail" src="https://img.test/t/{doc_id}-D0000{i}.png">'
            f'<meta itemprop="full" content="https://img.test/{doc_id}-D0000{i}.png">'
            '<ul><li itemprop="callouts" itemscope repeat>'
            '<meta itemprop="id" content="10"><meta itemprop="label" content="sensor">'
            "</li></ul></li>")
    if extra_img:
        figs += ('<li itemprop="images" itemscope repeat>'
                 '<meta itemprop="full" content="https://img.test/OTHER123A1-D00001.png">'
                 "</li>")
    date_meta = (f'<meta name="DC.date" content="{date}"'
                 + (f' scheme="{scheme}"' if scheme else "") + ">")
    return (f'<html><head><link rel="canonical" href="https://patents.google.com/patent/{doc_id}/en">'
            f'<meta name="{name}" content="{content}">{date_meta}'
            '<meta name="DC.date" content="2020-01-01" scheme="dateSubmitted">'
            '<meta name="citation_pdf_url" content="https://img.test/doc.pdf">'
            f"</head><body><section><h2>Images</h2><ul>{figs}</ul></section></body></html>")


def pub(family, number, kind, date="20240501", master=False, cc=None):
    return {"family_number": family, "publication_number": number,
            "kind_code": kind, "publication_date": date,
            "application_number": "", "application_date": "",
            "country_code": cc or number[:2],
            "position_in_source_row": "1",
            "is_master_publication": str(master), "source_rows": "x"}


class FakeHttp:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def get(self, url, head=False):
        self.calls.append((url, head))
        r = self.routes.get(url)
        if callable(r):
            r = r()
        if r is None:
            return 404, {}, b""
        if isinstance(r, Exception):
            raise r
        return r[0], {}, r[1]

    def n(self, prefix):
        return sum(1 for u, _ in self.calls if u.startswith(prefix))


class FakeVLM:
    model = "fake-vlm"

    def __init__(self, cfg="cfg1", fail_on=()):
        self.config_hash, self.calls, self.fail_on = cfg, [], set(fail_on)

    def describe(self, data, mime, fig_id):
        self.calls.append(fig_id)
        if fig_id in self.fail_on:
            raise efp.VlmError("boom")
        return ({"description": f"desc of {fig_id}", "figure_type": "x",
                 "purpose": "p", "components": [], "relations": [],
                 "text_in_figure": [], "uncertainties": [],
                 "process_or_mechanism": None}, {"prompt_tokens": 5,
                                                 "completion_tokens": 7})


def doc_route(doc_id, n_figs=2, date="2024-05-01", **kw):
    url = efp.document_url(doc_id)
    routes = {url: (200, page_html(doc_id, date, n_figs, **kw).encode())}
    for i in range(n_figs):
        routes[f"https://img.test/{doc_id}-D0000{i}.png"] = (200, png(i + hash(doc_id) % 7 * 10))
    return routes


def make_pipe(tmp_path, routes, vlm=None, **kw):
    state = efp.State(tmp_path / "out")
    kw.setdefault("sleep", NOSLEEP)
    kw.setdefault("backoff", 0)
    http = FakeHttp(routes)
    return efp.Pipeline(state, http, vlm, **kw), http, state


# ----------------------------------------------------- identifier handling
def test_normalize_us_application_and_grant():
    assert efp.candidate_document_ids("US", "US2020334401", "AA") == ["US20200334401A1"]
    assert efp.candidate_document_ids("US", "US11669085", "BB") == [
        "US11669085B2", "US11669085B1"]


def test_normalize_wo_year_expansion_and_whitespace():
    assert efp.candidate_document_ids("WO", " WO23247172 ", "A1") == ["WO2023247172A1"]


def test_normalize_plain_and_invalid():
    assert efp.candidate_document_ids("CN", "CN118709918", "A") == ["CN118709918A"]
    assert efp.candidate_document_ids("EP", "EP4544764", "A1") == ["EP4544764A1"]
    assert efp.candidate_document_ids("CN", "CN118709918", "") == []
    assert efp.candidate_document_ids("CN", "US118709918", "A") == []  # prefix mismatch
    assert len(efp.candidate_document_ids("US", "US11669085", "ZZ")) <= efp.MAX_CANDIDATES


def test_order_publications_master_first_then_country_rank():
    pubs = [pub("1", "CN1", "A"), pub("1", "EP2", "A1"),
            pub("1", "KR3", "A", master=True), pub("1", "US4", "AA")]
    order = [p["publication_number"] for p in efp.order_publications(pubs)]
    assert order == ["KR3", "US4", "EP2", "CN1"]


# ------------------------------------------------------------ page parsing
def test_parse_page_extracts_only_own_figures_and_labels():
    page = efp.parse_document_page(page_html("US20240000001A1", n_figs=3))
    assert page["document_id"] == "US20240000001A1"
    assert [f["figure_id"] for f in page["figures"]] == ["D00000", "D00001", "D00002"]
    assert page["figures"][0]["callout_labels"] == ["10:sensor"]
    assert all("OTHER123" not in f["url"] for f in page["figures"])
    assert page["pdf_url"] == "https://img.test/doc.pdf"


def test_parse_grant_page_without_kind_in_meta():
    html = page_html("US10084826B1", parts=2, scheme="issue").replace(
        'img.test/US10084826B1-', 'img.test/US10084826-')
    page = efp.parse_document_page(html)
    assert page["document_id"] == "US10084826B1"
    assert page["publication_date"] == "2024-05-01"
    assert len(page["figures"]) == 2   # file names without kind code accepted


def test_parse_non_patent_page_returns_none():
    assert efp.parse_document_page("<html><body>Please verify you are human</body></html>") is None


def test_verify_document_matches_and_mismatches():
    page = efp.parse_document_page(page_html("US20240000001A1", date="2024-05-01"))
    assert efp.verify_document(page, "US20240000001A1", "20240501") == (True, "pub_number+pub_date")
    assert efp.verify_document(page, "US20240000001A1", "") == (True, "pub_number_only")
    assert not efp.verify_document(page, "US20240000002A1", "20240501")[0]
    assert not efp.verify_document(page, "US20240000001A1", "20230101")[0]


# ------------------------------------------------------ VLM output validation
GOOD = {"figure_type": "block diagram", "purpose": "shows modules",
        "components": [{"label": "10", "name": "sensor", "evidence": "visible"}],
        "relations": [{"source": "10", "target": "12", "relation": "arrow", "evidence": "visible"}],
        "process_or_mechanism": None, "text_in_figure": ["A"],
        "uncertainties": [], "description": "A sensor feeds a module."}


def test_validate_description_accepts_good_and_fenced_json():
    assert efp.validate_description(dict(GOOD))["figure_type"] == "block diagram"
    assert efp.parse_vlm_text("```json\n" + json.dumps(GOOD) + "\n```")["purpose"]


@pytest.mark.parametrize("mutate", [
    lambda d: d.pop("description"),
    lambda d: d.update(description="  "),
    lambda d: d.update(components="x"),
    lambda d: d["components"][0].update(evidence="guess"),
    lambda d: d["relations"][0].pop("target"),
    lambda d: d.update(label="M"),          # classification-like key
])
def test_validate_description_rejects_bad(mutate):
    d = json.loads(json.dumps(GOOD))
    mutate(d)
    with pytest.raises(ValueError):
        efp.validate_description(d)


def test_parse_vlm_text_rejects_non_json():
    with pytest.raises(ValueError):
        efp.parse_vlm_text("The figure shows a box.")


# ------------------------------------------------------------ VLM client retry
class FakeResp:
    def __init__(self, status, payload=None, headers=None):
        self.status_code, self._p, self.headers = status, payload, headers or {}

    def json(self):
        if self._p is None:
            raise ValueError("no json")
        return self._p


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append((url, headers))
        r = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        return r


def ok_resp(obj=GOOD):
    return FakeResp(200, {"choices": [{"message": {"content": json.dumps(obj)}}],
                          "usage": {"prompt_tokens": 11, "completion_tokens": 22}})


def client(session, **kw):
    return efp.OpenAICompatibleVLM("https://x/v1", "m", "SECRET-KEY", session=session,
                                   sleep=NOSLEEP, **kw)


def test_vlm_retries_transient_then_succeeds():
    s = FakeSession([FakeResp(429, headers={"Retry-After": "1"}), FakeResp(503), ok_resp()])
    desc, usage = client(s).describe(png(1), "image/png", "D00001")
    assert len(s.calls) == 3 and usage["prompt_tokens"] == 11


def test_vlm_retry_limit_is_bounded():
    s = FakeSession([FakeResp(500)])
    with pytest.raises(efp.VlmError):
        client(s, max_retries=2).describe(png(1), "image/png", "D1")
    assert len(s.calls) == 3


def test_vlm_invalid_output_retried_then_fails():
    s = FakeSession([FakeResp(200, {"choices": [{"message": {"content": "no json"}}]})])
    with pytest.raises(efp.VlmError, match="invalid_output"):
        client(s, max_retries=1).describe(png(1), "image/png", "D1")
    assert len(s.calls) == 2


def test_vlm_auth_error_is_permanent_and_not_retried():
    s = FakeSession([FakeResp(401)])
    with pytest.raises(efp.VlmError) as ei:
        client(s).describe(png(1), "image/png", "D1")
    assert ei.value.permanent and len(s.calls) == 1


def test_vlm_from_env_missing_config(monkeypatch):
    for v in ("PATENT_VLM_BASE_URL", "PATENT_VLM_MODEL", "PATENT_VLM_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    with pytest.raises(efp.VlmError, match="not configured"):
        efp.OpenAICompatibleVLM.from_env()


def test_config_hash_changes_with_model():
    a = efp.OpenAICompatibleVLM("u", "m1", "k").config_hash
    b = efp.OpenAICompatibleVLM("u", "m2", "k").config_hash
    assert a != b


# --------------------------------------------------------- HTTP retry / robots
def test_get_with_retry_bounded():
    calls = []

    class H:
        def get(self, url, head=False):
            calls.append(url)
            return 503, {}, b""
    with pytest.raises(efp.TransientHttpError):
        efp.get_with_retry(H(), "u", max_retries=2, backoff=0, sleep=NOSLEEP)
    assert len(calls) == 3


def test_robots_longest_match_with_wildcards():
    txt = "User-agent: *\nDisallow: /*\nAllow: /$\nAllow: /patent/\n"
    assert efp.robots_allows(txt, "/patent/US1A1/en")
    assert not efp.robots_allows(txt, "/xhr/query")
    assert efp.robots_allows(txt, "/")


def test_rate_limiter_spaces_requests_per_host():
    t = [0.0]
    sleeps = []
    lim = efp.RateLimiter({"a.test": 2.0}, sleep=lambda s: (sleeps.append(s), t.__setitem__(0, t[0] + s)),
                          clock=lambda: t[0])
    lim.wait("https://a.test/x")
    lim.wait("https://a.test/y")
    lim.wait("https://b.test/y")
    assert sleeps == [2.0]


# ------------------------------------------------------------- image handling
def test_prepare_image_valid_and_invalid():
    mime, data, w, h = efp.prepare_image(png(3))
    assert mime == "image/png" and (w, h) == (60, 40)
    with pytest.raises(ValueError):
        efp.prepare_image(b"not an image")


# ----------------------------------------------------------- pipeline: flows
def test_master_with_figures_is_used_and_fallback_not_fetched(tmp_path):
    routes = {**doc_route("US20240000001A1"), **doc_route("EP9999999A1")}
    pipe, http, state = make_pipe(tmp_path, routes, FakeVLM())
    pubs = [pub("F1", "US2024000001", "AA", master=False),
            pub("F1", "EP9999999", "A1", master=True)]
    # master (EP) is tried first
    pubs[0]["publication_number"] = "US2024000001"
    pipe.process_family("F1", pubs)
    assert http.n(efp.document_url("EP9999999A1")) == 1
    assert http.n(efp.document_url("US20240000001A1")) == 0
    rows = list(state.figures.values())
    assert len(rows) == 2 and {r["publication_number"] for r in rows} == {"EP9999999"}
    assert all(r["publication_role"] == "master" and r["family_number"] == "F1" for r in rows)
    assert all(r["source_document_id"] == "EP9999999A1" for r in rows)


def test_fallback_to_other_publication_when_master_missing(tmp_path):
    routes = doc_route("US20240000001A1")            # CN master -> 404 (no route)
    pipe, http, state = make_pipe(tmp_path, routes, FakeVLM())
    pubs = [pub("F2", "CN111", "A", master=True), pub("F2", "US2024000001", "AA")]
    pipe.process_family("F2", pubs)
    res = {k[1]: v for k, v in state.resolutions.items()}
    assert res["CN111"]["resolution_status"] == "document_not_found"
    assert res["US2024000001"]["resolution_status"] == "resolved_with_figures"
    assert res["US2024000001"]["publication_role"] == "fallback"
    rows = list(state.figures.values())
    assert rows and all(r["publication_role"] == "fallback" for r in rows)
    summ = pipe.summarize({"F2": pubs})[0]
    assert summ["family_status"] == "described"
    assert summ["publication_used"] == "US2024000001 AA"
    assert summ["n_publications_examined"] == 2


def test_family_without_figures_not_treated_as_empty_evidence(tmp_path):
    routes = doc_route("CN111A", n_figs=0)
    pipe, http, state = make_pipe(tmp_path, routes, FakeVLM())
    pubs = [pub("F3", "CN111", "A", master=True)]
    pipe.process_family("F3", pubs)
    assert not state.figures                       # no fabricated figure rows
    s = pipe.summarize({"F3": pubs})[0]
    assert s["family_status"] == "no_figures_found"
    assert s["has_usable_description"] == "False" and s["failure_reason"]


def test_identifier_mismatch_rejected(tmp_path):
    url = efp.document_url("US20240000001A1")
    routes = {url: (200, page_html("US20240000009A1").encode())}   # wrong document
    pipe, _, state = make_pipe(tmp_path, routes, FakeVLM())
    pipe.process_family("F4", [pub("F4", "US2024000001", "AA", master=True)])
    r = next(iter(state.resolutions.values()))
    assert r["resolution_status"] == "identifier_mismatch" and not state.figures


def test_date_mismatch_rejected(tmp_path):
    routes = doc_route("US20240000001A1", date="2021-01-01")
    pipe, _, state = make_pipe(tmp_path, routes, FakeVLM())
    pipe.process_family("F5", [pub("F5", "US2024000001", "AA", date="20240501", master=True)])
    assert next(iter(state.resolutions.values()))["resolution_status"] == "identifier_mismatch"


def test_dry_run_makes_no_image_or_vlm_calls_except_head(tmp_path):
    routes = doc_route("US20240000001A1", n_figs=3)
    vlm = FakeVLM()
    pipe, http, state = make_pipe(tmp_path, routes, vlm, dry_run=True)
    pipe.process_family("F6", [pub("F6", "US2024000001", "AA", master=True)])
    assert vlm.calls == []
    assert all(head for u, head in http.calls if u.startswith("https://img.test"))
    st = [r["processing_status"] for r in state.figures.values()]
    assert st == ["pending"] * 3 and len(http.calls) == 2


def test_max_figures_marks_skipped_limit(tmp_path):
    routes = doc_route("US20240000001A1", n_figs=4)
    pipe, _, state = make_pipe(tmp_path, routes, FakeVLM(), max_figures=2)
    pubs = [pub("F7", "US2024000001", "AA", master=True)]
    pipe.process_family("F7", pubs)
    st = sorted(r["processing_status"] for r in state.figures.values())
    assert st == ["described", "described", "skipped_limit", "skipped_limit"]
    s = pipe.summarize({"F7": pubs})[0]
    assert s["n_figures_discovered"] == 4 and s["n_figures_skipped_limit"] == 2


# ----------------------------------------------- errors / isolation / retries
def test_image_failure_isolated_and_reported(tmp_path):
    routes = doc_route("US20240000001A1", n_figs=2)
    routes["https://img.test/US20240000001A1-D00000.png"] = (500, b"")
    pipe, _, state = make_pipe(tmp_path, routes, FakeVLM(), max_retries=1)
    pubs = [pub("F8", "US2024000001", "AA", master=True)]
    pipe.process_family("F8", pubs)
    by = {r["figure_id"]: r for r in state.figures.values()}
    assert by["D00000"]["processing_status"] == "image_inaccessible"
    assert by["D00001"]["processing_status"] == "described"
    assert pipe.summarize({"F8": pubs})[0]["n_figures_failed"] == 1


def test_vlm_failure_recorded_and_attempts_capped(tmp_path):
    routes = doc_route("US20240000001A1", n_figs=1)
    vlm = FakeVLM(fail_on={"D00000"})
    pipe, _, state = make_pipe(tmp_path, routes, vlm, max_attempts=2,
                               abort_after_vlm_failures=99)
    pubs = [pub("F9", "US2024000001", "AA", master=True)]
    for _ in range(4):
        pipe.process_family("F9", pubs)
    row = next(iter(state.figures.values()))
    assert row["processing_status"] == "vlm_failed" and row["attempts"] == 2
    assert len(vlm.calls) == 2                       # not retried beyond cap
    assert pipe.summarize({"F9": pubs})[0]["family_status"] == "vlm_failed"


def test_permanent_vlm_error_stops_run(tmp_path):
    class Dead(FakeVLM):
        def describe(self, *a):
            raise efp.VlmError("HTTP 401", permanent=True)
    routes = doc_route("US20240000001A1", n_figs=3)
    pipe, _, state = make_pipe(tmp_path, routes, Dead())
    pipe.process_family("FA", [pub("FA", "US2024000001", "AA", master=True)])
    assert pipe.stop.is_set()
    assert sum(r["processing_status"] == "vlm_failed" for r in state.figures.values()) == 1


def test_transient_document_error_is_retried_on_resume(tmp_path):
    url = efp.document_url("US20240000001A1")
    boom = efp.TransientHttpError("Timeout")
    routes = {url: boom}
    pipe, _, state = make_pipe(tmp_path, routes, FakeVLM(), dry_run=True, max_retries=1)
    pubs = [pub("FB", "US2024000001", "AA", master=True)]
    pipe.process_family("FB", pubs)
    assert next(iter(state.resolutions.values()))["resolution_status"] == "document_inaccessible"
    routes.update(doc_route("US20240000001A1"))
    pipe2, _, state2 = make_pipe(tmp_path, routes, FakeVLM(), dry_run=True)
    pipe2.process_family("FB", pubs)
    assert next(iter(state2.resolutions.values()))["resolution_status"] == "resolved_with_figures"


# ------------------------------------------------------------ resume / writes
def test_resume_skips_completed_work(tmp_path):
    routes = doc_route("US20240000001A1", n_figs=2)
    vlm = FakeVLM()
    pipe, http, _ = make_pipe(tmp_path, routes, vlm)
    pubs = [pub("FC", "US2024000001", "AA", master=True)]
    pipe.process_family("FC", pubs)
    assert len(vlm.calls) == 2
    vlm2 = FakeVLM()
    pipe2, http2, state2 = make_pipe(tmp_path, routes, vlm2)   # fresh process, same dir
    pipe2.process_family("FC", pubs)
    assert vlm2.calls == [] and http2.calls == []
    assert pipe2.counts["skipped_resume"] == 2
    assert all(r["processing_status"] == "described" for r in state2.figures.values())


def test_config_change_triggers_reprocessing(tmp_path):
    routes = doc_route("US20240000001A1", n_figs=1)
    pipe, _, _ = make_pipe(tmp_path, routes, FakeVLM("cfgA"))
    pubs = [pub("FD", "US2024000001", "AA", master=True)]
    pipe.process_family("FD", pubs)
    vlm2 = FakeVLM("cfgB")
    pipe2, _, state2 = make_pipe(tmp_path, routes, vlm2)
    pipe2.process_family("FD", pubs)
    assert len(vlm2.calls) == 1
    assert next(iter(state2.figures.values()))["vlm_config_hash"] == "cfgB"


def test_incremental_writes_and_torn_journal_tolerated(tmp_path):
    routes = doc_route("US20240000001A1", n_figs=2)
    vlm = FakeVLM(fail_on={"D00001"})
    pipe, _, state = make_pipe(tmp_path, routes, vlm)
    pipe.process_family("FE", [pub("FE", "US2024000001", "AA", master=True)])
    lines = (tmp_path / "out" / "figures.jsonl").read_text().splitlines()
    assert len(lines) == 2                           # one durable line per figure
    with open(tmp_path / "out" / "figures.jsonl", "a") as fh:
        fh.write('{"family_number": "FE", "trunc')   # simulated crash mid-write
    st2 = efp.State(tmp_path / "out")
    assert len(st2.figures) == 2


def test_stale_tmp_files_removed(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    (d / "figure_enrichments.csv.tmp").write_text("partial")
    efp.State(d)
    assert not list(d.glob("*.tmp"))


def test_duplicate_images_detected_by_checksum(tmp_path):
    routes = doc_route("US20240000001A1", n_figs=2)
    same = png(5)
    routes["https://img.test/US20240000001A1-D00000.png"] = (200, same)
    routes["https://img.test/US20240000001A1-D00001.png"] = (200, same)
    vlm = FakeVLM()
    pipe, _, state = make_pipe(tmp_path, routes, vlm)
    pipe.process_family("FF", [pub("FF", "US2024000001", "AA", master=True)])
    assert len(vlm.calls) == 1
    by = {r["figure_id"]: r for r in state.figures.values()}
    assert by["D00001"]["reused_from"].endswith("D00000")
    assert by["D00001"]["image_sha256"] == by["D00000"]["image_sha256"]


def test_exports_have_documented_columns_and_no_secrets(tmp_path, monkeypatch):
    routes = doc_route("US20240000001A1", n_figs=1)
    pipe, _, state = make_pipe(tmp_path, routes, FakeVLM())
    fam = {"FG": [pub("FG", "US2024000001", "AA", master=True)]}
    pipe.process_family("FG", fam["FG"])
    state.export_csvs(summaries=pipe.summarize(fam))
    import csv
    out = tmp_path / "out"
    with open(out / "figure_enrichments.csv", newline="") as fh:
        assert next(csv.reader(fh)) == efp.FIGURE_COLUMNS
    with open(out / "family_summary.csv", newline="") as fh:
        assert next(csv.reader(fh)) == efp.SUMMARY_COLUMNS
    master_schema = (ROOT / "master_dataset" / "figure_enrichments.csv")
    if master_schema.exists():       # extension columns keep the canonical 14 first
        head = master_schema.read_text().splitlines()[0].split(",")
        assert efp.FIGURE_COLUMNS[:len(head)] == head
    blob = "".join(p.read_text() for p in out.glob("*") if p.is_file())
    assert "SECRET-KEY" not in blob


def test_summary_not_yet_processed_and_all_not_found(tmp_path):
    pipe, _, _ = make_pipe(tmp_path, {}, FakeVLM())
    pubs = [pub("FH", "CN1", "A", master=True)]
    assert pipe.summarize({"FH": pubs})[0]["family_status"] == "not_yet_processed"
    pipe.process_family("FH", pubs)
    assert pipe.summarize({"FH": pubs})[0]["family_status"] == "document_not_found"


def test_select_families_deterministic_sample():
    grouped = {str(i): [] for i in range(50)}
    a = efp.select_families(grouped, sample=5, seed=1)
    b = efp.select_families(grouped, sample=5, seed=1)
    assert list(a) == list(b) and len(a) == 5
    assert list(efp.select_families(grouped, family_ids=["3", "4"])) == ["3", "4"]


# ------------------------------------------------------- local backend (fakes)
def _local(loader, **kw):
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from vlm_local import LocalTransformersVLM
    return LocalTransformersVLM(
        "org/model", "sys", "user {figure_id}", efp.parse_vlm_text, efp.VlmError,
        efp.PROMPT_VERSION, efp.IMAGE_PREP_VERSION, loader=loader, **kw)


class _T:
    def __init__(self, n):
        self.shape = (1, n)

    def __getitem__(self, k):
        return _T(self.shape[1] - (k[1].start or 0)) if isinstance(k, tuple) else self


def _fake_loader(outputs):
    outs = list(outputs)

    class Inputs(dict):
        def to(self, d):
            return self

    class Proc:
        def apply_chat_template(self, m, add_generation_prompt):
            return "p"

        def __call__(self, text, images, return_tensors):
            return Inputs(input_ids=_T(10))

        def batch_decode(self, new, skip_special_tokens):
            return [outs.pop(0)]

    class Mdl:
        device = "cpu"

        def generate(self, **kw):
            return _T(15)
    return lambda: (Mdl(), Proc())


def test_local_backend_describes_and_hashes_config():
    v = _local(_fake_loader([json.dumps(GOOD)]))
    desc, usage = v.describe(png(1), "image/png", "D1")
    assert desc["description"] and usage == {"prompt_tokens": 10, "completion_tokens": 5}
    assert v.config_hash != _local(None, revision="abc").config_hash
    assert _local(None, revision="abc").model_version == "abc"


def test_local_backend_invalid_output_bounded():
    v = _local(_fake_loader(["nope", "still nope"]), max_retries=1)
    with pytest.raises(efp.VlmError, match="invalid_output"):
        v.describe(png(1), "image/png", "D1")


def test_local_backend_missing_deps_is_permanent_blocker():
    v = _local(None)
    try:
        import transformers  # noqa: F401
        pytest.skip("transformers installed")
    except ImportError:
        pass
    with pytest.raises(efp.VlmError) as ei:
        v.describe(png(1), "image/png", "D1")
    assert ei.value.permanent


def test_make_vlm_local_requires_model(monkeypatch):
    monkeypatch.delenv("PATENT_VLM_LOCAL_MODEL", raising=False)
    a = efp.build_parser().parse_args(["--vlm-backend", "local"])
    with pytest.raises(efp.VlmError, match="local VLM not configured"):
        efp.make_vlm(a)


def test_validate_normalises_evidence_case_but_rejects_other_values():
    d = json.loads(json.dumps(GOOD))
    d["components"][0]["evidence"] = " Visible "
    d["relations"][0]["evidence"] = "INFERRED"
    out = efp.validate_description(d)
    assert out["components"][0]["evidence"] == "visible"
    assert out["relations"][0]["evidence"] == "inferred"
    d["components"][0]["evidence"] = "visible|inferred"
    with pytest.raises(ValueError):
        efp.validate_description(d)


# ------------------------------------------- reliability regression (task: POC)
def test_truncated_json_is_rejected_and_labelled_truncated_locally():
    cut = json.dumps(GOOD)[:-25]                      # cut mid-string, no closing brace
    v = _local(_fake_loader([cut]), max_new_tokens=5)  # fake generates 5 new tokens
    with pytest.raises(efp.VlmError, match="truncated_output"):
        v.describe(png(1), "image/png", "D1")


def test_incomplete_or_placeholder_outputs_rejected():
    for mutate in (lambda d: d.update(description="Too short."),
                   lambda d: d["components"][0].update(name="..."),
                   lambda d: d["relations"][0].update(source=" ")):
        d = json.loads(json.dumps(GOOD))
        mutate(d)
        with pytest.raises(ValueError):
            efp.validate_description(d)


def test_max_new_tokens_not_in_hash_but_recorded_and_legacy_hash_accepted():
    a, b = _local(None, max_new_tokens=1200), _local(None, max_new_tokens=2048)
    assert a.config_hash == b.config_hash
    assert json.loads(b.generation_config)["max_new_tokens"] == 2048
    assert _local(None, max_new_tokens=1200).config_hash not in b.legacy_config_hashes
    assert b._hash(max_new_tokens=1200) in b.legacy_config_hashes


def test_resume_keeps_legacy_hash_descriptions_and_retries_failed(tmp_path):
    routes = doc_route("US20240000001A1", n_figs=2)

    class Legacy(FakeVLM):
        legacy_config_hashes = ("old",)
    pipe, _, state = make_pipe(tmp_path, routes, Legacy("old", fail_on={"D00001"}),
                               max_attempts=3)
    pubs = [pub("FL", "US2024000001", "AA", master=True)]
    pipe.process_family("FL", pubs)
    new = Legacy("new")
    pipe2, _, st2 = make_pipe(tmp_path, routes, new, max_attempts=3)
    pipe2.process_family("FL", pubs)
    assert new.calls == ["D00001"]                    # only the failed figure re-run
    by = {r["figure_id"]: r for r in st2.figures.values()}
    assert by["D00000"]["vlm_config_hash"] == "old"
    assert by["D00001"]["processing_status"] == "described" and by["D00001"]["attempts"] == 2
    assert by["D00001"]["generation_config"] == ""   # fake has none; column exists
