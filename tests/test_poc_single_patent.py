"""Tests for the single-patent POC helpers and the raw-save hooks (synthetic data only)."""
import importlib.util
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


efp = _load("efp_poc", "scripts/enrich_patent_figures.py")
poc = _load("poc_mod", "scripts/poc_single_patent.py")


def test_page_text_extracts_sections_and_tolerates_absence():
    html = ('<meta name="DC.title" content="  A  title "><div itemprop="abstract">Abs  text</div>'
            '<section itemprop="claims">1. A thing</section>')
    t = efp._page_text(BeautifulSoup(html, "html.parser"))
    assert t == {"title": "A title", "abstract": "Abs text", "description": "",
                 "claims": "1. A thing"}


def test_raw_sinks_write_document_and_image(tmp_path):
    s = efp.make_raw_sinks(tmp_path)
    pub = {"publication_number": "US1", "kind_code": "A1"}
    page = {"figures": [{"figure_id": "D00000"}], "text": {"title": "t"},
            "publication_date": "2021-01-01", "application_number": "x"}
    s["document_sink"]("1", pub, "US1A1", "http://u", page, "2026-01-01T00:00:00Z")
    s["image_sink"]("1", {"document_id": "US1A1"}, {"figure_id": "D00000"}, b"\x89PNG....", "h")
    assert (tmp_path / "documents" / "US1A1.json").exists()
    assert (tmp_path / "figures" / "US1A1-D00000.png").read_bytes().startswith(b"\x89PNG")


def test_keyword_baseline_threshold_and_no_label_inputs():
    inp = {"title": "Monitor", "abstract": "We detect and monitor a digital twin.",
           "figure_descriptions": []}
    r = poc.keyword_baseline(inp)
    assert set(r) == set("MPRI")
    assert r["M"]["predicted"] == "Y" and r["P"]["predicted"] == "N"
    assert 0 <= r["M"]["raw_score"] <= 1
    assert poc.keyword_baseline(inp) == r  # deterministic
