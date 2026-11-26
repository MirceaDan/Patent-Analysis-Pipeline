"""Tests for scripts/figure_poc_tools.py on synthetic CSVs."""
import csv
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fpt", ROOT / "scripts" / "figure_poc_tools.py")
fpt = importlib.util.module_from_spec(spec)
sys.modules["fpt"] = fpt
spec.loader.exec_module(fpt)


def build(tmp):
    m, d = tmp / "m", tmp / "d"
    m.mkdir(); d.mkdir()
    fam = [("1", "Y", "G06N"), ("2", "N", "G06N"), ("3", "Y", "H04L"), ("4", "Y", "B25J")]
    fpt.write(m / "patent_families_annotated.csv",
              [{"Family Number": f, "is_annotated": "True", "Title": "t" + f,
                "adm_IPC": ipc + " 1/00", "F_MONITOR_REPRESENT": lab,
                "F_PREDICT_SIMULATE": "N", "F_RECOMMEND_DECIDE": "N",
                "F_INTERVENE_CONTROL": "N"} for f, lab, ipc in fam],
              ["Family Number", "is_annotated", "Title", "adm_IPC", *fpt.LABELS])
    rc = ["family_number", "publication_number", "kind_code", "publication_role",
          "document_id", "resolution_status", "n_figures_listed", "match_verification"]
    res = [dict(family_number="1", publication_number="CN1", kind_code="A", publication_role="master",
                document_id="CN1A", resolution_status="resolved_no_figures", n_figures_listed="0", match_verification="x"),
           dict(family_number="1", publication_number="US1", kind_code="AA", publication_role="fallback",
                document_id="US1A1", resolution_status="resolved_with_figures", n_figures_listed="3", match_verification="x"),
           dict(family_number="2", publication_number="CN2", kind_code="A", publication_role="master",
                document_id="CN2A", resolution_status="resolved_no_figures", n_figures_listed="0", match_verification="x"),
           dict(family_number="3", publication_number="US3", kind_code="AA", publication_role="master",
                document_id="US3A1", resolution_status="resolved_with_figures", n_figures_listed="2", match_verification="x"),
           dict(family_number="4", publication_number="US4", kind_code="AA", publication_role="master",
                document_id="US4A1", resolution_status="resolved_with_figures", n_figures_listed="2", match_verification="x")]
    fpt.write(d / "document_resolution.csv", res, rc)
    fc = ["family_number", "source_document_id", "processing_status"]
    fpt.write(d / "figure_enrichments.csv",
              [{"family_number": "4", "source_document_id": "US4A1",
                "processing_status": "image_inaccessible"}], fc)
    fpt.write(d / "family_summary.csv",
              [{"family_number": "2", "family_status": "no_figures_found"}], ["family_number", "family_status"])
    return m, d


def run(tmp, cmd, *extra):
    m, d = build(tmp) if not (tmp / "d").exists() else (tmp / "m", tmp / "d")
    fpt.main([cmd, "--data-dir", str(d), "--master-dir", str(m), *extra])
    return d


def test_eligibility_uses_resolver_evidence_not_country(tmp_path, capsys):
    d = run(tmp_path, "eligible")
    rows = list(csv.DictReader(open(d / "eligible_families.csv")))
    got = {r["family_number"]: r for r in rows}
    assert set(got) == {"1", "3"}                       # 4 excluded: image inaccessible
    assert got["1"]["publication_number"] == "US1" and got["1"]["country"] == "US"
    s = json.loads((d / "eligibility_summary.json").read_text())
    assert s["eligible_families"] == 2 and s["excluded_families"] == 2
    assert s["exclusion_reasons_by_family_status"]["no_figures_found"] == 1


def test_selection_is_deterministic_and_diverse(tmp_path, capsys):
    run(tmp_path, "eligible")
    d = run(tmp_path, "select", "-n", "2", "--seed", "1")
    a = (d / "pilot_families.csv").read_text()
    run(tmp_path, "select", "-n", "2", "--seed", "1")
    assert a == (d / "pilot_families.csv").read_text()
    assert {r["family_number"] for r in csv.DictReader(open(d / "pilot_families.csv"))} == {"1", "3"}
