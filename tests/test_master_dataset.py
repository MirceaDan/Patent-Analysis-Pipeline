"""Tests for scripts/build_master_dataset.py.

Fixture tests use small, explicitly synthetic files written to tmp_path.
Real-output tests run only if the built outputs exist in master_dataset/ and never
modify them.
"""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "database"          # original sources
MASTER = ROOT / "master_dataset"  # derived outputs
spec = importlib.util.spec_from_file_location(
    "build_master_dataset", ROOT / "scripts" / "build_master_dataset.py")
bmd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bmd)

COLS = ["Family Number", "Title", "Patent Number", "Kind Code",
        "Publication Date", "Application Number", "Application Date",
        "Country Code", "Master Patent Number", "Priority Numbers",
        "Priority Dates", "Forward Citations", "Abstract"]


def row(fam, title="T", pubs=("US1", "EP2"), kinds=("A1", "A1"),
        fwd="", abstract="abs"):
    n = len(pubs)
    return {
        "Family Number": fam, "Title": title,
        "Patent Number": "; ".join(pubs), "Kind Code": "; ".join(kinds),
        "Publication Date": "; ".join(["20240101"] * n),
        "Application Number": "; ".join(f"{p[:2]}A{i}" for i, p in enumerate(pubs)),
        "Application Date": "; ".join(["20230101"] * n),
        "Country Code": "; ".join(p[:2] for p in pubs),
        "Master Patent Number": f"{pubs[0]} {kinds[0]}",
        "Priority Numbers": "P1", "Priority Dates": "20220101",
        "Forward Citations": fwd, "Abstract": abstract,
    }


def make_inputs(tmp_path, files, coding, admin=None):
    src = tmp_path / "src"
    src.mkdir()
    names = []
    for i, rows in enumerate(files):
        name = f"part{i}.csv"
        names.append(name)
        pd.DataFrame(rows, columns=COLS).to_csv(src / name, index=False,
                                                encoding="utf-8-sig")
    if admin is None:
        admin = [{"APS_ID": r["APS_ID"], "Family Number": r["Family Number"],
                  "DESIGN_WEIGHT": "2.5", "INCLUSION_PROB": "0.4"}
                 for r in coding]
    with pd.ExcelWriter(src / "wb.xlsx") as xw:
        pd.DataFrame(coding).to_excel(xw, sheet_name=bmd.SHEET_CODING, index=False)
        pd.DataFrame(admin).to_excel(xw, sheet_name=bmd.SHEET_ADMIN, index=False)
    return src, names


def coding_row(aps, fam, m="Y", p="N", r="U", i="", **kw):
    d = {"APS_ID": aps, "Family Number": fam, "Title": "T", "Abstract": "abs",
         "F_MONITOR_REPRESENT": m, "F_PREDICT_SIMULATE": p,
         "F_RECOMMEND_DECIDE": r, "F_INTERVENE_CONTROL": i,
         "CONFIDENCE": "HIGH", "EVIDENCE_SOURCE": "CLAIMS"}
    d.update(kw)
    return d


@pytest.fixture
def built(tmp_path):
    f1 = [row("1"), row("2", pubs=("US3",), kinds=("B1",)), row("3")]
    f2 = [row("2", pubs=("US3",), kinds=("B1",)),          # exact duplicate
          row("3", fwd="X"),                                # metadata-only diff
          row("4", title="A"), row("5", pubs=("US9",), kinds=("A1",))]
    f3 = [row("4", title="B"),                              # conflicting content
          row("5", pubs=("US8",), kinds=("A1",))]           # identity differs
    coding = [coding_row("APS-1", "1"), coding_row("APS-2", "3", m="N", p="Y"),
              coding_row("APS-9", "999")]                   # unmatched
    src, names = make_inputs(tmp_path, [f1, f2, f3], coding)
    out = tmp_path / "out"
    summary = bmd.build(src, out, csv_files=names, workbook="wb.xlsx")
    return src, names, out, summary


def rd(p):
    return pd.read_csv(p, dtype=str, keep_default_na=False, na_filter=False)


def test_all_raw_rows_accounted_for(built):
    _, _, out, s = built
    raw = rd(out / bmd.OUT["raw"])
    assert len(raw) == s["raw_rows_total"] == 9
    assert list(raw.source_file.unique()) == ["part0.csv", "part1.csv", "part2.csv"]
    master = rd(out / bmd.OUT["master"])
    assert master["n_source_rows"].astype(int).sum() == len(raw)


def test_canonical_families_unique_and_counted(built):
    _, _, out, s = built
    master = rd(out / bmd.OUT["master"])
    assert master["Family Number"].is_unique
    assert len(master) == s["canonical_families"] == 5 == s["unique_family_numbers_raw"]


def test_duplicate_classification(built):
    _, _, out, _ = built
    m = rd(out / bmd.OUT["master"]).set_index("Family Number")
    assert m.at["1", "reconciliation_status"] == "unique"
    assert m.at["2", "reconciliation_status"] == "exact_duplicate"
    assert m.at["3", "reconciliation_status"] == "metadata_only_difference"
    assert m.at["4", "reconciliation_status"] == "conflicting_content"
    assert m.at["5", "reconciliation_status"] == "unreconcilable"
    # differing values are preserved, not dropped
    alt = json.loads(m.at["3", "field_alternates_json"])
    assert alt == {"Forward Citations": ["", "X"]}
    assert json.loads(m.at["4", "field_alternates_json"])["Title"] == ["A", "B"]
    rep = rd(out / bmd.OUT["dedup"]).set_index("family_number")
    assert set(rep.index) == {"2", "3", "4", "5"}
    assert rep.at["4", "differing_fields"] == "Title"


def test_publications_union_for_unreconcilable_family(built):
    _, _, out, _ = built
    pubs = rd(out / bmd.OUT["publications"])
    assert set(pubs[pubs.family_number == "5"].publication_number) == {"US9", "US8"}
    master = rd(out / bmd.OUT["master"])
    assert set(pubs.family_number) <= set(master["Family Number"])
    assert (pubs.publication_number != "").all()
    # exact duplicate does not duplicate publication rows
    assert (pubs.family_number == "2").sum() == 1


def test_misaligned_packed_fields_are_not_parsed(tmp_path):
    bad = row("7")
    bad["Kind Code"] = "A1"  # 1 token vs 2 publication numbers
    src, names = make_inputs(tmp_path, [[bad, row("8")]], [coding_row("A", "8")])
    s = bmd.build(src, tmp_path / "o", csv_files=names, workbook="wb.xlsx")
    pubs = rd(tmp_path / "o" / bmd.OUT["publications"])
    assert "7" not in set(pubs.family_number)
    assert s["source_rows_unparsed_for_publications"] == 1
    # the family itself is still in the master with its packed fields intact
    m = rd(tmp_path / "o" / bmd.OUT["master"]).set_index("Family Number")
    assert m.at["7", "Patent Number"] == "US1; EP2"


def test_labels_preserved_and_join_does_not_multiply(built):
    _, _, out, s = built
    master = rd(out / bmd.OUT["master"])
    ann = rd(out / bmd.OUT["annotated"])
    assert len(ann) == len(master)
    assert ann["Family Number"].is_unique
    a = ann.set_index("Family Number")
    assert a.at["1", "F_MONITOR_REPRESENT"] == "Y"
    assert a.at["1", "F_RECOMMEND_DECIDE"] == "U"          # U kept distinct
    assert a.at["1", "F_INTERVENE_CONTROL"] == ""          # missing stays missing
    assert a.at["3", "F_PREDICT_SIMULATE"] == "Y"
    assert a.at["1", "label_M"] == "Y" and a.at["1", "label_I"] == ""
    assert a.at["1", "adm_DESIGN_WEIGHT"] == "2.5"
    assert a.at["1", "ann_APS_ID"] == "APS-1"
    assert a.at["2", "is_annotated"] == "False"
    assert a.at["2", "F_MONITOR_REPRESENT"] == ""          # not labelled N
    assert s["annotation"]["annotated_families"] == 2


def test_unmatched_annotation_reported(built):
    _, _, out, s = built
    assert s["annotation"]["unmatched_annotations"] == [
        {"APS_ID": "APS-9", "Family Number": "999"}]
    assert "999" in (out / bmd.OUT["join_report"]).read_text()


def test_duplicate_annotation_keys_not_joined(tmp_path):
    coding = [coding_row("A1", "1"), coding_row("A2", "1", m="N"),
              coding_row("A3", "2")]
    src, names = make_inputs(tmp_path, [[row("1"), row("2")]], coding)
    s = bmd.build(src, tmp_path / "o", csv_files=names, workbook="wb.xlsx")
    ann = rd(tmp_path / "o" / bmd.OUT["annotated"]).set_index("Family Number")
    assert len(ann) == 2
    assert s["annotation"]["duplicate_annotation_keys"] == ["1"]
    assert ann.at["1", "is_annotated"] == "False"
    assert ann.at["2", "is_annotated"] == "True"


def test_schema_mismatch_rejected(tmp_path):
    src, names = make_inputs(tmp_path, [[row("1")], [row("2")]],
                             [coding_row("A", "1")])
    df = rd(src / names[1]).rename(columns={"Title": "Titel"})
    df.to_csv(src / names[1], index=False)
    with pytest.raises(ValueError, match="schemas differ"):
        bmd.build(src, tmp_path / "o", csv_files=names, workbook="wb.xlsx")


def test_figure_table_schema_and_empty(built):
    _, _, out, _ = built
    fig = rd(out / bmd.OUT["figures"])
    assert list(fig.columns) == bmd.FIGURE_SCHEMA
    assert len(fig) == 0


def test_rebuild_is_byte_identical(built, tmp_path):
    src, names, out, _ = built
    out2 = tmp_path / "out2"
    bmd.build(src, out2, csv_files=names, workbook="wb.xlsx")
    for key, fname in bmd.OUT.items():
        assert (out / fname).read_bytes() == (out2 / fname).read_bytes(), fname


def test_empty_family_key_rejected(tmp_path):
    src, names = make_inputs(tmp_path, [[row(""), row("2")]],
                             [coding_row("A", "2")])
    with pytest.raises(ValueError, match="empty Family Number"):
        bmd.build(src, tmp_path / "o", csv_files=names, workbook="wb.xlsx")


# ---------------------------------------------------------------- real outputs
REAL = MASTER / bmd.OUT["summary"]


@pytest.mark.skipif(not REAL.exists(), reason="real dataset not built")
class TestRealOutputs:
    def test_row_accounting_against_sources(self):
        s = json.loads(REAL.read_text())
        total = 0
        for f in bmd.CSV_FILES:
            n = len(bmd.read_csv_str(DB / f))
            assert s["raw_rows_per_file"][f] == n
            total += n
        assert total == s["raw_rows_total"] == 28431
        raw = rd(MASTER / bmd.OUT["raw"])
        assert len(raw) == total

    def test_master_unique_and_matches_dedup(self):
        raw = rd(MASTER / bmd.OUT["raw"])
        master = rd(MASTER / bmd.OUT["master"])
        assert master["Family Number"].is_unique
        assert set(master["Family Number"]) == set(raw["Family Number"].str.strip())
        assert master["n_source_rows"].astype(int).sum() == len(raw)

    def test_annotated_integrity(self):
        ann = rd(MASTER / bmd.OUT["annotated"])
        master = rd(MASTER / bmd.OUT["master"])
        assert len(ann) == len(master) and ann["Family Number"].is_unique
        lab = ann[ann.is_annotated == "True"]
        assert len(lab) == 600
        for f in bmd.LABEL_FIELDS:
            assert set(lab[f]) <= {"Y", "N", "U"}
            assert (ann.loc[ann.is_annotated == "False", f] == "").all()
        wb = pd.read_excel(DB / bmd.WORKBOOK, sheet_name=bmd.SHEET_CODING,
                           dtype=str, keep_default_na=False)
        for f in bmd.LABEL_FIELDS:
            assert lab[f].value_counts().to_dict() == wb[f].value_counts().to_dict()

    def test_publications_reference_families(self):
        pubs = rd(MASTER / bmd.OUT["publications"])
        master = rd(MASTER / bmd.OUT["master"])
        assert set(pubs.family_number) <= set(master["Family Number"])
        assert (pubs.publication_number.str.strip() != "").all()
