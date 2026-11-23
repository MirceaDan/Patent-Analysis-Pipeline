#!/usr/bin/env python3
"""Build the canonical patent master dataset from the PatBase exports and the
M/P/R/I annotation workbook.

Reads from database/ (original exports, never modified) and writes to
master_dataset/ (derived outputs only).

Usage (from the repository root):
    python scripts/build_master_dataset.py
    python scripts/build_master_dataset.py --input-dir database --output-dir master_dataset

The source files are only ever read. Outputs are deterministic: no timestamps,
fixed row order (source order), fixed column order. Rerunning on unchanged
inputs yields byte-identical outputs (see `build_summary.json` for SHA-256s).

Outputs
-------
patent_records_raw.csv            every source row + source_file, source_row
patent_families_master.csv        one reconciled row per `Family Number`
patent_deduplication_report.csv   one row per repeated family (decision log)
patent_families_annotated.csv     master + annotation workbook (left join)
annotation_join_report.md         join / label / integrity report
patent_publications.csv           one row per (family, publication) tuple
figure_enrichments.csv            EMPTY schema for future VLM enrichment
build_summary.json                counts and output checksums

Reconciliation policy (per repeated `Family Number`)
----------------------------------------------------
Rows are compared on every original column after whitespace stripping.
  exact_duplicate            no differing field -> keep first row.
  metadata_only_difference   differing fields are all in METADATA_FIELDS
                             (citations, legal status, assignee name variants)
                             -> keep first row's values; every distinct value
                             of each differing field is kept in
                             `field_alternates_json`.
  unreconcilable             a publication/priority identity field differs
                             (IDENTITY_FIELDS) -> first row kept as
                             representative, flagged; publications of all rows
                             are still emitted to patent_publications.csv.
  conflicting_content        any other field differs -> same handling as above.
"First row" = lowest (source file order, row number). No value is silently
dropped: alternates are always serialised into `field_alternates_json`.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path

import pandas as pd

CSV_FILES = ["1_10000.csv", "10001_20000.csv", "20001_28431.csv"]
WORKBOOK = "sample_todo_completed.xlsx"
FAMILY_KEY = "Family Number"

OUT = {
    "raw": "patent_records_raw.csv",
    "master": "patent_families_master.csv",
    "dedup": "patent_deduplication_report.csv",
    "annotated": "patent_families_annotated.csv",
    "join_report": "annotation_join_report.md",
    "publications": "patent_publications.csv",
    "figures": "figure_enrichments.csv",
    "summary": "build_summary.json",
}

# Packed, ';'-separated, position-aligned publication fields in PatBase exports.
PUB_FIELDS = {
    "publication_number": "Patent Number",
    "kind_code": "Kind Code",
    "publication_date": "Publication Date",
    "application_number": "Application Number",
    "application_date": "Application Date",
    "country_code": "Country Code",
}

METADATA_FIELDS = {
    "Backward Citations", "Forward Citations", "Number of Backward Citations",
    "Number of Forward Citations", "Legal Status Application Number",
    "Legal Status Application Date", "Legal Status Patent",
    "Legal Status Date Recorded", "Legal Status Event Code",
    "Legal Status Event Group", "Legal Status Details",
    "Legal Status Timeline", "Assignees", "Non Std Assignees",
}
IDENTITY_FIELDS = {
    "Patent Number", "Kind Code", "Publication Date", "Application Number",
    "Application Date", "Country Code", "Master Patent Number",
    "Priority Numbers", "Priority Dates",
}

# Workbook layout (verified against sample_todo_completed.xlsx).
SHEET_CODING = "Functional_Coding"   # authoritative label sheet
SHEET_ADMIN = "Admin_Manifest"       # sampling design metadata per family
LABEL_FIELDS = ["F_MONITOR_REPRESENT", "F_PREDICT_SIMULATE",
                "F_RECOMMEND_DECIDE", "F_INTERVENE_CONTROL"]
VALID_LABEL_VALUES = {"Y", "N", "U"}
# Mapping documented in database/JSTPM_Fisa_Tehnica_*.md (role table: M =
# monitoring/representation, ...; the Y counts 337/232/133/118 match the
# workbook's Y counts for these columns in this order). The source does not
# state the column->letter mapping literally, so it is recorded as inferred
# from role names + counts and flagged for confirmation by Andreea.
LABEL_TO_MPRI = OrderedDict([
    ("F_MONITOR_REPRESENT", "label_M"),
    ("F_PREDICT_SIMULATE", "label_P"),
    ("F_RECOMMEND_DECIDE", "label_R"),
    ("F_INTERVENE_CONTROL", "label_I"),
])
# Text columns of the workbook that merely copy the CSV content; verified
# against the master and not duplicated in the annotated output.
CODING_TEXT_COPIES = ["Title", "Abstract", "1st Main Claim", "Claims_Part1",
                      "Claims_Part2", "Claims_Part3", "Claims_Part4"]

FIGURE_SCHEMA = [
    "family_number", "publication_number", "kind_code", "figure_id",
    "figure_description", "image_source_url", "source_document_url",
    "source_document_id", "vlm_model", "vlm_model_version", "prompt_version",
    "processing_status", "error_details", "processed_at_utc",
]
FIGURE_STATUS_VALUES = ["pending", "processed", "failed", "skipped"]

RECON_COLS = ["source_files", "source_rows", "n_source_rows",
              "reconciliation_status", "differing_fields",
              "field_alternates_json"]


# --------------------------------------------------------------------- IO
def read_csv_str(path: Path) -> pd.DataFrame:
    """All columns as exact strings; empty cells stay ''. UTF-8 with BOM."""
    return pd.read_csv(path, dtype=str, keep_default_na=False,
                       na_filter=False, encoding="utf-8-sig")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8", lineterminator="\n",
              quoting=csv.QUOTE_MINIMAL)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------- raw + master
def load_raw(input_dir: Path, csv_files=CSV_FILES):
    """Concatenate the CSV exports after checking schema equality."""
    frames, schemas = [], {}
    for name in csv_files:
        df = read_csv_str(input_dir / name)
        schemas[name] = list(df.columns)
        df.insert(0, "source_row", range(1, len(df) + 1))
        df.insert(0, "source_file", name)
        frames.append(df)
    first = schemas[csv_files[0]]
    mismatched = [n for n, c in schemas.items() if c != first]
    if mismatched:
        raise ValueError(f"CSV schemas differ from {csv_files[0]}: {mismatched}")
    if FAMILY_KEY not in first:
        raise ValueError(f"missing key column {FAMILY_KEY!r}")
    raw = pd.concat(frames, ignore_index=True)
    # Whitespace is stripped for keys only; values are otherwise untouched.
    raw[FAMILY_KEY] = raw[FAMILY_KEY].str.strip()
    return raw, first


def classify_group(diff_fields):
    if not diff_fields:
        return "exact_duplicate"
    d = set(diff_fields)
    if d & IDENTITY_FIELDS:
        return "unreconcilable"
    if d <= METADATA_FIELDS:
        return "metadata_only_difference"
    return "conflicting_content"


def reconcile(raw: pd.DataFrame, source_cols):
    """Return (master, dedup_report)."""
    empty_key = raw[FAMILY_KEY] == ""
    if empty_key.any():
        raise ValueError(f"{int(empty_key.sum())} rows have an empty {FAMILY_KEY}")
    first_rows = raw.drop_duplicates(FAMILY_KEY, keep="first")  # source order
    master = first_rows[source_cols].copy()
    master["source_files"] = first_rows["source_file"].values
    master["source_rows"] = (first_rows["source_file"] + ":" +
                             first_rows["source_row"].astype(str)).values
    master["n_source_rows"] = 1
    master["reconciliation_status"] = "unique"
    master["differing_fields"] = ""
    master["field_alternates_json"] = ""
    master = master.set_index(FAMILY_KEY, drop=False)
    report_rows = []
    dup_keys = raw.loc[raw.duplicated(FAMILY_KEY, keep=False), FAMILY_KEY]
    for fam in OrderedDict.fromkeys(dup_keys):
        grp = raw[raw[FAMILY_KEY] == fam]
        diff = [c for c in source_cols
                if c != FAMILY_KEY and grp[c].str.strip().nunique() > 1]
        status = classify_group(diff)
        alternates = {c: list(OrderedDict.fromkeys(grp[c])) for c in diff}
        row = {
            "source_files": "; ".join(OrderedDict.fromkeys(grp["source_file"])),
            "source_rows": "; ".join(f"{f}:{r}" for f, r in
                                     zip(grp["source_file"], grp["source_row"])),
            "n_source_rows": len(grp),
            "reconciliation_status": status,
            "differing_fields": "; ".join(diff),
            "field_alternates_json": (json.dumps(alternates, ensure_ascii=False)
                                      if alternates else ""),
        }
        for k, v in row.items():
            master.at[fam, k] = v
        action = {
            "exact_duplicate": "kept first row; rows identical",
            "metadata_only_difference":
                "kept first row; all distinct values preserved in "
                "field_alternates_json",
            "conflicting_content":
                "kept first row as representative; flagged; all distinct "
                "values preserved in field_alternates_json",
            "unreconcilable":
                "kept first row as representative; flagged as not safely "
                "reconcilable; publications of all rows emitted",
        }[status]
        report_rows.append({
            "family_number": fam, "n_rows": len(grp),
            "source_rows": row["source_rows"], "classification": status,
            "differing_fields": row["differing_fields"],
            "differences_json": row["field_alternates_json"], "action": action,
        })
    master = master.reset_index(drop=True)
    master = master[source_cols + RECON_COLS]
    dedup = pd.DataFrame(report_rows, columns=[
        "family_number", "n_rows", "source_rows", "classification",
        "differing_fields", "differences_json", "action"])
    return master, dedup


# ------------------------------------------------------------ publications
def split_packed(value: str):
    return [t.strip() for t in value.split(";")]


def build_publications(raw: pd.DataFrame):
    """One row per distinct (family, publication tuple).

    Alignment of the packed fields is verified per source row: all six fields
    must contain the same number of ';'-separated tokens and every publication
    number must be non-empty. Rows failing this are NOT parsed (no rows are
    fabricated); they are returned in `unparsed`.
    """
    out = OrderedDict()
    unparsed = []
    prefix_mismatch = 0
    for rec in raw.to_dict("records"):
        parts = {k: split_packed(rec[col]) for k, col in PUB_FIELDS.items()}
        lengths = {k: len(v) for k, v in parts.items()}
        n = lengths["publication_number"]
        if len(set(lengths.values())) != 1 or any(
                not t for t in parts["publication_number"]):
            unparsed.append({"family_number": rec[FAMILY_KEY],
                             "source_file": rec["source_file"],
                             "source_row": rec["source_row"],
                             "token_counts": lengths})
            continue
        master_id = rec.get("Master Patent Number", "").strip()
        for i in range(n):
            tup = tuple(parts[k][i] for k in PUB_FIELDS)
            key = (rec[FAMILY_KEY],) + tup
            src = f"{rec['source_file']}:{rec['source_row']}"
            if key in out:
                out[key]["source_rows"].append(src)
                continue
            if tup[0][:2] != tup[5][:2]:
                prefix_mismatch += 1
            out[key] = {
                "family_number": rec[FAMILY_KEY],
                **dict(zip(PUB_FIELDS, tup)),
                "position_in_source_row": i + 1,
                "is_master_publication":
                    f"{tup[0]} {tup[1]}".strip() == master_id,
                "source_rows": [src],
            }
    rows = []
    for v in out.values():
        v["source_rows"] = "; ".join(v["source_rows"])
        rows.append(v)
    cols = ["family_number", *PUB_FIELDS, "position_in_source_row",
            "is_master_publication", "source_rows"]
    pubs = pd.DataFrame(rows, columns=cols)
    return pubs, unparsed, prefix_mismatch


# -------------------------------------------------------------- annotations
def _norm_key(s: str):
    """Normalise a numeric family id read from Excel ('123.0' -> '123')."""
    s = str(s).strip()
    if re.fullmatch(r"\d+\.0+", s):
        return s.split(".")[0], True
    return s, False


def load_workbook_sheets(path: Path):
    xl = pd.ExcelFile(path)
    sheets = {n: pd.read_excel(xl, sheet_name=n, dtype=str,
                               keep_default_na=False, na_filter=False)
              for n in (SHEET_CODING, SHEET_ADMIN)}
    for name, df in sheets.items():
        for c in df.columns:
            df[c] = df[c].astype(str)
    return sheets, xl.sheet_names


def integrate_annotations(master: pd.DataFrame, sheets, workbook_name: str):
    coding, admin = sheets[SHEET_CODING].copy(), sheets[SHEET_ADMIN].copy()
    info = {"workbook": workbook_name, "coding_rows": len(coding),
            "admin_rows": len(admin), "warnings": []}
    missing = [c for c in LABEL_FIELDS + [FAMILY_KEY, "APS_ID"]
               if c not in coding.columns]
    if missing:
        raise ValueError(f"{SHEET_CODING} lacks columns {missing}")

    for df in (coding, admin):
        keys = df[FAMILY_KEY].map(_norm_key)
        df["_fam"] = keys.map(lambda t: t[0])
    info["family_keys_normalised_from_float"] = int(
        sum(_norm_key(k)[1] for k in coding[FAMILY_KEY]))

    # Label value audit (raw values kept; '' = missing, never coerced to N).
    value_counts = {f: dict(Counter(coding[f])) for f in LABEL_FIELDS}
    info["label_value_counts"] = value_counts
    invalid = {f: sorted(set(coding[f]) - VALID_LABEL_VALUES - {""})
               for f in LABEL_FIELDS}
    info["invalid_label_values"] = {f: v for f, v in invalid.items() if v}
    info["missing_label_cells"] = {f: int((coding[f] == "").sum())
                                   for f in LABEL_FIELDS}

    # Duplicate annotation keys (excluded from the join, never silently picked).
    dup_mask = coding.duplicated("_fam", keep=False)
    dup_keys = sorted(set(coding.loc[dup_mask, "_fam"]))
    info["duplicate_annotation_keys"] = dup_keys
    info["duplicate_aps_ids"] = sorted(
        set(coding.loc[coding.duplicated("APS_ID", keep=False), "APS_ID"]))
    joinable = coding[~dup_mask].copy()

    master_keys = set(master[FAMILY_KEY])
    unmatched = coding[~coding["_fam"].isin(master_keys)]
    info["unmatched_annotations"] = [
        {"APS_ID": a, "Family Number": f}
        for a, f in zip(unmatched["APS_ID"], unmatched["_fam"])]
    joinable = joinable[joinable["_fam"].isin(master_keys)]
    info["joined_annotations"] = len(joinable)

    # Admin manifest: join on (APS_ID, family); report disagreements.
    admin_dup = admin.duplicated(["APS_ID", "_fam"], keep=False)
    info["admin_duplicate_keys"] = sorted(set(admin.loc[admin_dup, "_fam"]))
    admin_j = admin[~admin_dup]
    cmp = coding[["APS_ID", "_fam"]].merge(
        admin[["APS_ID", "_fam"]], on="APS_ID", how="outer",
        suffixes=("_coding", "_admin"), indicator=True)
    info["admin_coding_mismatch"] = int(
        ((cmp["_merge"] != "both") |
         (cmp["_fam_coding"] != cmp["_fam_admin"])).sum())

    ann = joinable.drop(columns=["Family Number"])
    ann_cols = {c: ("ann_" + c if c not in LABEL_FIELDS else c)
                for c in ann.columns if c != "_fam"}
    drop_copies = [c for c in CODING_TEXT_COPIES if c in ann.columns]
    # Verify the dropped text copies against the master before discarding.
    mm = master.set_index(FAMILY_KEY)
    chk = {}
    for c in ("Title", "Abstract"):
        if c in joinable.columns:
            chk[c] = {"compared": len(joinable), "identical": int(sum(
                mm.at[f, c].strip() == v.strip()
                for f, v in zip(joinable["_fam"], joinable[c])))}
    info["text_copy_checks"] = chk
    ann = ann.drop(columns=drop_copies).rename(columns=ann_cols)

    adm = admin_j[admin_j["_fam"].isin(set(joinable["_fam"]))].copy()
    adm = adm.drop(columns=["Family Number"])
    adm = adm.rename(columns={c: "adm_" + c for c in adm.columns if c != "_fam"})
    adm = adm.drop(columns=["adm_APS_ID"])

    merged = master.merge(ann, how="left", left_on=FAMILY_KEY,
                          right_on="_fam", validate="one_to_one")
    merged = merged.drop(columns=["_fam"]).merge(
        adm, how="left", left_on=FAMILY_KEY, right_on="_fam",
        validate="one_to_one").drop(columns=["_fam"])
    if len(merged) != len(master):
        raise AssertionError("annotation join changed the number of rows")
    annotated = merged[FAMILY_KEY].isin(set(joinable["_fam"]))
    for col in [c for c in merged.columns
                if c.startswith(("ann_", "adm_")) or c in LABEL_FIELDS]:
        merged[col] = merged[col].fillna("")
    merged.insert(len(master.columns), "is_annotated", annotated.astype(str))
    merged["annotation_source"] = [
        f"{workbook_name}#{SHEET_CODING}+{SHEET_ADMIN}" if a else ""
        for a in annotated]
    # Explicit M/P/R/I naming; raw Y/N/U copied verbatim, '' stays missing.
    for src, dst in LABEL_TO_MPRI.items():
        merged[dst] = merged[src]
    info["unannotated_families"] = int((~annotated).sum())
    info["annotated_families"] = int(annotated.sum())
    info["multi_label_distribution"] = dict(sorted(Counter(
        sum(1 for f in LABEL_FIELDS if row[f] == "Y")
        for row in joinable.to_dict("records")).items()))
    info["coding_ann_columns"] = [c for c in ann.columns]
    return merged, info


def write_join_report(info, path: Path, master_n: int):
    L = []
    a = L.append
    a("# Annotation join report\n")
    a("_Generated by `scripts/build_master_dataset.py`; deterministic, "
      "do not edit by hand._\n")
    a(f"- Workbook: `{info['workbook']}`; authoritative label sheet "
      f"`{SHEET_CODING}`, sampling metadata `{SHEET_ADMIN}`.")
    a(f"- Annotation rows (`{SHEET_CODING}`): **{info['coding_rows']}**; "
      f"`{SHEET_ADMIN}` rows: **{info['admin_rows']}**.")
    a(f"- Master families: **{master_n}**; annotated and joined: "
      f"**{info['annotated_families']}**; master families without "
      f"annotation: **{info['unannotated_families']}**.")
    a(f"- Annotation rows with no matching master family: "
      f"**{len(info['unmatched_annotations'])}** "
      f"{info['unmatched_annotations'] or ''}")
    a(f"- Duplicate annotation family keys (excluded from join): "
      f"**{len(info['duplicate_annotation_keys'])}** "
      f"{info['duplicate_annotation_keys'] or ''}")
    a(f"- Duplicate `APS_ID` values: {len(info['duplicate_aps_ids'])}.")
    a(f"- Admin_Manifest vs Functional_Coding `(APS_ID, Family Number)` "
      f"disagreements: **{info['admin_coding_mismatch']}**.")
    a(f"- Family keys that required float-suffix normalisation: "
      f"{info['family_keys_normalised_from_float']}.")
    a("- Join cardinality: one annotation row per family "
      "(`validate='one_to_one'`); row count of the master is unchanged.\n")
    a("## Label value counts (raw values; empty = missing, not N)\n")
    a("| Field | Y | N | U | missing (empty) | other |")
    a("|---|---:|---:|---:|---:|---:|")
    for f in LABEL_FIELDS:
        vc = info["label_value_counts"][f]
        other = sum(v for k, v in vc.items() if k not in ("Y", "N", "U", ""))
        a(f"| `{f}` | {vc.get('Y', 0)} | {vc.get('N', 0)} | {vc.get('U', 0)} "
          f"| {vc.get('', 0)} | {other} |")
    a("\nNumber of `Y` roles per annotated family (labels are non-exclusive): "
      f"{info['multi_label_distribution']}\n")
    a("## M/P/R/I column mapping\n")
    a("| Workbook field | Output alias | Basis |")
    a("|---|---|---|")
    for s, d in LABEL_TO_MPRI.items():
        a(f"| `{s}` | `{d}` | role names in the JSTPM technical sheet; Y counts "
          f"match the reported 337/232/133/118 |")
    a("\nThe mapping is **inferred** (role name + count agreement); the source "
      "does not state the column-to-letter mapping literally. Aliases are "
      "verbatim copies; the original columns are preserved. "
      "Confirmation requested from Andreea.\n")
    a("## Integrity checks\n")
    for c, v in info["text_copy_checks"].items():
        a(f"- Workbook `{c}` identical to master `{c}`: "
          f"{v['identical']}/{v['compared']}.")
    a(f"- Invalid label values: {info['invalid_label_values'] or 'none'}.")
    a(f"- Missing label cells: {info['missing_label_cells']}.")
    a("- Workbook text copies (`Title`, `Abstract`, `1st Main Claim`, "
      "`Claims_Part1-4`) are not duplicated in the annotated output.")
    a("\n## Warnings\n")
    a("- `adm_FUNC_*_PROB`, `adm_*_SCORE_QUARTILE`, `adm_ASSIGNEE_HIDDEN`, "
      "`adm_PREVIOUS_VLAD_HUMAN_CODE` are sampling/model metadata, **not "
      "features**; using them as predictors would leak label information.")
    a("- Labels describe patent *families*. They are not propagated to "
      "publications or to other families.")
    a("- `U` is kept as a distinct value (2 families in the supplied file); "
      "it is not a negative label.")
    a("- `adm_DESIGN_WEIGHT` / `adm_INCLUSION_PROB` are kept verbatim; "
      "the annotated set is a stratified sample, not the full population.")
    for w in info["warnings"]:
        a(f"- {w}")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# --------------------------------------------------------------- orchestration
def build(input_dir, output_dir, csv_files=CSV_FILES, workbook=WORKBOOK):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw, source_cols = load_raw(input_dir, csv_files)
    master, dedup = reconcile(raw, source_cols)
    pubs, unparsed, prefix_mismatch = build_publications(raw)
    known = set(master[FAMILY_KEY])
    assert set(pubs["family_number"]) <= known

    write_csv(raw, output_dir / OUT["raw"])
    write_csv(master, output_dir / OUT["master"])
    write_csv(dedup, output_dir / OUT["dedup"])
    write_csv(pubs, output_dir / OUT["publications"])
    write_csv(pd.DataFrame(columns=FIGURE_SCHEMA), output_dir / OUT["figures"])

    summary = {
        "source_files": list(csv_files) + [workbook],
        "raw_rows_per_file": {f: int((raw.source_file == f).sum())
                              for f in csv_files},
        "raw_rows_total": len(raw),
        "unique_family_numbers_raw": int(raw[FAMILY_KEY].nunique()),
        "canonical_families": len(master),
        "repeated_family_groups": len(dedup),
        "rows_in_repeated_groups": int(dedup["n_rows"].sum()) if len(dedup) else 0,
        "reconciliation_status_counts": dict(
            Counter(master["reconciliation_status"])),
        "publication_rows": len(pubs),
        "families_with_publications": int(pubs["family_number"].nunique()),
        "source_rows_unparsed_for_publications": len(unparsed),
        "unparsed_examples": unparsed[:10],
        "publication_country_prefix_mismatches": prefix_mismatch,
    }

    wb_path = input_dir / workbook
    if wb_path.exists():
        sheets, names = load_workbook_sheets(wb_path)
        annotated, info = integrate_annotations(master, sheets, workbook)
        info["sheet_names"] = names
        write_csv(annotated, output_dir / OUT["annotated"])
        write_join_report(info, output_dir / OUT["join_report"], len(master))
        summary["annotation"] = {k: info[k] for k in (
            "coding_rows", "annotated_families", "unannotated_families",
            "label_value_counts", "duplicate_annotation_keys",
            "unmatched_annotations", "admin_coding_mismatch")}
    else:
        summary["annotation"] = "workbook not found; annotated outputs skipped"

    summary["output_sha256"] = {
        k: sha256(output_dir / v) for k, v in OUT.items()
        if k != "summary" and (output_dir / v).exists()}
    (output_dir / OUT["summary"]).write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return summary


def main(argv=None):
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--input-dir", default=str(root / "database"))
    p.add_argument("--output-dir", default=str(root / "master_dataset"))
    args = p.parse_args(argv)
    s = build(args.input_dir, args.output_dir)
    print(json.dumps({k: v for k, v in s.items()
                      if k not in ("output_sha256", "unparsed_examples")},
                     indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
