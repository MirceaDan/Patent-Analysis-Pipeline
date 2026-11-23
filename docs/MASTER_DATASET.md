# Master dataset build

Reproducible build of the family-level master dataset from the three PatBase
CSV exports and `sample_todo_completed.xlsx`.

```bash
python scripts/build_master_dataset.py          # reads database/, writes master_dataset/
python -m pytest tests/test_master_dataset.py -q # fixtures + checks on built outputs
```

Requires `pandas` and `openpyxl`. `database/` holds only the original files and is never written to; all derived
files live in `master_dataset/`. Sources are read-only; outputs are
deterministic (byte-identical on rerun; SHA-256s in `master_dataset/build_summary.json`).
CSVs are UTF-8 with BOM, comma-delimited, 36 identical columns in all three
files; every value is read and written as a string (no type coercion).

Outputs (in `master_dataset/`):

| Output | Content |
|---|---|
| `patent_records_raw.csv` | all source rows + `source_file`, `source_row` |
| `patent_families_master.csv` | one row per `Family Number`, all 36 columns + `source_files`, `source_rows`, `n_source_rows`, `reconciliation_status`, `differing_fields`, `field_alternates_json` |
| `patent_deduplication_report.csv` | one row per repeated family: rows, differing fields, distinct values, action |
| `patent_families_annotated.csv` | master + `is_annotated`, label fields (original names), `label_M/P/R/I` aliases, `ann_*` (Functional_Coding), `adm_*` (Admin_Manifest) |
| `annotation_join_report.md` | match counts, label counts, warnings |
| `patent_publications.csv` | one row per (family, publication number, kind, dates, application, country) |
| `figure_enrichments.csv` | empty schema for future VLM descriptions (no rows) |
| `build_summary.json` | counts and checksums |

Reconciliation policy and the label mapping are documented in the script's
module docstring. The M/P/R/I column mapping is inferred (role names plus
matching Y counts) and awaits confirmation. Annotation labels exist only for
the 600 sampled families and are not propagated.

`figure_enrichments.csv` `processing_status` values: pending, processed,
failed, skipped. `image_source_url` is metadata only; images are not stored.
