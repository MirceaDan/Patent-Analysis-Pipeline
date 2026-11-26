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

## Enriched family dataset (figure descriptions integrated)

```bash
python scripts/build_master_dataset.py              # as above -> master_dataset/*
python scripts/integrate_figure_descriptions.py     # -> patent_families_enriched.csv + figure_enrichment_integration.json
python scripts/audit_training_dataset.py            # -> docs/TRAINING_DATASET_AUDIT.md + master_dataset/training_dataset_audit.json
```

`patent_families_enriched.csv` = `patent_families_annotated.csv` (same 28,311 rows and order, all original columns,
labels and `label_M/P/R/I` aliases untouched, `U` preserved) + five columns:

| Column | Meaning |
|---|---|
| `image_descriptions` | usable VLM descriptions joined by `\n` (master publication first, natural figure order); `""` if none |
| `n_image_descriptions` | number of lines in `image_descriptions` |
| `image_description_provenance` | JSON list aligned with the lines: publication, kind, figure id, source document id/url, image url, image sha256, VLM model + revision, prompt version, config hash, timestamp |
| `figure_enrichment_status` | `described`, `vlm_failed`, `image_inaccessible`, `figures_discovered_not_processed`, `no_figures_listed_by_source`, `document_not_found`, `document_inaccessible`, `not_enriched` |
| `figure_enrichment_detail` | JSON: publications total/examined, resolution and figure status counts |

Lineage: Google Patents crawl (`scripts/enrich_patent_figures.py`, `docker/run.sh`) -> `data/figure_enrichment/{figure_enrichments,document_resolution}.csv`
-> `integrate_figure_descriptions.py` (joined on family only; one row per family by construction, checked) -> enriched CSV.
`no_figures_listed_by_source` means Google Patents lists no figures (true for every CN page), not that the patent has none.
`not_enriched` = outside the crawl scope (all 27,711 unannotated families). Status is derived from the resolution and figure
tables, not `family_summary.csv`, which only reflects the last crawler run. 30 descriptions from the first pilot have no
recorded model revision (see the audit warning).
