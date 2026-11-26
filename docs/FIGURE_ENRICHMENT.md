# Patent figure retrieval and VLM enrichment

Script: `scripts/enrich_patent_figures.py` · Tests: `tests/test_figure_enrichment.py` (45, no network, no VLM).
Independent of the master builder and the classifier. Does not modify `master_dataset/`.

## Pipeline
1. Read the 600 annotated families (`master_dataset/patent_families_annotated.csv`, `is_annotated == True`) and their publications (`patent_publications.csv`).
2. Order publications: master first (`is_master_publication == "True"`, exactly one per family), then US, EP, WO, DE, KR, JP, ..., CN last.
3. Normalise each to a Google Patents document id (e.g. `US2020334401`+`AA` -> `US20200334401A1`; `WO23247172` -> `WO2023247172`).
4. Fetch `https://patents.google.com/patent/<id>/en` (allowed by robots.txt; rate limited; no auth/CAPTCHA bypass).
5. **Verify**: the page's own document id must equal the requested id, and the page date must equal the dataset publication date when both exist. Title similarity is never used. Result recorded as `match_verification`.
6. List figures (`itemprop=images`), keeping only images whose file name carries the page's own document id.
7. Download each image into memory only (size cap, Pillow validation, downscale to 2000 px). Nothing is written to disk.
8. Send to an OpenAI-compatible VLM with a strict JSON schema (`fig-desc-v1`): figure_type, purpose, components (visible/inferred), relations, process_or_mechanism, text_in_figure, uncertainties, description. No M/P/R/I classification and no confidence field.
9. Append to journals immediately; export CSVs atomically.

## Outputs (`data/figure_enrichment/` by default)
- `figure_enrichments.csv`: one row per figure. First 14 columns equal the canonical `master_dataset/figure_enrichments.csv`; extensions: image_sha256, image_bytes, figure_callout_labels, publication_role, match_verification, vlm_config_hash, token counts, attempts, reused_from, description_json.
- `document_resolution.csv`: one row per publication examined.
- `family_summary.csv`: per family counts, `has_usable_description`, `family_status`, `failure_reason`.
- `resolutions.jsonl`, `figures.jsonl`: append-only resume journals. `run.log`: log (no secrets).

Statuses
- Document: `resolved_with_figures`, `resolved_no_figures`, `document_not_found`, `document_inaccessible`, `identifier_mismatch`, `unsupported_identifier`.
- Figure: `pending`, `described`, `image_inaccessible`, `vlm_failed`, `skipped_limit`.
- Family: `not_yet_processed`, `document_not_found`, `document_inaccessible`, `no_figures_found`, `figures_inaccessible`, `vlm_failed`, `figures_discovered_pending_vlm`, `described`.

A missing description is never evidence that the patent lacks a capability.

Join: `family_number` = `Family Number`; and `publication_number` + `kind_code` (dataset form) = `patent_publications.csv`.

## Commands
```
# Dry run (no VLM), small sample
python scripts/enrich_patent_figures.py --dry-run --sample 40 --seed 20261123

# Check the VLM (synthetic image)
export PATENT_VLM_BASE_URL=... PATENT_VLM_MODEL=... PATENT_VLM_API_KEY=...
python scripts/enrich_patent_figures.py --vlm-check

# Pilot (families with figures only are meaningful; see coverage below)
python scripts/enrich_patent_figures.py --family-ids <ids> --max-families 8

# Full annotated set (600 families; default scope; not started)
# Never pass --all-families: that targets all 28,311 families.
python scripts/enrich_patent_figures.py --workers 1

# Resume: re-run the identical command; completed figures with the same URL and VLM config hash are skipped.
```
Exit codes: 0 ok, 2 VLM not configured, 3 `--vlm-check` failed, 4 stopped early.
Optional background run: `nohup <command> > data/figure_enrichment/nohup.out 2>&1 &` (none has been started).

## Measured coverage (Google Patents, dry run only)
- Random 40 families (mostly CN master): 62 documents resolved and verified; only 1 had figures. **CN documents on Google Patents expose no figure images.**
- 74 annotated families with a non-CN publication: 38 have figures (US/DE; 408 figures listed, median ~7), 30 no figures, 6 not found (IN, AU, ES, ZA and some DE kind codes).
- Estimate (not measured on all 600): ~6% of annotated families (~38) obtainable via Google; CN-only families (~526) are not.

## Architecture: VLM is offline preprocessing
The VLM is not part of the classifier. The crawler retrieves figures, runs inference and persists text plus provenance (publication number, figure id, URL, checksum, `vlm_model`, `vlm_model_version`, `vlm_config_hash`, `prompt_version`). Images live in memory only. The downstream classifier consumes text only (title, abstract, VLM descriptions): no image tensors, visual embeddings or VLM calls in its forward pass. Training code is untouched.

## VLM backends
`--vlm-backend api` (default): OpenAI-compatible endpoint via `PATENT_VLM_BASE_URL/MODEL/API_KEY`.

`--vlm-backend local`: separate module `scripts/vlm_local.py` (Hugging Face `transformers`, greedy decoding, lazy imports, no weights in the repo; weights download to the HF cache). Same prompts, parser and schema as the API backend.
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132   # CUDA build
pip install -r requirements-vlm-local.txt
python scripts/enrich_patent_figures.py --vlm-backend local \
    --local-model Qwen/Qwen2.5-VL-7B-Instruct --local-revision <commit> --local-4bit --vlm-check
```
Pin `--local-revision` for reproducibility (otherwise the resolved commit is recorded in `vlm_model_version`). Changing model, revision, dtype, quantisation or prompt changes `vlm_config_hash`, so affected figures are reprocessed on resume.

Environment inspected: RTX 4080 SUPER (16 GB), 125 GB RAM, Hugging Face reachable. Installed (verified): torch 2.15.0+cu132, torchvision 0.29.2+cu132, transformers 5.24.0, accelerate 1.16.0, bitsandbytes 0.50.2; CUDA works on the RTX 4080 SUPER. Local inference has **not yet been run** with real weights; only fake-model unit tests exist. A 7B model needs `--local-4bit` (or `Qwen/Qwen2.5-VL-3B-Instruct` in bf16) to fit in 16 GB. Output quality is unverified until a pilot.

## Limitations / open decisions
- No VLM is configured or verified in this environment, so no descriptions exist and no cost is known.
- CN figures need another source: render drawing pages of the PDF (`citation_pdf_url`; composite-page workflow, not implemented), or EPO OPS credentials.
- Google Patents terms of service for bulk use have not been reviewed beyond robots.txt; the image host's robots.txt returns 403, so images are fetched only from URLs referenced by allowed pages.
- `AD` / `D1` kind-code mappings are unverified.

## Running in a container (recommended; leaves the host Python untouched)
`docker/Dockerfile` + `docker/run.sh`. Base image is pulled from `mirror.gcr.io` (Docker Hub is not usable through the corporate proxy). The repo is mounted **read-only**; only `data/figure_enrichment/` is writable; model weights live in the named Docker volume `patent-hf-cache` (not `~/.cache/huggingface`).
```
docker/run.sh build
docker/run.sh gpu-check      # verified: torch 2.15.0+cu132, RTX 4080 SUPER visible
docker/run.sh test           # verified: 114 passed, 1 skipped
export PATENT_VLM_LOCAL_MODEL=Qwen/Qwen2.5-VL-3B-Instruct
docker/run.sh crawl --vlm-backend local --vlm-check
docker/run.sh crawl --vlm-backend local --sample 8 --seed 20261123     # pilot
```
Remove everything: `docker rmi patent-figure-enrich; docker volume rm patent-hf-cache`.

### Container networking notes (verified)
Behind the corporate TLS-inspecting proxy the container needs the host CA bundle (`docker/run.sh` mounts `/etc/ssl/certs/ca-certificates.crt` read-only) and `HF_HUB_DISABLE_XET=1` (the Xet CDN host is blocked). After the first download, `export HF_HUB_OFFLINE=1` (forwarded by run.sh) avoids Hub calls.

## Stage B pilot result (local Qwen/Qwen2.5-VL-3B-Instruct, bf16, RTX 4080 SUPER)
8 annotated families (7 US application publications, 1 DE), `--max-figures-per-family 5`, run 2026-11-24 in the container:
- 127 figures discovered, 40 attempted (cap), **30 described**, 10 `vlm_failed`, 87 `skipped_limit`. All 8 families have at least one description.
- Failures were output-format failures: 7 truncated JSON (the 1200-token limit), 2 invalid evidence value, 1 non-JSON. Greedy decoding makes retries identical, so these will fail again unless `max_new_tokens` or the prompt changes (which changes `vlm_config_hash`).
- Tokens: 109,817 prompt / 23,093 completion. Local inference: no monetary cost; about 20 s per figure wall-clock including image download delays.
- Descriptions were checked only for plausibility against family titles; they were not compared visually with every source image, and 3B-model errors (e.g. guessing meanings of reference numerals) are expected.

## PatBase client (`scripts/patbase_client.py`)
Source: official REST 2.0 document (https://www.patbase.com/rest/2.0/PatBaseRestAPI.pdf). Documented auth: login (user id + password -> `SessionFarm_GUID` cookie) or an API key (extra cost, issued by Minesoft); API access must be enabled on the account (support@minesoft.com). **An e-mail alone does not authenticate.** Figures: `getrecord&family=<Family Number>` returns publications with `IMG` (image page count); pages are fetched from `/phpmosaic/getone.php?pn=&kd=&pg=`. Dataset `Family Number` / `publication_number` / `kind_code` are PatBase identifiers and are used unchanged.
```
export PATBASE_EMAIL=<user id>          # never committed
read -rs PATBASE_PASSWORD; export PATBASE_PASSWORD   # or PATBASE_API_KEY; not echoed, not in history
python scripts/patbase_client.py --check-access
python scripts/patbase_client.py --fetch-figure --out-dir data/patbase_test
```
Status values: `authorized`, `no_credentials_supplied`, `wrong_credentials`, `api_not_enabled_for_account`, `authentication_required`, `insufficient_permissions`, `network_error`, `tls_error`, `unexpected_or_undocumented_response`. Single attempt per stage; no retries or fallback endpoints.
Observed (2026-11-24, from this machine): the unauthenticated `version` (2.0) and `getweek` calls succeed through the proxy; authenticated access and figure retrieval are untested (no password/API key available). Whether the image endpoint serves CN publications and returns usable single-figure images vs whole drawing sheets is unknown until tested. Image pages may be composite sheets.

## Google Patents POC (bounded, 20 families)

Reproduce (all VLM work in Docker; nothing installed on the host):

```bash
export PATENT_VLM_LOCAL_MODEL=Qwen/Qwen2.5-VL-3B-Instruct \
       PATENT_VLM_LOCAL_REVISION=66285546d2b821cf421d4f5eb2576359d3770cd3 HF_HUB_OFFLINE=1
docker/run.sh crawl --dry-run --max-figures-per-family 5 --page-delay 2   # eligibility discovery, no VLM
python scripts/figure_poc_tools.py eligible    # eligible_families.csv, eligibility_summary.json
python scripts/figure_poc_tools.py select      # pilot_families.csv (seeded greedy feature coverage)
docker/run.sh crawl --vlm-backend local --family-ids <ids from pilot_families.csv> \
       --max-figures-per-family 5 --page-delay 2 --image-delay 2
python scripts/figure_poc_tools.py review      # review_figures.csv, review_summary.json
```

Results (outputs in `data/figure_enrichment/`, separate from `master_dataset/`):

- **Eligible families: 38 of 600.** Eligibility = a verified document listing >=1 figure whose image was
  not marked inaccessible; country is not used (36 US, 2 DE happen to qualify). Excluded: 556
  `no_figures_found` (Google lists no figures; includes all CN), 6 `document_not_found`.
- **Pilot:** 20 families chosen by seeded greedy coverage of label values and IPC subclasses (21/21 covered),
  max 5 figures each. 186 figures discovered, 96 attempted, **80 described, 16 failed (83.3%)**;
  90 skipped by the 5-per-family cap. Failures: 10 `truncated_output` (still hit 2048 tokens),
  6 `invalid_output` (4 bad evidence value, 1 empty description, 1 extra JSON data).
  80 includes 30 descriptions reused from the earlier 8-family pilot.
- `max_new_tokens` 2048 did not remove truncation; greedy retries are pointless (deterministic).
- **Semantic quality is NOT assessed.** `review_figures.csv` has blank review fields
  (component/relationship/text-in-figure accuracy, overall usefulness) awaiting a human comparing each
  `image_source_url` to its description.

## Single-patent end-to-end POC (family 75198509)

One patent, run through crawl → VLM → classify → report. **n=1; not an evaluation.**
All commands run in Docker (export `PATENT_VLM_LOCAL_MODEL`, `PATENT_VLM_LOCAL_REVISION`, `HF_HUB_OFFLINE=1` first).

```
docker/run.sh poc select      # fixed rule; selection.json withholds human labels
docker/run.sh crawl --vlm-backend local --family-ids 75198509 --max-figures-per-family 10 \
  --page-delay 2 --image-delay 2 --out-dir data/figure_enrichment/poc_single_patent/crawl \
  --save-raw-dir data/figure_enrichment/poc_single_patent/raw
docker/run.sh poc assess      # classifies from title/abstract/descriptions only; writes predictions.sha256
docker/run.sh poc report      # verifies hash, THEN reads human labels; writes REPORT.md
```

Outputs: `data/figure_enrichment/poc_single_patent/` (`REPORT.md`, `predictions.json`, `comparison.json`, `raw/`, `crawl/`).
Classifier: existing `experiments/baseline_v1/checkpoints/A_repeat01.joblib` (family in its test split), raw uncalibrated scores,
plus a labelled keyword baseline. Limitations: 2 of 3 figures failed VLM generation (truncation at 2048 tokens), the page has no
description text, labels were coded from claims which the classifier did not see.
