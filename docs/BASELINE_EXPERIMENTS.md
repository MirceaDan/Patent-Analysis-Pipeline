# First M/P/R/I baselines (text-only vs text + image descriptions)

Code: `mpri_baseline/` (separate from the stdlib FCA prototype `patent_mpri/`, which is unchanged).
Audit: `docs/TRAINING_DATASET_AUDIT.md`. Artifacts: `experiments/baseline_v1/` (default grouping) and
`experiments/baseline_v1_assignee_groups/` (sensitivity). Each holds `config.json`, `splits.csv`, `predictions_*.csv`,
`metrics.json`, `results.md`, `run.log`, `checkpoints/*.joblib`.

```bash
python scripts/integrate_figure_descriptions.py && python scripts/audit_training_dataset.py
python -m mpri_baseline.run_experiments --out experiments/baseline_v1                 # ~1 min, CPU
python -m mpri_baseline.run_experiments --out experiments/baseline_v1_assignee_groups --group-mode assignee
python -m pytest tests -q
```
Needs `requirements-baseline.txt` (scikit-learn, pandas; already present in the Anaconda host environment).
No GPU or VLM is used here; the VLM is offline preprocessing.

## Formulation
- Targets: four **independent** binary labels y_k in {Y=1, N=0, unknown} (141 families are all-N, 153 have 2 Y, 93 have 3-4 Y; so no softmax/argmax).
- Mask m_ik = 1 iff y_ik in {Y,N}. Loss: L = -sum_ik m_ik [y log p + (1-y) log(1-p)] + (1/2C) sum_k ||w_k||^2 (per-label logistic regression trained on known entries only; `class_weight=balanced`). Metrics also use only known entries. U is never N.
- Eligible A: annotated, title and abstract present, >=1 known label -> **598** families (2 all-U rows excluded: no supervision). B/C: eligible and >=1 usable description -> **23** families, 90 descriptions.
- Features: TF-IDF (word 1-2 grams, min_df 2, max 50k, sublinear, English stop words, fitted on train only). Text = `Title . Abstract`; C appends `image_descriptions` (empty = nothing appended). Never used: `adm_*`, `ann_*`, assignee, claims, citations, ids (`assert_no_forbidden_features`).
- Model selection: C in {0.1, 1, 10, 100} by validation macro average precision (fallback C=1 if no label is evaluable). Threshold 0.5.
- Splits: 20 repeats (seeds 2026..2045), StratifiedGroupKFold(5): fold 0 test, fold 1 val, rest train (~60/20/20), stratified on the Y/N combination. One assignment on all 598 families is reused for A, B, C. Groups: priority/application/near-duplicate text links (none exist among the 600, so each family is its own group). Sensitivity: families sharing an assignee are grouped. Citation links (255) chain 155 families and are only counted.
- Experiments: **A** text on all 598; **B** text on the 23; **C** text+descriptions on the same 23, same splits/seeds; **A on BC-test** = model A scored on exactly the B/C test families.

## Results (pooled test predictions over 20 repeats; `baseline_v1/results.md` has per-label tables)
| Exp | test rows | macro AP | macro F1 | micro F1 | macro ROC-AUC |
|---|---|---|---|---|---|
| A | 2399 | 0.671 | 0.637 | 0.668 | 0.767 |
| A on BC-test | 97 | 0.416 | 0.294 | 0.392 | 0.568 |
| B | 97 | 0.261 | 0.246 | 0.314 | 0.445 |
| C | 97 | 0.294 | 0.307 | 0.358 | 0.520 |

A per label (AP / F1): M 0.808/0.751, P 0.662/0.632, R 0.598/0.576, I 0.615/0.590.

- **A vs B** (-0.41 macro AP): the image subset is a 23-family, non-random set; training on 8-13 families gives near-chance ROC-AUC (0.445). This reflects size, not a property of that subset. The cleaner contrast is **A on BC-test vs B** (same test families): B is 0.155 macro AP lower (CI -0.30..-0.01), i.e. more training data helps.
- **B vs C** (C minus B): macro AP +0.033 (CI -0.017..0.084), macro F1 +0.061 (-0.015..0.132), micro F1 +0.044. Per label AP: M +0.118, P -0.073, R +0.027, I +0.061; F1: M +0.063, P -0.079, R +0.123, I +0.137. All aggregate CIs include 0; P goes down. In the assignee-grouped sensitivity run the difference disappears (macro AP 0.441 vs 0.430, macro F1 0.248 vs 0.242). **No evidence that image descriptions help.**

## Limitations
23 families with 3 R and 3 I positives: B/C per-label numbers are noise; CIs resample families but repeats are not independent; subset is diversity-selected and US/DE-only; descriptions are unreviewed 3B-VLM output (30 lack a revision); sampling was stratified by assignee (up to 20 families per assignee) so scores are optimistic for unseen assignees (assignee-grouped A: macro AP 0.659, F1 0.605 vs 0.671/0.637); F_* -> M/P/R/I mapping is inferred; probabilities are uncalibrated.

## Next most valuable step
Describe more eligible families (15 more eligible families have figures but were outside the pilot scope) and, above all, solve the CN-only gap (PatBase/OPS); then human-review a sample of descriptions before re-running B vs C on >= ~150 families.
