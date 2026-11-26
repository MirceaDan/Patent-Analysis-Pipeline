# Baseline results (baseline_v1)

Pooled over repeated grouped splits (test predictions of all repeats concatenated; the same family appears in several repeats, so rows are NOT independent). `n/a` = undefined (no positives or no negatives in the test rows). Threshold 0.5 for F1.

| Experiment | test rows | macro AP | macro F1 | micro F1 | macro ROC-AUC | Hamming (known only) |
|---|---|---|---|---|---|---|
| A  text, all eligible | 2399 | 0.671 | 0.637 | 0.668 | 0.767 | 0.221 |
| A on BC-test (model A, B/C test families) | 97 | 0.416 | 0.294 | 0.392 | 0.568 | 0.247 |
| B  text, image subset | 97 | 0.261 | 0.246 | 0.314 | 0.445 | 0.371 |
| C  text+image descriptions, image subset | 97 | 0.294 | 0.307 | 0.358 | 0.520 | 0.397 |

### A  text, all eligible: per label (pooled)

| Role | pos | neg | unk excl. | prevalence | P | R | F1 | AP | ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|
| M | 1349 | 1050 | 0 | 0.562 | 0.752 | 0.749 | 0.751 | 0.808 | 0.765 |
| P | 946 | 1453 | 0 | 0.394 | 0.646 | 0.619 | 0.632 | 0.662 | 0.737 |
| R | 541 | 1858 | 0 | 0.226 | 0.636 | 0.527 | 0.576 | 0.598 | 0.774 |
| I | 482 | 1917 | 0 | 0.201 | 0.688 | 0.517 | 0.590 | 0.615 | 0.793 |

### A on BC-test (model A, B/C test families): per label (pooled)

| Role | pos | neg | unk excl. | prevalence | P | R | F1 | AP | ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|
| M | 48 | 49 | 0 | 0.495 | 0.657 | 0.479 | 0.554 | 0.712 | 0.634 |
| P | 26 | 71 | 0 | 0.268 | 0.200 | 0.115 | 0.146 | 0.249 | 0.476 |
| R | 13 | 84 | 0 | 0.134 | n/a | 0.000 | 0.000 | 0.112 | 0.391 |
| I | 12 | 85 | 0 | 0.124 | 0.556 | 0.417 | 0.476 | 0.589 | 0.771 |

### B  text, image subset: per label (pooled)

| Role | pos | neg | unk excl. | prevalence | P | R | F1 | AP | ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|
| M | 48 | 49 | 0 | 0.495 | 0.538 | 0.438 | 0.483 | 0.485 | 0.496 |
| P | 26 | 71 | 0 | 0.268 | 0.300 | 0.346 | 0.321 | 0.327 | 0.506 |
| R | 13 | 84 | 0 | 0.134 | 0.048 | 0.077 | 0.059 | 0.114 | 0.388 |
| I | 12 | 85 | 0 | 0.124 | 0.095 | 0.167 | 0.121 | 0.116 | 0.388 |

### C  text+image descriptions, image subset: per label (pooled)

| Role | pos | neg | unk excl. | prevalence | P | R | F1 | AP | ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|
| M | 48 | 49 | 0 | 0.495 | 0.529 | 0.562 | 0.545 | 0.603 | 0.587 |
| P | 26 | 71 | 0 | 0.268 | 0.200 | 0.308 | 0.242 | 0.254 | 0.457 |
| R | 13 | 84 | 0 | 0.134 | 0.129 | 0.308 | 0.182 | 0.141 | 0.530 |
| I | 12 | 85 | 0 | 0.124 | 0.211 | 0.333 | 0.258 | 0.177 | 0.503 |

## Per-repeat summary (mean ± sd over repeats where defined)

| Experiment | repeats | macro AP | macro F1 | micro F1 |
|---|---|---|---|---|
| A  text, all eligible | 20 | 0.733 ± 0.033 | 0.634 ± 0.045 | 0.667 ± 0.033 |
| A on BC-test (model A, B/C test families) | 20 | 0.641 ± 0.173 | 0.279 ± 0.274 | 0.360 ± 0.283 |
| B  text, image subset | 20 | 0.582 ± 0.188 | 0.210 ± 0.154 | 0.302 ± 0.217 |
| C  text+image descriptions, image subset | 20 | 0.623 ± 0.108 | 0.233 ± 0.146 | 0.325 ± 0.171 |

## Comparisons (pooled differences, family-level paired bootstrap 95% CI)

**A_vs_B** (diff = B minus A): effect of restricting training AND test to the image subset (different test sets; confounded by size and test composition)

| Metric | diff | 95% CI |
|---|---|---|
| macro_average_precision | -0.410 | None |
| macro_f1 | -0.391 | None |
| micro_f1 | -0.354 | None |
| M:average_precision | -0.323 | None |
| M:f1 | -0.268 | None |
| P:average_precision | -0.335 | None |
| P:f1 | -0.311 | None |
| R:average_precision | -0.484 | None |
| R:f1 | -0.518 | None |
| I:average_precision | -0.499 | None |
| I:f1 | -0.469 | None |

**A_on_BC_test_vs_B** (diff = B minus A_on_BC_test): same test families; training on all eligible families (A) vs only the image subset (B)

| Metric | diff | 95% CI |
|---|---|---|
| macro_average_precision | -0.155 | [-0.299, -0.011] |
| macro_f1 | -0.048 | [-0.219, 0.085] |
| micro_f1 | -0.078 | [-0.226, 0.062] |
| M:average_precision | -0.227 | [-0.388, -0.03] |
| M:f1 | -0.071 | [-0.214, 0.084] |
| P:average_precision | 0.078 | [-0.049, 0.265] |
| P:f1 | 0.175 | [-0.005, 0.367] |
| R:average_precision | 0.002 | [-0.04, 0.042] |
| R:f1 | 0.059 | [0.0, 0.167] |
| I:average_precision | -0.473 | [-0.946, -0.016] |
| I:f1 | -0.355 | [-0.842, 0.062] |

**B_vs_C** (diff = C minus B): contribution of image descriptions: same families, splits, seeds, hyperparameter grid

| Metric | diff | 95% CI |
|---|---|---|
| macro_average_precision | 0.033 | [-0.017, 0.084] |
| macro_f1 | 0.061 | [-0.015, 0.132] |
| micro_f1 | 0.044 | [-0.021, 0.103] |
| M:average_precision | 0.118 | [-0.006, 0.23] |
| M:f1 | 0.063 | [-0.045, 0.161] |
| P:average_precision | -0.073 | [-0.237, 0.04] |
| P:f1 | -0.079 | [-0.185, 0.025] |
| R:average_precision | 0.027 | [-0.009, 0.092] |
| R:f1 | 0.123 | [0.0, 0.255] |
| I:average_precision | 0.061 | [-0.007, 0.216] |
| I:f1 | 0.137 | [-0.008, 0.348] |

## Limitations

- B/C contain only 23 families; label support for R and I is ~3 positives. Per-label metrics are extremely noisy and the B-vs-C comparison cannot support a conclusion about image descriptions.
- The image subset is not a random sample (diversity-selected pilot) and is restricted to publications where Google Patents lists figures (US/DE here, no CN).
- Repeated splits reuse the same families, so pooled test rows and repeats are not independent; bootstrap CIs resample families but understate uncertainty from the small training sets.
- Descriptions come from a 3B VLM and have NOT been human-reviewed; 30 of them lack a recorded model revision.
- Default grouping has no related-family links (each family is its own group); citation and assignee relatedness are not controlled unless --group-mode assignee is used. The sample is stratified by assignee.
- Sampling was stratified (not a population sample); metrics are not population estimates; labels F_* -> M/P/R/I mapping is inferred.
- Checkpoints are scikit-learn pickles tied to the recorded library versions.
