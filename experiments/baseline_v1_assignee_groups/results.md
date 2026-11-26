# Baseline results (baseline_v1_assignee_groups)

Pooled over repeated grouped splits (test predictions of all repeats concatenated; the same family appears in several repeats, so rows are NOT independent). `n/a` = undefined (no positives or no negatives in the test rows). Threshold 0.5 for F1.

| Experiment | test rows | macro AP | macro F1 | micro F1 | macro ROC-AUC | Hamming (known only) |
|---|---|---|---|---|---|---|
| A  text, all eligible | 2400 | 0.659 | 0.605 | 0.638 | 0.752 | 0.227 |
| A on BC-test (model A, B/C test families) | 183 | 0.507 | 0.309 | 0.477 | 0.585 | 0.314 |
| B  text, image subset | 170 | 0.441 | 0.248 | 0.334 | 0.506 | 0.404 |
| C  text+image descriptions, image subset | 170 | 0.430 | 0.242 | 0.329 | 0.504 | 0.409 |

### A  text, all eligible: per label (pooled)

| Role | pos | neg | unk excl. | prevalence | P | R | F1 | AP | ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|
| M | 1353 | 1047 | 0 | 0.564 | 0.735 | 0.684 | 0.709 | 0.773 | 0.727 |
| P | 941 | 1459 | 0 | 0.392 | 0.675 | 0.579 | 0.623 | 0.682 | 0.746 |
| R | 541 | 1859 | 0 | 0.225 | 0.677 | 0.375 | 0.483 | 0.540 | 0.732 |
| I | 481 | 1919 | 0 | 0.200 | 0.733 | 0.514 | 0.604 | 0.640 | 0.805 |

### A on BC-test (model A, B/C test families): per label (pooled)

| Role | pos | neg | unk excl. | prevalence | P | R | F1 | AP | ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|
| M | 134 | 49 | 0 | 0.732 | 0.882 | 0.612 | 0.722 | 0.867 | 0.727 |
| P | 83 | 100 | 0 | 0.454 | 0.143 | 0.024 | 0.041 | 0.385 | 0.407 |
| R | 27 | 156 | 0 | 0.148 | n/a | 0.000 | 0.000 | 0.177 | 0.564 |
| I | 60 | 123 | 0 | 0.328 | 0.724 | 0.350 | 0.472 | 0.596 | 0.644 |

### B  text, image subset: per label (pooled)

| Role | pos | neg | unk excl. | prevalence | P | R | F1 | AP | ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|
| M | 126 | 44 | 0 | 0.741 | 0.644 | 0.302 | 0.411 | 0.681 | 0.365 |
| P | 78 | 92 | 0 | 0.459 | 0.651 | 0.359 | 0.463 | 0.595 | 0.640 |
| R | 25 | 145 | 0 | 0.147 | 0.120 | 0.120 | 0.120 | 0.154 | 0.521 |
| I | 57 | 113 | 0 | 0.335 | n/a | 0.000 | 0.000 | 0.335 | 0.500 |

### C  text+image descriptions, image subset: per label (pooled)

| Role | pos | neg | unk excl. | prevalence | P | R | F1 | AP | ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|
| M | 126 | 44 | 0 | 0.741 | 0.804 | 0.294 | 0.430 | 0.763 | 0.503 |
| P | 78 | 92 | 0 | 0.459 | 0.528 | 0.359 | 0.427 | 0.487 | 0.557 |
| R | 25 | 145 | 0 | 0.147 | 0.103 | 0.120 | 0.111 | 0.135 | 0.456 |
| I | 57 | 113 | 0 | 0.335 | n/a | 0.000 | 0.000 | 0.335 | 0.500 |

## Per-repeat summary (mean ± sd over repeats where defined)

| Experiment | repeats | macro AP | macro F1 | micro F1 |
|---|---|---|---|---|
| A  text, all eligible | 20 | 0.734 ± 0.023 | 0.603 ± 0.031 | 0.638 ± 0.025 |
| A on BC-test (model A, B/C test families) | 20 | 0.622 ± 0.069 | 0.304 ± 0.047 | 0.471 ± 0.080 |
| B  text, image subset | 19 | 0.549 ± 0.056 | 0.220 ± 0.099 | 0.319 ± 0.132 |
| C  text+image descriptions, image subset | 19 | 0.548 ± 0.051 | 0.183 ± 0.129 | 0.281 ± 0.191 |

## Comparisons (pooled differences, family-level paired bootstrap 95% CI)

**A_vs_B** (diff = B minus A): effect of restricting training AND test to the image subset (different test sets; confounded by size and test composition)

| Metric | diff | 95% CI |
|---|---|---|
| macro_average_precision | -0.217 | None |
| macro_f1 | -0.356 | None |
| micro_f1 | -0.304 | None |
| M:average_precision | -0.093 | None |
| M:f1 | -0.298 | None |
| P:average_precision | -0.087 | None |
| P:f1 | -0.160 | None |
| R:average_precision | -0.386 | None |
| R:f1 | -0.363 | None |
| I:average_precision | -0.305 | None |
| I:f1 | -0.604 | None |

**A_on_BC_test_vs_B** (diff = B minus A_on_BC_test): same test families; training on all eligible families (A) vs only the image subset (B)

| Metric | diff | 95% CI |
|---|---|---|
| macro_average_precision | -0.065 | [-0.223, 0.047] |
| macro_f1 | -0.060 | [-0.324, 0.138] |
| micro_f1 | -0.143 | [-0.47, 0.138] |
| M:average_precision | -0.187 | [-0.368, -0.024] |
| M:f1 | -0.312 | [-0.601, -0.012] |
| P:average_precision | 0.210 | [0.022, 0.406] |
| P:f1 | 0.422 | [0.138, 0.609] |
| R:average_precision | -0.023 | [-0.101, 0.089] |
| R:f1 | 0.120 | [0.0, 0.226] |
| I:average_precision | -0.261 | [-0.807, 0.068] |
| I:f1 | -0.472 | [-0.938, 0.0] |

**B_vs_C** (diff = C minus B): contribution of image descriptions: same families, splits, seeds, hyperparameter grid

| Metric | diff | 95% CI |
|---|---|---|
| macro_average_precision | -0.011 | [-0.067, 0.053] |
| macro_f1 | -0.006 | [-0.113, 0.1] |
| micro_f1 | -0.006 | [-0.169, 0.167] |
| M:average_precision | 0.083 | [-0.048, 0.211] |
| M:f1 | 0.019 | [-0.287, 0.329] |
| P:average_precision | -0.108 | [-0.278, 0.107] |
| P:f1 | -0.035 | [-0.274, 0.199] |
| R:average_precision | -0.019 | [-0.117, 0.021] |
| R:f1 | -0.009 | [-0.104, 0.136] |
| I:average_precision | 0.000 | [0.0, 0.0] |
| I:f1 | 0.000 | [0.0, 0.0] |

## Limitations

- B/C contain only 23 families; label support for R and I is ~3 positives. Per-label metrics are extremely noisy and the B-vs-C comparison cannot support a conclusion about image descriptions.
- The image subset is not a random sample (diversity-selected pilot) and is restricted to publications where Google Patents lists figures (US/DE here, no CN).
- Repeated splits reuse the same families, so pooled test rows and repeats are not independent; bootstrap CIs resample families but understate uncertainty from the small training sets.
- Descriptions come from a 3B VLM and have NOT been human-reviewed; 30 of them lack a recorded model revision.
- Default grouping has no related-family links (each family is its own group); citation and assignee relatedness are not controlled unless --group-mode assignee is used. The sample is stratified by assignee.
- Sampling was stratified (not a population sample); metrics are not population estimates; labels F_* -> M/P/R/I mapping is inferred.
- Checkpoints are scikit-learn pickles tied to the recorded library versions.
