# Single-patent M/P/R/I proof of concept: US2021063996 (family 75198509)

**One patent. This demonstrates that the pipeline runs end to end; it is not an evaluation and agreement or disagreement with one patent says nothing about accuracy.**

## 1. Patent identity

- Title (Google Patents): A method and apparatus for optimizing dynamically industrial production processes
- Dataset title: [EN] A METHOD AND APPARATUS FOR OPTIMIZING DYNAMICALLY INDUSTRIAL PRODUCTION PROCESSES
- Family: 75198509; publication US2021063996 (AA); document id US20210063996A1
- URL: https://patents.google.com/patent/US20210063996A1/en
- Retrieved (UTC): 2026-11-25T11:46:31Z; publication date 2021-03-04; application US:16/960,116
- Identity verification: crawler verified document id and publication date against the dataset (`pub_number+pub_date`); the dataset abstract and the Google abstract are the same text apart from the `[EN]` marker.
- Selection reason: First of 6 families passing the fixed rule, ordered by a salted hash of the family number; classifier outputs were not consulted. All four labels are known (no U) with a mix of Y and N, the title+abstract is long enough, and Google Patents lists 3 figures for the US20210063996A1 page.
- Selection rule: (fixed before any prediction): annotated, not in the earlier 20-family pilot,
Google-Patents-eligible (figures listed and reachable), ann_CONFIDENCE=HIGH, all four labels in
{Y,N}, at least one Y and one N, abstract >= 700 chars, 3..10 figures; first by
sha256("poc-single-patent-v1:" + family).

## 2. Figures and VLM descriptions

| Figure | File | Status | Description / error |
|---|---|---|---|
| D00000 | data/figure_enrichment/poc_single_patent/raw/figures/US20210063996A1-D00000.png | vlm_failed | gave up after 1 attempts: truncated_output: hit max_new_tokens=2048 (invalid JSON: Unterminated string starting at) |
| D00001 | data/figure_enrichment/poc_single_patent/raw/figures/US20210063996A1-D00001.png | vlm_failed | gave up after 1 attempts: truncated_output: hit max_new_tokens=2048 (invalid JSON: Unterminated string starting at) |
| D00002 | data/figure_enrichment/poc_single_patent/raw/figures/US20210063996A1-D00002.png | described | The figure presents a simple flowchart with three labeled steps or processes (S1, S2, S3), each connected by arrows indicating a sequential order. |

VLM: `Qwen/Qwen2.5-VL-3B-Instruct` revision `66285546d2b821cf421d4f5eb2576359d3770cd3`, prompt `fig-desc-v1`, config hash `9162e794a12a6892`, generation config `{"4bit": false, "backend": "local-transformers", "dtype": "bfloat16", "greedy": true, "image_prep": "p1-max2000-png", "max_new_tokens": 2048, "max_retries": 0}` (see `crawl/figure_enrichments.csv`).
Failed figures are reported, not hidden: a failed generation (truncated JSON at the 2048-token limit) yields NO description and is not passed to any classifier. The usable description has not been checked by a human against the image.

## 3. Text actually passed to the classifier

**Title + abstract:**

> A method and apparatus for optimizing dynamically industrial production processes . Provided is a a process optimizer apparatus for optimizing dynamically an industrial production process of a production plant including physical production modules, the process optimizer including a watchdog component adapted to monitor the production modules of the production plant to detect configuration changes within the production plant; a model comparator component adapted to evaluate a production plant data model of the production plant including digital twin data models related to physical production modules of the production plant to identify automatically deviating model elements of digital twin data models related to physical production modules of the production plant affected by the configuration changes detected by the watchdog component; and a process resequencer component adapted to perform a dynamic process optimization of the at least one production process of the production plant depending on the deviating model elements identified by the model comparator component.

**Title + abstract + validated figure descriptions:**

> A method and apparatus for optimizing dynamically industrial production processes . Provided is a a process optimizer apparatus for optimizing dynamically an industrial production process of a production plant including physical production modules, the process optimizer including a watchdog component adapted to monitor the production modules of the production plant to detect configuration changes within the production plant; a model comparator component adapted to evaluate a production plant data model of the production plant including digital twin data models related to physical production modules of the production plant to identify automatically deviating model elements of digital twin data models related to physical production modules of the production plant affected by the configuration changes detected by the watchdog component; and a process resequencer component adapted to perform a dynamic process optimization of the at least one production process of the production plant depending on the deviating model elements identified by the model comparator component. . The figure presents a simple flowchart with three labeled steps or processes (S1, S2, S3), each connected by arrows indicating a sequential order.

The Google Patents page exposes no usable description text for this application (`ASSIGNMENT OF ASSIGNORS INTEREST (SEE DOCUMENT FOR DETAILS).`), so none is used. Not provided: human labels, family id, assignee, `adm_*`/`ann_*`, claims, citations.

## 4. M/P/R/I definitions used

- **M**: Monitoring or representation: observation, representation and updates to the state of a physical or operational counterpart.
- **P**: Prediction or simulation: future-state estimation, forecasting and scenario analysis.
- **R**: Recommendation or decision support: selection or recommendation of actions.
- **I**: Intervention or control: action on the physical or managed system.

Labels are independent (non-exclusive); each dimension is predicted separately. The human coding is based on the CLAIMS (`ann_EVIDENCE_SOURCE=CLAIMS`), which the classifiers did not see.

## 5. Predictions (saved before labels were read)

`predictions.json` sha256 `2195930f9fb1c05b9c8ad811317b7f239fdfa066600f463d9e1e185624f6e827` verified unchanged before the human labels were loaded.

Scores are RAW, UNCALIBRATED model outputs, not probabilities.

| System | M | P | R | I |
|---|---|---|---|---|
| A_title_abstract | N (0.474) | N (0.419) | N (0.437) | N (0.398) |
| A_title_abstract_figdesc | N (0.465) | N (0.415) | N (0.429) | N (0.395) |
| C_title_abstract_figdesc | N (0.499) | N (0.289) | N (0.000) | N (0.103) |
| keyword_baseline | Y (0.143) | N (0.000) | N (0.000) | N (0.062) |

Trained model: `experiments/baseline_v1/checkpoints/A_repeat01.joblib` (sha256 `951788ddae2d247b643001837912c9e367170efea256f5a7e64ccbcba5f74aa8`), repeat 1, seed 2027; this family was in that repeat's TEST split (not train/val) and, for checkpoint C, not in the image-subset training families. Config: `experiments/baseline_v1/config.json` (TF-IDF + per-label L2 logistic regression).

### Evidence: A_title_abstract

PRIMARY. Checkpoint A (text-only), fed exactly its training input format.

- **M** N (raw 0.474): top terms models (0.0465), detected (0.0396), production (0.0394), dynamic (0.0377), including (0.0317), process (0.029). Evidence: [abstract] "a model comparator component adapted to evaluate a production plant data model of the production plant including digital twin data models related to physical production modules of the production plant to identify automat" / [abstract] "a model comparator component adapted to evaluate a production plant data model of the production plant including digital twin data models related to physical production modules of the production plant to identify automat" / [title] "A method and apparatus for optimizing dynamically industrial production processes"
- **P** N (raw 0.419): top terms process (0.0614), models (0.0407), optimization (0.0341), dynamically (0.0295), dynamic (0.026), process optimization (0.0238). Evidence: [title] "A method and apparatus for optimizing dynamically industrial production processes" / [abstract] "a model comparator component adapted to evaluate a production plant data model of the production plant including digital twin data models related to physical production modules of the production plant to identify automat" / [abstract] "and a process resequencer component adapted to perform a dynamic process optimization of the at least one production process of the production plant depending on the deviating model elements identified by the model compa"
- **R** N (raw 0.437): top terms production (0.1868), optimization (0.1087), plant (0.0525), process (0.0401), production process (0.0397), process optimization (0.0378). Evidence: [title] "A method and apparatus for optimizing dynamically industrial production processes" / [abstract] "and a process resequencer component adapted to perform a dynamic process optimization of the at least one production process of the production plant depending on the deviating model elements identified by the model compa" / [abstract] "Provided is a a process optimizer apparatus for optimizing dynamically an industrial production process of a production plant including physical production modules, the process optimizer including a watchdog component ad"
- **I** N (raw 0.398): top terms plant (0.0828), production (0.0789), optimization (0.0344), perform (0.025), process optimization (0.0229), depending (0.019). Evidence: [abstract] "Provided is a a process optimizer apparatus for optimizing dynamically an industrial production process of a production plant including physical production modules, the process optimizer including a watchdog component ad" / [title] "A method and apparatus for optimizing dynamically industrial production processes" / [abstract] "and a process resequencer component adapted to perform a dynamic process optimization of the at least one production process of the production plant depending on the deviating model elements identified by the model compa"

### Evidence: A_title_abstract_figdesc

SENSITIVITY. Checkpoint A fed figure descriptions it never saw in training (out-of-distribution).

- **M** N (raw 0.465): top terms models (0.0446), detected (0.0379), production (0.0377), dynamic (0.0361), including (0.0303), process (0.0277). Evidence: [abstract] "a model comparator component adapted to evaluate a production plant data model of the production plant including digital twin data models related to physical production modules of the production plant to identify automat" / [abstract] "a model comparator component adapted to evaluate a production plant data model of the production plant including digital twin data models related to physical production modules of the production plant to identify automat" / [title] "A method and apparatus for optimizing dynamically industrial production processes"
- **P** N (raw 0.415): top terms process (0.0588), models (0.039), optimization (0.0327), dynamically (0.0282), dynamic (0.0249), process optimization (0.0228). Evidence: [title] "A method and apparatus for optimizing dynamically industrial production processes" / [abstract] "a model comparator component adapted to evaluate a production plant data model of the production plant including digital twin data models related to physical production modules of the production plant to identify automat" / [abstract] "and a process resequencer component adapted to perform a dynamic process optimization of the at least one production process of the production plant depending on the deviating model elements identified by the model compa"
- **R** N (raw 0.429): top terms production (0.1789), optimization (0.1041), plant (0.0503), process (0.0384), production process (0.0381), process optimization (0.0362). Evidence: [title] "A method and apparatus for optimizing dynamically industrial production processes" / [abstract] "and a process resequencer component adapted to perform a dynamic process optimization of the at least one production process of the production plant depending on the deviating model elements identified by the model compa" / [abstract] "Provided is a a process optimizer apparatus for optimizing dynamically an industrial production process of a production plant including physical production modules, the process optimizer including a watchdog component ad"
- **I** N (raw 0.395): top terms plant (0.0793), production (0.0756), optimization (0.033), perform (0.024), process optimization (0.0219), depending (0.0182). Evidence: [abstract] "Provided is a a process optimizer apparatus for optimizing dynamically an industrial production process of a production plant including physical production modules, the process optimizer including a watchdog component ad" / [title] "A method and apparatus for optimizing dynamically industrial production processes" / [abstract] "and a process resequencer component adapted to perform a dynamic process optimization of the at least one production process of the production plant depending on the deviating model elements identified by the model compa"

### Evidence: C_title_abstract_figdesc

WEAK. Checkpoint C (text+descriptions) was trained on only ~8-13 families; not credible.

- **M** N (raw 0.499): top terms physical (0.7801), models (0.3589), process (0.277), connected (0.098), indicating (0.0624), steps (0.0438). Evidence: [abstract] "Provided is a a process optimizer apparatus for optimizing dynamically an industrial production process of a production plant including physical production modules, the process optimizer including a watchdog component ad" / [abstract] "a model comparator component adapted to evaluate a production plant data model of the production plant including digital twin data models related to physical production modules of the production plant to identify automat" / [title] "A method and apparatus for optimizing dynamically industrial production processes"
- **P** N (raw 0.289): top terms models (0.5521), processes (0.2082), model (0.2009), process (0.1883), including (0.1729), component (0.1291). Evidence: [abstract] "a model comparator component adapted to evaluate a production plant data model of the production plant including digital twin data models related to physical production modules of the production plant to identify automat" / [title] "A method and apparatus for optimizing dynamically industrial production processes" / [abstract] "a model comparator component adapted to evaluate a production plant data model of the production plant including digital twin data models related to physical production modules of the production plant to identify automat"
- **R** N (raw 0.000): top terms . Evidence: none found
- **I** N (raw 0.103): top terms physical (0.5067), model (0.2156), steps (0.1695), provided (0.141), perform (0.1324), flowchart (0.0603). Evidence: [abstract] "Provided is a a process optimizer apparatus for optimizing dynamically an industrial production process of a production plant including physical production modules, the process optimizer including a watchdog component ad" / [abstract] "a model comparator component adapted to evaluate a production plant data model of the production plant including digital twin data models related to physical production modules of the production plant to identify automat" / [figure D00002] "The figure presents a simple flowchart with three labeled steps or processes (S1, S2, S3), each connected by arrows indicating a sequential order."

### Evidence: keyword baseline (experimental, unvalidated lexicon)

- **M** Y: 3 distinct cue terms of the M lexicon matched (rule: >= 2 => Y): ['detect', 'digital twin', 'monitor']. Evidence: [abstract] "Provided is a a process optimizer apparatus for optimizing dynamically an industrial production process of a production plant including physical production modules, the process optimizer including a watchdog component ad" (monitor, detect) / [abstract] "a model comparator component adapted to evaluate a production plant data model of the production plant including digital twin data models related to physical production modules of the production plant to identify automat" (detect, digital twin)
- **P** N: 0 distinct cue terms of the P lexicon matched (rule: >= 2 => Y): []. Evidence: none
- **R** N: 0 distinct cue terms of the R lexicon matched (rule: >= 2 => Y): []. Evidence: none
- **I** N: 1 distinct cue terms of the I lexicon matched (rule: >= 2 => Y): ['automatically']. Evidence: [abstract] "a model comparator component adapted to evaluate a production plant data model of the production plant including digital twin data models related to physical production modules of the production plant to identify automat" (automatically)

## 6. Human annotation and agreement (revealed after the predictions were saved)

Human: M=Y, P=N, R=N, I=N (coder confidence HIGH).

| System | M | P | R | I | agree |
|---|---|---|---|---|---|
| A_title_abstract | DISAGREE (N vs Y) | agree (N vs N) | agree (N vs N) | agree (N vs N) | 3/4 |
| A_title_abstract_figdesc | DISAGREE (N vs Y) | agree (N vs N) | agree (N vs N) | agree (N vs N) | 3/4 |
| C_title_abstract_figdesc | DISAGREE (N vs Y) | agree (N vs N) | agree (N vs N) | agree (N vs N) | 3/4 |
| keyword_baseline | agree (Y vs Y) | agree (N vs N) | agree (N vs N) | agree (N vs N) | 4/4 |

## 7. Limitations and assumptions

- n = 1; four binary comparisons; no statistical meaning. The prevalence of N in the human labels means a model that predicts mostly N will look good.
- Human labels were derived from claims; the classifiers saw only title, abstract and (at most) one figure description. Claims were deliberately withheld.
- 2 of 3 figures failed VLM generation (truncated JSON); only one description exists, and it is unverified against its image.
- The page lists no description text for this US application; only title, abstract (and claims, unused) were retrieved.
- Checkpoint A never saw figure descriptions in training; A+descriptions is out of distribution. Checkpoint C was trained on a handful of families and is not credible. No checkpoint trained on text + image descriptions at adequate scale exists yet.
- Scores are uncalibrated. The keyword lexicon is an arbitrary operationalisation of the definitions, not validated.
- Reproducibility: the VLM is greedy and the revision is pinned, but GPU kernels may not be bit-identical across hardware.
- The agent that ran this POC saw the selection labels while choosing by rule; the classification code never read them before predictions were saved.
