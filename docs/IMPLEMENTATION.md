# Implementation of `patent_organizational_inference_math.tex`

The repository held only the spec, with no existing code or data. `patent_mpri/` is a new stdlib-only Python package.
Nothing has been run on real patents. **No accuracy, calibration, or trained teacher–student model is claimed.**
Tests and the example use hand-made synthetic records (`examples/synthetic_data.py`, `SYNTHETIC = True`).

## Run

```
python3 -m pytest -q tests            # 49 tests: unit, audit regression, synthetic integration
python3 examples/run_synthetic_example.py
```

## Naming of scores (important)

| Name in code | Meaning | Calibrated? |
|---|---|---|
| `raw_scores` (`s_k`) | evidence-based ratio in [0,1] | no |
| `softmax_scores` (`softmax(s_k/T)`) | normalisation of raw scores, `None` when there is no evidence | **no** (spec's "calibrated π_k", but T is unvalidated) |
| `calibrated_probabilities` | raises `NotImplementedError` until T is fitted on held-out data | n/a |
| `ConceptScorer.class_priors` | training label frequencies, kept separate, never added to `s_k` | n/a |
| org `compatibility_score` | heuristic structure score, **not a probability** | no |

## Audit corrections (this revision)

- **Empty-intent concept.** The top concept has an empty intent when no attribute is shared by all patents. It matches every patent regardless of cues, so it injected the training label frequencies as if they were evidence. It is now excluded from `s_k`, `Δ`, and the denominator by default (`include_empty_intent=False`). `True` reproduces the literal sum over all of `C` and is tested to show the artifact.
- **No evidence → abstain.** The raw scores are all 0, `softmax_scores` is `None`, and `predicted` is `None`. A softmax of zeros would otherwise fake a uniform distribution, and an arg-max over zeros would pick "M" by tie-break. The same applies if every matched concept has ρ = 0.
- **ρ and μ checked against the spec.** `compute_rho` is tested against an independent evaluation of the formula. Binary `μ(p,c)=1[Y⊆A_p]` equals extent membership for every training patent and concept (`test_binary_mu_equals_extent_membership`).
- **Org edges.** Co-occurrence and direct relation counts are stored separately (`edge_components`). The weight is `a_cooc·cooc + a_rel·rel`, with unvalidated defaults of 1 and 1. Setting `cooc_weight=0` gives relations only.
- **Modularity.** The partition is now validated: disjoint units, covering exactly the graph's nodes, no edge endpoint outside the partition. Isolated nodes have degree 0 and contribute 0.
- **M/P/R/I-category grouping is opt-in** (`include_category_hypothesis=True`). Nothing in the spec says organizational units mirror the four function types.
- **New candidate `external_constrained`.** It is built from independent same-unit assertions, and the different-unit assertions are respected by the scoring.
- **Patent-only abstention.** Modularity favours splitting into dense clusters, so patents alone cannot separate "distinct technical clusters of one organization" from "separate units". By default the module abstains unless independent external assertions apply (`allow_patent_only=True` returns technical-cluster candidates only). Abstentions list `missing_evidence`.

## Status table

Statuses: **UT** = implemented and unit-tested; **NV** = implemented but not validated on real data (applies to everything in this package, so it's omitted from the list where UT already holds); **IF** = interface only; **NI** = not implemented; **DATA** = requires additional labeled data.
Every UT row below is also NV, since no real data has been used.

| Spec item | Status | Source | Test |
|---|---|---|---|
| §1 patent input `x_p`, `y_p`, `c(p)`, family | UT | `schema.py::Patent`, `classifier_input` | `test_patent_malformed`, `test_company_not_classifier_input`, `test_company_not_leaked_into_classifier` |
| §3 cue `q=(s,r,l,e,u)`, `Q_p` | UT | `schema.py::Cue/Relation/Annotation/SourceRef` | `test_cue_validation`, `test_annotation_validation` |
| §3 teacher `T_ω` (LLM) | IF + DATA | `distillation.py::Teacher`, `PrecomputedAnnotator` | `test_integration.py::test_pipeline_end_to_end` (precomputed synthetic cues only) |
| §2 encoders `f_θ, g_φ, F_ψ` | NI | none | none |
| §4 student `S_η` | IF + DATA | `distillation.py::Student` | none |
| §4 `π̂=softmax(Wz+b)` | UT | `distillation.py::affine_softmax` | `test_affine_softmax` |
| §4 `L_CE`, `L_KD^cls` | UT | `cross_entropy`, `kl_divergence`, `kd_loss` | `test_kd_and_losses` |
| §4 `L_student` | UT for CE+KD; `L_span`, `L_rel` DATA/NI | `student_loss` | `test_kd_and_losses` |
| §5.1 `K=(P,A,R)`, `B`, `X'`, `Y'` | UT | `lattice.py::FormalContext` | `test_derivation_operators`, `test_bad_incidence` |
| §5.1 concepts, order, lattice | UT | `FormalContext.concepts` (NextClosure), `ConceptLattice` | `test_lattice_concepts_match_manual`, `test_nextclosure_equals_bruteforce`, `test_lattice_order_and_covers` |
| §5.1 weighted variant | partial: only the matching `graded_mean_mu` (extension, not FCA); weighted lattice NI | `scoring.py::graded_mean_mu` | `test_graded_mu_reduces_to_binary_and_is_partial` |
| §5.2 `ρ(c,k)` | UT | `scoring.py::compute_rho` | `test_rho_and_scores_hand_computed`, `test_rho_matches_spec_formula_independently` |
| §5.2 `μ(p,c)` | UT (spec leaves open; binary containment default) | `binary_containment_mu` | `test_binary_mu_equals_extent_membership` |
| §6 `s_k(p*)` | UT | `ConceptScorer.raw_scores` | `test_adding_and_removing_evidence`, `test_empty_intent_concept_present_but_excluded` |
| §6 `π_k` softmax(s/T) | UT as a softmax; calibration is NOT established | `ConceptScorer.softmax_scores` | `test_softmax_vs_calibrated`, `test_deterministic_fit_and_ties` |
| §6 calibration of `T`, concept weights `w(c)` | DATA (`w` defaults to 1; `T` to 1) | `ConceptScorer.calibrated_probabilities` (raises) | `test_softmax_vs_calibrated` |
| §6 `ŷ=argmax` | UT | `ConceptScorer.predict` | `test_matched_concepts_with_zero_rho_abstain`, `test_deterministic_fit_and_ties` |
| §6 multi-label | NI | none | none |
| §7 `Δ_{c,k}` | UT | `ConceptScorer.deltas` | `test_rho_and_scores_hand_computed` |
| §7 evidence `d_i` | UT | `schema.py::EvidenceRecord`, `evidence_from_annotation`; `Prediction.supporting` | `test_prediction_provenance`, `test_graph_aggregation_and_provenance` |
| §8 `G_c=(V,E,X)` | UT | `graph.py::build_capability_graphs` | `test_graph_aggregation_and_provenance`, `test_cooccurrence_and_relation_kept_separate` |
| §8 `P(G_c|D_c,E_c)` | **NI as a probability**; baseline heuristic only; DATA (reference org structures) | `orginference.py::infer_organization`, `OrgPosteriorModel` (IF) | `test_org_modularity_hand_value_and_external_evidence`, `test_adversarial_*`, `test_org_abstains_*` |
| §9 `L_lattice`, `L_org`, joint `L` | NI (spec says undefined); DATA | `distillation.py::lattice_loss/org_loss` raise | `test_kd_and_losses` (lattice_loss raises) |
| §10 evaluation metrics | UT (functions only; no results) | `evaluation.py` | `test_metrics` |

## Unresolved mathematical ambiguities

1. **μ(p,c).** The spec doesn't define it. The default is binary containment. The graded mean is an extension.
2. **Attribute universe A.** It is the normalised cue term. Roles and relations are not part of the attributes.
3. **Concepts whose extent is all training patents.** Even with a non-empty intent they carry only the label prior, since ρ is the class frequency. The spec's sum includes them and they are kept. Whether they should also be excluded is open.
4. **Empty-extent concepts** (the bottom concept) have ρ=0 but add μ·w to the denominator when matched. This follows the spec literally and dilutes the scores.
5. **ε** sits in the denominators, so `Σ_k s_k` is slightly below the matched mass (value 1e-9, not specified).
6. **w(c) and T** have no learning objective in the spec. `s_k ∈ [0,1]` also makes a softmax at T=1 nearly flat.
7. **Evidence score** in `d_i` is undefined, so cue confidence `1-u` is used.
8. **Edge-weight coefficients** `a_cooc`, `a_rel`, λ_ext, and the abstention thresholds are arbitrary defaults.
9. **ρ leakage.** ρ is estimated on the training patents. Evaluate on held-out families with `evaluation.group_split`.
10. **`P(G_c|D_c,E_c)`.** The spec gives no likelihood, prior, or candidate-space definition.

## Missing data / remaining work

- LLM teacher annotations (audited), gold spans and relations, and a calibrated teacher distribution (for `L_span`, `L_rel`, `L_KD`).
- A held-out validation set to fit `T` and `w(c)` and to compare against a no-lattice baseline.
- A concrete definition of `L_lattice`, or an alternating procedure.
- Reference organizational structures for many companies, split by company, before any trainable organizational model.
- Multi-label support, if the labelling scheme allows overlap.
- Scalability of the lattice code to a large corpus.
