# Implementation of `patent_organizational_inference_math.tex`

The repository held only the spec, with no existing code or data. This is a new stdlib-only Python package (`patent_mpri/`). There was no existing M/P/R/I pipeline to preserve or integrate with.

## Run

```
python3 -m pytest -q tests            # 31 tests (unit + synthetic integration)
python3 examples/run_synthetic_example.py
```

The only dependency is Python ≥3.9. `pytest` is needed for the tests.

## Equation → code map

| Spec | Code | Status |
|---|---|---|
| §1 `x_p, y_p, c(p)`, family id | `schema.Patent` (`classifier_input()` excludes company/label) | done |
| §3 cue `q=(s,r,l,e,u)`, `Q_p`, relations | `schema.Cue/Relation/Annotation/SourceRef` (+`validate_against`) | done |
| §3 teacher `T_ω`, student `S_η` | `distillation.Teacher/Student` protocols, `PrecomputedAnnotator` | interface only (needs LLM / trained student) |
| §2 `f_θ, g_φ, F_ψ` | not implemented | needs model + data |
| §4 `π̂=softmax(W z+b)` | `distillation.affine_softmax` | done |
| §4 `L_KD^cls` | `distillation.kd_loss` (raises if teacher not calibrated) | done |
| §4 `L_CE`, `L_student` | `cross_entropy`, `student_loss` (no value → error, never invented) | done; `L_span`, `L_rel` undefined |
| §5.1 `K=(P,A,R)`, `B`, `X'`, `Y'`, concepts, order | `lattice.FormalContext/ConceptLattice` (NextClosure, Hasse covers) | done, **classical binary FCA** |
| §5.1 weighted variant | `scoring.graded_mean_mu` — only the matching weight, labelled EXTENSION | partial; no weighted lattice |
| §5.2 `ρ(c,k)` | `scoring.compute_rho` | done |
| §6 `s_k`, `π_k`, `ŷ` | `ConceptScorer.scores/probabilities/predict` | done |
| §6 multi-label | not implemented (spec says "should be used" if labels overlap; labels are mutually exclusive in the given scheme) | open |
| §7 `Δ_{c,k}` | `ConceptScorer.deltas`, `Prediction.supporting` (with cue provenance) | done |
| §7 `d_i` | `schema.EvidenceRecord`, `evidence_from_annotation` | done |
| §8 `G_c=(V,E,X)` | `graph.build_capability_graphs` | done |
| §8 `P(G_c|D_c,E_c)` | `orginference` baseline + `OrgPosteriorModel` protocol | **baseline only; not a probability** |
| §9 `L_lattice`, `L_org` | raise `NotImplementedError` | undefined in spec |
| §10 metrics | `evaluation` (macro-F1, balanced acc., ECE, span/relation/graph P-R-F1, partition pair F1, group split) | interfaces, no results reported |

## Ambiguities and decisions

1. **`μ(p,c)` undefined.** The default is classical containment `1[Y ⊆ A_p]`. The alternative `graded_mean_mu` is mean cue confidence over the intent, an extension that is *not* classical FCA.
2. **Attribute universe `A`.** It is the lowercased, whitespace-normalised cue term. Roles are ignored, and an optional `min_confidence` threshold sets which cues enter `R` (default 0, so every cue counts).
3. **Top concept (empty intent).** It matches every patent. By default it is kept, so `C` is the whole lattice as the spec says. It acts as a class-prior term, so abstention never happens in the default setup (`predict` returns `None` only if the total match mass is 0).
4. **`ε`.** `1e-9`. Because `ε` sits in the denominator, `Σ_k s_k` is slightly below the true mass.
5. **`w(c)`.** Default is 1. The spec gives no objective for learning it, so `ConceptScorer` accepts external weights.
6. **`T`.** Default is 1. `s_k ∈ [0,1]`, so softmax at T=1 is nearly flat. `T` must be tuned on validation data before `π` is called calibrated; it is not.
7. **`ρ` leakage.** `ρ` is estimated on the training patents, including the patent being scored if you evaluate in-sample. Evaluate on held-out patent families (`group_split`).
8. **Evidence `score`.** The spec doesn't define it; the code uses cue confidence `1-u`.
9. **Organizational baseline.**
   - Candidates: single unit, connected components, and grouping by dominant predicted M/P/R/I category (the last assumes the organization mirrors M/P/R/I, which is unverified).
   - Score: Newman modularity on co-occurrence/relation weights, plus an optional agreement term with independent external assertions.
   - Thresholds `min_patents=3` and `min_margin=0.05` are arbitrary user-adjustable defaults with no validation behind them.
   - Outputs are explicitly "heuristic_uncalibrated". The module can abstain, and it keeps ranked alternatives.
   - Co-occurrence inside a patent does not imply a shared team.

## Missing data / remaining work

- **Teacher/student training:** LLM cue annotations (audited) and the existing labelled patents. `L_span` and `L_rel` need gold spans and relations.
- **Calibrated teacher distribution:** required for `L_KD`.
- **`L_lattice`:** needs a concrete definition, e.g. held-out likelihood of concept→category associations. A differentiable lattice relaxation, or the alternating procedure the spec recommends, is also missing.
- **Learned `w(c)` and `T`:** need a validation set.
- **Organizational model:** needs reference organizational structures (units, membership, hierarchy) for many companies, split by company. Until then no Bayesian or latent-variable model can honestly be trained.
- **Multi-label M/P/R/I:** only if the labelling scheme permits overlap.
- **Real-data evaluation:** none has been run, and no performance numbers are claimed. The tests and example use hand-made synthetic data in `examples/synthetic_data.py`, flagged `SYNTHETIC = True`.
- **Scale:** the NextClosure and cover computations are fine for small corpora. A large corpus needs an optimized FCA algorithm and attribute pruning.
