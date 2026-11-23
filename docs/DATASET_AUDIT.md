# Dataset Forensic Audit: `database/` vs. the M/P/R/I architecture

Read-only analysis. No code, data, spec or config was modified; no model was trained and nothing was fetched.
The only file created is this report. Statistics were computed with pandas from the three CSVs (loaded as strings).

Evidence tags: **[V]** verified directly in the supplied data; **[D]** stated only in the companion documents (`JSTPM_Fisa_Tehnica…md`), not verifiable here; **[H]** hypothesis; **[?]** unresolved.

## 1. Executive summary

- **The three CSVs contain no M/P/R/I labels [V].** I checked every column for single-letter M/P/R/I values and for low-cardinality label-like columns, and found none. The 36 columns are an unmodified PatBase export. Title, abstract and claims are present, and there are no figures, images, figure captions or full descriptions.
- **Labels exist elsewhere and were not supplied [D].** The companion tech sheet says they are human Y/N/U codes for four **non-exclusive** roles, assigned by one coder ("Vlad") to **598 complete patent families**, plus 2 "U" cases (600 sampled). They live in `sample_todo_completed.xlsx` (sheet `Functional_Coding`, hidden `Admin_Manifest`), plus a 120-case validation workbook and an earlier 100+48 development set. None of these workbooks is in the repository [V: searched for `*.xlsx`].
- **Each CSV row is one patent family [V].** There are 28,431 rows, 28,311 unique `Family Number` values, and 120 duplicated rows across files. Publications are packed into one cell per family, as `;`-separated values.
- **Label unit is the family, and labels were coded from claims.** Per the tech sheet [D] the coder saw title, abstract and claims, with machine-translated claims for CN/KR/DE/JP. This differs from the spec's input `x_p=(title, abstract, figure descriptions)`.
- **Two structural mismatches with the current implementation:**
  - **Labels are multi-label and binary.** Roles are non-exclusive, with 0–4 positive roles per family. The prototype's `ConceptScorer` is single-label and mutually exclusive: a softmax over four classes with one `y_p` per patent.
  - **The labelled set is small and sampled.** It is 598 of 28,311 families, a design-weighted sample from a 9,917-family relevance-filtered frame, so it is not a random sample of the CSV.
- **Valid evaluation needs a family-cluster split**, and ideally an assignee-disjoint one. Related families exist that `Family Number` does not merge.

## 2. Inventory and schema

| File | Rows | Size | Purpose / finding |
|---|---:|---:|---|
| `1_10000.csv` | 10,000 | 121 MB | PatBase export part 1 [V] |
| `10001_20000.csv` | 10,000 | 118 MB | part 2 [V] |
| `20001_28431.csv` | 8,431 | 104 MB | part 3 [V] |
| `JSTPM_Fisa_Tehnica_…md` | – | 111 KB | Romanian internal data-lineage sheet. It is the **only** source of label information. |
| `JSTPM_Editorial_DNA_…md`, two `JSTPM_*Diagnostics*.docx` | – | – | Journal-fit audit and manuscript/supplement. Context only. |

The files share an identical 36-column header (UTF-8 with BOM) [V]. Rows total 28,431 and there are 28,311 unique families.

Fill rates (non-empty share, 28,431 rows):

| Field group | Fields (fill) |
|---|---|
| Keys | `Family Number` 100%, `Patent Number` 100%, `Application Number` 100%, `Master Patent Number` 100% |
| Text | `Title` 100%, `Abstract` 99.9% (16 empty), `Claims` / `1st Main Claim` 99.1% (243 empty) |
| Dates | `Earliest Priority Date` 100% (1 empty), `Priority Dates` 100%, `Publication Date` 100% |
| Assignees | `Assignees` 97.1%, `Non Std Assignees` 94.0%, `Probable Patent Assignee` ~100% (12 empty) |
| Inventors | `Inventors` 96.7%, `Non Std Inventors` 92.0% |
| Classification | `International Class` 99.3%, `Cooperative Patent Class` 84.1% |
| Citations | `Backward Citations` 61.0%, `Forward Citations` 62.6% (counts are always present) |
| Other | `Designated States` 8.3%, `Legal Status *` 100% (8 columns) |

**Field meanings checked against the data:**

- **Multi-valued cells.** `Patent Number`, `Kind Code`, `Publication Date`, `Application Number` and `Country Code` are aligned `;`-separated lists, one entry per publication. 12,109 rows have more than one entry. Per-family publication counts range from 1 to 684.
- **`Patent Number` does not hold only the master.** Of 28,311 families, 9,274 repeat the same publication number with different kind codes (A/B).
- **`Master Patent Number` is formatted `"<pubno> <kind>"`**, for example `CN114897241 A`. In 100% of rows its number part is one of the family's publication numbers [V].
- **`Probable Patent Assignee` is a single normalised string** with no `;`, versus `Assignees`, which is a list.
- **Title/abstract/claims language prefixes.** Titles and abstracts carry `[EN]` or `[MT]` (452 titles and 516 abstracts are `[MT]`, which I read as machine-translated, but the export does not define the tag [?]). 70 abstracts are `[DE]` and 6 are `[FR]`. Claims are tagged `[CN]` 22,414, `[EN]` 4,013, `[KR]` 1,061, `[DE]` 289, `[JP]` 124; 83% of claims text is CJK script.
- **Not verified:** the exact semantics of `Legal Status Event Group` codes, and whether `Family Number` is PatBase simple or extended family (see section 4).

## 3. Where the M/P/R/I labels are

**Verified: not in the CSVs.** No column holds M/P/R/I, and no column has ≤15 distinct values other than the file-source tag I added.

**Documented elsewhere [D], not verifiable here:**

| Property | What the tech sheet states |
|---|---|
| Storage | `sample_todo_completed.xlsx` > `Functional_Coding` (22 cols, four Y/N/U columns for M, P, R, I); `Admin_Manifest` (hidden, 28 cols, including `FUNC_*_PROB` model scores); separate 120-case workbook; earlier development set (100 cases + 48 coded) |
| Entity labelled | Patent **family** (`Family Number`), coded from claims |
| Cardinality | Non-exclusive; 0–4 positive roles per family. All-N is a legitimate outcome |
| Counts (598 complete) | M 337, P 232, R 133, I 118, at least two roles 246, no role 141; 2 cases are "U" on all roles |
| Source terminology | M monitoring/representation; P prediction/simulation; R recommendation/decision support; I intervention/control. The definition document supplied is described as a **reconstructed codebook, not the original** |
| Coder | One coder (Vlad) for all 600. Inter-rater agreement (kappa 0.765–0.958, with Alexa) is stated only historically and was not verified from source matrices. 12 families coded twice by the same coder differ in 10 of 12 on at least one label |
| Model-derived scores | Gate 5 scored all 28,311 families with model probabilities. These are **not** human ground truth, and in-sample AUC 0.997 on the training families is inflated |

**Unresolved [?]:** label propagation. The sheet says no automatic hard labels were applied to the other families. I found no evidence of inherited or hierarchical labels, and also nothing that would let me test for them, because the labels themselves are absent.

## 4. Identities, families and related records

**Verified:**

| Quantity | Value |
|---|---:|
| Rows | 28,431 |
| Unique `Family Number` | **28,311** |
| Repeated rows | 120 families appear twice (103 across files 1 and 2, 17 across files 2 and 3) |
| Of those, 112 are fully identical; 8 differ | Differences are only in citations, forward-citation counts, legal-status fields and one `Assignees` list |
| Unique publication numbers | 43,296 (53,525 publication entries before dedup of kind variants) |
| Unique application numbers | 39,470 |
| Publication numbers shared by two or more families | **0** |
| Application numbers shared by two or more families | **0** |
| Families by earliest priority year | 2019–2024: 18,785; 2025: 9,031; before 2019: 494; 1 undated. These match the tech sheet |
| Assignee entries with at least 10 families in 2019–2024 | 229 (6,274 families), matching the sheet |

**Relationship levels:**

1. **Same document twice:** 120 duplicated rows (above). Keep the latest-snapshot row or the one with fuller legal status; document the choice.
2. **Different publications of the same family:** explicit. They are packed into one row and linked by `Family Number`. No separate publication table exists.
3. **Technically related but different families:** not explicit. Evidence of such relations:
   - **Shared priority numbers.** 170 priority numbers are shared by more than one `Family Number`, linking 87 families into 13 clusters (largest 22). In 7 of the 13 clusters the `Probable Patent Assignee` string varies (name variants such as "STRONG FORCE VCN 2019 PORTFOLIO LLC" vs "STRONG FORCE VCN PORTFOLIO 2019 LLC").
   - **Near-identical abstracts across families.** 36 normalised-abstract groups cover 85 families (30 groups have one assignee). An approximate MinHash screen of English abstracts found 103 pairs with 5-gram Jaccard ≥ 0.8 (126 families), 88 of them same-assignee. This is a lower bound because the screen is approximate.
   - **Titles.** 692 families share a title with another family.
   - **In-dataset citation edges.** 33,491 family-to-family backward-citation edges, resolved by publication number. Citations also point outside the dataset (34,210 of 163,173 backward citations resolve inside).
4. **Only a common company or label:** same `Probable Patent Assignee` with different families. There are 13,367 distinct strings (most are singletons). Top: NVIDIA 571, Siemens 250, IBM 201, Beihang University 161.

I found no explicit parent/child, continuation or divisional field. Those can only be partially inferred from shared priority numbers and from within-family publication lists (for example, family 65567142 has 684 publication entries, so `Family Number` seems closer to an extended grouping for that case [H]).

**Safe vs. untrusted identifiers:**

| Identifier | Trust |
|---|---|
| `Family Number` | Safe as the primary key for rows (unique after dedup, but see the 8 discrepant duplicates). Its definition (simple vs extended) is unverified. |
| Publication numbers (`CN114897241`) | Safe for linking to full documents; unique across families |
| `Master Patent Number` | Safe after stripping the kind suffix |
| Application / priority numbers | Safe as keys; priority numbers must be used for cluster detection |
| `Probable Patent Assignee` | Not safe as a company key (name variants, 13,367 strings, 12 empty). Normalisation is a methodological decision |

## 5. Proposed logical data model

```
Assignee string (Probable Patent Assignee)  1 ── * Family (Family Number)
Family (Family Number)                      1 ── * Publication (Patent Number, Kind Code, Pub Date, App Number, Country)
Family                                      * ── * Family       (shared priority numbers, citations; derived, not explicit)
Family                                      1 ── 1 Text bundle  (Title, Abstract, Claims, 1st Main Claim; one language tag each)
Family                                      1 ── * CPC/IPC codes (multi-valued list)
Family                                      0..1 ── 0..1 Human label set (M,P,R,I each Y/N/U)  [not supplied; 598 of 28,311]
Family                                      0..1 ── 1 Model-derived scores (28,311) [not supplied; not ground truth]
Family                                      0 ── * Figure/image                                  [absent]
```

| Link | Evidence | Explicit/derived | Cardinality | Issues |
|---|---|---|---|---|
| Assignee → Family | `Probable Patent Assignee` | Explicit but name-normalisation-dependent | 1:* | 12 empty, name variants, no co-assignee handling |
| Family → Publication | packed lists in `Patent Number` and siblings | Explicit, requires parsing | 1:* | 9,274 families repeat a number with several kinds; 120 dup rows |
| Family ↔ Family | shared priority numbers; shared or near-identical abstracts; citations | Derived/uncertain | *:* | Not part of the export's identity key |
| Family → Label | label workbooks | Not in the supplied data | 1:0..1 | 598 only; a family may have 0–4 positives |

## 6. Semantic evidence coverage

| Evidence | Coverage | Notes |
|---|---|---|
| Title | 100% | `[EN]` 98%, `[MT]` 1.6% |
| Abstract | 99.9% (16 missing) | English for almost all; short (median 1,242 characters) |
| Claims | 99.1% (243 missing) | Original language (83% CJK); median 2,386 characters; machine translation was used by the human coder |
| Full description | 0% | Absent |
| Figure captions / images / VLM descriptions | 0% | No column mentions figures; "FIG" appears in 1.1% of abstracts |
| Inventors / assignees | 96.7% / 97.1% | |
| Priority / publication dates | ~100% | |
| Citations | 61–63% non-empty | |
| CPC / IPC | 84% / 99% | |

**Assessment:**

- **Cue and relation extraction.** Title plus abstract (and translated claims) could support cue extraction with source-span localisation by character offset. Claims are not in the spec's `x_p`, though they are what the human labels were based on [D].
- **Figure side.** Because no figure data exists, the spec's `v_p` is empty for every record. A VLM enrichment step needs figure images, which the export does not contain. Whether it is needed cannot be decided until the label evidence source is clarified: the tech sheet says the coder used claims, not figures [D].
- **Insufficient records:** 16 with no abstract; 243 with no claims (about 1% each). Missing text must stay "unknown", not "no cues".
- **Language caveat.** The `[MT]` abstracts and the machine-translated claims carry translation error into cues, so each cue's provenance should record the language tag.

## 7. Mapping to the mathematical spec and the repository

| Mathematical component | Required input | Available field(s) | Coverage / limitation | Status |
|---|---|---|---|---|
| Patent representation `x_p=(t,a,v)` (`schema.Patent`) | title, abstract, figure text | `Title`, `Abstract`; claims as an extra input; no figure text | `v_p` is empty for all rows; text tags `[EN]/[MT]` must be stripped; `patent_id` must be chosen | Supported after deterministic preprocessing (without `v_p`) |
| Normalisation | clean text, dedup | `[EN]` prefix, `Family Number`, 120 duplicate rows | Dedup rule needed for 8 discrepant duplicates | Supported after deterministic preprocessing |
| Semantic cues `q=(s,r,l,e,u)` | teacher/student output | none; text available | No teacher, no gold spans or roles | Requires annotation or external enrichment |
| Relations and provenance | relation annotations; spans | none | Same as above; spans can be located only after extraction | Requires annotation or external enrichment |
| Patent-to-concept membership `μ(p,c)` | cues | derives from cues | Definition of `μ` is open in the spec | Requires a methodological decision |
| Concept lattice (FCA) | `(P, A, R)` | `P` = family ids; `A` and `R` require cues | Cannot be built until cues exist (a deterministic term baseline is possible but is a methodological decision) | Requires annotation or external enrichment |
| Extents/intents | the lattice | – | Implemented and unit-tested on synthetic data; not exercised on real data | Interface/code ready; no real input |
| Concept-to-category relevance `ρ(c,k)` | labels `y_p` | label workbooks (not supplied) | **Mismatch:** spec and code assume one `y_p ∈ {M,P,R,I}`; real labels are 4 independent Y/N/U flags. Options: four binary relevances (one per role), or treat label sets as classes. Not decided | Requires a methodological decision; labels not supplied |
| M/P/R/I scores and abstention | scorer, labels | – | Current code: one softmax over four classes. Multi-label needs independent per-role scores; U needs a representation | Requires a methodological decision |
| Teacher annotations, student training, `L_KD` | LLM outputs, calibrated teacher distribution, gold spans/relations | none | No teacher output; no gold spans/relations; teacher calibration unknown | Cannot be assessed with the available data |
| Company capability graph | company, cues, relations | `Probable Patent Assignee`; cues absent | Entity resolution needed; no cues | Supported after preprocessing + enrichment |
| Organizational inference | independent org evidence `E_c` | none in the CSVs | Citation and assignee data are not organizational structure | Cannot be assessed with the available data |
| Training/validation/calibration/evaluation | labelled set, splits | 598 labels (not supplied), `Family Number`, priority numbers | Small sample, design weights, 12 repeat-coded families | Requires a methodological decision |

Spec formulas are not to be changed to fit the data. The two real gaps are (a) multi-label/binary targets and (b) the family-level, claims-based label provenance.

## 8. Leakage and evaluation risks

1. **Family-level duplication.** The same family appears twice for 120 rows, in different files. A file-based split would leak it. Dedup first.
2. **Related-but-distinct families.** 87 families form 13 priority-linked clusters; 126 families are in near-duplicate abstract pairs; there are 33,491 in-corpus citation edges. Splitting by `Family Number` alone allows these to land on both sides.
3. **Company shortcut.** Assignee is correlated with roles in the author's own analysis [D], and the same company contributes many families (NVIDIA 571). Assignee-disjoint folds are already used in the author's benchmark [D].
4. **Training evidence reuse.** The 598 families were also training data for the model-derived scores; Gate 5 reports in-sample AUC near 0.997 [D]. Scores for those families must not be reused as features or targets. The 12 families shared by the 600 and 120 workbooks, and 19 previously coded families, must be tracked [D].
5. **Lattice and ρ built from held-out examples.** The prototype fits ρ on whatever it is given; the split must be applied before `fit`.
6. **Label source differs from model input.** Labels are from claims, but the spec's input is title, abstract and figures. A text-only model may find the label evidence missing, and the 83%-CJK claims have a translation dependency.
7. **Sampling design.** The 598 are a stratified, probability-weighted sample from 9,917 relevance-filtered families; unweighted metrics do not describe the 28,311 corpus. The relevance filter itself used a model with a reported threshold that 7 of 600 sampled families fall below [D].
8. **Single coder.** No independent second labels are supplied, so label noise is unmeasured.

**Recommended split:** group by connected components over (a) `Family Number`, (b) shared priority numbers, (c) normalised-abstract near-duplicates, then keep whole assignee strings (after normalisation) on one side where feasible. Family-level grouping is mandatory; the rest are strongly recommended. Publication-number splitting is insufficient because publications are nested inside families.

## 9. Blockers, ranked

1. **Label files not supplied** (`sample_todo_completed.xlsx` and its siblings). No supervised step is possible.
2. **Label model mismatch.** Non-exclusive binary Y/N/U flags vs the single-label scorer. Needs a decision.
3. **No semantic cues.** No teacher or gold annotations, so the lattice has no attributes.
4. **Label provenance vs input mismatch.** Claims-based labels vs title/abstract-based model input; translation dependence.
5. **Company identity.** Assignee normalisation is undefined and no organizational ground truth exists.
6. **No figures.** The `v_p` channel is empty. Priority depends on item 4.
7. **Small labelled sample with design weights.**

## 10. Questions for Andreea

1. Can the label workbooks (`sample_todo_completed.xlsx`, the 120-case workbook, the 100+48 development set) and the 9,917-family list be supplied, with the exception register and the codebook (original, not reconstructed)?
2. Should M/P/R/I be modelled as four independent binary roles (as coded), or is a single-label formulation intended? How should "U" and the all-N case be treated?
3. Is the target input the title and abstract, or the claims as the human coder used them? Are figures, which this export lacks, expected to matter?
4. Secondary: what does PatBase `Family Number` denote (simple vs extended), how should the 120 duplicate rows and the 8 discrepant ones be resolved, and what organizational evidence (if any) is available for the company stage?

## 11. Next-step plan

- **Deterministic preprocessing** (no new decisions): dedup families, split publication lists, strip language tags, record the 16/243 missing-text flags, derive priority-number/near-duplicate/citation clusters and split groups.
- **Enrichment** (needs approval): obtain label workbooks; decide on the translation of claims; teacher annotation of cues (not started, and no teacher exists); figure retrieval only if shown necessary.
- **Methodological decisions:** multi-label scoring and `μ(p,c)`; handling of U; company normalisation; split policy and the weighting of the sample.
- **Implementation (later):** a loader mapping CSV rows to `Patent` (family id as `family_id`, one chosen `patent_id`), a label adapter for multi-label targets, and group-aware splitting. None of this is started.

## 12. Verified, hypothesis, unresolved

- **Verified:** counts, field fill, duplicates, absence of labels, absence of figures, family/publication/application uniqueness, priority clusters, citation resolution.
- **Reasonable hypotheses:** `[MT]` means machine-translated; `Family Number` for very large families (up to 684 publications) is an extended grouping; label noise exists because of the 12-family recoding.
- **Unresolved:** label content and propagation, codebook originals, the exact selection procedure, the meaning of legal-status codes, assignee entity resolution.

## 13. Direct answers

- **Where are the labels stored?** Not in the supplied CSVs. According to the tech sheet they are in `sample_todo_completed.xlsx` (`Functional_Coding`), which is not supplied.
- **What entity does a label describe?** A patent family (`Family Number`), per the sheet [D]. This is not verified here.
- **How many unique entities can we identify?** 28,311 families, 43,296 unique publication numbers and 39,470 unique application numbers [V], plus 28,431 raw rows with 120 repeated.
- **Are family or related-patent relationships explicit?** Family membership is explicit (one row per family, publications packed). Relations between different families are not explicit; they can be inferred from shared priority numbers (13 clusters, 87 families), near-duplicate text and citation edges.
- **Which fields can feed the model directly?** `Title` and `Abstract` (with preprocessing), and optionally `Claims`; `Family Number` for grouping. No labels, cues or figure text are present.
- **What is missing for the lattice and teacher?** Cue/attribute annotations (teacher or gold), relation annotations, the label table, a multi-label formulation, and a decision on `μ`.
- **Is figure retrieval or VLM enrichment needed?** Not yet determinable. The export has no figure data and the human coder reportedly used claims [D].
- **Most serious risk to valid evaluation?** Related-family leakage: the same family appears twice in the export, 13 priority-linked clusters, near-duplicate abstracts and citation links join distinct `Family Number` values, and heavily represented assignees compound it.
- **Three most important questions:** the label files (Q1), multi-label vs single-label (Q2), and title/abstract vs claims as the input (Q3).
- **Smallest defensible next step:** obtain the label workbooks, then run a deterministic, no-model preprocessing pass (dedup, group-cluster IDs, text flags) and re-check the label counts against the sheet's 598.
