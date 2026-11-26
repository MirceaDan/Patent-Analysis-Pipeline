"""Phase-2 audit of the integrated training dataset.

Writes master_dataset/training_dataset_audit.json (machine-readable) and
docs/TRAINING_DATASET_AUDIT.md (human-readable). Reads only; never alters labels.
Run: python scripts/audit_training_dataset.py
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpri_baseline import data as D  # noqa: E402
from mpri_baseline.groups import related_family_groups  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402


def read_str(p, **kw):
    return pd.read_csv(p, dtype=str, keep_default_na=False, na_filter=False, encoding="utf-8-sig", **kw)


def norm(s):
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in s).split())


def run(master: Path, enrich: Path):
    checks, out = [], {}

    def check(name, ok, detail=""):
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    # ---- structure and joins
    enr = read_str(master / "patent_families_enriched.csv")
    ann = read_str(master / "patent_families_annotated.csv")
    mas = read_str(master / "patent_families_master.csv", usecols=["Family Number"])
    out["rows"] = {"enriched": len(enr), "annotated_file": len(ann), "master": len(mas),
                   "unique_family_numbers": int(enr["Family Number"].nunique())}
    check("one row per family (enriched)", len(enr) == enr["Family Number"].nunique() == len(mas))
    check("family order/ids identical to annotated and master files",
          list(enr["Family Number"]) == list(ann["Family Number"]) == list(mas["Family Number"]))
    same = all((enr[c] == ann[c]).all() for c in ann.columns)
    check("all original annotated columns unchanged (incl. labels and label_M/P/R/I)", same)
    check("label aliases equal source label columns", D.alias_consistent(enr))
    a = enr[enr["is_annotated"] == "True"].reset_index(drop=True)
    out["annotated_families"] = len(a)
    check("600 annotated families", len(a) == 600)
    check("unannotated families have empty labels",
          (enr.loc[enr["is_annotated"] != "True", D.LABEL_COLUMNS] == "").all().all())
    pubs = read_str(master / "patent_publications.csv", usecols=["family_number", "publication_number"])
    check("every family has >=1 publication", set(pubs["family_number"]) >= set(enr["Family Number"]))
    figs = read_str(enrich / "figure_enrichments.csv")
    res = read_str(enrich / "document_resolution.csv")
    check("enrichment families all exist in dataset",
          (set(figs["family_number"]) | set(res["family_number"])) <= set(enr["Family Number"]))
    check("enrichment families all annotated",
          (set(figs["family_number"]) | set(res["family_number"])) <= set(a["Family Number"]))

    # ---- labels
    lab = {}
    for c, n in zip(D.LABEL_COLUMNS, D.LABEL_NAMES):
        vc = a[c].value_counts().to_dict()
        lab[n] = {"column": c, **{k: int(vc.get(k, 0)) for k in ["Y", "N", "U", ""]}}
    out["label_counts"] = lab
    check("label values only Y/N/U (no empty among annotated)",
          all(lab[n][""] == 0 for n in lab) and
          all(sum(lab[n][k] for k in "YNU") == 600 for n in lab))
    y, known = D.label_matrix(a)
    ny = (y * known).sum(1).astype(int)
    out["n_Y_per_family"] = {int(k): int(v) for k, v in collections.Counter(ny).items()}
    out["label_combinations_Y_N_U"] = a[D.LABEL_COLUMNS].apply(lambda r: "".join(r), axis=1).value_counts().to_dict()
    ph = {}
    for i, ni in enumerate(D.LABEL_NAMES):
        for j, nj in enumerate(D.LABEL_NAMES):
            if i < j:
                m = known[:, i] & known[:, j]
                ph[f"{ni}-{nj}"] = round(float(np.corrcoef(y[m, i], y[m, j])[0, 1]), 3)
    out["label_phi_correlation"] = ph
    out["rows_all_N"] = int(((y == 0) & known).all(1).sum())
    out["rows_all_U"] = int((~known).all(1).sum())
    out["rows_partially_U"] = int((known.any(1) & ~known.all(1)).sum())
    out["label_semantics"] = ("independent binary multi-label: {} rows have >=2 Y, {} rows have 0 Y "
                              "(all N); mutual exclusivity and argmax are NOT justified").format(
        int((ny >= 2).sum()), out["rows_all_N"])
    check("labels are not mutually exclusive (so no softmax/argmax formulation)", (ny >= 2).sum() > 0)

    # ---- missing fields
    miss = {c: int((a[c].str.strip() == "").sum()) for c in
            ["Title", "Abstract", "Patent Number", "Family Number", "Priority Numbers"]}
    out["missing_in_annotated"] = miss
    out["abstract_chars"] = {k: float(v) for k, v in a["Abstract"].str.len().describe().round(1).items()}
    check("no missing titles/abstracts/family numbers among annotated",
          miss["Title"] == miss["Abstract"] == miss["Family Number"] == 0, str(miss))

    # ---- enrichment / provenance
    st = a["figure_enrichment_status"].value_counts().to_dict()
    out["enrichment_status_annotated"] = st
    nd = a["n_image_descriptions"].astype(int)
    out["families_with_usable_description"] = int((nd > 0).sum())
    out["families_without_usable_description"] = int((nd == 0).sum())
    out["usable_descriptions_total"] = int(nd.sum())
    prov_ok, miss_model, no_rev, desc_texts = True, 0, 0, []
    for t, p, n in zip(a["image_descriptions"], a["image_description_provenance"], nd):
        pl = json.loads(p)
        lines = t.split("\n") if t else []
        prov_ok &= len(pl) == len(lines) == n
        miss_model += sum(1 for x in pl if not x["vlm_model"] or not x["publication_number"]
                          or not x["figure_id"] or not x["prompt_version"])
        no_rev += sum(1 for x in pl if not x["vlm_model_version"])
        desc_texts += lines
    check("provenance list aligned with description lines and counts", prov_ok)
    check("every description has model id, prompt version, publication and figure ids", miss_model == 0,
          f"{miss_model} incomplete")
    out["warnings"] = []
    if no_rev:
        out["warnings"].append(
            f"{no_rev} descriptions (first 8-family pilot) have NO recorded model revision because the "
            "revision was not pinned then; they carry vlm_config_hash fe6c9a962df2e6f2 (max_new_tokens=1200). "
            "Same weights are likely (one cached snapshot) but this is not recorded. The other "
            "descriptions use revision 66285546d2b821cf421d4f5eb2576359d3770cd3 with max_new_tokens=2048.")
    dups = {k: v for k, v in collections.Counter(desc_texts).items() if v > 1}
    out["duplicate_descriptions"] = {"distinct_texts_repeated": len(dups),
                                     "total_repeats": int(sum(v - 1 for v in dups.values()))}
    out["figure_rows_by_status"] = figs["processing_status"].value_counts().to_dict()
    out["failed_figure_records"] = {
        "vlm_failed": int((figs["processing_status"] == "vlm_failed").sum()),
        "image_inaccessible": int((figs["processing_status"] == "image_inaccessible").sum()),
        "families_with_failures_and_no_usable_description": int(sum(
            1 for f, g in figs.groupby("family_number")
            if (g["processing_status"] == "vlm_failed").any()
            and not (a.loc[a["Family Number"] == f, "n_image_descriptions"].astype(int) > 0).any()))}
    out["failure_reasons"] = collections.Counter(
        e.split(":")[1].strip().split(" ")[0] if "gave up" in e and ":" in e else e[:50]
        for e in figs.loc[figs["processing_status"] == "vlm_failed", "error_details"]).most_common()
    out["vlm_models_in_dataset"] = sorted({(x["vlm_model"], x["vlm_model_version"])
                                           for p in a["image_description_provenance"] for x in json.loads(p)})

    # ---- eligibility
    el = D.eligibility(a)
    A, BC = el["eligible_A"].to_numpy(), el["eligible_BC"].to_numpy()
    out["eligibility"] = {
        "A_all_eligible_annotated": int(A.sum()),
        "A_excluded": {"no_known_label": int((~el["any_known_label"]).sum()),
                       "missing_title": int((~el["has_title"]).sum()),
                       "missing_abstract": int((~el["has_abstract"]).sum())},
        "BC_image_description_subset": int(BC.sum()),
        "BC_excluded_because": {"no_usable_image_description_among_A": int((A & ~el["has_image_description"]).sum())}}
    sup = {}
    for nm, mask in (("A", A), ("BC", BC)):
        sup[nm] = {n: {"Y": int((y[mask, i] * known[mask, i]).sum()),
                       "N": int(((1 - y[mask, i]) * known[mask, i]).sum()),
                       "U": int((~known[mask, i]).sum())} for i, n in enumerate(D.LABEL_NAMES)}
    out["label_support_by_subset"] = sup
    sub = a[BC]
    src = collections.Counter(x["publication_number"][:2] for p, e in
                              zip(sub["image_description_provenance"], BC[BC]) for x in json.loads(p))
    out["BC_selection_note"] = ("The image-enriched subset is NOT a random sample: the 20-family pilot was chosen by "
                                "a seeded greedy rule maximising coverage of label values and IPC classes (plus 3 "
                                "families from the earlier 8-family pilot). Only publications on which Google Patents "
                                "lists figures can be described (no CN). Country prefix of the publications the "
                                "descriptions came from: " + json.dumps(dict(src)))
    check("B/C subset is a subset of A", bool((BC <= A).all()))
    check("each experiment-eligible row has at least one known label", bool(known[A].any(1).all()))

    # ---- leakage
    ta = (a["Title"] + " . " + a["Abstract"]).map(norm)
    out["leakage"] = {
        "exact_duplicate_title_abstract_pairs": int(ta.duplicated().sum()),
        "duplicate_titles": int(a["Title"].map(norm).duplicated().sum()),
        "duplicate_abstracts": int(a["Abstract"].map(norm).duplicated().sum())}
    X = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(
        (a["Title"] + " . " + a["Abstract"]).tolist())
    S = (X @ X.T).toarray()
    np.fill_diagonal(S, 0)
    out["leakage"]["max_pairwise_tfidf_cosine"] = round(float(S.max()), 3)
    out["leakage"]["pairs_cosine_ge_0.5"] = int((np.triu(S) >= 0.5).sum())
    g_def, links = related_family_groups(a, pubs, return_links=True)
    g_as = related_family_groups(a, pubs, use_assignee=True)
    g_cit = related_family_groups(a, pubs, use_citations=True)
    sz = lambda g: sorted(collections.Counter(g.values()).values(), reverse=True)[:5]
    out["leakage"]["link_counts"] = {k: len(v) for k, v in links.items()}
    out["leakage"]["groups"] = {
        "default_priority_application_textdup": {"n_groups": len(set(g_def.values())), "largest": sz(g_def)},
        "plus_assignee": {"n_groups": len(set(g_as.values())), "largest": sz(g_as)},
        "plus_citations": {"n_groups": len(set(g_cit.values())), "largest": sz(g_cit)}}
    out["leakage"]["note"] = (
        "No priority/application/near-duplicate links among the 600 families, so default grouping = one "
        "group per family. Citation links exist and chain families; assignee is a strong confounder "
        "(the sample is stratified by assignee, up to 20 families each). Neither is controlled by "
        "the default split; a stricter assignee-grouped split is provided as a sensitivity analysis.")
    out["leakage"]["max_families_per_assignee"] = int(a["Probable Patent Assignee"].str.upper().value_counts().max())
    check("no exact duplicate title+abstract among annotated", out["leakage"]["exact_duplicate_title_abstract_pairs"] == 0)

    # ---- features
    try:
        D.assert_no_forbidden_features(D.FEATURE_COLUMNS)
        ok = True
    except ValueError:
        ok = False
    check("classifier feature set excludes adm_*, ann_*, assignee, claims, citations, ids", ok,
          f"features={D.FEATURE_COLUMNS}")
    out["feature_columns"] = D.FEATURE_COLUMNS
    out["excluded_columns_note"] = ("adm_* (sampling design, assignee, model probabilities), ann_* (coder "
                                    "evidence/notes), Assignees/Probable Patent Assignee, Claims, citations, "
                                    "identifiers and labels are never features; some are used only to build groups.")
    out["checks"] = checks
    out["all_checks_passed"] = all(c["passed"] for c in checks)
    return out


def md(o):
    L = ["# Training dataset audit (integrated family dataset)", "",
         "Generated by `scripts/audit_training_dataset.py`; machine-readable twin: "
         "`master_dataset/training_dataset_audit.json`. Source labels are never altered.", "",
         "## Checks", "", "| Check | Result | Detail |", "|---|---|---|"]
    L += [f"| {c['check']} | {'PASS' if c['passed'] else '**FAIL**'} | {c['detail']} |" for c in o["checks"]]
    L += ["", f"**All checks passed: {o['all_checks_passed']}**", ""] + [f"> WARNING: {w}" for w in o["warnings"]] + [""]
    L += [ "## Rows", "",
          f"`{json.dumps(o['rows'])}`; annotated families: {o['annotated_families']}.", "",
          "## Labels (annotated families)", "", "| Role | Source column | Y | N | U | empty |", "|---|---|---|---|---|---|"]
    L += [f"| {n} | {v['column']} | {v['Y']} | {v['N']} | {v['U']} | {v['']} |" for n, v in o["label_counts"].items()]
    L += ["", f"Label semantics: {o['label_semantics']}.", "",
          f"Y-count per family: `{o['n_Y_per_family']}`. All-U rows: {o['rows_all_U']}; partially-U rows: "
          f"{o['rows_partially_U']}. Pairwise phi: `{o['label_phi_correlation']}`.", "",
          "Unknown (`U`) is kept as unknown: masked in loss and metrics, never converted to N. The mapping of "
          "`F_*` columns to the letters M/P/R/I is inferred in the master build and still awaits confirmation.", "",
          "## Missing fields", "", f"`{json.dumps(o['missing_in_annotated'])}`; abstract length chars: `{json.dumps(o['abstract_chars'])}`.", "",
          "## Figure descriptions", "", f"Enrichment status (annotated): `{json.dumps(o['enrichment_status_annotated'])}`.", "",
          f"- With usable description: **{o['families_with_usable_description']}**; without: {o['families_without_usable_description']}; "
          f"total usable descriptions: {o['usable_descriptions_total']}.",
          f"- Duplicate descriptions: `{json.dumps(o['duplicate_descriptions'])}`.",
          f"- Figure rows by status: `{json.dumps(o['figure_rows_by_status'])}`; failures: `{json.dumps(o['failed_figure_records'])}`; "
          f"failure reasons: `{json.dumps(o['failure_reasons'])}`.",
          f"- Models/revisions: `{json.dumps(o['vlm_models_in_dataset'])}`.",
          "- No human review of descriptions has been done: semantic accuracy is unknown.", "",
          "## Eligibility", "", f"`{json.dumps(o['eligibility'])}`", "", "Label support (Y/N/U):", ""]
    for k, v in o["label_support_by_subset"].items():
        L.append(f"- {k}: `{json.dumps(v)}`")
    L += ["", o["BC_selection_note"], "", "## Leakage", "", f"`{json.dumps(o['leakage'], indent=1)}`", "",
          "## Features", "", f"Allowed: `{o['feature_columns']}`. {o['excluded_columns_note']}", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--master-dir", default=str(ROOT / "master_dataset"))
    ap.add_argument("--enrich-dir", default=str(ROOT / "data" / "figure_enrichment"))
    ap.add_argument("--docs-dir", default=str(ROOT / "docs"))
    a = ap.parse_args(argv)
    o = run(Path(a.master_dir), Path(a.enrich_dir))
    (Path(a.master_dir) / "training_dataset_audit.json").write_text(json.dumps(o, indent=2, default=str) + "\n")
    (Path(a.docs_dir) / "TRAINING_DATASET_AUDIT.md").write_text(md(o))
    print(json.dumps({"all_checks_passed": o["all_checks_passed"],
                      "failed": [c for c in o["checks"] if not c["passed"]]}, indent=2))


if __name__ == "__main__":
    main()
