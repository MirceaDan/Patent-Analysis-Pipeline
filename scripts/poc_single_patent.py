"""Single-patent end-to-end proof of concept (NOT a training or evaluation task).

Stages (each writes into data/figure_enrichment/poc_single_patent/):
  select    choose the patent by a fixed rule that never looks at any classifier output
            -> selection.json  (human labels are recorded there for provenance, but are
            NOT read by `assess`)
  assess    read ONLY title/abstract/figure descriptions from the crawl output, run
            (1) an existing trained checkpoint (mpri_baseline, held-out for this family),
            (2) a transparent keyword-evidence baseline against the documented definitions
            -> classifier_input.json, predictions.json (+ its sha256 printed and stored)
  report    verify predictions.json is unchanged, only now load the human labels, write
            comparison.json and REPORT.md
Selection rule (fixed before any prediction): annotated, not in the earlier 20-family pilot,
Google-Patents-eligible (figures listed and reachable), ann_CONFIDENCE=HIGH, all four labels in
{Y,N}, at least one Y and one N, abstract >= 700 chars, 3..10 figures; first by
sha256("poc-single-patent-v1:" + family).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

csv.field_size_limit(1 << 28)
ROOT = Path(__file__).resolve().parents[1]
POC = ROOT / "data" / "figure_enrichment" / "poc_single_patent"
LABEL_COLS = {"M": "F_MONITOR_REPRESENT", "P": "F_PREDICT_SIMULATE",
              "R": "F_RECOMMEND_DECIDE", "I": "F_INTERVENE_CONTROL"}
SALT = "poc-single-patent-v1:"

# Definitions quoted from database/JSTPM_Fisa_Tehnica_Completa_*.md (A2.1 role codebook).
DEFINITIONS = {
    "M": "Monitoring or representation: observation, representation and updates to the state of a physical or operational counterpart.",
    "P": "Prediction or simulation: future-state estimation, forecasting and scenario analysis.",
    "R": "Recommendation or decision support: selection or recommendation of actions.",
    "I": "Intervention or control: action on the physical or managed system.",
}
# Cue lexicon of the keyword baseline. This is an ad-hoc design choice, NOT validated and not
# derived from the dataset; it only operationalises the wording of DEFINITIONS.
CUES = {
    "M": ["monitor", "monitoring", "sensor", "sensing", "measure", "measuring", "measurement", "detect",
          "observe", "observation", "acquire", "collect", "digital twin", "state of", "status", "visualiz",
          "display", "represent", "real-time", "real time", "telemetry"],
    "P": ["predict", "prediction", "forecast", "simulat", "estimate the future", "future state", "scenario",
          "prognos", "projected", "anticipate", "remaining useful life", "what-if", "expected"],
    "R": ["recommend", "decision support", "decision-making", "decision making", "select an action",
          "suggest", "advise", "optimal action", "choose", "rank", "decide", "decision"],
    "I": ["control", "controlling", "actuat", "adjust", "regulat", "operate", "operating", "command signal",
          "switch", "intervene", "intervention", "valve", "motor", "drive", "execute an action", "automatically"],
}


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_rows(p):
    with open(p, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.;!?])\s+|\n", text) if s.strip()]


# --------------------------------------------------------------------------- select
def select(a):
    d = ROOT / "data" / "figure_enrichment"
    elig = read_rows(d / "eligible_families.csv")
    pilot = {r["family_number"] for r in read_rows(d / "pilot_families.csv")}
    ann = {r["Family Number"]: r for r in read_rows(ROOT / "master_dataset" / "patent_families_annotated.csv")
           if r["is_annotated"] == "True"}
    cands = []
    for e in elig:
        f = e["family_number"]
        r = ann[f]
        labs = [r[c] for c in LABEL_COLS.values()]
        if (f not in pilot and r["ann_CONFIDENCE"] == "HIGH" and all(x in "YN" and x for x in labs)
                and "Y" in labs and "N" in labs and len(r["Abstract"]) >= 700
                and 3 <= int(e["n_figures_discovered"]) <= 10):
            cands.append((hashlib.sha256((SALT + f).encode()).hexdigest(), f, e))
    cands.sort()
    _, f, e = cands[0]
    r = ann[f]
    sel = {"rule": __doc__.split("Selection rule")[1].strip(),
           "n_candidates": len(cands), "candidate_families_in_rank_order": [c[1] for c in cands],
           "family_number": f, "publication_number": e["publication_number"], "kind_code": e["kind_code"],
           "document_id": e["document_id"], "country": e["country"],
           "url": f"https://patents.google.com/patent/{e['document_id']}/en",
           "title": r["Title"], "n_figures_discovered_in_dry_run": int(e["n_figures_discovered"]),
           "annotation_confidence": r["ann_CONFIDENCE"],
           "human_annotations": "withheld here; read from the annotated master only by the `report` stage",
           "reason": ("First of %d families passing the fixed rule, ordered by a salted hash of the family number; "
                      "classifier outputs were not consulted. All four labels are known (no U) with a mix of Y and N, "
                      "the title+abstract is long enough, and Google Patents lists %s figures for the %s page."
                      % (len(cands), e["n_figures_discovered"], e["document_id"]))}
    POC.mkdir(parents=True, exist_ok=True)
    (POC / "selection.json").write_text(json.dumps(sel, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: sel[k] for k in ("family_number", "publication_number", "title", "url", "n_candidates")},
                     indent=2))


# --------------------------------------------------------------------------- inputs
def load_inputs():
    """Classifier input = Google-Patents title + abstract + validated figure descriptions ONLY."""
    sel = json.loads((POC / "selection.json").read_text())
    doc = json.loads((POC / "raw" / "documents" / f"{sel['document_id']}.json").read_text())
    if doc["document_id"] != sel["document_id"] or doc["family_number"] != sel["family_number"]:
        raise SystemExit("identity check failed: retrieved document != selected patent")
    figs = [r for r in read_rows(POC / "crawl" / "figure_enrichments.csv")
            if r["family_number"] == sel["family_number"] and r["source_document_id"] == sel["document_id"]]
    abstract = re.sub(r"^Abstract\s+", "", doc["text"]["abstract"]).strip()
    described = sorted((f for f in figs if f["processing_status"] == "described" and f["figure_description"].strip()),
                       key=lambda f: f["figure_id"])
    return sel, doc, figs, {"title": doc["text"]["title"].strip(), "abstract": abstract,
                            "figure_descriptions": [{"figure_id": f["figure_id"], "text": f["figure_description"].strip()}
                                                    for f in described]}


def passed_text(inp, images):
    parts = [inp["title"], inp["abstract"]] + ([d["text"] for d in inp["figure_descriptions"]] if images else [])
    return " . ".join(p for p in parts if p)


def segments(inp):
    """(source, sentence) pairs for evidence search."""
    out = [("title", inp["title"])] + [("abstract", x) for x in sentences(inp["abstract"])]
    for d in inp["figure_descriptions"]:
        out += [(f"figure {d['figure_id']}", x) for x in sentences(d["text"])]
    return [(a, b) for a, b in out if b]


# --------------------------------------------------------------------------- baselines
def keyword_baseline(inp, min_distinct=2):
    """Transparent rule baseline. raw score = distinct cue terms matched / lexicon size (NOT a
    probability). Predicted Y iff >= `min_distinct` distinct cues match (arbitrary, unvalidated)."""
    segs, res = segments(inp), {}
    for k, cues in CUES.items():
        hit, ev = set(), []
        for src, sent in segs:
            low = sent.lower()
            m = [c for c in cues if c in low]
            if m:
                hit |= set(m)
                ev.append({"source": src, "text": sent, "matched_cues": m})
        res[k] = {"raw_score": len(hit) / len(cues), "distinct_cues_matched": sorted(hit),
                  "predicted": "Y" if len(hit) >= min_distinct else "N", "evidence": ev[:6],
                  "explanation": (f"{len(hit)} distinct cue terms of the {k} lexicon matched "
                                  f"(rule: >= {min_distinct} => Y): {sorted(hit)}")}
    return res


def checkpoint_choice(family, exp_dir):
    """Lowest repeat index in which `family` is in the TEST split (never train or validation)."""
    rows = read_rows(exp_dir / "splits.csv")
    for r in rows:
        if r["family"] == family and r["split"] == "test":
            return int(r["repeat"]), int(r["seed"])
    raise SystemExit("family is never in a test split; no held-out checkpoint")


def model_prediction(mdl, text, top=6):
    import numpy as np
    X = mdl.vectorizer.transform([text])
    names = mdl.vectorizer.get_feature_names_out()
    res = {}
    for k, name in enumerate("MPRI"):
        head = mdl.heads[k]
        if not hasattr(head, "predict_proba"):
            res[name] = {"raw_score": float(head), "predicted": "Y" if head >= .5 else "N", "top_terms": [],
                         "explanation": "constant prior head"}
            continue
        sc = float(head.predict_proba(X)[0, 1])
        contrib = X.multiply(head.coef_[0]).toarray()[0]
        idx = np.argsort(-contrib)[:top]
        terms = [{"term": names[i], "contribution": round(float(contrib[i]), 4)} for i in idx if contrib[i] > 0]
        res[name] = {"raw_score": sc, "predicted": "Y" if sc >= 0.5 else "N", "top_terms": terms,
                     "bias": float(head.intercept_[0]),
                     "decision_value": float(head.decision_function(X)[0])}
    return res


def attach_model_evidence(preds, inp):
    segs = segments(inp)
    for name, p in preds.items():
        ev = []
        for t in p.get("top_terms", []):
            words = t["term"].split()
            for src, sent in segs:
                if all(w in re.sub(r"[^a-z0-9 ]", " ", sent.lower()) for w in words):
                    ev.append({"source": src, "text": sent, "matched_term": t["term"]})
                    break
        p["evidence"] = ev
        p["explanation"] = ("raw logistic-regression score (uncalibrated; class_weight=balanced so it is not a prior-"
                            "adjusted probability) = sigmoid(bias + sum of tf-idf*weight); evidence = sentences "
                            "containing the highest-contributing terms. Terms are statistical, not definitions.")
    return preds


# --------------------------------------------------------------------------- assess
def assess(a):
    import joblib
    sys.path.insert(0, str(ROOT))
    sel, doc, figs, inp = load_inputs()
    exp = ROOT / "experiments" / "baseline_v1"
    rep, seed = checkpoint_choice(sel["family_number"], exp)
    cfg = json.loads((exp / "config.json").read_text())
    ck = {}
    for name in ("A", "C"):
        path = exp / "checkpoints" / f"{name}_repeat{rep:02d}.joblib"
        ck[name] = {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path), "model": joblib.load(path)}
    # C was trained only on image-subset families; the selected family must not be among them
    bc = {r["family"] for r in read_rows(exp / "splits.csv") if r["in_image_subset"] == "True"}
    c_leak = sel["family_number"] in bc
    runs = {
        "A_title_abstract": ("A", False, "PRIMARY. Checkpoint A (text-only), fed exactly its training input format"),
        "A_title_abstract_figdesc": ("A", True, "SENSITIVITY. Checkpoint A fed figure descriptions it never saw in training (out-of-distribution)"),
        "C_title_abstract_figdesc": ("C", True, "WEAK. Checkpoint C (text+descriptions) was trained on only ~8-13 families; not credible")}
    out = {"model_predictions": {}, "keyword_baseline": None}
    for key, (m, imgs, note) in runs.items():
        txt = passed_text(inp, imgs)
        out["model_predictions"][key] = {
            "note": note, "text_passed_to_classifier": txt, "checkpoint": {k: v for k, v in ck[m].items() if k != "model"},
            "repeat": rep, "seed": seed, "trained_on_this_family": False if m == "C" and not c_leak else None,
            "scores_are": "raw uncalibrated logistic-regression outputs; NOT calibrated probabilities; threshold 0.5",
            "per_label": attach_model_evidence(model_prediction(ck[m]["model"], txt), inp)}
    out["keyword_baseline"] = {"note": "EXPERIMENTAL transparent rule baseline; unvalidated cue lexicon (see CUES in this script)",
                               "text_scope": "title + abstract + figure descriptions", "per_label": keyword_baseline(inp)}
    out["checkpoint_split_check"] = {"experiment": "experiments/baseline_v1", "repeat": rep, "seed": seed,
                                     "family_split_in_that_repeat_for_A": "test (never used for fitting or C selection)",
                                     "family_in_image_subset_used_to_train_C": c_leak,
                                     "checkpoint_config_sha256": sha256_file(exp / "config.json")}
    if c_leak:
        raise SystemExit("leakage: selected family is in the image subset used to train C")
    POC.mkdir(exist_ok=True)
    (POC / "classifier_input.json").write_text(json.dumps(
        {"family_number": sel["family_number"], "document_id": sel["document_id"], **inp,
         "text_title_abstract": passed_text(inp, False), "text_title_abstract_figdesc": passed_text(inp, True),
         "fields_NOT_provided": ["human labels", "family id", "assignee/company", "adm_* / ann_* fields",
                                 "claims", "citations", "description text (page has none)"]}, indent=2, ensure_ascii=False) + "\n")
    pp = POC / "predictions.json"
    pp.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print("predictions saved; sha256 =", sha256_file(pp))
    (POC / "predictions.sha256").write_text(sha256_file(pp) + "\n")


# --------------------------------------------------------------------------- report
def report(a):
    sel, doc, figs, inp = load_inputs()
    pp = POC / "predictions.json"
    if sha256_file(pp) != (POC / "predictions.sha256").read_text().strip():
        raise SystemExit("predictions.json changed after being saved; refusing to reveal labels")
    preds = json.loads(pp.read_text())
    ann = next(r for r in read_rows(ROOT / "master_dataset" / "patent_families_annotated.csv")
               if r["Family Number"] == sel["family_number"])          # labels first read HERE
    human = {k: ann[c] for k, c in LABEL_COLS.items()}
    systems = {k: {n: v["predicted"] for n, v in p["per_label"].items()} for k, p in preds["model_predictions"].items()}
    systems["keyword_baseline"] = {n: v["predicted"] for n, v in preds["keyword_baseline"]["per_label"].items()}
    comp = {s: {k: {"predicted": pr[k], "human": human[k], "agree": pr[k] == human[k]} for k in "MPRI"}
            for s, pr in systems.items()}
    (POC / "comparison.json").write_text(json.dumps({"human_annotations": human, "annotation_confidence": ann["ann_CONFIDENCE"],
                                                    "annotation_evidence_source": ann["ann_EVIDENCE_SOURCE"],
                                                    "comparison": comp}, indent=2) + "\n")
    fmt = lambda x: f"{x:.3f}"
    L = [f"# Single-patent M/P/R/I proof of concept: {sel['publication_number']} (family {sel['family_number']})", "",
         "**One patent. This demonstrates that the pipeline runs end to end; it is not an evaluation and agreement or "
         "disagreement with one patent says nothing about accuracy.**", "",
         "## 1. Patent identity", "", f"- Title (Google Patents): {inp['title']}",
         f"- Dataset title: {sel['title']}", f"- Family: {sel['family_number']}; publication {sel['publication_number']} ({sel['kind_code']}); document id {sel['document_id']}",
         f"- URL: {doc['source_url']}", f"- Retrieved (UTC): {doc['retrieved_at_utc']}; publication date {doc['publication_date']}; application {doc['application_number']}",
         "- Identity verification: crawler verified document id and publication date against the dataset (`pub_number+pub_date`); the "
         "dataset abstract and the Google abstract are the same text apart from the `[EN]` marker.",
         f"- Selection reason: {sel['reason']}", f"- Selection rule: {sel['rule']}", "",
         "## 2. Figures and VLM descriptions", "",
         "| Figure | File | Status | Description / error |", "|---|---|---|---|"]
    for f in sorted(figs, key=lambda f: f["figure_id"]):
        fn = next(iter((POC / "raw" / "figures").glob(f"{f['source_document_id']}-{f['figure_id']}.*")), None)
        L.append(f"| {f['figure_id']} | {fn.relative_to(ROOT) if fn else 'not saved'} | {f['processing_status']} | "
                 f"{(f['figure_description'] or f['error_details']).replace('|', '/')[:400]} |")
    d0 = next(iter(figs), {})
    L += ["", f"VLM: `{d0.get('vlm_model')}` revision `{d0.get('vlm_model_version')}`, prompt `{d0.get('prompt_version')}`, "
          f"config hash `{d0.get('vlm_config_hash')}`, generation config `{d0.get('generation_config')}` (see `crawl/figure_enrichments.csv`).",
          "Failed figures are reported, not hidden: a failed generation (truncated JSON at the 2048-token limit) yields NO description and "
          "is not passed to any classifier. The usable description has not been checked by a human against the image.", "",
          "## 3. Text actually passed to the classifier", "", "**Title + abstract:**", "", "> " + passed_text(inp, False), "",
          "**Title + abstract + validated figure descriptions:**", "", "> " + passed_text(inp, True), "",
          f"The Google Patents page exposes no usable description text for this application (`{doc['text']['description'][:80]}`), so none is used. "
          "Not provided: human labels, family id, assignee, `adm_*`/`ann_*`, claims, citations.", "",
          "## 4. M/P/R/I definitions used", ""] + [f"- **{k}**: {v}" for k, v in DEFINITIONS.items()]
    L += ["", "Labels are independent (non-exclusive); each dimension is predicted separately. The human coding is based on the CLAIMS "
          f"(`ann_EVIDENCE_SOURCE={ann['ann_EVIDENCE_SOURCE']}`), which the classifiers did not see.", "", "## 5. Predictions (saved before labels were read)", "",
          f"`predictions.json` sha256 `{(POC / 'predictions.sha256').read_text().strip()}` verified unchanged before the human labels were loaded.", "",
          "Scores are RAW, UNCALIBRATED model outputs, not probabilities.", "",
          "| System | M | P | R | I |", "|---|---|---|---|---|"]
    for s, p in preds["model_predictions"].items():
        L.append(f"| {s} | " + " | ".join(f"{p['per_label'][k]['predicted']} ({fmt(p['per_label'][k]['raw_score'])})" for k in "MPRI") + " |")
    kb = preds["keyword_baseline"]["per_label"]
    L.append("| keyword_baseline | " + " | ".join(f"{kb[k]['predicted']} ({fmt(kb[k]['raw_score'])})" for k in "MPRI") + " |")
    ck = preds["model_predictions"]["A_title_abstract"]["checkpoint"]
    L += ["", f"Trained model: `{ck['path']}` (sha256 `{ck['sha256']}`), repeat {preds['checkpoint_split_check']['repeat']}, seed "
          f"{preds['checkpoint_split_check']['seed']}; this family was in that repeat's TEST split (not train/val) and, for checkpoint C, "
          "not in the image-subset training families. Config: `experiments/baseline_v1/config.json` (TF-IDF + per-label L2 logistic regression).", ""]
    for s, p in preds["model_predictions"].items():
        L += [f"### Evidence: {s}", "", p["note"] + ".", ""]
        for k in "MPRI":
            v = p["per_label"][k]
            L += [f"- **{k}** {v['predicted']} (raw {fmt(v['raw_score'])}): top terms " +
                  ", ".join(f"{t['term']} ({t['contribution']})" for t in v["top_terms"]) + ". Evidence: " +
                  (" / ".join(f"[{e['source']}] \"{e['text'][:220]}\"" for e in v["evidence"][:3]) or "none found")]
        L.append("")
    L += ["### Evidence: keyword baseline (experimental, unvalidated lexicon)", ""]
    for k in "MPRI":
        v = kb[k]
        L += [f"- **{k}** {v['predicted']}: {v['explanation']}. Evidence: " +
              (" / ".join(f"[{e['source']}] \"{e['text'][:220]}\" ({', '.join(e['matched_cues'])})" for e in v['evidence'][:3]) or "none")]
    L += ["", "## 6. Human annotation and agreement (revealed after the predictions were saved)", "",
          f"Human: " + ", ".join(f"{k}={v}" for k, v in human.items()) + f" (coder confidence {ann['ann_CONFIDENCE']}).", "",
          "| System | M | P | R | I | agree |", "|---|---|---|---|---|---|"]
    for s, c in comp.items():
        n = sum(c[k]["agree"] for k in "MPRI")
        L.append(f"| {s} | " + " | ".join(("agree" if c[k]["agree"] else "DISAGREE") + f" ({c[k]['predicted']} vs {c[k]['human']})" for k in "MPRI") + f" | {n}/4 |")
    L += ["", "## 7. Limitations and assumptions", "",
          "- n = 1; four binary comparisons; no statistical meaning. The prevalence of N in the human labels means a model that predicts mostly N will look good.",
          "- Human labels were derived from claims; the classifiers saw only title, abstract and (at most) one figure description. Claims were deliberately withheld.",
          "- 2 of 3 figures failed VLM generation (truncated JSON); only one description exists, and it is unverified against its image.",
          "- The page lists no description text for this US application; only title, abstract (and claims, unused) were retrieved.",
          "- Checkpoint A never saw figure descriptions in training; A+descriptions is out of distribution. Checkpoint C was trained on a handful of families and is not credible. "
          "No checkpoint trained on text + image descriptions at adequate scale exists yet.",
          "- Scores are uncalibrated. The keyword lexicon is an arbitrary operationalisation of the definitions, not validated.",
          "- Reproducibility: the VLM is greedy and the revision is pinned, but GPU kernels may not be bit-identical across hardware.",
          "- The agent that ran this POC saw the selection labels while choosing by rule; the classification code never read them before predictions were saved.", ""]
    (POC / "REPORT.md").write_text("\n".join(L))
    print("\n".join(L[L.index("## 6. Human annotation and agreement (revealed after the predictions were saved)"):][:9]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["select", "assess", "report"])
    a = ap.parse_args(argv)
    {"select": select, "assess": assess, "report": report}[a.stage](a)


if __name__ == "__main__":
    main()
