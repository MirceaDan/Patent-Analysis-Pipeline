"""Run controlled baselines A / B / C and write all artifacts.

  A  text-only (title+abstract) on ALL eligible annotated families
  B  text-only on the image-description subset (families with >=1 usable description)
  C  text + image descriptions on exactly the B families, same splits/seeds as B
  A_on_BC_test  model A evaluated only on the B/C test families of each repeat (same test
             families as B and C but trained on all eligible training families)

Usage: python -m mpri_baseline.run_experiments --out experiments/baseline_v1 [--group-mode default|assignee]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from . import data as D
from . import model as M
from .groups import related_family_groups
from .metrics import aggregate
from .splits import make_assignments

ROOT = Path(__file__).resolve().parents[1]
PRED_COLS = [f"p_{n}" for n in D.LABEL_NAMES]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _arrays(df):
    y, known = D.label_matrix(df)
    return y, known


def run_repeat(df, assign_r, in_bc, seed):
    """Fit A, B, C for one repeat. Returns dict of prediction frames and selection logs."""
    split = dict(zip(assign_r["family"], assign_r["split"]))
    sp = df["Family Number"].map(split)
    out, logs, models = {}, {}, {}

    def fit_eval(name, sub, use_images):
        s = sp.loc[sub.index]
        tr, va, te = sub[s == "train"], sub[s == "val"], sub[s == "test"]
        if len(tr) == 0 or len(va) == 0 or len(te) == 0:
            return None
        ytr, ktr = _arrays(tr)
        yva, kva = _arrays(va)
        mdl, log = M.fit(D.build_text(tr, use_images), ytr, ktr, D.build_text(va, use_images), yva, kva, seed=seed)
        yte, kte = _arrays(te)
        p = mdl.predict_proba(D.build_text(te, use_images))
        frame = pd.DataFrame({"family": te["Family Number"].to_numpy(),
                              **{f"y_{n}": yte[:, i] for i, n in enumerate(D.LABEL_NAMES)},
                              **{f"known_{n}": kte[:, i] for i, n in enumerate(D.LABEL_NAMES)},
                              **{f"p_{n}": p[:, i] for i, n in enumerate(D.LABEL_NAMES)}})
        logs[name] = {"selected_C": mdl.C, "validation": log,
                      "n_train": len(tr), "n_val": len(va), "n_test": len(te)}
        models[name] = mdl
        return frame

    fa = fit_eval("A", df, False)
    out["A"] = fa
    if fa is not None:
        out["A_on_BC_test"] = fa[fa["family"].isin(set(df.loc[in_bc, "Family Number"]))]
    sub = df[in_bc]
    out["B"] = fit_eval("B", sub, False)
    out["C"] = fit_eval("C", sub, True)
    return out, logs, models


def frame_metrics(fr):
    y = fr[[f"y_{n}" for n in D.LABEL_NAMES]].to_numpy()
    k = fr[[f"known_{n}" for n in D.LABEL_NAMES]].to_numpy().astype(bool)
    p = fr[PRED_COLS].to_numpy()
    return aggregate(y, k, p)


def bootstrap_diff(f1, f2, n_boot=1000, seed=0):
    """Family-level paired bootstrap of metric(f2) - metric(f1) on pooled predictions."""
    fams = np.array(sorted(set(f1["family"]) & set(f2["family"])))
    g1, g2 = {k: v for k, v in f1.groupby("family")}, {k: v for k, v in f2.groupby("family")}
    rng = np.random.default_rng(seed)
    keys = ["macro_average_precision", "macro_f1", "micro_f1"]
    acc = {k: [] for k in keys}
    lab = {(n, m): [] for n in D.LABEL_NAMES for m in ("average_precision", "f1")}
    for _ in range(n_boot):
        pick = rng.choice(fams, size=len(fams), replace=True)
        m1 = frame_metrics(pd.concat([g1[f] for f in pick]))
        m2 = frame_metrics(pd.concat([g2[f] for f in pick]))
        for k in keys:
            acc[k].append(m2[k] - m1[k])
        for (n, m) in lab:
            lab[(n, m)].append(m2["per_label"][n][m] - m1["per_label"][n][m])
    ci = lambda v: ([float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
                    if np.isfinite(v).sum() >= 0.5 * len(v) else None)
    return {"n_families": int(len(fams)), "n_boot": n_boot,
            "aggregate": {k: {"ci95": ci(np.array(v))} for k, v in acc.items()},
            "per_label": {f"{n}:{m}": {"ci95": ci(np.array(v))} for (n, m), v in lab.items()}}


def fmt(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def results_md(res, cfg):
    names = {"A": "A  text, all eligible", "A_on_BC_test": "A on BC-test (model A, B/C test families)",
             "B": "B  text, image subset", "C": "C  text+image descriptions, image subset"}
    L = [f"# Baseline results ({cfg['run_id']})", "",
         "Pooled over repeated grouped splits (test predictions of all repeats concatenated; the "
         "same family appears in several repeats, so rows are NOT independent). `n/a` = undefined "
         "(no positives or no negatives in the test rows). Threshold 0.5 for F1.", "",
         "| Experiment | test rows | macro AP | macro F1 | micro F1 | macro ROC-AUC | Hamming (known only) |", "|---|---|---|---|---|---|---|"]
    for k, nm in names.items():
        if k in res["pooled"]:
            m = res["pooled"][k]
            L.append(f"| {nm} | {res['n_test_rows'][k]} | {fmt(m['macro_average_precision'])} | {fmt(m['macro_f1'])} | "
                     f"{fmt(m['micro_f1'])} | {fmt(m['macro_roc_auc'])} | {fmt(m['hamming_loss_known_only'])} |")
    for k, nm in names.items():
        if k not in res["pooled"]:
            continue
        L += ["", f"### {nm}: per label (pooled)", "", "| Role | pos | neg | unk excl. | prevalence | P | R | F1 | AP | ROC-AUC |", "|---|---|---|---|---|---|---|---|---|---|"]
        for n, v in res["pooled"][k]["per_label"].items():
            L.append(f"| {n} | {v['support_pos']} | {v['support_neg']} | {v['n_unknown_excluded']} | {fmt(v['prevalence'])} | "
                     f"{fmt(v['precision'])} | {fmt(v['recall'])} | {fmt(v['f1'])} | {fmt(v['average_precision'])} | {fmt(v['roc_auc'])} |")
    L += ["", "## Per-repeat summary (mean ± sd over repeats where defined)", "", "| Experiment | repeats | macro AP | macro F1 | micro F1 |", "|---|---|---|---|---|"]
    for k, nm in names.items():
        if k in res["per_repeat"]:
            r = res["per_repeat"][k]
            L.append(f"| {nm} | {r['n_repeats']} | " + " | ".join(
                f"{fmt(r[m]['mean'])} ± {fmt(r[m]['sd'])}" for m in ("macro_average_precision", "macro_f1", "micro_f1")) + " |")
    L += ["", "## Comparisons (pooled differences, family-level paired bootstrap 95% CI)", ""]
    for cname, c in res["comparisons"].items():
        L += [f"**{cname}** (diff = {c['b']} minus {c['a']}): {c['description']}", "", "| Metric | diff | 95% CI |", "|---|---|---|"]
        for k, v in c["aggregate"].items():
            L.append(f"| {k} | {fmt(v['diff'])} | {v['ci95'] and [round(x, 3) for x in v['ci95']]} |")
        for k, v in c["per_label"].items():
            L.append(f"| {k} | {fmt(v['diff'])} | {v['ci95'] and [round(x, 3) for x in v['ci95']]} |")
        L.append("")
    L += ["## Limitations", ""] + [f"- {x}" for x in cfg["limitations"]]
    return "\n".join(L) + "\n"


def run(args):
    out = Path(args.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out / "run.log", mode="w"), logging.StreamHandler()])
    df_all = D.load_annotated(Path(args.dataset))
    el = D.eligibility(df_all)
    df = df_all[el["eligible_A"].to_numpy()].reset_index(drop=True)
    in_bc = (df["n_image_descriptions"].astype(int) > 0).to_numpy()
    groups = related_family_groups(df, use_assignee=(args.group_mode == "assignee"))
    seeds = [args.seed + r for r in range(args.repeats)]
    assign = make_assignments(df, groups, seeds)
    assign["in_image_subset"] = assign["family"].isin(set(df.loc[in_bc, "Family Number"]))
    assign.to_csv(out / "splits.csv", index=False)
    logging.info("eligible A=%d, image subset B/C=%d, repeats=%d, group_mode=%s",
                 len(df), int(in_bc.sum()), args.repeats, args.group_mode)

    preds = {k: [] for k in ("A", "A_on_BC_test", "B", "C")}
    sel = []
    for r, seed in enumerate(seeds):
        fr, logs, models = run_repeat(df, assign[assign["repeat"] == r], in_bc, seed)
        for k, v in fr.items():
            if v is not None:
                preds[k].append(v.assign(repeat=r))
        sel.append({"repeat": r, "seed": seed, **logs})
        for k, mdl in models.items():
            joblib.dump(mdl, out / "checkpoints" / f"{k}_repeat{r:02d}.joblib")
        logging.info("repeat %d: %s", r, {k: (v["selected_C"], v["n_train"], v["n_val"], v["n_test"]) for k, v in logs.items()})

    pooled, n_rows, per_rep = {}, {}, {}
    for k, lst in preds.items():
        if not lst:
            continue
        allp = pd.concat(lst, ignore_index=True)
        allp.to_csv(out / f"predictions_{k}.csv", index=False)
        pooled[k], n_rows[k] = frame_metrics(allp), len(allp)
        vals = {m: [] for m in ("macro_average_precision", "macro_f1", "micro_f1")}
        for fr in lst:
            mm = frame_metrics(fr)
            for m in vals:
                vals[m].append(mm[m])
        per_rep[k] = {"n_repeats": len(lst), **{m: {"mean": float(np.nanmean(v)) if np.isfinite(v).any() else None,
                                                     "sd": float(np.nanstd(v)) if np.isfinite(v).any() else None}
                                                 for m, v in vals.items()}}

    comps = {}
    for cname, (a, b, desc) in {
        "A_vs_B": ("A", "B", "effect of restricting training AND test to the image subset (different test sets; confounded by size and test composition)"),
        "A_on_BC_test_vs_B": ("A_on_BC_test", "B", "same test families; training on all eligible families (A) vs only the image subset (B)"),
        "B_vs_C": ("B", "C", "contribution of image descriptions: same families, splits, seeds, hyperparameter grid")}.items():
        if a not in preds or b not in preds or not preds[a] or not preds[b]:
            continue
        fa, fb = pd.concat(preds[a]), pd.concat(preds[b])
        ma, mb = pooled[a], pooled[b]
        diff = lambda x, y: None if np.isnan(x) or np.isnan(y) else float(y - x)
        if cname == "A_vs_B":
            bs = None  # different families: no paired bootstrap
        else:
            bs = bootstrap_diff(fa, fb, seed=args.seed)
        comps[cname] = {"description": desc, "a": a, "b": b,
                        "aggregate": {k: {"diff": diff(ma[k], mb[k]), "ci95": bs and bs["aggregate"][k]["ci95"]}
                                      for k in ("macro_average_precision", "macro_f1", "micro_f1")},
                        "per_label": {f"{n}:{m}": {"diff": diff(ma["per_label"][n][m], mb["per_label"][n][m]),
                                                   "ci95": bs and bs["per_label"][f"{n}:{m}"]["ci95"]}
                                      for n in D.LABEL_NAMES for m in ("average_precision", "f1")},
                        "n_paired_families": bs and bs["n_families"]}
    cfg = {"run_id": out.name, "seed_base": args.seed, "seeds": seeds, "repeats": args.repeats,
           "group_mode": args.group_mode, "n_folds": 5,
           "split": "fold0=test, fold1=val, folds2-4=train (StratifiedGroupKFold, shuffle)",
           "model": {"tfidf": {**M.TFIDF, "ngram_range": list(M.TFIDF["ngram_range"])}, "classifier":
                     "LogisticRegression(l2, liblinear, class_weight=balanced) per label, known targets only",
                     "C_grid": list(M.C_GRID), "fallback_C": M.DEFAULT_C, "threshold": 0.5,
                     "selection": "validation macro average precision"},
           "feature_columns": {"text_only": D.FEATURE_COLUMNS[:2], "text_plus_images": D.FEATURE_COLUMNS},
           "n_eligible_A": len(df), "n_BC": int(in_bc.sum()),
           "dataset": str(Path(args.dataset).relative_to(ROOT)) if Path(args.dataset).is_relative_to(ROOT) else args.dataset,
           "dataset_sha256": sha256(args.dataset),
           "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                        "numpy": np.__version__, "pandas": pd.__version__},
           "limitations": [
               f"B/C contain only {int(in_bc.sum())} families; label support for R and I is ~3 positives. Per-label metrics are extremely noisy and the B-vs-C comparison cannot support a conclusion about image descriptions.",
               "The image subset is not a random sample (diversity-selected pilot) and is restricted to publications where Google Patents lists figures (US/DE here, no CN).",
               "Repeated splits reuse the same families, so pooled test rows and repeats are not independent; bootstrap CIs resample families but understate uncertainty from the small training sets.",
               "Descriptions come from a 3B VLM and have NOT been human-reviewed; 30 of them lack a recorded model revision.",
               "Default grouping has no related-family links (each family is its own group); citation and assignee relatedness are not controlled unless --group-mode assignee is used. The sample is stratified by assignee.",
               "Sampling was stratified (not a population sample); metrics are not population estimates; labels F_* -> M/P/R/I mapping is inferred.",
               "Checkpoints are scikit-learn pickles tied to the recorded library versions."]}
    res = {"pooled": pooled, "n_test_rows": n_rows, "per_repeat": per_rep, "comparisons": comps,
           "selection": sel}
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (out / "metrics.json").write_text(json.dumps(res, indent=2, default=lambda o: None if o != o else o) + "\n")
    (out / "results.md").write_text(results_md(res, cfg))
    logging.info("wrote %s", out)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "experiments" / "baseline_v1"))
    ap.add_argument("--dataset", default=str(D.ENRICHED))
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--group-mode", choices=["default", "assignee"], default="default")
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
