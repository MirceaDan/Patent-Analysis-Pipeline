"""Offline helpers for the Google-Patents figure-enrichment POC.

Reads crawler outputs (data/figure_enrichment/*) and the master dataset; makes
no network or VLM calls.

  eligible : eligible_families.csv + eligibility_summary.json
             A family is eligible iff the existing resolver verified a document
             (publication number [+ date]) that lists >=1 figure image and the
             HEAD probe did not mark that image inaccessible. Country is never
             used as a proxy; it is only reported.
  select   : pilot_families.csv, a deterministic greedy max-coverage selection
             of <=N eligible families over feature sets {F_*=value} + {IPC
             subclass}. Ties are broken by sha256(seed:family).
  review   : review_figures.csv, one row per described figure with blank human
             review fields (never pre-filled) + review_summary.json (generation
             statistics only; semantic quality is NOT assessed here).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

csv.field_size_limit(1 << 28)
ROOT = Path(__file__).resolve().parents[1]
LABELS = ["F_MONITOR_REPRESENT", "F_PREDICT_SIMULATE", "F_RECOMMEND_DECIDE",
          "F_INTERVENE_CONTROL"]
REVIEW_FIELDS = ["review_component_accuracy", "review_relationship_accuracy",
                 "review_text_in_figure_accuracy", "review_overall_usefulness",
                 "reviewer", "review_notes"]


def read(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write(path, rows, cols):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, cols, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def annotated(master):
    return {r["Family Number"]: r for r in
            read(master / "patent_families_annotated.csv") if r["is_annotated"] == "True"}


def eligible(a):
    d, master = Path(a.data_dir), Path(a.master_dir)
    fams = annotated(master)
    res = read(d / "document_resolution.csv")
    figs = read(d / "figure_enrichments.csv")
    summ = {r["family_number"]: r for r in read(d / "family_summary.csv")}
    bad_img = defaultdict(int)
    for f in figs:
        if f["processing_status"] == "image_inaccessible":
            bad_img[(f["family_number"], f["source_document_id"])] += 1
    rows, seen = [], set()
    for r in sorted(res, key=lambda r: (r["family_number"], r["publication_role"] != "master")):
        fam = r["family_number"]
        if fam in seen or fam not in fams:
            continue
        n = int(r["n_figures_listed"] or 0)
        if r["resolution_status"] == "resolved_with_figures" and n > 0 \
                and not bad_img.get((fam, r["document_id"])):
            seen.add(fam)
            rows.append({"family_number": fam,
                         "publication_number": r["publication_number"],
                         "kind_code": r["kind_code"],
                         "country": r["publication_number"][:2],
                         "document_id": r["document_id"],
                         "publication_role": r["publication_role"],
                         "n_figures_discovered": n,
                         "resolution_status": r["resolution_status"],
                         "match_verification": r["match_verification"]})
    cols = list(rows[0]) if rows else ["family_number"]
    write(d / "eligible_families.csv", rows, cols)
    excl = Counter(summ.get(f, {}).get("family_status", "not_yet_processed")
                   for f in fams if f not in seen)
    res_by = Counter(r["resolution_status"] for r in res if r["family_number"] in fams)
    out = {"annotated_families": len(fams), "eligible_families": len(rows),
           "excluded_families": len(fams) - len(rows),
           "exclusion_reasons_by_family_status": dict(sorted(excl.items())),
           "eligible_by_country": dict(sorted(Counter(r["country"] for r in rows).items())),
           "document_resolution_status_counts": dict(sorted(res_by.items())),
           "eligible_total_figures": sum(r["n_figures_discovered"] for r in rows),
           "definition": __doc__.split("eligible :")[1].split("select")[0].strip()}
    (d / "eligibility_summary.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


def features(row):
    f = {f"{k}={row.get(k, '')}" for k in LABELS}
    ipc = (row.get("adm_IPC") or "").replace(" ", "")
    f.add("IPC=" + (ipc[:4] if ipc else "none"))
    return f


def select(a):
    d, master = Path(a.data_dir), Path(a.master_dir)
    fams = annotated(master)
    elig = read(d / "eligible_families.csv")
    feats = {e["family_number"]: features(fams[e["family_number"]]) for e in elig}
    tie = lambda f: hashlib.sha256(f"{a.seed}:{f}".encode()).hexdigest()
    covered, chosen = set(), []
    pool = sorted(feats)
    while pool and len(chosen) < a.n:
        best = min(pool, key=lambda f: (-len(feats[f] - covered), tie(f)))
        chosen.append((best, sorted(feats[best] - covered)))
        covered |= feats[best]
        pool.remove(best)
    by = {e["family_number"]: e for e in elig}
    rows = [{**by[f], "title": fams[f]["Title"], "ipc": fams[f].get("adm_IPC", ""),
             **{k: fams[f][k] for k in LABELS}, "new_features_covered": " ; ".join(new),
             "selection_rank": i + 1} for i, (f, new) in enumerate(chosen)]
    write(d / "pilot_families.csv", rows, list(rows[0]))
    print(f"selected {len(rows)} of {len(elig)} eligible; features covered {len(covered)}/"
          f"{len(set().union(*feats.values()))}")
    print(",".join(f for f, _ in chosen))


def review(a):
    d = Path(a.data_dir)
    fam_filter = set(read(d / "pilot_families.csv") and
                     [r["family_number"] for r in read(d / "pilot_families.csv")]) \
        if (d / "pilot_families.csv").exists() and not a.all else None
    figs = [r for r in read(d / "figure_enrichments.csv")
            if fam_filter is None or r["family_number"] in fam_filter]
    rows = []
    for f in figs:
        if f["processing_status"] != "described":
            continue
        dj = json.loads(f["description_json"] or "{}")
        rows.append({
            "family_number": f["family_number"], "publication_number": f["publication_number"],
            "kind_code": f["kind_code"], "figure_id": f["figure_id"],
            "image_source_url": f["image_source_url"], "image_sha256": f["image_sha256"],
            "source_document_url": f["source_document_url"],
            "figure_type": dj.get("figure_type", ""),
            "figure_description": f["figure_description"],
            "components": "; ".join(f"{c.get('label') or ''} {c['name']} [{c['evidence']}]"
                                    for c in dj.get("components", [])),
            "relations": "; ".join(f"{r['source']} -> {r['target']} ({r['relation']}) [{r['evidence']}]"
                                   for r in dj.get("relations", [])),
            "text_in_figure": " | ".join(dj.get("text_in_figure", [])),
            "uncertainties": " | ".join(dj.get("uncertainties", [])),
            "vlm_model": f["vlm_model"], "vlm_model_version": f["vlm_model_version"],
            "prompt_version": f["prompt_version"], "reused_from": f["reused_from"],
            **{k: "" for k in REVIEW_FIELDS}})
    write(d / "review_figures.csv", rows, list(rows[0]) if rows else ["family_number"])
    scope = [r for r in read(d / "figure_enrichments.csv")
             if fam_filter is None or r["family_number"] in fam_filter]
    attempted = [r for r in scope if int(r["attempts"] or 0) > 0]
    st = Counter(r["processing_status"] for r in attempted)
    reasons = Counter()
    for r in attempted:
        if r["processing_status"] != "described":
            e = r["error_details"]
            reasons[e.split(":")[1].strip().split(" ")[0] if "gave up" in e and ":" in e
                    else e[:60]] += 1
    fam_with = {r["family_number"] for r in scope if r["processing_status"] == "described"}
    out = {"families_in_scope": len(fam_filter) if fam_filter else len({r["family_number"] for r in scope}),
           "families_with_>=1_description": len(fam_with),
           "figures_discovered": len(scope),
           "figures_attempted": len(attempted),
           "attempted_status_counts": dict(st),
           "generation_success_rate": round(st["described"] / len(attempted), 4) if attempted else None,
           "failure_reasons": dict(reasons),
           "skipped_limit": sum(r["processing_status"] == "skipped_limit" for r in scope),
           "human_review_completed_rows": 0,
           "note": "Review fields are blank; no semantic accuracy is claimed."}
    (d / "review_summary.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["eligible", "select", "review"])
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "figure_enrichment"))
    ap.add_argument("--master-dir", default=str(ROOT / "master_dataset"))
    ap.add_argument("-n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261124)
    ap.add_argument("--all", action="store_true", help="review: all families, not only the pilot")
    a = ap.parse_args(argv)
    {"eligible": eligible, "select": select, "review": review}[a.cmd](a)


if __name__ == "__main__":
    main()
