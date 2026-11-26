"""Integrate VLM figure descriptions into the family-level master dataset.

Inputs (read-only):
  master_dataset/patent_families_annotated.csv   one row per family (28,311)
  master_dataset/patent_publications.csv         publications per family
  data/figure_enrichment/figure_enrichments.csv  latest row per figure
  data/figure_enrichment/document_resolution.csv one row per examined publication

Outputs (new files; no input is modified):
  master_dataset/patent_families_enriched.csv    SAME rows/columns as the annotated
      file (one row per family, labels and `label_M/P/R/I` aliases untouched, `U`
      preserved) plus the enrichment columns below.
  master_dataset/figure_enrichment_integration.json   counts + input checksums

Join keys: the family key is `Family Number` (= `family_number` in the figure
files). Figures are attached by family only; the publication number, kind code and
figure id of every description travel in `image_description_provenance`.
The enrichment tables are never joined row-wise onto the family table, so multiple
figures/publications cannot multiply family rows.

Added columns
  image_descriptions            usable descriptions joined by "\\n" in a deterministic
                                order, no identifiers inside the text; "" if none.
  n_image_descriptions          number of usable descriptions in `image_descriptions`.
  image_description_provenance  JSON list aligned index-wise with the text lines:
                                publication_number, kind_code, figure_id,
                                source_document_id/url, image_source_url,
                                image_sha256, vlm_model, vlm_model_version,
                                prompt_version, vlm_config_hash, processed_at_utc.
  figure_enrichment_status      one of STATUSES (below).
  figure_enrichment_detail      JSON: publication / resolution / figure counts.

A description is *usable* iff processing_status == "described" and the text is
non-empty. Order: publication_role (master first), publication_number, figure id
(natural order).

Status semantics (derived here from document_resolution + figure rows, not from
family_summary.csv, which only reflects the most recent crawler run):
  described                       >=1 usable description
  vlm_failed                      figures were attempted, none usable, >=1 vlm_failed
  image_inaccessible              figures attempted, none usable, image download failed
  figures_discovered_not_processed figures listed by the source but none attempted
                                  (per-family cap / pending); NOT an absence of figures
  no_figures_listed_by_source     every examined document resolved and lists no figures.
                                  This is a statement about Google Patents, not about the
                                  patent (all CN pages list none).
  document_not_found / document_inaccessible   no examined document could be resolved
  not_enriched                    no resolution record (family outside the crawl scope)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STATUSES = ["described", "vlm_failed", "image_inaccessible",
            "figures_discovered_not_processed", "no_figures_listed_by_source",
            "document_not_found", "document_inaccessible", "not_enriched"]
ADDED = ["image_descriptions", "n_image_descriptions", "image_description_provenance",
         "figure_enrichment_status", "figure_enrichment_detail"]
PROV_FIELDS = ["publication_number", "kind_code", "figure_id", "source_document_id",
               "source_document_url", "image_source_url", "image_sha256", "vlm_model",
               "vlm_model_version", "prompt_version", "vlm_config_hash", "processed_at_utc"]


def read_str(path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False,
                       encoding="utf-8-sig")


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _natural(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s or "")]


def family_status(res_statuses, fig_statuses):
    """Map examined-document and figure statuses of one family to STATUSES."""
    if "described" in fig_statuses:
        return "described"
    if fig_statuses:
        if "vlm_failed" in fig_statuses:
            return "vlm_failed"
        if "image_inaccessible" in fig_statuses:
            return "image_inaccessible"
        return "figures_discovered_not_processed"
    if not res_statuses:
        return "not_enriched"
    if "resolved_no_figures" in res_statuses:
        return "no_figures_listed_by_source"
    if "document_inaccessible" in res_statuses:
        return "document_inaccessible"
    if "resolved_with_figures" in res_statuses:  # listed but no figure rows recorded
        return "figures_discovered_not_processed"
    return "document_not_found"


def build_enrichment(families: pd.DataFrame, pubs: pd.DataFrame, figs: pd.DataFrame,
                     res: pd.DataFrame) -> pd.DataFrame:
    """Return `families` with the ADDED columns; row count and order unchanged."""
    if families["Family Number"].duplicated().any():
        raise ValueError("input family table has duplicate Family Number")
    n_pub = pubs.groupby("family_number").size().to_dict()
    res_by = {k: g for k, g in res.groupby("family_number")}
    fig_by = {k: g for k, g in figs.groupby("family_number")}
    texts, counts, provs, stats, details = [], [], [], [], []
    for fam in families["Family Number"]:
        r = res_by.get(fam)
        f = fig_by.get(fam)
        res_st = list(r["resolution_status"]) if r is not None else []
        fig_st = list(f["processing_status"]) if f is not None else []
        usable = []
        if f is not None:
            u = f[(f["processing_status"] == "described")
                  & (f["figure_description"].str.strip() != "")]
            order = sorted(u.index, key=lambda i: (
                u.at[i, "publication_role"] != "master", u.at[i, "publication_number"],
                _natural(u.at[i, "figure_id"])))
            usable = [u.loc[i] for i in order]
        texts.append("\n".join(x["figure_description"].strip().replace("\n", " ")
                               for x in usable))
        counts.append(len(usable))
        provs.append(json.dumps([{k: x.get(k, "") for k in PROV_FIELDS} for x in usable],
                                ensure_ascii=False))
        stats.append(family_status(res_st, fig_st))
        details.append(json.dumps({
            "n_publications_total": n_pub.get(fam, 0),
            "n_publications_examined": len(res_st),
            "resolution_status_counts": {k: res_st.count(k) for k in sorted(set(res_st))},
            "figure_status_counts": {k: fig_st.count(k) for k in sorted(set(fig_st))},
        }, sort_keys=True))
    out = families.copy()
    out["image_descriptions"] = texts
    out["n_image_descriptions"] = [str(c) for c in counts]
    out["image_description_provenance"] = provs
    out["figure_enrichment_status"] = stats
    out["figure_enrichment_detail"] = details
    return out


def integrate(master_dir: Path, enrich_dir: Path, out_dir: Path | None = None):
    out_dir = out_dir or master_dir
    paths = {"families": master_dir / "patent_families_annotated.csv",
             "publications": master_dir / "patent_publications.csv",
             "figures": enrich_dir / "figure_enrichments.csv",
             "resolution": enrich_dir / "document_resolution.csv"}
    fam, pubs = read_str(paths["families"]), read_str(paths["publications"])
    figs, res = read_str(paths["figures"]), read_str(paths["resolution"])
    unknown = (set(figs["family_number"]) | set(res["family_number"])) - set(fam["Family Number"])
    if unknown:
        raise ValueError(f"enrichment families missing from master: {sorted(unknown)[:5]}")
    if figs.duplicated(["family_number", "publication_number", "figure_id"]).any():
        raise ValueError("figure table must hold one row per figure")
    out = build_enrichment(fam, pubs, figs, res)
    assert len(out) == len(fam) and out["Family Number"].is_unique
    out.to_csv(out_dir / "patent_families_enriched.csv", index=False,
               encoding="utf-8-sig", lineterminator="\n")
    summary = {
        "families": len(out),
        "annotated_families": int((out["is_annotated"] == "True").sum()),
        "status_counts_all_families": out["figure_enrichment_status"].value_counts().to_dict(),
        "status_counts_annotated": out.loc[out["is_annotated"] == "True",
                                           "figure_enrichment_status"].value_counts().to_dict(),
        "families_with_usable_description": int((out["n_image_descriptions"] != "0").sum()),
        "usable_descriptions_total": int(out["n_image_descriptions"].astype(int).sum()),
        "input_sha256": {k: sha256(p) for k, p in paths.items()},
        "output_sha256": sha256(out_dir / "patent_families_enriched.csv"),
    }
    (out_dir / "figure_enrichment_integration.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return out, summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--master-dir", default=str(ROOT / "master_dataset"))
    ap.add_argument("--enrich-dir", default=str(ROOT / "data" / "figure_enrichment"))
    a = ap.parse_args(argv)
    _, s = integrate(Path(a.master_dir), Path(a.enrich_dir))
    print(json.dumps({k: v for k, v in s.items() if k != "input_sha256"}, indent=2))


if __name__ == "__main__":
    main()
