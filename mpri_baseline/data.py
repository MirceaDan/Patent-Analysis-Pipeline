"""Dataset loading, label masking, eligibility and text features.

Target formulation (verified in docs/TRAINING_DATASET_AUDIT.md): four INDEPENDENT binary
labels, y_k in {1 (Y), 0 (N), unknown (U or empty)}; mask m_k = 1 iff y_k is known.
Unknown is never turned into 0. Rows with no known label carry no supervision and are
ineligible. There is no argmax / single-class formulation.

Classifier inputs are limited to FEATURE_COLUMNS. Everything else (assignee fields,
all `adm_*` sampling/model-score columns, `ann_*` coder fields, claims, citations,
identifiers) is explicitly forbidden as a feature; `assert_no_forbidden_features`
enforces it.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ENRICHED = ROOT / "master_dataset" / "patent_families_enriched.csv"
PUBLICATIONS = ROOT / "master_dataset" / "patent_publications.csv"

LABEL_COLUMNS = ["F_MONITOR_REPRESENT", "F_PREDICT_SIMULATE", "F_RECOMMEND_DECIDE",
                 "F_INTERVENE_CONTROL"]
LABEL_ALIASES = ["label_M", "label_P", "label_R", "label_I"]
LABEL_NAMES = ["M", "P", "R", "I"]
FEATURE_COLUMNS = ["Title", "Abstract", "image_descriptions"]
FORBIDDEN_PREFIXES = ("adm_", "ann_")
FORBIDDEN_COLUMNS = {"Assignees", "Non Std Assignees", "Probable Patent Assignee", "Inventors",
                     "Non Std Inventors", "Claims", "1st Main Claim", "Backward Citations",
                     "Forward Citations", "Family Number", "Patent Number",
                     "Priority Numbers", "Application Number"} | set(LABEL_COLUMNS) | set(LABEL_ALIASES)
# Used ONLY to build leakage-control groups, never as features.
GROUPING_COLUMNS = ["Priority Numbers", "Application Number", "Backward Citations",
                    "Forward Citations", "Probable Patent Assignee"]
LOAD_COLUMNS = (["Family Number", "is_annotated", "figure_enrichment_status",
                 "n_image_descriptions"] + FEATURE_COLUMNS + LABEL_COLUMNS + LABEL_ALIASES
                + GROUPING_COLUMNS)

csv.field_size_limit(1 << 28)


def assert_no_forbidden_features(columns: Sequence[str]) -> None:
    bad = [c for c in columns if c in FORBIDDEN_COLUMNS or c.startswith(FORBIDDEN_PREFIXES)]
    if bad:
        raise ValueError(f"forbidden classifier feature columns: {bad}")
    unknown = [c for c in columns if c not in FEATURE_COLUMNS]
    if unknown:
        raise ValueError(f"columns outside the allowed feature set: {unknown}")


def load_annotated(path: Path = ENRICHED) -> pd.DataFrame:
    """Annotated families only, strings, index = row order of the source file."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False,
                     encoding="utf-8-sig", usecols=LOAD_COLUMNS)
    df = df[df["is_annotated"] == "True"].reset_index(drop=True)
    if not df["Family Number"].is_unique:
        raise ValueError("duplicate Family Number in enriched dataset")
    return df


def label_matrix(df: pd.DataFrame):
    """(Y, M): Y in {0,1} float, M mask bool (True = known). Unknown entries have Y=0 but M=False."""
    raw = df[LABEL_COLUMNS].to_numpy()
    if not (np.isin(raw, ["Y", "N", "U", ""]).all()):
        raise ValueError("labels must be in {Y, N, U, ''}")
    known = np.isin(raw, ["Y", "N"])
    y = (raw == "Y").astype(float)
    return y, known


def alias_consistent(df: pd.DataFrame) -> bool:
    return all((df[a] == df[c]).all() for a, c in zip(LABEL_ALIASES, LABEL_COLUMNS))


def eligibility(df: pd.DataFrame) -> pd.DataFrame:
    """Boolean columns: eligible_A (all eligible annotated families) and eligible_BC
    (A-eligible AND >=1 usable image description). Reasons are returned for the audit."""
    _, known = label_matrix(df)
    has_title = df["Title"].str.strip() != ""
    has_abs = df["Abstract"].str.strip() != ""
    any_known = known.any(axis=1)
    has_desc = df["n_image_descriptions"].astype(int) > 0
    out = pd.DataFrame({"family": df["Family Number"], "has_title": has_title,
                        "has_abstract": has_abs, "any_known_label": any_known,
                        "has_image_description": has_desc})
    out["eligible_A"] = has_title & has_abs & any_known
    out["eligible_BC"] = out["eligible_A"] & has_desc
    return out


def build_text(df: pd.DataFrame, use_images: bool) -> List[str]:
    """Title + abstract [+ image descriptions]. Empty descriptions simply add nothing."""
    cols = FEATURE_COLUMNS if use_images else FEATURE_COLUMNS[:2]
    assert_no_forbidden_features(cols)
    parts = [df[c].str.strip() for c in cols]
    return [" . ".join(p for p in row if p) for row in zip(*parts)]
