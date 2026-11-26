"""Deterministic, group-aware, label-stratified repeated splits.

For repeat r (seed s_r): StratifiedGroupKFold(5, shuffle=True, random_state=s_r) over the
eligible families; fold 0 = test (~20%), fold 1 = validation (~20%), folds 2-4 = train (~60%).
Groups never straddle splits. The stratification key is the Y/N label combination
(`Y`/`N` per role, unknown as `U`), with combinations rarer than `min_count` pooled into 'rare'.
The assignment is computed ONCE on all eligible families (experiment A); the image-subset
experiments B and C reuse the identical assignment restricted to their families, so
A, B and C share split identities and seeds.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .data import LABEL_COLUMNS


def strat_key(df: pd.DataFrame, min_count: int = 10) -> np.ndarray:
    key = df[LABEL_COLUMNS].apply(lambda r: "".join(r), axis=1)
    counts = key.value_counts()
    return np.where(key.map(counts) >= min_count, key, "rare")


def make_assignments(df: pd.DataFrame, groups: Dict[str, str], seeds: Sequence[int],
                     n_folds: int = 5) -> pd.DataFrame:
    """Rows: repeat, seed, family, split in {train, val, test}. df = eligible families."""
    fam = df["Family Number"].to_numpy()
    y = strat_key(df)
    g = np.array([groups[f] for f in fam])
    rows: List[pd.DataFrame] = []
    for r, seed in enumerate(seeds):
        fold = np.full(len(fam), -1)
        sgk = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=int(seed))
        for k, (_, te) in enumerate(sgk.split(fam, y, g)):
            fold[te] = k
        split = np.where(fold == 0, "test", np.where(fold == 1, "val", "train"))
        rows.append(pd.DataFrame({"repeat": r, "seed": int(seed), "family": fam, "split": split}))
    out = pd.concat(rows, ignore_index=True)
    # group integrity
    gm = dict(zip(fam, g))
    out["group"] = out["family"].map(gm)
    bad = out.groupby(["repeat", "group"])["split"].nunique().max()
    if bad > 1:
        raise AssertionError("a group straddles splits")
    return out.drop(columns="group")
