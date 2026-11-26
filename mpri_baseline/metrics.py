"""Masked multi-label metrics. y: (n,4) in {0,1}; known: (n,4) bool; p: (n,4) probabilities.
Entries with known=False are excluded from every metric (never counted as negatives).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .data import LABEL_NAMES

NAN = float("nan")


def per_label(y, known, p, threshold=0.5):
    out = {}
    for k, name in enumerate(LABEL_NAMES):
        m = known[:, k]
        yt, pk = y[m, k].astype(int), p[m, k]
        pos, neg = int(yt.sum()), int((1 - yt).sum())
        pred = (pk >= threshold).astype(int)
        tp = int(((pred == 1) & (yt == 1)).sum())
        fp = int(((pred == 1) & (yt == 0)).sum())
        fn = int(((pred == 0) & (yt == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else NAN
        rec = tp / (tp + fn) if tp + fn else NAN
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else NAN
        out[name] = {"n_known": int(m.sum()), "n_unknown_excluded": int((~m).sum()),
                     "support_pos": pos, "support_neg": neg,
                     "prevalence": pos / max(pos + neg, 1) if pos + neg else NAN,
                     "precision": prec, "recall": rec, "f1": f1,
                     "average_precision": float(average_precision_score(yt, pk)) if pos and neg else NAN,
                     "roc_auc": float(roc_auc_score(yt, pk)) if pos and neg else NAN}
    return out


def aggregate(y, known, p, threshold=0.5):
    pl = per_label(y, known, p, threshold)
    nanmean = lambda key: (float(np.nanmean([v[key] for v in pl.values()]))
                           if any(not np.isnan(v[key]) for v in pl.values()) else NAN)
    pred = (p >= threshold)
    tp = int(((pred & (y == 1)) & known).sum())
    fp = int(((pred & (y == 0)) & known).sum())
    fn = int(((~pred & (y == 1)) & known).sum())
    wrong = int(((pred != (y == 1)) & known).sum())
    return {"macro_f1": nanmean("f1"), "macro_average_precision": nanmean("average_precision"),
            "macro_roc_auc": nanmean("roc_auc"),
            "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else NAN,
            "hamming_loss_known_only": wrong / max(int(known.sum()), 1), "per_label": pl}
