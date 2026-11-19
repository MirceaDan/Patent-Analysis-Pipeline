"""Evaluation interfaces (spec §9). Metrics only; no results are reported or hard-coded.
Run them on real labelled data when available."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Hashable, Iterable, List, Optional, Sequence, Set, Tuple

from .schema import LABELS


def confusion(y_true: Sequence[str], y_pred: Sequence[Optional[str]]) -> Dict[str, Dict[Optional[str], int]]:
    if len(y_true) != len(y_pred):
        raise ValueError("length mismatch")
    m: Dict[str, Dict[Optional[str], int]] = {k: defaultdict(int) for k in LABELS}
    for t, p in zip(y_true, y_pred):
        m[t][p] += 1  # p may be None (abstention): counts as an error
    return m


def per_class_prf(y_true, y_pred) -> Dict[str, Tuple[float, float, float]]:
    out = {}
    for k in LABELS:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == k and p == k)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != k and p == k)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == k and p != k)
        out[k] = prf(tp, fp, fn)
    return out


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def macro_f1(y_true, y_pred) -> float:
    """Unweighted mean F1 over classes present in y_true or y_pred."""
    present = [k for k in LABELS if k in set(y_true) | set(y_pred)]
    if not present:
        raise ValueError("empty input")
    f = per_class_prf(y_true, y_pred)
    return sum(f[k][2] for k in present) / len(present)


def balanced_accuracy(y_true, y_pred) -> float:
    """Mean recall over classes present in y_true."""
    present = [k for k in LABELS if k in set(y_true)]
    if not present:
        raise ValueError("empty input")
    f = per_class_prf(y_true, y_pred)
    return sum(f[k][1] for k in present) / len(present)


def expected_calibration_error(probs: Sequence[Dict[str, float]], y_true: Sequence[str], n_bins: int = 10) -> float:
    """ECE over top-1 confidence."""
    if not probs:
        raise ValueError("empty input")
    bins: List[List[Tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for p, t in zip(probs, y_true):
        k = max(LABELS, key=lambda c: (p[c], -LABELS.index(c)))
        conf = p[k]
        bins[min(int(conf * n_bins), n_bins - 1)].append((conf, k == t))
    n = len(probs)
    return sum(len(b) / n * abs(sum(c for c, _ in b) / len(b) - sum(o for _, o in b) / len(b))
               for b in bins if b)


def set_prf(predicted: Iterable[Hashable], reference: Iterable[Hashable]) -> Tuple[float, float, float]:
    """P/R/F1 between two sets: use for spans (patent_id,modality,start,end), relations
    (head,rel,tail), graph nodes, and graph edges."""
    P, R = set(predicted), set(reference)
    return prf(len(P & R), len(P - R), len(R - P))


def partition_pair_f1(pred_units: Sequence[Set[str]], ref_units: Sequence[Set[str]]) -> Tuple[float, float, float]:
    """Org-structure quality vs a reference: pairs of nodes placed in the same unit."""
    def pairs(units):
        s = set()
        for u in units:
            u = sorted(u)
            s |= {(u[i], u[j]) for i in range(len(u)) for j in range(i + 1, len(u))}
        return s
    return set_prf(pairs(pred_units), pairs(ref_units))


def group_split(ids: Sequence[str], groups: Sequence[str], test_fraction: float, seed: int = 0
                ) -> Tuple[List[str], List[str]]:
    """Deterministic split keeping every group (patent family or company) on one side."""
    import random
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction in (0,1)")
    uniq = sorted(set(groups))
    random.Random(seed).shuffle(uniq)
    test_g = set(uniq[: max(1, round(len(uniq) * test_fraction))]) if len(uniq) > 1 else set()
    tr = [i for i, g in zip(ids, groups) if g not in test_g]
    te = [i for i, g in zip(ids, groups) if g in test_g]
    return tr, te
