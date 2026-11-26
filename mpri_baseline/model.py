"""TF-IDF + one L2 logistic regression per label, trained on KNOWN targets only.

For label k the training set is {i : m_ik = 1}; unknown targets contribute nothing to the
loss (masked), which is exactly the masked binary cross-entropy
    L = - sum_k sum_i m_ik [ y_ik log p_ik + (1 - y_ik) log(1 - p_ik) ] + (1/2C) sum_k ||w_k||^2
because the per-label problems are independent. class_weight='balanced' reweights Y/N within
each label. The vectorizer is fitted on the TRAINING texts only. Regularisation C is chosen on
the validation split by mean average precision over labels having both classes in validation;
if no label qualifies (tiny validation sets) the fixed default C=1.0 is used (documented fallback).
If a label has a single class in training, the prior is predicted for it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .metrics import aggregate

C_GRID = (0.1, 1.0, 10.0, 100.0)
DEFAULT_C = 1.0
TFIDF = dict(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True,
             stop_words="english", lowercase=True)


@dataclass
class MPRIModel:
    C: float
    vectorizer: TfidfVectorizer
    heads: List = field(default_factory=list)  # LogisticRegression | float prior

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        cols = [h.predict_proba(X)[:, 1] if hasattr(h, "predict_proba") else np.full(X.shape[0], h)
                for h in self.heads]
        return np.column_stack(cols)


def _fit_heads(X, y, known, C, seed):
    heads = []
    for k in range(y.shape[1]):
        m = known[:, k]
        yk = y[m, k].astype(int)
        if len(np.unique(yk)) < 2:
            heads.append(float(yk.mean()) if len(yk) else 0.5)
            continue
        clf = LogisticRegression(C=C, class_weight="balanced", solver="liblinear", random_state=seed)
        heads.append(clf.fit(X[m], yk))
    return heads


def fit(train_text, y_tr, known_tr, val_text, y_va, known_va, seed=0, c_grid=C_GRID):
    """Returns (MPRIModel, selection_log). Vectorizer fitted on train_text only."""
    vec = TfidfVectorizer(**TFIDF)
    Xtr = vec.fit_transform(train_text)
    Xva = vec.transform(val_text)
    log, best, best_score = [], None, -np.inf
    for C in c_grid:
        heads = _fit_heads(Xtr, y_tr, known_tr, C, seed)
        m = MPRIModel(C, vec, heads)
        p = np.column_stack([h.predict_proba(Xva)[:, 1] if hasattr(h, "predict_proba")
                             else np.full(Xva.shape[0], h) for h in heads])
        s = aggregate(y_va, known_va, p)["macro_average_precision"]
        log.append({"C": C, "val_macro_average_precision": None if np.isnan(s) else float(s)})
        if not np.isnan(s) and s > best_score:
            best, best_score = m, s
    if best is None:  # validation gave no usable metric
        best = MPRIModel(DEFAULT_C, vec, _fit_heads(Xtr, y_tr, known_tr, DEFAULT_C, seed))
        log.append({"fallback_default_C": DEFAULT_C})
    return best, log
