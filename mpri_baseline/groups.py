"""Related-family grouping used to keep related patents on the same side of a split.

Two annotated families are linked (union-find) if ANY of:
  priority  - share a priority number token (`Priority Numbers`, ';'-separated)
  appl      - share an application number token
  citation  - (only if use_citations) a backward/forward citation of one equals a publication
              number of the other (kind code ignored). Links are always COUNTED, but they
              chain 155 of the 600 annotated families into one component, so they are
              unioned only for the strict sensitivity grouping.
  textdup   - TF-IDF cosine(title+abstract) >= `text_threshold` (design choice, default 0.8)
  assignee  - (optional, off by default) same `Probable Patent Assignee`
A group is a connected component. Limitations: PatBase `Family Number` is already a
family-level key; links outside the 600 annotated families are not visible; the text
threshold is not validated against ground truth.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .data import PUBLICATIONS


def _tokens(s: str):
    return {t.strip() for t in s.split(";") if t.strip()}


def _cited_numbers(s: str):
    out = set()
    for t in s.split(";"):
        t = t.strip().split(" ")[0] if t.strip() else ""
        if t:
            out.add(re.sub(r"\s+", "", t))
    return out


class _UF:
    def __init__(self, items):
        self.p = {i: i for i in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def related_family_groups(df: pd.DataFrame, publications: Optional[pd.DataFrame] = None,
                          text_threshold: float = 0.8, use_assignee: bool = False,
                          use_citations: bool = False,
                          return_links: bool = False):
    """Return {family: group_id} (group id = smallest member family string)."""
    fams = list(df["Family Number"])
    uf = _UF(fams)
    links = defaultdict(set)

    def add(kind, a, b, union=True):
        if a != b:
            links[kind].add(tuple(sorted((a, b))))
            if union:
                uf.union(a, b)

    for col, kind in (("Priority Numbers", "priority"), ("Application Number", "appl")):
        owner = defaultdict(list)
        for f, v in zip(fams, df[col]):
            for t in _tokens(v):
                owner[t].append(f)
        for fl in owner.values():
            for g in fl[1:]:
                add(kind, fl[0], g)

    if publications is None:
        publications = pd.read_csv(PUBLICATIONS, dtype=str, keep_default_na=False,
                                   usecols=["family_number", "publication_number"])
    famset = set(fams)
    pub_owner = {p: f for f, p in zip(publications["family_number"],
                                      publications["publication_number"]) if f in famset}
    for f, b, fw in zip(fams, df["Backward Citations"], df["Forward Citations"]):
        for n in _cited_numbers(b) | _cited_numbers(fw):
            o = pub_owner.get(n)
            if o:
                add("citation", f, o, union=use_citations)

    text = (df["Title"] + " . " + df["Abstract"]).tolist()
    X = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(text)
    sim = (X @ X.T).toarray()
    np.fill_diagonal(sim, 0.0)
    for i, j in zip(*np.where(np.triu(sim) >= text_threshold)):
        add("textdup", fams[i], fams[j])

    if use_assignee:
        owner = defaultdict(list)
        for f, a in zip(fams, df["Probable Patent Assignee"]):
            if a.strip():
                owner[a.strip().upper()].append(f)
        for fl in owner.values():
            for g in fl[1:]:
                add("assignee", fl[0], g)

    groups: Dict[str, str] = {f: uf.find(f) for f in fams}
    return (groups, {k: sorted(v) for k, v in links.items()}) if return_links else groups
