"""Concept-category relevance and patent scoring (spec §5.2, §6, §7).

Symbols
  C            set of formal concepts (from lattice.ConceptLattice)
  mu(p,c) in [0,1]   concept-to-patent matching weight  (spec leaves it undefined; see below)
  rho(c,k)     = sum_p 1[y_p=k] mu(p,c) / (sum_p mu(p,c) + eps)
  w(c) >= 0    concept weight (spec: "learned or experimentally validated"; default 1)
  s_k(p*)      = sum_c mu(p*,c) w(c) rho(c,k) / (sum_c mu(p*,c) w(c) + eps)
  pi_k(p*)     = softmax_k(s_k / T)
  Delta_{c,k}  = mu(p*,c) w(c) rho(c,k)

Matching weights (design decision, the spec only says "measures the association"):
  * binary_containment_mu  -- CLASSICAL: mu = 1[Y subseteq A_p]  (p lies in c's extent
    when A_p is p's crisp attribute set). Consistent with binary FCA.
  * graded_mean_mu         -- EXTENSION (NOT classical FCA): mu = mean_{a in Y} m_p(a),
    where m_p(a) in [0,1] is the confidence of the best cue for a (0 if absent).
    Reduces to the binary version when all m_p(a) in {0,1}. A concept with empty
    intent (the top concept) has mu = 1 in both.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Mapping, Optional, Sequence

from .lattice import FormalConcept
from .schema import LABELS

AttrConf = Mapping[str, float]  # attribute -> confidence in [0,1] (crisp set: all 1.0)
MuFn = Callable[[AttrConf, FormalConcept], float]

EPS_DEFAULT = 1e-9  # spec: "epsilon > 0", value unspecified


def binary_containment_mu(attrs: AttrConf, concept: FormalConcept) -> float:
    """Classical: 1 iff every intent attribute is present in the patent."""
    return 1.0 if all(a in attrs and attrs[a] > 0 for a in concept.intent) else 0.0


def graded_mean_mu(attrs: AttrConf, concept: FormalConcept) -> float:
    """EXTENSION (not classical FCA): mean attribute confidence over the intent."""
    if not concept.intent:
        return 1.0
    return sum(min(1.0, max(0.0, attrs.get(a, 0.0))) for a in concept.intent) / len(concept.intent)


def stable_softmax(values: Sequence[float], temperature: float = 1.0) -> list:
    """softmax(v/T) with max-subtraction for numerical stability."""
    if not (temperature > 0 and math.isfinite(temperature)):
        raise ValueError("temperature must be a finite positive number")
    if not values:
        raise ValueError("empty input")
    z = [v / temperature for v in values]
    m = max(z)
    e = [math.exp(x - m) for x in z]
    t = sum(e)
    return [x / t for x in e]


def compute_rho(concepts: Sequence[FormalConcept],
                train_attrs: Mapping[str, AttrConf],
                train_labels: Mapping[str, str],
                mu: MuFn = binary_containment_mu,
                eps: float = EPS_DEFAULT) -> list:
    """rho[i][k] for concept index i and category k. Overlap across categories is kept."""
    if eps <= 0:
        raise ValueError("eps must be > 0")
    rho = []
    for c in concepts:
        num = {k: 0.0 for k in LABELS}
        den = 0.0
        for pid, attrs in train_attrs.items():
            m = mu(attrs, c)
            den += m
            num[train_labels[pid]] += m
        rho.append({k: num[k] / (den + eps) for k in LABELS})
    return rho


class ConceptScorer:
    """Fitted scoring model: concepts, rho, w(c), T, eps."""

    def __init__(self, concepts: Sequence[FormalConcept], rho: Sequence[Mapping[str, float]],
                 mu: MuFn = binary_containment_mu,
                 weights: Optional[Sequence[float]] = None,
                 temperature: float = 1.0, eps: float = EPS_DEFAULT):
        if len(concepts) != len(rho):
            raise ValueError("concepts and rho differ in length")
        if weights is None:
            weights = [1.0] * len(concepts)
        if len(weights) != len(concepts) or any((not math.isfinite(w)) or w < 0 for w in weights):
            raise ValueError("weights must be finite, non-negative, one per concept")
        if eps <= 0 or not (temperature > 0):
            raise ValueError("eps>0 and temperature>0 required")
        self.concepts, self.rho, self.mu = list(concepts), list(rho), mu
        self.weights, self.temperature, self.eps = list(weights), temperature, eps

    @classmethod
    def fit(cls, concepts, train_attrs, train_labels, mu=binary_containment_mu, **kw):
        """Estimate rho from labelled training patents. w and T are NOT learned here
        (the spec gives no objective for them); pass them via kw."""
        eps = kw.get("eps", EPS_DEFAULT)
        return cls(concepts, compute_rho(concepts, train_attrs, train_labels, mu, eps), mu, **kw)

    def deltas(self, attrs: AttrConf) -> list:
        """Delta[i][k] = mu(p*,c_i) w(c_i) rho(c_i,k)."""
        out = []
        for c, r, w in zip(self.concepts, self.rho, self.weights):
            m = self.mu(attrs, c) * w
            out.append({k: m * r[k] for k in LABELS})
        return out

    def scores(self, attrs: AttrConf) -> Dict[str, float]:
        """s_k(p*). Denominator mass is returned via match_mass()."""
        d = self.match_mass(attrs)
        dl = self.deltas(attrs)
        return {k: sum(x[k] for x in dl) / (d + self.eps) for k in LABELS}

    def match_mass(self, attrs: AttrConf) -> float:
        return sum(self.mu(attrs, c) * w for c, w in zip(self.concepts, self.weights))

    def probabilities(self, attrs: AttrConf) -> Dict[str, float]:
        """pi_k = softmax(s_k / T). T is a calibration temperature; no calibration is
        performed here, so these are 'pi' in the spec's notation but NOT validated-calibrated."""
        s = self.scores(attrs)
        p = stable_softmax([s[k] for k in LABELS], self.temperature)
        return dict(zip(LABELS, p))

    def predict(self, attrs: AttrConf) -> Optional[str]:
        """argmax_k pi_k; None (abstain) if no concept matches at all (mass == 0).
        Ties resolve by the fixed order M,P,R,I for determinism."""
        if self.match_mass(attrs) <= 0:
            return None
        p = self.probabilities(attrs)
        return max(LABELS, key=lambda k: (p[k], -LABELS.index(k)))
