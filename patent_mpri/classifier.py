"""End-to-end M/P/R/I classifier: annotator -> formal context -> lattice -> rho -> scores.

The classifier only ever sees x_p=(title, abstract, figure descriptions). Company and
label are stripped before annotation (spec §1: c(p) is not a classifier input).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .lattice import ConceptLattice, FormalConcept, FormalContext
from .schema import LABELS, Annotation, Cue, Patent, SchemaError
from .scoring import (EPS_DEFAULT, ConceptScorer, MuFn, binary_containment_mu,
                      graded_mean_mu)


def strip_to_classifier_input(p: Patent) -> Patent:
    """Copy of p without company/label/family (no shortcut features)."""
    t, a, v = p.classifier_input()
    return Patent(p.patent_id, t, a, v)


def attribute_confidences(ann: Annotation, min_confidence: float = 0.0, crisp: bool = True) -> Dict[str, float]:
    """A_p as attribute -> m_p(a). crisp=True gives 1.0 for present attributes."""
    out: Dict[str, float] = {}
    for c in ann.cues:
        if c.confidence >= min_confidence:
            out[c.attribute] = max(out.get(c.attribute, 0.0), 1.0 if crisp else c.confidence)
    return out


@dataclass(frozen=True)
class SupportingConcept:
    intent: Tuple[str, ...]
    delta: float  # Delta_{c,k} for the predicted class
    cues: Tuple[Cue, ...]  # cues of p* that realise the intent (provenance)


@dataclass(frozen=True)
class Prediction:
    patent_id: str
    scores: Dict[str, float]  # s_k
    probabilities: Dict[str, float]  # pi_k (uncalibrated unless T was validated)
    predicted: Optional[str]  # None = abstain (no concept matched)
    supporting: Tuple[SupportingConcept, ...] = ()


class MPRIClassifier:
    def __init__(self, annotator, graded: bool = False, min_confidence: float = 0.0,
                 temperature: float = 1.0, eps: float = EPS_DEFAULT, top_support: int = 5):
        """graded=False: classical binary FCA + containment matching.
        graded=True: still classical FCA lattice, but EXTENSION matching mu (graded_mean_mu)."""
        self.annotator, self.graded = annotator, graded
        self.min_confidence, self.temperature, self.eps = min_confidence, temperature, eps
        self.top_support = top_support
        self.mu: MuFn = graded_mean_mu if graded else binary_containment_mu
        self.lattice: Optional[ConceptLattice] = None
        self.scorer: Optional[ConceptScorer] = None
        self.train_annotations: List[Annotation] = []

    def fit(self, patents: Sequence[Patent]) -> "MPRIClassifier":
        if not patents:
            raise ValueError("cannot fit on an empty patent collection")
        if any(p.label is None for p in patents):
            raise SchemaError("all training patents need a label y_p")
        if len({p.patent_id for p in patents}) != len(patents):
            raise SchemaError("duplicate patent_id in training set")
        anns = []
        for p in patents:
            a = self.annotator.annotate(strip_to_classifier_input(p))
            a.validate_against(p)
            anns.append(a)
        self.train_annotations = anns
        ctx = FormalContext.from_annotations(anns, self.min_confidence)
        self.lattice = ConceptLattice(ctx)
        attrs = {a.patent_id: attribute_confidences(a, self.min_confidence, crisp=not self.graded)
                 for a in anns}
        labels = {p.patent_id: p.label for p in patents}
        self.scorer = ConceptScorer.fit(self.lattice.concepts, attrs, labels, self.mu,
                                        temperature=self.temperature, eps=self.eps)
        return self

    def predict(self, patent: Patent, annotation: Optional[Annotation] = None) -> Prediction:
        if self.scorer is None:
            raise RuntimeError("classifier not fitted")
        clean = strip_to_classifier_input(patent)
        ann = annotation or self.annotator.annotate(clean)
        ann.validate_against(clean)
        attrs = attribute_confidences(ann, self.min_confidence, crisp=not self.graded)
        s = self.scorer
        pred = s.predict(attrs)
        sup: Tuple[SupportingConcept, ...] = ()
        if pred is not None:
            ds = s.deltas(attrs)
            order = sorted(range(len(ds)), key=lambda i: -ds[i][pred])
            items = []
            for i in order[: self.top_support]:
                if ds[i][pred] <= 0:
                    break
                intent = tuple(sorted(s.concepts[i].intent))
                cues = tuple(c for c in ann.cues if c.attribute in s.concepts[i].intent)
                items.append(SupportingConcept(intent, ds[i][pred], cues))
            sup = tuple(items)
        return Prediction(patent.patent_id, s.scores(attrs), s.probabilities(attrs), pred, sup)

    def shared_concepts(self, min_categories: int = 2, min_rho: float = 0.0) -> List[Tuple[FormalConcept, List[str]]]:
        """Concepts with rho(c,k) > min_rho for >= min_categories categories (overlap retained)."""
        out = []
        for c, r in zip(self.scorer.concepts, self.scorer.rho):
            ks = [k for k in LABELS if r[k] > min_rho]
            if len(ks) >= min_categories:
                out.append((c, ks))
        return out
