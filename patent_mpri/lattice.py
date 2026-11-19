"""Formal context and CLASSICAL binary Formal Concept Analysis (spec §5.1).

K = (P, A, R);  B_pa = 1[(p,a) in R]
X' = {a : forall p in X, (p,a) in R}      (common attributes)
Y' = {p : forall a in Y, (p,a) in R}      (common objects)
Concept: (X, Y) with X' = Y and Y' = X.  Order: (X1,Y1) <= (X2,Y2) iff X1 subset X2.

Everything here is crisp/binary. Graded variants live in scoring.py and are
explicitly labelled extensions.  Implementation: bitmask sets + Ganter's NextClosure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Iterable, List, Sequence, Tuple

from .schema import Annotation


@dataclass(frozen=True)
class FormalConcept:
    extent: FrozenSet[str]  # X
    intent: FrozenSet[str]  # Y


class FormalContext:
    """K=(P,A,R). Objects/attributes are strings; incidence is a set of pairs."""

    def __init__(self, objects: Iterable[str], attributes: Iterable[str],
                 incidence: Iterable[Tuple[str, str]]):
        self.objects: Tuple[str, ...] = tuple(sorted(set(objects)))
        self.attributes: Tuple[str, ...] = tuple(sorted(set(attributes)))
        self._oi = {o: i for i, o in enumerate(self.objects)}
        self._ai = {a: i for i, a in enumerate(self.attributes)}
        self._obj_attrs = [0] * len(self.objects)  # per object: bitmask over attributes
        self._attr_objs = [0] * len(self.attributes)  # per attribute: bitmask over objects
        for o, a in incidence:
            if o not in self._oi or a not in self._ai:
                raise ValueError(f"incidence pair ({o!r},{a!r}) outside P x A")
            self._obj_attrs[self._oi[o]] |= 1 << self._ai[a]
            self._attr_objs[self._ai[a]] |= 1 << self._oi[o]

    @classmethod
    def from_annotations(cls, annotations: Sequence[Annotation],
                         min_confidence: float = 0.0) -> "FormalContext":
        """R = {(p,a): some cue of p with attribute a has confidence >= min_confidence}.
        The threshold is a design choice the spec leaves open (default keeps all cues)."""
        objs, attrs, inc = [], set(), []
        for ann in annotations:
            objs.append(ann.patent_id)
            for c in ann.cues:
                if c.confidence >= min_confidence:
                    attrs.add(c.attribute)
                    inc.append((ann.patent_id, c.attribute))
        return cls(objs, attrs, inc)

    def holds(self, p: str, a: str) -> bool:
        return bool(self._obj_attrs[self._oi[p]] >> self._ai[a] & 1)

    def attributes_of(self, p: str) -> FrozenSet[str]:
        return self._decode_attrs(self._obj_attrs[self._oi[p]])

    def _extent_mask(self, y_mask: int) -> int:  # Y'
        m = (1 << len(self.objects)) - 1
        for i in range(len(self.attributes)):
            if y_mask >> i & 1:
                m &= self._attr_objs[i]
        return m

    def _intent_mask(self, x_mask: int) -> int:  # X'
        m = (1 << len(self.attributes)) - 1
        for i in range(len(self.objects)):
            if x_mask >> i & 1:
                m &= self._obj_attrs[i]
        return m

    def _decode_attrs(self, m: int) -> FrozenSet[str]:
        return frozenset(a for i, a in enumerate(self.attributes) if m >> i & 1)

    def _decode_objs(self, m: int) -> FrozenSet[str]:
        return frozenset(o for i, o in enumerate(self.objects) if m >> i & 1)

    def common_attributes(self, X: Iterable[str]) -> FrozenSet[str]:
        """X'."""
        m = 0
        for p in X:
            m |= 1 << self._oi[p]
        return self._decode_attrs(self._intent_mask(m))

    def common_objects(self, Y: Iterable[str]) -> FrozenSet[str]:
        """Y'."""
        m = 0
        for a in Y:
            m |= 1 << self._ai[a]
        return self._decode_objs(self._extent_mask(m))

    def concepts(self) -> List[FormalConcept]:
        """All formal concepts via NextClosure over intents (closure Y -> Y'')."""
        n = len(self.attributes)
        full = (1 << n) - 1
        closure = lambda m: self._intent_mask(self._extent_mask(m))
        cur = closure(0)
        out = [self._make(cur)]
        while cur != full:
            for i in range(n - 1, -1, -1):
                if cur >> i & 1:
                    continue
                low = (1 << i) - 1
                cand = closure((cur & low) | (1 << i))
                if (cand & low) == (cur & low):  # lectically next closed set
                    cur = cand
                    break
            out.append(self._make(cur))
        return out

    def _make(self, y_mask: int) -> FormalConcept:
        return FormalConcept(self._decode_objs(self._extent_mask(y_mask)), self._decode_attrs(y_mask))


class ConceptLattice:
    """Concepts ordered by extent inclusion, with the Hasse (cover) relation."""

    def __init__(self, context: FormalContext):
        self.context = context
        self.concepts: List[FormalConcept] = sorted(
            context.concepts(), key=lambda c: (-len(c.extent), sorted(c.intent)))
        self._covers = self._compute_covers()

    @staticmethod
    def leq(c1: FormalConcept, c2: FormalConcept) -> bool:
        return c1.extent <= c2.extent

    def _compute_covers(self) -> List[Tuple[int, int]]:
        """(i, j): concept j is a lower cover of concept i."""
        cs, edges = self.concepts, []
        for i, hi in enumerate(cs):
            below = [j for j, lo in enumerate(cs) if lo.extent < hi.extent]
            for j in below:
                if not any(cs[j].extent < cs[k].extent for k in below if k != j):
                    edges.append((i, j))
        return edges

    @property
    def covers(self) -> List[Tuple[int, int]]:
        return list(self._covers)

    def top(self) -> FormalConcept:
        return max(self.concepts, key=lambda c: len(c.extent))

    def bottom(self) -> FormalConcept:
        return min(self.concepts, key=lambda c: len(c.extent))
