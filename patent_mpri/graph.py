"""Company-level technical capability graph G_c = (V_c, E_c, X_c) (spec §7).

Nodes = concepts (capability candidates) observed in the company's patents.
Edges = typed relations between concepts, aggregated across patents.
X_c   = node/edge attributes: supporting patents, mean confidence, evidence-type counts,
        and the full list of provenance records.
This graph records OBSERVED technical evidence. It does not assert departments or
reporting lines. Company identity is used only to group patents here; it is never
fed to the M/P/R/I classifier.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from .schema import Annotation, EvidenceRecord, Patent, evidence_from_annotation


@dataclass
class CapabilityNode:
    concept: str
    patent_ids: set = field(default_factory=set)
    evidence: List[EvidenceRecord] = field(default_factory=list)
    category_counts: Counter = field(default_factory=Counter)  # predicted M/P/R/I of supporting patents

    @property
    def mean_confidence(self) -> float:
        return sum(e.score for e in self.evidence) / len(self.evidence) if self.evidence else 0.0

    @property
    def evidence_types(self) -> Counter:
        return Counter(e.evidence_type for e in self.evidence)


@dataclass
class CapabilityEdge:
    head: str
    relation: str
    tail: str
    patent_ids: set = field(default_factory=set)
    uncertainties: List[float] = field(default_factory=list)
    evidence: List[Tuple[EvidenceRecord, EvidenceRecord]] = field(default_factory=list)  # (head,tail) cue evidence

    @property
    def mean_uncertainty(self) -> float:
        return sum(self.uncertainties) / len(self.uncertainties)


@dataclass
class CapabilityGraph:
    company: str
    nodes: Dict[str, CapabilityNode] = field(default_factory=dict)
    edges: Dict[Tuple[str, str, str], CapabilityEdge] = field(default_factory=dict)
    patent_ids: set = field(default_factory=set)
    # concept sets per patent, for co-occurrence statistics
    patent_concepts: Dict[str, FrozenSet[str]] = field(default_factory=dict)

    def cooccurrence(self) -> Dict[Tuple[str, str], int]:
        """Number of patents in which two concepts co-occur (undirected, sorted key)."""
        co: Dict[Tuple[str, str], int] = defaultdict(int)
        for cs in self.patent_concepts.values():
            s = sorted(cs)
            for i in range(len(s)):
                for j in range(i + 1, len(s)):
                    co[(s[i], s[j])] += 1
        return dict(co)


def build_capability_graphs(patents: Sequence[Patent], annotations: Sequence[Annotation],
                            predicted_labels: Optional[Dict[str, Optional[str]]] = None
                            ) -> Tuple[Dict[str, CapabilityGraph], List[str]]:
    """Aggregate patent-level cues/relations into per-company graphs.
    Returns (graphs by company, ids of patents skipped for lacking company metadata)."""
    ann_by_id = {a.patent_id: a for a in annotations}
    graphs: Dict[str, CapabilityGraph] = {}
    skipped: List[str] = []
    for p in patents:
        ann = ann_by_id.get(p.patent_id)
        if ann is None or not p.company:
            skipped.append(p.patent_id)
            continue
        g = graphs.setdefault(p.company, CapabilityGraph(p.company))
        g.patent_ids.add(p.patent_id)
        cue_ev = {c.cue_id: ev for c, ev in zip(ann.cues, evidence_from_annotation(ann))}
        cat = (predicted_labels or {}).get(p.patent_id)
        for c in ann.cues:
            n = g.nodes.setdefault(c.attribute, CapabilityNode(c.attribute))
            n.patent_ids.add(p.patent_id)
            n.evidence.append(cue_ev[c.cue_id])
        for cname in {c.attribute for c in ann.cues}:
            if cat:
                g.nodes[cname].category_counts[cat] += 1
        g.patent_concepts[p.patent_id] = frozenset(c.attribute for c in ann.cues)
        by_id = {c.cue_id: c for c in ann.cues}
        for r in ann.relations:
            h, t = by_id[r.head], by_id[r.tail]
            e = g.edges.setdefault((h.attribute, r.relation, t.attribute),
                                   CapabilityEdge(h.attribute, r.relation, t.attribute))
            e.patent_ids.add(p.patent_id)
            e.uncertainties.append(r.uncertainty)
            e.evidence.append((cue_ev[h.cue_id], cue_ev[t.cue_id]))
    return graphs, skipped
