"""Organisational inference (spec §8): BASELINE candidate generation + heuristic ranking.

IMPORTANT: patents reveal technical capabilities, not departments or reporting lines.
No labelled organisational structures are available, therefore:
  * nothing here is a trained model; nothing is a probability P(G_c | D_c, E_c);
  * scores are uncalibrated heuristic "compatibility" values, comparable only within
    one company's candidate list;
  * the result may abstain; every hypothesis keeps its derivation + assumptions;
  * each unit is a hypothesis about a *capability cluster*, labelled HYPOTHESIS.

Baseline compatibility score of a candidate partition U of the capability nodes:
    score(U) = Q(U) + lambda_ext * ext(U)
  Q(U)   = sum_{u in U} [ L_u/m - (d_u/(2m))^2 ]   Newman modularity on the undirected
           graph with edge weight w_ij = (#patents where i,j co-occur) + (#patents
           supporting a relation between i and j); L_u = in-unit weight, d_u = sum of
           node degrees in u, m = total weight.  Q(single unit) = 0.
  ext(U) = fraction of independent external assertions ("same_unit"/"different_unit")
           satisfied by U (0 if none are applicable).
Abstention when: too few patents, < 2 nodes, zero graph weight, < 2 distinct
candidates, or the top-two score margin < min_margin (a user-set, unvalidated threshold).

Data required for a trainable probabilistic model (NOT available here): company-level
reference organisational graphs (units, membership, hierarchy) from verifiable sources,
covering enough companies for train/validation/test splits by company.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Protocol, Sequence, Tuple

from .graph import CapabilityGraph

STATUS = "heuristic_uncalibrated_not_a_probability"


@dataclass(frozen=True)
class ExternalEvidence:
    """Independent organisational evidence from E_c (e.g. an org page, job ads)."""
    source: str
    kind: str  # "same_unit" | "different_unit"
    concepts: Tuple[str, str]

    def __post_init__(self):
        if self.kind not in ("same_unit", "different_unit"):
            raise ValueError("kind must be same_unit or different_unit")
        if len(self.concepts) != 2:
            raise ValueError("concepts must be a pair")


@dataclass(frozen=True)
class OrgHypothesis:
    name: str
    units: Tuple[FrozenSet[str], ...]  # partition of capability nodes
    derived_from: str
    assumptions: Tuple[str, ...]
    evidence_type: str = "capability_hypothesis"


@dataclass(frozen=True)
class ScoredHypothesis:
    hypothesis: OrgHypothesis
    modularity: float
    external_support: Optional[float]
    score: float


@dataclass(frozen=True)
class OrgInferenceResult:
    company: str
    ranked: Tuple[ScoredHypothesis, ...]
    abstained: bool
    reason: Optional[str]
    margin: Optional[float]
    status: str = STATUS

    @property
    def best(self) -> Optional[ScoredHypothesis]:
        return None if self.abstained or not self.ranked else self.ranked[0]


class OrgPosteriorModel(Protocol):
    """Interface for a future trainable model of P(G_c | D_c, E_c). Needs reference graphs."""
    def fit(self, graphs: Sequence[CapabilityGraph], reference_structures: Sequence[object]) -> None: ...
    def log_posterior(self, graph: CapabilityGraph, hypothesis: OrgHypothesis) -> float: ...


def edge_weights(g: CapabilityGraph) -> Dict[Tuple[str, str], float]:
    w: Dict[Tuple[str, str], float] = defaultdict(float)
    for k, v in g.cooccurrence().items():
        w[k] += v
    for (h, _r, t), e in g.edges.items():
        if h != t:
            w[tuple(sorted((h, t)))] += len(e.patent_ids)
    return dict(w)


def modularity(units: Sequence[FrozenSet[str]], w: Dict[Tuple[str, str], float]) -> float:
    m = sum(w.values())
    if m <= 0:
        raise ValueError("graph has no edge weight")
    unit_of = {n: i for i, u in enumerate(units) for n in u}
    deg: Dict[str, float] = defaultdict(float)
    inner = [0.0] * len(units)
    for (a, b), x in w.items():
        deg[a] += x
        deg[b] += x
        if a in unit_of and unit_of[a] == unit_of.get(b):
            inner[unit_of[a]] += x
    q = 0.0
    for i, u in enumerate(units):
        d = sum(deg[n] for n in u)
        q += inner[i] / m - (d / (2 * m)) ** 2
    return q


def _components(nodes: Sequence[str], w) -> List[FrozenSet[str]]:
    adj = defaultdict(set)
    for (a, b) in w:
        adj[a].add(b)
        adj[b].add(a)
    seen, out = set(), []
    for n in sorted(nodes):
        if n in seen:
            continue
        stack, comp = [n], set()
        while stack:
            x = stack.pop()
            if x in comp:
                continue
            comp.add(x)
            stack.extend(adj[x] - comp)
        seen |= comp
        out.append(frozenset(comp))
    return out


def generate_candidates(g: CapabilityGraph) -> List[OrgHypothesis]:
    """Baseline candidate generation (no learning). Duplicated partitions are merged."""
    nodes = sorted(g.nodes)
    w = edge_weights(g)
    cands = [
        OrgHypothesis("single_unit", (frozenset(nodes),), "all capabilities in one unit",
                      ("no internal differentiation",)),
        OrgHypothesis("connected_components", tuple(_components(nodes, w)),
                      "connected components of co-occurrence/relation graph",
                      ("capabilities that never co-occur belong to different units",
                       "co-occurrence in a patent implies a shared unit (often false: "
                       "cross-functional inventions)")),
    ]
    cat: Dict[str, set] = defaultdict(set)
    for n in nodes:
        cc = g.nodes[n].category_counts
        cat[max(sorted(cc), key=lambda k: cc[k]) if cc else "unassigned"].add(n)
    if len(cat) > 1 or "unassigned" not in cat:
        cands.append(OrgHypothesis("by_dominant_mpri_category",
                                   tuple(frozenset(v) for _, v in sorted(cat.items())),
                                   "group by most frequent predicted M/P/R/I category",
                                   ("organisation mirrors M/P/R/I function types (strong, unverified)",)))
    seen, uniq = set(), []
    for h in cands:
        key = frozenset(h.units)
        if key not in seen:
            seen.add(key)
            uniq.append(h)
    return uniq


def _external_support(h: OrgHypothesis, ext: Sequence[ExternalEvidence]) -> Optional[float]:
    unit_of = {n: i for i, u in enumerate(h.units) for n in u}
    applicable = [e for e in ext if all(c in unit_of for c in e.concepts)]
    if not applicable:
        return None
    ok = sum((unit_of[e.concepts[0]] == unit_of[e.concepts[1]]) == (e.kind == "same_unit")
             for e in applicable)
    return ok / len(applicable)


def infer_organization(g: CapabilityGraph, external: Sequence[ExternalEvidence] = (),
                       lambda_ext: float = 1.0, min_patents: int = 3,
                       min_margin: float = 0.05) -> OrgInferenceResult:
    def abstain(reason, ranked=(), margin=None):
        return OrgInferenceResult(g.company, tuple(ranked), True, reason, margin)

    if len(g.patent_ids) < min_patents:
        return abstain(f"insufficient evidence: {len(g.patent_ids)} patents < {min_patents}")
    if len(g.nodes) < 2:
        return abstain("fewer than 2 capability nodes")
    w = edge_weights(g)
    if not w:
        return abstain("no co-occurrence or relation evidence between capabilities")
    scored = []
    for h in generate_candidates(g):
        ext = _external_support(h, external)
        q = modularity(h.units, w)
        scored.append(ScoredHypothesis(h, q, ext, q + lambda_ext * (ext or 0.0)))
    scored.sort(key=lambda s: (-s.score, s.hypothesis.name))
    if len(scored) < 2:
        return abstain("only one distinct candidate structure; alternatives cannot be compared", scored)
    margin = scored[0].score - scored[1].score
    if margin < min_margin:
        return abstain(f"top-two margin {margin:.4f} < min_margin {min_margin}: evidence "
                       "cannot distinguish competing hypotheses", scored, margin)
    return OrgInferenceResult(g.company, tuple(scored), False, None, margin)
