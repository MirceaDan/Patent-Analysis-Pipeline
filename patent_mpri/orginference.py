"""Organisational inference (spec §8): BASELINE candidate generation + heuristic ranking.

IMPORTANT: patents reveal technical capabilities, not departments or reporting lines.
No labelled organisational structures are available, therefore:
  * nothing here is a trained model; nothing is a probability P(G_c | D_c, E_c);
  * scores are uncalibrated heuristic "compatibility" values, comparable only within
    one company's candidate list;
  * the result may abstain; every hypothesis keeps its derivation + assumptions;
  * each unit is a hypothesis about a *capability cluster*, labelled HYPOTHESIS.

Baseline compatibility score of a candidate partition U of the capability nodes:
    compatibility(U) = Q(U) + lambda_ext * ext(U)   (heuristic; NOT a probability)
  Q(U)   = sum_{u in U} [ L_u/m - (d_u/(2m))^2 ]   Newman modularity on the undirected
           graph with edge weight w_ij = a_cooc * cooc_ij + a_rel * rel_ij, where
             cooc_ij = #patents where i,j co-occur            (weak: shared document only)
             rel_ij  = sum over typed relations (i,r,j)/(j,r,i) of #patents asserting it
                       (direct semantic relation extracted from the text)
           The two components are stored separately (edge_components); a_cooc, a_rel are
           explicit, UNVALIDATED parameters (defaults 1.0 each). Setting a_cooc=0 uses direct
           relations only. L_u = in-unit weight, d_u = sum of node degrees in u, m = total
           weight. Q(single unit) = 0. Every node of the graph must appear in exactly one unit
           (checked); isolated nodes have degree 0 and contribute 0.
  Modularity favours splitting a graph into dense clusters, so patent evidence alone cannot
  tell "distinct technical clusters inside ONE organisation" from "separate units". Hence by
  default (allow_patent_only=False) the module abstains unless independent external
  assertions apply to the candidate nodes.
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
    external_support: Optional[float]  # None = no applicable external assertion
    compatibility_score: float  # heuristic; not a probability, comparable only within one company


@dataclass(frozen=True)
class OrgInferenceResult:
    company: str
    ranked: Tuple[ScoredHypothesis, ...]
    abstained: bool
    reason: Optional[str]
    margin: Optional[float]
    missing_evidence: Tuple[str, ...] = ()  # which evidence is insufficient (when abstained)
    status: str = STATUS

    @property
    def best(self) -> Optional[ScoredHypothesis]:
        return None if self.abstained or not self.ranked else self.ranked[0]


class OrgPosteriorModel(Protocol):
    """Interface for a future trainable model of P(G_c | D_c, E_c). Needs reference graphs."""
    def fit(self, graphs: Sequence[CapabilityGraph], reference_structures: Sequence[object]) -> None: ...
    def log_posterior(self, graph: CapabilityGraph, hypothesis: OrgHypothesis) -> float: ...


def edge_components(g: CapabilityGraph) -> Dict[Tuple[str, str], Tuple[float, float]]:
    """pair -> (cooccurrence_count, relation_count); the two kinds of evidence stay separate."""
    comp: Dict[Tuple[str, str], List[float]] = defaultdict(lambda: [0.0, 0.0])
    for k, v in g.cooccurrence().items():
        comp[k][0] += v
    for (h, _r, t), e in g.edges.items():
        if h != t:
            comp[tuple(sorted((h, t)))][1] += len(e.patent_ids)
    return {k: (v[0], v[1]) for k, v in comp.items()}


def edge_weights(g: CapabilityGraph, cooc_weight: float = 1.0,
                 relation_weight: float = 1.0) -> Dict[Tuple[str, str], float]:
    """w_ij = a_cooc*cooc_ij + a_rel*rel_ij (zero-weight pairs dropped)."""
    if cooc_weight < 0 or relation_weight < 0:
        raise ValueError("edge weight coefficients must be >= 0")
    w = {k: cooc_weight * c + relation_weight * r for k, (c, r) in edge_components(g).items()}
    return {k: v for k, v in w.items() if v > 0}


def modularity(units: Sequence[FrozenSet[str]], w: Dict[Tuple[str, str], float],
               nodes: Optional[Sequence[str]] = None) -> float:
    """Newman modularity of a partition. Validates that units are disjoint, and (if `nodes`
    is given) cover exactly `nodes`; every edge endpoint must be in some unit."""
    m = sum(w.values())
    if m <= 0:
        raise ValueError("graph has no edge weight")
    unit_of: Dict[str, int] = {}
    for i, u in enumerate(units):
        for n in u:
            if n in unit_of:
                raise ValueError(f"node {n!r} appears in more than one unit")
            unit_of[n] = i
    if nodes is not None and set(nodes) != set(unit_of):
        raise ValueError("partition does not cover exactly the graph's nodes")
    deg: Dict[str, float] = defaultdict(float)
    inner = [0.0] * len(units)
    for (a, b), x in w.items():
        if a not in unit_of or b not in unit_of:
            raise ValueError(f"edge ({a!r},{b!r}) has an endpoint outside the partition")
        deg[a] += x
        deg[b] += x
        if unit_of[a] == unit_of[b]:
            inner[unit_of[a]] += x
    return sum(inner[i] / m - (sum(deg[n] for n in u) / (2 * m)) ** 2 for i, u in enumerate(units))


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


def generate_candidates(g: CapabilityGraph, external: Sequence[ExternalEvidence] = (),
                        include_category_hypothesis: bool = False,
                        cooc_weight: float = 1.0, relation_weight: float = 1.0) -> List[OrgHypothesis]:
    """Baseline candidate generation (no learning). Duplicated partitions are merged.
    The M/P/R/I-category grouping is OPT-IN: it assumes the organisation mirrors the four
    function types, which nothing in the spec supports."""
    nodes = sorted(g.nodes)
    w = edge_weights(g, cooc_weight, relation_weight)
    cands = [
        OrgHypothesis("single_unit", (frozenset(nodes),), "all capabilities in one unit",
                      ("no internal differentiation",)),
        OrgHypothesis("connected_components", tuple(_components(nodes, w)),
                      "connected components of the weighted capability graph (isolated nodes = singleton units)",
                      ("capabilities with no co-occurrence/relation evidence belong to different units",
                       "co-occurrence in a patent implies a shared unit (often false: "
                       "cross-functional inventions)")),
    ]
    node_set = set(nodes)
    pairs = [e.concepts for e in external if e.kind == "same_unit" and all(c in node_set for c in e.concepts)]
    if any(all(c in node_set for c in e.concepts) for e in external):
        same_w = {tuple(sorted(pr)): 1.0 for pr in pairs}
        cands.append(OrgHypothesis("external_constrained", tuple(_components(nodes, same_w)),
                                   "transitive closure of independent same_unit assertions; all other nodes singleton (different_unit assertions are respected by scoring)",
                                   ("only asserted same-unit links are merged",)))
    if include_category_hypothesis:
        cat: Dict[str, set] = defaultdict(set)
        for n in nodes:
            cc = g.nodes[n].category_counts
            cat[max(sorted(cc), key=lambda k: cc[k]) if cc else "unassigned"].add(n)
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
                       min_margin: float = 0.05, allow_patent_only: bool = False,
                       include_category_hypothesis: bool = False,
                       cooc_weight: float = 1.0, relation_weight: float = 1.0) -> OrgInferenceResult:
    def abstain(reason, missing, ranked=(), margin=None):
        return OrgInferenceResult(g.company, tuple(ranked), True, reason, margin, tuple(missing))

    if len(g.patent_ids) < min_patents:
        return abstain(f"insufficient evidence: {len(g.patent_ids)} patents < {min_patents}",
                       ["more patents for this company"])
    if len(g.nodes) < 2:
        return abstain("fewer than 2 capability nodes", ["at least two distinct capabilities"])
    w = edge_weights(g, cooc_weight, relation_weight)
    if not w:
        return abstain("no co-occurrence or relation evidence between capabilities",
                       ["co-occurrence or extracted relations linking capabilities"])
    nodes = sorted(g.nodes)
    scored = []
    for h in generate_candidates(g, external, include_category_hypothesis, cooc_weight, relation_weight):
        ext = _external_support(h, external)
        q = modularity(h.units, w, nodes)
        scored.append(ScoredHypothesis(h, q, ext, q + lambda_ext * (ext or 0.0)))
    scored.sort(key=lambda s: (-s.compatibility_score, s.hypothesis.name))
    n_ext = sum(1 for e in external if all(c in g.nodes for c in e.concepts))
    if n_ext == 0 and not allow_patent_only:
        return abstain("no independent organisational evidence applies to this graph: patent "
                       "co-occurrence cannot distinguish technical clusters from organisational units",
                       ["independent organisational evidence (E_c) about these capabilities"], scored)
    if len(scored) < 2:
        return abstain("only one distinct candidate structure; alternatives cannot be compared",
                       ["evidence that separates the capabilities into alternative structures"], scored)
    margin = scored[0].compatibility_score - scored[1].compatibility_score
    if margin < min_margin:
        return abstain(f"top-two margin {margin:.4f} < min_margin {min_margin}: evidence "
                       "cannot distinguish competing hypotheses",
                       ["evidence that favours one candidate structure over its closest alternative"],
                       scored, margin)
    return OrgInferenceResult(g.company, tuple(scored), False, None, margin)
