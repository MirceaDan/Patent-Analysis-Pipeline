"""Regression tests for the audit corrections (empty-intent exclusion, score naming, org inference)."""
import os, sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from patent_mpri import *  # noqa
from patent_mpri.classifier import attribute_confidences
from patent_mpri.distillation import PrecomputedAnnotator
from patent_mpri.graph import CapabilityGraph, CapabilityNode
from patent_mpri.orginference import edge_components, edge_weights, generate_candidates, modularity

EPS = 1e-6


def cue(i, term, s=0):
    return Cue(i, term, "component", SourceRef(Modality.ABSTRACT, s, s + 1), 0.0)


@pytest.fixture
def fitted():
    """M:{a,s},{a,t}  P:{b,s},{b,t}. No attribute is shared by all -> top concept has empty intent."""
    spec = [("p1", "M", "as"), ("p2", "M", "at"), ("p3", "P", "bs"), ("p4", "P", "bt")]
    anns = [Annotation(pid, [cue(f"{pid}{t}", t, i) for i, t in enumerate(ts)]) for pid, _, ts in spec]
    ctx = FormalContext.from_annotations(anns)
    lat = ConceptLattice(ctx)
    attrs = {a.patent_id: attribute_confidences(a) for a in anns}
    labels = {pid: lab for pid, lab, _ in spec}
    return lat, attrs, labels


def sc_of(fitted, **kw):
    lat, attrs, labels = fitted
    return ConceptScorer.fit(lat.concepts, attrs, labels, **kw)


# ---- P1.1 / P1.3 empty-intent concept excluded; no misleading scores without evidence
def test_empty_intent_concept_present_but_excluded(fitted):
    sc = sc_of(fitted)
    assert any(not c.intent for c in sc.concepts)
    assert sc.match_mass({}) == 0.0
    assert sc.raw_scores({}) == {k: 0.0 for k in LABELS}
    assert sc.softmax_scores({}) is None and sc.predict({}) is None and not sc.has_evidence({})


def test_unseen_terms_give_no_evidence(fitted):
    sc = sc_of(fitted)
    assert sc.raw_scores({"zzz": 1.0}) == {k: 0.0 for k in LABELS}
    assert sc.predict({"zzz": 1.0}) is None


def test_literal_sum_over_all_concepts_is_opt_in_and_shows_the_artifact(fitted):
    sc = sc_of(fitted, include_empty_intent=True)
    s = sc.raw_scores({})  # no cues at all, yet the empty-intent concept injects the training prior
    assert s["M"] == pytest.approx(0.5, abs=EPS) and s["P"] == pytest.approx(0.5, abs=EPS)


def test_classifier_abstains_without_relevant_cues():
    anns = [Annotation("p1", [cue("c1", "a")]), Annotation("p2", [cue("c2", "b")])]
    ps = [Patent("p1", "t", "x", label="M"), Patent("p2", "t", "x", label="P")]
    clf = MPRIClassifier(PrecomputedAnnotator(anns)).fit(ps)
    q = Patent("q", "t", "x")
    pr = clf.predict(q, annotation=Annotation("q", [cue("z", "unrelated")]))
    assert pr.predicted is None and pr.softmax_scores is None and pr.supporting == ()
    assert pr.raw_scores == {k: 0.0 for k in LABELS}
    assert pr.calibration_status == "uncalibrated"


def test_matched_concepts_with_zero_rho_abstain():
    c = FormalConcept(frozenset(), frozenset({"a"}))
    sc = ConceptScorer([c], [{k: 0.0 for k in LABELS}])
    assert sc.match_mass({"a": 1.0}) == 1.0 and sc.predict({"a": 1.0}) is None


# ---- P1.4 scores change for the right reasons (hand-computed)
def test_adding_and_removing_evidence(fitted):
    sc = sc_of(fitted)
    approx = lambda v: pytest.approx(v, abs=EPS)
    s = sc.raw_scores({"a": 1.0})  # matches only concept {a} (rho M=1)
    assert s["M"] == approx(1.0) and s["P"] == 0.0
    s = sc.raw_scores({"a": 1.0, "s": 1.0})  # +{s} (M .5,P .5) +{a,s} (M 1): M=2.5/3, P=.5/3
    assert s["M"] == approx(2.5 / 3) and s["P"] == approx(0.5 / 3)
    assert sc.raw_scores({"a": 1.0})["M"] == approx(1.0)  # removing 's' restores the score
    s = sc.raw_scores({"a": 1.0, "b": 1.0})  # {a}->M, {b}->P: tie
    assert s["M"] == approx(0.5) and s["P"] == approx(0.5)
    assert sc.raw_scores({"b": 1.0})["P"] == approx(1.0)
    # irrelevant extra cue changes nothing
    assert sc.raw_scores({"a": 1.0, "zzz": 1.0}) == sc.raw_scores({"a": 1.0})


# ---- P1.2 priors separate
def test_priors_are_separate_from_evidence(fitted):
    sc = sc_of(fitted)
    assert sc.class_priors == {"M": 0.5, "P": 0.5, "R": 0.0, "I": 0.0}
    assert sc.raw_scores({}) == {k: 0.0 for k in LABELS}  # priors never leak into s_k


# ---- P1.5 naming
def test_softmax_vs_calibrated(fitted):
    sc = sc_of(fitted)
    sm = sc.softmax_scores({"a": 1.0})
    assert sum(sm.values()) == pytest.approx(1.0)
    assert not hasattr(sc, "probabilities") and not hasattr(sc, "scores")
    with pytest.raises(NotImplementedError):
        sc.calibrated_probabilities({"a": 1.0})
    assert ConceptScorer.CALIBRATION_STATUS == "uncalibrated"


# ---- P1.6 exactness of rho and mu
def test_binary_mu_equals_extent_membership(fitted):
    lat, attrs, _ = fitted
    for c in lat.concepts:
        for pid, a in attrs.items():
            assert (binary_containment_mu(a, c) == 1.0) == (pid in c.extent)


def test_rho_matches_spec_formula_independently(fitted):
    lat, attrs, labels = fitted
    sc = sc_of(fitted, eps=1e-9)
    for c, r in zip(sc.concepts, sc.rho):
        for k in LABELS:
            num = sum(binary_containment_mu(attrs[p], c) for p in attrs if labels[p] == k)
            den = sum(binary_containment_mu(attrs[p], c) for p in attrs)
            assert r[k] == pytest.approx(num / (den + 1e-9), abs=1e-15)


def test_empty_extent_concept_dilutes_mass_documented_behaviour(fitted):
    # Bottom concept (empty extent, rho=0) still adds mu*w to the denominator when matched,
    # exactly as the spec's sum over C prescribes. Tested so the behaviour is explicit.
    sc = sc_of(fitted)
    bottom = [i for i, c in enumerate(sc.concepts) if not c.extent][0]
    assert all(v == 0 for v in sc.rho[bottom].values())
    allattrs = {a: 1.0 for a in "abst"}
    assert sc.mu(allattrs, sc.concepts[bottom]) == 1.0


# ---- P2.1/2.2 trace and direct relation vs co-occurrence
def _annot_graph():
    ps = [Patent(f"p{i}", "t", "abc", company="Co") for i in range(1, 4)]
    an = [
        Annotation("p1", [cue("x1", "x"), cue("y1", "y", 1)], [Relation("x1", "enables", "y1", 0.1)]),
        Annotation("p2", [cue("x2", "x"), cue("y2", "y", 1)]),
        Annotation("p3", [cue("x3", "x"), cue("z3", "z", 1)]),
    ]
    g, _ = build_capability_graphs(ps, an)
    return g["Co"]


def test_cooccurrence_and_relation_kept_separate():
    g = _annot_graph()
    assert edge_components(g) == {("x", "y"): (2.0, 1.0), ("x", "z"): (1.0, 0.0)}
    assert edge_weights(g) == {("x", "y"): 3.0, ("x", "z"): 1.0}
    assert edge_weights(g, relation_weight=0.0) == {("x", "y"): 2.0, ("x", "z"): 1.0}
    assert edge_weights(g, cooc_weight=0.0) == {("x", "y"): 1.0}  # direct relations only
    with pytest.raises(ValueError):
        edge_weights(g, cooc_weight=-1)


def _g(concept_sets):
    g = CapabilityGraph("Co")
    for i, cs in enumerate(concept_sets):
        pid = f"p{i}"
        g.patent_ids.add(pid)
        g.patent_concepts[pid] = frozenset(cs)
        for c in cs:
            g.nodes.setdefault(c, CapabilityNode(c)).patent_ids.add(pid)
    return g


# ---- P2.3 modularity with disconnected graph and isolated nodes
def test_modularity_isolated_node_and_disconnected():
    g = _g([{"a", "b"}, {"a", "b"}, {"c", "d"}, {"c", "d"}, {"e"}])
    w = edge_weights(g)
    nodes = sorted(g.nodes)
    comps = generate_candidates(g)[1]
    assert comps.name == "connected_components" and frozenset({"e"}) in comps.units
    assert modularity(comps.units, w, nodes) == pytest.approx(0.5)
    # isolated node placed into a cluster: degree 0, so Q unchanged
    q = modularity([frozenset("abe"), frozenset("cd")], w, nodes)
    assert q == pytest.approx(0.5)
    assert modularity([frozenset(nodes)], w, nodes) == pytest.approx(0.0)


def test_modularity_validates_partition():
    w = {("a", "b"): 1.0}
    with pytest.raises(ValueError): modularity([frozenset("a")], w)  # endpoint outside partition
    with pytest.raises(ValueError): modularity([frozenset("ab"), frozenset("b")], w)  # overlap
    with pytest.raises(ValueError): modularity([frozenset("ab")], w, ["a", "b", "c"])  # missing node
    with pytest.raises(ValueError): modularity([frozenset("ab")], {})


# ---- P2.4 M/P/R/I grouping is opt-in
def test_category_hypothesis_opt_in():
    g = _g([{"a", "b"}, {"a", "b"}, {"c", "d"}])
    g.nodes["a"].category_counts["M"] = 2
    names = lambda **k: {h.name for h in generate_candidates(g, **k)}
    assert "by_dominant_mpri_category" not in names()
    opt = [h for h in generate_candidates(g, include_category_hypothesis=True) if h.name == "by_dominant_mpri_category"]
    assert opt and any("unverified" in a for a in opt[0].assumptions)


# ---- P2.6 adversarial cases
def test_adversarial_distinct_clusters_one_organisation():
    g = _g([{"a", "b"}, {"a", "b"}, {"c", "d"}, {"c", "d"}])
    # patent-only: the clusters look like separate units, but that is not organisational evidence
    r0 = infer_organization(g)
    assert r0.abstained and r0.missing_evidence and "organisational" in r0.reason
    # independent evidence says a and c are in the same unit -> single unit must win
    ext = [ExternalEvidence("org-page", "same_unit", ("a", "c"))]
    r = infer_organization(g, ext)
    assert not r.abstained and r.best.hypothesis.name == "single_unit"
    names = [s.hypothesis.name for s in r.ranked]
    assert "connected_components" in names  # alternative retained
    cc = next(s for s in r.ranked if s.hypothesis.name == "connected_components")
    assert cc.external_support == 0.0


def test_adversarial_similar_capabilities_separate_units():
    g = _g([{"a", "b"}] * 4)  # a and b always co-occur
    assert infer_organization(g, allow_patent_only=True).abstained  # one candidate only
    ext = [ExternalEvidence("job-ads", "different_unit", ("a", "b"))]
    r = infer_organization(g, ext)
    assert not r.abstained
    assert r.best.hypothesis.name == "external_constrained"
    assert [sorted(u) for u in r.best.hypothesis.units] == [["a"], ["b"]]
    assert r.best.external_support == 1.0


# ---- P2.5 / P2.7
def test_scores_are_heuristic_not_probabilities_and_abstention_explains():
    g = _g([{"a", "b"}, {"a", "b"}, {"c", "d"}, {"c", "d"}])
    r = infer_organization(g, allow_patent_only=True)
    assert r.status == "heuristic_uncalibrated_not_a_probability"
    assert not hasattr(r.ranked[0], "probability")
    assert hasattr(r.ranked[0], "compatibility_score")
    few = infer_organization(_g([{"a", "b"}]))
    assert few.abstained and few.missing_evidence == ("more patents for this company",)


def test_affine_softmax():
    from patent_mpri.distillation import affine_softmax
    p = affine_softmax([[1.0, 0.0], [0.0, 1.0]], [2.0, 0.0], [0.0, 0.0])
    assert p[0] > p[1] and sum(p) == pytest.approx(1.0)
    assert p[0] == pytest.approx(2.718281828 ** 2 / (2.718281828 ** 2 + 1), rel=1e-6)
