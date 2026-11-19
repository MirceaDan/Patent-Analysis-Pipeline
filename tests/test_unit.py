import itertools, math, os, sys
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from patent_mpri import *  # noqa
from patent_mpri.classifier import attribute_confidences
from patent_mpri.distillation import (PrecomputedAnnotator, cross_entropy, kd_loss, kl_divergence,
                                      lattice_loss, student_loss)
from patent_mpri.evaluation import (balanced_accuracy, expected_calibration_error, group_split,
                                    macro_f1, partition_pair_f1, set_prf)
from patent_mpri.graph import CapabilityGraph
from patent_mpri.orginference import generate_candidates, modularity, edge_weights


def cue(i, term, u=0.0, mod=Modality.ABSTRACT, s=0, e=1, **kw):
    return Cue(i, term, "component", SourceRef(mod, s, e), u, **kw)


# ---- 1. schema validation
def test_cue_validation():
    with pytest.raises(SchemaError): cue("c", "  ")
    with pytest.raises(SchemaError): cue("c", "x", u=1.5)
    with pytest.raises(SchemaError): cue("c", "x", u=float("nan"))
    with pytest.raises(SchemaError): SourceRef(Modality.ABSTRACT, 5, 2)
    with pytest.raises(SchemaError): SourceRef(Modality.ABSTRACT, 5, None)
    with pytest.raises(SchemaError): SourceRef(Modality.FIGURE)
    assert cue("c", " Sensor  DATA ").attribute == "sensor data"
    assert cue("c", "x", u=0.25).confidence == 0.75


def test_annotation_validation():
    a, b = cue("a", "x"), cue("b", "y")
    with pytest.raises(SchemaError): Annotation("p", [a, a])
    with pytest.raises(SchemaError): Annotation("p", [a], [Relation("a", "r", "zzz", 0.1)])
    p = Patent("p", "t", "abc", {"F1": "fig"})
    with pytest.raises(SchemaError): Annotation("p", [cue("a", "x", s=0, e=99)]).validate_against(p)
    with pytest.raises(SchemaError):
        Annotation("p", [Cue("f", "x", "r", SourceRef(Modality.FIGURE, figure_id="F9"), 0.1)]).validate_against(p)
    Annotation("p", [a, b], [Relation("a", "r", "b", 0.1)]).validate_against(p)


# ---- 7. malformed / missing
def test_patent_malformed():
    with pytest.raises(SchemaError): Patent("", "t", "a")
    with pytest.raises(SchemaError): Patent("p", "", "  ")
    with pytest.raises(SchemaError): Patent("p", "t", "a", label="X")
    with pytest.raises(SchemaError): Patent.from_dict({"patent_id": "p"})
    with pytest.raises(SchemaError): Patent.from_dict("nope")
    with pytest.raises(SchemaError): Patent("p", None, "a")
    p = Patent.from_dict({"patent_id": "p", "title": "t", "abstract": ""})  # missing optional fields OK
    assert p.figure_descriptions == {} and p.company is None


def test_company_not_classifier_input():
    p = Patent("p", "t", "a", {"F": "f"}, company="Acme", label="M")
    assert p.classifier_input() == ("t", "a", {"F": "f"})


def test_empty_collections():
    clf = MPRIClassifier(PrecomputedAnnotator([]))
    with pytest.raises(ValueError): clf.fit([])
    with pytest.raises(RuntimeError): clf.predict(Patent("p", "t", "a"))
    ctx = FormalContext([], [], [])
    assert [(set(c.extent), set(c.intent)) for c in ConceptLattice(ctx).concepts] == [(set(), set())]
    assert build_capability_graphs([], []) == ({}, [])
    with pytest.raises(SchemaError):
        MPRIClassifier(PrecomputedAnnotator([])).fit([Patent("p", "t", "a")])  # unlabeled


# ---- 3. FCA on a hand-verified example
@pytest.fixture
def ctx():
    inc = [("1", "a"), ("1", "b"), ("2", "a"), ("2", "c"), ("3", "a"), ("3", "b"), ("3", "c"), ("4", "d")]
    return FormalContext("1234", "abcd", inc)


def test_derivation_operators(ctx):
    assert ctx.common_attributes({"1", "2"}) == {"a"}
    assert ctx.common_objects({"a", "b"}) == {"1", "3"}
    assert ctx.common_attributes(set()) == set("abcd")
    assert ctx.common_objects(set()) == set("1234")


def test_lattice_concepts_match_manual(ctx):
    got = {(c.extent, c.intent) for c in ConceptLattice(ctx).concepts}
    manual = {
        (frozenset("1234"), frozenset()),
        (frozenset("123"), frozenset("a")),
        (frozenset("13"), frozenset("ab")),
        (frozenset("23"), frozenset("ac")),
        (frozenset("3"), frozenset("abc")),
        (frozenset("4"), frozenset("d")),
        (frozenset(), frozenset("abcd")),
    }
    assert got == manual


def test_nextclosure_equals_bruteforce(ctx):
    brute = set()
    for r in range(5):
        for Y in itertools.combinations(ctx.attributes, r):
            X = ctx.common_objects(Y)
            if ctx.common_attributes(X) == frozenset(Y):
                brute.add((X, frozenset(Y)))
    assert {(c.extent, c.intent) for c in ctx.concepts()} == brute
    assert len(ctx.concepts()) == len(brute)  # no duplicates


def test_lattice_order_and_covers(ctx):
    L = ConceptLattice(ctx)
    idx = {(c.extent, c.intent): i for i, c in enumerate(L.concepts)}
    cov = {(L.concepts[i].intent, L.concepts[j].intent) for i, j in L.covers}
    assert (frozenset(), frozenset("a")) in cov
    assert (frozenset("a"), frozenset("ab")) in cov and (frozenset("a"), frozenset("ac")) in cov
    assert (frozenset("ab"), frozenset("abc")) in cov
    assert (frozenset(), frozenset("d")) in cov
    assert (frozenset("a"), frozenset("abc")) not in cov  # transitive, not a cover
    assert L.top().intent == frozenset() and L.bottom().extent == frozenset()
    for c in L.concepts:  # order equivalence: X1<=X2 iff Y2<=Y1
        for d in L.concepts:
            assert (c.extent <= d.extent) == (d.intent <= c.intent)


def test_bad_incidence():
    with pytest.raises(ValueError): FormalContext(["p"], ["a"], [("p", "zzz")])


# ---- 2/4. scoring: hand-computed values
def _tiny():
    ann = [Annotation("p1", [cue("a1", "a")]), Annotation("p2", [cue("a2", "a"), cue("b2", "b", s=1, e=2)])]
    lat = ConceptLattice(FormalContext.from_annotations(ann))
    attrs = {a.patent_id: attribute_confidences(a) for a in ann}
    return lat, attrs, {"p1": "M", "p2": "P"}


def test_rho_and_scores_hand_computed():
    lat, attrs, labels = _tiny()
    sc = ConceptScorer.fit(lat.concepts, attrs, labels, eps=1e-12)
    by_intent = {frozenset(c.intent): r for c, r in zip(sc.concepts, sc.rho)}
    assert by_intent[frozenset("a")]["M"] == pytest.approx(0.5)
    assert by_intent[frozenset("a")]["P"] == pytest.approx(0.5)
    assert by_intent[frozenset("ab")]["P"] == pytest.approx(1.0)
    assert by_intent[frozenset("ab")]["M"] == 0.0
    s = sc.raw_scores({"a": 1.0, "b": 1.0})  # matches both concepts: (0.5+..)/2
    assert s["M"] == pytest.approx(0.25) and s["P"] == pytest.approx(0.75) and s["R"] == 0 == s["I"]
    assert sc.predict({"a": 1.0, "b": 1.0}) == "P"
    d = sc.deltas({"a": 1.0, "b": 1.0})
    assert sum(x["P"] for x in d) == pytest.approx(1.5)  # Delta sums to numerator
    probs = sc.softmax_scores({"a": 1.0, "b": 1.0})
    assert sum(probs.values()) == pytest.approx(1.0)


def test_weights_change_scores():
    lat, attrs, labels = _tiny()
    w = [0.0 if len(c.intent) == 1 else 1.0 for c in lat.concepts]
    sc = ConceptScorer.fit(lat.concepts, attrs, labels, weights=w, eps=1e-12)
    assert sc.raw_scores({"a": 1.0, "b": 1.0})["P"] == pytest.approx(1.0)
    with pytest.raises(ValueError): ConceptScorer.fit(lat.concepts, attrs, labels, weights=[-1] * len(lat.concepts))


def test_abstain_when_nothing_matches():
    c = FormalConcept(frozenset({"p"}), frozenset({"a"}))
    sc = ConceptScorer([c], [{k: 0.25 for k in LABELS}])
    assert sc.predict({"zzz": 1.0}) is None
    assert sc.raw_scores({"zzz": 1.0}) == {k: 0.0 for k in LABELS}


def test_graded_mu_reduces_to_binary_and_is_partial():
    c = FormalConcept(frozenset(), frozenset("ab"))
    assert graded_mean_mu({"a": 1.0, "b": 1.0}, c) == binary_containment_mu({"a": 1.0, "b": 1.0}, c) == 1.0
    assert graded_mean_mu({"a": 1.0}, c) == 0.5 and binary_containment_mu({"a": 1.0}, c) == 0.0
    assert graded_mean_mu({"a": 0.4, "b": 0.8}, c) == pytest.approx(0.6)


# ---- 8. numerical stability / reproducibility
def test_softmax_stable_and_validated():
    p = stable_softmax([1000.0, 1001.0, -1000.0])
    assert sum(p) == pytest.approx(1.0) and all(math.isfinite(x) for x in p)
    assert stable_softmax([0.2, 0.2, 0.2, 0.2], 1e-3) == pytest.approx([0.25] * 4)
    for T in (0, -1, float("inf")):
        with pytest.raises(ValueError): stable_softmax([1.0], T)
    with pytest.raises(ValueError): stable_softmax([])


def test_deterministic_fit_and_ties():
    from examples.synthetic_data import build
    ps, an = build()
    r1 = MPRIClassifier(PrecomputedAnnotator(an)).fit(ps)
    r2 = MPRIClassifier(PrecomputedAnnotator(list(reversed(an)))).fit(list(reversed(ps)))
    assert r1.scorer.rho == [dict(x) for x in r1.scorer.rho]
    k = lambda c: sorted(c.intent)
    d1 = {tuple(k(c)): r for c, r in zip(r1.scorer.concepts, r1.scorer.rho)}
    d2 = {tuple(k(c)): r for c, r in zip(r2.scorer.concepts, r2.scorer.rho)}
    assert d1 == d2
    assert r1.predict(ps[0]).raw_scores == r1.predict(ps[0]).raw_scores
    tied = ConceptScorer([FormalConcept(frozenset(), frozenset({"a"}))], [{k: 0.25 for k in LABELS}])
    assert tied.predict({"a": 1.0}) == "M"  # fixed tie-break


def test_eps_effect_small_and_validated():
    lat, attrs, labels = _tiny()
    a = ConceptScorer.fit(lat.concepts, attrs, labels, eps=1e-9).raw_scores({"a": 1.0, "b": 1.0})
    b = ConceptScorer.fit(lat.concepts, attrs, labels, eps=1e-12).raw_scores({"a": 1.0, "b": 1.0})
    assert all(abs(a[k] - b[k]) < 1e-8 for k in LABELS)
    with pytest.raises(ValueError): compute_rho(lat.concepts, attrs, labels, eps=0)


# ---- distillation
def test_kd_and_losses():
    assert kl_divergence([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0)
    assert kl_divergence([1, 0], [0.5, 0.5]) == pytest.approx(math.log(2))
    assert cross_entropy([0.25, 0.75], 1) == pytest.approx(-math.log(0.75))
    t = [[0.7, 0.1, 0.1, 0.1]]
    s = [[2.0, 0.0, 0.0, 0.0]]
    assert kd_loss(t, s, 2.0, True) >= 0
    assert kd_loss(t, [[math.log(.7), math.log(.1), math.log(.1), math.log(.1)]], 1.0, True) == pytest.approx(0, abs=1e-9)
    with pytest.raises(ValueError): kd_loss(t, s, 2.0, False)  # uncalibrated teacher -> omit
    assert student_loss(0.5, 2.0) == 1.0
    with pytest.raises(NotImplementedError): student_loss(0.5, 1.0, lam_span=1.0)
    with pytest.raises(NotImplementedError): lattice_loss()


# ---- 4. shared concepts
def test_shared_concepts_preserved():
    from examples.synthetic_data import build
    ps, an = build()
    clf = MPRIClassifier(PrecomputedAnnotator(an)).fit(ps)
    shared = {frozenset(c.intent): ks for c, ks in clf.shared_concepts()}
    assert set(shared[frozenset({"sensor data"})]) == {"M", "P", "I"}
    assert set(shared[frozenset({"forecast model"})]) == {"P", "R"}


# ---- 5. provenance & graph aggregation
def test_prediction_provenance():
    from examples.synthetic_data import build
    ps, an = build()
    clf = MPRIClassifier(PrecomputedAnnotator(an)).fit(ps)
    pr = clf.predict(ps[0])
    assert pr.supporting and all(s.cues for s in pr.supporting if s.intent)  # top concept has empty intent
    ids = {c.cue_id for s in pr.supporting for c in s.cues}
    assert ids <= {c.cue_id for c in an[0].cues}


def test_graph_aggregation_and_provenance():
    from examples.synthetic_data import build
    ps, an = build()
    graphs, skipped = build_capability_graphs(ps, an, {p.patent_id: p.label for p in ps})
    assert skipped == [] and set(graphs) == {"AcmeCo", "BetaCorp"}
    g = graphs["AcmeCo"]
    n = g.nodes["sensor data"]
    assert n.patent_ids == {"SYN-1", "SYN-2", "SYN-3"}
    assert len(n.evidence) == 6  # abstract cue + figure cue per patent
    assert all(e.patent_id in n.patent_ids and e.cue_id for e in n.evidence)
    assert any(e.source.modality is Modality.FIGURE for e in n.evidence)
    assert n.category_counts == {"M": 2, "P": 1}
    e = g.edges[("sensor data", "input_to", "anomaly detection")]
    assert e.patent_ids == {"SYN-2"} and len(e.evidence) == 1
    assert g.cooccurrence()[("anomaly detection", "forecast model")] == 1


def test_graph_skips_missing_company_or_annotation():
    ps = [Patent("a", "t", "x"), Patent("b", "t", "x", company="Z")]
    g, sk = build_capability_graphs(ps, [Annotation("a", [cue("c", "k")])])
    assert g == {} and sorted(sk) == ["a", "b"]


def test_company_not_leaked_into_classifier():
    seen = []
    class Spy:
        def annotate(self, p):
            seen.append((p.company, p.label, p.family_id))
            return Annotation(p.patent_id, [cue("c", "k")])
    clf = MPRIClassifier(Spy()).fit([Patent("p", "t", "a", company="X", label="M", family_id="F")])
    clf.predict(Patent("q", "t", "a", company="X"))
    assert seen and all(s == (None, None, None) for s in seen)


# ---- 6. org inference with insufficient evidence
def _g(n_patents, concept_sets, company="Co"):
    g = CapabilityGraph(company)
    from patent_mpri.graph import CapabilityNode
    for i, cs in enumerate(concept_sets[:n_patents]):
        pid = f"p{i}"
        g.patent_ids.add(pid)
        g.patent_concepts[pid] = frozenset(cs)
        for c in cs:
            g.nodes.setdefault(c, CapabilityNode(c)).patent_ids.add(pid)
    return g


def test_org_abstains_few_patents():
    r = infer_organization(_g(2, [{"a", "b"}, {"c", "d"}]), min_patents=3)
    assert r.abstained and "insufficient" in r.reason and r.best is None
    assert r.status.startswith("heuristic_uncalibrated")


def test_org_abstains_no_relations_or_single_node():
    assert infer_organization(_g(3, [{"a"}, {"a"}, {"a"}]), allow_patent_only=True).abstained
    assert infer_organization(_g(3, [{"a"}, {"b"}, {"c"}]), allow_patent_only=True).reason.startswith("no co-occurrence")
    assert infer_organization(CapabilityGraph("Empty")).abstained


def test_org_abstains_on_indistinguishable():
    # all patents contain all concepts -> one component == single unit -> one distinct candidate
    r = infer_organization(_g(3, [{"a", "b", "c"}] * 3), allow_patent_only=True)
    assert r.abstained and "one distinct" in r.reason
    # margin threshold
    g = _g(4, [{"a", "b"}, {"a", "b"}, {"c", "d"}, {"c", "d"}])
    assert not infer_organization(g, allow_patent_only=True).abstained
    assert infer_organization(g, min_margin=10, allow_patent_only=True).abstained


def test_org_modularity_hand_value_and_external_evidence():
    g = _g(4, [{"a", "b"}, {"a", "b"}, {"c", "d"}, {"c", "d"}])
    w = edge_weights(g)
    assert w == {("a", "b"): 2, ("c", "d"): 2}
    q = modularity([frozenset("ab"), frozenset("cd")], w, list("abcd"))
    assert q == pytest.approx(0.5)  # 2*(2/4 - (4/8)^2)
    assert modularity([frozenset("abcd")], w) == pytest.approx(0.0)
    ext = [ExternalEvidence("org-page", "different_unit", ("a", "c"))]
    r = infer_organization(g, ext)
    top = r.best
    assert top.hypothesis.name == "connected_components" and top.external_support == 1.0
    assert top.hypothesis.evidence_type == "capability_hypothesis" and top.hypothesis.assumptions
    assert len(r.ranked) >= 2  # alternatives preserved
    with pytest.raises(ValueError): ExternalEvidence("s", "bad", ("a", "b"))


# ---- evaluation interfaces
def test_metrics():
    yt, yp = ["M", "M", "P", "R"], ["M", "P", "P", None]
    assert macro_f1(yt, yp) == pytest.approx((2/3 + 2/3 + 0) / 3)  # M:P=1,R=.5; P:P=.5,R=1; R:0
    assert balanced_accuracy(["M", "M", "P"], ["M", "P", "P"]) == pytest.approx(0.75)
    assert macro_f1(["M", "P"], ["M", "P"]) == 1.0
    assert set_prf({1, 2}, {2, 3}) == (0.5, 0.5, 0.5)
    assert partition_pair_f1([{"a", "b"}], [{"a", "b"}]) == (1.0, 1.0, 1.0)
    pr = [{"M": 1.0, "P": 0, "R": 0, "I": 0}]
    assert expected_calibration_error(pr, ["M"]) == 0.0 and expected_calibration_error(pr, ["P"]) == 1.0
    with pytest.raises(ValueError): macro_f1([], [])
    tr, te = group_split(list("abcd"), ["f1", "f1", "f2", "f3"], 0.34, seed=1)
    assert not ({g for i, g in zip("abcd", ["f1", "f1", "f2", "f3"]) if i in te} &
                {g for i, g in zip("abcd", ["f1", "f1", "f2", "f3"]) if i in tr})
