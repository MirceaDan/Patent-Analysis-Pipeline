"""Integration tests on SYNTHETIC data only (examples/synthetic_data.py)."""
import os, sys, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from examples.synthetic_data import SYNTHETIC, build
from patent_mpri import MPRIClassifier, build_capability_graphs, infer_organization
from patent_mpri.distillation import PrecomputedAnnotator


def test_pipeline_end_to_end():
    assert SYNTHETIC
    ps, an = build()
    clf = MPRIClassifier(PrecomputedAnnotator(an)).fit(ps)
    preds = {p.patent_id: clf.predict(p).predicted for p in ps}
    assert all(v in ("M", "P", "R", "I") for v in preds.values())
    graphs, _ = build_capability_graphs(ps, an, preds)
    res = {c: infer_organization(g, allow_patent_only=True) for c, g in graphs.items()}
    assert res["BetaCorp"].best.hypothesis.name == "connected_components"
    assert all(r.status.startswith("heuristic_uncalibrated") for r in res.values())


def test_graded_variant_runs_and_unlabeled_inference():
    ps, an = build()
    clf = MPRIClassifier(PrecomputedAnnotator(an), graded=True).fit(ps)
    ps_unl, _ = build(with_labels=False)
    pr = clf.predict(ps_unl[0])
    assert abs(sum(pr.softmax_scores.values()) - 1) < 1e-12


def test_example_script_runs():
    out = subprocess.run([sys.executable, os.path.join(ROOT, "examples", "run_synthetic_example.py")],
                         capture_output=True, text=True, check=True).stdout
    assert "SYNTHETIC" in out
