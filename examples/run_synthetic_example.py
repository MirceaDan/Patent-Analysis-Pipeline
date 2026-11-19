"""Minimal runnable demo on SYNTHETIC data:  python examples/run_synthetic_example.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patent_mpri import MPRIClassifier, build_capability_graphs, infer_organization
from patent_mpri.distillation import PrecomputedAnnotator
from synthetic_data import build

patents, anns = build()
print("*** SYNTHETIC DATA: outputs below demonstrate mechanics, not performance ***")
clf = MPRIClassifier(PrecomputedAnnotator(anns)).fit(patents)
print(f"lattice: {len(clf.lattice.concepts)} concepts, "
      f"{len(clf.shared_concepts())} shared across >=2 categories")
preds = {}
for p in patents:  # in-sample on purpose; no held-out metric is claimed
    pr = clf.predict(p)
    preds[p.patent_id] = pr.predicted
    print(p.patent_id, "label", p.label, "pred", pr.predicted,
          {k: round(v, 3) for k, v in pr.scores.items()})
graphs, skipped = build_capability_graphs(patents, anns, preds)
for company, g in graphs.items():
    res = infer_organization(g, min_patents=3)
    print(f"\n[{company}] nodes={sorted(g.nodes)} status={res.status}")
    print("  abstained:", res.abstained, "|", res.reason)
    for s in res.ranked:
        print(f"  {s.hypothesis.name}: score={s.score:.3f} units={[sorted(u) for u in s.hypothesis.units]}")
