"""SYNTHETIC, hand-made records for demos/tests only. NOT real patents, NOT real labels,
NOT for training or evaluating any real model. Kept apart from any real dataset."""
from patent_mpri import (Annotation, Cue, EvidenceType, Modality, Patent, Relation, SourceRef)

SYNTHETIC = True

# (patent_id, company, label, [terms in abstract], [(head_idx, rel, tail_idx)])
_SPEC = [
    ("SYN-1", "AcmeCo", "M", ["sensor data", "alert"], [(0, "triggers", 1)]),
    ("SYN-2", "AcmeCo", "M", ["sensor data", "anomaly detection"], [(0, "input_to", 1)]),
    ("SYN-3", "AcmeCo", "P", ["sensor data", "forecast model"], [(0, "input_to", 1)]),
    ("SYN-4", "AcmeCo", "P", ["forecast model", "anomaly detection"], []),
    ("SYN-5", "BetaCorp", "R", ["recommendation engine", "forecast model"], [(1, "feeds", 0)]),
    ("SYN-6", "BetaCorp", "R", ["recommendation engine", "user profile"], []),
    ("SYN-7", "BetaCorp", "I", ["actuator control", "alert"], [(1, "triggers", 0)]),
    ("SYN-8", "BetaCorp", "I", ["actuator control", "sensor data"], [(1, "input_to", 0)]),
]


def build(with_labels=True):
    patents, annotations = [], []
    for pid, co, lab, terms, rels in _SPEC:
        abstract = "The system uses " + " and ".join(terms) + "."
        p = Patent(pid, f"Synthetic {pid}", abstract, {"FIG1": f"Block diagram showing {terms[0]}."},
                   company=co, label=lab if with_labels else None, family_id=f"FAM-{pid}")
        cues = []
        for i, t in enumerate(terms):
            s = abstract.find(t)
            cues.append(Cue(f"{pid}-c{i}", t, "component", SourceRef(Modality.ABSTRACT, s, s + len(t)),
                            0.1 + 0.1 * i, EvidenceType.EXPLICIT))
        cues.append(Cue(f"{pid}-fig", terms[0], "component", SourceRef(Modality.FIGURE, figure_id="FIG1"),
                        0.3, EvidenceType.INTERPRETATION))
        rr = [Relation(f"{pid}-c{h}", r, f"{pid}-c{t}", 0.2) for h, r, t in rels]
        annotations.append(Annotation(pid, cues, rr, producer="synthetic-handwritten"))
        patents.append(p)
    return patents, annotations
