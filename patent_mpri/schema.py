"""Data schema (spec §1, §3, §6).

Notation (spec -> code)
  x_p = (t_p, a_p, v_p)      -> Patent.title / .abstract / .figure_descriptions
  y_p in {M,P,R,I}           -> Patent.label (optional; None at inference)
  c(p)                       -> Patent.company  (metadata; NEVER part of classifier_input())
  q = (s, r, l, e, u)        -> Cue(term, role, source, uncertainty) + Relation (the e's)
  d_i = (concept, source, location, score, uncertainty, evidence type) -> EvidenceRecord
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

LABELS: Tuple[str, ...] = ("M", "P", "R", "I")


class SchemaError(ValueError):
    """Raised for malformed patent / cue / relation records."""


class Modality(str, Enum):
    TITLE = "title"
    ABSTRACT = "abstract"
    FIGURE = "figure"  # VLM-generated description (may contain VLM errors)


class EvidenceType(str, Enum):
    """Spec §6: explicit statement / semantic interpretation / capability hypothesis."""
    EXPLICIT = "explicit_statement"
    INTERPRETATION = "semantic_interpretation"
    CAPABILITY_HYPOTHESIS = "capability_hypothesis"


def _check_unit(name: str, v: float) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or math.isnan(v) or not 0.0 <= v <= 1.0:
        raise SchemaError(f"{name} must be a number in [0,1], got {v!r}")
    return float(v)


@dataclass(frozen=True)
class SourceRef:
    """Location l_pi: a character span in a text modality, or a figure id."""
    modality: Modality
    start: Optional[int] = None
    end: Optional[int] = None
    figure_id: Optional[str] = None

    def __post_init__(self):
        if not isinstance(self.modality, Modality):
            raise SchemaError(f"modality must be Modality, got {self.modality!r}")
        if (self.start is None) != (self.end is None):
            raise SchemaError("start and end must be given together")
        if self.start is not None and not (0 <= self.start <= self.end):
            raise SchemaError(f"invalid span [{self.start},{self.end})")
        if self.modality is Modality.FIGURE and self.figure_id is None:
            raise SchemaError("figure source requires figure_id")


@dataclass(frozen=True)
class Cue:
    """q_pi: term s, role r, location l, uncertainty u in [0,1] (confidence = 1-u)."""
    cue_id: str
    term: str
    role: str
    source: SourceRef
    uncertainty: float
    evidence_type: EvidenceType = EvidenceType.INTERPRETATION

    def __post_init__(self):
        if not isinstance(self.term, str) or not self.term.strip():
            raise SchemaError("cue term must be a non-empty string")
        if not isinstance(self.role, str) or not self.role.strip():
            raise SchemaError("cue role must be a non-empty string")
        if not self.cue_id:
            raise SchemaError("cue_id required")
        if not isinstance(self.source, SourceRef):
            raise SchemaError("cue source must be a SourceRef")
        object.__setattr__(self, "uncertainty", _check_unit("uncertainty", self.uncertainty))
        if not isinstance(self.evidence_type, EvidenceType):
            raise SchemaError("evidence_type must be EvidenceType")

    @property
    def confidence(self) -> float:
        return 1.0 - self.uncertainty

    @property
    def attribute(self) -> str:
        """Default attribute in the universe A: the normalised term."""
        return " ".join(self.term.lower().split())


@dataclass(frozen=True)
class Relation:
    """Element of e_pi / R_hat_p: typed relation between two cues of the same patent."""
    head: str
    relation: str
    tail: str
    uncertainty: float

    def __post_init__(self):
        if not (self.head and self.tail and self.relation):
            raise SchemaError("relation needs head, relation, tail")
        object.__setattr__(self, "uncertainty", _check_unit("uncertainty", self.uncertainty))


@dataclass(frozen=True)
class Annotation:
    """Q_p together with relations; same schema for teacher and student output."""
    patent_id: str
    cues: Tuple[Cue, ...] = ()
    relations: Tuple[Relation, ...] = ()
    producer: str = "unspecified"  # e.g. "teacher:<model>", "student:<ckpt>", "human"

    def __post_init__(self):
        object.__setattr__(self, "cues", tuple(self.cues))
        object.__setattr__(self, "relations", tuple(self.relations))
        ids = [c.cue_id for c in self.cues]
        if len(set(ids)) != len(ids):
            raise SchemaError("duplicate cue_id in annotation")
        known = set(ids)
        for r in self.relations:
            if r.head not in known or r.tail not in known:
                raise SchemaError(f"relation references unknown cue: {r}")

    def validate_against(self, patent: "Patent") -> None:
        """Spans must lie inside the text they point to; figure ids must exist."""
        if patent.patent_id != self.patent_id:
            raise SchemaError("annotation/patent id mismatch")
        for c in self.cues:
            s = c.source
            if s.modality is Modality.FIGURE:
                if s.figure_id not in patent.figure_descriptions:
                    raise SchemaError(f"cue {c.cue_id}: unknown figure {s.figure_id!r}")
                text = patent.figure_descriptions[s.figure_id]
            else:
                text = patent.title if s.modality is Modality.TITLE else patent.abstract
            if s.end is not None and s.end > len(text):
                raise SchemaError(f"cue {c.cue_id}: span exceeds source text")


@dataclass(frozen=True)
class Patent:
    patent_id: str
    title: str
    abstract: str
    figure_descriptions: Dict[str, str] = field(default_factory=dict)  # figure_id -> VLM text
    company: Optional[str] = None  # organisational metadata only
    label: Optional[str] = None
    family_id: Optional[str] = None  # for leakage-free splits (spec §8)

    def __post_init__(self):
        if not isinstance(self.patent_id, str) or not self.patent_id:
            raise SchemaError("patent_id must be a non-empty string")
        for n in ("title", "abstract"):
            if not isinstance(getattr(self, n), str):
                raise SchemaError(f"{n} must be a string")
        if not isinstance(self.figure_descriptions, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in self.figure_descriptions.items()
        ):
            raise SchemaError("figure_descriptions must be Dict[str,str]")
        if not (self.title.strip() or self.abstract.strip()
                or any(v.strip() for v in self.figure_descriptions.values())):
            raise SchemaError("patent has no text at all")
        if self.label is not None and self.label not in LABELS:
            raise SchemaError(f"label must be one of {LABELS}, got {self.label!r}")

    def classifier_input(self) -> Tuple[str, str, Dict[str, str]]:
        """x_p = (t, a, v). Deliberately excludes company and label."""
        return self.title, self.abstract, dict(self.figure_descriptions)

    @staticmethod
    def from_dict(d: dict) -> "Patent":
        if not isinstance(d, dict):
            raise SchemaError("patent record must be a dict")
        missing = [k for k in ("patent_id", "title", "abstract") if k not in d]
        if missing:
            raise SchemaError(f"missing fields: {missing}")
        return Patent(
            patent_id=d["patent_id"], title=d["title"], abstract=d["abstract"],
            figure_descriptions=d.get("figure_descriptions") or {},
            company=d.get("company"), label=d.get("label"), family_id=d.get("family_id"),
        )


@dataclass(frozen=True)
class EvidenceRecord:
    """d_i: provenance record carried to the organisational stage (spec §6)."""
    concept: str
    patent_id: str
    source: SourceRef
    score: float
    uncertainty: float
    evidence_type: EvidenceType
    cue_id: Optional[str] = None


def evidence_from_annotation(ann: Annotation) -> List[EvidenceRecord]:
    """One record per cue; score = cue confidence (the spec leaves 'scor' undefined)."""
    return [
        EvidenceRecord(c.attribute, ann.patent_id, c.source, c.confidence, c.uncertainty,
                       c.evidence_type, c.cue_id)
        for c in ann.cues
    ]
