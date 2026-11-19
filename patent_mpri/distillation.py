"""Teacher/student interfaces and the distillation terms that ARE fully specified (spec §3-4, §9).

Implemented:   L_CE, L_KD^cls = tau^2 * sum_p KL(pi^T_{p,tau} || pi_hat_{p,tau}),
               pi_hat = softmax(W_y z + b_y), and the weighted sum L_student.
NOT implemented (spec gives no operational definition): L_span, L_rel, L_lattice, L_org,
               encoders f_theta, g_phi, F_psi, teacher T_omega. They are exposed as
               Protocols / required callables; nothing is trained in this package.
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Protocol, Sequence

from .schema import Annotation, Patent
from .scoring import stable_softmax


class Teacher(Protocol):
    """T_omega: x_p -> Q_p (cues + relations). Needs an external LLM; not provided."""
    def annotate(self, patent: Patent) -> Annotation: ...


class Student(Protocol):
    """S_eta: x_p -> (pi_hat, Q_hat, R_hat). Requires training data + a model; not provided."""
    def annotate(self, patent: Patent) -> Annotation: ...


class PrecomputedAnnotator:
    """Looks up offline-generated annotations (spec: Q_p 'can be generated offline')."""

    def __init__(self, annotations: Sequence[Annotation]):
        self._by_id = {a.patent_id: a for a in annotations}

    def annotate(self, patent: Patent) -> Annotation:
        try:
            return self._by_id[patent.patent_id]
        except KeyError:
            raise KeyError(f"no precomputed annotation for {patent.patent_id!r}") from None


def affine_softmax(W: Sequence[Sequence[float]], z: Sequence[float], b: Sequence[float],
                   temperature: float = 1.0) -> list:
    """pi_hat = softmax(W_y z + b_y) (/tau when tau != 1)."""
    logits = [sum(wi * zi for wi, zi in zip(row, z)) + bi for row, bi in zip(W, b)]
    return stable_softmax(logits, temperature)


def cross_entropy(probs: Sequence[float], target_index: int, eps: float = 1e-12) -> float:
    """L_CE for one patent: -log pi_hat[y]."""
    return -math.log(max(probs[target_index], eps))


def kl_divergence(p: Sequence[float], q: Sequence[float], eps: float = 1e-12) -> float:
    """D_KL(p || q) with 0 log 0 = 0; q clipped at eps to avoid infinities."""
    return sum(pi * math.log(pi / max(qi, eps)) for pi, qi in zip(p, q) if pi > 0)


def kd_loss(teacher_logits_or_probs: Sequence[Sequence[float]],
            student_logits: Sequence[Sequence[float]], tau: float,
            teacher_is_calibrated: bool, teacher_given_as_logits: bool = False) -> float:
    """L_KD^cls = tau^2 * sum_p KL(pi^T_{p,tau} || pi_hat_{p,tau}).

    Spec: only defined when the teacher yields a *calibrated* distribution; otherwise omit.
    A teacher distribution given as probabilities is tempered via p^(1/tau), renormalised
    (equivalent to softmax(log p / tau))."""
    if not teacher_is_calibrated:
        raise ValueError("L_KD^cls requires a calibrated teacher distribution (spec §4); omit the term")
    if tau <= 0:
        raise ValueError("tau must be > 0")
    total = 0.0
    for t, s in zip(teacher_logits_or_probs, student_logits):
        if teacher_given_as_logits:
            pt = stable_softmax(t, tau)
        else:
            pt = stable_softmax([math.log(max(x, 1e-12)) for x in t], tau)
        total += kl_divergence(pt, stable_softmax(s, tau))
    return tau ** 2 * total


def student_loss(l_ce: float, lam_cls: float,
                 l_kd: Optional[float] = None, lam_kd: float = 0.0,
                 l_span: Optional[float] = None, lam_span: float = 0.0,
                 l_rel: Optional[float] = None, lam_rel: float = 0.0) -> float:
    """L_student = lam_cls L_CE + lam_KD L_KD + lam_span L_span + lam_rel L_rel.
    A term with non-zero weight but no value raises: no loss is invented."""
    total = lam_cls * l_ce
    for name, v, lam in (("KD", l_kd, lam_kd), ("span", l_span, lam_span), ("rel", l_rel, lam_rel)):
        if lam != 0.0:
            if v is None:
                raise NotImplementedError(
                    f"L_{name} has weight {lam} but no value supplied; the spec does not define "
                    f"how to compute it without annotations/teacher outputs")
            total += lam * v
    return total


def lattice_loss(*_a, **_k):
    raise NotImplementedError("L_lattice is undefined in the spec (§9: 'must be defined concretely')")


def org_loss(*_a, **_k):
    raise NotImplementedError("L_org needs reference organisational graphs, which are unavailable")
