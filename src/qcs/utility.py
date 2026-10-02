"""
utility.py
==========
Per-session utility functions.

A scheduling policy needs some scalar objective to optimize. The
network-utility-maximization (NUM) tradition models this as each session
(application) having its own utility function of the rate and fidelity it is
delivered, and the network-wide objective is some aggregate (usually the
sum, sometimes a fairness-weighted sum) of those per-session utilities. This
module collects a few generic, textbook-standard utility shapes so the
scheduler modules can stay agnostic to which one a given session uses.

None of these are tied to a specific published utility function -- they are
the standard building blocks (log utility for proportional fairness,
alpha-fairness, a coherence-time-gated piecewise penalty) used throughout
the broader network-utility-maximization and quantum-networking-scheduling
literature.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class UtilityKind(Enum):
    LINEAR = "linear"           # alpha*F_hat + beta*R_hat - gamma*L_hat
    LOG_RATE = "log_rate"       # proportional-fair-style log(rate), gated by a fidelity floor
    NEGATIVITY = "negativity"   # entanglement-negativity-style utility, favors high fidelity


@dataclass
class UtilityWeights:
    """Weights for the LINEAR utility kind. R_ref/L_ref are normalization
    scales (so sessions with very different raw rate/latency magnitudes can
    be combined into one additive objective without one dominating)."""
    alpha_fidelity: float = 1.0
    beta_rate: float = 1.0
    gamma_latency: float = 0.25
    rate_ref_ebit_s: float = 100.0
    latency_ref_s: float = 1.0e-3


def coherence_gate(fidelity: float, hold_time_s: float, coherence_time_s: float) -> float:
    """Multiplicative penalty in [0, 1] for holding an entangled pair past a
    memory's coherence time. Below the cutoff the state is usable at full
    value; past it, usefulness decays smoothly rather than cutting off
    sharply, matching the fact that decoherence is a continuous process, not
    a hard deadline.
    """
    if coherence_time_s <= 0:
        return 1.0 if hold_time_s <= 0 else 0.0
    ratio = hold_time_s / coherence_time_s
    return float(np.exp(-max(0.0, ratio)))


def session_utility(
    kind: UtilityKind,
    fidelity: float,
    rate_ebit_s: float,
    latency_s: float,
    fidelity_floor: float,
    weights: UtilityWeights | None = None,
) -> float:
    """Evaluate one session's utility given what it was actually delivered.

    A session delivered fidelity below its own fidelity_floor gets a hard
    zero -- an unusable entangled pair (e.g. for QKD key distillation, or for
    a distributed computation that needs a verified Bell pair) contributes
    nothing regardless of how much "rate" was nominally delivered, which is
    what makes the fidelity floor a real constraint rather than just another
    soft term in the objective.
    """
    if fidelity < fidelity_floor:
        return 0.0

    weights = weights or UtilityWeights()

    if kind == UtilityKind.LINEAR:
        f_hat = fidelity
        r_hat = min(1.0, rate_ebit_s / max(1e-9, weights.rate_ref_ebit_s))
        l_hat = min(1.0, latency_s / max(1e-9, weights.latency_ref_s))
        return (
            weights.alpha_fidelity * f_hat
            + weights.beta_rate * r_hat
            - weights.gamma_latency * l_hat
        )

    if kind == UtilityKind.LOG_RATE:
        # Proportional-fair-style: diminishing returns on rate, standard in
        # the NUM literature (Kelly-style log utility), gated by the
        # fidelity floor above.
        return float(np.log1p(max(0.0, rate_ebit_s)))

    if kind == UtilityKind.NEGATIVITY:
        # A simple monotone stand-in for entanglement negativity as a
        # fidelity-quality measure: rewards fidelity super-linearly above
        # the floor, reflecting that marginal fidelity gains near w=1 are
        # disproportionately valuable for applications doing further
        # distillation or verification.
        margin = max(0.0, fidelity - fidelity_floor) / max(1e-9, 1.0 - fidelity_floor)
        return float(margin ** 2) * max(0.0, rate_ebit_s) ** 0.5

    raise ValueError(f"Unknown utility kind: {kind!r}")


def jains_fairness_index(values: list[float]) -> float:
    """Standard fairness measure in [1/n, 1]; 1.0 is perfectly fair."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 1.0
    num = arr.sum() ** 2
    den = arr.size * np.sum(arr ** 2)
    if den <= 0:
        return 1.0
    return float(num / den)
