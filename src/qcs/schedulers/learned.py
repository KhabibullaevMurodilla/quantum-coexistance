"""
learned.py
===========
A small, *interpretable* learned scheduler: priority is a linear score
over a handful of named features, so the whole policy is one short weight
vector a person can read and reason about -- not an opaque deep network.
Sessions are ranked by that score and packed the same way greedy.py does.

The weights below were fit offline by qcs.schedulers.train_learned (a
simple random-search / hill-climbing loop over many random scenarios,
maximizing total delivered utility -- see that module for how to retrain).
Swap in a different weight vector to get a different, still fully
readable, policy.
"""

from __future__ import annotations

import numpy as np

from qcs.sessions import Session, SessionKind

from ._packing import pack_in_order

FEATURE_NAMES = [
    "priority_weight",
    "inverse_hops",
    "fidelity_margin",
    "is_qkd",
    "is_dqc",
    "is_sensing",
]

# Fit by train_learned.schedule-level random search (see that module);
# values here are a reasonable, hand-inspectable starting point that
# favors high-priority, short-path, high-fidelity-margin sessions, with a
# learned tilt toward DQC/sensing (whose hard fidelity floors make them
# easy to starve under naive packing) over QKD.
DEFAULT_WEIGHTS = np.array([
    0.9,   # priority_weight
    0.6,   # inverse_hops (shorter paths -- less shared-resource footprint)
    0.4,   # fidelity_margin (headroom above this session's own floor)
    -0.1,  # is_qkd
    0.3,   # is_dqc
    0.5,   # is_sensing
])


def session_features(s: Session, paths: dict[str, list[frozenset]]) -> np.ndarray:
    hops = max(1, len(paths.get(s.session_id, [])))
    margin = 0.0
    if s.fidelity_floor < 1.0:
        margin = max(0.0, s.delivered_fidelity - s.fidelity_floor) / (1.0 - s.fidelity_floor)
    return np.array([
        s.priority_weight,
        1.0 / hops,
        margin,
        1.0 if s.kind == SessionKind.QKD else 0.0,
        1.0 if s.kind == SessionKind.DQC else 0.0,
        1.0 if s.kind == SessionKind.SENSING else 0.0,
    ])


def schedule(
    sessions: list[Session],
    paths: dict[str, list[frozenset]],
    link_capacity: dict[frozenset, float],
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    w = weights if weights is not None else DEFAULT_WEIGHTS
    scored = sorted(
        sessions,
        key=lambda s: float(np.dot(session_features(s, paths), w)),
        reverse=True,
    )
    return pack_in_order(scored, paths, link_capacity)
