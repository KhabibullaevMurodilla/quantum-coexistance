"""
sessions.py
===========
Session (application request) generation.

A "session" is one end-to-end entanglement request between two nodes, with
its own rate target, fidelity floor, and utility shape. Real deployments
would see a mix of application classes with very different requirements --
this module generates a heterogeneous mix of three generic archetypes so
scheduling policies have something non-trivial to arbitrate between:

  - QKD-style: wants high rate, tolerates a comparatively low fidelity
    floor (post-processing / privacy amplification can absorb some noise).
  - Distributed-computation-style ("DQC"): wants a high fidelity floor
    (a noisy Bell pair corrupts the computation outright) at modest rate.
  - Sensing-style: wants very high fidelity at low rate (a small number of
    high-quality entangled probes beats many noisy ones).

These three archetypes are standard, qualitative descriptions used across
the quantum-networking applications literature in general -- they are not
drawn from, or attributed to, any single paper.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum

import networkx as nx
import numpy as np

from .utility import UtilityKind


class SessionKind(Enum):
    QKD = "qkd"
    DQC = "dqc"
    SENSING = "sensing"


# (fidelity_floor_range, rate_request_range_ebit_s, default utility kind, weight_range)
_ARCHETYPES = {
    SessionKind.QKD: dict(
        fidelity_floor=(0.80, 0.88),
        rate_request_ebit_s=(20.0, 80.0),
        utility_kind=UtilityKind.LOG_RATE,
        priority_weight=(0.8, 1.2),
    ),
    SessionKind.DQC: dict(
        fidelity_floor=(0.92, 0.97),
        rate_request_ebit_s=(2.0, 15.0),
        utility_kind=UtilityKind.LINEAR,
        priority_weight=(1.0, 1.8),
    ),
    SessionKind.SENSING: dict(
        fidelity_floor=(0.95, 0.99),
        rate_request_ebit_s=(0.5, 4.0),
        utility_kind=UtilityKind.NEGATIVITY,
        priority_weight=(1.2, 2.0),
    ),
}


@dataclass
class Session:
    session_id: str
    src: str
    dst: str
    kind: SessionKind
    fidelity_floor: float
    rate_request_ebit_s: float
    utility_kind: UtilityKind
    priority_weight: float = 1.0
    # Filled in by the simulation loop each timestep -- not part of the
    # request itself, just a convenient place to record what was delivered.
    delivered_rate_ebit_s: float = field(default=0.0, repr=False)
    delivered_fidelity: float = field(default=0.0, repr=False)


def generate_sessions(
    graph: nx.Graph,
    n_sessions: int = 12,
    mix: dict[SessionKind, float] | None = None,
    seed: int = 0,
) -> list[Session]:
    """Generate a random mix of sessions between distinct node pairs in
    `graph`. `mix` gives the relative proportion of each archetype
    (defaults to roughly even thirds); node pairs are sampled without
    replacement from all simple paths' endpoints so every session has at
    least one route to be scheduled over.
    """
    rng = np.random.default_rng(seed)
    mix = mix or {SessionKind.QKD: 1.0, SessionKind.DQC: 1.0, SessionKind.SENSING: 1.0}
    kinds = list(mix.keys())
    probs = np.array([mix[k] for k in kinds], dtype=float)
    probs /= probs.sum()

    nodes = list(graph.nodes)
    if len(nodes) < 2:
        raise ValueError("Graph needs at least 2 nodes to generate sessions")

    all_pairs = [(a, b) for a, b in itertools.permutations(nodes, 2) if nx.has_path(graph, a, b)]
    if not all_pairs:
        raise ValueError("No connected node pairs available for session endpoints")

    chosen_pairs_idx = rng.choice(len(all_pairs), size=min(n_sessions, len(all_pairs) * 4), replace=True)

    sessions: list[Session] = []
    for i, pair_idx in enumerate(chosen_pairs_idx[:n_sessions]):
        src, dst = all_pairs[pair_idx % len(all_pairs)]
        kind = kinds[rng.choice(len(kinds), p=probs)]
        spec = _ARCHETYPES[kind]

        f_lo, f_hi = spec["fidelity_floor"]
        r_lo, r_hi = spec["rate_request_ebit_s"]
        w_lo, w_hi = spec["priority_weight"]

        sessions.append(
            Session(
                session_id=f"S{i:03d}_{kind.value}",
                src=src,
                dst=dst,
                kind=kind,
                fidelity_floor=float(rng.uniform(f_lo, f_hi)),
                rate_request_ebit_s=float(rng.uniform(r_lo, r_hi)),
                utility_kind=spec["utility_kind"],
                priority_weight=float(rng.uniform(w_lo, w_hi)),
            )
        )
    return sessions
