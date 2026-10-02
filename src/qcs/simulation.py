"""
simulation.py
===============
Discrete time-slot simulation loop: drives any scheduler (by name, via
qcs.schedulers.registry) over a fixed topology and session set, under a
classical-traffic coexistence process, and records per-slot metrics.

Each slot:
  1. Classical traffic load is sampled for this slot (one value per link,
     currently shared across all links for simplicity -- see
     `per_link_load` to vary that).
  2. Every link's clean Werner parameter is degraded by that load via
     qcs.coexistence.apply_coexistence_noise, giving this slot's capacity
     (qcs.network.Link.rate) and, combined with each session's fixed
     route, this slot's delivered end-to-end fidelity.
  3. A session whose route can't meet its own fidelity floor this slot is
     given an empty path for this slot only -- it contributes (and can
     receive) nothing, matching qcs.utility.session_utility's hard floor
     rule, and is automatically excluded by every scheduler without any
     scheduler needing its own feasibility check.
  4. The chosen scheduler allocates rate to the (now feasibility-filtered)
     sessions; qcs.utility.session_utility scores what was actually
     delivered; Jain's fairness index summarizes the spread.

This is deliberately the same loop for all seven scheduler policies, which
is what makes the policies comparable at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from .coexistence import CoexistenceConfig, apply_coexistence_noise, classical_load_profile
from .network import end_to_end_fidelity, link_key, shortest_path_links
from .schedulers.registry import SCHEDULERS
from .sessions import Session
from .utility import UtilityWeights, jains_fairness_index, session_utility


@dataclass
class SlotResult:
    t: int
    total_utility: float
    total_delivered_rate: float
    jain_fairness: float
    feasible_sessions: int
    violated_sessions: int
    mean_classical_load: float
    per_session_rate: dict[str, float] = field(default_factory=dict)
    per_session_utility: dict[str, float] = field(default_factory=dict)
    per_session_fidelity: dict[str, float] = field(default_factory=dict)


def run_simulation(
    graph: nx.Graph,
    sessions: list[Session],
    scheduler: str,
    n_slots: int = 100,
    clean_w: float = 0.96,
    coexistence_cfg: CoexistenceConfig | None = None,
    load_kind: str = "diurnal",
    utility_weights: UtilityWeights | None = None,
    seed: int = 0,
    scheduler_kwargs: dict | None = None,
) -> list[SlotResult]:
    if scheduler not in SCHEDULERS:
        raise ValueError(f"Unknown scheduler {scheduler!r}; choose from {list(SCHEDULERS)}")
    schedule_fn = SCHEDULERS[scheduler]
    scheduler_kwargs = scheduler_kwargs or {}
    cfg = coexistence_cfg or CoexistenceConfig()
    weights = utility_weights or UtilityWeights()

    link_keys = [link_key(u, v) for u, v in graph.edges()]
    link_objs = {link_key(u, v): data["link"] for u, v, data in graph.edges(data=True)}

    # Fixed routes: computed once since the topology doesn't change
    # mid-run, only each link's noise level does.
    static_paths: dict[str, list[frozenset]] = {}
    for s in sessions:
        try:
            static_paths[s.session_id] = shortest_path_links(graph, s.src, s.dst)
        except nx.NetworkXNoPath:
            static_paths[s.session_id] = []

    classical_load = classical_load_profile(n_slots, kind=load_kind, seed=seed)

    results: list[SlotResult] = []
    for t in range(n_slots):
        load_t = float(classical_load[t])

        w_eff: dict[frozenset, float] = {}
        link_capacity: dict[frozenset, float] = {}
        for lk in link_keys:
            w_eff[lk] = apply_coexistence_noise(clean_w, load_t, cfg)
            link_capacity[lk] = link_objs[lk].rate(w_eff[lk])

        paths_t: dict[str, list[frozenset]] = {}
        for s in sessions:
            route = static_paths[s.session_id]
            if not route:
                paths_t[s.session_id] = []
                s.delivered_fidelity = 0.0
                continue
            fidelity = end_to_end_fidelity([w_eff[lk] for lk in route])
            s.delivered_fidelity = fidelity
            paths_t[s.session_id] = route if fidelity >= s.fidelity_floor else []

        allocation = schedule_fn(sessions, paths_t, link_capacity, **scheduler_kwargs)

        per_session_rate: dict[str, float] = {}
        per_session_utility: dict[str, float] = {}
        per_session_fidelity: dict[str, float] = {}
        feasible = 0
        violated = 0

        for s in sessions:
            rate = allocation.get(s.session_id, 0.0)
            s.delivered_rate_ebit_s = rate
            route = static_paths[s.session_id]
            latency = sum(link_objs[lk].latency(w_eff[lk]) for lk in route) if route else 0.0

            u = session_utility(
                s.utility_kind,
                fidelity=s.delivered_fidelity,
                rate_ebit_s=rate,
                latency_s=latency,
                fidelity_floor=s.fidelity_floor,
                weights=weights,
            )
            per_session_rate[s.session_id] = rate
            per_session_utility[s.session_id] = u
            per_session_fidelity[s.session_id] = s.delivered_fidelity

            if route and s.delivered_fidelity >= s.fidelity_floor:
                feasible += 1
            elif route:
                violated += 1

        results.append(
            SlotResult(
                t=t,
                total_utility=float(sum(per_session_utility.values())),
                total_delivered_rate=float(sum(per_session_rate.values())),
                jain_fairness=jains_fairness_index(list(per_session_rate.values())),
                feasible_sessions=feasible,
                violated_sessions=violated,
                mean_classical_load=load_t,
                per_session_rate=per_session_rate,
                per_session_utility=per_session_utility,
                per_session_fidelity=per_session_fidelity,
            )
        )

    return results


def summarize(results: list[SlotResult]) -> dict:
    """Collapse a run's per-slot results into the handful of headline
    numbers used in reports.py and the dashboard."""
    if not results:
        return {}
    return {
        "mean_total_utility": float(np.mean([r.total_utility for r in results])),
        "mean_total_rate": float(np.mean([r.total_delivered_rate for r in results])),
        "mean_jain_fairness": float(np.mean([r.jain_fairness for r in results])),
        "mean_feasible_sessions": float(np.mean([r.feasible_sessions for r in results])),
        "mean_violated_sessions": float(np.mean([r.violated_sessions for r in results])),
    }
