"""
optimal.py
==========
Centralized social optimum: maximize the sum of each session's *actual*
utility (qcs.utility.session_utility's shape, not a generic log proxy),
given the fidelity each session's fixed path is currently delivering.
This is the benchmark every other policy is measured against -- "what's
the best any allocation could achieve with full information and full
coordination" -- which is exactly the role Project B's "centralized
optimum" needs to play against best_response.py's self-interested play.

Requires each session's `delivered_fidelity` to already be set (the
simulation loop computes this once per slot from the path's current,
possibly coexistence-degraded, Werner parameters -- fidelity is a
property of the *route*, not of how much rate gets allocated, in this
rate/fidelity-decoupled model). A session whose floor isn't met by that
fidelity is constrained to zero rate, matching session_utility's hard cutoff.
"""

from __future__ import annotations

import cvxpy as cp

from qcs.sessions import Session, UtilityKind

from ._convex import solve_rate_allocation

_EPS = 1e-6


def _objective(s: Session, x: cp.Variable) -> cp.Expression:
    if s.delivered_fidelity < s.fidelity_floor:
        # Infeasible under this route/noise -- contributes nothing, and
        # solve_rate_allocation's caller still respects x <= rate_request,
        # we just make sure a nonzero x buys no utility.
        return 0.0 * x

    if s.utility_kind == UtilityKind.LOG_RATE:
        return s.priority_weight * cp.log(1 + x)

    if s.utility_kind == UtilityKind.NEGATIVITY:
        margin = max(0.0, s.delivered_fidelity - s.fidelity_floor) / max(1e-9, 1.0 - s.fidelity_floor)
        return s.priority_weight * (margin ** 2) * cp.sqrt(x + _EPS)

    # LINEAR (and fallback): fidelity/latency terms are constants given a
    # fixed path, so only the rate term is a function of the decision
    # variable and matters for the optimizer.
    return s.priority_weight * x


def schedule(
    sessions: list[Session],
    paths: dict[str, list[frozenset]],
    link_capacity: dict[frozenset, float],
) -> dict[str, float]:
    return solve_rate_allocation(sessions, paths, link_capacity, _objective)
