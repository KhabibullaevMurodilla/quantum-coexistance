"""
_convex.py
==========
Shared convex-program plumbing for optimal.py and proportional_fair.py.
Both solve the same shape of problem -- maximize a sum of concave
per-session objectives subject to link-capacity constraints -- and differ
only in which objective each session gets, so the constraint-building code
(decision variables, capacity constraints, request caps) lives here once.
"""

from __future__ import annotations

from collections.abc import Callable

import cvxpy as cp

from qcs.sessions import Session


def solve_rate_allocation(
    sessions: list[Session],
    paths: dict[str, list[frozenset]],
    link_capacity: dict[frozenset, float],
    per_session_objective: Callable[[Session, cp.Variable], cp.Expression],
) -> dict[str, float]:
    routed = [s for s in sessions if paths.get(s.session_id)]
    if not routed:
        return {s.session_id: 0.0 for s in sessions}

    x = {s.session_id: cp.Variable(nonneg=True) for s in routed}

    constraints = [x[s.session_id] <= s.rate_request_ebit_s for s in routed]

    links_used: dict[frozenset, list[str]] = {}
    for s in routed:
        for lk in paths[s.session_id]:
            links_used.setdefault(lk, []).append(s.session_id)

    for lk, sids in links_used.items():
        cap = link_capacity.get(lk, 0.0)
        constraints.append(cp.sum([x[sid] for sid in sids]) <= cap)

    objective = cp.Maximize(cp.sum([per_session_objective(s, x[s.session_id]) for s in routed]))
    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(solver=cp.ECOS)
    except cp.error.SolverError:
        problem.solve(solver=cp.SCS)

    allocation = {s.session_id: 0.0 for s in sessions}
    if x[routed[0].session_id].value is not None:
        for s in routed:
            allocation[s.session_id] = max(0.0, float(x[s.session_id].value))
    return allocation
