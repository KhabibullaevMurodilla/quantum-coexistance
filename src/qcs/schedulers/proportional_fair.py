"""
proportional_fair.py
=====================
Centralized proportional-fair allocation: maximize sum_s w_s * log(x_s),
the standard Kelly-style logarithmic utility that produces the classic
"proportional fairness" allocation -- every session gets a share roughly
proportional to its priority weight, with diminishing returns preventing
any one session from monopolizing a link the way a pure throughput-max
objective would. Solved centrally here as a convex program; primal_dual.py
reaches (an approximation of) the same allocation without central
coordination.
"""

from __future__ import annotations

import cvxpy as cp

from qcs.sessions import Session

from ._convex import solve_rate_allocation

_EPS = 1e-3  # avoids log(0); negligible next to realistic rate requests


def _objective(s: Session, x: cp.Variable) -> cp.Expression:
    return s.priority_weight * cp.log(x + _EPS)


def schedule(
    sessions: list[Session],
    paths: dict[str, list[frozenset]],
    link_capacity: dict[frozenset, float],
) -> dict[str, float]:
    return solve_rate_allocation(sessions, paths, link_capacity, _objective)
