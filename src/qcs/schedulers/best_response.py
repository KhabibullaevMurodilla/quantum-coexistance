"""
best_response.py
==================
Selfish, uncoordinated rate choice under congestion pricing -- a
Nash/Wardrop-equilibrium-style policy. Unlike primal_dual.py (whose prices
are set by a hard capacity constraint that the whole system is jointly
satisfying), here each session just reacts to the *congestion cost* its
current load is causing on each link of its path, with no planner enforcing
a global capacity bound and no session accounting for how its own choice
raises the cost everyone else faces -- the standard selfish-routing /
congestion-game setup. Iterating best responses settles into a Nash
equilibrium of that game.

Comparing this policy's aggregate welfare against optimal.py's coordinated
social optimum is exactly the "price of decentralization" (also called the
price of anarchy in the congestion-games literature) this project set out
to quantify: how much utility is lost to selfish, uncoordinated play versus
what a central planner with full information could achieve.
"""

from __future__ import annotations

import numpy as np

from qcs.sessions import Session, UtilityKind

_ROUNDS = 150
_CONGESTION_COEFF = 2.0  # scales how sharply cost grows with load/capacity


def _congestion_price(load: float, capacity: float) -> float:
    """Marginal congestion cost per extra unit of rate on a link already
    carrying `load` out of `capacity`: cost(load) = coeff*(load/cap)^2, so
    marginal cost = 2*coeff*load/cap^2. Soft and always finite (no hard
    cutoff), which is what makes this a *selfish* game rather than a
    constrained optimization -- a session can always push more rate
    through, it just gets costlier."""
    cap = max(capacity, 1e-6)
    return 2.0 * _CONGESTION_COEFF * load / (cap ** 2)


def _best_response_rate(s: Session, price_sum: float) -> float:
    price_sum = max(price_sum, 1e-6)

    if s.delivered_fidelity < s.fidelity_floor:
        return 0.0

    if s.utility_kind == UtilityKind.LOG_RATE:
        x = s.priority_weight / price_sum - 1.0
    elif s.utility_kind == UtilityKind.NEGATIVITY:
        margin = max(0.0, s.delivered_fidelity - s.fidelity_floor) / max(1e-9, 1.0 - s.fidelity_floor)
        coeff = s.priority_weight * (margin ** 2)
        x = (coeff / (2.0 * price_sum)) ** 2 if coeff > 0 else 0.0
    else:
        x = s.rate_request_ebit_s if s.priority_weight > price_sum else 0.0

    return float(np.clip(x, 0.0, s.rate_request_ebit_s))


def schedule(
    sessions: list[Session],
    paths: dict[str, list[frozenset]],
    link_capacity: dict[frozenset, float],
) -> dict[str, float]:
    routed = [s for s in sessions if paths.get(s.session_id)]
    allocation = {s.session_id: 0.0 for s in sessions}

    for _ in range(_ROUNDS):
        load: dict[frozenset, float] = {lk: 0.0 for lk in link_capacity}
        for s in routed:
            for lk in paths[s.session_id]:
                load[lk] += allocation[s.session_id]

        # Asynchronous-style sweep: each session best-responds to the load
        # left by everyone else (its own prior contribution removed first),
        # which is the standard way to avoid a session "congesting itself".
        for s in routed:
            path = paths[s.session_id]
            others_load = {
                lk: max(0.0, load.get(lk, 0.0) - allocation[s.session_id]) for lk in path
            }
            price_sum = sum(
                _congestion_price(others_load[lk], link_capacity.get(lk, 1e-6)) for lk in path
            )
            new_rate = _best_response_rate(s, price_sum)
            for lk in path:
                load[lk] = load[lk] - allocation[s.session_id] + new_rate
            allocation[s.session_id] = new_rate

    # Selfish play has no planner enforcing the hard capacity bound, so a
    # link can still end up oversubscribed at equilibrium; rescale each
    # link's contending sessions down proportionally if so, which is the
    # natural "the fiber only has so many ebits/s" physical reality
    # reasserting itself after the game settles.
    final_load: dict[frozenset, float] = {lk: 0.0 for lk in link_capacity}
    for s in routed:
        for lk in paths[s.session_id]:
            final_load[lk] += allocation[s.session_id]

    scale: dict[frozenset, float] = {}
    for lk, cap in link_capacity.items():
        scale[lk] = min(1.0, cap / final_load[lk]) if final_load[lk] > cap else 1.0

    for s in routed:
        path = paths[s.session_id]
        if path:
            allocation[s.session_id] *= min(scale[lk] for lk in path)

    return allocation
