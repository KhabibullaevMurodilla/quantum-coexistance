"""
primal_dual.py
===============
Distributed dual-decomposition (shadow-price) rate control -- the classic
Kelly / Low-Lapsley style algorithm: no central solver ever sees the whole
network. Instead, each link maintains its own congestion price, each
session reacts only to the sum of prices along its own path (computing its
own utility-maximizing rate in closed form), and the two sides iterate
until prices stop moving. Under the standard concave-utility / convex-
constraint conditions this converges to the same allocation optimal.py
finds centrally -- the point is to reach it *without* any node needing
global knowledge of the network.
"""

from __future__ import annotations

import numpy as np

from qcs.sessions import Session, UtilityKind

_ITERATIONS = 300
_PRICE_STEP = 0.05
_AVERAGE_LAST = 50  # average the tail of iterates to damp oscillation


def _best_response_rate(s: Session, price_sum: float) -> float:
    """Closed-form rate maximizing (utility(x) - price_sum * x) for each
    utility shape, clipped to [0, rate_request]."""
    price_sum = max(price_sum, 1e-6)

    if s.delivered_fidelity < s.fidelity_floor:
        return 0.0

    if s.utility_kind == UtilityKind.LOG_RATE:
        # d/dx [w*log(1+x)] = w/(1+x) = price_sum  =>  x = w/price_sum - 1
        x = s.priority_weight / price_sum - 1.0
    elif s.utility_kind == UtilityKind.NEGATIVITY:
        margin = max(0.0, s.delivered_fidelity - s.fidelity_floor) / max(1e-9, 1.0 - s.fidelity_floor)
        coeff = s.priority_weight * (margin ** 2)
        # d/dx [coeff*sqrt(x)] = coeff/(2*sqrt(x)) = price_sum => x = (coeff/(2*price_sum))^2
        x = (coeff / (2.0 * price_sum)) ** 2 if coeff > 0 else 0.0
    else:
        # LINEAR: marginal utility is the constant priority_weight -- bang-bang
        # against price: take the full request if it's worth more than its price.
        x = s.rate_request_ebit_s if s.priority_weight > price_sum else 0.0

    return float(np.clip(x, 0.0, s.rate_request_ebit_s))


def schedule(
    sessions: list[Session],
    paths: dict[str, list[frozenset]],
    link_capacity: dict[frozenset, float],
) -> dict[str, float]:
    routed = [s for s in sessions if paths.get(s.session_id)]
    prices: dict[frozenset, float] = {lk: 0.1 for lk in link_capacity}

    allocation = {s.session_id: 0.0 for s in sessions}
    tail_sum = {s.session_id: 0.0 for s in sessions}
    tail_count = 0

    for it in range(_ITERATIONS):
        for s in routed:
            path = paths[s.session_id]
            price_sum = sum(prices.get(lk, 0.0) for lk in path)
            allocation[s.session_id] = _best_response_rate(s, price_sum)

        load: dict[frozenset, float] = {lk: 0.0 for lk in link_capacity}
        for s in routed:
            for lk in paths[s.session_id]:
                load[lk] += allocation[s.session_id]

        for lk, cap in link_capacity.items():
            gradient = load.get(lk, 0.0) - cap
            prices[lk] = max(0.0, prices[lk] + _PRICE_STEP * gradient)

        if it >= _ITERATIONS - _AVERAGE_LAST:
            tail_count += 1
            for s in routed:
                tail_sum[s.session_id] += allocation[s.session_id]

    # The bang-bang closed form for LINEAR-utility sessions (and, to a
    # lesser extent, sub-gradient price updates in general) never quite
    # settles to a fixed point -- it oscillates around the optimum. The
    # standard fix is to report the time-average of the last iterations
    # rather than the final (possibly over-shooting) one.
    averaged = {
        s.session_id: tail_sum[s.session_id] / max(1, tail_count) for s in routed
    }

    # A sub-gradient price process can still leave a link briefly
    # oversubscribed in that average; the physical fiber doesn't care how
    # the price iteration behaved, so rescale any still-violated link's
    # contending sessions down proportionally -- the same final feasibility
    # projection best_response.py uses for the same underlying reason.
    final_load: dict[frozenset, float] = {lk: 0.0 for lk in link_capacity}
    for s in routed:
        for lk in paths[s.session_id]:
            final_load[lk] += averaged[s.session_id]

    scale: dict[frozenset, float] = {}
    for lk, cap in link_capacity.items():
        scale[lk] = min(1.0, cap / final_load[lk]) if final_load[lk] > cap else 1.0

    result = {s.session_id: 0.0 for s in sessions}
    for s in routed:
        path = paths[s.session_id]
        factor = min((scale[lk] for lk in path), default=1.0)
        result[s.session_id] = averaged[s.session_id] * factor

    return result
