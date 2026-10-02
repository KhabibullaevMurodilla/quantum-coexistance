"""
_packing.py
===========
Shared greedy bottleneck-respecting packing used by fifo.py and greedy.py.
Not a scheduler itself -- the leading underscore marks it as an internal
helper the two simplest policies both build on.
"""

from __future__ import annotations

from qcs.sessions import Session


def pack_in_order(
    ordered_sessions: list[Session],
    paths: dict[str, list[frozenset]],
    link_capacity: dict[frozenset, float],
) -> dict[str, float]:
    """Walk sessions in the given order; give each as much of its requested
    rate as the tightest (most depleted) link on its path still has left,
    then deduct that allocation from every link on the path."""
    remaining = dict(link_capacity)
    allocation: dict[str, float] = {}

    for s in ordered_sessions:
        path = paths.get(s.session_id, [])
        if not path:
            allocation[s.session_id] = 0.0
            continue
        bottleneck = min(remaining.get(lk, 0.0) for lk in path)
        granted = max(0.0, min(s.rate_request_ebit_s, bottleneck))
        allocation[s.session_id] = granted
        for lk in path:
            remaining[lk] = remaining.get(lk, 0.0) - granted

    return allocation
