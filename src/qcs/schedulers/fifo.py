"""
fifo.py
=======
First-come-first-served greedy packing, in session-list order. The naive
baseline every other policy should beat: it ignores priority, fairness,
and congestion entirely.
"""

from __future__ import annotations

from qcs.sessions import Session

from ._packing import pack_in_order


def schedule(
    sessions: list[Session],
    paths: dict[str, list[frozenset]],
    link_capacity: dict[frozenset, float],
) -> dict[str, float]:
    return pack_in_order(sessions, paths, link_capacity)
