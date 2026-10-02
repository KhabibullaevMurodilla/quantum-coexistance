"""
greedy.py
=========
Priority-weighted greedy packing: sessions are served in descending order
of (priority_weight / path hop-count), so a high-priority session with a
short route is favored over a low-priority one hogging a long multi-hop
path. Still centralized and still a single pass -- no iterative
re-optimization -- which is what distinguishes it from `optimal` and
`proportional_fair`.
"""

from __future__ import annotations

from qcs.sessions import Session

from ._packing import pack_in_order


def schedule(
    sessions: list[Session],
    paths: dict[str, list[frozenset]],
    link_capacity: dict[frozenset, float],
) -> dict[str, float]:
    def value_density(s: Session) -> float:
        hops = max(1, len(paths.get(s.session_id, [])))
        return s.priority_weight / hops

    ordered = sorted(sessions, key=value_density, reverse=True)
    return pack_in_order(ordered, paths, link_capacity)
