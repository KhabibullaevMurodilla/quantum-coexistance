"""
registry.py
============
Name -> schedule() function lookup, so the simulation loop and dashboard
can select a policy by string (e.g. from a CLI flag or a dropdown) instead
of importing every module by hand.
"""

from __future__ import annotations

from . import best_response, fifo, greedy, learned, optimal, primal_dual, proportional_fair

SCHEDULERS = {
    "fifo": fifo.schedule,
    "greedy": greedy.schedule,
    "proportional_fair": proportional_fair.schedule,
    "optimal": optimal.schedule,
    "primal_dual": primal_dual.schedule,
    "best_response": best_response.schedule,
    "learned": learned.schedule,
}

SCHEDULER_NAMES = list(SCHEDULERS.keys())
