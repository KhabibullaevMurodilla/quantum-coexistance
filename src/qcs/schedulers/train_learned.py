"""
train_learned.py
==================
Fits learned.py's weight vector by random-search hill-climbing: repeatedly
perturb the current best weights, run a short simulation under each
candidate, and keep whichever version produced higher mean total utility
across a handful of random scenarios. No gradients, no neural network --
the whole "model" being trained is the same six-number vector learned.py
already documents, which is the point: the policy stays as readable after
training as before it.

Run directly (`python -m qcs.schedulers.train_learned`) to reproduce or
improve on the committed DEFAULT_WEIGHTS.
"""

from __future__ import annotations

import copy

import numpy as np

from ..coexistence import CoexistenceConfig
from ..network import build_topology
from ..sessions import generate_sessions
from ..simulation import run_simulation, summarize
from . import learned


def _evaluate(weights: np.ndarray, n_scenarios: int, seed: int) -> float:
    scores = []
    for i in range(n_scenarios):
        g = build_topology("metro_mesh", n=8, seed=seed + i)
        sessions = generate_sessions(g, n_sessions=16, seed=100 + seed + i)
        results = run_simulation(
            g,
            copy.deepcopy(sessions),
            scheduler="learned",
            n_slots=20,
            clean_w=0.96,
            coexistence_cfg=CoexistenceConfig(),
            seed=seed + i,
            scheduler_kwargs={"weights": weights},
        )
        scores.append(summarize(results)["mean_total_utility"])
    return float(np.mean(scores))


def train(
    n_rounds: int = 60,
    n_scenarios: int = 4,
    step: float = 0.25,
    seed: int = 0,
    verbose: bool = True,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    best_weights = learned.DEFAULT_WEIGHTS.copy()
    best_score = _evaluate(best_weights, n_scenarios, seed)

    for r in range(n_rounds):
        candidate = best_weights + rng.normal(0, step, size=best_weights.shape)
        score = _evaluate(candidate, n_scenarios, seed)
        if score > best_score:
            best_weights, best_score = candidate, score
            if verbose:
                pairs = ", ".join(
                    f"{name}={w:+.2f}" for name, w in zip(learned.FEATURE_NAMES, best_weights)
                )
                print(f"[round {r:3d}] new best mean_utility={best_score:.3f}  ({pairs})")

    return best_weights


if __name__ == "__main__":
    w = train()
    print("\nFinal learned weights:")
    for name, value in zip(learned.FEATURE_NAMES, w):
        print(f"  {name:18s} {value:+.3f}")
