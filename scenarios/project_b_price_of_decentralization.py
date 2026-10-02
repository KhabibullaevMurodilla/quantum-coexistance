"""
Project B -- The price of decentralization.

Question: how much total utility is lost when scheduling decisions are
made by self-interested sessions playing a Nash/Wardrop-equilibrium-style
congestion game (best_response.py), instead of by a central planner
solving for the true social optimum (optimal.py) -- and does a
distributed-but-coordinated algorithm (primal_dual.py) recover most of
that gap without any central solver?

This script runs many random scenarios (varying topology, session mix,
and classical load) and reports, for each, the welfare gap:

    gap(policy) = (optimal - policy) / optimal

averaged across runs, which is this project's empirical answer to "what
does decentralization cost here." Run:
`python scenarios/project_b_price_of_decentralization.py`
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qcs.coexistence import CoexistenceConfig, Propagation  # noqa: E402
from qcs.network import build_topology  # noqa: E402
from qcs.sessions import generate_sessions  # noqa: E402
from qcs.simulation import run_simulation, summarize  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "reports"
OUT_DIR.mkdir(exist_ok=True)

POLICIES = ["optimal", "primal_dual", "best_response"]


def one_scenario(seed: int, n_slots: int = 40) -> dict:
    g = build_topology("metro_mesh", n=np.random.default_rng(seed).integers(6, 14), seed=seed)
    sessions = generate_sessions(g, n_sessions=int(np.random.default_rng(seed + 1).integers(10, 30)), seed=seed + 1)
    cfg = CoexistenceConfig(propagation=Propagation.COUNTER)

    utilities = {}
    for policy in POLICIES:
        sess_copy = copy.deepcopy(sessions)
        results = run_simulation(
            g, sess_copy, scheduler=policy, n_slots=n_slots,
            clean_w=0.96, coexistence_cfg=cfg, load_kind="diurnal", seed=seed,
        )
        utilities[policy] = summarize(results)["mean_total_utility"]
    return utilities


def main(n_scenarios: int = 15, seed0: int = 42) -> pd.DataFrame:
    rows = []
    for i in range(n_scenarios):
        utilities = one_scenario(seed0 + i)
        opt = utilities["optimal"]
        row = {"scenario": i, **utilities}
        for policy in POLICIES:
            row[f"gap_{policy}"] = (opt - utilities[policy]) / opt if opt > 0 else 0.0
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = OUT_DIR / "project_b_results.csv"
    df.to_csv(out_path, index=False)

    print("Mean welfare gap relative to the centralized optimum:")
    for policy in POLICIES:
        print(f"  {policy:15s} gap = {df[f'gap_{policy}'].mean():.1%}")
    print(f"\nFull results written to {out_path}")
    return df


if __name__ == "__main__":
    main()
