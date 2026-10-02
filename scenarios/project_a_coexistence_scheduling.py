"""
Project A -- Utility-based adaptive coexistence scheduling.

Question: when a quantum network shares fiber with classical traffic,
how much does the *scheduling policy* recover of what's lost to
coexistence noise, and how does the best propagation choice change as
classical load grows?

This script sweeps classical load levels and propagation schemes (co-,
counter-propagating, time-interleaved) and compares a naive baseline
(fifo) against a utility-aware policy (greedy) and the centralized
optimum (optimal), reporting mean total utility and feasibility for each
combination. Run: `python scenarios/project_a_coexistence_scheduling.py`
from the repo root (with `src` on PYTHONPATH, or the package installed).
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qcs.coexistence import CoexistenceConfig, Propagation  # noqa: E402
from qcs.network import build_topology  # noqa: E402
from qcs.sessions import generate_sessions  # noqa: E402
from qcs.simulation import run_simulation, summarize  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "reports"
OUT_DIR.mkdir(exist_ok=True)

SCHEDULERS = ["fifo", "greedy", "optimal"]
LOAD_KINDS = ["constant", "diurnal", "bursty"]


def main(n_slots: int = 60, n_sessions: int = 20, seed: int = 7) -> pd.DataFrame:
    g = build_topology("metro_mesh", n=10, seed=seed)
    base_sessions = generate_sessions(g, n_sessions=n_sessions, seed=seed + 1)

    rows = []
    for propagation in Propagation:
        for load_kind in LOAD_KINDS:
            cfg = CoexistenceConfig(propagation=propagation)
            for scheduler in SCHEDULERS:
                sessions = copy.deepcopy(base_sessions)
                results = run_simulation(
                    g, sessions, scheduler=scheduler, n_slots=n_slots,
                    clean_w=0.97, coexistence_cfg=cfg, load_kind=load_kind, seed=seed,
                )
                summary = summarize(results)
                rows.append({
                    "propagation": propagation.value,
                    "load_kind": load_kind,
                    "scheduler": scheduler,
                    **summary,
                })

    df = pd.DataFrame(rows)
    out_path = OUT_DIR / "project_a_results.csv"
    df.to_csv(out_path, index=False)
    print(df.pivot_table(index=["propagation", "load_kind"], columns="scheduler", values="mean_total_utility"))
    print(f"\nFull results written to {out_path}")
    return df


if __name__ == "__main__":
    main()
