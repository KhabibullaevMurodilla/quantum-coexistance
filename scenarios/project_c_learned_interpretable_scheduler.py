"""
Project C -- A learned, interpretable scheduler.

Question: can a scheduler that *learns* its priority rule from simulated
experience still be read and understood by a person, unlike a typical
deep-RL policy -- and how close does it get to the heuristic (greedy) and
coordinated (proportional_fair) baselines?

qcs.schedulers.learned scores each session with a six-weight linear
function (qcs.schedulers.learned.FEATURE_NAMES); this script (re)trains
those weights with the random-search loop in
qcs.schedulers.train_learned, prints the resulting weights in plain
English, and benchmarks the learned policy against fifo/greedy/
proportional_fair/optimal on held-out random scenarios (different seeds
than training used). Run:
`python scenarios/project_c_learned_interpretable_scheduler.py`
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qcs.coexistence import CoexistenceConfig  # noqa: E402
from qcs.network import build_topology  # noqa: E402
from qcs.schedulers import learned  # noqa: E402
from qcs.schedulers.train_learned import train  # noqa: E402
from qcs.sessions import generate_sessions  # noqa: E402
from qcs.simulation import run_simulation, summarize  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "reports"
OUT_DIR.mkdir(exist_ok=True)

BASELINES = ["fifo", "greedy", "proportional_fair", "optimal"]


def describe_weights(weights) -> None:
    print("Learned priority rule (higher score = scheduled first):\n")
    for name, value in zip(learned.FEATURE_NAMES, weights):
        sign = "+" if value >= 0 else "-"
        print(f"  {sign} {abs(value):.2f} x {name}")
    print(
        "\nIn plain terms: a session scores higher the more priority weight it "
        "carries, the fewer hops its route needs, the more fidelity headroom it "
        "has above its own floor, and (per the sign/size of the three kind "
        "indicators above) depending on whether it's a QKD-, DQC-, or "
        "sensing-style request."
    )


def main(train_rounds: int = 40, n_eval_scenarios: int = 6, train_seed: int = 0, eval_seed0: int = 900) -> pd.DataFrame:
    print(f"Training the linear policy ({train_rounds} rounds of random search)...\n")
    weights = train(n_rounds=train_rounds, n_scenarios=4, seed=train_seed, verbose=False)
    describe_weights(weights)

    rows = []
    for i in range(n_eval_scenarios):
        seed = eval_seed0 + i
        g = build_topology("metro_mesh", n=10, seed=seed)
        sessions = generate_sessions(g, n_sessions=20, seed=seed + 1)
        cfg = CoexistenceConfig()

        for policy in BASELINES + ["learned"]:
            sess_copy = copy.deepcopy(sessions)
            kwargs = {"weights": weights} if policy == "learned" else {}
            results = run_simulation(
                g, sess_copy, scheduler=policy, n_slots=40,
                clean_w=0.96, coexistence_cfg=cfg, load_kind="diurnal",
                seed=seed, scheduler_kwargs=kwargs,
            )
            summary = summarize(results)
            rows.append({"scenario": i, "scheduler": policy, **summary})

    df = pd.DataFrame(rows)
    out_path = OUT_DIR / "project_c_results.csv"
    df.to_csv(out_path, index=False)

    print("\nMean total utility by policy, averaged over held-out scenarios:")
    print(df.groupby("scheduler")["mean_total_utility"].mean().sort_values(ascending=False))
    print(f"\nFull results written to {out_path}")
    return df


if __name__ == "__main__":
    main()
