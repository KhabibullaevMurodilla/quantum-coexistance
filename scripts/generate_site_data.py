"""
generate_site_data.py
=======================
Runs a fixed, modest-sized set of simulations and writes web/data.json --
everything the static dashboard at web/index.html needs to render, with
no backend of its own. This is what .github/workflows/pages.yml runs
before every deploy, mirroring the same "compute once, serve a static
JSON" pattern used by the sibling Voyager forecast project.

Kept intentionally small (few nodes/sessions/slots, few training rounds)
so a GitHub Actions run finishes in well under a minute -- this is a
snapshot for a public page to look at, not a research-grade sweep (run
the scenarios/ scripts directly, or the Streamlit dashboard, for that).
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qcs.coexistence import CoexistenceConfig, Propagation, apply_coexistence_noise, classical_load_profile  # noqa: E402
from qcs.network import build_topology  # noqa: E402
from qcs.schedulers import learned  # noqa: E402
from qcs.schedulers.registry import SCHEDULER_NAMES  # noqa: E402
from qcs.schedulers.train_learned import train  # noqa: E402
from qcs.sessions import generate_sessions  # noqa: E402
from qcs.simulation import run_simulation, summarize  # noqa: E402

OUT_PATH = ROOT / "web" / "data.json"

N_NODES = 10
N_SESSIONS = 20
N_SLOTS = 40
CLEAN_W = 0.96
SEED = 7


def build_scheduler_comparison() -> list[dict]:
    g = build_topology("metro_mesh", n=N_NODES, seed=SEED)
    sessions = generate_sessions(g, n_sessions=N_SESSIONS, seed=SEED + 1)
    cfg = CoexistenceConfig(propagation=Propagation.COUNTER)

    rows = []
    for name in SCHEDULER_NAMES:
        results = run_simulation(
            g, copy.deepcopy(sessions), scheduler=name, n_slots=N_SLOTS,
            clean_w=CLEAN_W, coexistence_cfg=cfg, load_kind="diurnal", seed=SEED,
        )
        summary = summarize(results)
        rows.append({"scheduler": name, **summary})
    return rows


def build_price_of_decentralization(n_scenarios: int = 6) -> dict:
    policies = ["optimal", "primal_dual", "best_response"]
    gaps = {p: [] for p in policies}

    for i in range(n_scenarios):
        seed = 200 + i
        g = build_topology("metro_mesh", n=N_NODES, seed=seed)
        sessions = generate_sessions(g, n_sessions=N_SESSIONS, seed=seed + 1)
        cfg = CoexistenceConfig(propagation=Propagation.COUNTER)

        utilities = {}
        for p in policies:
            results = run_simulation(
                g, copy.deepcopy(sessions), scheduler=p, n_slots=25,
                clean_w=CLEAN_W, coexistence_cfg=cfg, load_kind="diurnal", seed=seed,
            )
            utilities[p] = summarize(results)["mean_total_utility"]

        opt = utilities["optimal"]
        for p in policies:
            gaps[p].append((opt - utilities[p]) / opt if opt > 0 else 0.0)

    return {p: sum(v) / len(v) for p, v in gaps.items()}


def build_coexistence_noise(n_slots: int = 60) -> dict:
    load = classical_load_profile(n_slots, kind="diurnal", seed=1)
    series = {"classical_load": [round(float(x), 4) for x in load]}
    for prop in Propagation:
        cfg = CoexistenceConfig(propagation=prop)
        series[prop.value] = [
            round(apply_coexistence_noise(CLEAN_W, float(x), cfg), 4) for x in load
        ]
    return series


def build_learned_weights(train_rounds: int = 30) -> dict:
    weights = train(n_rounds=train_rounds, n_scenarios=3, seed=1, verbose=False)
    return {name: round(float(w), 3) for name, w in zip(learned.FEATURE_NAMES, weights)}


def main() -> None:
    data = {
        "generated_with": {
            "n_nodes": N_NODES, "n_sessions": N_SESSIONS, "n_slots": N_SLOTS, "clean_w": CLEAN_W,
        },
        "scheduler_comparison": build_scheduler_comparison(),
        "price_of_decentralization": build_price_of_decentralization(),
        "coexistence_noise": build_coexistence_noise(),
        "learned_weights": build_learned_weights(),
    }

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(data, indent=2))
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
