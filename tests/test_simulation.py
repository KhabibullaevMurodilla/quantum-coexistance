import pytest

from qcs.coexistence import CoexistenceConfig
from qcs.network import build_topology
from qcs.schedulers.registry import SCHEDULER_NAMES
from qcs.sessions import generate_sessions
from qcs.simulation import run_simulation, summarize


@pytest.mark.parametrize("name", SCHEDULER_NAMES)
def test_run_simulation_smoke(name):
    g = build_topology("metro_mesh", n=6, seed=0)
    sessions = generate_sessions(g, n_sessions=10, seed=1)
    results = run_simulation(g, sessions, scheduler=name, n_slots=5, seed=0)
    assert len(results) == 5
    for r in results:
        assert r.total_utility >= 0.0
        assert 0.0 <= r.jain_fairness <= 1.0
        assert r.feasible_sessions + r.violated_sessions <= len(sessions)


def test_summarize_empty():
    assert summarize([]) == {}


def test_summarize_keys():
    g = build_topology("metro_mesh", n=6, seed=0)
    sessions = generate_sessions(g, n_sessions=8, seed=1)
    results = run_simulation(g, sessions, scheduler="greedy", n_slots=5, seed=0)
    summary = summarize(results)
    for key in (
        "mean_total_utility", "mean_total_rate", "mean_jain_fairness",
        "mean_feasible_sessions", "mean_violated_sessions",
    ):
        assert key in summary


def test_optimal_is_not_worse_than_fifo_on_mean_utility():
    g = build_topology("metro_mesh", n=10, seed=3)
    sessions = generate_sessions(g, n_sessions=20, seed=4)

    import copy
    fifo_results = run_simulation(g, copy.deepcopy(sessions), scheduler="fifo", n_slots=20,
                                   coexistence_cfg=CoexistenceConfig(), seed=2)
    optimal_results = run_simulation(g, copy.deepcopy(sessions), scheduler="optimal", n_slots=20,
                                      coexistence_cfg=CoexistenceConfig(), seed=2)

    assert summarize(optimal_results)["mean_total_utility"] >= summarize(fifo_results)["mean_total_utility"] - 1e-6


def test_unknown_scheduler_raises():
    g = build_topology("metro_mesh", n=5, seed=0)
    sessions = generate_sessions(g, n_sessions=3, seed=0)
    with pytest.raises(ValueError):
        run_simulation(g, sessions, scheduler="not_a_real_scheduler", n_slots=2)
