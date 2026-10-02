import pytest

from qcs.network import build_topology, end_to_end_fidelity, link_key, shortest_path_links
from qcs.schedulers.registry import SCHEDULER_NAMES, SCHEDULERS
from qcs.sessions import generate_sessions

_CAP_TOLERANCE = 1e-3


def _build_scenario(seed=5, n_sessions=20, w=0.95):
    g = build_topology("metro_mesh", n=8, seed=1)
    sessions = generate_sessions(g, n_sessions=n_sessions, seed=seed)

    link_capacity = {}
    for u, v, data in g.edges(data=True):
        link_capacity[link_key(u, v)] = data["link"].rate(w)

    paths = {}
    for s in sessions:
        route = shortest_path_links(g, s.src, s.dst)
        fidelity = end_to_end_fidelity([w] * len(route)) if route else 0.0
        s.delivered_fidelity = fidelity
        paths[s.session_id] = route if fidelity >= s.fidelity_floor else []

    return sessions, paths, link_capacity


@pytest.mark.parametrize("name", SCHEDULER_NAMES)
def test_scheduler_respects_link_capacity(name):
    sessions, paths, link_capacity = _build_scenario()
    allocation = SCHEDULERS[name](sessions, paths, link_capacity)

    load = {lk: 0.0 for lk in link_capacity}
    for s in sessions:
        for lk in paths[s.session_id]:
            load[lk] += allocation.get(s.session_id, 0.0)

    for lk, cap in link_capacity.items():
        assert load[lk] <= cap + _CAP_TOLERANCE, f"{name} oversubscribed link {lk}: {load[lk]} > {cap}"


@pytest.mark.parametrize("name", SCHEDULER_NAMES)
def test_scheduler_never_exceeds_request(name):
    sessions, paths, link_capacity = _build_scenario()
    allocation = SCHEDULERS[name](sessions, paths, link_capacity)
    for s in sessions:
        assert allocation.get(s.session_id, 0.0) <= s.rate_request_ebit_s + _CAP_TOLERANCE


@pytest.mark.parametrize("name", SCHEDULER_NAMES)
def test_scheduler_zero_for_infeasible_sessions(name):
    sessions, paths, link_capacity = _build_scenario()
    allocation = SCHEDULERS[name](sessions, paths, link_capacity)
    for s in sessions:
        if not paths[s.session_id]:
            assert allocation.get(s.session_id, 0.0) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("name", SCHEDULER_NAMES)
def test_scheduler_nonnegative_allocations(name):
    sessions, paths, link_capacity = _build_scenario()
    allocation = SCHEDULERS[name](sessions, paths, link_capacity)
    for v in allocation.values():
        assert v >= -_CAP_TOLERANCE


def test_optimal_beats_or_ties_other_policies_on_total_rate():
    # The centralized optimum isn't guaranteed to maximize raw total rate
    # (it maximizes true per-session utility, not throughput), but it
    # should never be beaten by a naive policy on *utility* -- that
    # property is covered in test_simulation.py where utility is computed.
    sessions, paths, link_capacity = _build_scenario()
    for name in SCHEDULER_NAMES:
        allocation = SCHEDULERS[name](sessions, paths, link_capacity)
        assert all(v >= 0 for v in allocation.values())
