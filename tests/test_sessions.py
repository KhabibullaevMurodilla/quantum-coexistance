import pytest

from qcs.network import build_topology
from qcs.sessions import SessionKind, generate_sessions


def test_generate_sessions_count_and_endpoints():
    g = build_topology("metro_mesh", n=8, seed=0)
    sessions = generate_sessions(g, n_sessions=10, seed=1)
    assert len(sessions) == 10
    nodes = set(g.nodes)
    for s in sessions:
        assert s.src in nodes and s.dst in nodes
        assert s.src != s.dst
        assert 0.0 < s.fidelity_floor <= 1.0
        assert s.rate_request_ebit_s > 0.0
        assert isinstance(s.kind, SessionKind)


def test_generate_sessions_mix_respects_kinds():
    g = build_topology("metro_mesh", n=10, seed=0)
    sessions = generate_sessions(g, n_sessions=30, mix={SessionKind.QKD: 1.0}, seed=2)
    assert all(s.kind == SessionKind.QKD for s in sessions)


def test_generate_sessions_deterministic_with_seed():
    g = build_topology("metro_mesh", n=8, seed=0)
    a = generate_sessions(g, n_sessions=5, seed=42)
    b = generate_sessions(g, n_sessions=5, seed=42)
    assert [(s.src, s.dst, s.kind) for s in a] == [(s.src, s.dst, s.kind) for s in b]


def test_generate_sessions_too_few_nodes_raises():
    import networkx as nx

    g = nx.Graph()
    g.add_node("only_one")
    with pytest.raises(ValueError):
        generate_sessions(g, n_sessions=3)
