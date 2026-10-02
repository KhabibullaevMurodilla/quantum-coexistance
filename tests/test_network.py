import networkx as nx
import pytest

from qcs.network import Link, build_topology, end_to_end_fidelity, link_key, shortest_path_links


def test_end_to_end_fidelity_bounds():
    assert end_to_end_fidelity([]) == 1.0
    assert end_to_end_fidelity([1.0, 1.0]) == 1.0
    assert 0.25 <= end_to_end_fidelity([0.5, 0.7, 0.9]) <= 1.0


def test_end_to_end_fidelity_monotonic_in_w():
    lo = end_to_end_fidelity([0.5, 0.5])
    hi = end_to_end_fidelity([0.9, 0.9])
    assert hi > lo


def test_link_rate_and_latency():
    link = Link("A", "B", length_km=40.0, base_rate_ebit_s=1000.0)
    assert link.rate(1.0) == 0.0  # perfect fidelity -> rate collapses to 0 by construction
    assert link.rate(0.0) == pytest.approx(1000.0)
    assert link.latency(1.0, purification_rounds=0) > 0
    assert link.latency(1.0, purification_rounds=2) > link.latency(1.0, purification_rounds=0)


@pytest.mark.parametrize("n", [5, 8, 15])
def test_metro_mesh_is_connected(n):
    g = build_topology("metro_mesh", n=n, seed=3)
    assert nx.is_connected(g)
    assert g.number_of_nodes() == n
    for _, _, data in g.edges(data=True):
        assert isinstance(data["link"], Link)


def test_star_repeater_shape():
    g = build_topology("star_repeater", n_leaders=2, m_end=3, l_repeaters=1, seed=0)
    leaders = [n for n, d in g.nodes(data=True) if d["role"] == "leader"]
    ends = [n for n, d in g.nodes(data=True) if d["role"] == "end"]
    assert len(leaders) == 2
    assert len(ends) == 6
    assert nx.is_connected(g)


def test_unknown_topology_raises():
    with pytest.raises(ValueError):
        build_topology("not_a_real_preset")


def test_shortest_path_links_matches_graph():
    g = build_topology("metro_mesh", n=6, seed=1)
    nodes = list(g.nodes)
    path = shortest_path_links(g, nodes[0], nodes[-1])
    assert all(isinstance(lk, frozenset) for lk in path)
    # Every consecutive pair in the underlying node path must be a real edge.
    node_path = nx.shortest_path(g, nodes[0], nodes[-1], weight=lambda u, v, d: d["link"].length_km)
    for a, b in zip(node_path[:-1], node_path[1:]):
        assert link_key(a, b) in path
