"""
network.py
==========
Topology and link model.

Each link is parameterized by a Werner-state fidelity parameter w in [0,1]
(standard in the quantum-network-utility-maximization literature: a link
producing Werner states with parameter w gives end-to-end fidelity, after
n such links are swapped together, of F = (3/4) * prod(w_i) + 1/4).

A link's raw entanglement-generation rate d_l (ebits/s, before any
fidelity-rate tradeoff) and its current allocation decide the delivered
rate: R_l = d_l * (1 - w_l) in the standard formulation used by this
literature's "rate vs. fidelity" knob -- lowering w_l (worse fidelity,
more purification/filtering) raises the deliverable rate, and vice versa.
This one-parameter-per-link abstraction is what lets scheduling policy be
studied independently of a specific physical-layer protocol; see
coexistence.py for how shared-fiber classical traffic degrades w_l
further, and calibration.py for pulling realistic w_l/latency
distributions from a physical-layer simulator instead of assuming them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np


@dataclass
class Link:
    """One physical link between two nodes."""

    u: str
    v: str
    length_km: float
    base_rate_ebit_s: float  # max raw entanglement-generation rate, d_l
    coherence_time_s: float = 1.0e-3  # memory decoherence cutoff
    # Base propagation + processing latency at w=1 (no purification rounds).
    base_latency_s: float = 0.0

    def __post_init__(self):
        # Propagation at ~2e5 km/s in fiber (roughly 1.5x slower than
        # vacuum c, standard fiber group-velocity figure), one-way.
        self.base_latency_s = self.base_latency_s or (self.length_km / 2.0e5)

    def rate(self, w: float) -> float:
        """Deliverable entanglement rate at fidelity-parameter w."""
        return self.base_rate_ebit_s * max(0.0, 1.0 - w)

    def latency(self, w: float, purification_rounds: int = 0) -> float:
        """Latency grows with purification effort (more rounds -> better w,
        but more round trips). purification_rounds is informational here;
        the scheduler modules decide how many rounds buy a given w."""
        return self.base_latency_s * (1 + purification_rounds)


def end_to_end_fidelity(w_values: list[float]) -> float:
    """Standard Werner-state swapping composition: F = 3/4 * prod(w) + 1/4.
    This is literature-standard (not something a scheduling policy gets to
    choose), so every scheduler module composes fidelity the same way --
    only the chosen w_l per link differs between policies."""
    prod = float(np.prod(w_values)) if w_values else 1.0
    return 0.75 * prod + 0.25


def build_topology(preset: str = "metro_mesh", **kwargs) -> nx.Graph:
    """Build one of the named topology presets used across scenarios.

    - "metro_mesh": a small random geometric mesh of `n` nodes (default 8),
      representative of a metro-area quantum network testbed.
    - "star_repeater": a generic hub-and-spoke topology with `n_leaders`
      leader/hub nodes, `m_end` end-nodes per leader, and `l_repeaters`
      repeater hops between leader pairs -- a commonly studied shape in the
      entanglement-distribution literature for comparing centralized vs.
      distributed resource allocation, used here as a stress-test topology,
      not attributed to any single source.
    """
    rng = np.random.default_rng(kwargs.get("seed", 0))

    if preset == "metro_mesh":
        n = kwargs.get("n", 8)
        g = nx.random_geometric_graph(n, radius=kwargs.get("radius", 0.55), seed=kwargs.get("seed", 0))
        g = nx.relabel_nodes(g, {i: f"N{i}" for i in g.nodes})
        if not nx.is_connected(g):
            # Densify minimally until connected (rare with these defaults).
            comps = list(nx.connected_components(g))
            for i in range(len(comps) - 1):
                a = next(iter(comps[i]))
                b = next(iter(comps[i + 1]))
                g.add_edge(a, b)
        for u, v in g.edges():
            pu, pv = g.nodes[u].get("pos", (0, 0)), g.nodes[v].get("pos", (0, 0))
            length_km = 1.0 + 40.0 * float(np.hypot(pu[0] - pv[0], pu[1] - pv[1]))
            g.edges[u, v]["link"] = Link(u, v, length_km=length_km,
                                          base_rate_ebit_s=kwargs.get("base_rate_ebit_s", 1000.0))
        return g

    if preset == "star_repeater":
        n_leaders = kwargs.get("n_leaders", 3)
        m_end = kwargs.get("m_end", 4)
        l_repeaters = kwargs.get("l_repeaters", 2)
        g = nx.Graph()
        leaders = [f"L{i}" for i in range(n_leaders)]
        for li in leaders:
            g.add_node(li, role="leader")
            for mi in range(m_end):
                end_node = f"{li}_E{mi}"
                g.add_node(end_node, role="end")
                length_km = 2.0 + 8.0 * rng.random()
                g.add_edge(li, end_node, link=Link(li, end_node, length_km,
                           base_rate_ebit_s=kwargs.get("base_rate_ebit_s", 1000.0)))
        # Chain of repeaters between each consecutive pair of leaders.
        for i in range(len(leaders) - 1):
            prev = leaders[i]
            for r in range(l_repeaters):
                rep = f"R{i}_{r}"
                g.add_node(rep, role="repeater")
                length_km = 5.0 + 10.0 * rng.random()
                g.add_edge(prev, rep, link=Link(prev, rep, length_km,
                           base_rate_ebit_s=kwargs.get("base_rate_ebit_s", 1000.0)))
                prev = rep
            length_km = 5.0 + 10.0 * rng.random()
            g.add_edge(prev, leaders[i + 1], link=Link(prev, leaders[i + 1], length_km,
                       base_rate_ebit_s=kwargs.get("base_rate_ebit_s", 1000.0)))
        return g

    raise ValueError(f"Unknown topology preset: {preset!r}")


def link_key(u: str, v: str) -> frozenset:
    """Undirected identity for a link, independent of traversal direction."""
    return frozenset((u, v))


def shortest_path_links(graph: nx.Graph, src: str, dst: str) -> list[frozenset]:
    """Shortest path (by physical length) between src and dst, returned as
    an ordered list of link_key()s. One fixed route per session is a
    simplification -- this package studies rate/fidelity scheduling given a
    route, not joint routing+scheduling -- but it is the standard
    simplification made in most of the scheduling literature so policies
    can be compared on equal footing.
    """
    node_path = nx.shortest_path(graph, src, dst, weight=lambda u, v, d: d["link"].length_km)
    return [link_key(a, b) for a, b in zip(node_path[:-1], node_path[1:])]
