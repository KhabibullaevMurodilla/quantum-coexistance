"""
qcs.schedulers
==============
Every scheduler module in this package exposes one function:

    schedule(sessions, paths, link_capacity) -> dict[session_id, float]

- `sessions`: list[qcs.sessions.Session] -- the requests competing this slot.
- `paths`: dict[session_id, list[frozenset]] -- each session's fixed route,
  as an ordered list of qcs.network.link_key() values (see
  qcs.network.shortest_path_links).
- `link_capacity`: dict[frozenset, float] -- each link's available
  entanglement-generation rate (ebit/s) this slot, after whatever
  coexistence degradation has already been applied upstream.

Return value: allocated rate (ebit/s) for each session_id, respecting
every link's capacity (the sum of allocations for sessions whose path uses
a given link must not exceed that link's capacity) and never exceeding a
session's own rate_request_ebit_s.

Seven policies are provided, spanning the design space this project set
out to compare:

  - fifo            : arrival-order greedy packing (naive baseline)
  - greedy           : priority-weighted greedy packing (better baseline)
  - proportional_fair: centralized log-utility water-filling (fair baseline)
  - optimal          : centralized convex optimum of the true per-session
                        utilities (the benchmark "what's achievable"
                        centralized solution)
  - primal_dual      : distributed dual-decomposition / shadow-price
                        algorithm converging toward the same optimum as
                        `optimal`, without any central solver -- each
                        session only ever reacts to link prices
  - best_response    : selfish, self-interested rate choice under
                        congestion pricing (a Nash/Wardrop-equilibrium-style
                        policy) -- deliberately NOT coordinated, so it can
                        be compared against `optimal` to quantify the
                        "price of decentralization"
  - learned          : a small, interpretable linear policy (feature
                        weights map directly to a priority score) trained
                        with a simple policy-gradient loop in
                        qcs.schedulers.train_learned
"""
