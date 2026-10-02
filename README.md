# Quantum Coexistence Scheduler (qcs)

A network-utility-maximization (NUM) testbed for **scheduling entanglement
distribution** in a quantum network that shares optical fiber with
classical (DWDM) traffic, and for comparing how different scheduling
policies -- from naive to centrally optimal to fully decentralized -- cope
with that shared-fiber noise.

This is a portfolio/demo project, not a research paper: it implements
standard, textbook building blocks from the quantum-networking and
network-utility-maximization literature (Werner-state link fidelity,
rate/fidelity tradeoffs, coherence-time cutoffs, Raman-scattering-induced
coexistence noise, Nash/Wardrop-equilibrium congestion games, Kelly-style
primal-dual rate control) and puts them together into one runnable,
tested simulator with a dashboard, rather than proposing a new result.

## Why this problem

Entanglement-based quantum networks are a scarce, shared resource: a
handful of physical fiber links have to serve many competing
applications (QKD, distributed quantum computation, networked sensing),
each with its own rate and fidelity requirements, often while the same
fiber is also carrying ordinary classical data traffic that degrades the
quantum signal via spontaneous Raman scattering. Deciding *who gets how
much, at what fidelity, right now* is exactly the kind of resource
allocation problem that network-utility-maximization and game theory were
built for -- which is what this project explores computationally.

## What's in here

```
src/qcs/
  network.py        Link model (Werner parameter w, rate/fidelity tradeoff,
                     latency), topology builders, shortest-path routing.
  coexistence.py     Classical-traffic-driven Raman noise on a shared
                     fiber: co-/counter-propagating/time-interleaved.
  utility.py          Per-session utility shapes (linear, log-rate,
                     negativity-style) with a hard fidelity-floor cutoff,
                     plus Jain's fairness index.
  sessions.py         Heterogeneous session generator (QKD-/DQC-/
                     sensing-style request archetypes).
  simulation.py       The discrete time-slot loop that drives any
                     scheduler under a coexistence noise process and
                     records per-slot metrics.
  schedulers/
    fifo.py                 naive arrival-order baseline
    greedy.py                priority-weighted greedy packing
    proportional_fair.py     centralized log-utility water-filling
    optimal.py                centralized convex optimum of the true
                              per-session utilities (the benchmark)
    primal_dual.py            distributed Kelly/Low-style shadow-price
                              algorithm -- no central solver
    best_response.py          selfish Nash/Wardrop-equilibrium congestion
                              game -- no coordination at all
    learned.py                 small, fully-readable linear policy
                              (six weights) fit by random search
    train_learned.py           the random-search training loop for
                              learned.py's weights

dashboard/app.py       Streamlit dashboard: coexistence noise, scheduler
                       comparison, price-of-decentralization views.

scenarios/              Three standalone studies (see below), each saving
                       a CSV to reports/.

web/index.html,        Static public page (see "Public interface" below)
web/data.json          and the fixed-snapshot data it renders.
scripts/
  generate_site_data.py Builds web/data.json from a small simulation.

tests/                   pytest suite: capacity feasibility, request caps,
                       fidelity-floor enforcement, and end-to-end
                       simulation smoke tests for every scheduler.
```

## The three studies

**A -- Utility-based adaptive coexistence scheduling**
(`scenarios/project_a_coexistence_scheduling.py`): sweeps classical-load
patterns and propagation schemes against a naive, a utility-aware, and the
centrally-optimal scheduler, to see how much scheduling policy recovers of
what coexistence noise costs, and whether the best propagation choice
changes with load.

**B -- The price of decentralization**
(`scenarios/project_b_price_of_decentralization.py`): runs many random
scenarios comparing the centralized optimum against a coordinated-but-
distributed algorithm (`primal_dual`) and a fully selfish congestion game
(`best_response`), reporting the mean welfare gap of each relative to the
optimum -- an empirical price of anarchy for this setting.

**C -- A learned, interpretable scheduler**
(`scenarios/project_c_learned_interpretable_scheduler.py`): trains
`learned.py`'s six-weight linear priority rule with random search, prints
the weights in plain language, and benchmarks it against the heuristic
and centralized baselines on held-out scenarios -- the point being that a
learned policy here stays as easy to read as the baselines it's compared
against, unlike a typical opaque deep-RL policy.

## Scope and honest limitations

- This operates at the *link-utility* level (links characterized by one
  Werner parameter and a rate/fidelity tradeoff), not the physical-layer
  photonics level. For calibrating that abstraction against a real
  protocol stack, an open-source discrete-event simulator such as
  SeQUeNCe is the natural next step; this project doesn't attempt that.
- Routing is fixed (one shortest path per session, computed once); only
  *rate and fidelity scheduling* over fixed routes is studied, not joint
  routing-and-scheduling.
- The coexistence noise coefficients and congestion-cost function are
  illustrative, order-of-magnitude choices consistent with the qualitative
  co- vs. counter-propagation gap reported across the coexistence
  literature -- they are tunable knobs (`CoexistenceConfig`), not
  calibrated physical constants.
- `learned.py`'s "learning" is random-search hill-climbing over a 6-number
  linear policy, chosen deliberately over a deep network so the result
  stays inspectable; it is not a claim that this outperforms a properly
  trained deep-RL agent.

## Running it

```bash
pip install -e ".[dev,dashboard]"

pytest                                   # full test suite

python scenarios/project_a_coexistence_scheduling.py
python scenarios/project_b_price_of_decentralization.py
python scenarios/project_c_learned_interpretable_scheduler.py

streamlit run dashboard/app.py           # interactive dashboard
```

Each scenario script writes its results to `reports/*.csv` in addition to
printing a summary.

## Public interface

This repo publishes two different front ends, because the two jobs they
do aren't the same thing:

- **A static page** (`web/index.html` + `web/data.json`), deployed by
  `.github/workflows/pages.yml` to GitHub Pages on every push to `main`
  and weekly on a schedule. It shows one fixed snapshot -- scheduler
  comparison, price of decentralization, coexistence noise, the learned
  policy's weights -- computed by `scripts/generate_site_data.py`. No
  backend, no server cost, works for anyone with the link. <https://khabibullaevmurodilla.github.io/quantum-coexistance/>
  **One-time setup**: in the repo's Settings -> Pages, set Source to
  "GitHub Actions" (not "Deploy from a branch"), then run the
  "Build data + deploy GitHub Pages" workflow once (Actions tab ->
  workflow_dispatch) or just push to `main`.
- **The interactive Streamlit dashboard** (`dashboard/app.py`), for
  anyone who wants to pick their own topology, session mix, scheduler,
  and classical-load pattern and see it run live -- this needs an actual
  Python backend (the `optimal`/`proportional_fair` schedulers solve a
  convex program per run), so it can't be static. To get a public URL
  for it: go to <https://share.streamlit.io>, sign in, "New app", point
  it at this repo and `dashboard/app.py` on the `main` branch. Streamlit
  Cloud installs from the root `requirements.txt` already in this repo.
  It redeploys automatically on every push to `main`. <https://quantum-coexistance-jfeugpexj7lh9gavdodq4x.streamlit.app/>
