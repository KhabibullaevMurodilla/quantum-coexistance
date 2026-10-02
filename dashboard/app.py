"""
Streamlit dashboard for the Quantum Coexistence Scheduler testbed.

Run with:  streamlit run dashboard/app.py
(from the repo root, with the package installed or PYTHONPATH=src set --
see the top-level README).

Three tabs map directly onto the three study questions this project set
out to answer:

  1. Coexistence  -- how much does sharing fiber with classical traffic
     cost quantum throughput/fidelity, co- vs. counter-propagating vs.
     time-interleaved?
  2. Scheduler comparison -- across all seven policies, who wins on
     utility, fairness, and feasibility, and by how much?
  3. Price of decentralization -- optimal vs. primal_dual vs.
     best_response, isolating the welfare gap uncoordinated, selfish
     scheduling leaves on the table.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qcs.coexistence import CoexistenceConfig, Propagation, classical_load_profile  # noqa: E402
from qcs.network import build_topology  # noqa: E402
from qcs.schedulers.registry import SCHEDULER_NAMES  # noqa: E402
from qcs.sessions import generate_sessions  # noqa: E402
from qcs.simulation import run_simulation, summarize  # noqa: E402

st.set_page_config(page_title="Quantum Coexistence Scheduler", layout="wide")
st.title("Quantum Coexistence Scheduler")
st.caption(
    "A network-utility-maximization testbed for entanglement scheduling under "
    "classical-fiber coexistence noise."
)

with st.sidebar:
    st.header("Scenario")
    topology = st.selectbox("Topology", ["metro_mesh", "star_repeater"])
    n_nodes = st.slider("Nodes (metro_mesh only)", 5, 20, 8)
    n_sessions = st.slider("Sessions", 4, 40, 16)
    n_slots = st.slider("Time slots", 10, 200, 60)
    clean_w = st.slider("Clean link Werner parameter (w)", 0.80, 0.999, 0.97)
    propagation = st.selectbox(
        "Classical/quantum fiber sharing",
        [p.value for p in Propagation],
        index=1,
    )
    load_kind = st.selectbox("Classical traffic pattern", ["diurnal", "bursty", "constant"])
    seed = st.number_input("Random seed", value=1, step=1)

    st.header("Scheduler(s)")
    chosen = st.multiselect("Policies to run", SCHEDULER_NAMES, default=SCHEDULER_NAMES)
    run_button = st.button("Run simulation", type="primary")

tab1, tab2, tab3 = st.tabs(["Coexistence noise", "Scheduler comparison", "Price of decentralization"])

with tab1:
    st.subheader("Classical traffic load and its noise cost, by propagation scheme")
    load = classical_load_profile(n_slots, kind=load_kind, seed=seed)
    st.line_chart(pd.DataFrame({"classical_load": load}))

    rows = []
    for prop in Propagation:
        cfg = CoexistenceConfig(propagation=prop)
        from qcs.coexistence import apply_coexistence_noise

        w_series = [apply_coexistence_noise(clean_w, float(l), cfg) for l in load]
        rows.append(pd.Series(w_series, name=prop.value))
    df_w = pd.concat(rows, axis=1)
    st.line_chart(df_w)
    st.caption(
        "Effective Werner parameter delivered under each propagation scheme, for the "
        "same classical load trace. Co-propagating traffic (same direction as the "
        "quantum signal) costs noticeably more fidelity than counter-propagating; "
        "interleaving (time-division) nearly eliminates the noise at the cost of "
        "halved classical throughput."
    )

if run_button:
    g = build_topology(topology, n=n_nodes, seed=seed)
    sessions = generate_sessions(g, n_sessions=n_sessions, seed=seed + 1)
    cfg = CoexistenceConfig(propagation=Propagation(propagation))

    summary_rows = []
    per_run_results = {}
    for name in chosen:
        sess_copy = copy.deepcopy(sessions)
        results = run_simulation(
            g, sess_copy, scheduler=name, n_slots=n_slots, clean_w=clean_w,
            coexistence_cfg=cfg, load_kind=load_kind, seed=seed,
        )
        per_run_results[name] = results
        summary = summarize(results)
        summary["scheduler"] = name
        summary_rows.append(summary)

    df_summary = pd.DataFrame(summary_rows).set_index("scheduler")

    with tab2:
        st.subheader("Headline metrics by scheduler")
        st.dataframe(df_summary.style.highlight_max(axis=0, color="#1f6f3f"))

        st.subheader("Total utility over time")
        util_over_time = pd.DataFrame(
            {name: [r.total_utility for r in res] for name, res in per_run_results.items()}
        )
        st.line_chart(util_over_time)

        st.subheader("Jain fairness over time")
        fair_over_time = pd.DataFrame(
            {name: [r.jain_fairness for r in res] for name, res in per_run_results.items()}
        )
        st.line_chart(fair_over_time)

    with tab3:
        st.subheader("Price of decentralization: optimal vs. distributed vs. selfish")
        key_policies = [p for p in ("optimal", "primal_dual", "best_response") if p in df_summary.index]
        if len(key_policies) < 2:
            st.info("Select 'optimal' and at least one of 'primal_dual' / 'best_response' in the sidebar to see this comparison.")
        else:
            sub = df_summary.loc[key_policies, ["mean_total_utility", "mean_jain_fairness"]]
            st.bar_chart(sub["mean_total_utility"])
            if "optimal" in sub.index:
                opt = sub.loc["optimal", "mean_total_utility"]
                gap = {
                    p: (opt - sub.loc[p, "mean_total_utility"]) / opt if opt else 0.0
                    for p in key_policies if p != "optimal"
                }
                st.write(
                    "Welfare lost relative to the centralized optimum "
                    "(fraction of `optimal`'s mean total utility):"
                )
                st.table(pd.Series(gap, name="welfare_gap").map(lambda x: f"{x:.1%}"))
else:
    st.info("Set a scenario in the sidebar and click **Run simulation**.")
