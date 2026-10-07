#!/usr/bin/env python3
"""Live CF-ISAC network simulator frontend.

The frontend is intentionally thin:
    UI -> LiveCFISACEngine -> real project algorithms

No physics, prediction, clustering, CVaR, beamforming, or metric
calculation is implemented here.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Project import path
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import streamlit as st

@st.cache_resource
def get_auto_executor():
    return ThreadPoolExecutor(max_workers=1)


from live_sim.engine import LiveCFISACEngine, LiveConfig

try:
    import plotly.graph_objects as go
except ImportError as exc:
    raise SystemExit("Install plotly first: pip install plotly") from exc


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CF-ISAC Live Network Simulator",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)
# ---- Required session-state defaults ----
_SESSION_DEFAULTS = {
    "cvar_alpha": 0.90,
    "enforce_deficiency_cvar": False,
    "max_comm_cvar": 1_000_000.0,
    "max_sensing_cvar": 1e-9,
    "min_rate_bps": 1_000_000.0,
    "epsilon_trk": 1e-9,
}

for _key, _value in _SESSION_DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _value



# ---------------------------------------------------------------------------
# Professional engineering UI
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>

.stApp {
    background:
        radial-gradient(circle at 15% 0%, #14202b 0%, transparent 28%),
        radial-gradient(circle at 90% 10%, #101b25 0%, transparent 25%),
        #090d12;
    color: #e7edf4;
}

/* Main width */
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

/* Headings */
h1 {
    letter-spacing: -0.7px;
    margin-bottom: 0.15rem;
}

h2, h3 {
    letter-spacing: -0.25px;
}

/* Top network header */
.network-header {
    background: linear-gradient(135deg, #101923, #0d141c);
    border: 1px solid #273545;
    border-radius: 14px;
    padding: 18px 22px;
    margin: 8px 0 14px 0;
}

.network-title {
    font-size: 1.55rem;
    font-weight: 750;
    color: #f1f5f9;
}

.network-subtitle {
    color: #8d9aaa;
    margin-top: 4px;
    font-size: 0.88rem;
}

/* Runtime cards */
.runtime-bar {
    display: flex;
    gap: 9px;
    margin: 10px 0 18px 0;
    flex-wrap: wrap;
}

.runtime-card {
    background: #101720;
    border: 1px solid #273545;
    border-radius: 9px;
    padding: 8px 13px;
    min-width: 135px;
}

.runtime-label {
    color: #7f8da0;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.runtime-value {
    color: #edf2f7;
    font-size: 0.95rem;
    font-weight: 650;
    margin-top: 2px;
}

/* Status */
.live-status {
    border-radius: 9px;
    padding: 9px 13px;
    text-align: center;
    font-weight: 700;
    font-size: 0.78rem;
    letter-spacing: 0.06em;
    background: #102219;
    border: 1px solid #315d45;
    color: #65d79b;
}

.paused-status {
    border-radius: 9px;
    padding: 9px 13px;
    text-align: center;
    font-weight: 700;
    font-size: 0.78rem;
    letter-spacing: 0.06em;
    background: #151a20;
    border: 1px solid #303b48;
    color: #9aa8b8;
}

/* Section headers */
.section-title {
    border-left: 3px solid #5dade2;
    padding-left: 10px;
    margin: 22px 0 11px 0;
    color: #e8edf3;
    font-size: 1.03rem;
    font-weight: 700;
}

/* Explanation cards */
.explain-card {
    background: #101720;
    border: 1px solid #273545;
    border-radius: 10px;
    padding: 12px 14px;
    min-height: 88px;
}

.explain-title {
    color: #e8edf3;
    font-weight: 650;
    font-size: 0.88rem;
}

.explain-value {
    color: #f3f6f9;
    font-size: 1.25rem;
    font-weight: 750;
    margin-top: 4px;
}

.explain-help {
    color: #7f8da0;
    font-size: 0.72rem;
    margin-top: 4px;
    line-height: 1.35;
}

/* Decision banner */

.control-pipeline {
    display: flex;
    align-items: stretch;
    gap: 8px;
    margin: 8px 0 22px 0;
    padding: 14px;
    background: #0f151d;
    border: 1px solid #263241;
    border-radius: 12px;
    overflow-x: auto;
}

.pipeline-step {
    flex: 1 1 0;
    min-width: 145px;
    padding: 12px;
    background: #131b25;
    border: 1px solid #2a3544;
    border-radius: 9px;
}

.pipeline-step:hover {
    border-color: #52677f;
}

.pipeline-icon {
    font-size: 1.25rem;
    margin-bottom: 5px;
}

.pipeline-name {
    color: #e8edf3;
    font-weight: 650;
    font-size: 0.9rem;
}

.pipeline-help {
    color: #8997a8;
    font-size: 0.72rem;
    line-height: 1.4;
    margin-top: 6px;
}

.pipeline-arrow {
    display: flex;
    align-items: center;
    color: #647386;
    font-size: 1.1rem;
}

.pipeline-decision {
    border-color: #536b83;
}

.decision-keep {
    border: 1px solid #3d7257;
    background: #0f2018;
    color: #9de1bc;
    padding: 13px 16px;
    border-radius: 10px;
    font-weight: 700;
}

.decision-reconfigure {
    border: 1px solid #8b5050;
    background: #261719;
    color: #ffb9b9;
    padding: 13px 16px;
    border-radius: 10px;
    font-weight: 700;
}

/* Buttons */
.stButton > button {
    border-radius: 8px;
    border: 1px solid #303d4c;
    background: #121a23;
    color: #e8edf3;
    min-height: 42px;
    font-weight: 600;
    transition: all 0.15s ease;
}

.stButton > button:hover {
    border-color: #5dade2;
    background: #182431;
}

/* Metrics */
div[data-testid="stMetric"] {
    background: #101720;
    border: 1px solid #273545;
    border-radius: 9px;
    padding: 10px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d1218;
    border-right: 1px solid #26323f;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border-radius: 9px;
    overflow: hidden;
}

/* Network graph container */
.network-panel {
    background: #090d12;
    border: 1px solid #273545;
    border-radius: 12px;
    padding: 4px;
}

/* Legend */
.legend-box {
    background: #101720;
    border: 1px solid #273545;
    border-radius: 9px;
    padding: 10px 14px;
    color: #9ba8b7;
    font-size: 0.76rem;
    line-height: 1.8;
}

.legend-symbol {
    color: #e7edf4;
    font-weight: 700;
}

/* Timeline */
.timeline-box {
    background: #101720;
    border: 1px solid #273545;
    border-radius: 10px;
    padding: 12px;
}

/* Small note */
.small-note {
    color: #7f8da0;
    font-size: 0.74rem;
    line-height: 1.45;
}

</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Engine creation
# ---------------------------------------------------------------------------

def make_engine() -> LiveCFISACEngine:
    return LiveCFISACEngine.create(
        LiveConfig(
            num_aps=st.session_state.M,
            num_users=st.session_state.K,
            num_targets=st.session_state.Q,
            num_antennas=st.session_state.N,
            area_size=st.session_state.area,
            dt=st.session_state.dt,
            mobility_mode=st.session_state.mobility_mode,
            T_slow=st.session_state.T_slow,
            prediction_horizon=st.session_state.horizon,
            predictor_type=st.session_state.predictor_type,
            history_length=10,
            P_max=st.session_state.P_max,
            precoder=st.session_state.precoder,
            alpha_power=st.session_state.alpha_power,
            beta_power=st.session_state.beta_power,
            num_scenarios=st.session_state.num_scenarios,
            cvar_alpha=st.session_state.cvar_alpha,
        enforce_deficiency_cvar=st.session_state.enforce_deficiency_cvar,
        max_comm_cvar=st.session_state.max_comm_cvar,
        max_sensing_cvar=st.session_state.max_sensing_cvar,
        min_rate_bps=st.session_state.min_rate_bps,
        epsilon_trk=st.session_state.epsilon_trk,
        p1_mode=st.session_state.p1_mode,
        p1_max_candidates=st.session_state.p1_max_candidates,
            lambda_churn=st.session_state.lambda_churn,
            seed=st.session_state.seed,
            base_fast_seed=7,
        )
    )


def init_defaults() -> None:
    defaults = {
        "M": 20,
        "K": 5,
        "Q": 2,
        "N": 4,
        "area": 500.0,
        "dt": 1.0,
        "mobility_mode": "constant_velocity",
        "T_slow": 5,
        "horizon": 5,
        "predictor_type": "cv",
        "P_max": 1.0,
        "precoder": "mrt",
        "alpha_power": 1.0,
        "beta_power": 1.0,
        "num_scenarios": 100,
        "cvar_alpha": 0.9,
        "lambda_churn": 1.0,
        "p1_max_candidates": 20,
        "seed": 42,
        "auto_run": False,
        "auto_pause_reconfigure": True,
        "auto_future": None,
        "auto_next_time": 0.0,
        "sim_speed": 1.0,
    }

    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


init_defaults()


# ---------------------------------------------------------------------------
# Sidebar configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Scenario")

    st.session_state.M = st.number_input(
        "Access Points",
        4,
        40,
        st.session_state.M,
    )

    st.session_state.K = st.number_input(
        "Mobile Users",
        1,
        15,
        st.session_state.K,
    )

    st.session_state.Q = st.number_input(
        "Sensing Targets",
        1,
        10,
        st.session_state.Q,
    )

    st.session_state.N = st.number_input(
        "Antennas / AP",
        1,
        16,
        st.session_state.N,
    )

    st.session_state.area = st.number_input(
        "Area size",
        50.0,
        2000.0,
        st.session_state.area,
        50.0,
    )

    st.session_state.dt = st.number_input(
        "Simulation timestep Δt (s)",
        0.1,
        5.0,
        st.session_state.dt,
        0.1,
    )

    mobility_options = ["constant_velocity", "stochastic"]

    st.session_state.mobility_mode = st.selectbox(
        "Mobility model",
        mobility_options,
        index=mobility_options.index(st.session_state.mobility_mode),
    )

    st.divider()

    st.header("Network control")

    st.session_state.T_slow = st.number_input(
        "Slow update interval",
        1,
        20,
        st.session_state.T_slow,
    )

    st.session_state.horizon = st.number_input(
        "Prediction horizon",
        1,
        20,
        st.session_state.horizon,
    )

    predictor_options = ["cv", "lstm"]

    st.session_state.predictor_type = st.selectbox(
        "Prediction model",
        predictor_options,
        index=predictor_options.index(st.session_state.predictor_type),
    )

    precoder_options = ["mrt", "rzf"]

    st.session_state.precoder = st.selectbox(
        "Beamforming method",
        precoder_options,
        index=precoder_options.index(st.session_state.precoder),
    )

    st.session_state.P_max = st.number_input(
        "Maximum power / AP (W)",
        0.01,
        10.0,
        st.session_state.P_max,
        0.1,
    )

    st.session_state.alpha_power = st.number_input(
        "Communication objective weight",
        0.0,
        20.0,
        st.session_state.alpha_power,
        0.1,
    )

    st.session_state.beta_power = st.number_input(
        "Sensing objective weight",
        0.0,
        20.0,
        st.session_state.beta_power,
        0.1,
    )

    st.session_state.lambda_churn = st.number_input(
        "Cost of changing clusters",
        0.0,
        20.0,
        st.session_state.lambda_churn,
        0.1,
    )

    st.session_state.num_scenarios = st.number_input(
        "Risk scenarios",
        0,
        500,
        st.session_state.num_scenarios,
    )


    st.session_state.cvar_alpha = st.slider(
        "CVaR risk level α",
        0.50,
        0.99,
        st.session_state.cvar_alpha,
        0.01,
    )
    

    st.session_state.p1_mode = st.checkbox(
        "Full P1 Decomposition Mode",
        value=st.session_state.get("p1_mode", False)
    )
    if st.session_state.p1_mode:
        st.session_state.p1_max_candidates = st.number_input(
            "P1 Max Candidates",
            1, 100, st.session_state.get("p1_max_candidates", 20),
        )

    st.session_state.enforce_deficiency_cvar = st.checkbox(
        "Enforce Deficiency CVaR Constraints",
        value=st.session_state.get("enforce_deficiency_cvar", False)
    )
    if st.session_state.enforce_deficiency_cvar:
        st.session_state.max_comm_cvar = st.number_input(
            "Max Comm CVaR Deficiency (bps)",
            0.0, 1e8, st.session_state.get("max_comm_cvar", 1e6), step=1000.0, format="%f"
        )
        st.session_state.max_sensing_cvar = st.number_input(
            "Max Sensing CVaR Deficiency",
            0.0, 1.0, st.session_state.get("max_sensing_cvar", 1e-9), format="%.1e"
        )
        st.session_state.min_rate_bps = st.number_input(
            "Target Comm Rate (bps)",
            0.0, 1e8, st.session_state.get("min_rate_bps", 1e6), step=1000.0, format="%f"
        )
        st.session_state.epsilon_trk = st.number_input(
            "Target Sensing Variance",
            0.0, 1.0, st.session_state.get("epsilon_trk", 1e-9), format="%.1e"
        )


    st.session_state.seed = st.number_input(
        "Simulation seed",
        0,
        999999,
        st.session_state.seed,
    )

    st.divider()

    if st.button("Apply settings / restart", width="stretch"):
        st.session_state.auto_run = False
        st.session_state.engine = make_engine()
        st.rerun()


# ---------------------------------------------------------------------------
# Engine initialization
# ---------------------------------------------------------------------------

if "engine" not in st.session_state:
    st.session_state.engine = make_engine()

engine: LiveCFISACEngine = st.session_state.engine


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

status_text = (
    "● AUTO SIMULATION RUNNING"
    if st.session_state.auto_run
    else "● SIMULATION PAUSED"
)

status_class = (
    "live-status"
    if st.session_state.auto_run
    else "paused-status"
)

st.title("📡 CF-ISAC Network Control Simulator")

st.caption(
    "Predictive mobility · overlapping clustering · churn-aware control · "
    "risk-aware decisions · communication + sensing"
)

status_text = (
    "🟢 AUTO SIMULATION RUNNING"
    if st.session_state.auto_run
    else "⏸️ SIMULATION PAUSED"
)

h1, h2, h3, h4 = st.columns(4)

with h1:
    st.metric("Simulation status", status_text)

with h2:
    st.metric("Timestep", engine.step)

with h3:
    st.metric("Access points", engine.cfg.num_aps)

with h4:
    st.metric("Mobile users", engine.cfg.num_users)

h5, h6, h7 = st.columns(3)

with h5:
    st.metric("Sensing targets", engine.cfg.num_targets)

with h6:
    st.metric("Slow update", f"{engine.cfg.T_slow} steps")

with h7:
    st.metric("Prediction horizon", f"{engine.cfg.prediction_horizon} steps")

st.divider()


# ---------------------------------------------------------------------------
# Simulation controls
# ---------------------------------------------------------------------------


st.markdown(
    '<div class="section-title">Simulation Controls</div>',
    unsafe_allow_html=True,
)

c1, c2, c3, c4, c5 = st.columns([1.25, 1.0, 1.0, 1.0, 1.2])

with c1:
    if st.session_state.auto_run:
        if st.button("⏸  Pause", width="stretch", type="primary"):
            st.session_state.auto_run = False
            st.rerun()
    else:
        if st.button("▶  Auto Run", width="stretch", type="primary"):
            st.session_state.auto_run = True
            st.rerun()

with c2:
    if st.button("⏭  Next step", width="stretch"):
        st.session_state.auto_run = False
        engine.step_once()
        st.rerun()

with c3:
    if st.button("⚡  Evaluate", width="stretch"):
        engine.evaluate_current()
        st.rerun()

with c4:
    if st.button("↻  Reset", width="stretch"):
        st.session_state.auto_run = False
        st.session_state.engine = make_engine()
        st.rerun()

with c5:
    st.session_state.sim_speed = st.select_slider(
        "Auto speed",
        options=[0.25, 0.5, 1.0, 2.0, 4.0],
        value=st.session_state.sim_speed,
        format_func=lambda x: f"{x:g}×",
    )

a1, a2 = st.columns([1, 1])

with a1:
    st.session_state.auto_pause_reconfigure = st.checkbox(
        "Pause automatically when the controller chooses RECONFIGURE",
        value=st.session_state.auto_pause_reconfigure,
    )

with a2:
    st.markdown(
        f'<div class="{status_class}">{status_text}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Interactive object control
# ---------------------------------------------------------------------------

with st.expander("Move a user or sensing target manually", expanded=False):

    st.caption(
        "This changes the actual simulator position before the next physical "
        "simulation step. It does not modify the prediction or clustering code."
    )

    a, b, c = st.columns(3)

    obj_type = a.selectbox(
        "Object",
        ["user", "target"],
    )

    idx_max = (
        engine.cfg.num_users - 1
        if obj_type == "user"
        else engine.cfg.num_targets - 1
    )

    idx = b.number_input(
        "Object index",
        0,
        idx_max,
        0,
    )

    delta = c.number_input(
        "Movement amount",
        -100.0,
        100.0,
        0.0,
        1.0,
    )

    d1, d2 = st.columns(2)

    if d1.button("Move in +X direction", width="stretch"):
        (
            engine.nudge_user
            if obj_type == "user"
            else engine.nudge_target
        )(int(idx), float(delta), 0.0)

        st.rerun()

    if d2.button("Move in +Y direction", width="stretch"):
        (
            engine.nudge_user
            if obj_type == "user"
            else engine.nudge_target
        )(int(idx), 0.0, float(delta))

        st.rerun()


# ---------------------------------------------------------------------------
# Get current real runtime state
# ---------------------------------------------------------------------------

if not engine.records:
    record = engine.evaluate_current()
else:
    record = engine.records[-1]


# ---------------------------------------------------------------------------
# Network visualization
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="section-title">Live Network</div>',
    unsafe_allow_html=True,
)

aps = np.asarray(record["aps"])
users = np.asarray(record["users"])
targets = np.asarray(record["targets"])

x_cluster = np.asarray(record["x"])
y_tx = np.asarray(record["y_tx"])
y_rx = np.asarray(record["y_rx"])

fig = go.Figure()


# ---------------------------------------------------------------------------
# Active communication links
# ---------------------------------------------------------------------------

for m, k in zip(*np.where(x_cluster > 0)):
    fig.add_trace(
        go.Scatter(
            x=[aps[m, 0], users[k, 0]],
            y=[aps[m, 1], users[k, 1]],
            mode="lines",
            line=dict(width=1.5),
            opacity=0.45,
            showlegend=False,
            hoverinfo="skip",
        )
    )


# ---------------------------------------------------------------------------
# Sensing transmitter links
# ---------------------------------------------------------------------------

for m, q in zip(*np.where(y_tx > 0)):
    fig.add_trace(
        go.Scatter(
            x=[aps[m, 0], targets[q, 0]],
            y=[aps[m, 1], targets[q, 1]],
            mode="lines",
            line=dict(width=2, dash="dash"),
            opacity=0.65,
            showlegend=False,
            hoverinfo="skip",
        )
    )


# ---------------------------------------------------------------------------
# Sensing receiver links
# ---------------------------------------------------------------------------

for m, q in zip(*np.where(y_rx > 0)):
    fig.add_trace(
        go.Scatter(
            x=[aps[m, 0], targets[q, 0]],
            y=[aps[m, 1], targets[q, 1]],
            mode="lines",
            line=dict(width=1, dash="dot"),
            opacity=0.55,
            showlegend=False,
            hoverinfo="skip",
        )
    )


# ---------------------------------------------------------------------------
# AP nodes
# ---------------------------------------------------------------------------

fig.add_trace(
    go.Scatter(
        x=aps[:, 0],
        y=aps[:, 1],
        mode="markers+text",
        text=[f"AP {i}" for i in range(len(aps))],
        textposition="top center",
        marker=dict(
            size=10,
            symbol="square",
            line=dict(width=1),
        ),
        name="Access Points",
        hovertemplate="AP %{text}<br>x=%{x:.1f}<br>y=%{y:.1f}<extra></extra>",
    )
)


# ---------------------------------------------------------------------------
# User nodes
# ---------------------------------------------------------------------------

fig.add_trace(
    go.Scatter(
        x=users[:, 0],
        y=users[:, 1],
        mode="markers+text",
        text=[f"U {i}" for i in range(len(users))],
        textposition="bottom center",
        marker=dict(
            size=13,
            symbol="circle",
            line=dict(width=1),
        ),
        name="Mobile Users",
        hovertemplate="User %{text}<br>x=%{x:.1f}<br>y=%{y:.1f}<extra></extra>",
    )
)


# ---------------------------------------------------------------------------
# Target nodes
# ---------------------------------------------------------------------------

fig.add_trace(
    go.Scatter(
        x=targets[:, 0],
        y=targets[:, 1],
        mode="markers+text",
        text=[f"T {i}" for i in range(len(targets))],
        textposition="bottom center",
        marker=dict(
            size=15,
            symbol="diamond",
            line=dict(width=1),
        ),
        name="Sensing Targets",
        hovertemplate="Target %{text}<br>x=%{x:.1f}<br>y=%{y:.1f}<extra></extra>",
    )
)


# ---------------------------------------------------------------------------
# Predicted user trajectories
# ---------------------------------------------------------------------------

if record["pred_users"] is not None:

    for k in range(record["pred_users"].shape[0]):

        p = record["pred_users"][k]

        fig.add_trace(
            go.Scatter(
                x=p[:, 0],
                y=p[:, 1],
                mode="lines+markers",
                line=dict(width=1.5, dash="dot"),
                marker=dict(size=3),
                showlegend=False,
                hoverinfo="skip",
            )
        )


# ---------------------------------------------------------------------------
# Predicted target trajectories
# ---------------------------------------------------------------------------

if record["pred_targets"] is not None:

    for q in range(record["pred_targets"].shape[0]):

        p = record["pred_targets"][q]

        fig.add_trace(
            go.Scatter(
                x=p[:, 0],
                y=p[:, 1],
                mode="lines+markers",
                line=dict(width=1.5, dash="dashdot"),
                marker=dict(size=3),
                showlegend=False,
                hoverinfo="skip",
            )
        )


# ---------------------------------------------------------------------------
# Graph styling
# ---------------------------------------------------------------------------

fig.update_layout(
    height=680,
    margin=dict(l=10, r=10, t=55, b=10),
    paper_bgcolor="#090d12",
    plot_bgcolor="#090d12",
    font=dict(color="#dce5ee"),
    title=dict(
        text=(
            f"<b>LIVE NETWORK TOPOLOGY</b> "
            f"<span style='font-size:13px'>· STEP {record['step']}</span>"
        ),
        x=0.02,
        xanchor="left",
    ),
    hoverlabel=dict(
        bgcolor="#111720",
        font_size=12,
    ),
    xaxis=dict(
        range=[0, engine.cfg.area_size],
        title="X position",
        gridcolor="#202a36",
        zeroline=False,
    ),
    yaxis=dict(
        range=[0, engine.cfg.area_size],
        title="Y position",
        gridcolor="#202a36",
        zeroline=False,
        scaleanchor="x",
        scaleratio=1,
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.01,
        xanchor="right",
        x=1,
    ),
)

st.markdown(
    '<div class="network-panel">',
    unsafe_allow_html=True,
)

st.plotly_chart(
    fig,
    width="stretch",
    config={
        "displaylogo": False,
        "scrollZoom": True,
        "modeBarButtonsToRemove": [
            "lasso2d",
            "select2d",
        ],
    },
)

st.markdown(
    '</div>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Network legend
# ---------------------------------------------------------------------------

st.markdown(
    """
<div class="legend-box">
    <span class="legend-symbol">■ AP</span> = Access Point &nbsp;&nbsp;
    <span class="legend-symbol">● User</span> = Mobile user &nbsp;&nbsp;
    <span class="legend-symbol">◆ Target</span> = Sensing target
    <br>
    <span class="legend-symbol">────</span> Communication link &nbsp;&nbsp;
    <span class="legend-symbol">- - -</span> Sensing TX &nbsp;&nbsp;
    <span class="legend-symbol">····</span> Sensing RX &nbsp;&nbsp;
    <span class="legend-symbol">╌╌╌</span> Predicted movement
</div>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Explainable control pipeline
# ---------------------------------------------------------------------------

st.markdown("### How the Network Controller Works")

pipeline = [
    (
        "📍",
        "Mobility",
        "The simulation updates user and target positions using the selected mobility model.",
    ),
    (
        "🔮",
        "Prediction",
        "The predictor estimates future user positions over the configured prediction horizon.",
    ),
    (
        "🔗",
        "Candidate Clustering",
        "Predicted positions are mapped to overlapping AP clusters that could serve the users.",
    ),
    (
        "⚠️",
        "Uncertainty / CVaR",
        "Prediction uncertainty is evaluated and the potential future benefit is adjusted for downside risk.",
    ),
    (
        "⚙️",
        "KEEP / RECONFIGURE",
        "The risk-adjusted benefit of changing clusters is compared with the modeled churn cost.",
    ),
    (
        "📡",
        "Physical Layer",
        "The selected configuration is evaluated using beamforming, communication rates, power and sensing performance.",
    ),
]

# Three stages per row: stage → arrow → stage.
for row_start in range(0, len(pipeline), 2):
    if row_start + 1 < len(pipeline):
        left, arrow, right = st.columns([1, 0.12, 1])

        icon, name, description = pipeline[row_start]
        with left:
            st.markdown(f"**{icon} {name}**")
            st.caption(description)

        with arrow:
            st.markdown("### →")

        icon, name, description = pipeline[row_start + 1]
        with right:
            st.markdown(f"**{icon} {name}**")
            st.caption(description)
    else:
        icon, name, description = pipeline[row_start]
        st.markdown(f"**{icon} {name}**")
        st.caption(description)


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------



st.markdown(
    '<div class="section-title">What is the Network Controller Doing?</div>',
    unsafe_allow_html=True,
)

decision = record["decision"]

# BOOTSTRAP is the real engine state before the first
# slow-timescale controller evaluation.
if decision == "BOOTSTRAP":

    st.markdown(
        """
<div class="decision-keep">
    ◉ INITIALIZED
    <br>
    The network is using its initial cluster configuration.
    The first risk-aware controller decision will occur at the
    configured slow-update boundary.
</div>
""",
        unsafe_allow_html=True,
    )

elif decision == "RECONFIGURE":

    st.markdown(
        """
<div class="decision-reconfigure">
    ⚠ RECONFIGURE
    <br>
    The controller decided that the current cluster configuration should change.
</div>
""",
        unsafe_allow_html=True,
    )

else:

    st.markdown(
        """
<div class="decision-keep">
    ✓ KEEP
    <br>
    The controller evaluated the predicted future state and retained
    the current cluster configuration.
</div>
""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Decision explanation
# ---------------------------------------------------------------------------

d1, d2, d3, d4 = st.columns(4)

bootstrap = record["decision"] == "BOOTSTRAP"

predicted_gain_text = (
    "—" if bootstrap else f"{record['predicted_gain']:.3g}"
)
risk_adjusted_gain_text = (
    "—" if bootstrap else f"{record['risk_adjusted_gain']:.3g}"
)
raw_churn_text = (
    "—" if bootstrap else f"{record['raw_churn']:.0f}"
)

with d1:
    st.markdown(
        f"""
<div class="explain-card">
    <div class="explain-title">Predicted benefit</div>
    <div class="explain-value">{predicted_gain_text}</div>
    <div class="explain-help">
        {"Waiting for the first slow-timescale controller evaluation."
         if bootstrap
         else "Expected benefit from the controller's predicted future state."}
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

with d2:
    st.markdown(
        f"""
<div class="explain-card">
    <div class="explain-title">Risk-adjusted benefit</div>
    <div class="explain-value">{risk_adjusted_gain_text}</div>
    <div class="explain-help">
        {"Risk evaluation has not yet been performed."
         if bootstrap
         else "Predicted benefit after accounting for uncertainty and risk."}
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

with d3:
    st.markdown(
        f"""
<div class="explain-card">
    <div class="explain-title">Cluster changes</div>
    <div class="explain-value">{raw_churn_text}</div>
    <div class="explain-help">
        {"No controller comparison has been performed yet."
         if bootstrap
         else "Number of AP association changes considered by the controller."}
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

with d4:
    st.markdown(
        f"""
<div class="explain-card">
    <div class="explain-title">Reconfigurations</div>
    <div class="explain-value">{record['reconfigurations']}</div>
    <div class="explain-help">
        Number of times the network has actually changed its configuration.
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Communication
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="section-title">Communication Performance</div>',
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Total data rate",
        f"{record['rate_bps'] / 1e6:.2f} Mbps",
    )

with c2:
    st.metric(
        "Minimum user rate",
        f"{record['min_user_rate_bps'] / 1e6:.2f} Mbps",
    )

with c3:
    total_pcomm = float(np.sum(np.asarray(record["P_comm"])))
    st.metric(
        "Communication power",
        f"{total_pcomm:.3f} W",
    )

with c4:
    st.metric(
        "Physical objective",
        f"{record['objective']:.3g}",
    )

st.caption(
    "Communication rate measures aggregate user throughput. "
    "Minimum user rate shows the lowest individual user rate. "
    "Communication power is the total transmit power used for communication."
)


# ---------------------------------------------------------------------------
# Sensing
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="section-title">Sensing Performance</div>',
    unsafe_allow_html=True,
)

s1, s2, s3 = st.columns(3)

with s1:
    st.metric(
        "Sensing information",
        f"{record['sensing_info']:.3f} nats",
    )

with s2:
    total_psens = float(np.sum(np.asarray(record["P_sens"])))
    st.metric(
        "Sensing power",
        f"{total_psens:.3f} W",
    )

with s3:
    st.metric(
        "Total transmit power",
        f"{float(np.sum(np.asarray(record['P_tx']))):.3f} W",
    )

st.caption(
    "Sensing information represents the information gain produced by the "
    "current sensing configuration. Higher values indicate more sensing "
    "information in the current scenario."
)


# ---------------------------------------------------------------------------
# Risk and stability
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="section-title">Risk & Network Stability</div>',
    unsafe_allow_html=True,
)

r1, r2, r3, r4 = st.columns(4)

with r1:
    st.metric(
        "Risk threshold (VaR)",
        f"{record['var']:.3g}",
    )

with r2:
    st.metric(
        "Worst-case risk (CVaR)",
        f"{record['cvar']:.3g}",
    )

with r3:
    st.metric(
        "Cumulative cluster changes",
        f"{record['churn_total']:.0f}",
    )

with r4:
    st.metric(
        "Network reconfigurations",
        f"{record['reconfigurations']}",
    )

st.caption(
    "VaR identifies the selected risk threshold. CVaR summarizes the "
    "average outcome in the worst tail of the predicted scenarios. "
    "Cluster changes measure association churn; reconfigurations count "
    "controller decisions that actually changed the configuration."
)


# ---------------------------------------------------------------------------
# Physical feasibility
# ---------------------------------------------------------------------------

feasible_text = "✓ Feasible" if record["feasible"] else "⚠ Constraint issue"

if record["feasible"]:
    st.success(
        f"{feasible_text} · "
        f"maximum power violation = {record['max_power_violation']:.3e}"
    )
else:
    st.warning(
        f"{feasible_text} · "
        f"maximum power violation = {record['max_power_violation']:.3e}"
    )


# ---------------------------------------------------------------------------
# Live performance history
# ---------------------------------------------------------------------------

hist = engine.history_table()

st.markdown(
    '<div class="section-title">Performance Over Time</div>',
    unsafe_allow_html=True,
)

if len(hist) > 1:

    p1, p2, p3 = st.columns(3)

    with p1:
        st.caption("Communication rate")
        st.line_chart(
            hist.set_index("step")[["rate_bps"]]
        )

    with p2:
        st.caption("Sensing information and risk")
        st.line_chart(
            hist.set_index("step")[["sensing_info", "cvar"]]
        )

    with p3:
        st.caption("Cluster changes and controller gain")
        st.line_chart(
            hist.set_index("step")[["raw_churn", "net_gain"]]
        )

else:

    st.info(
        "Advance the simulation to build the live performance history."
    )


# ---------------------------------------------------------------------------
# Cluster state
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="section-title">Current Cluster Configuration</div>',
    unsafe_allow_html=True,
)

st.caption(
    "These matrices are the actual cluster assignments used by the runtime."
)

ca, cb, cc = st.columns(3)

with ca:
    st.markdown("**User ↔ AP communication**")
    st.dataframe(
        record["x"],
        width="stretch",
    )

with cb:
    st.markdown("**Target ↔ AP sensing transmitters**")
    st.dataframe(
        record["y_tx"],
        width="stretch",
    )

with cc:
    st.markdown("**Target ↔ AP sensing receivers**")
    st.dataframe(
        record["y_rx"],
        width="stretch",
    )


# ---------------------------------------------------------------------------
# Runtime history
# ---------------------------------------------------------------------------

with st.expander("Detailed runtime history", expanded=False):

    st.dataframe(
        hist,
        width="stretch",
    )

    csv = hist.to_csv(index=False).encode()

    st.download_button(
        "Export runtime CSV",
        csv,
        file_name="cfisac_live_runtime.csv",
        mime="text/csv",
        width="stretch",
    )


# ---------------------------------------------------------------------------
# Auto simulation
# ---------------------------------------------------------------------------

auto_run_every = 0.5 if st.session_state.auto_run else None


@st.fragment(run_every=auto_run_every)
def auto_simulation_worker():
    if not st.session_state.auto_run:
        return

    executor = get_auto_executor()
    future = st.session_state.get("auto_future")

    if future is not None:
        if future.done():
            try:
                new_record = future.result()
                st.session_state.last_auto_record = new_record
                st.session_state.auto_future = None

                delay = max(
                    0.20,
                    1.0 / float(st.session_state.sim_speed),
                )
                st.session_state.auto_next_time = (
                    time.monotonic() + delay
                )

                st.rerun()

            except Exception as exc:
                st.session_state.auto_future = None
                st.session_state.auto_run = False
                st.error(f"Auto Run failed: {exc}")
                return
        else:
            st.caption(
                "⏳ Auto Run: P1 + Deficiency CVaR is calculating..."
            )
            return

    if time.monotonic() < st.session_state.get("auto_next_time", 0.0):
        return

    st.session_state.auto_future = executor.submit(engine.step_once)
    st.caption("⚙️ Auto Run: simulation step running...")


auto_simulation_worker()


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown(
    """
<div class="small-note">
    <b>Runtime provenance:</b>
    every timestep is computed by the live CF-ISAC engine.
    Auto Run repeatedly calls the same physical simulation step used by
    manual stepping. The frontend does not preload a trajectory, generate
    substitute physics, or replace the prediction, clustering, risk,
    beamforming, communication, or sensing algorithms.
</div>
""",
    unsafe_allow_html=True,
)
