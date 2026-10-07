#!/usr/bin/env python3
from __future__ import annotations

import copy
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.cfisac_runtime_adapter import build_ap_positions, build_predictor, build_simulator

st.set_page_config(
    page_title="CF-ISAC Network Control",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Visual direction: compact operator/engineering dashboard inspired by
# Plotly's manufacturing-monitoring pattern and network-topology dashboards.
st.markdown(
    """
    <style>
    :root {
        --bg:#070b12; --panel:#0d131d; --panel2:#101824; --line:#1e2a38;
        --text:#eef4fb; --muted:#8b9aad; --cyan:#48d5ff; --green:#48e0a3;
        --amber:#ffbf5a; --red:#ff6b7a; --purple:#a88bff;
    }
    [data-testid="stAppViewContainer"] { background:var(--bg); }
    [data-testid="stHeader"] { background:rgba(7,11,18,.92); }
    .block-container { max-width:1720px; padding:1rem 1.25rem 2rem; }
    section[data-testid="stSidebar"] { background:#090f17; border-right:1px solid var(--line); }
    section[data-testid="stSidebar"] .block-container { padding-top:1.0rem; }
    h1,h2,h3,h4,p,div,span { font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    .brand { display:flex; justify-content:space-between; gap:18px; align-items:flex-end; margin:0 0 12px; }
    .brand-left .eyebrow { color:var(--cyan); font-size:.68rem; font-weight:800; letter-spacing:.17em; }
    .brand-left .title { color:var(--text); font-size:1.65rem; line-height:1.1; font-weight:800; letter-spacing:-.03em; margin-top:4px; }
    .brand-left .sub { color:var(--muted); font-size:.78rem; margin-top:5px; }
    .brand-right { text-align:right; color:#77879a; font-size:.7rem; line-height:1.5; }
    .statusbar { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:14px; }
    .pill { display:inline-flex; align-items:center; gap:6px; padding:6px 9px; border:1px solid var(--line); border-radius:999px; background:#0b1119; color:#aebdcd; font-size:.67rem; font-weight:750; letter-spacing:.04em; text-transform:uppercase; }
    .dot { width:7px; height:7px; border-radius:50%; display:inline-block; background:#59697c; }
    .dot.green { background:var(--green); box-shadow:0 0 9px rgba(72,224,163,.45); }
    .dot.cyan { background:var(--cyan); box-shadow:0 0 9px rgba(72,213,255,.4); }
    .dot.amber { background:var(--amber); }
    .panel { background:linear-gradient(180deg,#0e151f,#0b1119); border:1px solid var(--line); border-radius:14px; padding:14px; }
    .panel-title { color:#dce6f1; font-weight:800; font-size:.74rem; letter-spacing:.12em; text-transform:uppercase; margin-bottom:10px; }
    .panel-note { color:#7f90a4; font-size:.72rem; line-height:1.45; }
    .metric-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin:12px 0; }
    .metric { background:linear-gradient(180deg,#101824,#0c131c); border:1px solid var(--line); border-radius:13px; padding:12px 13px; min-height:88px; }
    .metric .k { color:#7f90a4; font-size:.64rem; text-transform:uppercase; letter-spacing:.11em; font-weight:750; }
    .metric .v { color:var(--text); font-size:1.26rem; font-weight:800; margin-top:5px; letter-spacing:-.02em; }
    .metric .s { color:#77899b; font-size:.68rem; margin-top:3px; }
    .action-card { background:#0c121a; border:1px solid var(--line); border-radius:14px; padding:14px; height:100%; }
    .action { display:inline-flex; padding:6px 11px; border-radius:8px; font-size:.78rem; font-weight:850; letter-spacing:.06em; }
    .action.keep { background:rgba(72,224,163,.11); color:var(--green); border:1px solid rgba(72,224,163,.28); }
    .action.reconfig { background:rgba(255,191,90,.12); color:var(--amber); border:1px solid rgba(255,191,90,.28); }
    .action.fast { background:rgba(72,213,255,.1); color:var(--cyan); border:1px solid rgba(72,213,255,.25); }
    .reason { color:#bdc9d6; font-size:.77rem; line-height:1.5; margin:10px 0 12px; }
    .minirow { display:flex; justify-content:space-between; gap:12px; padding:8px 0; border-top:1px solid #192532; }
    .minirow:first-child { border-top:0; }
    .minirow .label { color:#7f90a4; font-size:.69rem; }
    .minirow .value { color:#e8eef5; font-size:.75rem; font-weight:700; text-align:right; }
    .legend-row { display:flex; gap:14px; flex-wrap:wrap; color:#8798a9; font-size:.68rem; margin-top:-3px; }
    .legend-item { display:flex; align-items:center; gap:5px; }
    .legend-line { width:20px; border-top:2px solid var(--cyan); }
    .legend-dash { width:20px; border-top:2px dashed var(--amber); }
    .legend-dot { width:20px; border-top:2px dotted var(--purple); }
    .callout { border:1px solid #20303f; background:#0b121a; border-radius:11px; padding:11px 12px; color:#b8c6d5; font-size:.74rem; line-height:1.5; }
    .callout strong { color:#eef4fb; }
    .footer-note { color:#617286; font-size:.64rem; margin-top:12px; }
    div[data-testid="stMetric"] { background:#0e151f; border:1px solid var(--line); border-radius:12px; }
    .stTabs [data-baseweb="tab-list"] { gap:4px; border-bottom:1px solid var(--line); }
    .stTabs [data-baseweb="tab"] { color:#8798aa; font-size:.76rem; font-weight:700; padding:8px 12px; }
    .stTabs [aria-selected="true"] { color:#eff5fb !important; }
    button[kind="primary"] { border-radius:9px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _f(v: Any, default=np.nan) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _g(d: dict[str, Any], *keys: str, default=np.nan) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return default


def fmt_rate(v: float) -> str:
    return "—" if not np.isfinite(v) else f"{v/1e6:,.1f} Mbps"


def fmt_num(v: float, digits: int = 2) -> str:
    return "—" if not np.isfinite(v) else f"{v:,.{digits}f}"


def metric_card(label: str, value: str, sub: str = "") -> None:
    st.markdown(
        f'<div class="metric"><div class="k">{label}</div><div class="v">{value}</div>'
        f'{f"<div class=\"s\">{sub}</div>" if sub else ""}</div>',
        unsafe_allow_html=True,
    )


def run_observable_trace(sim: Any, predictor: Any, ap_positions: np.ndarray, *, num_antennas: int,
                         t_total: int, t_slow: int, p_max: float, precoder: str,
                         alpha_power: float, beta_power: float, num_scenarios: int,
                         cvar_alpha: float, base_seed: int, scenario_base_seed: int) -> dict[str, Any]:
    from src.optimization.baselines import cluster_for_positions
    from src.optimization.risk_aware import risk_aware_slow_update
    from src.optimization.two_timescale import fast_update

    x, y_tx, y_rx = cluster_for_positions(ap_positions, sim.user_positions, sim.target_positions)
    records: list[dict[str, Any]] = []
    tau = -1

    for t in range(t_total):
        users_now = np.asarray(sim.user_positions, dtype=float).copy()
        targets_now = np.asarray(sim.target_positions, dtype=float).copy()
        slow = None
        if t % t_slow == 0:
            tau += 1
            scenario_seed = scenario_base_seed * 1_000_003 + tau
            upd = risk_aware_slow_update(
                sim, predictor, ap_positions, x, y_tx, y_rx, t_slow,
                num_scenarios=num_scenarios, alpha=cvar_alpha,
                scenario_seed=scenario_seed,
                cluster_kwargs={}, perf_kwargs={}, churn_kwargs={},
                ref=None, horizon_aggregation="sum",
            )
            x, y_tx, y_rx = np.asarray(upd["x"]).copy(), np.asarray(upd["y_tx"]).copy(), np.asarray(upd["y_rx"]).copy()
            slow = {
                "decision": str(upd["decision"]),
                "candidate_x": np.asarray(upd["candidate_x"]).copy(),
                "candidate_y_tx": np.asarray(upd["candidate_y_tx"]).copy(),
                "candidate_y_rx": np.asarray(upd["candidate_y_rx"]).copy(),
                "pred_users": np.asarray(upd["pred_users"]).copy(),
                "pred_targets": np.asarray(upd["pred_targets"]).copy(),
                "decision_info": copy.deepcopy(upd["decision_info"]),
                "var_prev": _f(upd.get("var_prev")), "cvar_prev": _f(upd.get("cvar_prev")),
                "var_cand": _f(upd.get("var_cand")), "cvar_cand": _f(upd.get("cvar_cand")),
                "risk_aware": bool(upd.get("risk_aware", False)),
                "num_scenarios": int(upd.get("num_scenarios", num_scenarios)),
                "alpha": _f(upd.get("alpha", cvar_alpha)),
            }
        fast_seed = base_seed * 1_000_003 + t
        fast = copy.deepcopy(fast_update(
            ap_positions, sim.user_positions, sim.target_positions,
            x, y_tx, y_rx, num_antennas, p_max,
            precoder=precoder, alpha=alpha_power, beta=beta_power,
            fast_seed=fast_seed, phys_kwargs={}, sens_kwargs={}, joint_kwargs={},
        ))
        records.append({"t": t, "tau": tau, "users": users_now, "targets": targets_now,
                        "x": x.copy(), "y_tx": y_tx.copy(), "y_rx": y_rx.copy(),
                        "slow": slow, "fast": fast})
        if t < t_total - 1:
            sim.step()
    return {"steps": records, "ap_positions": np.asarray(ap_positions, dtype=float).copy(),
            "meta": {"t_total": t_total, "t_slow": t_slow, "num_antennas": num_antennas,
                     "p_max": p_max, "precoder": precoder, "alpha": alpha_power, "beta": beta_power,
                     "num_scenarios": num_scenarios, "cvar_alpha": cvar_alpha,
                     "base_seed": base_seed, "scenario_base_seed": scenario_base_seed}}


def network_figure(trace: dict[str, Any], idx: int, show_prediction: bool, show_candidate: bool) -> go.Figure:
    r = trace["steps"][idx]; ap = trace["ap_positions"]; u = r["users"]; tar = r["targets"]
    fig = go.Figure()
    # APs / users / targets
    fig.add_trace(go.Scatter(x=ap[:,0], y=ap[:,1], mode="markers+text",
        text=[f"AP {i+1}" for i in range(len(ap))], textposition="top center",
        name="Access points", marker=dict(symbol="square", size=9, color="#48d5ff", line=dict(color="#c9f6ff", width=1))))
    fig.add_trace(go.Scatter(x=u[:,0], y=u[:,1], mode="markers+text",
        text=[f"U{i+1}" for i in range(len(u))], textposition="bottom center",
        name="Users", marker=dict(symbol="circle", size=11, color="#48e0a3", line=dict(color="#d6fff0", width=1))))
    fig.add_trace(go.Scatter(x=tar[:,0], y=tar[:,1], mode="markers+text",
        text=[f"T{i+1}" for i in range(len(tar))], textposition="bottom center",
        name="Targets", marker=dict(symbol="diamond", size=13, color="#ffbf5a", line=dict(color="#fff0cf", width=1))))

    x, yt, yr = r["x"], r["y_tx"], r["y_rx"]
    for m in range(x.shape[0]):
        for k in range(x.shape[1]):
            if x[m,k]:
                fig.add_trace(go.Scatter(x=[ap[m,0],u[k,0]], y=[ap[m,1],u[k,1]], mode="lines",
                    line=dict(color="#48d5ff", width=1.5), opacity=.48, showlegend=False, hoverinfo="skip"))
    for m in range(yt.shape[0]):
        for q in range(yt.shape[1]):
            if yt[m,q]:
                fig.add_trace(go.Scatter(x=[ap[m,0],tar[q,0]], y=[ap[m,1],tar[q,1]], mode="lines",
                    line=dict(color="#ffbf5a", width=1.6), opacity=.42, showlegend=False, hoverinfo="skip"))
    for m in range(yr.shape[0]):
        for q in range(yr.shape[1]):
            if yr[m,q]:
                fig.add_trace(go.Scatter(x=[tar[q,0],ap[m,0]], y=[tar[q,1],ap[m,1]], mode="lines",
                    line=dict(color="#a88bff", width=1.2), opacity=.30, showlegend=False, hoverinfo="skip"))

    if show_candidate and r["slow"]:
        cx = r["slow"]["candidate_x"]
        for m in range(cx.shape[0]):
            for k in range(cx.shape[1]):
                if cx[m,k] and not x[m,k]:
                    fig.add_trace(go.Scatter(x=[ap[m,0],u[k,0]], y=[ap[m,1],u[k,1]], mode="lines",
                        line=dict(color="#ffbf5a", dash="dash", width=1), opacity=.35, showlegend=False, hoverinfo="skip"))
    if show_prediction and r["slow"]:
        pu, pt = r["slow"]["pred_users"], r["slow"]["pred_targets"]
        for k in range(pu.shape[0]):
            fig.add_trace(go.Scatter(x=pu[k,:,0], y=pu[k,:,1], mode="lines",
                line=dict(color="#a88bff", dash="dot", width=1.4), opacity=.7, showlegend=False, hoverinfo="skip"))
        for q in range(pt.shape[0]):
            fig.add_trace(go.Scatter(x=pt[q,:,0], y=pt[q,:,1], mode="lines",
                line=dict(color="#a88bff", dash="dot", width=1.4), opacity=.7, showlegend=False, hoverinfo="skip"))
    fig.update_layout(
        height=565, margin=dict(l=8,r=8,t=8,b=8),
        plot_bgcolor="#0b1119", paper_bgcolor="#0b1119",
        font=dict(color="#aebdcd", size=10),
        xaxis=dict(title="x position", gridcolor="#17222f", zeroline=False),
        yaxis=dict(title="y position", gridcolor="#17222f", zeroline=False),
        legend=dict(orientation="h", y=1.02, x=0, font=dict(size=9)),
        hovermode="closest",
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def make_df(trace: dict[str, Any]) -> pd.DataFrame:
    rows=[]
    for r in trace["steps"]:
        f, s = r["fast"], r["slow"]; d = s["decision_info"] if s else {}
        rows.append({
            "step": r["t"], "cycle": r["tau"],
            "action": s["decision"] if s else "FAST",
            "reconfigure": int(bool(s and s["decision"] == "RECONFIGURE")),
            "cluster_changes": _f(_g(d,"raw_churn")),
            "net_benefit": _f(_g(d,"net_gain")),
            "expected_benefit": _f(_g(d,"predicted_gain")),
            "sum_rate_bps": _f(_g(f,"sum_rate_bps","communication_sum_rate_bps")),
            "objective": _f(_g(f,"objective")),
            "P_comm": _f(_g(f,"P_comm")), "P_sens": _f(_g(f,"P_sens")), "P_tx": _f(_g(f,"P_tx")),
            "power_violation": _f(_g(f,"max_power_violation")),
            "VaR": _f(s["var_cand"]) if s else np.nan,
            "CVaR": _f(s["cvar_cand"]) if s else np.nan,
        })
    return pd.DataFrame(rows)


def cluster_table(x: np.ndarray, yt: np.ndarray, yr: np.ndarray) -> pd.DataFrame:
    rows=[]
    for k in range(x.shape[1]):
        rows.append({"Entity": f"User U{k+1}", "Role":"Communication", "Serving APs": ", ".join(f"AP{i+1}" for i in np.flatnonzero(x[:,k])) or "—"})
    for q in range(yt.shape[1]):
        rows.append({"Entity": f"Target T{q+1}", "Role":"Sensing TX", "Serving APs": ", ".join(f"AP{i+1}" for i in np.flatnonzero(yt[:,q])) or "—"})
        rows.append({"Entity": f"Target T{q+1}", "Role":"Sensing RX", "Serving APs": ", ".join(f"AP{i+1}" for i in np.flatnonzero(yr[:,q])) or "—"})
    return pd.DataFrame(rows)


def chart_layout(fig: go.Figure, height: int = 330) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=8,r=8,t=24,b=8), plot_bgcolor="#0b1119", paper_bgcolor="#0b1119",
                      font=dict(color="#aebdcd", size=10), legend=dict(font=dict(size=9)),
                      xaxis=dict(gridcolor="#17222f", zeroline=False), yaxis=dict(gridcolor="#17222f", zeroline=False))
    return fig


def action_block(s: dict[str, Any] | None, df: pd.DataFrame, idx: int) -> None:
    if not s:
        st.markdown('<div class="action-card"><div class="panel-title">Controller status</div><span class="action fast">FAST UPDATE</span><div class="reason">The cluster configuration is being held between slow decision points while the physical layer adapts to the current positions.</div><div class="minirow"><span class="label">Next cluster review</span><span class="value">controller schedule</span></div><div class="minirow"><span class="label">Current step rate</span><span class="value">'+fmt_rate(df.iloc[idx].sum_rate_bps)+'</span></div></div>', unsafe_allow_html=True)
        return
    decision = str(s["decision"])
    d = s["decision_info"]
    if decision == "KEEP":
        reason = "Current clusters are retained because the risk-adjusted benefit of changing them is not sufficient at this decision point."
    elif decision == "RECONFIGURE":
        reason = "The controller selects a new cluster configuration because the expected future benefit exceeds the modeled switching cost after uncertainty is considered."
    else:
        reason = f"Controller action: {decision}."
    badge = "keep" if decision == "KEEP" else "reconfig"
    st.markdown(
        f'<div class="action-card"><div class="panel-title">Cluster controller</div><span class="action {badge}">{decision}</span>'
        f'<div class="reason">{reason}</div>'
        f'<div class="minirow"><span class="label">Current objective</span><span class="value">{fmt_num(_f(_g(d,"current_objective")),2)}</span></div>'
        f'<div class="minirow"><span class="label">Predicted objective</span><span class="value">{fmt_num(_f(_g(d,"predicted_objective")),2)}</span></div>'
        f'<div class="minirow"><span class="label">Expected benefit of changing</span><span class="value">{fmt_num(_f(_g(d,"predicted_gain")),2)}</span></div>'
        f'<div class="minirow"><span class="label">Reconfiguration cost</span><span class="value">{fmt_num(_f(_g(d,"churn_cost")),2)}</span></div>'
        f'<div class="minirow"><span class="label">Risk-adjusted net benefit</span><span class="value">{fmt_num(_f(_g(d,"net_gain")),2)}</span></div>'
        f'<div class="minirow"><span class="label">Cluster changes</span><span class="value">{fmt_num(_f(_g(d,"raw_churn")),0)}</span></div>'
        f'</div>', unsafe_allow_html=True)


def main() -> None:
    st.markdown(
        '<div class="brand"><div class="brand-left"><div class="eyebrow">CF-ISAC // NETWORK CONTROL</div>'
        '<div class="title">Predictive Mobile Cell-Free ISAC</div>'
        '<div class="sub">Observe movement → forecast future positions → control clusters → adapt the physical layer</div></div>'
        '<div class="brand-right">ACTUAL PROJECT RUNTIME<br>NO FALLBACK PHYSICS</div></div>',
        unsafe_allow_html=True,
    )

    if "trace" not in st.session_state: st.session_state.trace = None
    if "error" not in st.session_state: st.session_state.error = None
    if "playing" not in st.session_state: st.session_state.playing = False
    if "step_idx" not in st.session_state: st.session_state.step_idx = 0

    trace = st.session_state.trace
    if trace is not None:
        st.markdown('<div class="statusbar"><span class="pill"><span class="dot green"></span>Runtime connected</span>'
                    '<span class="pill"><span class="dot cyan"></span>Two-timescale control</span>'
                    '<span class="pill"><span class="dot amber"></span>Risk-aware decision</span>'
                    '<span class="pill"><span class="dot"></span>Replay mode</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="statusbar"><span class="pill"><span class="dot"></span>Waiting for simulation</span>'
                    '<span class="pill"><span class="dot cyan"></span>Actual project runtime</span></div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("## Scenario")
        M = st.number_input("Access points", 1, 100, 20, help="Number of APs in the deployment.")
        K = st.number_input("Communication users", 1, 50, 5)
        Q = st.number_input("Sensing targets", 1, 20, 2)
        N = st.number_input("Antennas per AP", 1, 32, 4)
        area = st.number_input("Deployment area", 10.0, 5000.0, 500.0, 50.0)

        st.markdown("## Time & control")
        total = st.number_input("Simulation steps", 1, 500, 30)
        slow = st.number_input("Cluster review every", 1, 100, 5, help="Slow-timescale interval. Fast beamforming runs every step.")
        pmax = st.number_input("Maximum AP transmit power", 1e-9, 100.0, 1.0, 0.1)
        pred_kind = st.selectbox("Movement prediction", ["cv","lstm"], format_func=lambda x: "Constant velocity" if x == "cv" else "LSTM")
        prec = st.selectbox("Beamforming", ["mrt","rzf"], format_func=lambda x: "MRT — matched filter" if x == "mrt" else "RZF — interference-aware")

        st.markdown("## Comm + sensing")
        alpha = st.number_input("Communication weight", 0.0, 100.0, 1.0, 0.1)
        beta = st.number_input("Sensing weight", 0.0, 100.0, 1.0, 0.1)

        st.markdown("## Risk model")
        scenarios = st.number_input("Empirical future scenarios", 0, 500, 100)
        cvar_a = st.slider("CVaR confidence", 0.50, 0.99, 0.90, 0.01)
        with st.expander("Seeds / reproducibility"):
            fast_seed = st.number_input("Fast-update seed", 0, 1000000, 7)
            scenario_seed = st.number_input("Scenario seed", 0, 1000000, 0)
            mobility_seed = st.number_input("Mobility seed", 0, 1000000, 42)
        with st.expander("Map options"):
            show_pred = st.checkbox("Show predicted paths", True)
            show_cand = st.checkbox("Show candidate links", True)
        run = st.button("▶  RUN SIMULATION", use_container_width=True, type="primary")
        cols = st.columns(2)
        with cols[0]: clear = st.button("Clear", use_container_width=True)
        with cols[1]:
            if trace is not None:
                play = st.button("Pause" if st.session_state.playing else "Play", use_container_width=True)
            else:
                play = False
        st.markdown('<div class="footer-note">The dashboard is an observability layer. Algorithms and physics remain in src/.</div>', unsafe_allow_html=True)

    if clear:
        st.session_state.trace = None; st.session_state.error = None; st.session_state.playing = False; st.session_state.step_idx = 0; st.rerun()
    if play:
        st.session_state.playing = not st.session_state.playing
        st.rerun()
    if run:
        try:
            t0 = time.perf_counter()
            sim = build_simulator(num_users=int(K), num_targets=int(Q), area_size=float(area), dt=1.0, seed=int(mobility_seed))
            aps = build_ap_positions(num_aps=int(M), area_size=float(area), seed=int(mobility_seed))
            predictor = build_predictor(pred_kind)
            st.session_state.trace = run_observable_trace(
                sim, predictor, aps, num_antennas=int(N), t_total=int(total), t_slow=int(slow), p_max=float(pmax),
                precoder=prec, alpha_power=float(alpha), beta_power=float(beta), num_scenarios=int(scenarios),
                cvar_alpha=float(cvar_a), base_seed=int(fast_seed), scenario_base_seed=int(scenario_seed),
            )
            st.session_state.trace["runtime_seconds"] = time.perf_counter() - t0
            st.session_state.error = None
            st.session_state.step_idx = 0
        except Exception as exc:
            st.session_state.trace = None; st.session_state.error = str(exc); st.session_state.playing = False

    if st.session_state.error:
        st.error("The actual project runtime stopped before the dashboard could render.")
        st.code(st.session_state.error)
        st.info("No surrogate physics is used. Fix the existing project runtime/adapter and run again.")
        return
    trace = st.session_state.trace
    if trace is None:
        st.markdown('<div class="panel" style="margin-top:18px"><div class="panel-title">Ready</div>'
                    '<div style="font-size:1.05rem;font-weight:750;color:#edf3fa">Configure the scenario and press <b>RUN SIMULATION</b>.</div>'
                    '<div class="panel-note" style="margin-top:6px">The live view will show APs, moving users, sensing targets, active associations, predicted movement, controller decisions, physical-layer performance, and empirical CVaR telemetry.</div></div>', unsafe_allow_html=True)
        return

    df = make_df(trace)
    idx = int(st.session_state.step_idx)
    idx = max(0, min(idx, len(df)-1))
    st.session_state.step_idx = idx
    r = trace["steps"][idx]; s = r["slow"]; f = r["fast"]; d = s["decision_info"] if s else {}

    rate = _f(_g(f,"sum_rate_bps","communication_sum_rate_bps"))
    cvar = s["cvar_cand"] if s else np.nan
    changes = _f(_g(d,"raw_churn"))
    action = s["decision"] if s else "FAST"
    metric_grid = st.container()
    with metric_grid:
        st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
        cols = st.columns(4)
        with cols[0]: metric_card("Simulation time", f"Step {idx} / {len(df)-1}", f"Controller cycle {r['tau']}")
        with cols[1]: metric_card("Cluster action", action, "Slow controller" if s else "Between reviews")
        with cols[2]: metric_card("Communication rate", fmt_rate(rate), "Current physical-layer result")
        with cols[3]: metric_card("Worst-case average", fmt_num(cvar,2) if np.isfinite(cvar) else "—", "CVaR at this decision point")
        st.markdown('</div>', unsafe_allow_html=True)

    # Timeline controls
    c1,c2,c3 = st.columns([5,1,1])
    with c1:
        new_idx = st.slider("Simulation timeline", 0, len(df)-1, idx, label_visibility="collapsed")
        if new_idx != st.session_state.step_idx:
            st.session_state.step_idx = new_idx; st.rerun()
    with c2:
        if st.button("◀", use_container_width=True, help="Previous step"):
            st.session_state.step_idx = max(0, idx-1); st.rerun()
    with c3:
        if st.button("▶", use_container_width=True, help="Next step"):
            st.session_state.step_idx = min(len(df)-1, idx+1); st.rerun()

    tabs = st.tabs(["Network overview", "Performance", "Associations", "Research details"])

    with tabs[0]:
        left,right = st.columns([1.7,1.0], gap="medium")
        with left:
            st.markdown('<div class="panel"><div class="panel-title">Live network topology</div>', unsafe_allow_html=True)
            st.plotly_chart(network_figure(trace, idx, show_pred, show_cand), use_container_width=True, config={"displaylogo":False, "scrollZoom":True})
            st.markdown('<div class="legend-row"><span class="legend-item"><span class="legend-line"></span>Active communication</span>'
                        '<span class="legend-item"><span class="legend-line" style="border-top-color:#ffbf5a"></span>Sensing transmit</span>'
                        '<span class="legend-item"><span class="legend-line" style="border-top-color:#a88bff"></span>Sensing receive</span>'
                        '<span class="legend-item"><span class="legend-dash"></span>Candidate link</span>'
                        '<span class="legend-item"><span class="legend-dot"></span>Predicted path</span></div></div>', unsafe_allow_html=True)
        with right:
            action_block(s, df, idx)
            st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
            st.markdown('<div class="panel"><div class="panel-title">What the controller sees</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="callout"><strong>Current movement:</strong> {len(r["users"])} users and {len(r["targets"])} sensing targets are being tracked from the project simulator.</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="callout" style="margin-top:8px"><strong>Decision schedule:</strong> clusters are reconsidered every <b>{trace["meta"]["t_slow"]}</b> simulation steps; beamforming/power adaptation runs every step.</div>', unsafe_allow_html=True)
            if s and s["risk_aware"]:
                st.markdown(f'<div class="callout" style="margin-top:8px"><strong>Risk layer:</strong> the controller evaluates <b>{s["num_scenarios"]}</b> empirical future-motion scenarios at <b>{s["alpha"]:.0%}</b> CVaR confidence.</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with tabs[1]:
        c1,c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Scatter(x=df.step, y=df.sum_rate_bps/1e6, mode="lines+markers", line=dict(color="#48d5ff",width=2), marker=dict(size=5), name="Communication rate"))
            fig.add_vline(x=idx, line_dash="dot", line_color="#ffffff", opacity=.55)
            st.markdown('<div class="panel"><div class="panel-title">Communication throughput</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_layout(fig), use_container_width=True, config={"displaylogo":False})
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.step,y=df.P_comm,mode="lines",name="Communication",line=dict(color="#48d5ff",width=2)))
            fig.add_trace(go.Scatter(x=df.step,y=df.P_sens,mode="lines",name="Sensing",line=dict(color="#ffbf5a",width=2)))
            fig.add_trace(go.Scatter(x=df.step,y=df.P_tx,mode="lines",name="Total",line=dict(color="#48e0a3",width=2)))
            fig.add_vline(x=idx, line_dash="dot", line_color="#ffffff", opacity=.55)
            st.markdown('<div class="panel"><div class="panel-title">Transmit-power allocation</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_layout(fig), use_container_width=True, config={"displaylogo":False})
            st.markdown('</div>', unsafe_allow_html=True)
        c3,c4 = st.columns(2)
        with c3:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.step,y=df.net_benefit,mode="lines+markers",name="Risk-adjusted net benefit",line=dict(color="#48e0a3",width=2)))
            fig.add_trace(go.Scatter(x=df.step,y=df.expected_benefit,mode="lines",name="Expected benefit",line=dict(color="#a88bff",width=1.5,dash="dash")))
            fig.add_vline(x=idx, line_dash="dot", line_color="#ffffff", opacity=.55)
            st.markdown('<div class="panel"><div class="panel-title">Why clusters change</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_layout(fig), use_container_width=True, config={"displaylogo":False})
            st.markdown('</div>', unsafe_allow_html=True)
        with c4:
            rdf = df.dropna(subset=["VaR","CVaR"])
            fig = go.Figure()
            if len(rdf):
                fig.add_trace(go.Scatter(x=rdf.step,y=rdf.VaR,mode="lines+markers",name="VaR",line=dict(color="#ffbf5a",width=2)))
                fig.add_trace(go.Scatter(x=rdf.step,y=rdf.CVaR,mode="lines+markers",name="CVaR",line=dict(color="#ff6b7a",width=2)))
                fig.add_vline(x=idx, line_dash="dot", line_color="#ffffff", opacity=.55)
            st.markdown('<div class="panel"><div class="panel-title">Prediction uncertainty</div>', unsafe_allow_html=True)
            if len(rdf): st.plotly_chart(chart_layout(fig), use_container_width=True, config={"displaylogo":False})
            else: st.info("Risk telemetry is reported at slow decision epochs.")
            st.markdown('</div>', unsafe_allow_html=True)

    with tabs[2]:
        st.markdown('<div class="panel"><div class="panel-title">Current cluster membership</div>', unsafe_allow_html=True)
        ct = cluster_table(r["x"],r["y_tx"],r["y_rx"])
        st.dataframe(ct, use_container_width=True, hide_index=True, column_config={
            "Entity": st.column_config.TextColumn("Entity", width="small"),
            "Role": st.column_config.TextColumn("Role", width="medium"),
            "Serving APs": st.column_config.TextColumn("Connected access points", width="large"),
        })
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
        st.markdown('<div class="panel"><div class="panel-title">Step-by-step controller log</div>', unsafe_allow_html=True)
        logdf = df[["step","cycle","action","expected_benefit","net_benefit","cluster_changes","VaR","CVaR"]].copy()
        logdf.columns=["Step","Cycle","Action","Expected benefit","Net benefit","Cluster changes","VaR","CVaR"]
        st.dataframe(logdf, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tabs[3]:
        st.markdown('<div class="panel"><div class="panel-title">Technical configuration</div>', unsafe_allow_html=True)
        st.json({"project_root": str(PROJECT_ROOT), "runtime_seconds": trace.get("runtime_seconds"), **trace["meta"],
                 "backend": "src.optimization.risk_aware.risk_aware_slow_update + src.optimization.two_timescale.fast_update",
                 "fallback_physics": False})
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
        st.markdown('<div class="callout"><strong>Interpretation:</strong> CVaR is shown as a risk summary of the empirical future-motion scenarios used by the existing risk-aware controller. The dashboard does not introduce an additional risk model.</div>', unsafe_allow_html=True)
        st.download_button("Export controller trace (CSV)", df.to_csv(index=False).encode(), "cfisac_controller_trace.csv", "text/csv")

    if st.session_state.playing and idx < len(df)-1:
        time.sleep(0.7)
        st.session_state.step_idx = idx + 1
        st.rerun()
    elif st.session_state.playing:
        st.session_state.playing = False


if __name__ == "__main__":
    main()
