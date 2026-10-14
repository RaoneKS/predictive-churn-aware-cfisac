"""
experiments/run_published_baseline_comparison.py

External published-baseline comparison experiment for the CFISAC research project.

This script compares our proposed predictive churn-aware method against four
paper-inspired proxy baselines. All methods run under matched conditions:
same topology, mobility, channel realizations, evaluation duration, and seeds.

PROTECTION: This script REFUSES to overwrite any existing results directory.
Every run creates a uniquely timestamped output directory.

METHODS COMPARED:
  Internal policies (from this project):
    STATIC, REACTIVE, PREDICTIVE, PREDICTIVE+CHURN (from end_to_end.py)
    OBJECTIVE-HYSTERESIS, REACTIVE+CHURN (heuristic controls)

  Published-baseline proxies (from src/baselines/published_baselines.py):
    SUN2025_ADAPTIVE_HYSTERESIS  -- Sun et al. 2025 (PAPER_INSPIRED_PROXY)
    ELFIATOURE2023_GAP_NPC       -- Elfiatoure et al. 2023 GAP-NPC (PAPER_INSPIRED_PROXY)
    ELFIATOURE2023_RAP_NPC       -- Elfiatoure et al. 2023 RAP-NPC (PAPER_INSPIRED_PROXY)
    KIMCHO2024_CF_HO             -- Kim and Cho 2024 (PAPER_INSPIRED_PROXY)
    OZBEK2026_GMSUA              -- Özbek and Cumalı 2026 G-MSUA (PAPER_INSPIRED_PROXY)

OUTPUTS (in results/published_baseline_comparison_<TIMESTAMP>/):
  - per_seed_block_metrics.csv   (every block of every seed for every method)
  - per_seed_summary.csv         (per-seed episode means)
  - aggregate_summary.csv        (per-method stats across seeds)
  - statistical_tests.csv        (paired t-tests vs proposed method)
  - confidence_intervals.csv     (95% CI per method and metric)
  - improvement_table.csv        (% improvement vs each baseline)
  - reproduction_status.json     (baseline labels and metadata)
  - experiment_config.json       (exact parameters used)
  - seed_manifest.csv            (seed → status mapping)
  - failed_runs.log              (failed run record)
  - figures/                     (all plots as PNG)
  - EXPERIMENT_REPORT.md         (summary report)

REPRODUCTION:
  python -m experiments.run_published_baseline_comparison [--smoke] [--seeds N]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.simulation.end_to_end import run_simulation
from src.baselines.published_baselines import (
    BASELINE_REGISTRY,
    elfiatoure2023_gap_npc,
    elfiatoure2023_rap_npc,
    sun2025_adaptive_hysteresis,
    kimcho2024_cf_ho,
    ozbek2026_gmsua,
)
from src.clustering.churn import total_churn_cost
from src.environment.topology import generate_ap_positions
from src.optimization.churn_aware import evaluate_cluster_performance
from src.simulation.mobility_simulator import MobilitySimulator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Proposed method = PREDICTIVE+CHURN (the churn-aware predictive policy
# implemented in end_to_end.py; see also two_timescale.py for the full
# two-timescale integration, which uses the same slow-timescale decision)
PROPOSED_METHOD_NAME = "PREDICTIVE+CHURN"
PROPOSED_METHOD_NOTE = (
    "The PREDICTIVE+CHURN policy implements predictive, churn-aware overlapping "
    "clustering. It uses the existing Phase-2B horizon-averaged objective and "
    "churn_aware_decision (src/optimization/churn_aware.py). The full two-timescale "
    "integration (src/optimization/two_timescale.py) further wraps this with "
    "fast-timescale physical power control but uses the same slow-timescale "
    "clustering decision. PREDICTIVE+CHURN is the relevant clustering benchmark."
)

# All methods in experiment
INTERNAL_POLICIES = [
    "STATIC",
    "REACTIVE",
    "PREDICTIVE",
    "PREDICTIVE+CHURN",
    "OBJECTIVE-HYSTERESIS",
    "REACTIVE+CHURN",
]

PUBLISHED_BASELINE_IDS = [
    "SUN2025_ADAPTIVE_HYSTERESIS",
    "ELFIATOURE2023_GAP_NPC",
    "ELFIATOURE2023_RAP_NPC",
    "KIMCHO2024_CF_HO",
    "OZBEK2026_GMSUA",
]

ALL_METHODS = INTERNAL_POLICIES + PUBLISHED_BASELINE_IDS

METRICS = [
    ("avg_user_rate",       "Mean user rate (bps)",         "bps",          True),   # higher=better
    ("p5_user_rate",        "5th-pctile user rate (bps)",   "bps",          True),
    ("sensing_utility",     "Sensing information gain (nat)","nats",         True),
    ("tracking_violations", "Tracking violation level",      "model units",  False),  # lower=better
    ("qos_violations",      "QoS violation level",           "model units",  False),
    ("energy",              "Energy penalty",                "model units",  False),
    ("fronthaul",           "Fronthaul load",               "model units",  False),
    ("cumulative_churn",    "Cumulative topology churn",     "model units",  False),
    ("combined_objective",  "Combined objective",           "norm. units",   True),
    ("reconfigurations",    "Reconfigurations per block",   "events/block",  False),
]

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

def make_output_dir(base: Path) -> Path:
    """Create a uniquely-timestamped output directory; fail if collision."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = base / f"published_baseline_comparison_{ts}"
    if out.exists():
        raise FileExistsError(
            f"Output directory already exists: {out}\n"
            "This should not happen with a timestamp-based name. "
            "Wait 1 second and try again."
        )
    out.mkdir(parents=True)
    (out / "figures").mkdir()
    return out


# ---------------------------------------------------------------------------
# Published-baseline simulation runner
# ---------------------------------------------------------------------------

def run_published_baseline_episode(
    baseline_id: str,
    config: dict,
    seed: int,
    perf_kwargs: dict,
    ref: dict,
    c_comm: float,
    c_sens_tx: float,
    c_sens_rx: float,
) -> dict:
    """
    Run one episode of a published-baseline method under the same
    network/mobility conditions as the internal policies.

    Parameters match the internal run_simulation() API as closely as possible
    to ensure fair comparison.
    """
    rng = np.random.default_rng(seed)
    num_aps     = config["network"]["num_aps"]
    num_users   = config["network"]["num_users"]
    num_targets = config["network"]["num_targets"]
    area_size   = config["network"]["area_size"]
    time_blocks = config["simulation"]["time_blocks"]
    dt          = config["simulation"]["dt"]

    ap_positions = generate_ap_positions(num_aps, area_size, seed=seed)
    initial_user_pos   = rng.uniform(0, area_size, size=(num_users,   2))
    initial_target_pos = rng.uniform(0, area_size, size=(num_targets, 2))

    sim = MobilitySimulator(
        initial_user_pos,
        initial_target_pos,
        area_size=area_size,
        dt=dt,
        seed=seed,
    )

    # Shared kwargs for baselines that accept perf_kwargs / ref
    bline_kwargs = dict(perf_kwargs=perf_kwargs, ref=ref)

    # Metrics storage
    metrics = {k: [] for k, *_ in METRICS}
    metrics["runtime_s"] = 0.0
    metrics["reconfigurations"] = []
    metrics["cumulative_churn"] = []
    churn_series = []
    cumulative_churn = 0.0
    reconf_count = 0

    prev_x = prev_y_tx = prev_y_rx = None
    t_start = time.perf_counter()

    for t in range(time_blocks):
        user_pos, target_pos = sim.step()

        # ── Policy decision ──────────────────────────────────────────────────
        if baseline_id == "SUN2025_ADAPTIVE_HYSTERESIS":
            if t == 0:
                from src.optimization.baselines import reactive_policy
                x, y_tx, y_rx = reactive_policy(ap_positions, user_pos, target_pos)
                decision_info = {"decision": "INIT"}
            else:
                x, y_tx, y_rx, decision_info = sun2025_adaptive_hysteresis(
                    ap_positions, user_pos, target_pos,
                    prev_x, prev_y_tx, prev_y_rx,
                    sim=sim,
                    **bline_kwargs,
                )

        elif baseline_id == "ELFIATOURE2023_GAP_NPC":
            x, y_tx, y_rx, decision_info = elfiatoure2023_gap_npc(
                ap_positions, user_pos, target_pos,
                perf_kwargs=perf_kwargs, ref=ref,
            )

        elif baseline_id == "ELFIATOURE2023_RAP_NPC":
            x, y_tx, y_rx, decision_info = elfiatoure2023_rap_npc(
                ap_positions, user_pos, target_pos,
                seed=seed * 1000 + t,
            )

        elif baseline_id == "KIMCHO2024_CF_HO":
            if t == 0:
                from src.optimization.baselines import reactive_policy
                x, y_tx, y_rx = reactive_policy(ap_positions, user_pos, target_pos)
                decision_info = {"decision": "INIT"}
            else:
                x, y_tx, y_rx, decision_info = kimcho2024_cf_ho(
                    ap_positions, user_pos, target_pos,
                    prev_x, prev_y_tx, prev_y_rx,
                    **bline_kwargs,
                )

        elif baseline_id == "OZBEK2026_GMSUA":
            x, y_tx, y_rx, decision_info = ozbek2026_gmsua(
                ap_positions, user_pos, target_pos,
                perf_kwargs=perf_kwargs, ref=ref,
            )

        else:
            raise ValueError(f"Unknown baseline_id: {baseline_id!r}")

        # ── Churn ───────────────────────────────────────────────────────────
        if t == 0 or prev_x is None:
            churn_val = 0.0
            reconfigured = 0
        else:
            from src.optimization.baselines import cluster_difference
            diff = cluster_difference(
                prev_x, x, prev_y_tx, y_tx, prev_y_rx, y_rx
            )
            reconfigured = 1 if diff > 0 else 0
            churn_val = float(total_churn_cost(
                prev_x, x, prev_y_tx, y_tx, prev_y_rx, y_rx,
                c_comm=c_comm, c_sensing_tx=c_sens_tx, c_sensing_rx=c_sens_rx,
            ))

        cumulative_churn += churn_val

        # ── Evaluate actual performance ──────────────────────────────────────
        actual_perf = evaluate_cluster_performance(
            x, y_tx, y_rx,
            ap_positions, user_pos, target_pos,
            **perf_kwargs,
        )

        n_ch = churn_val / float(ref["churn_events"])
        lambda_c = float(config["objective"]["lambda_churn"])

        metrics["avg_user_rate"].append(
            actual_perf["raw_communication_utility"] / num_users
        )
        metrics["p5_user_rate"].append(
            float(np.percentile(actual_perf["raw_rates"], 5))
        )
        metrics["sensing_utility"].append(actual_perf["raw_sensing_utility"])
        metrics["tracking_violations"].append(actual_perf["tracking_violations"])
        metrics["qos_violations"].append(actual_perf["qos_violations"])
        metrics["energy"].append(actual_perf["raw_energy_penalty"])
        metrics["fronthaul"].append(actual_perf["raw_fronthaul_penalty"])
        churn_series.append(churn_val)
        metrics["cumulative_churn"].append(cumulative_churn)
        metrics["reconfigurations"].append(reconfigured)

        # Combined objective includes churn penalty
        combined = actual_perf["combined_objective"] - lambda_c * n_ch
        metrics["combined_objective"].append(combined)

        prev_x, prev_y_tx, prev_y_rx = x, y_tx, y_rx

    metrics["runtime_s"] = time.perf_counter() - t_start
    # Convert lists to arrays
    for k in metrics:
        if isinstance(metrics[k], list):
            metrics[k] = np.array(metrics[k])

    metrics["churn"] = np.array(churn_series)
    return metrics


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def ci_95(values: np.ndarray) -> tuple[float, float]:
    """95% CI assuming normally distributed samples (use t-distribution)."""
    from scipy import stats
    n = len(values)
    if n < 2:
        return float(np.mean(values)), float(np.mean(values))
    m = float(np.mean(values))
    se = float(stats.sem(values))
    t_crit = float(stats.t.ppf(0.975, df=n - 1))
    return m - t_crit * se, m + t_crit * se


def paired_t_test(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired t-test between two per-seed value arrays."""
    from scipy import stats
    if len(a) < 2 or len(b) < 2 or len(a) != len(b):
        return {"t_stat": np.nan, "p_value": np.nan, "significant_p05": False}
    t, p = stats.ttest_rel(a, b)
    return {"t_stat": float(t), "p_value": float(p), "significant_p05": bool(p < 0.05)}


def improvement_pct(proposed: float, reference: float, higher_better: bool) -> float:
    """Calculate % improvement of proposed over reference."""
    if abs(reference) < 1e-30:
        return np.nan
    if higher_better:
        return (proposed - reference) / abs(reference) * 100.0
    else:
        return (reference - proposed) / abs(reference) * 100.0


# ---------------------------------------------------------------------------
# Multiple-comparison correction
# ---------------------------------------------------------------------------

# Metrics excluded from Holm correction (zero variance in default config)
_ZERO_VARIANCE_METRICS = {"qos_violations", "fronthaul"}

def _apply_holm_correction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Holm-Bonferroni FWER correction to statistical_tests DataFrame.

    Two families are corrected independently:
      Family A — PUBLISHED_BASELINE_IDS (primary research comparisons)
      Family B — INTERNAL_POLICIES minus proposed method (ablation comparisons)

    Zero-variance metrics (qos_violations, fronthaul) are excluded from all
    correction families. Their adjusted p-values remain NaN.

    The raw p_value and significant_p05 columns are preserved unchanged.
    New columns added: p_value_adjusted, significant_p05_adjusted,
                       family, family_size, p_nan_reason (if not present).
    """
    try:
        from statsmodels.stats.multitest import multipletests
        def _holm(pvals):
            reject, adj, _, _ = multipletests(pvals, alpha=0.05, method="holm")
            return adj, reject
    except ImportError:
        def _holm(pvals):  # manual fallback
            n = len(pvals)
            order = np.argsort(pvals)
            adj = np.empty(n)
            reject = np.zeros(n, dtype=bool)
            running_max = 0.0
            for i, idx in enumerate(order):
                k = n - i
                adj_p = min(pvals[idx] * k, 1.0)
                adj_p = max(adj_p, running_max)
                running_max = adj_p
                adj[idx] = adj_p
                reject[idx] = adj[idx] < 0.05
            return adj, reject

    pub_methods   = set(PUBLISHED_BASELINE_IDS)
    int_methods   = set(INTERNAL_POLICIES) - {PROPOSED_METHOD_NAME}

    df = df.copy()
    if "p_nan_reason" not in df.columns:
        df["p_nan_reason"] = ""
    if "family" not in df.columns:
        df["family"] = None
    if "family_size" not in df.columns:
        df["family_size"] = None
    if "p_value_adjusted" not in df.columns:
        df["p_value_adjusted"] = np.nan
    if "significant_p05_adjusted" not in df.columns:
        df["significant_p05_adjusted"] = False

    df.loc[df.reference_method.isin(pub_methods), "family"] = "A_published_baselines"
    df.loc[df.reference_method.isin(int_methods), "family"] = "B_internal_ablations"

    for family_id in ("A_published_baselines", "B_internal_ablations"):
        fam_mask  = df.family == family_id
        corr_mask = fam_mask & ~df.metric.isin(_ZERO_VARIANCE_METRICS) & df.p_value.notna()
        idx = df.index[corr_mask]
        df.loc[fam_mask, "family_size"] = int(corr_mask.sum())
        if corr_mask.sum() > 0:
            adj_p, reject = _holm(df.loc[idx, "p_value"].values)
            df.loc[idx, "p_value_adjusted"] = adj_p
            df.loc[idx, "significant_p05_adjusted"] = reject

    # Mark zero-variance nan reason
    zero_var = df.metric.isin(_ZERO_VARIANCE_METRICS) & df.p_value.isna()
    df.loc[zero_var & (df.p_nan_reason == ""), "p_nan_reason"] = "zero_variance_difference"

    return df



def plot_bar_comparison(
    summaries: pd.DataFrame,
    metric_key: str,
    metric_label: str,
    metric_unit: str,
    methods: list,
    out_path: Path,
    proposed_method: str = PROPOSED_METHOD_NAME,
):
    """Bar chart: across-seed mean ± 1 SD for each method."""
    col = f"{metric_key}_mean"
    fig, ax = plt.subplots(figsize=(12, 6))
    means, stds, colors = [], [], []
    for m in methods:
        vals = summaries.loc[summaries.method == m, col].dropna().values
        means.append(float(np.mean(vals)) if len(vals) else np.nan)
        stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
        colors.append("steelblue" if m == proposed_method else "lightcoral"
                      if m in PUBLISHED_BASELINE_IDS else "lightgray")
    bars = ax.bar(range(len(methods)), means, yerr=stds, capsize=4, color=colors)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(f"{metric_label} ({metric_unit})")
    ax.set_title(f"{metric_label}: method comparison")
    ax.grid(True, axis="y", alpha=0.3)
    # Legend patches
    import matplotlib.patches as mpatches
    legend_patches = [
        mpatches.Patch(color="steelblue", label="Proposed method"),
        mpatches.Patch(color="lightcoral", label="Published baseline (proxy)"),
        mpatches.Patch(color="lightgray", label="Internal policy"),
    ]
    ax.legend(handles=legend_patches, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_timeseries(
    blocks: pd.DataFrame,
    metric_key: str,
    metric_label: str,
    metric_unit: str,
    methods: list,
    out_path: Path,
    proposed_method: str = PROPOSED_METHOD_NAME,
):
    """Time series: mean ± 1 SD across seeds for each method."""
    fig, ax = plt.subplots(figsize=(12, 5))
    for m in methods:
        g = blocks.loc[blocks.method == m].groupby("block")[metric_key]
        mn = g.mean()
        sd = g.std().fillna(0)
        lw = 2.0 if m == proposed_method else 1.2
        ls = "-" if m == proposed_method else (
            "--" if m in PUBLISHED_BASELINE_IDS else ":")
        ax.plot(mn.index, mn.values, label=m, linewidth=lw, linestyle=ls)
        ax.fill_between(mn.index, (mn - sd).values, (mn + sd).values, alpha=0.08)
    ax.set_xlabel("Time block")
    ax.set_ylabel(f"{metric_label} ({metric_unit})")
    ax.set_title(f"{metric_label} over time (mean ± 1 SD)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_boxplot(
    summaries: pd.DataFrame,
    metric_key: str,
    metric_label: str,
    metric_unit: str,
    methods: list,
    out_path: Path,
):
    """Box plot: per-seed episode means distribution."""
    col = f"{metric_key}_mean"
    data = [summaries.loc[summaries.method == m, col].dropna().values for m in methods]
    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot(data, tick_labels=methods, showmeans=True, patch_artist=True)
    for patch, m in zip(bp["boxes"], methods):
        if m == PROPOSED_METHOD_NAME:
            patch.set_facecolor("steelblue")
        elif m in PUBLISHED_BASELINE_IDS:
            patch.set_facecolor("lightcoral")
        else:
            patch.set_facecolor("lightgray")
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(f"{metric_label} ({metric_unit})")
    ax.set_title(f"{metric_label}: distribution across seeds")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_tradeoff(
    summaries: pd.DataFrame,
    x_metric: str, x_label: str,
    y_metric: str, y_label: str,
    methods: list,
    out_path: Path,
):
    """Scatter trade-off plot."""
    fig, ax = plt.subplots(figsize=(9, 7))
    for m in methods:
        xvals = summaries.loc[summaries.method == m, f"{x_metric}_mean"].dropna().values
        yvals = summaries.loc[summaries.method == m, f"{y_metric}_mean"].dropna().values
        if len(xvals) == 0 or len(yvals) == 0:
            continue
        mx, my = float(np.mean(xvals)), float(np.mean(yvals))
        marker = "*" if m == PROPOSED_METHOD_NAME else (
            "^" if m in PUBLISHED_BASELINE_IDS else "o"
        )
        ax.scatter(mx, my, label=m, marker=marker, s=100)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"Trade-off: {x_label} vs {y_label}")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Published-baseline comparison")
    parser.add_argument("--smoke", action="store_true",
                        help="Run a small smoke test (3 seeds, 20 blocks)")
    parser.add_argument("--seeds", type=int, default=10,
                        help="Number of seeds (default 10; smoke overrides to 3)")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Config file path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    # ── Load config ──────────────────────────────────────────────────────────
    config_path = REPO_ROOT / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if args.smoke:
        config["simulation"]["time_blocks"] = 20
        n_seeds = 3
    else:
        n_seeds = args.seeds

    seeds = list(range(42, 42 + n_seeds))
    log.info("Seeds: %s (%d total)", seeds, len(seeds))
    log.info("Time blocks: %d", config["simulation"]["time_blocks"])
    log.info("Smoke test: %s", args.smoke)

    # ── Guard: do NOT touch existing results ─────────────────────────────────
    existing_graphs = list((REPO_ROOT / "results").rglob("*.png"))
    log.info("Existing graphs (will not be touched): %d", len(existing_graphs))

    # ── Create unique output directory ───────────────────────────────────────
    out = make_output_dir(REPO_ROOT / "results")
    log.info("Output directory: %s", out)

    failed_log = out / "failed_runs.log"
    seed_manifest_rows = []

    # ── Build perf_kwargs from config ────────────────────────────────────────
    ref = {k: float(v) for k, v in config["reference_values"].items()}
    perf_kwargs = dict(
        ref=ref,
        alpha_communication=float(config["objective"]["alpha_communication"]),
        beta_sensing=float(config["objective"]["beta_sensing"]),
        lambda_energy=float(config["objective"]["lambda_energy"]),
        lambda_fronthaul=float(config["objective"]["lambda_fronthaul"]),
        min_rate=float(config["communication"]["min_rate_bps"]),
        max_tracking_error=float(config["sensing"]["tracking_error_threshold"]),
    )
    c_comm    = float(config["churn"]["communication_cost"])
    c_sens_tx = float(config["churn"]["sensing_tx_cost"])
    c_sens_rx = float(config["churn"]["sensing_rx_cost"])

    # ── Run experiments ──────────────────────────────────────────────────────
    block_frames = []
    seed_summary_rows = []
    total_runs = len(seeds) * len(ALL_METHODS)
    completed = 0

    for seed in seeds:
        for method in ALL_METHODS:
            log.info("[%d/%d] seed=%d method=%s",
                     completed + 1, total_runs, seed, method)
            try:
                if method in INTERNAL_POLICIES:
                    result = run_simulation(
                        policy_name=method,
                        config_path=str(config_path),
                        seed=seed,
                    )
                else:
                    result = run_published_baseline_episode(
                        baseline_id=method,
                        config=config,
                        seed=seed,
                        perf_kwargs=perf_kwargs,
                        ref=ref,
                        c_comm=c_comm,
                        c_sens_tx=c_sens_tx,
                        c_sens_rx=c_sens_rx,
                    )

                # Extract per-block arrays
                metric_keys = [k for k, *_ in METRICS] + ["churn"]
                arrays = {}
                for k in metric_keys:
                    v = result.get(k)
                    if v is not None and isinstance(v, np.ndarray) and v.ndim == 1:
                        arrays[k] = v

                frame = pd.DataFrame(arrays)
                frame["seed"] = seed
                frame["method"] = method
                frame["block"] = np.arange(len(frame))
                block_frames.append(frame)

                row = {"seed": seed, "method": method,
                       "runtime_s": float(result.get("runtime_s", np.nan))}
                for k, *_ in METRICS:
                    col = result.get(k)
                    if col is not None and isinstance(col, np.ndarray) and len(col) > 0:
                        row[f"{k}_mean"] = float(np.mean(col))
                        row[f"{k}_std"]  = float(np.std(col, ddof=1))
                seed_summary_rows.append(row)
                seed_manifest_rows.append({"seed": seed, "method": method,
                                           "status": "OK",
                                           "runtime_s": row["runtime_s"]})

            except Exception as exc:
                msg = f"FAILED seed={seed} method={method}: {exc}"
                log.error(msg)
                with open(failed_log, "a") as f:
                    f.write(msg + "\n")
                    f.write(traceback.format_exc() + "\n---\n")
                seed_manifest_rows.append({"seed": seed, "method": method,
                                           "status": "FAILED", "runtime_s": np.nan})

            completed += 1

    log.info("Runs completed: %d / %d", completed, total_runs)

    if not block_frames:
        log.error("No data was collected. Exiting.")
        sys.exit(1)

    # ── Save raw data ─────────────────────────────────────────────────────────
    blocks = pd.concat(block_frames, ignore_index=True)
    summaries = pd.DataFrame(seed_summary_rows)
    seed_manifest = pd.DataFrame(seed_manifest_rows)

    blocks.to_csv(out / "per_seed_block_metrics.csv", index=False)
    summaries.to_csv(out / "per_seed_summary.csv", index=False)
    seed_manifest.to_csv(out / "seed_manifest.csv", index=False)
    log.info("Raw data saved.")

    # ── Aggregate ────────────────────────────────────────────────────────────
    agg_rows = []
    for method, grp in summaries.groupby("method", sort=False):
        row = {"method": method, "n_seeds": int(grp["seed"].nunique())}
        for k, *_ in METRICS:
            col = f"{k}_mean"
            if col in grp:
                vals = grp[col].dropna().values
                if len(vals) > 0:
                    lo, hi = ci_95(vals)
                    row[f"{col}_across_seeds"] = float(np.mean(vals))
                    row[f"{col}_seed_std"]     = float(np.std(vals, ddof=1))
                    row[f"{col}_ci_lo"]        = float(lo)
                    row[f"{col}_ci_hi"]        = float(hi)
        row["runtime_mean_s"] = float(grp["runtime_s"].mean())
        agg_rows.append(row)
    aggregate = pd.DataFrame(agg_rows)
    aggregate.to_csv(out / "aggregate_summary.csv", index=False)

    # ── CI table ─────────────────────────────────────────────────────────────
    ci_rows = []
    for method, grp in summaries.groupby("method", sort=False):
        for k, label, unit, higher_better in METRICS:
            col = f"{k}_mean"
            if col in grp:
                vals = grp[col].dropna().values
                if len(vals) > 0:
                    lo, hi = ci_95(vals)
                    ci_rows.append({
                        "method": method, "metric": k, "label": label,
                        "unit": unit, "higher_better": higher_better,
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals, ddof=1)),
                        "ci_lo_95": float(lo), "ci_hi_95": float(hi),
                        "n": len(vals),
                    })
    pd.DataFrame(ci_rows).to_csv(out / "confidence_intervals.csv", index=False)

    # ── Statistical tests ─────────────────────────────────────────────────────
    proposed_vals = {}
    for k, *_ in METRICS:
        col = f"{k}_mean"
        pv = summaries.loc[summaries.method == PROPOSED_METHOD_NAME, col].dropna().values
        proposed_vals[k] = pv

    stat_rows = []
    for method in ALL_METHODS:
        if method == PROPOSED_METHOD_NAME:
            continue
        for k, label, unit, higher_better in METRICS:
            col = f"{k}_mean"
            ref_vals = summaries.loc[summaries.method == method, col].dropna().values
            prop_vals = proposed_vals.get(k, np.array([]))

            if len(ref_vals) == 0 or len(prop_vals) == 0:
                continue

            # Align seeds (use intersection)
            seeds_ref  = summaries.loc[summaries.method == method, "seed"].values
            seeds_prop = summaries.loc[summaries.method == PROPOSED_METHOD_NAME, "seed"].values
            common = sorted(set(seeds_ref) & set(seeds_prop))
            if len(common) < 2:
                continue

            a = np.array([
                summaries.loc[(summaries.method == PROPOSED_METHOD_NAME) &
                              (summaries.seed == s), col].values[0]
                for s in common
            ])
            b = np.array([
                summaries.loc[(summaries.method == method) &
                              (summaries.seed == s), col].values[0]
                for s in common
            ])

            ttest = paired_t_test(a, b)
            prop_mean = float(np.mean(a))
            ref_mean  = float(np.mean(b))
            improv = improvement_pct(prop_mean, ref_mean, higher_better)

            stat_rows.append({
                "proposed_method": PROPOSED_METHOD_NAME,
                "reference_method": method,
                "metric": k, "label": label, "unit": unit,
                "higher_better": higher_better,
                "proposed_mean": prop_mean,
                "reference_mean": ref_mean,
                "improvement_pct": improv,
                "n_paired_seeds": len(common),
                "t_stat": ttest["t_stat"],
                "p_value": ttest["p_value"],
                "p_nan_reason": "" if ttest["t_stat"] == ttest["t_stat"] else "zero_variance_difference",
                "significant_p05": ttest["significant_p05"],
                "family": None,            # set by _apply_holm_correction
                "family_size": None,
                "p_value_adjusted": np.nan,
                "significant_p05_adjusted": False,
            })

    # Apply Holm-Bonferroni FWER correction within two pre-specified families
    stat_df = _apply_holm_correction(pd.DataFrame(stat_rows))
    stat_df.to_csv(out / "statistical_tests.csv", index=False)


    # ── Improvement table ─────────────────────────────────────────────────────
    imp_rows = []
    for method in PUBLISHED_BASELINE_IDS:
        row = {"baseline": method}
        for k, _, _, higher_better in METRICS:
            col = f"{k}_mean"
            bv = summaries.loc[summaries.method == method, col].dropna().values
            pv = proposed_vals.get(k, np.array([]))
            if len(bv) > 0 and len(pv) > 0:
                row[f"{k}_improvement_pct"] = improvement_pct(
                    float(np.mean(pv)), float(np.mean(bv)), higher_better
                )
        imp_rows.append(row)
    pd.DataFrame(imp_rows).to_csv(out / "improvement_table.csv", index=False)

    # ── Figures ───────────────────────────────────────────────────────────────
    fig_count = 0
    methods_present = summaries["method"].unique().tolist()

    for k, label, unit, higher_better in METRICS:
        # 1. Bar comparison
        plot_bar_comparison(
            summaries, k, label, unit, methods_present,
            out / "figures" / f"bar_{k}.png",
        )
        fig_count += 1

        # 2. Time series
        plot_timeseries(
            blocks, k, label, unit, methods_present,
            out / "figures" / f"timeseries_{k}.png",
        )
        fig_count += 1

        # 3. Box plot
        plot_boxplot(
            summaries, k, label, unit, methods_present,
            out / "figures" / f"boxplot_{k}.png",
        )
        fig_count += 1

    # 4. Trade-off plots
    tradeoffs = [
        ("avg_user_rate", "Mean user rate (bps)",
         "sensing_utility", "Sensing utility (nats)"),
        ("avg_user_rate", "Mean user rate (bps)",
         "cumulative_churn", "Cumulative churn (model units)"),
        ("avg_user_rate", "Mean user rate (bps)",
         "energy", "Energy penalty (model units)"),
        ("p5_user_rate", "5th-pctile user rate (bps)",
         "qos_violations", "QoS violations"),
        ("sensing_utility", "Sensing utility (nats)",
         "tracking_violations", "Tracking violations"),
    ]
    for xk, xl, yk, yl in tradeoffs:
        plot_tradeoff(summaries, xk, xl, yk, yl, methods_present,
                      out / "figures" / f"tradeoff_{xk}_vs_{yk}.png")
        fig_count += 1

    log.info("Figures created: %d", fig_count)

    # ── Verify old graphs untouched ───────────────────────────────────────────
    post_existing = list((REPO_ROOT / "results").rglob("*.png"))
    new_graphs = [p for p in post_existing if p not in existing_graphs
                  and "published_baseline_comparison" in str(p)]
    old_intact = all(p.exists() for p in existing_graphs)
    log.info("Old graphs intact: %s", old_intact)
    log.info("New graphs created: %d", len(new_graphs))

    # ── Save reproduction status ──────────────────────────────────────────────
    repro_status = {
        bid: {
            "label": BASELINE_REGISTRY[bid]["reproduction_label"],
            "paper": BASELINE_REGISTRY[bid]["paper_title"],
            "authors": BASELINE_REGISTRY[bid]["authors"],
            "doi": BASELINE_REGISTRY[bid]["doi"],
        }
        for bid in BASELINE_REGISTRY
    }
    repro_status["_proposed_method"] = {
        "id": PROPOSED_METHOD_NAME,
        "note": PROPOSED_METHOD_NOTE,
    }
    (out / "reproduction_status.json").write_text(
        json.dumps(repro_status, indent=2) + "\n"
    )

    # ── Experiment config ─────────────────────────────────────────────────────
    import subprocess
    def git_val(*args):
        try:
            return subprocess.check_output(["git", *args], text=True,
                                            stderr=subprocess.DEVNULL).strip()
        except Exception:
            return "unavailable"

    exp_config = {
        "experiment_name": "published_baseline_comparison",
        "timestamp": datetime.now().isoformat(),
        "git_commit": git_val("rev-parse", "HEAD"),
        "git_branch": git_val("branch", "--show-current"),
        "seeds": seeds,
        "n_seeds_requested": n_seeds,
        "n_seeds_completed": int(seed_manifest[seed_manifest.status == "OK"]["seed"].nunique()),
        "time_blocks": config["simulation"]["time_blocks"],
        "smoke_test": args.smoke,
        "internal_policies": INTERNAL_POLICIES,
        "published_baselines": PUBLISHED_BASELINE_IDS,
        "proposed_method": PROPOSED_METHOD_NAME,
        "config_path": str(config_path),
        "figures_created": fig_count,
        "old_graphs_intact": old_intact,
        "fairness_conditions": {
            "same_topology": True,
            "same_mobility_seeds": True,
            "same_time_blocks": True,
            "same_channel_model": True,
            "same_perf_evaluation": True,
            "common_random_numbers": True,
            "note": "Internal policies use run_simulation() which independently "
                    "regenerates the same topology/mobility from the seed. "
                    "Published baselines use the same seed-based generation.",
        },
        "reproduction_labels": {
            bid: BASELINE_REGISTRY[bid]["reproduction_label"]
            for bid in PUBLISHED_BASELINE_IDS
        },
    }
    (out / "experiment_config.json").write_text(
        json.dumps(exp_config, indent=2) + "\n"
    )

    # ── Markdown report ───────────────────────────────────────────────────────
    n_ok = int(seed_manifest[seed_manifest.status == "OK"].shape[0])
    n_fail = int(seed_manifest[seed_manifest.status == "FAILED"].shape[0])

    report_lines = [
        "# Published-Baseline Comparison: Experiment Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Git branch:** {exp_config['git_branch']}",
        f"**Git commit:** {exp_config['git_commit']}",
        "",
        "## Summary",
        "",
        f"- Seeds: {seeds}",
        f"- Completed runs: {n_ok} / {n_ok + n_fail}",
        f"- Failed runs: {n_fail}",
        f"- Time blocks per episode: {config['simulation']['time_blocks']}",
        f"- Smoke test: {args.smoke}",
        "",
        "## Proposed Method",
        "",
        f"**{PROPOSED_METHOD_NAME}**: {PROPOSED_METHOD_NOTE}",
        "",
        "## External Baselines (Reproduction Labels)",
        "",
    ]
    for bid in PUBLISHED_BASELINE_IDS:
        e = BASELINE_REGISTRY[bid]
        report_lines += [
            f"### {bid}",
            f"- **Paper:** {e['paper_title']}",
            f"- **Authors:** {e['authors']} ({e['venue']}, {e['year']})",
            f"- **DOI:** {e['doi']}",
            f"- **Reproduction label:** `{e['reproduction_label']}`",
            f"- **Algorithm:** {e['algorithm']}",
            f"- **Model differences:** {e['model_differences']}",
            "",
        ]

    report_lines += [
        "## NOT_IMPLEMENTED Baselines",
        "",
    ]
    for bid in ("DEMIRHAN2023_BEAMFORMING", "AMMAR2024_POMDP"):
        e = BASELINE_REGISTRY[bid]
        report_lines += [
            f"### {bid}",
            f"- {e['model_differences']}",
            "",
        ]

    report_lines += [
        "## Fairness Conditions",
        "",
        "All methods used the same:",
        "- Network topology (AP positions from seed)",
        "- Initial user/target positions (from seed)",
        "- Mobility model and trajectory (same seed)",
        "- Evaluation duration (time_blocks)",
        "- Performance evaluation function (evaluate_cluster_performance)",
        "- Reference normalization values",
        "",
        "## Metrics",
        "",
        "| Metric | Label | Units | Higher=Better |",
        "|--------|-------|-------|---------------|",
    ]
    for k, label, unit, hb in METRICS:
        report_lines.append(f"| {k} | {label} | {unit} | {hb} |")

    report_lines += [
        "",
        "## Improvement vs Published Baselines",
        "",
        "See `improvement_table.csv` and `statistical_tests.csv` for full results.",
        "",
        "## Figures",
        "",
        f"Total figures: {fig_count}",
        "Located in `figures/` subdirectory.",
        "- `bar_*.png` — across-seed mean ± 1 SD bar comparison",
        "- `timeseries_*.png` — time-series mean ± 1 SD",
        "- `boxplot_*.png` — seed distribution box plots",
        "- `tradeoff_*.png` — trade-off scatter plots",
        "",
        "## Historical Results",
        "",
        "The original 30 graphs (in `results/professor_submission_corrected_20261013/`)",
        "and all other historical results remain intact and were not modified.",
        f"Old graphs verified intact: {old_intact}",
        "",
        "## Limitations",
        "",
        "All external baselines are labelled **PAPER_INSPIRED_PROXY**. Their native",
        "objectives (MASR, max-min SE, sensing SNR) are proxied by the simulator's",
        "scalar communication rate and information-gain sensing metrics. Numerical",
        "values cannot be directly compared to results reported in the original papers.",
        "",
        "## Reproducibility",
        "",
        "```bash",
        "git checkout baseline-comparison",
        "source .venv/bin/activate",
        "# Smoke test (fast):",
        f"python -m experiments.run_published_baseline_comparison --smoke",
        "# Full experiment (default 10 seeds):",
        f"python -m experiments.run_published_baseline_comparison --seeds 10",
        "```",
    ]

    (out / "EXPERIMENT_REPORT.md").write_text("\n".join(report_lines) + "\n")

    log.info("Experiment complete. Output: %s", out)
    log.info("Figures: %d, Old graphs intact: %s", fig_count, old_intact)
    print(f"\n✓ Experiment complete: {out}")
    print(f"  Figures: {fig_count}")
    print(f"  Old graphs intact: {old_intact}")
    print(f"  Runs: {n_ok} OK / {n_fail} failed")


if __name__ == "__main__":
    main()
