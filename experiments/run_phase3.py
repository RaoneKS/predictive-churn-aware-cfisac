"""
experiments/run_phase3.py

Phase 3 — Four-Policy Research Experiment.

Runs a 2 (mobility) × 2 (predictor) × 4 (policy) = 16-condition evaluation.

Mobility:
  - CV (constant_velocity)
  - Stochastic (OU-process correlated velocity)

Predictor (for PREDICTIVE and PREDICTIVE+CHURN only):
  - CV: constant-velocity predictor (uses simulator velocity state)
  - LSTM: trained LSTM (requires checkpoints in results/prediction/)

Policies:
  - STATIC
  - REACTIVE
  - PREDICTIVE
  - PREDICTIVE+CHURN

STATIC and REACTIVE do not depend on a predictor — they are included once
per mobility condition (not doubled).

Metrics saved:
  - results/phase3/block_metrics.csv     — per-block per-condition
  - results/phase3/summary.csv           — aggregated summary
  - results/phase3/plots/               — time-series plots per metric

Usage:
  PYTHONPATH=. python experiments/run_phase3.py [--seed 42] [--blocks 100] [--force]
"""

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.environment.mobility_configs import STOCHASTIC_CONFIG
from src.prediction.predictor_interface import CVPredictor, make_predictor
from src.simulation.end_to_end import run_simulation


CONFIG_PATH = "configs/default.yaml"
RESULTS_DIR = "results/phase3"

# Policies that depend on a predictor
PREDICTIVE_POLICIES = ["PREDICTIVE", "PREDICTIVE+CHURN"]
# Policies that are mobility-only (no predictor dependency)
BASELINE_POLICIES   = ["STATIC", "REACTIVE"]


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def summarise(block_df, warmup_blocks=0):
    """
    Aggregate per-block metrics to a per-condition summary.

    Two evaluation windows are always reported:

      window = "full"         : every simulated block (nothing discarded)
      window = "post_warmup"  : blocks >= warmup_blocks only

    Rationale
    ---------
    LSTMPredictor requires `history_length` steps of position history before it
    can run; until then it legitimately falls back to constant-velocity
    prediction.  During those blocks the CV and LSTM conditions are IDENTICAL
    BY CONSTRUCTION, so predictor-specific comparisons over the full episode
    are contaminated.  The full episode is still reported in its entirety --
    no data is deleted -- and the warm-up blocks are flagged in
    block_metrics.csv via the `warmup` column.
    """
    frames = [("full", block_df)]
    if warmup_blocks > 0:
        frames.append(("post_warmup", block_df[block_df["block"] >= warmup_blocks]))

    rows = []
    for window, frame in frames:
        for (mob, pred, pol), sub in frame.groupby(["mobility", "predictor", "policy"]):
            rows.append({
                "window":             window,
                "mobility":           mob,
                "predictor":          pred,
                "policy":             pol,
                "avg_rate_mean":      sub["avg_user_rate"].mean(),
                "p5_rate_mean":       sub["p5_user_rate"].mean(),
                "sensing_mean":       sub["sensing_utility"].mean(),
                "tracking_viol_mean": sub["tracking_violations"].mean(),
                "energy_mean":        sub["energy"].mean(),
                "fronthaul_mean":     sub["fronthaul"].mean(),
                "churn_mean":         sub["churn"].mean(),
                "cum_churn":          sub["churn"].sum(),
                "reconfigs":          sub["reconfigurations"].sum(),
                "objective_mean":     sub["combined_objective"].mean(),
                "qos_viol_mean":      sub["qos_violations"].mean(),
            })
    return pd.DataFrame(rows)


def plot_metric(block_df, metric, title, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, mob in zip(axes, ["CV mobility", "Stochastic mobility"]):
        sub = block_df[block_df["mobility"] == mob]
        for (pred, pol), grp in sub.groupby(["predictor", "policy"]):
            grp = grp.sort_values("block")
            ax.plot(grp["block"], grp[metric], label=f"{pol}/{pred}", linewidth=0.9)
        ax.set_title(mob)
        ax.set_xlabel("Time block")
        ax.set_ylabel(metric)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{metric}.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def run_condition(
    policy, mobility_name, mob_config, predictor,
    seed, time_blocks, pred_label,
):
    """Run one simulation condition and return annotated block DataFrame."""
    t0 = time.perf_counter()
    metrics = run_simulation(
        policy_name=policy,
        config_path=CONFIG_PATH,
        seed=seed,
        predictor=predictor,
        mobility_config=mob_config,
        time_blocks=time_blocks,
    )
    elapsed = time.perf_counter() - t0

    # Build per-block DataFrame
    block_arrays = {k: v for k, v in metrics.items() if isinstance(v, np.ndarray)}
    df = pd.DataFrame(block_arrays)
    df["block"]    = np.arange(len(df))
    df["policy"]   = policy
    df["mobility"] = mobility_name
    df["predictor"]= pred_label
    df["seed"]     = seed
    df["runtime_s"]= elapsed

    print(
        f"  {policy:20s} | {mobility_name:22s} | pred={pred_label:4s} | "
        f"obj_mean={df['combined_objective'].mean():.4f} | "
        f"recon={df['reconfigurations'].sum():4d} | "
        f"t={elapsed:.1f}s"
    )
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--blocks", type=int, default=100)
    parser.add_argument("--force",  action="store_true",
                        help="Retrain LSTM checkpoints if missing")
    parser.add_argument("--warmup-blocks", type=int, default=None,
                        help="Blocks during which the LSTM predictor lacks "
                             "sufficient history and falls back to CV. "
                             "Defaults to prediction.history_length. Flagged "
                             "in block_metrics.csv and used as the cut point "
                             "for the post_warmup summary window. No data is "
                             "discarded.")
    parser.add_argument("--out-dir", type=str, default=RESULTS_DIR,
                        help="Output directory (kept separate per experiment "
                             "variant so earlier result sets are never "
                             "overwritten)")
    args = parser.parse_args()

    cfg       = load_config()
    area_size = float(cfg["network"]["area_size"])
    L         = int(cfg["prediction"]["history_length"])

    warmup   = args.warmup_blocks if args.warmup_blocks is not None else L
    out_dir  = args.out_dir
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # ── Predictor setup ───────────────────────────────────────────────────────
    cv_pred = CVPredictor()

    # LSTM checkpoints from Phase 1 / 2×2 experiment
    # Try stochastic checkpoints first (trained on stochastic mobility), fall
    # back to CV-mobility-trained ones if stochastic ones don't exist yet.
    def _lstm_ckpt(entity, mob):
        """Return the best available checkpoint path for entity and mobility."""
        candidates = [
            f"results/prediction/lstm_{entity}_{mob}.pt",  # from compare_predictors_grid
            f"results/prediction/lstm_{entity}.pt",         # from original training
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return None

    lstm_cv_user    = _lstm_ckpt("users",   "cv")
    lstm_cv_target  = _lstm_ckpt("targets", "cv")
    lstm_st_user    = _lstm_ckpt("users",   "stochastic")
    lstm_st_target  = _lstm_ckpt("targets", "stochastic")

    # If LSTM checkpoints are missing, train them inline
    def _ensure_lstm(mob_label, mob_config, user_ckpt, target_ckpt):
        if user_ckpt and target_ckpt:
            return user_ckpt, target_ckpt
        print(f"  Training LSTM for {mob_label} mobility (missing checkpoints)...")
        from src.prediction.train_lstm import train as train_lstm
        for entity in ["users", "targets"]:
            train_lstm(
                config_path=CONFIG_PATH,
                entity=entity,
                num_steps=2000,
                checkpoint_dir="results/prediction",
                verbose=False,
                mobility_config=mob_config,
                ckpt_suffix=f"_{mob_label}",
            )
        return (
            f"results/prediction/lstm_users_{mob_label}.pt",
            f"results/prediction/lstm_targets_{mob_label}.pt",
        )

    lstm_cv_user,   lstm_cv_target  = _ensure_lstm(
        "cv", None, lstm_cv_user, lstm_cv_target)
    lstm_st_user,   lstm_st_target  = _ensure_lstm(
        "stochastic", STOCHASTIC_CONFIG, lstm_st_user, lstm_st_target)

    def _make_lstm(user_ckpt, target_ckpt):
        """
        Build an LSTM predictor.

        Normally this is the torch-backed LSTMPredictor.  If torch is not
        installed in the execution environment, fall back to the numpy
        re-implementation in tools/np_lstm.py, which has been validated
        against torch-produced reference metrics (tools/validate_np_lstm.py).
        The fallback is announced loudly so results are never silently
        attributed to the wrong implementation.
        """
        try:
            return make_predictor("lstm",
                                  user_ckpt_path=user_ckpt,
                                  target_ckpt_path=target_ckpt,
                                  area_size=area_size,
                                  history_length=L), "torch"
        except ImportError:
            from tools.np_lstm import NumpyLSTMPredictor
            return NumpyLSTMPredictor(user_ckpt_path=user_ckpt,
                                      target_ckpt_path=target_ckpt,
                                      area_size=area_size,
                                      history_length=L), "numpy"

    lstm_cv_pred,    _backend = _make_lstm(lstm_cv_user, lstm_cv_target)
    lstm_stoch_pred, _backend = _make_lstm(lstm_st_user, lstm_st_target)
    if _backend == "numpy":
        print("  !! torch unavailable -- using VALIDATED numpy LSTM forward pass")
    print(f"  LSTM backend: {_backend}")

    # ── Experiment matrix ─────────────────────────────────────────────────────
    # Each entry: (mobility_name, mob_config, predictor_object, pred_label)
    mobility_conditions = [
        ("CV mobility",         None,           cv_pred,        lstm_cv_pred,    "cv",    "lstm_cv"),
        ("Stochastic mobility", STOCHASTIC_CONFIG, cv_pred,    lstm_stoch_pred, "cv",   "lstm_stoch"),
    ]

    all_blocks = []

    print("=" * 80)
    print("PHASE 3 — 4-POLICY × 2-MOBILITY × 2-PREDICTOR EXPERIMENT")
    print(f"  seed={args.seed}  blocks={args.blocks}  warmup_blocks={warmup}")
    print(f"  Blocks 0..{warmup - 1} are LSTM warm-up (CV fallback, identical "
          f"to the CV condition by construction).")
    print("=" * 80)

    for mob_name, mob_cfg, cv_p, lstm_p, cv_label, lstm_label in mobility_conditions:
        print(f"\n── {mob_name} ──")

        # Baseline policies (no predictor dependency, run once)
        for pol in BASELINE_POLICIES:
            df = run_condition(pol, mob_name, mob_cfg, None,
                               args.seed, args.blocks, "N/A")
            all_blocks.append(df)

        # Predictive policies with CV predictor
        for pol in PREDICTIVE_POLICIES:
            df = run_condition(pol, mob_name, mob_cfg, cv_p,
                               args.seed, args.blocks, cv_label)
            all_blocks.append(df)

        # Predictive policies with LSTM predictor
        for pol in PREDICTIVE_POLICIES:
            df = run_condition(pol, mob_name, mob_cfg, lstm_p,
                               args.seed, args.blocks, lstm_label)
            all_blocks.append(df)

    # ── Save results ──────────────────────────────────────────────────────────
    block_df   = pd.concat(all_blocks, ignore_index=True)
    block_df["warmup"] = block_df["block"] < warmup
    summary_df = summarise(block_df, warmup_blocks=warmup)

    block_path   = os.path.join(out_dir, "block_metrics.csv")
    summary_path = os.path.join(out_dir, "summary.csv")
    block_df.to_csv(block_path,   index=False)
    summary_df.to_csv(summary_path, index=False)

    # ── Plots ─────────────────────────────────────────────────────────────────
    for metric, title in [
        ("avg_user_rate",       "Average User Rate (bps)"),
        ("p5_user_rate",        "5th-Percentile User Rate (bps)"),
        ("sensing_utility",     "Sensing Information Gain (nats)"),
        ("tracking_violations", "Tracking Violations"),
        ("energy",              "Energy (W)"),
        ("fronthaul",           "Fronthaul Load"),
        ("churn",               "Churn Cost"),
        ("cumulative_churn",    "Cumulative Churn"),
        ("combined_objective",  "Combined Objective"),
        ("reconfigurations",    "Reconfiguration Events"),
    ]:
        try:
            plot_metric(block_df, metric, title, plot_dir)
        except Exception as e:
            print(f"  Warning: could not plot {metric}: {e}")

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("PHASE 3 SUMMARY")
    print("=" * 100)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(summary_df.to_string(index=False))
    print("=" * 100)
    print(f"\nSaved: {block_path}")
    print(f"Saved: {summary_path}")
    print(f"Plots: {plot_dir}/")


if __name__ == "__main__":
    main()
