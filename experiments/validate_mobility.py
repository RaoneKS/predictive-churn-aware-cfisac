"""
experiments/validate_mobility.py

Mobility validation for the constant-velocity and stochastic
correlated-velocity regimes.

This script does NOT depend on torch / the LSTM predictor -- it only
exercises src.simulation.mobility_simulator and
src.environment.stochastic_mobility, so it can be run and verified
independently of experiments/compare_mobility.py.

Reports, per mobility regime and per entity (users / targets):
  - speed statistics (mean, std, min, max, median)
  - velocity autocorrelation (lag 1..K)
  - a GENUINE boundary-reflection event rate, from the simulator's real
    geometric reflection events (MobilitySimulator.*_boundary_event_counts)
  - velocity_sign_flip_rate: a SEPARATE, explicitly-labeled diagnostic.
    This is NOT a boundary-interaction metric. Under the stochastic
    regime, additive velocity noise can flip a velocity component's sign
    with no boundary contact at all, so this number must never be
    interpreted as "how often objects hit a wall".
  - sample trajectory plots

Outputs: results/mobility_validation/
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.simulation.mobility_simulator import MobilitySimulator
from src.environment.mobility_configs import STOCHASTIC_CONFIG


CONFIG_PATH = "configs/default.yaml"
RESULTS_DIR = "results/mobility_validation"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def run_regime(mobility_name, mobility_config, net_cfg, seed, num_steps):
    """
    Run the mobility simulator for num_steps and collect raw trajectories,
    velocities, and genuine boundary-event counts for users and targets.
    """
    rng = np.random.default_rng(seed)
    area_size = float(net_cfg["area_size"])
    num_users = int(net_cfg["num_users"])
    num_targets = int(net_cfg["num_targets"])

    init_users = rng.uniform(0, area_size, size=(num_users, 2))
    init_targets = rng.uniform(0, area_size, size=(num_targets, 2))

    sim = MobilitySimulator(
        init_users,
        init_targets,
        area_size=area_size,
        dt=1.0,
        seed=seed,
        mobility_config=mobility_config,
        history_capacity=num_steps + 1,
    )

    user_pos_hist, user_vel_hist = [], []
    target_pos_hist, target_vel_hist = [], []

    for _ in range(num_steps):
        sim.step()
        user_pos_hist.append(sim.user_positions.copy())
        user_vel_hist.append(sim.user_velocities.copy())
        target_pos_hist.append(sim.target_positions.copy())
        target_vel_hist.append(sim.target_velocities.copy())

    return {
        "mobility_name": mobility_name,
        "user_positions": np.asarray(user_pos_hist),     # (T, N_u, 2)
        "user_velocities": np.asarray(user_vel_hist),     # (T, N_u, 2)
        "target_positions": np.asarray(target_pos_hist),  # (T, N_t, 2)
        "target_velocities": np.asarray(target_vel_hist),  # (T, N_t, 2)
        "user_boundary_event_counts": sim.user_boundary_event_counts.copy(),
        "target_boundary_event_counts": sim.target_boundary_event_counts.copy(),
        "num_steps": num_steps,
    }


def speed_statistics(velocities):
    """velocities: (T, N, 2) -> dict of scalar stats over all objects/steps."""
    speeds = np.linalg.norm(velocities, axis=-1)   # (T, N)
    return {
        "mean_speed_m_per_block": float(np.mean(speeds)),
        "std_speed_m_per_block": float(np.std(speeds)),
        "min_speed_m_per_block": float(np.min(speeds)),
        "max_speed_m_per_block": float(np.max(speeds)),
        "median_speed_m_per_block": float(np.median(speeds)),
    }


def velocity_autocorrelation(velocities, max_lag=10):
    """
    Per-component (x, y) velocity autocorrelation, averaged across objects
    and both components, for lags 1..max_lag.

    Returns
    -------
    dict: {"lag_1": corr, "lag_2": corr, ...}
    """
    T, N, _ = velocities.shape
    max_lag = min(max_lag, T - 2)
    result = {}

    for lag in range(1, max_lag + 1):
        corrs = []
        for obj in range(N):
            for comp in range(2):
                series = velocities[:, obj, comp]
                a = series[:-lag]
                b = series[lag:]
                if np.std(a) < 1e-10 or np.std(b) < 1e-10:
                    continue
                c = np.corrcoef(a, b)[0, 1]
                if np.isfinite(c):
                    corrs.append(c)
        result[f"lag_{lag}"] = float(np.mean(corrs)) if corrs else float("nan")

    return result


def velocity_sign_flip_rate(velocities):
    """
    Fraction of consecutive-step transitions where a velocity component's
    sign changed, averaged over objects and both components.

    IMPORTANT: this is NOT a boundary-interaction metric. Under stochastic
    mobility, additive velocity noise can flip a component's sign with no
    boundary contact whatsoever. Use the genuine
    *_boundary_event_counts-derived rate for actual boundary interactions.
    """
    T, N, _ = velocities.shape
    signs = np.sign(velocities)
    signs[signs == 0] = 1.0
    flips = (signs[1:] != signs[:-1])   # (T-1, N, 2)
    return float(np.mean(flips))


def genuine_boundary_event_rate(boundary_event_counts, num_steps):
    """Mean fraction of steps, per object, that had a real boundary reflection."""
    return float(np.mean(boundary_event_counts) / num_steps)


def plot_sample_trajectories(regime_results, entity, n_samples=3):
    fig, axes = plt.subplots(1, len(regime_results), figsize=(6 * len(regime_results), 5))
    if len(regime_results) == 1:
        axes = [axes]

    key = "user_positions" if entity == "users" else "target_positions"

    for ax, res in zip(axes, regime_results):
        positions = res[key]   # (T, N, 2)
        n = min(n_samples, positions.shape[1])
        for obj in range(n):
            ax.plot(positions[:, obj, 0], positions[:, obj, 1], marker=".", markersize=2, label=f"object {obj}")
        ax.set_title(f"{res['mobility_name']} — {entity}")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_xlim(0, 500)
        ax.set_ylim(0, 500)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

    fig.suptitle(f"Sample Trajectories — {entity}")
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, f"trajectories_{entity}.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_speed_distributions(regime_results, entity):
    key = "user_velocities" if entity == "users" else "target_velocities"

    fig, ax = plt.subplots(figsize=(7, 5))
    for res in regime_results:
        speeds = np.linalg.norm(res[key], axis=-1).ravel()
        ax.hist(speeds, bins=40, alpha=0.5, density=True, label=res["mobility_name"])
    ax.set_xlabel("Speed (m/block)")
    ax.set_ylabel("Density")
    ax.set_title(f"Speed Distribution — {entity}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, f"speed_distribution_{entity}.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_autocorrelation(regime_results, entity):
    key = "user_velocities" if entity == "users" else "target_velocities"

    fig, ax = plt.subplots(figsize=(7, 5))
    for res in regime_results:
        ac = velocity_autocorrelation(res[key], max_lag=10)
        lags = sorted(range(1, len(ac) + 1))
        values = [ac[f"lag_{l}"] for l in lags]
        ax.plot(lags, values, marker="o", label=res["mobility_name"])
    ax.set_xlabel("Lag (blocks)")
    ax.set_ylabel("Velocity autocorrelation")
    ax.set_title(f"Velocity Autocorrelation — {entity}")
    ax.axhline(0.0, color="grey", linewidth=0.8)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, f"velocity_autocorrelation_{entity}.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  Saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    cfg = load_config()
    net_cfg = cfg["network"]

    conditions = [
        ("constant_velocity", None),
        ("stochastic", STOCHASTIC_CONFIG),
    ]

    print("=" * 80)
    print("MOBILITY VALIDATION")
    print("=" * 80)
    print(f"Seed: {args.seed}   Steps: {args.num_steps}")
    print()

    regime_results = []
    summary_rows = []
    autocorr_rows = []

    for mobility_name, mobility_config in conditions:
        print(f"Running regime: {mobility_name}")
        res = run_regime(mobility_name, mobility_config, net_cfg, args.seed, args.num_steps)
        regime_results.append(res)

        for entity, pos_key, vel_key, bcounts_key in [
            ("users", "user_positions", "user_velocities", "user_boundary_event_counts"),
            ("targets", "target_positions", "target_velocities", "target_boundary_event_counts"),
        ]:
            vel = res[vel_key]
            stats = speed_statistics(vel)
            sign_flip = velocity_sign_flip_rate(vel)
            boundary_rate = genuine_boundary_event_rate(res[bcounts_key], res["num_steps"])
            ac = velocity_autocorrelation(vel, max_lag=5)

            row = {
                "mobility": mobility_name,
                "entity": entity,
                **stats,
                "boundary_event_rate": boundary_rate,
                "velocity_sign_flip_rate": sign_flip,
                **{f"velocity_autocorr_{k}": v for k, v in ac.items()},
            }
            summary_rows.append(row)

            print(
                f"  [{mobility_name:>17s} / {entity:7s}] "
                f"mean_speed={stats['mean_speed_m_per_block']:.3f} m/block  "
                f"std={stats['std_speed_m_per_block']:.3f}  "
                f"boundary_event_rate={boundary_rate:.4f}  "
                f"velocity_sign_flip_rate={sign_flip:.4f} "
                f"(NOTE: sign-flip rate is NOT a boundary metric)"
            )

            for lag_key, corr in ac.items():
                autocorr_rows.append({
                    "mobility": mobility_name, "entity": entity,
                    "lag": int(lag_key.split("_")[1]), "autocorrelation": corr,
                })

    summary_df = pd.DataFrame(summary_rows)
    autocorr_df = pd.DataFrame(autocorr_rows)

    summary_path = os.path.join(RESULTS_DIR, "mobility_comparison_summary.csv")
    autocorr_path = os.path.join(RESULTS_DIR, "velocity_autocorrelation.csv")
    summary_df.to_csv(summary_path, index=False)
    autocorr_df.to_csv(autocorr_path, index=False)

    for entity in ["users", "targets"]:
        plot_sample_trajectories(regime_results, entity)
        plot_speed_distributions(regime_results, entity)
        plot_autocorrelation(regime_results, entity)

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    # ── Honest speed-matching statement (do not claim exact speed matching
    #    unless actually achieved) ──────────────────────────────────────
    print()
    print("-" * 80)
    print("MOBILITY REGIME COMPARISON NOTE")
    print("-" * 80)
    for entity in ["users", "targets"]:
        cv_row = summary_df[(summary_df.mobility == "constant_velocity") & (summary_df.entity == entity)].iloc[0]
        st_row = summary_df[(summary_df.mobility == "stochastic") & (summary_df.entity == entity)].iloc[0]
        cv_mean = cv_row["mean_speed_m_per_block"]
        st_mean = st_row["mean_speed_m_per_block"]
        pct_diff = 100.0 * (st_mean - cv_mean) / cv_mean if cv_mean != 0 else float("nan")
        print(
            f"{entity}: constant-velocity mean speed = {cv_mean:.3f} m/block, "
            f"stochastic mean speed = {st_mean:.3f} m/block "
            f"({pct_diff:+.1f}% relative difference)."
        )
    print(
        "The stochastic regime is NOT a speed-matched nonlinear version of "
        "constant-velocity mobility -- it is a materially different mobility "
        "regime with its own speed statistics, reported above. This is "
        "acceptable for a first controlled mobility-regime comparison, but "
        "must not be described as speed-matched."
    )
    print("=" * 80)
    print(f"Saved: {summary_path}")
    print(f"Saved: {autocorr_path}")


if __name__ == "__main__":
    main()
