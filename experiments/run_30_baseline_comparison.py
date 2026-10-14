"""Reproducible 30-figure CFISAC baseline comparison.

The two added heuristic policies are transparent, project-compatible controls;
they are NOT claimed to be exact reproductions of published algorithms.
External-paper reproduction should only be added after its full equations,
constraints, and channel model have been checked for compatibility.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.simulation.end_to_end import run_simulation

OUT = Path("results/literature_baseline_comparison_30graphs_20261015")
SEEDS = list(range(42, 52))
POLICIES = [
    "STATIC",
    "REACTIVE",
    "PREDICTIVE",
    "PREDICTIVE+CHURN",
    "OBJECTIVE-HYSTERESIS",
    "REACTIVE+CHURN",
]
METRICS = [
    ("avg_user_rate", "Mean user rate", "bps"),
    ("p5_user_rate", "5th-percentile user rate", "bps"),
    ("sensing_utility", "Sensing information gain", "nats"),
    ("tracking_violations", "Tracking violations", "model units"),
    ("qos_violations", "QoS violations", "model units"),
    ("energy", "Modeled energy penalty", "model units"),
    ("fronthaul", "Fronthaul load", "model units"),
    ("cumulative_churn", "Cumulative topology churn", "model units"),
    ("combined_objective", "Combined objective", "normalized objective"),
    ("reconfigurations", "Reconfiguration events", "events per block"),
]


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unavailable"


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing results: {OUT}")
    OUT.mkdir(parents=True)
    (OUT / "decision_traces").mkdir()
    block_frames = []
    seed_summaries = []

    for seed in SEEDS:
        for policy in POLICIES:
            print(f"[run] seed={seed} policy={policy}", flush=True)
            result = run_simulation(
                policy_name=policy,
                config_path="configs/default.yaml",
                seed=seed,
            )
            arrays = {k: v for k, v in result.items()
                      if isinstance(v, np.ndarray) and v.ndim == 1}
            frame = pd.DataFrame(arrays)
            frame["seed"] = seed
            frame["policy"] = policy
            frame["block"] = np.arange(len(frame))
            block_frames.append(frame)

            row = {"seed": seed, "policy": policy,
                   "runtime_s": float(result.get("runtime_s", np.nan))}
            for key, _, _ in METRICS:
                if key in frame:
                    row[f"{key}_mean"] = float(frame[key].mean())
                    row[f"{key}_std_block"] = float(frame[key].std(ddof=1))
            row["total_reconfigurations"] = float(frame["reconfigurations"].sum())
            row["final_cumulative_churn"] = float(frame["cumulative_churn"].iloc[-1])
            row["keep_decisions"] = int(result.get("keep_decisions", 0))
            row["reconfigure_decisions"] = int(result.get("reconfigure_decisions", 0))
            seed_summaries.append(row)

            # The simulator writes the proposed method's decision trace to a
            # shared path; archive each trace immediately to avoid collisions.
            trace = Path("results/decision_trace.csv")
            if policy == "PREDICTIVE+CHURN" and trace.exists():
                shutil.copy2(trace, OUT / "decision_traces" / f"decision_trace_seed_{seed}.csv")

    blocks = pd.concat(block_frames, ignore_index=True)
    summaries = pd.DataFrame(seed_summaries)
    blocks.to_csv(OUT / "block_metrics.csv", index=False)
    summaries.to_csv(OUT / "per_seed_summary.csv", index=False)

    # Policy-level mean and standard deviation across independent seeds.
    agg_rows = []
    for policy, group in summaries.groupby("policy", sort=False):
        row = {"policy": policy, "n_seeds": int(group["seed"].nunique())}
        for key, _, _ in METRICS:
            col = f"{key}_mean"
            if col in group:
                row[f"{col}_across_seeds"] = float(group[col].mean())
                row[f"{col}_seed_std"] = float(group[col].std(ddof=1))
        row["reconfigurations_seed_mean"] = float(group["total_reconfigurations"].mean())
        row["reconfigurations_seed_std"] = float(group["total_reconfigurations"].std(ddof=1))
        row["runtime_s_seed_mean"] = float(group["runtime_s"].mean())
        agg_rows.append(row)
    aggregate = pd.DataFrame(agg_rows)
    aggregate.to_csv(OUT / "aggregate_summary.csv", index=False)

    # Exactly 30 plots: 10 multi-seed time traces, 10 across-seed mean
    # comparisons, and 10 per-seed distribution plots.
    for key, title, unit in METRICS:
        # 1-10: mean trajectory over the paired seed set.
        fig, ax = plt.subplots(figsize=(10, 6))
        for policy in POLICIES:
            g = blocks[blocks["policy"] == policy].groupby("block")[key]
            mean = g.mean()
            std = g.std().fillna(0)
            ax.plot(mean.index, mean.values, label=policy)
            ax.fill_between(mean.index, (mean - std).values, (mean + std).values, alpha=0.12)
        ax.set(title=f"{title} vs time (mean ± 1 SD across seeds)",
               xlabel="Time block", ylabel=f"{title} ({unit})")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize="small")
        fig.tight_layout()
        fig.savefig(OUT / f"01_timeseries_{key}.png", dpi=180)
        plt.close(fig)

        # 11-20: across-seed mean policy comparison.
        col = f"{key}_mean"
        fig, ax = plt.subplots(figsize=(10, 6))
        data = [summaries.loc[summaries.policy == p, col].dropna().values for p in POLICIES]
        means = [float(np.mean(v)) if len(v) else np.nan for v in data]
        stds = [float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for v in data]
        ax.bar(POLICIES, means, yerr=stds, capsize=4)
        ax.set(title=f"{title}: baseline comparison across {len(SEEDS)} paired seeds",
               xlabel="Policy", ylabel=f"Per-seed episode mean ({unit})")
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / f"02_seed_means_{key}.png", dpi=180)
        plt.close(fig)

        # 21-30: distribution of per-seed episode means, not per-block pseudoreplicates.
        fig, ax = plt.subplots(figsize=(10, 6))
        data = [summaries.loc[summaries.policy == p, col].dropna().values for p in POLICIES]
        ax.boxplot(data, tick_labels=POLICIES, showmeans=True)
        ax.set(title=f"{title}: distribution across seeds",
               xlabel="Policy", ylabel=f"Per-seed episode mean ({unit})")
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / f"03_seed_distributions_{key}.png", dpi=180)
        plt.close(fig)

    manifest = {
        "experiment": "30-figure CFISAC baseline comparison",
        "seeds": SEEDS,
        "policies": POLICIES,
        "time_blocks_per_episode": int(blocks["block"].max() + 1),
        "metrics": [{"key": k, "label": label, "unit": unit} for k, label, unit in METRICS],
        "figures_expected": 30,
        "figures_created": len(list(OUT.glob("*.png"))),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_branch": git_value("branch", "--show-current"),
        "interpretation_note": (
            "STATIC/REACTIVE/PREDICTIVE/PREDICTIVE+CHURN are existing project policies. "
            "OBJECTIVE-HYSTERESIS and REACTIVE+CHURN are transparent heuristic controls, "
            "not exact reproductions of published papers. The plots are internal controlled "
            "comparisons and must not be described as external literature reproductions."
        ),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    if manifest["figures_created"] != 30:
        raise RuntimeError(f"Expected 30 figures, got {manifest['figures_created']}")


if __name__ == "__main__":
    main()
