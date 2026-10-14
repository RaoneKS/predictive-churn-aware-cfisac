#!/usr/bin/env python3
"""Run a corrected, reproducible four-policy, ten-seed CF-ISAC comparison."""

import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Ensure imports work when this script is launched from experiments/.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simulation import end_to_end

ROOT = Path.cwd()
CONFIG = ROOT / "configs/default.yaml"
OUT = ROOT / "results/multiseed_comparison_thresholds_fixed_20261013"
POLICIES = ["STATIC", "REACTIVE", "PREDICTIVE", "PREDICTIVE+CHURN"]
SEEDS = list(range(42, 52))
TIME_BLOCKS = 100

if OUT.exists():
    raise SystemExit(
        f"Refusing to overwrite existing output directory:\n{OUT}"
    )

OUT.mkdir(parents=True)

def git_output(*args):
    try:
        return subprocess.run(
            ["git", *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except OSError:
        return "unavailable"

config_bytes = CONFIG.read_bytes()
config = yaml.safe_load(config_bytes)
config_hash = hashlib.sha256(config_bytes).hexdigest()
commit = git_output("rev-parse", "HEAD")
status_before = git_output("status", "--short")

expected_min_rate = float(config["communication"]["min_rate_bps"])
expected_tracking_threshold = float(
    config["sensing"]["tracking_error_threshold"]
)

print("Output directory:", OUT)
print("Git commit:", commit)
print("Seeds:", SEEDS)
print("Policies:", POLICIES)
print("Time blocks per run:", TIME_BLOCKS)
print("Configured minimum rate (bps):", expected_min_rate)
print("Configured tracking threshold:", expected_tracking_threshold)
print("Config SHA-256:", config_hash)
print()
print("The existing results/decision_trace.csv will not be overwritten.")

# Redirect only the simulation engine's shared decision-trace write.
# All other DataFrame.to_csv calls retain their original behavior.
original_to_csv = pd.DataFrame.to_csv
active_trace_path = None

def redirected_to_csv(self, path_or_buf=None, *args, **kwargs):
    if path_or_buf is not None:
        try:
            supplied_path = os.fspath(path_or_buf)
        except TypeError:
            supplied_path = None

        if supplied_path == "results/decision_trace.csv":
            if active_trace_path is None:
                raise RuntimeError(
                    "Unexpected shared decision-trace write outside a run."
                )
            path_or_buf = str(active_trace_path)

    return original_to_csv(self, path_or_buf, *args, **kwargs)

pd.DataFrame.to_csv = redirected_to_csv

all_blocks = []
seed_summaries = []
trace_files = []
started = time.perf_counter()

try:
    for seed in SEEDS:
        for policy in POLICIES:
            print(
                f"[{len(seed_summaries) + 1:02d}/40] "
                f"seed={seed}, policy={policy}",
                flush=True,
            )

            trace_path = OUT / f"decision_trace_seed_{seed}.csv"
            active_trace_path = (
                trace_path if policy == "PREDICTIVE+CHURN" else None
            )

            run_started = time.perf_counter()
            metrics = end_to_end.run_simulation(
                policy_name=policy,
                config_path=str(CONFIG),
                seed=seed,
                time_blocks=TIME_BLOCKS,
            )
            wall_runtime = time.perf_counter() - run_started

            # Collect all per-block NumPy metrics without changing units.
            block_data = {
                key: value
                for key, value in metrics.items()
                if isinstance(value, np.ndarray)
                and value.ndim == 1
                and len(value) == TIME_BLOCKS
            }

            if not block_data:
                raise RuntimeError(
                    f"No per-block metrics returned for {policy}, seed {seed}."
                )

            block_df = pd.DataFrame(block_data)
            block_df["seed"] = seed
            block_df["policy"] = policy
            block_df["block"] = np.arange(TIME_BLOCKS)
            all_blocks.append(block_df)

            # Match the original per-seed summary conventions.
            seed_summaries.append({
                "seed": seed,
                "policy": policy,
                "avg_user_rate_mbps":
                    float(np.mean(metrics["avg_user_rate"])) / 1e6,
                "p5_user_rate_mbps":
                    float(np.mean(metrics["p5_user_rate"])) / 1e6,
                "sensing_utility_mean":
                    float(np.mean(metrics["sensing_utility"])),
                "tracking_violations_mean":
                    float(np.mean(metrics["tracking_violations"])),
                "qos_violations_mean":
                    float(np.mean(metrics["qos_violations"])),
                "energy_mean": float(np.mean(metrics["energy"])),
                "fronthaul_mean": float(np.mean(metrics["fronthaul"])),
                "cumulative_churn":
                    float(metrics["cumulative_churn"][-1]),
                "reconfigurations":
                    int(np.sum(metrics["reconfigurations"])),
                "keep_decisions":
                    int(metrics.get("keep_decisions", 0)),
                "reconfigure_decisions":
                    int(metrics.get("reconfigure_decisions", 0)),
                "combined_objective_mean":
                    float(np.mean(metrics["combined_objective"])),
                "runtime_s": float(metrics.get("runtime_s", wall_runtime)),
            })

            print(
                f"    avg rate={np.mean(metrics['avg_user_rate']) / 1e6:.3f} Mbps; "
                f"QoS magnitude={np.mean(metrics['qos_violations']):.6g}; "
                f"tracking magnitude={np.mean(metrics['tracking_violations']):.6g}; "
                f"runtime={wall_runtime:.2f}s",
                flush=True,
            )

            active_trace_path = None

            if policy == "PREDICTIVE+CHURN":
                if not trace_path.exists() or trace_path.stat().st_size == 0:
                    raise RuntimeError(
                        f"Missing or empty decision trace for seed {seed}: "
                        f"{trace_path}"
                    )
                trace_files.append(trace_path.name)

finally:
    pd.DataFrame.to_csv = original_to_csv

elapsed = time.perf_counter() - started

blocks = pd.concat(all_blocks, ignore_index=True)
per_seed = pd.DataFrame(seed_summaries)

if len(per_seed) != 40:
    raise RuntimeError(f"Expected 40 per-seed rows; got {len(per_seed)}")

if len(blocks) != 40 * TIME_BLOCKS:
    raise RuntimeError(
        f"Expected {40 * TIME_BLOCKS} block rows; got {len(blocks)}"
    )

blocks.to_csv(OUT / "block_metrics_all_seeds.csv", index=False)
per_seed.to_csv(OUT / "per_seed_summary.csv", index=False)

# Aggregate across independent seeds. Use sample standard deviation (ddof=1).
summary_columns = [
    column for column in per_seed.columns
    if column not in {"seed", "policy", "runtime_s"}
]

aggregate_rows = []
for policy in POLICIES:
    subset = per_seed[per_seed["policy"] == policy]
    row = {"policy": policy}

    for column in summary_columns:
        values = subset[column].astype(float)
        row[f"{column}_mean"] = float(values.mean())
        row[f"{column}_std"] = float(values.std(ddof=1))
        row[f"{column}_sem"] = float(values.std(ddof=1) / np.sqrt(len(values)))

    aggregate_rows.append(row)

aggregate = pd.DataFrame(aggregate_rows)
aggregate.to_csv(OUT / "aggregate_summary.csv", index=False)

metadata = {
    "experiment": "corrected_multiseed_four_policy_comparison",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "seeds": SEEDS,
    "policies": POLICIES,
    "time_blocks_per_run": TIME_BLOCKS,
    "expected_simulations": 40,
    "completed_simulations": len(per_seed),
    "expected_block_rows": 4000,
    "actual_block_rows": len(blocks),
    "configuration": "configs/default.yaml",
    "configuration_sha256": config_hash,
    "configured_min_rate_bps": expected_min_rate,
    "configured_tracking_error_threshold": expected_tracking_threshold,
    "threshold_fix": (
        "run_simulation passes configured communication minimum rate and "
        "tracking threshold to evaluate_cluster_performance; explicit "
        "min_rate and max_tracking_error overrides remain supported"
    ),
    "git_commit_at_start": commit,
    "git_status_before": status_before,
    "git_status_after": git_output("status", "--short"),
    "elapsed_seconds": elapsed,
    "decision_trace_files": trace_files,
    "original_shared_decision_trace_overwritten": False,
    "aggregation": (
        "per-seed means; aggregate mean, sample standard deviation (ddof=1), "
        "and standard error across ten seeds"
    ),
    "metric_interpretation": {
        "qos_violations": (
            "Evaluator's per-block QoS violation magnitude; not a count "
            "of violating users or blocks."
        ),
        "tracking_violations": (
            "Evaluator's per-block tracking violation magnitude; not a count "
            "of violating targets or blocks."
        ),
        "energy": (
            "Simulation-model energy penalty/metric; not physical power "
            "measured on hardware."
        ),
    },
}
(OUT / "experiment_metadata.json").write_text(
    json.dumps(metadata, indent=2) + "\n"
)

print("\n========== EXPERIMENT COMPLETE ==========")
print("Output directory:", OUT)
print(f"Simulations: {len(per_seed)}/40")
print(f"Block rows: {len(blocks)}/4000")
print(f"Elapsed: {elapsed:.2f}s")
print("\nAggregate summary:")
print(aggregate.to_string(index=False))
print("\nOutput files:")
for path in sorted(OUT.iterdir()):
    print(f"  {path.name} ({path.stat().st_size:,} bytes)")
