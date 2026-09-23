"""
experiments/run_phase4a_beamforming_validation.py

Phase 4A — small, deterministic validation experiment comparing:

    (1) EXISTING scalar baseline  (src.metrics.system.communication_sinr /
        communication_rate, exactly as used today in
        src.optimization.churn_aware.evaluate_cluster_performance)
    (2) NEW physical-layer path with MRT (matched-filter) beamforming
    (3) NEW physical-layer path with RZF beamforming

All three conditions share the EXACT same topology (AP/user positions)
and the EXACT same AP-user association matrix (from the existing
communication_clustering function), so the comparison isolates the
effect of the beamforming/physical-layer method itself.

This does not touch prediction, clustering policy, or churn logic -- it
is a static, single-snapshot physical-layer comparison, as scoped for
Phase 4A (no two-timescale optimization, no joint comm+sensing).

Usage:
    PYTHONPATH=. python experiments/run_phase4a_beamforming_validation.py
"""

import argparse
import json
import os
import time

import numpy as np
import yaml

from src.channels.communication import generate_comm_gains
from src.channels.physical import generate_physical_channel
from src.clustering.overlapping import communication_clustering
from src.environment.topology import generate_ap_positions, generate_positions
from src.beamforming.precoders import (
    mrt_weights,
    rzf_weights,
    normalize_per_ap_power,
)
from src.metrics.physical_layer import physical_sinr, physical_rate
from src.metrics.system import communication_sinr, communication_rate


def load_config(path="configs/default.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_scalar_baseline(ap_positions, user_positions, x, transmit_power, noise_power):
    """
    EXACT reproduction of the existing scalar communication model used in
    src.optimization.churn_aware.evaluate_cluster_performance, isolated
    to just the communication metric for a direct comparison.
    """
    t0 = time.perf_counter()

    G_comm = generate_comm_gains(ap_positions, user_positions)
    signal_power = np.sum(x * G_comm, axis=0) * transmit_power
    active_aps = np.sum(x, axis=1)
    total_tx = active_aps * transmit_power
    total_received_power = np.sum(total_tx[:, None] * G_comm, axis=0)
    interference_power = np.maximum(total_received_power - signal_power, 0.0)

    sinr = communication_sinr(signal_power, interference_power, noise_power=noise_power)
    rates = communication_rate(sinr)

    # Total/maximum "transmit power" under the scalar model: every active
    # AP transmits `transmit_power` regardless of how many users it serves
    # (the scalar model has no per-antenna/per-beam power budget).
    per_ap_power = total_tx
    runtime = time.perf_counter() - t0

    return {
        "sinr": sinr,
        "rates": rates,
        "per_ap_power": per_ap_power,
        "runtime_s": runtime,
    }


def run_physical_layer(
    H, x, precoder, tx_power_per_ap, noise_power, bandwidth_hz, pilot_fraction
):
    t0 = time.perf_counter()

    if precoder == "mrt":
        W = mrt_weights(H, x)
    elif precoder == "rzf":
        W = rzf_weights(H, x)
    else:
        raise ValueError(f"Unknown precoder: {precoder}")

    W, per_ap_power = normalize_per_ap_power(W, tx_power_per_ap=tx_power_per_ap)

    sinr = physical_sinr(H, W, noise_power=noise_power)
    rates = physical_rate(sinr, bandwidth_hz=bandwidth_hz, pilot_fraction=pilot_fraction)

    runtime = time.perf_counter() - t0

    return {
        "sinr": sinr,
        "rates": rates,
        "per_ap_power": per_ap_power,
        "runtime_s": runtime,
    }


def summarize(name, result):
    rates_bps = result["rates"]
    sinr = result["sinr"]
    per_ap_power = result["per_ap_power"]

    rates_mbps = rates_bps / 1e6

    return {
        "condition": name,
        "avg_rate_mbps": float(np.mean(rates_mbps)),
        "p5_rate_mbps": float(np.percentile(rates_mbps, 5)),
        "avg_sinr_linear": float(np.mean(sinr)),
        "avg_sinr_db": float(10.0 * np.log10(np.mean(sinr) + 1e-30)),
        "total_tx_power_w": float(np.sum(per_ap_power)),
        "max_per_ap_power_w": float(np.max(per_ap_power)),
        "runtime_s": result["runtime_s"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-aps", type=int, default=None)
    parser.add_argument("--num-users", type=int, default=None)
    parser.add_argument("--aps-per-user", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="results/phase4a")
    args = parser.parse_args()

    config = load_config()
    phys_cfg = config["physical_layer"]
    net_cfg = config["network"]
    comm_cfg = config["communication"]

    num_aps = args.num_aps if args.num_aps is not None else net_cfg["num_aps"]
    num_users = args.num_users if args.num_users is not None else net_cfg["num_users"]
    area_size = net_cfg["area_size"]

    ap_positions = generate_ap_positions(num_aps, area_size, seed=args.seed)
    user_positions = generate_positions(num_users, area_size, seed=args.seed + 1)

    G_comm = generate_comm_gains(ap_positions, user_positions)
    x = communication_clustering(G_comm, num_users=num_users, aps_per_user=args.aps_per_user)

    num_antennas = phys_cfg["num_antennas_per_ap"]
    tx_power_per_ap = phys_cfg["tx_power_per_ap_w"]
    noise_power = phys_cfg["noise_power_w"]
    bandwidth_hz = comm_cfg["bandwidth_hz"]
    pilot_fraction = 0.1

    H = generate_physical_channel(
        ap_positions,
        user_positions,
        num_antennas=num_antennas,
        antenna_spacing=phys_cfg["antenna_spacing"],
        rician_k_factor=phys_cfg["rician_k_factor"],
        small_scale_fading=phys_cfg["small_scale_fading"],
        seed=phys_cfg["seed"],
    )

    # For a fair comparison with the physical-layer path (which spends
    # tx_power_per_ap per active AP), the scalar baseline uses the same
    # per-AP transmit power budget.
    scalar_result = run_scalar_baseline(
        ap_positions, user_positions, x,
        transmit_power=tx_power_per_ap,
        noise_power=noise_power,
    )
    mrt_result = run_physical_layer(
        H, x, "mrt", tx_power_per_ap, noise_power, bandwidth_hz, pilot_fraction
    )
    rzf_result = run_physical_layer(
        H, x, "rzf", tx_power_per_ap, noise_power, bandwidth_hz, pilot_fraction
    )

    summary = [
        summarize("scalar_baseline", scalar_result),
        summarize("mrt", mrt_result),
        summarize("rzf", rzf_result),
    ]

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "beamforming_validation.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "config": {
                    "seed": args.seed,
                    "num_aps": num_aps,
                    "num_users": num_users,
                    "aps_per_user": args.aps_per_user,
                    "num_antennas_per_ap": num_antennas,
                    "tx_power_per_ap_w": tx_power_per_ap,
                    "noise_power_w": noise_power,
                    "bandwidth_hz": bandwidth_hz,
                },
                "results": summary,
            },
            f,
            indent=2,
        )

    print(f"Phase 4A beamforming validation ({num_aps} APs, {num_users} users, "
          f"{num_antennas} antennas/AP, seed={args.seed})\n")
    header = (
        f"{'condition':<16}{'avg_rate_Mbps':>15}{'p5_rate_Mbps':>15}"
        f"{'avg_SINR_dB':>13}{'total_Ptx_W':>13}{'max_AP_Ptx_W':>14}{'runtime_s':>12}"
    )
    print(header)
    print("-" * len(header))
    for row in summary:
        print(
            f"{row['condition']:<16}"
            f"{row['avg_rate_mbps']:>15.4f}"
            f"{row['p5_rate_mbps']:>15.4f}"
            f"{row['avg_sinr_db']:>13.4f}"
            f"{row['total_tx_power_w']:>13.4f}"
            f"{row['max_per_ap_power_w']:>14.4f}"
            f"{row['runtime_s']:>12.6f}"
        )

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
