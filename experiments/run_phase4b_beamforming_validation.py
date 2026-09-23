"""
experiments/run_phase4b_beamforming_validation.py

Phase 4B — deterministic validation experiment comparing, on the SAME
topology, AP-user association, and per-AP power budget:

    (1) MRT        (Phase 4A, per-AP local, src.beamforming.precoders)
    (2) RZF        (Phase 4A, per-AP local, src.beamforming.precoders)
    (3) SINR-constrained / QoS beamforming at a moderate uniform target
        (Phase 4B, global/joint, src.beamforming.constrained)
    (4) SINR-constrained bisection: the largest uniform SINR floor the
        network can guarantee every user under the same power budget
        (Phase 4B)

This is a static, single-snapshot physical-layer comparison -- it does
not touch prediction, clustering policy, or churn logic.

Usage:
    PYTHONPATH=. python experiments/run_phase4b_beamforming_validation.py
"""

import argparse
import json
import os
import time

import numpy as np
import yaml

from src.channels.physical import generate_physical_channel
from src.channels.communication import generate_comm_gains
from src.clustering.overlapping import communication_clustering
from src.environment.topology import generate_ap_positions, generate_positions
from src.beamforming.precoders import mrt_weights, rzf_weights, normalize_per_ap_power
from src.beamforming.constrained import (
    constrained_sinr_beamforming,
    bisection_sinr_beamforming,
)
from src.metrics.physical_layer import physical_sinr, physical_rate


def load_config(path="configs/default.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def summarize(name, H, W, noise_power, bandwidth_hz, pilot_fraction, runtime_s, extra=None):
    sinr = physical_sinr(H, W, noise_power=noise_power)
    rates = physical_rate(sinr, bandwidth_hz=bandwidth_hz, pilot_fraction=pilot_fraction)
    ap_power = np.sum(np.abs(W) ** 2, axis=(1, 2))
    row = {
        "condition": name,
        "avg_rate_mbps": float(np.mean(rates) / 1e6),
        "min_sinr_linear": float(np.min(sinr)),
        "min_sinr_db": float(10.0 * np.log10(np.min(sinr) + 1e-30)),
        "avg_sinr_db": float(10.0 * np.log10(np.mean(sinr) + 1e-30)),
        "total_tx_power_w": float(np.sum(ap_power)),
        "max_per_ap_power_w": float(np.max(ap_power)),
        "runtime_s": runtime_s,
    }
    if extra:
        row.update(extra)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-aps", type=int, default=None)
    parser.add_argument("--num-users", type=int, default=None)
    parser.add_argument("--aps-per-user", type=int, default=2)
    parser.add_argument("--target-sinr", type=float, default=1.0,
                         help="Uniform SINR target (linear) for the fixed-gamma run.")
    parser.add_argument("--bisection-tol", type=float, default=1e-2)
    parser.add_argument("--output-dir", type=str, default="results/phase4b")
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
        ap_positions, user_positions, num_antennas=num_antennas,
        antenna_spacing=phys_cfg["antenna_spacing"],
        rician_k_factor=phys_cfg["rician_k_factor"],
        small_scale_fading=phys_cfg["small_scale_fading"],
        seed=phys_cfg["seed"],
    )

    summary = []

    t0 = time.perf_counter()
    W_mrt = mrt_weights(H, x)
    W_mrt, _ = normalize_per_ap_power(W_mrt, tx_power_per_ap=tx_power_per_ap)
    summary.append(summarize("mrt", H, W_mrt, noise_power, bandwidth_hz, pilot_fraction,
                              time.perf_counter() - t0))

    t0 = time.perf_counter()
    W_rzf = rzf_weights(H, x)
    W_rzf, _ = normalize_per_ap_power(W_rzf, tx_power_per_ap=tx_power_per_ap)
    summary.append(summarize("rzf", H, W_rzf, noise_power, bandwidth_hz, pilot_fraction,
                              time.perf_counter() - t0))

    t0 = time.perf_counter()
    W_qos, feasible, sinr_qos, info_qos = constrained_sinr_beamforming(
        H, gamma=args.target_sinr, tx_power_per_ap=tx_power_per_ap, noise_power=noise_power,
    )
    summary.append(summarize(
        f"sinr_constrained(target={args.target_sinr})", H, W_qos, noise_power,
        bandwidth_hz, pilot_fraction, time.perf_counter() - t0,
        extra={"feasible": bool(feasible), "outer_iterations": info_qos["outer_iterations"],
               "solver_reason": info_qos["reason"]},
    ))

    t0 = time.perf_counter()
    W_bis, gamma_star, sinr_bis, history = bisection_sinr_beamforming(
        H, tx_power_per_ap=tx_power_per_ap, noise_power=noise_power, tol=args.bisection_tol,
    )
    summary.append(summarize(
        "sinr_constrained(bisection_max_min)", H, W_bis, noise_power,
        bandwidth_hz, pilot_fraction, time.perf_counter() - t0,
        extra={"gamma_star_linear": float(gamma_star),
               "gamma_star_db": float(10.0 * np.log10(max(gamma_star, 1e-30))),
               "bisection_steps": len(history)},
    ))

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "beamforming_validation_4b.json")
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
                    "target_sinr_linear": args.target_sinr,
                    "bisection_tol": args.bisection_tol,
                },
                "results": summary,
                "bisection_history": history,
            },
            f,
            indent=2,
        )

    print(f"Phase 4B beamforming validation ({num_aps} APs, {num_users} users, "
          f"{num_antennas} antennas/AP, seed={args.seed})\n")
    header = (
        f"{'condition':<38}{'avg_rate_Mbps':>14}{'min_SINR_dB':>13}"
        f"{'max_AP_Ptx_W':>14}{'runtime_s':>11}"
    )
    print(header)
    print("-" * len(header))
    for row in summary:
        print(
            f"{row['condition']:<38}"
            f"{row['avg_rate_mbps']:>14.4f}"
            f"{row['min_sinr_db']:>13.4f}"
            f"{row['max_per_ap_power_w']:>14.6f}"
            f"{row['runtime_s']:>11.6f}"
        )

    print(f"\nQoS-target run feasible: {summary[2]['feasible']} "
          f"(outer_iterations={summary[2]['outer_iterations']})")
    print(f"Bisection max-min SINR floor: {summary[3]['gamma_star_linear']:.4f} "
          f"linear ({summary[3]['gamma_star_db']:.2f} dB) in "
          f"{summary[3]['bisection_steps']} steps")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
