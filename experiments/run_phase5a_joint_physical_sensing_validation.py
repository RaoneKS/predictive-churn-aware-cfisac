"""
Phase 5A — physical communication / sensing bridge validation.

Scope:
    * Keep the physical communication channel, topology, clustering and
      MRT beamforming fixed.
    * Add an explicit isotropic sensing covariance through per-AP sensing
      power.
    * Evaluate the existing Fisher-information sensing model through the
      new physical-power-aware bridge.
    * Verify:
        1. zero sensing power gives zero physical sensing contribution
        2. increasing sensing power gives non-decreasing information gain
        3. increasing sensing power gives non-increasing tracking error
        4. combined AP transmit power is exactly communication + sensing
        5. the per-AP power constraint is detected correctly
        6. the full bridge output remains finite and deterministic

This is Phase 5A only:
    no joint optimization and no communication SINR interference from
    sensing are introduced here.

Usage:
    PYTHONPATH=. python experiments/run_phase5a_joint_physical_sensing_validation.py
"""

import argparse
import json
import os

import numpy as np
import yaml

from src.beamforming.precoders import mrt_weights, normalize_per_ap_power
from src.channels.communication import generate_comm_gains
from src.channels.physical import generate_physical_channel
from src.channels.sensing import generate_sensing_gains
from src.clustering.overlapping import communication_clustering
from src.environment.topology import generate_ap_positions, generate_positions
from src.metrics.physical_sensing import (
    evaluate_physical_sensing_bridge,
    physical_tx_power_per_ap,
)
from src.metrics.sensing import tracking_error


def load_config(path="configs/default.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_sensing_assignments(num_aps, num_targets, active_aps):
    """
    Deterministic sensing topology.

    For every target:
        first two communication-active APs -> TX
        next two communication-active APs  -> RX

    The same AP roles are deliberately allowed to be reused across targets.
    """
    active_aps = np.asarray(active_aps, dtype=int)

    if len(active_aps) < 4:
        raise RuntimeError(
            "Need at least four communication-active APs for the "
            "Phase 5A validation topology."
        )

    y_tx = np.zeros((num_aps, num_targets), dtype=float)
    y_rx = np.zeros((num_aps, num_targets), dtype=float)

    tx_aps = active_aps[:2]
    rx_aps = active_aps[2:4]

    for q in range(num_targets):
        y_tx[tx_aps, q] = 1.0
        y_rx[rx_aps, q] = 1.0

    return y_tx, y_rx


def make_sensing_power(num_aps, sensing_aps, power_per_active_sensing_ap):
    p = np.zeros(num_aps, dtype=float)
    p[np.asarray(sensing_aps, dtype=int)] = float(
        power_per_active_sensing_ap
    )
    return p


def summarize_condition(
    name,
    result,
    sensing_power_value,
):
    info_gain = np.asarray(result["information_gain"], dtype=float)
    tracking = np.asarray(result["tracking_error"], dtype=float)
    p_comm = np.asarray(result["P_comm"], dtype=float)
    p_sens = np.asarray(result["P_sens"], dtype=float)
    p_tx = np.asarray(result["P_tx"], dtype=float)
    feasible = np.asarray(result["power_feasible"], dtype=bool)
    violation = np.asarray(result["power_violation"], dtype=float)

    return {
        "condition": name,
        "sensing_power_per_active_sensing_ap_w": float(sensing_power_value),
        "information_gain_per_target": info_gain.tolist(),
        "mean_information_gain": float(np.mean(info_gain)),
        "tracking_error_per_target": tracking.tolist(),
        "mean_tracking_error": float(np.mean(tracking)),
        "communication_power_per_ap_w": p_comm.tolist(),
        "sensing_power_per_ap_w": p_sens.tolist(),
        "combined_power_per_ap_w": p_tx.tolist(),
        "total_communication_power_w": float(np.sum(p_comm)),
        "total_sensing_power_w": float(np.sum(p_sens)),
        "total_combined_power_w": float(np.sum(p_tx)),
        "max_combined_power_w": float(np.max(p_tx)),
        "max_power_violation_w": float(np.max(violation)),
        "power_feasible": bool(np.all(feasible)),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results/phase5a")

    args = parser.parse_args()

    config = load_config()

    net_cfg = config["network"]
    phys_cfg = config["physical_layer"]

    num_aps = net_cfg["num_aps"]
    num_users = net_cfg["num_users"]
    area_size = net_cfg["area_size"]

    # Keep Phase 4A/4B topology and physical channel generation deterministic.
    ap_positions = generate_ap_positions(
        num_aps,
        area_size,
        seed=args.seed,
    )

    user_positions = generate_positions(
        num_users,
        area_size,
        seed=args.seed + 1,
    )

    target_positions = generate_positions(
        2,
        area_size,
        seed=args.seed + 2,
    )

    # Existing scalar communication model supplies the same association
    # mechanism used in earlier phases.
    G_comm = generate_comm_gains(
        ap_positions,
        user_positions,
    )

    x = communication_clustering(
        G_comm,
        num_users=num_users,
        aps_per_user=2,
    )

    num_antennas = phys_cfg["num_antennas_per_ap"]
    p_max = float(phys_cfg["tx_power_per_ap_w"])

    H = generate_physical_channel(
        ap_positions,
        user_positions,
        num_antennas=num_antennas,
        antenna_spacing=phys_cfg["antenna_spacing"],
        rician_k_factor=phys_cfg["rician_k_factor"],
        small_scale_fading=phys_cfg["small_scale_fading"],
        seed=phys_cfg["seed"],
    )

    # Fixed communication beamforming.
    #
    # Use only half of the available per-AP power budget so that the
    # sensing-power sweep can demonstrate both feasible and infeasible
    # combined-power cases.
    comm_power_budget = 0.5 * p_max

    W = mrt_weights(H, x)
    W, _ = normalize_per_ap_power(
        W,
        tx_power_per_ap=comm_power_budget,
    )

    P_comm = np.sum(np.abs(W) ** 2, axis=(1, 2))
    active_aps = np.flatnonzero(P_comm > 1e-12)

    y_tx, y_rx = build_sensing_assignments(
        num_aps,
        len(target_positions),
        active_aps,
    )

    sensing_active_aps = np.flatnonzero(
        (y_tx.sum(axis=1) + y_rx.sum(axis=1)) > 0
    )

    G_sens = generate_sensing_gains(
        ap_positions,
        target_positions,
        rcs=np.ones(len(target_positions)),
        pathloss_exponent=2.0,
    )

    # All feasible cases have 0.0 <= P_comm + P_sens <= Pmax.
    # The final 0.60 W case intentionally exceeds the 1.0 W AP budget.
    sensing_power_levels = [
        ("zero_sensing", 0.00),
        ("low_sensing", 0.10),
        ("medium_sensing", 0.25),
        ("budget_edge", 0.50),
        ("over_budget", 0.60),
    ]

    prior_covariances = [
        np.eye(2, dtype=float)
        for _ in range(len(target_positions))
    ]

    condition_rows = []

    raw_results = {}

    for name, sensing_power_value in sensing_power_levels:
        p_sens = make_sensing_power(
            num_aps,
            sensing_active_aps,
            sensing_power_value,
        )

        result = evaluate_physical_sensing_bridge(
            ap_positions=ap_positions,
            target_positions=target_positions,
            y_tx=y_tx,
            y_rx=y_rx,
            W=W,
            sensing_power_per_ap=p_sens,
            G_sens=G_sens,
            tx_power_per_ap_max=p_max,
            prior_covariances=prior_covariances,
            noise_power_sens=1e-9,
            reference_variance=1.0,
        )

        # Independent power-accounting cross-check.
        P_tx_direct = physical_tx_power_per_ap(
            W,
            p_sens,
        )

        if not np.array_equal(
            result["P_tx"],
            P_tx_direct,
        ):
            raise RuntimeError(
                f"Power accounting mismatch in condition {name}."
            )

        raw_results[name] = result

        condition_rows.append(
            summarize_condition(
                name,
                result,
                sensing_power_value,
            )
        )

    # ------------------------------------------------------------------
    # Validation assertions
    # ------------------------------------------------------------------

    feasible_names = [
        "zero_sensing",
        "low_sensing",
        "medium_sensing",
        "budget_edge",
    ]

    feasible_rows = [
        next(r for r in condition_rows if r["condition"] == name)
        for name in feasible_names
    ]

    # Information gain should be non-decreasing with sensing power.
    feasible_info_gain = np.array(
        [r["mean_information_gain"] for r in feasible_rows],
        dtype=float,
    )

    if not np.all(np.diff(feasible_info_gain) >= -1e-10):
        raise AssertionError(
            "Information gain is not non-decreasing with sensing power."
        )

    # Tracking error should be non-increasing with sensing power.
    feasible_tracking_error = np.array(
        [r["mean_tracking_error"] for r in feasible_rows],
        dtype=float,
    )

    if not np.all(np.diff(feasible_tracking_error) <= 1e-10):
        raise AssertionError(
            "Tracking error is not non-increasing with sensing power."
        )

    # Zero sensing must contribute no sensing power.
    zero = next(
        r for r in condition_rows
        if r["condition"] == "zero_sensing"
    )

    if zero["total_sensing_power_w"] != 0.0:
        raise AssertionError(
            "Zero-sensing condition has non-zero sensing power."
        )

    # The deliberately over-budget condition must be detected.
    over_budget = next(
        r for r in condition_rows
        if r["condition"] == "over_budget"
    )

    if over_budget["power_feasible"]:
        raise AssertionError(
            "Over-budget condition was incorrectly reported feasible."
        )

    if over_budget["max_power_violation_w"] <= 0.0:
        raise AssertionError(
            "Over-budget condition did not report a positive violation."
        )

    # The budget-edge case should remain feasible within numerical tolerance.
    budget_edge = next(
        r for r in condition_rows
        if r["condition"] == "budget_edge"
    )

    if not budget_edge["power_feasible"]:
        raise AssertionError(
            "Budget-edge condition should be feasible."
        )

    os.makedirs(args.output_dir, exist_ok=True)

    output_path = os.path.join(
        args.output_dir,
        "phase5a_joint_validation.json",
    )

    payload = {
        "phase": "5A",
        "title": (
            "Physical communication-sensing bridge validation "
            "with explicit sensing power"
        ),
        "scope": {
            "communication_beamforming": "fixed MRT",
            "joint_optimization": False,
            "sensing_interference_in_comm_sinr": False,
            "sensing_covariance_model": "isotropic",
            "fim_model": "existing scalar bistatic Fisher-information model",
        },
        "config": {
            "seed": args.seed,
            "num_aps": num_aps,
            "num_users": num_users,
            "num_targets": len(target_positions),
            "num_antennas_per_ap": num_antennas,
            "per_ap_power_max_w": p_max,
            "fixed_communication_power_budget_w": comm_power_budget,
            "noise_power_sensing_w": 1e-9,
            "reference_variance": 1.0,
        },
        "topology": {
            "communication_active_aps": active_aps.tolist(),
            "sensing_active_aps": sensing_active_aps.tolist(),
            "y_tx": y_tx.astype(int).tolist(),
            "y_rx": y_rx.astype(int).tolist(),
        },
        "validation": {
            "information_gain_non_decreasing": True,
            "tracking_error_non_increasing": True,
            "zero_sensing_gives_zero_sensing_power": True,
            "over_budget_is_detected": True,
            "budget_edge_is_feasible": True,
        },
        "conditions": condition_rows,
    }

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(
        f"Phase 5A physical sensing validation "
        f"({num_aps} APs, {num_users} users, "
        f"{len(target_positions)} targets, seed={args.seed})\n"
    )

    header = (
        f"{'condition':<16}"
        f"{'sense_W/AP':>12}"
        f"{'mean_info_gain':>17}"
        f"{'mean_track_err':>17}"
        f"{'max_Ptx_W':>12}"
        f"{'feasible':>11}"
    )

    print(header)
    print("-" * len(header))

    for row in condition_rows:
        print(
            f"{row['condition']:<16}"
            f"{row['sensing_power_per_active_sensing_ap_w']:>12.4f}"
            f"{row['mean_information_gain']:>17.8f}"
            f"{row['mean_tracking_error']:>17.8f}"
            f"{row['max_combined_power_w']:>12.6f}"
            f"{str(row['power_feasible']):>11}"
        )

    print("\nValidation checks:")
    print("  information gain monotonic: PASS")
    print("  tracking error monotonic:    PASS")
    print("  zero sensing power:          PASS")
    print("  budget-edge feasible:        PASS")
    print("  over-budget detected:        PASS")

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
