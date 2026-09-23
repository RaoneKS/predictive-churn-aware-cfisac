"""
src/simulation/end_to_end.py

Core simulation engine for the Predictive Churn-Aware CF-ISAC project.

Changes since Phase 1
---------------------
Phase 2:
  - `predictor` parameter: pass any object with .predict(sim, horizon) → (users, targets).
    Default is None, which falls back to the built-in CVPredictor (backward-compatible).
  - `mobility_config` parameter: dict passed to MobilitySimulator for stochastic mode.

Phase 2B (horizon-aware churn):
  - PREDICTIVE and PREDICTIVE+CHURN now evaluate the objective at EACH of the H
    predicted future positions and use the horizon-averaged objective for the
    churn-aware decision.  The clustering candidate is still derived from the
    LAST predicted step (h=H), preserving the original policy semantics.

All four policies remain clearly separated:
  STATIC         — one-time clustering at t=0, no updates
  REACTIVE       — re-cluster on current positions every step
  PREDICTIVE     — re-cluster on predicted positions (last horizon step)
  PREDICTIVE+CHURN — re-cluster on predicted positions, but only if the
                     horizon-averaged gain exceeds the churn cost
"""

import os
import time

import numpy as np
import pandas as pd
import yaml

from src.clustering.churn import total_churn_cost
from src.environment.topology import generate_ap_positions
from src.optimization.baselines import (
    cluster_difference,
    predictive_policy,
    reactive_policy,
    static_policy,
)
from src.optimization.churn_aware import (
    churn_aware_decision,
    evaluate_cluster_performance,
)
from src.simulation.mobility_simulator import MobilitySimulator


# ---------------------------------------------------------------------------
# Internal helper: default CV predictor (no import of the interface needed
# for backward compatibility when predictor=None)
# ---------------------------------------------------------------------------

class _DefaultCVPredictor:
    """Fallback used when predictor=None."""

    def predict(self, sim, horizon):
        return sim.predict_users(horizon), sim.predict_targets(horizon)


# ---------------------------------------------------------------------------
# Horizon-aware objective helper
# ---------------------------------------------------------------------------

def _horizon_objective(
    x, y_tx, y_rx,
    ap_positions,
    pred_users_H,    # (N_u, H, 2)
    pred_targets_H,  # (N_t, H, 2)
    perf_kwargs,     # dict for evaluate_cluster_performance
    aggregation="sum",
):
    """
    Evaluate cluster performance at each of H predicted future positions and
    aggregate the combined objective over the horizon.

    aggregation
    -----------
    "sum"  : total predicted utility accrued over the horizon (formulation A)
    "mean" : per-block average predicted utility          (formulation B)

    Why this matters
    ----------------
    The candidate configuration is held for the whole horizon, so utility
    accrues at every one of the H blocks while the reconfiguration cost is
    paid exactly ONCE, at the moment of switching.  The decision rule compares

        aggregate_gain  vs  lambda_churn * normalized_churn

    Under "sum" both sides are per-decision-epoch totals and the comparison is
    dimensionally consistent.

    Under "mean" the utility side is divided by H but the churn side is not.
    Crucially this is NOT a harmless rescaling: dividing BOTH sides by H would
    leave every decision unchanged, whereas dividing only the utility side
    shrinks the gain relative to the cost by a factor of exactly H and
    therefore biases the policy toward KEEP.  "mean" is retained only so the
    two formulations can be compared under controlled conditions.
    """
    H = pred_users_H.shape[1]
    objectives = []
    for h in range(H):
        u_pos = pred_users_H[:, h, :]    # (N_u, 2) at step h+1
        t_pos = pred_targets_H[:, h, :]  # (N_t, 2) at step h+1
        perf  = evaluate_cluster_performance(
            x, y_tx, y_rx, ap_positions, u_pos, t_pos, **perf_kwargs
        )
        objectives.append(perf["combined_objective"])
    if aggregation == "sum":
        return float(np.sum(objectives))
    if aggregation == "mean":
        return float(np.mean(objectives))
    raise ValueError(f"Unknown aggregation {aggregation!r}; use 'sum' or 'mean'.")


def _horizon_averaged_objective(
    x, y_tx, y_rx, ap_positions, pred_users_H, pred_targets_H, perf_kwargs,
):
    """Backward-compatible alias for the Phase-2B mean formulation."""
    return _horizon_objective(
        x, y_tx, y_rx, ap_positions, pred_users_H, pred_targets_H,
        perf_kwargs, aggregation="mean",
    )


# ---------------------------------------------------------------------------
# Main simulation function
# ---------------------------------------------------------------------------

def run_simulation(
    policy_name,
    config_path="configs/default.yaml",
    seed=42,
    predictor=None,
    mobility_config=None,
    **kwargs,
):
    """
    Run one episode of the CF-ISAC simulation.

    Parameters
    ----------
    policy_name    : "STATIC" | "REACTIVE" | "PREDICTIVE" | "PREDICTIVE+CHURN"
    config_path    : path to YAML configuration
    seed           : integer random seed (controls topology + mobility)
    predictor      : object with .predict(sim, horizon) → (users_pred, targets_pred)
                     each shape (N_obj, H, 2).  If None, uses built-in CV predictor.
    mobility_config: dict or None — passed to MobilitySimulator; None = constant_velocity
    **kwargs       : override any config value by name

    Returns
    -------
    metrics : dict of np.ndarray per time step plus scalar summaries
    """
    t_start = time.perf_counter()

    # ── Load config ──────────────────────────────────────────────────────────
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    num_aps     = kwargs.get("num_aps",     config["network"]["num_aps"])
    num_users   = kwargs.get("num_users",   config["network"]["num_users"])
    num_targets = kwargs.get("num_targets", config["network"]["num_targets"])
    area_size   = kwargs.get("area_size",   config["network"]["area_size"])
    time_blocks = kwargs.get("time_blocks", config["simulation"]["time_blocks"])
    dt          = kwargs.get("dt",          config["simulation"]["dt"])
    horizon     = kwargs.get("horizon",     config["simulation"]["prediction_horizon"])

    # Churn costs
    c_comm    = float(config["churn"]["communication_cost"])
    c_sens_tx = float(config["churn"]["sensing_tx_cost"])
    c_sens_rx = float(config["churn"]["sensing_rx_cost"])

    # Objective weights
    alpha_comm = float(kwargs.get("alpha_communication", config["objective"]["alpha_communication"]))
    beta_sens  = float(kwargs.get("beta_sensing",        config["objective"]["beta_sensing"]))
    lambda_e   = float(kwargs.get("lambda_energy",       config["objective"]["lambda_energy"]))
    lambda_f   = float(kwargs.get("lambda_fronthaul",    config["objective"]["lambda_fronthaul"]))
    lambda_c   = float(kwargs.get("lambda_churn",        config["objective"]["lambda_churn"]))

    # Horizon aggregation for the PREDICTIVE+CHURN decision: "sum" (default,
    # dimensionally consistent with a once-per-epoch churn cost) or "mean"
    # (legacy Phase-2B behaviour).  See _horizon_objective.
    horizon_aggregation = str(kwargs.get(
        "horizon_aggregation",
        config["objective"].get("horizon_aggregation", "sum"),
    ))

    # Reference values for normalization
    ref = {k: float(kwargs.get(k, v)) for k, v in config["reference_values"].items()}

    # Shared kwargs for evaluate_cluster_performance
    perf_kwargs = dict(
        alpha_communication=alpha_comm,
        beta_sensing=beta_sens,
        lambda_energy=lambda_e,
        lambda_fronthaul=lambda_f,
        ref=ref,
    )

    # ── Predictor ─────────────────────────────────────────────────────────────
    if predictor is None:
        predictor = _DefaultCVPredictor()

    # ── Environment setup ────────────────────────────────────────────────────
    ap_positions = generate_ap_positions(num_aps, area_size, seed=seed)

    rng = np.random.default_rng(seed)
    initial_user_positions   = rng.uniform(0, area_size, size=(num_users,   2))
    initial_target_positions = rng.uniform(0, area_size, size=(num_targets, 2))

    sim = MobilitySimulator(
        initial_user_positions,
        initial_target_positions,
        area_size=area_size,
        dt=dt,
        seed=seed,
        mobility_config=mobility_config,
    )

    # ── Metrics storage ──────────────────────────────────────────────────────
    metrics = {
        "avg_user_rate":        [],
        "p5_user_rate":         [],
        "qos_violations":       [],
        "sensing_utility":      [],
        "tracking_violations":  [],
        "energy":               [],
        "fronthaul":            [],
        "churn":                [],
        "cumulative_churn":     [],
        "combined_objective":   [],
        "reconfigurations":     [],
        "keep_decisions":       0,
        "reconfigure_decisions": 0,
        # Raw / normalized / weighted diagnostics
        "raw_communication":        [],
        "raw_sensing":              [],
        "raw_energy":               [],
        "raw_fronthaul":            [],
        "raw_churn":                [],
        "normalized_communication": [],
        "normalized_sensing":       [],
        "normalized_energy":        [],
        "normalized_fronthaul":     [],
        "normalized_churn":         [],
        "weighted_communication":   [],
        "weighted_sensing":         [],
        "weighted_energy":          [],
        "weighted_fronthaul":       [],
        "weighted_churn":           [],
    }

    decision_trace = []

    x_prev    = None
    y_tx_prev = None
    y_rx_prev = None
    cumulative_churn_val = 0.0

    # ── Main loop ─────────────────────────────────────────────────────────────
    for t in range(time_blocks):
        user_pos, target_pos = sim.step()

        # ── Policy decision ──────────────────────────────────────────────────
        if policy_name == "STATIC":
            if t == 0:
                x_curr, y_tx_curr, y_rx_curr = static_policy(
                    ap_positions, user_pos, target_pos
                )
            else:
                x_curr, y_tx_curr, y_rx_curr = x_prev, y_tx_prev, y_rx_prev

        elif policy_name == "REACTIVE":
            x_curr, y_tx_curr, y_rx_curr = reactive_policy(
                ap_positions, user_pos, target_pos
            )

        elif policy_name == "PREDICTIVE":
            # Get full H-step predictions; cluster on the LAST predicted step
            pred_users_H, pred_targets_H = predictor.predict(sim, horizon)
            pred_user_final   = pred_users_H[:,   -1, :]
            pred_target_final = pred_targets_H[:, -1, :]
            x_curr, y_tx_curr, y_rx_curr = predictive_policy(
                ap_positions, pred_user_final, pred_target_final
            )

        elif policy_name == "PREDICTIVE+CHURN":
            pred_users_H, pred_targets_H = predictor.predict(sim, horizon)
            pred_user_final   = pred_users_H[:,   -1, :]
            pred_target_final = pred_targets_H[:, -1, :]

            if t == 0:
                x_curr, y_tx_curr, y_rx_curr = predictive_policy(
                    ap_positions, pred_user_final, pred_target_final
                )
            else:
                x_cand, y_tx_cand, y_rx_cand = predictive_policy(
                    ap_positions, pred_user_final, pred_target_final
                )

                # ── Phase 2B: horizon-averaged objective ──────────────────────
                curr_avg_obj = _horizon_objective(
                    x_prev, y_tx_prev, y_rx_prev,
                    ap_positions, pred_users_H, pred_targets_H,
                    perf_kwargs, aggregation=horizon_aggregation,
                )
                cand_avg_obj = _horizon_objective(
                    x_cand, y_tx_cand, y_rx_cand,
                    ap_positions, pred_users_H, pred_targets_H,
                    perf_kwargs, aggregation=horizon_aggregation,
                )

                # Horizon-averaged objectives are passed to the tested
                # decision function as performance proxies.  Only the
                # "combined_objective" key is consumed by
                # estimate_reconfiguration_gain().
                curr_perf_proxy = {"combined_objective": curr_avg_obj}
                cand_perf_proxy = {"combined_objective": cand_avg_obj}

                # Single source of truth for the KEEP/RECONFIGURE rule:
                # src.optimization.churn_aware.churn_aware_decision().
                # It computes the raw churn internally via total_churn_cost,
                # normalizes by ref["churn_events"], weights by lambda_churn,
                # and returns RECONFIGURE iff gain - lambda*norm_churn > 0.
                decision_info = churn_aware_decision(
                    x_prev, y_tx_prev, y_rx_prev,
                    x_cand, y_tx_cand, y_rx_cand,
                    curr_perf_proxy, cand_perf_proxy,
                    c_comm=c_comm,
                    c_sensing_tx=c_sens_tx,
                    c_sensing_rx=c_sens_rx,
                    ref=ref,
                    lambda_churn=lambda_c,
                )

                decision = decision_info["decision"]

                decision_trace.append({
                    "time":               t,
                    "current_objective":  decision_info["current_objective"],
                    "candidate_objective":decision_info["predicted_objective"],
                    "predicted_gain":     decision_info["predicted_gain"],
                    "raw_churn":          decision_info["raw_churn"],
                    "normalized_churn":   decision_info["raw_churn"] / float(ref["churn_events"]),
                    "net_gain":           decision_info["net_gain"],
                    "decision":           decision,
                })

                if decision == "RECONFIGURE":
                    x_curr, y_tx_curr, y_rx_curr = x_cand, y_tx_cand, y_rx_cand
                    metrics["reconfigure_decisions"] += 1
                else:
                    x_curr, y_tx_curr, y_rx_curr = x_prev, y_tx_prev, y_rx_prev
                    metrics["keep_decisions"] += 1

        else:
            raise ValueError(f"Unknown policy {policy_name!r}. "
                             "Choose STATIC, REACTIVE, PREDICTIVE, or PREDICTIVE+CHURN.")

        # ── Churn measurement ────────────────────────────────────────────────
        if t == 0:
            reconfigured = 0
            churn_val    = 0.0
        else:
            diff = cluster_difference(
                x_prev, x_curr, y_tx_prev, y_tx_curr, y_rx_prev, y_rx_curr
            )
            reconfigured = 1 if diff > 0 else 0
            churn_val    = total_churn_cost(
                x_prev, x_curr, y_tx_prev, y_tx_curr, y_rx_prev, y_rx_curr,
                c_comm=c_comm, c_sensing_tx=c_sens_tx, c_sensing_rx=c_sens_rx,
            )

        # ── True performance on actual positions ─────────────────────────────
        actual_perf = evaluate_cluster_performance(
            x_curr, y_tx_curr, y_rx_curr,
            ap_positions, user_pos, target_pos,
            **perf_kwargs,
        )

        cumulative_churn_val += churn_val

        # ── Record metrics ───────────────────────────────────────────────────
        metrics["avg_user_rate"].append(
            actual_perf["raw_communication_utility"] / num_users
        )
        metrics["p5_user_rate"].append(
            np.percentile(actual_perf["raw_rates"], 5)
        )
        metrics["qos_violations"].append(actual_perf["qos_violations"])
        metrics["sensing_utility"].append(actual_perf["raw_sensing_utility"])
        metrics["tracking_violations"].append(actual_perf["tracking_violations"])
        metrics["energy"].append(actual_perf["raw_energy_penalty"])
        metrics["fronthaul"].append(actual_perf["raw_fronthaul_penalty"])
        metrics["churn"].append(churn_val)
        metrics["cumulative_churn"].append(cumulative_churn_val)
        metrics["reconfigurations"].append(reconfigured)

        n_comm = actual_perf["raw_communication_utility"] / ref["communication_rate_bps"]
        n_sens = actual_perf["raw_sensing_utility"]       / ref["sensing_information_gain"]
        n_eng  = actual_perf["raw_energy_penalty"]        / ref["energy_watts"]
        n_fh   = actual_perf["raw_fronthaul_penalty"]     / ref["fronthaul_links"]
        n_ch   = churn_val                                / ref["churn_events"]

        metrics["raw_communication"].append(actual_perf["raw_communication_utility"])
        metrics["raw_sensing"].append(actual_perf["raw_sensing_utility"])
        metrics["raw_energy"].append(actual_perf["raw_energy_penalty"])
        metrics["raw_fronthaul"].append(actual_perf["raw_fronthaul_penalty"])
        metrics["raw_churn"].append(churn_val)

        metrics["normalized_communication"].append(n_comm)
        metrics["normalized_sensing"].append(n_sens)
        metrics["normalized_energy"].append(n_eng)
        metrics["normalized_fronthaul"].append(n_fh)
        metrics["normalized_churn"].append(n_ch)

        metrics["weighted_communication"].append(alpha_comm * n_comm)
        metrics["weighted_sensing"].append(beta_sens  * n_sens)
        metrics["weighted_energy"].append(lambda_e   * n_eng)
        metrics["weighted_fronthaul"].append(lambda_f   * n_fh)
        metrics["weighted_churn"].append(lambda_c   * n_ch)

        metrics["combined_objective"].append(
            actual_perf["combined_objective"] - lambda_c * n_ch
        )

        x_prev, y_tx_prev, y_rx_prev = x_curr, y_tx_curr, y_rx_curr

    # ── Finalise ──────────────────────────────────────────────────────────────
    for k in metrics:
        if isinstance(metrics[k], list):
            metrics[k] = np.array(metrics[k])

    metrics["runtime_s"] = time.perf_counter() - t_start

    # Write decision trace if PREDICTIVE+CHURN
    if policy_name == "PREDICTIVE+CHURN" and len(decision_trace) > 0:
        os.makedirs("results", exist_ok=True)
        pd.DataFrame(decision_trace).to_csv("results/decision_trace.csv", index=False)

    return metrics
