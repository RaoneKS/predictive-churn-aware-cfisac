"""
src/optimization/two_timescale.py

Phase 6 -- two-timescale optimization.

Slow-timescale association/clustering decisions (Phase 2/3 churn-aware
policy, reused unmodified) parameterize a fast-timescale per-AP power
allocation loop (Phase 5B, reused unmodified) that reacts to the
instantaneous physical channel every step.

This module is ADDITIVE. It reuses, read-only and unmodified:

    * Phase 2/3  src.optimization.baselines      (cluster_for_positions,
                                                    static_policy)
    * Phase 2/3  src.optimization.churn_aware     (evaluate_cluster_performance,
                                                    churn_aware_decision)
    * Phase 2B   src.simulation.end_to_end        (_horizon_objective)
    * Phase 4A   src.channels.physical            (generate_physical_channel)
    * Phase 4A/5B src.optimization.joint_comm_sensing
                                                   (comm_beam_directions,
                                                    optimize_joint_allocation)
    * Phase 1    src.channels.sensing             (generate_sensing_gains)
    * Phase 2    src.simulation.mobility_simulator (MobilitySimulator)
    * predictor  src.prediction.predictor_interface (predict(sim, horizon))

Nothing in Phases 1-5B is modified.

===========================================================================
1. Problem formulation
===========================================================================

Slow-timescale variables (held fixed for T_slow fast steps, one "epoch"):

    x_tau     in {0,1}^(M,K)   communication association    (Phase 2)
    y_tx_tau  in {0,1}^(M,Q)   sensing TX role               (Phase 3)
    y_rx_tau  in {0,1}^(M,Q)   sensing RX role               (Phase 3)

    tau = 0, 1, 2, ...  indexes slow epochs.  Epoch tau spans fast steps
    t = tau*T_slow, ..., (tau+1)*T_slow - 1.

Fast-timescale variables (re-optimized every fast step t):

    c^(t) in R_+^M   per-AP communication power   (Phase 5B P_comm)
    s^(t) in R_+^M   per-AP sensing power          (Phase 5B P_sens)

Fast problem (solved every step, Phase 5B verbatim, association fixed):

    maximize_{c,s>=0}  alpha*U_comm(c; x_tau, H^(t))
                        + beta*U_sens(s; y_tx_tau, y_rx_tau, G_sens^(t))
    subject to          c_m + s_m <= P_max,m   for every AP m

    where H^(t) is drawn fresh every fast step from the ACTUAL current
    positions (Phase 4A, with small-scale fading) -- this is what makes the
    fast layer react to "instantaneous CSI" in the two-timescale sense of
    the reference (vendor/reference_today/powercontrol, Miretti et al.,
    "Two-timescale joint power control and beamforming design").

Slow decision (evaluated once per epoch, churn_aware_decision verbatim):

    Candidate (x_cand, y_tx_cand, y_rx_cand) is built from predicted
    positions H_slow-steps ahead (predictor.predict(sim, T_slow), last
    step).  Both the previous and the candidate association are scored by
    the existing horizon-aggregated scalar objective
    (src.simulation.end_to_end._horizon_objective, reused unmodified) over
    ALL T_slow predicted steps, then compared via churn_aware_decision:

        net_gain = (candidate_horizon_objective - previous_horizon_objective)
                   - lambda_churn * norm_churn_cost(prev, cand)

        RECONFIGURE  if net_gain > 0   else KEEP

    This is intentionally the SCALAR model (not the physical Phase 5B
    objective): predicted positions exist, but predicted small-scale
    fading realizations do not, so the physical objective cannot honestly
    be evaluated at future predicted steps.  This is a real model boundary
    inherited from the existing project (see PROJECT_CONTEXT / Phase 2B),
    not something this module introduces -- it is documented here rather
    than hidden.

===========================================================================
2. Coupling between timescales
===========================================================================

    x_tau, y_tx_tau, y_rx_tau parameterize the fast Phase 5B problem:
      - x_tau determines which (AP, user) pairs get nonzero W_dir via
        comm_beam_directions (Phase 4A precoders under the hood).
      - y_tx_tau, y_rx_tau determine which APs are sensing-active and
        hence which per-AP power variables the Phase 5B solver has to
        decide (see _JointProblem.var_idx / held_idx, unmodified).
    The fast layer never feeds back into the slow decision within an
    epoch; slow decisions only observe PREDICTED positions, never fast
    per-step power outcomes.  This one-directional coupling matches the
    reference's separation of "functions of instantaneous CSI" (fast)
    from "long-term statistical" decisions (slow).

===========================================================================
3. Assumptions / model mismatches (explicit, not hidden)
===========================================================================

    (A1) Slow decisions use the scalar model; fast decisions use the
         physical model. Two different objective spaces, inherited from
         the existing project (Phase 2B docstring already establishes
         this convention for the per-step PREDICTIVE+CHURN policy).
    (A2) Fixed K (users) and Q (targets) for the duration of a run -- no
         phase in this project supports dynamic array resizing; dynamic
         user/target arrival-departure is out of scope here too.
    (A3) T_slow defaults to the caller-provided predictor horizon for
         backward-compatible convenience ONLY -- it is a fully independent
         explicit parameter, never hard-wired to the predictor's horizon.
    (A4) No claim of joint (slow, fast) global optimality: each layer is
         optimized given the other held fixed, exactly the same caveat
         Phase 5B already carries (`global_optimality_claimed: False`).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from src.channels.physical import generate_physical_channel
from src.channels.sensing import generate_sensing_gains
from src.optimization.baselines import cluster_for_positions
from src.optimization.churn_aware import churn_aware_decision
from src.optimization.joint_comm_sensing import (
    comm_beam_directions,
    optimize_joint_allocation,
)
from src.simulation.end_to_end import _horizon_objective


# ---------------------------------------------------------------------------
# Slow -> fast boundary validation (explicit, per Phase 6 requirement)
# ---------------------------------------------------------------------------

def validate_slow_to_fast(x, y_tx, y_rx, num_aps, num_users, num_targets):
    """
    Validate that a slow-timescale decision (x, y_tx, y_rx) is a legal
    input to the fast-timescale Phase 5B problem for this scenario size.

    Raises ValueError with a specific message on any mismatch. Returns
    (x, y_tx, y_rx) as float ndarrays with the validated shapes.
    """
    x = np.asarray(x, dtype=float)
    y_tx = np.asarray(y_tx, dtype=float)
    y_rx = np.asarray(y_rx, dtype=float)

    if x.shape != (num_aps, num_users):
        raise ValueError(
            f"x_tau must have shape (num_aps={num_aps}, num_users="
            f"{num_users}); got {x.shape}."
        )
    if y_tx.shape != (num_aps, num_targets):
        raise ValueError(
            f"y_tx_tau must have shape (num_aps={num_aps}, num_targets="
            f"{num_targets}); got {y_tx.shape}."
        )
    if y_rx.shape != (num_aps, num_targets):
        raise ValueError(
            f"y_rx_tau must have shape (num_aps={num_aps}, num_targets="
            f"{num_targets}); got {y_rx.shape}."
        )

    for name, arr in (("x_tau", x), ("y_tx_tau", y_tx), ("y_rx_tau", y_rx)):
        if not np.all(np.isin(arr, (0.0, 1.0))):
            raise ValueError(f"{name} must be binary (entries in {{0,1}}).")

    return x, y_tx, y_rx


# ---------------------------------------------------------------------------
# Slow-timescale update
# ---------------------------------------------------------------------------

def slow_update(
    sim,
    predictor,
    ap_positions,
    prev_x,
    prev_y_tx,
    prev_y_rx,
    T_slow,
    cluster_kwargs=None,
    perf_kwargs=None,
    churn_kwargs=None,
    ref=None,
    horizon_aggregation="sum",
):
    """
    One slow-timescale epoch decision.

    Reuses, unmodified:
      * predictor.predict(sim, T_slow)                (prediction interface)
      * cluster_for_positions                          (Phase 2/3 baselines)
      * _horizon_objective                              (Phase 2B end_to_end)
      * churn_aware_decision                             (Phase 3 churn_aware)

    Returns
    -------
    dict with keys:
        x, y_tx, y_rx           -- the (possibly unchanged) new slow decision
        decision                -- "KEEP" or "RECONFIGURE"
        candidate_x/y_tx/y_rx   -- the candidate that was scored
        pred_users, pred_targets -- (N, T_slow, 2) predictions used
        decision_info           -- raw churn_aware_decision() output
    """
    cluster_kwargs = dict(cluster_kwargs or {})
    perf_kwargs = dict(perf_kwargs or {})
    churn_kwargs = dict(churn_kwargs or {})

    pred_users_H, pred_targets_H = predictor.predict(sim, T_slow)
    pred_users_H = np.asarray(pred_users_H, dtype=float)
    pred_targets_H = np.asarray(pred_targets_H, dtype=float)

    cand_x, cand_y_tx, cand_y_rx = cluster_for_positions(
        ap_positions,
        pred_users_H[:, -1, :],
        pred_targets_H[:, -1, :],
        **cluster_kwargs,
    )

    prev_obj = _horizon_objective(
        prev_x, prev_y_tx, prev_y_rx,
        ap_positions, pred_users_H, pred_targets_H,
        perf_kwargs, aggregation=horizon_aggregation,
    )
    cand_obj = _horizon_objective(
        cand_x, cand_y_tx, cand_y_rx,
        ap_positions, pred_users_H, pred_targets_H,
        perf_kwargs, aggregation=horizon_aggregation,
    )

    # Small adapter: churn_aware_decision expects dict-shaped performance
    # records with a "combined_objective" key (see estimate_reconfiguration_
    # gain). _horizon_objective returns a scalar, so we wrap it -- no part
    # of the churn objective itself is reimplemented.
    current_performance = {"combined_objective": prev_obj}
    predicted_performance = {"combined_objective": cand_obj}

    decision_info = churn_aware_decision(
        prev_x, prev_y_tx, prev_y_rx,
        cand_x, cand_y_tx, cand_y_rx,
        current_performance,
        predicted_performance,
        ref=ref,
        **churn_kwargs,
    )

    if decision_info["decision"] == "RECONFIGURE":
        new_x, new_y_tx, new_y_rx = cand_x, cand_y_tx, cand_y_rx
    else:
        new_x, new_y_tx, new_y_rx = prev_x, prev_y_tx, prev_y_rx

    return {
        "x": new_x,
        "y_tx": new_y_tx,
        "y_rx": new_y_rx,
        "decision": decision_info["decision"],
        "candidate_x": cand_x,
        "candidate_y_tx": cand_y_tx,
        "candidate_y_rx": cand_y_rx,
        "pred_users": pred_users_H,
        "pred_targets": pred_targets_H,
        "decision_info": decision_info,
    }


# ---------------------------------------------------------------------------
# Fast-timescale update
# ---------------------------------------------------------------------------

def fast_update(
    ap_positions,
    user_positions,
    target_positions,
    x_tau,
    y_tx_tau,
    y_rx_tau,
    num_antennas,
    P_max,
    precoder="mrt",
    alpha=1.0,
    beta=1.0,
    fast_seed=0,
    phys_kwargs=None,
    sens_kwargs=None,
    joint_kwargs=None,
):
    """
    One fast-timescale step: draw the instantaneous physical channel from
    ACTUAL current positions, hold (x_tau, y_tx_tau, y_rx_tau) fixed, and
    solve the Phase 5B joint per-AP power allocation problem.

    Reuses, unmodified:
      * generate_physical_channel        (Phase 4A)
      * generate_sensing_gains           (Phase 1)
      * comm_beam_directions             (Phase 5B wrapper of 4A precoders)
      * optimize_joint_allocation        (Phase 5B)

    Returns the Phase 5B `optimize_joint_allocation` output dict, augmented
    with a few bookkeeping fields (fast_seed, num_aps/users/targets).
    """
    phys_kwargs = dict(phys_kwargs or {})
    sens_kwargs = dict(sens_kwargs or {})
    joint_kwargs = dict(joint_kwargs or {})

    ap_positions = np.asarray(ap_positions, dtype=float)
    user_positions = np.asarray(user_positions, dtype=float)
    target_positions = np.asarray(target_positions, dtype=float)

    num_aps = ap_positions.shape[0]
    num_users = user_positions.shape[0]
    num_targets = target_positions.shape[0]

    x_tau, y_tx_tau, y_rx_tau = validate_slow_to_fast(
        x_tau, y_tx_tau, y_rx_tau, num_aps, num_users, num_targets,
    )

    H = generate_physical_channel(
        ap_positions,
        user_positions,
        num_antennas,
        seed=fast_seed,
        **phys_kwargs,
    )

    W_dir = comm_beam_directions(H, x_tau, precoder=precoder)

    G_sens = generate_sensing_gains(
        ap_positions,
        target_positions,
        **sens_kwargs,
    )

    result = optimize_joint_allocation(
        H,
        W_dir,
        ap_positions,
        target_positions,
        y_tx_tau,
        y_rx_tau,
        G_sens,
        P_max,
        alpha=alpha,
        beta=beta,
        x=x_tau,
        **joint_kwargs,
    )

    result["fast_seed"] = int(fast_seed)
    result["num_aps"] = int(num_aps)
    result["num_users"] = int(num_users)
    result["num_targets"] = int(num_targets)

    return result


# ---------------------------------------------------------------------------
# Full two-timescale driver
# ---------------------------------------------------------------------------

@dataclass
class TwoTimescaleResult:
    fast: Dict[str, list] = field(default_factory=dict)
    slow: Dict[str, list] = field(default_factory=dict)
    num_fast_updates: int = 0
    num_slow_updates: int = 0
    T_slow: int = 0


def run_two_timescale(
    sim,
    predictor,
    ap_positions,
    num_antennas,
    T_total,
    T_slow,
    P_max=1.0,
    precoder="mrt",
    alpha=1.0,
    beta=1.0,
    cluster_kwargs=None,
    perf_kwargs=None,
    churn_kwargs=None,
    ref=None,
    horizon_aggregation="sum",
    phys_kwargs=None,
    sens_kwargs=None,
    joint_kwargs=None,
    base_seed=0,
    advance_sim=True,
    p1_mode=False,
    p1_kwargs=None,
):
    """
    Run the full two-timescale loop for T_total fast steps.

    T_slow is an explicit, independent parameter (Phase 6 requirement):
    it is NOT derived from the predictor's own horizon. Callers that want
    backward-compatible behaviour can pass T_slow=config["simulation"]
    ["prediction_horizon"], but nothing in this function assumes that.

    Parameters
    ----------
    sim           : MobilitySimulator (advanced in place if advance_sim)
    predictor     : object with .predict(sim, horizon) -> (users, targets)
    ap_positions  : (M, 2)
    num_antennas  : N, per-AP antenna count (Phase 4A)
    T_total       : total number of fast steps to run
    T_slow        : slow epoch length in fast steps (>= 1)
    base_seed     : seed root for the per-fast-step physical channel draw

    Returns
    -------
    TwoTimescaleResult
    """
    if T_slow < 1:
        raise ValueError("T_slow must be >= 1.")
    if T_total < 1:
        raise ValueError("T_total must be >= 1.")

    ap_positions = np.asarray(ap_positions, dtype=float)
    num_users = len(sim.user_positions)
    num_targets = len(sim.target_positions)

    # Bootstrap: slow epoch 0 uses clustering on the ACTUAL initial
    # positions (same convention as the existing STATIC baseline), since
    # there is no previous association to compare a candidate against yet.
    x, y_tx, y_rx = cluster_for_positions(
        ap_positions,
        sim.user_positions,
        sim.target_positions,
        **(cluster_kwargs or {}),
    )

    fast_log = {
        "tau": [], "t": [],
        "P_comm": [], "P_sens": [], "P_tx": [], "P_max": [],
        "feasible": [], "max_power_violation": [],
        "comm_utility_raw": [], "sensing_utility_raw": [],
        "objective": [],
    }
    slow_log = {
        "tau": [], "t": [], "decision": [], "net_gain": [],
        "raw_churn": [], "predicted_gain": [],
    }

    tau = -1
    for t in range(T_total):
        if t % T_slow == 0:
            tau += 1
            upd = slow_update(
                sim,
                predictor,
                ap_positions,
                x, y_tx, y_rx,
                T_slow,
                cluster_kwargs=cluster_kwargs,
                perf_kwargs=perf_kwargs,
                churn_kwargs=churn_kwargs,
                ref=ref,
                horizon_aggregation=horizon_aggregation,
            )
            x, y_tx, y_rx = upd["x"], upd["y_tx"], upd["y_rx"]

            slow_log["tau"].append(tau)
            slow_log["t"].append(t)
            slow_log["decision"].append(upd["decision"])
            slow_log["net_gain"].append(upd["decision_info"]["net_gain"])
            slow_log["raw_churn"].append(upd["decision_info"]["raw_churn"])
            slow_log["predicted_gain"].append(
                upd["decision_info"]["predicted_gain"]
            )

        fast_seed = base_seed * 1_000_003 + t

        res = fast_update(
            ap_positions,
            sim.user_positions,
            sim.target_positions,
            x, y_tx, y_rx,
            num_antennas,
            P_max,
            precoder=precoder,
            alpha=alpha,
            beta=beta,
            fast_seed=fast_seed,
            phys_kwargs=phys_kwargs,
            sens_kwargs=sens_kwargs,
            joint_kwargs=joint_kwargs,
        )

        fast_log["tau"].append(tau)
        fast_log["t"].append(t)
        fast_log["P_comm"].append(res["P_comm"])
        fast_log["P_sens"].append(res["P_sens"])
        fast_log["P_tx"].append(res["P_tx"])
        fast_log["P_max"].append(res["P_max"])
        fast_log["feasible"].append(res["feasible"])
        fast_log["max_power_violation"].append(res["max_power_violation"])
        fast_log["comm_utility_raw"].append(res["comm_utility_raw"])
        fast_log["sensing_utility_raw"].append(res["sensing_utility_raw"])
        fast_log["objective"].append(res["objective"])

        if advance_sim and t < T_total - 1:
            sim.step()

    return TwoTimescaleResult(
        fast=fast_log,
        slow=slow_log,
        num_fast_updates=T_total,
        num_slow_updates=tau + 1,
        T_slow=T_slow,
    )


__all__ = [
    "validate_slow_to_fast",
    "slow_update",
    "fast_update",
    "run_two_timescale",
    "TwoTimescaleResult",
]
