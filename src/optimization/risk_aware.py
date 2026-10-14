"""
src/optimization/risk_aware.py

Phase 7 -- risk-aware slow-timescale integration.

This module is ADDITIVE. It reuses, read-only and unmodified:

    * Phase 6   src.optimization.two_timescale   (slow_update, fast_update,
                                                    validate_slow_to_fast)
    * Phase 2/3 src.optimization.baselines        (cluster_for_positions)
    * Phase 2/3 src.optimization.churn_aware      (churn_aware_decision)
    * Phase 2B  src.simulation.end_to_end         (_horizon_objective)
    * Phase 7   src.optimization.uncertainty      (measured_cv_residuals,
                                                     generate_scenarios)
    * Phase 7   src.optimization.cvar             (cvar_risk_adjusted_objective)

Nothing in Phases 1-6 (including the Phase 7 CVaR core and uncertainty
modules themselves) is modified.

===========================================================================
1. What this module adds
===========================================================================

Phase 6's slow_update() scores exactly one predicted trajectory (the
deterministic CV/LSTM point forecast) and compares candidate vs. previous
association on that single number. This module replaces that single
number with a risk measure over a SET of scenario trajectories, built by
perturbing the deterministic forecast with resampled historical CV
residuals (src.optimization.uncertainty), while leaving every other part
of the Phase 6 decision path untouched:

    * the candidate association (x_cand, y_tx_cand, y_rx_cand) is still
      built once, from the deterministic (unperturbed) last predicted
      step -- scenarios are used only to risk-adjust how that candidate
      is SCORED, never to change what candidate gets generated;
    * the churn cost (total_churn_cost, via churn_aware_decision) is
      reused completely unmodified -- it depends only on the (binary)
      association matrices, which never carry uncertainty;
    * fast_update / run_two_timescale (Phase 6) are reused unmodified and
      remain fully compatible with the association this module returns.

===========================================================================
2. The sign-domain fix (why this module exists in its current form)
===========================================================================

churn_aware.estimate_reconfiguration_gain defines a GAIN:

    gain = predicted_objective - current_objective        (bigger = better)

cvar.py (Phase 7 CVaR core) is defined over LOSSES (bigger = worse) and is
explicit that its ONLY sign transformation is l_i := -g_i (module
docstring, cvar.py Section 1).

The bug this module must NOT contain: computing per-scenario
reconfiguration gains g_i = cand_obj_i - prev_obj_i and then calling
empirical_var / empirical_cvar DIRECTLY on the g_i array. That silently
treats gains as if they were losses. Concretely, with alpha=0.9:

    * On losses, VaR/CVaR at alpha=0.9 correctly select the worst
      (highest-loss) 10% tail.
    * Called directly on gains (no sign flip), the "worst" tail picked
      out by the same order statistic is the HIGHEST-gain 10% -- i.e.
      the best-case scenarios, not the worst-case ones. The resulting
      "risk-adjusted" gain would be an optimistic best-case average
      dressed up as a conservative risk measure, and would even reverse
      the qualitative alpha ordering (raising alpha, meant to widen the
      protected tail, would instead narrow it towards the optimistic
      extreme).

The fix: every scenario-gain array in this module is risk-adjusted via
cvar.cvar_risk_adjusted_objective(gains, alpha) -- NEVER via a direct
empirical_var/empirical_cvar call on gains. That function performs the
l_i = -g_i flip internally (cvar.py Section 1), evaluates CVaR in the
correct loss domain (Section 3), and flips back
(G_CVaR_alpha = -CVaR_alpha({l_i})) so the value this module reports as
"risk_adjusted_gain" is genuinely the (RU-weighted) mean gain over the
worst (1-alpha) fraction of scenarios -- pessimistic relative to the
plain scenario mean, as a risk-averse measure must be. See
CVaRDecisionInfo below: `var` and `cvar` are always reported in the loss
domain exactly as cvar.py returns them; `risk_adjusted_gain` is always in
the gain domain (module Section 3 of cvar.py, restated for gains).

===========================================================================
3. Scenario construction
===========================================================================

For a slow epoch scoring horizon T_slow and requested `num_scenarios`:

    1. pred_users_H, pred_targets_H = predictor.predict(sim, T_slow)
       -- the same deterministic forecast Phase 6 uses.
    2. Candidate association is clustered from the deterministic last
       predicted step (identical to two_timescale.slow_update).
    3. Measured CV residuals are computed independently for users and
       targets from sim.position_history (Phase 2, read-only) via
       uncertainty.measured_cv_residuals(..., horizon=T_slow).
    4. num_scenarios draws are resampled from that residual pool via
       uncertainty.generate_scenarios (users and targets use seed and
       seed+1 respectively, so the two populations are not coupled
       through a shared resample index).
    5. Each scenario's predicted trajectory is
       pred_H + resampled_residual_draw (elementwise); the deterministic
       forecast is scenario 0's "center", never itself discarded.
    6. cand_obj_i, prev_obj_i are each evaluated via
       src.simulation.end_to_end._horizon_objective on the SAME
       (cand_x, cand_y_tx, cand_y_rx) / (prev_x, prev_y_tx, prev_y_rx)
       used throughout, at the perturbed trajectory for scenario i.
    7. gain_i := cand_obj_i - prev_obj_i (Section 2's g_i).
    8. risk_adjusted_gain, cvar_result = cvar_risk_adjusted_objective(
           gains, alpha)
    9. The final decision reuses churn_aware_decision UNCHANGED, fed a
       synthetic (current=0, predicted=risk_adjusted_gain) performance
       pair so that estimate_reconfiguration_gain recovers exactly
       risk_adjusted_gain as "gain", and the existing churn-cost formula
       is applied to it exactly as in Phase 6.

===========================================================================
4. Edge cases (explicit, not hidden)
===========================================================================

(E1) num_scenarios == 0 -- EXACT PHASE 6 PASSTHROUGH.
     No scenario/CVaR machinery runs at all; this module delegates
     directly to two_timescale.slow_update(...) and returns its result
     bit-for-bit (plus risk-metadata fields set to None/0 so the return
     shape is uniform). This is a deliberate hard branch, not an
     approximation: a caller who wants Phase-6-identical behaviour gets
     literally the Phase 6 code path.

(E2) Empty / insufficient residual history (e.g. slow epoch 0, before
     sim.step() has ever been called, or fewer than T_slow+1 recorded
     steps) with num_scenarios > 0 -- DETERMINISTIC REDUCTION.
     uncertainty.measured_cv_residuals legitimately returns a
     zero-anchor pool in this case. Rather than routing that through
     uncertainty.generate_scenarios (whose own zero-residual-pool branch
     infers N from the residual array itself, which is only correct when
     at least one anchor exists -- see uncertainty.py), this module
     builds the "no perturbation available" scenario batch directly from
     the ACTUAL number of users/targets (taken from pred_users_H /
     pred_targets_H, never from the possibly-empty residual pool), i.e.
     num_scenarios copies of an all-zero perturbation. Every scenario is
     then identical to the deterministic forecast, so gain_1 = ... =
     gain_N = the single deterministic gain, and by the CVaR module's own
     structural identical-scenario property (cvar.py Section 4 /
     tests.test_phase7_cvar.TestZeroUncertaintyReduction) VaR = CVaR =
     that one value for every alpha, and risk_adjusted_gain reduces
     EXACTLY to the deterministic Phase 6 gain -- without this module
     special-casing that equality itself; it falls out of the RU formula.

(E3) num_scenarios > 0 but the residual pool has fewer than 2 DISTINCT
     residual vectors -- resampling degenerates to at most that many
     realizations, but this is not treated as an error: generate_scenarios
     (unmodified) samples with replacement, which remains well-defined.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from src.optimization.baselines import cluster_for_positions
from src.optimization.churn_aware import churn_aware_decision
from src.optimization.cvar import CVaRResult, cvar_risk_adjusted_objective
from src.optimization.deficiency_cvar import evaluate_deficiency_constraints
from src.optimization.two_timescale import fast_update, slow_update  # noqa: F401  (fast_update re-exported for convenience)
from src.optimization.uncertainty import generate_scenarios, measured_cv_residuals
from src.simulation.end_to_end import _horizon_objective

__all__ = [
    "risk_aware_slow_update",
    "run_risk_aware_two_timescale",
    "RiskAwareTwoTimescaleResult",
]


# ---------------------------------------------------------------------------
# Internal: scenario perturbations, with the epoch-0 / empty-history fix
# ---------------------------------------------------------------------------

def _scenario_perturbations(position_history, T_slow, num_scenarios, seed,
                             actual_n, dt):
    """
    Build (num_scenarios, actual_n, T_slow, 2) perturbation draws.

    position_history : list of (N, 2) ndarrays (chronological), e.g. the
                        user- or target-half of sim.position_history.
    actual_n          : the REAL object count (from pred_H.shape[0]), used
                         whenever the residual pool cannot supply it (E2
                         above) -- never inferred from a possibly-empty
                         residual array.
    """
    if len(position_history) == 0:
        residuals = np.zeros((0, actual_n, T_slow, 2), dtype=float)
    else:
        residuals = measured_cv_residuals(position_history, horizon=T_slow, dt=dt)

    if residuals.shape[0] == 0:
        # E2: no measured anchors available -- zero-perturbation scenarios,
        # built from the actual object count, not from residuals.shape[1].
        return np.zeros((num_scenarios, actual_n, T_slow, 2), dtype=float)

    return generate_scenarios(residuals, num_scenarios, seed)


# ---------------------------------------------------------------------------
# Risk-adjusted slow-timescale update
# ---------------------------------------------------------------------------

def risk_aware_slow_update(
    sim,
    predictor,
    ap_positions,
    prev_x,
    prev_y_tx,
    prev_y_rx,
    T_slow,
    num_scenarios=0,
    alpha=0.9,
    scenario_seed=0,
    residual_dt=None,
    cluster_kwargs=None,
    perf_kwargs=None,
    churn_kwargs=None,
    ref=None,
    horizon_aggregation="sum",
    enforce_deficiency_cvar=False,
    max_comm_cvar=None,
    max_sensing_cvar=None,
    min_rate_bps=None,
    epsilon_trk=None,
):
    """
    One risk-aware slow-timescale epoch decision.

    Identical to two_timescale.slow_update in every respect (candidate
    generation, churn cost, KEEP/RECONFIGURE rule) EXCEPT the scalar
    "predicted_gain" fed into churn_aware_decision is a CVaR-risk-adjusted
    gain over `num_scenarios` resampled-residual scenarios instead of the
    single deterministic forecast (module docstring, Sections 1-3).

    Parameters
    ----------
    num_scenarios : int, >= 0.
        0 => exact Phase 6 passthrough (module Section 4, E1).
    alpha : float, 0 < alpha < 1.
        CVaR confidence level. Only validated/used when num_scenarios > 0
        (matches cvar.py's own alpha domain).
    scenario_seed : int.
        Root seed for residual resampling (users use scenario_seed,
        targets use scenario_seed + 1; see module Section 3).
    residual_dt : float or None.
        dt used for measured_cv_residuals; defaults to sim.dt.

    Returns
    -------
    dict with keys:
        x, y_tx, y_rx             -- new slow decision (post KEEP/RECONFIGURE)
        decision                  -- "KEEP" or "RECONFIGURE"
        candidate_x/y_tx/y_rx     -- the candidate that was scored
        pred_users, pred_targets  -- (N, T_slow, 2) deterministic predictions
        decision_info             -- churn_aware_decision(...) output,
                                      UNCHANGED in shape/keys from Phase 6
        num_scenarios             -- int, as requested
        alpha                     -- float or None (None iff num_scenarios==0)
        risk_adjusted_gain        -- float; GAIN domain (module Section 2)
        deterministic_gain        -- float; the Phase-6 single-forecast gain,
                                      for direct comparison
        var, cvar                 -- float or None; LOSS domain, straight
                                      from cvar.py (None iff num_scenarios==0)
        cvar_result                -- cvar.CVaRResult or None
        scenario_gains             -- ndarray, shape (num_scenarios,), or
                                       None iff num_scenarios == 0
    """
    cluster_kwargs = dict(cluster_kwargs or {})
    perf_kwargs = dict(perf_kwargs or {})
    churn_kwargs = dict(churn_kwargs or {})

    if not isinstance(num_scenarios, (int, np.integer)):
        raise TypeError("num_scenarios must be an integer.")
    if num_scenarios < 0:
        raise ValueError("num_scenarios must be >= 0.")

    # ---- E1: exact Phase 6 passthrough -----------------------------------
    if num_scenarios == 0:
        base = slow_update(
            sim, predictor, ap_positions,
            prev_x, prev_y_tx, prev_y_rx,
            T_slow,
            cluster_kwargs=cluster_kwargs,
            perf_kwargs=perf_kwargs,
            churn_kwargs=churn_kwargs,
            ref=ref,
            horizon_aggregation=horizon_aggregation,
        )
        base["num_scenarios"] = 0
        base["alpha"] = None
        base["risk_adjusted_gain"] = base["decision_info"]["predicted_gain"]
        base["deterministic_gain"] = base["decision_info"]["predicted_gain"]
        base["var"] = None
        base["cvar"] = None
        base["cvar_result"] = None
        base["scenario_gains"] = None
        return base

    dt = float(residual_dt) if residual_dt is not None else float(sim.dt)

    pred_users_H, pred_targets_H = predictor.predict(sim, T_slow)
    pred_users_H = np.asarray(pred_users_H, dtype=float)
    pred_targets_H = np.asarray(pred_targets_H, dtype=float)

    cand_x, cand_y_tx, cand_y_rx = cluster_for_positions(
        ap_positions,
        pred_users_H[:, -1, :],
        pred_targets_H[:, -1, :],
        **cluster_kwargs,
    )

    deterministic_prev_obj = _horizon_objective(
        prev_x, prev_y_tx, prev_y_rx,
        ap_positions, pred_users_H, pred_targets_H,
        perf_kwargs, aggregation=horizon_aggregation,
    )
    deterministic_cand_obj = _horizon_objective(
        cand_x, cand_y_tx, cand_y_rx,
        ap_positions, pred_users_H, pred_targets_H,
        perf_kwargs, aggregation=horizon_aggregation,
    )
    deterministic_gain = float(deterministic_cand_obj - deterministic_prev_obj)

    user_history = [h[0] for h in sim.position_history]
    target_history = [h[1] for h in sim.position_history]

    user_pert = _scenario_perturbations(
        user_history, T_slow, num_scenarios, scenario_seed,
        pred_users_H.shape[0], dt,
    )
    target_pert = _scenario_perturbations(
        target_history, T_slow, num_scenarios, scenario_seed + 1,
        pred_targets_H.shape[0], dt,
    )

    from src.optimization.churn_aware import evaluate_cluster_performance

    gains = np.empty(num_scenarios, dtype=float)

    # For CVaR deficiency evaluation
    # We want to collect the rates and traces for the candidate across the scenario horizon.
    # To keep things simple, we take the average rate and average trace over the horizon blocks
    # for each scenario, matching how 'mean' horizon aggregation works, or we can just take
    # the last step. The PDF says expected deficiency, but we are evaluating constraints.
    # We evaluate deficiency CVaR on the mean performance over the horizon block H for each scenario.
    scen_rates = []
    scen_traces = []

    for i in range(num_scenarios):
        pu = pred_users_H + user_pert[i]
        pt = pred_targets_H + target_pert[i]

        cand_obj_i = _horizon_objective(
            cand_x, cand_y_tx, cand_y_rx,
            ap_positions, pu, pt,
            perf_kwargs, aggregation=horizon_aggregation,
        )
        prev_obj_i = _horizon_objective(
            prev_x, prev_y_tx, prev_y_rx,
            ap_positions, pu, pt,
            perf_kwargs, aggregation=horizon_aggregation,
        )
        gains[i] = cand_obj_i - prev_obj_i

        if enforce_deficiency_cvar:
            H_blocks = pu.shape[1]
            r_acc = 0.0
            t_acc = 0.0
            for h in range(H_blocks):
                u_pos = pu[:, h, :]
                t_pos = pt[:, h, :]
                perf = evaluate_cluster_performance(
                    cand_x, cand_y_tx, cand_y_rx, ap_positions, u_pos, t_pos, **perf_kwargs
                )
                r_acc = r_acc + np.array(perf["raw_rates"])
                t_acc = t_acc + np.array(perf["raw_traces"])
            scen_rates.append(r_acc / H_blocks)
            scen_traces.append(t_acc / H_blocks)

    if enforce_deficiency_cvar:
        rates_scenarios = np.array(scen_rates)
        traces_scenarios = np.array(scen_traces)
        deficiency_results = evaluate_deficiency_constraints(
            rates_scenarios, traces_scenarios,
            min_rate_bps=min_rate_bps or 0.0,
            epsilon_trk=epsilon_trk or np.zeros(pred_targets_H.shape[0]),
            max_comm_cvar=max_comm_cvar,
            max_sensing_cvar=max_sensing_cvar
        )
        cvar_satisfied = deficiency_results["satisfied"]
    else:
        cvar_satisfied = True
        deficiency_results = None

    # ---- The sign-domain-safe call (module Section 2) --------------------
    risk_adjusted_gain, cvar_result = cvar_risk_adjusted_objective(gains, alpha)

    # Reuse churn_aware_decision COMPLETELY UNCHANGED
    current_performance = {"combined_objective": 0.0}
    predicted_performance = {"combined_objective": float(risk_adjusted_gain)}

    decision_info = churn_aware_decision(
        prev_x, prev_y_tx, prev_y_rx,
        cand_x, cand_y_tx, cand_y_rx,
        current_performance,
        predicted_performance,
        ref=ref,
        **churn_kwargs,
    )
    # Restore human-readable objective bookkeeping (diagnostics only --
    # does not affect the decision, which was already made above).
    decision_info["current_objective"] = float(deterministic_prev_obj)
    decision_info["predicted_objective"] = float(
        deterministic_prev_obj + risk_adjusted_gain
    )

    if decision_info["decision"] == "RECONFIGURE" and cvar_satisfied:
        new_x, new_y_tx, new_y_rx = cand_x, cand_y_tx, cand_y_rx
    else:
        new_x, new_y_tx, new_y_rx = prev_x, prev_y_tx, prev_y_rx

    # If the candidate was rejected SOLELY because of the CVaR constraints, mark it.
    if decision_info["decision"] == "RECONFIGURE" and not cvar_satisfied:
        decision_info["decision"] = "KEEP (CVaR Constraint Violated)"
        decision_info["cvar_rejected"] = True
    else:
        decision_info["cvar_rejected"] = False

    decision_info["deficiency_cvar_results"] = deficiency_results

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
        "num_scenarios": int(num_scenarios),
        "alpha": float(alpha),
        "risk_adjusted_gain": float(risk_adjusted_gain),
        "deterministic_gain": deterministic_gain,
        "var": float(cvar_result.var),
        "cvar": float(cvar_result.cvar),
        "cvar_result": cvar_result,
        "scenario_gains": gains,
    }


# ---------------------------------------------------------------------------
# Full risk-aware two-timescale driver (mirrors two_timescale.run_two_timescale)
# ---------------------------------------------------------------------------

@dataclass
class RiskAwareTwoTimescaleResult:
    fast: Dict[str, list] = None
    slow: Dict[str, list] = None
    num_fast_updates: int = 0
    num_slow_updates: int = 0
    T_slow: int = 0

    def __post_init__(self):
        if self.fast is None:
            self.fast = {}
        if self.slow is None:
            self.slow = {}


def run_risk_aware_two_timescale(
    sim,
    predictor,
    ap_positions,
    num_antennas,
    T_total,
    T_slow,
    P_max=1.0,
    precoder="mrt",
    alpha_power=1.0,
    beta_power=1.0,
    num_scenarios=0,
    cvar_alpha=0.9,
    scenario_seed=0,
    residual_dt=None,
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
    enforce_deficiency_cvar=False,
    max_comm_cvar=None,
    max_sensing_cvar=None,
    min_rate_bps=None,
    epsilon_trk=None,
):
    """
    Run the full risk-aware two-timescale loop for T_total fast steps.

    Identical driver structure to two_timescale.run_two_timescale (fast
    layer completely unmodified -- Phase 6's fast_update is called
    verbatim every step), except the slow layer calls
    risk_aware_slow_update instead of slow_update.

    `alpha_power` / `beta_power` name the Phase 5B/6 comm/sensing power
    weights (kept distinct from `cvar_alpha`, the CVaR confidence level,
    to avoid any parameter-name collision between the two "alpha"s used
    in this project).
    """
    if T_slow < 1:
        raise ValueError("T_slow must be >= 1.")
    if T_total < 1:
        raise ValueError("T_total must be >= 1.")

    ap_positions = np.asarray(ap_positions, dtype=float)

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
        "risk_adjusted_gain": [], "deterministic_gain": [],
        "var": [], "cvar": [],
    }

    tau = -1
    for t in range(T_total):
        if t % T_slow == 0:
            tau += 1
            upd = risk_aware_slow_update(
                sim, predictor, ap_positions,
                x, y_tx, y_rx,
                T_slow,
                num_scenarios=num_scenarios,
                alpha=cvar_alpha,
                scenario_seed=scenario_seed,
                residual_dt=residual_dt,
                cluster_kwargs=cluster_kwargs,
                perf_kwargs=perf_kwargs,
                churn_kwargs=churn_kwargs,
                ref=ref,
                horizon_aggregation=horizon_aggregation,
                enforce_deficiency_cvar=enforce_deficiency_cvar,
                max_comm_cvar=max_comm_cvar,
                max_sensing_cvar=max_sensing_cvar,
                min_rate_bps=min_rate_bps,
                epsilon_trk=epsilon_trk,
            )
            x, y_tx, y_rx = upd["x"], upd["y_tx"], upd["y_rx"]

            slow_log["tau"].append(tau)
            slow_log["t"].append(t)
            slow_log["decision"].append(upd["decision"])
            slow_log["net_gain"].append(upd["decision_info"]["net_gain"])
            slow_log["raw_churn"].append(upd["decision_info"]["raw_churn"])
            slow_log["predicted_gain"].append(upd["decision_info"]["predicted_gain"])
            slow_log["risk_adjusted_gain"].append(upd["risk_adjusted_gain"])
            slow_log["deterministic_gain"].append(upd["deterministic_gain"])
            slow_log["var"].append(upd["var"])
            slow_log["cvar"].append(upd["cvar"])

        fast_seed = base_seed * 1_000_003 + t

        res = fast_update(
            ap_positions,
            sim.user_positions,
            sim.target_positions,
            x, y_tx, y_rx,
            num_antennas,
            P_max,
            precoder=precoder,
            alpha=alpha_power,
            beta=beta_power,
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

    return RiskAwareTwoTimescaleResult(
        fast=fast_log,
        slow=slow_log,
        num_fast_updates=T_total,
        num_slow_updates=tau + 1,
        T_slow=T_slow,
    )
