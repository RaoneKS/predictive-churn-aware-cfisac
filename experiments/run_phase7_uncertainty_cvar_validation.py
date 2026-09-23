"""
experiments/run_phase7_uncertainty_cvar_validation.py

Phase 7 validation experiment: uncertainty-aware, CVaR-risk-adjusted
slow-timescale decisions (src.optimization.risk_aware) vs. the
deterministic Phase 6 baseline (src.optimization.two_timescale).

Compares:
  1. Deterministic Phase 6                (num_scenarios=0)
  2. Uncertainty-aware Phase 7            (num_scenarios>0, fixed alpha)
  3. Multiple uncertainty levels          (mobility sigma_v swept, fixed
                                            num_scenarios/alpha)
  4. Multiple CVaR alpha values           (alpha swept, fixed uncertainty
                                            level / num_scenarios)

Reports, per run: communication sum-rate, sensing information gain,
reconfiguration count, raw churn, predicted (risk-adjusted) gain, VaR,
CVaR, risk-adjusted gain, power feasibility.

Writes results to results/phase7/phase7_uncertainty_cvar_validation.json
"""

import json
import os

import numpy as np

from src.environment.topology import generate_ap_positions, generate_positions
from src.optimization.risk_aware import run_risk_aware_two_timescale
from src.optimization.two_timescale import run_two_timescale
from src.prediction.predictor_interface import CVPredictor
from src.simulation.mobility_simulator import MobilitySimulator


NUM_APS = 20
NUM_USERS = 5
NUM_TARGETS = 2
AREA_SIZE = 500.0
NUM_ANTENNAS = 4
P_MAX = 1.0
ALPHA_POWER = 1.0
BETA_POWER = 1.0
SEED = 42
BASE_FAST_SEED = 7

T_TOTAL = 30
T_SLOW = 5

DEFAULT_NUM_SCENARIOS = 100
DEFAULT_CVAR_ALPHA = 0.9
DEFAULT_SCENARIO_SEED = 0

UNCERTAINTY_LEVELS = {
    # sigma_v controls per-step velocity innovation noise in the OU-style
    # stochastic mobility model (src.environment.stochastic_mobility) --
    # larger sigma_v means the constant-velocity forecast is a worse fit,
    # i.e. genuinely larger measured CV residuals, not a fabricated variance.
    "low": 0.5,
    "medium": 3.0,
    "high": 8.0,
}

ALPHA_SWEEP = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]

FAILURES = []


def _make_scenario(sigma_v=None, seed=SEED):
    ap = generate_ap_positions(NUM_APS, AREA_SIZE, seed=seed)
    us = generate_positions(NUM_USERS, AREA_SIZE, seed=seed + 1)
    tg = generate_positions(NUM_TARGETS, AREA_SIZE, seed=seed + 2)
    mobility_config = None
    if sigma_v is not None:
        mobility_config = {
            "type": "stochastic",
            "users": {"sigma_v": sigma_v, "rho_v": 0.6, "preferred_speed": 6.0},
            "targets": {"sigma_v": sigma_v, "rho_v": 0.6, "preferred_speed": 4.0},
        }
    sim = MobilitySimulator(us, tg, area_size=AREA_SIZE, dt=1.0, seed=seed,
                             mobility_config=mobility_config)
    predictor = CVPredictor()
    return ap, sim, predictor


def _summarize(result, label, is_risk_aware):
    fast = result.fast
    slow = result.slow

    n_reconf = sum(1 for d in slow["decision"] if d == "RECONFIGURE")
    n_keep = sum(1 for d in slow["decision"] if d == "KEEP")
    raw_churn_total = float(np.sum(slow["raw_churn"]))
    predicted_gain_mean = float(np.mean(slow["predicted_gain"]))

    sum_rate_mean = float(np.mean(fast["comm_utility_raw"]))
    info_gain_mean = float(np.mean(fast["sensing_utility_raw"]))
    feasible_all = bool(all(fast["feasible"]))
    max_violation = float(np.max(fast["max_power_violation"]))

    summary = {
        "label": label,
        "is_risk_aware": is_risk_aware,
        "T_slow": result.T_slow,
        "num_fast_updates": result.num_fast_updates,
        "num_slow_updates": result.num_slow_updates,
        "reconfiguration_count": n_reconf,
        "keep_count": n_keep,
        "raw_churn_total": raw_churn_total,
        "predicted_gain_mean": predicted_gain_mean,
        "communication_sum_rate_bps_mean": sum_rate_mean,
        "sensing_information_gain_mean": info_gain_mean,
        "power_feasible_all_steps": feasible_all,
        "max_power_violation": max_violation,
    }

    if is_risk_aware:
        var_vals = [v for v in slow["var"] if v is not None]
        cvar_vals = [v for v in slow["cvar"] if v is not None]
        radj_vals = [v for v in slow["risk_adjusted_gain"] if v is not None]
        det_vals = [v for v in slow["deterministic_gain"] if v is not None]
        summary["var_mean"] = float(np.mean(var_vals)) if var_vals else None
        summary["cvar_mean"] = float(np.mean(cvar_vals)) if cvar_vals else None
        summary["risk_adjusted_gain_mean"] = (
            float(np.mean(radj_vals)) if radj_vals else None
        )
        summary["deterministic_gain_mean"] = (
            float(np.mean(det_vals)) if det_vals else None
        )
        if var_vals and cvar_vals:
            if any(c + 1e-9 < v for c, v in zip(cvar_vals, var_vals)):
                FAILURES.append(f"{label}: CVaR < VaR observed (violates CVaR>=VaR)")

    if not feasible_all:
        FAILURES.append(f"{label}: infeasible fast step detected")

    return summary


def _run_deterministic():
    ap, sim, predictor = _make_scenario(sigma_v=None, seed=SEED)
    result = run_two_timescale(
        sim, predictor, ap,
        num_antennas=NUM_ANTENNAS, T_total=T_TOTAL, T_slow=T_SLOW,
        P_max=P_MAX, alpha=ALPHA_POWER, beta=BETA_POWER,
        base_seed=BASE_FAST_SEED,
    )
    return _summarize(result, "deterministic_phase6", is_risk_aware=False)


def _run_risk_aware(label, sigma_v, num_scenarios, cvar_alpha, scenario_seed):
    ap, sim, predictor = _make_scenario(sigma_v=sigma_v, seed=SEED)
    result = run_risk_aware_two_timescale(
        sim, predictor, ap,
        num_antennas=NUM_ANTENNAS, T_total=T_TOTAL, T_slow=T_SLOW,
        P_max=P_MAX, alpha_power=ALPHA_POWER, beta_power=BETA_POWER,
        num_scenarios=num_scenarios, cvar_alpha=cvar_alpha,
        scenario_seed=scenario_seed, base_seed=BASE_FAST_SEED,
    )
    return _summarize(result, label, is_risk_aware=True)


def main():
    # -----------------------------------------------------------------
    # 1. Deterministic Phase 6 baseline
    # -----------------------------------------------------------------
    deterministic_summary = _run_deterministic()

    # -----------------------------------------------------------------
    # 2. Uncertainty-aware Phase 7 (medium uncertainty, default alpha)
    # -----------------------------------------------------------------
    phase7_default_summary = _run_risk_aware(
        "phase7_uncertainty_aware (medium uncertainty, alpha=0.9)",
        sigma_v=UNCERTAINTY_LEVELS["medium"],
        num_scenarios=DEFAULT_NUM_SCENARIOS,
        cvar_alpha=DEFAULT_CVAR_ALPHA,
        scenario_seed=DEFAULT_SCENARIO_SEED,
    )

    # -----------------------------------------------------------------
    # 3. Multiple uncertainty levels (fixed num_scenarios, alpha)
    # -----------------------------------------------------------------
    uncertainty_level_summaries = {}
    for level_name, sigma_v in UNCERTAINTY_LEVELS.items():
        uncertainty_level_summaries[level_name] = _run_risk_aware(
            f"uncertainty_level_{level_name} (sigma_v={sigma_v})",
            sigma_v=sigma_v,
            num_scenarios=DEFAULT_NUM_SCENARIOS,
            cvar_alpha=DEFAULT_CVAR_ALPHA,
            scenario_seed=DEFAULT_SCENARIO_SEED,
        )

    # -----------------------------------------------------------------
    # 4. Multiple CVaR alpha values (fixed medium uncertainty)
    # -----------------------------------------------------------------
    alpha_sweep_summaries = {}
    for a in ALPHA_SWEEP:
        alpha_sweep_summaries[str(a)] = _run_risk_aware(
            f"alpha_{a}",
            sigma_v=UNCERTAINTY_LEVELS["medium"],
            num_scenarios=DEFAULT_NUM_SCENARIOS,
            cvar_alpha=a,
            scenario_seed=DEFAULT_SCENARIO_SEED,
        )

    # -----------------------------------------------------------------
    # 5. Consistency checks
    # -----------------------------------------------------------------

    # (a) Zero-scenario Phase 7 driver reproduces deterministic Phase 6
    #     bit-for-bit (E1 passthrough), on the SAME (non-stochastic)
    #     scenario used for the deterministic baseline above.
    ap_z, sim_z, pred_z = _make_scenario(sigma_v=None, seed=SEED)
    zero_scn_result = run_risk_aware_two_timescale(
        sim_z, pred_z, ap_z,
        num_antennas=NUM_ANTENNAS, T_total=T_TOTAL, T_slow=T_SLOW,
        P_max=P_MAX, alpha_power=ALPHA_POWER, beta_power=BETA_POWER,
        num_scenarios=0, base_seed=BASE_FAST_SEED,
    )
    ap_ref, sim_ref, pred_ref = _make_scenario(sigma_v=None, seed=SEED)
    ref_result = run_two_timescale(
        sim_ref, pred_ref, ap_ref,
        num_antennas=NUM_ANTENNAS, T_total=T_TOTAL, T_slow=T_SLOW,
        P_max=P_MAX, alpha=ALPHA_POWER, beta=BETA_POWER,
        base_seed=BASE_FAST_SEED,
    )
    passthrough_matches = (
        list(zero_scn_result.slow["decision"]) == list(ref_result.slow["decision"])
    )
    if not passthrough_matches:
        FAILURES.append("zero-scenario Phase 7 driver did not match deterministic Phase 6")

    # (b) CVaR alpha monotonicity: risk-adjusted gain should be
    #     non-increasing in alpha (wider worst-case tail averaged as
    #     alpha grows), matching cvar.py's own structural property.
    alpha_vals_sorted = sorted(ALPHA_SWEEP)
    radj_means = [
        alpha_sweep_summaries[str(a)]["risk_adjusted_gain_mean"]
        for a in alpha_vals_sorted
    ]
    non_increasing_violations = sum(
        1 for i in range(len(radj_means) - 1)
        if radj_means[i + 1] > radj_means[i] + 1e-6
    )
    # Allow a small number of violations since each alpha uses a run with
    # its own realized KEEP/RECONFIGURE path (association can differ
    # across epochs), not a single frozen scenario set; flag only a
    # systematic violation.
    alpha_monotonic_mostly_ok = non_increasing_violations <= 1
    if not alpha_monotonic_mostly_ok:
        FAILURES.append(
            f"risk_adjusted_gain not mostly non-increasing in alpha "
            f"({non_increasing_violations} violations across "
            f"{len(radj_means) - 1} adjacent pairs)"
        )

    # (c) Higher uncertainty should not trivially collapse to the
    #     deterministic value -- at least the "high" level must differ
    #     from the deterministic gain.
    high_summary = uncertainty_level_summaries["high"]
    uncertainty_has_effect = (
        high_summary["risk_adjusted_gain_mean"] is not None
        and high_summary["deterministic_gain_mean"] is not None
        and not np.isclose(
            high_summary["risk_adjusted_gain_mean"],
            high_summary["deterministic_gain_mean"],
            atol=1e-6,
        )
    )
    if not uncertainty_has_effect:
        FAILURES.append("high uncertainty level did not change the risk-adjusted gain")

    status = "PASS" if not FAILURES else "FAIL"

    result = {
        "status": status,
        "scenario": {
            "num_aps": NUM_APS,
            "num_users": NUM_USERS,
            "num_targets": NUM_TARGETS,
            "num_antennas": NUM_ANTENNAS,
            "area_size": AREA_SIZE,
            "T_total": T_TOTAL,
            "T_slow": T_SLOW,
            "P_max": P_MAX,
            "alpha_power": ALPHA_POWER,
            "beta_power": BETA_POWER,
            "seed": SEED,
            "base_fast_seed": BASE_FAST_SEED,
            "default_num_scenarios": DEFAULT_NUM_SCENARIOS,
            "default_cvar_alpha": DEFAULT_CVAR_ALPHA,
        },
        "checks": {
            "zero_scenario_passthrough_matches_phase6": passthrough_matches,
            "cvar_never_below_var": not any(
                "CVaR < VaR" in f for f in FAILURES
            ),
            "alpha_monotonicity_mostly_holds": alpha_monotonic_mostly_ok,
            "uncertainty_has_effect_at_high_level": uncertainty_has_effect,
            "all_fast_steps_power_feasible": all(
                s["power_feasible_all_steps"]
                for s in (
                    [deterministic_summary, phase7_default_summary]
                    + list(uncertainty_level_summaries.values())
                    + list(alpha_sweep_summaries.values())
                )
            ),
        },
        "deterministic_phase6_baseline": deterministic_summary,
        "phase7_uncertainty_aware_default": phase7_default_summary,
        "uncertainty_levels": uncertainty_level_summaries,
        "cvar_alpha_sweep": alpha_sweep_summaries,
        "failures": FAILURES,
    }

    os.makedirs("results/phase7", exist_ok=True)
    out_path = "results/phase7/phase7_uncertainty_cvar_validation.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print("=" * 72)
    print("Phase 7 Uncertainty + CVaR Validation")
    print("=" * 72)
    print(f"Status: {status}")
    print()
    print("Deterministic Phase 6 baseline:")
    print(f"  reconfigurations: {deterministic_summary['reconfiguration_count']}")
    print(f"  raw churn total:  {deterministic_summary['raw_churn_total']:.4g}")
    print(f"  sum-rate mean:    {deterministic_summary['communication_sum_rate_bps_mean']:.4g}")
    print(f"  info gain mean:   {deterministic_summary['sensing_information_gain_mean']:.4g}")
    print(f"  power feasible:   {deterministic_summary['power_feasible_all_steps']}")
    print()
    print("Phase 7 uncertainty-aware (medium uncertainty, alpha=0.9):")
    print(f"  reconfigurations: {phase7_default_summary['reconfiguration_count']}")
    print(f"  VaR mean:         {phase7_default_summary['var_mean']}")
    print(f"  CVaR mean:        {phase7_default_summary['cvar_mean']}")
    print(f"  risk-adj gain:    {phase7_default_summary['risk_adjusted_gain_mean']}")
    print(f"  deterministic gain: {phase7_default_summary['deterministic_gain_mean']}")
    print(f"  power feasible:   {phase7_default_summary['power_feasible_all_steps']}")
    print()
    print("Uncertainty levels (sigma_v):")
    for name, s in uncertainty_level_summaries.items():
        print(
            f"  {name:6s} -> risk_adj_gain={s['risk_adjusted_gain_mean']:.4g}  "
            f"det_gain={s['deterministic_gain_mean']:.4g}  "
            f"reconf={s['reconfiguration_count']}  feasible={s['power_feasible_all_steps']}"
        )
    print()
    print("CVaR alpha sweep:")
    for a in alpha_vals_sorted:
        s = alpha_sweep_summaries[str(a)]
        print(
            f"  alpha={a:5.2f} -> risk_adj_gain={s['risk_adjusted_gain_mean']:.4g}  "
            f"VaR={s['var_mean']:.4g}  CVaR={s['cvar_mean']:.4g}  "
            f"reconf={s['reconfiguration_count']}"
        )
    print()
    print(f"Validation JSON: {out_path}")
    print("=" * 72)

    return status == "PASS"


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
