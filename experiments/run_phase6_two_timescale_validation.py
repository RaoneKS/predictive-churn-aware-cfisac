"""
experiments/run_phase6_two_timescale_validation.py

Deterministic Phase 6 validation experiment.

Compares the new two-timescale policy (T_slow > 1, slow association held
fixed across several fast steps) against a per-step reconfiguration policy.

IMPORTANT — how "the existing per-step predictive/churn policy" is realized
here (explicit assumption, not hidden):

    The existing src.simulation.end_to_end PREDICTIVE+CHURN policy operates
    entirely in the SCALAR channel model (src.optimization.churn_aware /
    src.metrics.system) and never touches the physical layer (Phase 4A-5B).
    Phase 6's fast layer, by construction, operates in the PHYSICAL model
    (per-antenna channel, MRT/RZF beamforming, Phase 5B power split).
    A "sum-rate in bps" or "information gain in nats" number from the
    scalar model is not the same quantity as the physical model's, so
    comparing them directly would be comparing apples to oranges rather
    than showing the effect of the timescale separation.

    Instead, the "per-step predictive/churn policy" comparator is realized
    as T_slow=1 of the EXACT SAME two-timescale code path used for the
    slower policy (T_slow=T_SLOW_MULTI). Both comparators therefore share
    the identical physical model, the identical Phase 5B fast optimizer,
    and the identical churn_aware_decision slow layer -- the ONLY thing
    that differs is the update cadence T_slow. This isolates exactly the
    effect Phase 6 introduces (the timescale separation itself) rather
    than conflating it with the pre-existing scalar-vs-physical model
    difference.

Writes results to results/phase6/phase6_two_timescale_validation.json
"""

import json
import os

import numpy as np

from src.environment.topology import generate_ap_positions, generate_positions
from src.prediction.predictor_interface import CVPredictor
from src.simulation.mobility_simulator import MobilitySimulator
from src.optimization.two_timescale import run_two_timescale


NUM_APS = 20
NUM_USERS = 5
NUM_TARGETS = 2
AREA_SIZE = 500.0
NUM_ANTENNAS = 4
P_MAX = 1.0
ALPHA = 1.0
BETA = 1.0
SEED = 42
BASE_FAST_SEED = 7

T_TOTAL = 30
T_SLOW_PER_STEP = 1
T_SLOW_MULTI = 5

FAILURES = []


def _make_scenario():
    ap = generate_ap_positions(NUM_APS, AREA_SIZE, seed=SEED)
    us = generate_positions(NUM_USERS, AREA_SIZE, seed=SEED + 1)
    tg = generate_positions(NUM_TARGETS, AREA_SIZE, seed=SEED + 2)
    sim = MobilitySimulator(us, tg, area_size=AREA_SIZE, dt=1.0, seed=SEED)
    predictor = CVPredictor()
    return ap, sim, predictor


def _run(T_slow, lambda_churn):
    ap, sim, predictor = _make_scenario()
    return run_two_timescale(
        sim, predictor, ap,
        num_antennas=NUM_ANTENNAS,
        T_total=T_TOTAL,
        T_slow=T_slow,
        P_max=P_MAX,
        alpha=ALPHA,
        beta=BETA,
        churn_kwargs={"lambda_churn": lambda_churn},
        base_seed=BASE_FAST_SEED,
    )


def _summarize(result, label):
    fast = result.fast
    slow = result.slow

    n_reconf = sum(1 for d in slow["decision"] if d == "RECONFIGURE")
    n_keep = sum(1 for d in slow["decision"] if d == "KEEP")
    raw_churn_total = float(np.sum(slow["raw_churn"]))

    sum_rate = float(np.sum(fast["comm_utility_raw"]))
    mean_sum_rate = float(np.mean(fast["comm_utility_raw"]))
    info_gain = float(np.sum(fast["sensing_utility_raw"]))
    mean_info_gain = float(np.mean(fast["sensing_utility_raw"]))

    feasible_all = bool(all(fast["feasible"]))
    max_violation = float(np.max(fast["max_power_violation"]))

    summary = {
        "label": label,
        "T_slow": result.T_slow,
        "num_fast_updates": result.num_fast_updates,
        "num_slow_updates": result.num_slow_updates,
        "reconfiguration_count": n_reconf,
        "keep_count": n_keep,
        "raw_churn_total": raw_churn_total,
        "communication_sum_rate_bps_total": sum_rate,
        "communication_sum_rate_bps_mean": mean_sum_rate,
        "sensing_information_gain_total": info_gain,
        "sensing_information_gain_mean": mean_info_gain,
        "power_feasible_all_steps": feasible_all,
        "max_power_violation": max_violation,
    }

    if not feasible_all:
        FAILURES.append(f"{label}: infeasible fast step detected")
    if result.num_fast_updates != T_TOTAL:
        FAILURES.append(f"{label}: fast update count mismatch")

    return summary


def main():
    # -----------------------------------------------------------------
    # 1. Determinism check: run the multi-step policy twice
    # -----------------------------------------------------------------
    r1 = _run(T_SLOW_MULTI, lambda_churn=1.0)
    r2 = _run(T_SLOW_MULTI, lambda_churn=1.0)
    deterministic = (
        r1.slow["decision"] == r2.slow["decision"]
        and all(
            np.allclose(a, b)
            for a, b in zip(r1.fast["P_comm"], r2.fast["P_comm"])
        )
    )
    if not deterministic:
        FAILURES.append("multi-step policy is not deterministic")

    # -----------------------------------------------------------------
    # 2. Per-step (T_slow=1) vs two-timescale (T_slow=5) comparison,
    #    same lambda_churn, isolating the effect of the schedule itself.
    # -----------------------------------------------------------------
    per_step = _run(T_SLOW_PER_STEP, lambda_churn=1.0)
    multi_step = _run(T_SLOW_MULTI, lambda_churn=1.0)

    per_step_summary = _summarize(per_step, "per_step (T_slow=1)")
    multi_step_summary = _summarize(multi_step, "two_timescale (T_slow=5)")

    # -----------------------------------------------------------------
    # 3. Zero / low / high churn regimes at T_slow=5
    # -----------------------------------------------------------------
    zero_churn = _run(T_SLOW_MULTI, lambda_churn=0.0)
    low_churn = _run(T_SLOW_MULTI, lambda_churn=0.1)
    high_churn = _run(T_SLOW_MULTI, lambda_churn=1e9)

    zero_summary = _summarize(zero_churn, "zero_churn (lambda=0)")
    low_summary = _summarize(low_churn, "low_churn (lambda=0.1)")
    high_summary = _summarize(high_churn, "high_churn (lambda=1e9)")

    if high_summary["reconfiguration_count"] != 0:
        FAILURES.append("high-churn regime reconfigured at least once")
    if zero_summary["reconfiguration_count"] < high_summary["reconfiguration_count"]:
        FAILURES.append("zero-churn regime reconfigured less than high-churn regime")

    # -----------------------------------------------------------------
    # 4. Slow-vs-fast update counts (schedule sanity)
    # -----------------------------------------------------------------
    expected_multi_epochs = -(-T_TOTAL // T_SLOW_MULTI)  # ceil division
    if multi_step_summary["num_slow_updates"] != expected_multi_epochs:
        FAILURES.append(
            f"multi-step epoch count {multi_step_summary['num_slow_updates']} "
            f"!= expected {expected_multi_epochs}"
        )
    if per_step_summary["num_slow_updates"] != T_TOTAL:
        FAILURES.append("per-step policy did not reconfigure-check every step")

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
            "P_max": P_MAX,
            "alpha": ALPHA,
            "beta": BETA,
            "seed": SEED,
            "base_fast_seed": BASE_FAST_SEED,
        },
        "checks": {
            "deterministic": deterministic,
            "per_step_power_feasible": per_step_summary["power_feasible_all_steps"],
            "multi_step_power_feasible": multi_step_summary["power_feasible_all_steps"],
            "high_churn_zero_reconfigurations": high_summary["reconfiguration_count"] == 0,
            "zero_churn_gte_high_churn_reconfigurations": (
                zero_summary["reconfiguration_count"]
                >= high_summary["reconfiguration_count"]
            ),
            "schedule_counts_correct": (
                multi_step_summary["num_slow_updates"] == expected_multi_epochs
                and per_step_summary["num_slow_updates"] == T_TOTAL
            ),
        },
        "comparison": {
            "per_step": per_step_summary,
            "two_timescale": multi_step_summary,
        },
        "churn_regimes": {
            "zero_churn": zero_summary,
            "low_churn": low_summary,
            "high_churn": high_summary,
        },
        "assumption_note": (
            "The 'per-step predictive/churn policy' comparator is T_slow=1 "
            "of this same physical two-timescale code path, NOT the "
            "existing scalar-model end_to_end.PREDICTIVE+CHURN policy -- "
            "the two use different objective spaces (scalar vs physical) "
            "and are not directly comparable on sum-rate/information-gain. "
            "See module docstring."
        ),
        "failures": FAILURES,
    }

    os.makedirs("results/phase6", exist_ok=True)
    out_path = "results/phase6/phase6_two_timescale_validation.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print("=" * 72)
    print("Phase 6 Two-Timescale Optimization Validation")
    print("=" * 72)
    print(f"Status: {status}")
    print(f"Scenario: M={NUM_APS}, K={NUM_USERS}, Q={NUM_TARGETS}, N={NUM_ANTENNAS}")
    print()
    print(f"Deterministic: {deterministic}")
    print()
    print("Per-step (T_slow=1):")
    print(f"  reconfigurations: {per_step_summary['reconfiguration_count']}")
    print(f"  fast updates: {per_step_summary['num_fast_updates']}")
    print(f"  slow updates: {per_step_summary['num_slow_updates']}")
    print(f"  sum-rate [bps, mean/step]: {per_step_summary['communication_sum_rate_bps_mean']:.4g}")
    print(f"  info gain [mean/step]: {per_step_summary['sensing_information_gain_mean']:.4g}")
    print(f"  power feasible: {per_step_summary['power_feasible_all_steps']}")
    print()
    print("Two-timescale (T_slow=5):")
    print(f"  reconfigurations: {multi_step_summary['reconfiguration_count']}")
    print(f"  fast updates: {multi_step_summary['num_fast_updates']}")
    print(f"  slow updates: {multi_step_summary['num_slow_updates']}")
    print(f"  sum-rate [bps, mean/step]: {multi_step_summary['communication_sum_rate_bps_mean']:.4g}")
    print(f"  info gain [mean/step]: {multi_step_summary['sensing_information_gain_mean']:.4g}")
    print(f"  power feasible: {multi_step_summary['power_feasible_all_steps']}")
    print()
    print("Churn regimes (T_slow=5):")
    for k, s in (("zero", zero_summary), ("low", low_summary), ("high", high_summary)):
        print(
            f"  {k:5s} lambda -> reconf={s['reconfiguration_count']:2d}  "
            f"raw_churn={s['raw_churn_total']:.3g}  "
            f"sum_rate_mean={s['communication_sum_rate_bps_mean']:.4g}"
        )
    print()
    print(f"Validation JSON: {out_path}")
    print("=" * 72)

    return status == "PASS"


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
