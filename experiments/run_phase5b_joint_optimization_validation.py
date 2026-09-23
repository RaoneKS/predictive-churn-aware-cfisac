"""
Phase 5B validation experiment.

Uses the authoritative deterministic fixture from
tests/test_phase5b_joint_optimization.py.

This script does NOT invent or override optimizer defaults such as:
    noise_power_comm
    ref_sum_rate
    noise_power_sens
    reference_variance

The current optimize_joint_allocation() implementation owns those defaults.

Validation performed:
    1. MRT communication-only boundary
    2. MRT sensing-only boundary
    3. MRT joint optimization
    4. RZF joint optimization
    5. dual_role vs all communication-power scope
    6. beta sweep
    7. deterministic repeatability
    8. per-AP power feasibility
    9. exact combined-power accounting
    10. communication/sensing evaluation consistency
    11. positive sensing utility for joint operation
    12. JSON result export
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np

from src.optimization.joint_comm_sensing import (
    comm_beam_directions,
    evaluate_joint_allocation,
    optimize_joint_allocation,
)

from tests.test_phase5b_joint_optimization import _SC


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _supported_kwargs(func, supplied):
    """
    Pass only arguments accepted by the current function signature.

    Required parameters without defaults are checked explicitly.
    Parameters with defaults are allowed to use the implementation's
    own defaults.
    """
    sig = inspect.signature(func)
    kwargs = {}

    for name, param in sig.parameters.items():
        if name == "self":
            continue

        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        if name in supplied:
            kwargs[name] = supplied[name]
        elif param.default is inspect.Parameter.empty:
            raise TypeError(
                f"{func.__name__} requires parameter '{name}', "
                "but the Phase 5B validation fixture does not provide it."
            )

    return kwargs


def _run_optimizer(
    sc,
    W_dir,
    *,
    alpha,
    beta,
    comm_power_control="dual_role",
):
    # Match the authoritative Phase 5B test fixture exactly:
    # _args(sc) passes every non-private scenario field.
    supplied = {
        k: v
        for k, v in sc.items()
        if not k.startswith("_")
    }

    # Override the communication beam directions for the requested
    # MRT/RZF experiment.
    supplied["W_dir"] = W_dir

    # Experiment controls.
    supplied["alpha"] = alpha
    supplied["beta"] = beta
    supplied["comm_power_control"] = comm_power_control

    kwargs = _supported_kwargs(
        optimize_joint_allocation,
        supplied,
    )

    return optimize_joint_allocation(**kwargs)


def _evaluate(
    sc,
    W_dir,
    P_comm,
    P_sens,
):
    # Again mirror the authoritative fixture. All non-private fields
    # are available to the evaluator, while the allocation and W_dir
    # are replaced for this specific evaluation.
    supplied = {
        k: v
        for k, v in sc.items()
        if not k.startswith("_")
    }

    supplied["W_dir"] = W_dir
    supplied["P_comm_setpoint"] = np.asarray(
        P_comm,
        dtype=float,
    )
    supplied["sensing_power"] = np.asarray(
        P_sens,
        dtype=float,
    )

    kwargs = _supported_kwargs(
        evaluate_joint_allocation,
        supplied,
    )

    return evaluate_joint_allocation(**kwargs)


def _array(result, key):
    return np.asarray(result[key], dtype=float)


def _max_power_violation(result):
    p_comm = _array(result, "P_comm")
    p_sens = _array(result, "P_sens")

    if "P_max" in result:
        p_max = _array(result, "P_max")
    else:
        p_max = np.asarray(_SC["P_max"], dtype=float)

    return float(
        np.max(
            p_comm + p_sens - p_max
        )
    )


def _power_accounting_error(result):
    p_comm = _array(result, "P_comm")
    p_sens = _array(result, "P_sens")

    if "P_tx" not in result:
        p_tx = p_comm + p_sens
    else:
        p_tx = _array(result, "P_tx")

    return float(
        np.max(
            np.abs(
                p_tx - p_comm - p_sens
            )
        )
    )


def _summary(result):
    rates = _array(result, "rates_bps")
    info_gain = _array(result, "information_gain")

    return {
        "sum_rate_bps": float(np.sum(rates)),
        "information_gain": float(np.sum(info_gain)),
        "objective": float(result["objective"]),
        "max_power_violation": _max_power_violation(result),
        "power_accounting_error": _power_accounting_error(result),
        "P_comm": _array(result, "P_comm").tolist(),
        "P_sens": _array(result, "P_sens").tolist(),
        "P_tx": (
            _array(result, "P_tx").tolist()
            if "P_tx" in result
            else (
                _array(result, "P_comm")
                + _array(result, "P_sens")
            ).tolist()
        ),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    sc = _SC

    H = np.asarray(sc["H"])
    M, K, N = H.shape

    # The authoritative fixture stores the communication association
    # as "_x".
    association = sc["_x"]

    # ---------------------------------------------------------------
    # Build communication directions from the same authoritative
    # topology used by the Phase 5B tests.
    # ---------------------------------------------------------------

    W_mrt = comm_beam_directions(
        H,
        association,
        precoder="mrt",
    )

    W_rzf = comm_beam_directions(
        H,
        association,
        precoder="rzf",
    )

    # ---------------------------------------------------------------
    # Boundary cases
    # ---------------------------------------------------------------

    mrt_comm_only = _run_optimizer(
        sc,
        W_mrt,
        alpha=1.0,
        beta=0.0,
        comm_power_control="dual_role",
    )

    mrt_sensing_only = _run_optimizer(
        sc,
        W_mrt,
        alpha=0.0,
        beta=1.0,
        comm_power_control="dual_role",
    )

    # ---------------------------------------------------------------
    # Main joint cases
    # ---------------------------------------------------------------

    mrt_joint = _run_optimizer(
        sc,
        W_mrt,
        alpha=1.0,
        beta=1.0,
        comm_power_control="dual_role",
    )

    rzf_joint = _run_optimizer(
        sc,
        W_rzf,
        alpha=1.0,
        beta=1.0,
        comm_power_control="dual_role",
    )

    mrt_joint_all = _run_optimizer(
        sc,
        W_mrt,
        alpha=1.0,
        beta=1.0,
        comm_power_control="all",
    )

    # ---------------------------------------------------------------
    # Beta sweep
    # ---------------------------------------------------------------

    beta_values = [
        0.0,
        0.01,
        0.1,
        1.0,
        10.0,
        100.0,
    ]

    beta_results = {}

    for beta in beta_values:
        beta_results[str(beta)] = _run_optimizer(
            sc,
            W_mrt,
            alpha=1.0,
            beta=beta,
            comm_power_control="dual_role",
        )

    # ---------------------------------------------------------------
    # Determinism
    # ---------------------------------------------------------------

    repeat_a = _run_optimizer(
        sc,
        W_mrt,
        alpha=1.0,
        beta=1.0,
        comm_power_control="dual_role",
    )

    repeat_b = _run_optimizer(
        sc,
        W_mrt,
        alpha=1.0,
        beta=1.0,
        comm_power_control="dual_role",
    )

    deterministic = bool(
        np.array_equal(
            _array(repeat_a, "P_comm"),
            _array(repeat_b, "P_comm"),
        )
        and
        np.array_equal(
            _array(repeat_a, "P_sens"),
            _array(repeat_b, "P_sens"),
        )
    )

    # ---------------------------------------------------------------
    # Evaluate the returned allocations again using the current
    # evaluator. This validates optimizer -> evaluator consistency.
    # ---------------------------------------------------------------

    comm_eval = _evaluate(
        sc,
        W_mrt,
        mrt_comm_only["P_comm"],
        mrt_comm_only["P_sens"],
    )

    sensing_eval = _evaluate(
        sc,
        W_mrt,
        mrt_sensing_only["P_comm"],
        mrt_sensing_only["P_sens"],
    )

    joint_eval = _evaluate(
        sc,
        W_mrt,
        mrt_joint["P_comm"],
        mrt_joint["P_sens"],
    )

    comm_eval_rate_error = float(
        np.max(
            np.abs(
                _array(comm_eval, "rates_bps")
                -
                _array(mrt_comm_only, "rates_bps")
            )
        )
    )

    sensing_eval_info_error = float(
        np.max(
            np.abs(
                _array(sensing_eval, "information_gain")
                -
                _array(mrt_sensing_only, "information_gain")
            )
        )
    )

    joint_eval_rate_error = float(
        np.max(
            np.abs(
                _array(joint_eval, "rates_bps")
                -
                _array(mrt_joint, "rates_bps")
            )
        )
    )

    joint_eval_info_error = float(
        np.max(
            np.abs(
                _array(joint_eval, "information_gain")
                -
                _array(mrt_joint, "information_gain")
            )
        )
    )

    # ---------------------------------------------------------------
    # Boundary checks
    # ---------------------------------------------------------------

    comm_only_sensing_power = float(
        np.max(
            np.abs(
                _array(mrt_comm_only, "P_sens")
            )
        )
    )

    sensing_only_comm_power = float(
        np.max(
            np.abs(
                _array(mrt_sensing_only, "P_comm")
            )
        )
    )

    joint_info_gain = float(
        np.sum(
            _array(mrt_joint, "information_gain")
        )
    )

    # ---------------------------------------------------------------
    # Per-result feasibility
    # ---------------------------------------------------------------

    named_results = {
        "mrt_comm_only": mrt_comm_only,
        "mrt_sensing_only": mrt_sensing_only,
        "mrt_joint": mrt_joint,
        "rzf_joint": rzf_joint,
        "mrt_joint_all": mrt_joint_all,
    }

    feasibility = {}

    for name, result in named_results.items():
        violation = _max_power_violation(result)
        accounting = _power_accounting_error(result)

        feasibility[name] = {
            "feasible": bool(
                violation <= 1e-9
            ),
            "max_power_violation": violation,
            "power_accounting_error": accounting,
        }

    for beta_key, result in beta_results.items():
        violation = _max_power_violation(result)
        accounting = _power_accounting_error(result)

        feasibility[f"beta_{beta_key}"] = {
            "feasible": bool(
                violation <= 1e-9
            ),
            "max_power_violation": violation,
            "power_accounting_error": accounting,
        }

    # ---------------------------------------------------------------
    # Hard validation gates
    # ---------------------------------------------------------------

    failures = []

    if not deterministic:
        failures.append(
            "optimizer is not deterministic"
        )

    if any(
        not item["feasible"]
        for item in feasibility.values()
    ):
        failures.append(
            "one or more allocations violate "
            "P_comm + P_sens <= P_max"
        )

    if any(
        item["power_accounting_error"] > 1e-9
        for item in feasibility.values()
    ):
        failures.append(
            "P_tx != P_comm + P_sens"
        )

    if comm_only_sensing_power > 1e-9:
        failures.append(
            "communication-only case has nonzero sensing power"
        )

    if sensing_only_comm_power > 1e-9:
        failures.append(
            "sensing-only case has nonzero communication power"
        )

    if comm_eval_rate_error > 1e-6:
        failures.append(
            f"communication-only evaluator mismatch: "
            f"{comm_eval_rate_error} bps"
        )

    if sensing_eval_info_error > 1e-9:
        failures.append(
            f"sensing-only evaluator mismatch: "
            f"{sensing_eval_info_error}"
        )

    if joint_eval_rate_error > 1e-6:
        failures.append(
            f"joint evaluator rate mismatch: "
            f"{joint_eval_rate_error} bps"
        )

    if joint_eval_info_error > 1e-9:
        failures.append(
            f"joint evaluator sensing mismatch: "
            f"{joint_eval_info_error}"
        )

    if joint_info_gain <= 0.0:
        failures.append(
            "joint optimization produced no positive sensing information gain"
        )

    status = "PASS" if not failures else "FAIL"

    # ---------------------------------------------------------------
    # JSON output
    # ---------------------------------------------------------------

    output = {
        "status": status,
        "scenario": {
            "M": int(M),
            "K": int(K),
            "N": int(N),
            "P_max": np.asarray(
                sc["P_max"],
                dtype=float,
            ).tolist(),
        },
        "checks": {
            "deterministic": deterministic,
            "all_cases_feasible": all(
                item["feasible"]
                for item in feasibility.values()
            ),
            "power_accounting": all(
                item["power_accounting_error"] <= 1e-9
                for item in feasibility.values()
            ),
            "comm_only_has_no_sensing_power": (
                comm_only_sensing_power <= 1e-9
            ),
            "sensing_only_has_no_comm_power": (
                sensing_only_comm_power <= 1e-9
            ),
            "positive_joint_sensing_information_gain": (
                joint_info_gain > 0.0
            ),
            "comm_eval_rate_error_bps": comm_eval_rate_error,
            "sensing_eval_info_error": sensing_eval_info_error,
            "joint_eval_rate_error_bps": joint_eval_rate_error,
            "joint_eval_info_error": joint_eval_info_error,
        },
        "feasibility": feasibility,
        "results": {
            "mrt_comm_only": _summary(mrt_comm_only),
            "mrt_sensing_only": _summary(mrt_sensing_only),
            "mrt_joint": _summary(mrt_joint),
            "rzf_joint": _summary(rzf_joint),
            "mrt_joint_all": _summary(mrt_joint_all),
        },
        "beta_sweep": {
            beta: _summary(result)
            for beta, result in beta_results.items()
        },
        "failures": failures,
    }

    output_path = Path(
        "results/phase5b/"
        "phase5b_joint_optimization_validation.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            output,
            indent=2,
        )
    )

    # ---------------------------------------------------------------
    # Console report
    # ---------------------------------------------------------------

    print("=" * 72)
    print("Phase 5B Joint Communication + Sensing Validation")
    print("=" * 72)
    print(f"Status: {status}")
    print(
        f"Scenario: M={M}, K={K}, N={N}"
    )
    print()

    print(
        "Deterministic:",
        deterministic,
    )

    print(
        "All cases feasible:",
        all(
            item["feasible"]
            for item in feasibility.values()
        ),
    )

    print(
        "Power accounting:",
        all(
            item["power_accounting_error"] <= 1e-9
            for item in feasibility.values()
        ),
    )

    print(
        "Communication-only sensing power:",
        comm_only_sensing_power,
    )

    print(
        "Sensing-only communication power:",
        sensing_only_comm_power,
    )

    print(
        "Joint sensing information gain:",
        joint_info_gain,
    )

    print()
    print("MRT communication-only:")
    print(
        "  sum-rate [bps]:",
        np.sum(
            _array(
                mrt_comm_only,
                "rates_bps",
            )
        ),
    )

    print()
    print("MRT sensing-only:")
    print(
        "  info gain:",
        np.sum(
            _array(
                mrt_sensing_only,
                "information_gain",
            )
        ),
    )

    print()
    print("MRT joint:")
    print(
        "  sum-rate [bps]:",
        np.sum(
            _array(
                mrt_joint,
                "rates_bps",
            )
        ),
    )
    print(
        "  info gain:",
        joint_info_gain,
    )
    print(
        "  objective:",
        mrt_joint["objective"],
    )

    print()
    print("RZF joint:")
    print(
        "  sum-rate [bps]:",
        np.sum(
            _array(
                rzf_joint,
                "rates_bps",
            )
        ),
    )
    print(
        "  info gain:",
        np.sum(
            _array(
                rzf_joint,
                "information_gain",
            )
        ),
    )
    print(
        "  objective:",
        rzf_joint["objective"],
    )

    print()
    print("Beta sweep:")

    for beta, result in beta_results.items():
        print(
            f"  beta={beta}: "
            f"sum_rate="
            f"{np.sum(_array(result, 'rates_bps')):.6g}, "
            f"info_gain="
            f"{np.sum(_array(result, 'information_gain')):.6g}, "
            f"objective="
            f"{float(result['objective']):.6g}, "
            f"violation="
            f"{_max_power_violation(result):.3e}"
        )

    print()
    print(
        "Validation JSON:",
        output_path,
    )
    print("=" * 72)

    if failures:
        print()
        print("FAILURES:")
        for failure in failures:
            print(f"  - {failure}")

        raise SystemExit(1)


if __name__ == "__main__":
    main()
