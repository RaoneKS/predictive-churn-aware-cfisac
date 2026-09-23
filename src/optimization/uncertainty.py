"""
src/optimization/uncertainty.py

Phase 7 -- uncertainty sourcing and scenario generation.

This module is ADDITIVE. It reuses, read-only and unmodified:

    * src.simulation.mobility_simulator.MobilitySimulator.position_history
      (Phase 2) -- the only genuinely MEASURED trajectory record kept by
      the project.

It does NOT touch, modify, or peek inside src.prediction.*.

MEASURED
--------
measured_cv_residuals() computes the empirical error of a deterministic
constant-velocity extrapolation against recorded future positions.

For anchor step t0:

    v_hat(t0)   = (pos[t0] - pos[t0-1]) / dt
    cv_pred(t0,h) = pos[t0] + h * v_hat(t0) * dt
    residual(t0,h) = pos[t0+h] - cv_pred(t0,h)

These are empirical historical prediction errors computed from measured
trajectory records.

DERIVED
-------
generate_scenarios() resamples the measured residual pool with replacement.
Those perturbations are derived scenarios, not measurements.

Determinism
-----------
measured_cv_residuals() is deterministic.
generate_scenarios() is deterministic for fixed inputs and seed.
"""

from __future__ import annotations

import numpy as np


def measured_cv_residuals(position_series, horizon, dt=1.0):
    """
    Compute empirical constant-velocity extrapolation residuals.

    Parameters
    ----------
    position_series : sequence of ndarray, each shape (N, 2)
        Recorded positions in chronological order.
    horizon : int
        Number of future steps per anchor.
    dt : float
        Time spacing between recorded samples.

    Returns
    -------
    residuals : ndarray, shape (S, N, H, 2)
        Empirical historical CV prediction errors.
    """
    if not isinstance(horizon, (int, np.integer)):
        raise TypeError("horizon must be an integer.")

    if horizon < 1:
        raise ValueError("horizon must be >= 1.")

    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and > 0.")

    series = [
        np.asarray(p, dtype=float)
        for p in position_series
    ]

    if not series:
        return np.zeros(
            (0, 0, horizon, 2),
            dtype=float,
        )

    for p in series:
        if p.ndim != 2 or p.shape[1] != 2:
            raise ValueError(
                "Each position history entry must have shape (N, 2)."
            )

        if not np.all(np.isfinite(p)):
            raise ValueError(
                "Position history must contain only finite values."
            )

    N = series[0].shape[0]

    if any(p.shape != (N, 2) for p in series):
        raise ValueError(
            "All position history entries must have the same shape."
        )

    T = len(series)

    # Need:
    #   one step before t0 for the velocity estimate
    #   horizon steps after t0 for observed future positions
    n_anchors = T - horizon - 1

    if n_anchors <= 0:
        return np.zeros(
            (0, N, horizon, 2),
            dtype=float,
        )

    residuals = np.zeros(
        (n_anchors, N, horizon, 2),
        dtype=float,
    )

    for i, t0 in enumerate(
        range(1, T - horizon)
    ):
        v_hat = (
            series[t0] - series[t0 - 1]
        ) / dt

        for h in range(1, horizon + 1):
            cv_pred = (
                series[t0]
                + h * v_hat * dt
            )

            residuals[
                i, :, h - 1, :
            ] = (
                series[t0 + h]
                - cv_pred
            )

    return residuals


def generate_scenarios(
    residuals,
    num_scenarios,
    seed,
):
    """
    DERIVED: resample a measured residual pool into synthetic scenarios.

    Parameters
    ----------
    residuals : ndarray, shape (S, N, H, 2)
        Empirical residual pool.
    num_scenarios : int
        Number of scenarios to generate.
    seed : int
        Random seed.

    Returns
    -------
    scenarios : ndarray, shape (num_scenarios, N, H, 2)
        Derived perturbation scenarios.
    """
    residuals = np.asarray(
        residuals,
        dtype=float,
    )

    if residuals.ndim != 4:
        raise ValueError(
            "residuals must have shape (S, N, H, 2)."
        )

    if residuals.shape[-1] != 2:
        raise ValueError(
            "residuals must have final dimension 2."
        )

    if not np.all(np.isfinite(residuals)):
        raise ValueError(
            "residuals must contain only finite values."
        )

    if not isinstance(
        num_scenarios,
        (int, np.integer),
    ):
        raise TypeError(
            "num_scenarios must be an integer."
        )

    if num_scenarios < 0:
        raise ValueError(
            "num_scenarios must be >= 0."
        )

    _, N, H, _ = residuals.shape

    if num_scenarios == 0:
        return np.zeros(
            (0, N, H, 2),
            dtype=float,
        )

    S = residuals.shape[0]

    if S == 0:
        return np.zeros(
            (num_scenarios, N, H, 2),
            dtype=float,
        )

    rng = np.random.default_rng(seed)

    indices = rng.integers(
        0,
        S,
        size=num_scenarios,
    )

    return residuals[indices].copy()


__all__ = [
    "measured_cv_residuals",
    "generate_scenarios",
]
