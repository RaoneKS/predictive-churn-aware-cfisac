"""
src/metrics/prediction.py

Trajectory prediction accuracy metrics.

All metrics operate on arrays of shape (N, H, 2) where:
  N = number of samples
  H = prediction horizon
  2 = [x, y] coordinates in metres

Units
-----
All position-based metrics are in the same units as input (metres in this project).
"""

import numpy as np


def _check_shape(y_true, y_pred, name="array"):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    assert y_true.shape == y_pred.shape, (
        f"{name}: shape mismatch {y_true.shape} vs {y_pred.shape}"
    )
    return y_true, y_pred


# ─────────────────────────────────────────────────────────────────────────────
# Element-wise errors
# ─────────────────────────────────────────────────────────────────────────────

def displacement_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    L2 distance at every predicted step.

    Returns
    -------
    errors : (N, H)   metres
    """
    y_true, y_pred = _check_shape(y_true, y_pred, "displacement_error")
    return np.linalg.norm(y_pred - y_true, axis=-1)   # (N, H)


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate metrics — all return scalars
# ─────────────────────────────────────────────────────────────────────────────

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error across all samples, steps, and coordinates.

    Units : metres
    """
    y_true, y_pred = _check_shape(y_true, y_pred, "mae")
    return float(np.mean(np.abs(y_pred - y_true)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Square Error across all samples, steps, and coordinates.

    Units : metres
    """
    y_true, y_pred = _check_shape(y_true, y_pred, "rmse")
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def final_step_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean L2 displacement at the last predicted horizon step only.

    Units : metres
    """
    y_true, y_pred = _check_shape(y_true, y_pred, "final_step_error")
    return float(np.mean(np.linalg.norm(y_pred[:, -1, :] - y_true[:, -1, :], axis=-1)))


def ade(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Average Displacement Error — mean L2 displacement across all steps and samples.

    ADE = (1 / (N * H)) sum_{n,h} || pred_{n,h} - true_{n,h} ||_2

    Units : metres
    """
    de = displacement_error(y_true, y_pred)   # (N, H)
    return float(np.mean(de))


def fde(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Final Displacement Error — mean L2 displacement at the last step only.

    FDE = (1/N) sum_n || pred_{n,H} - true_{n,H} ||_2

    Units : metres
    """
    y_true, y_pred = _check_shape(y_true, y_pred, "fde")
    return float(np.mean(np.linalg.norm(y_pred[:, -1, :] - y_true[:, -1, :], axis=-1)))


def error_vs_horizon(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Mean L2 displacement at each horizon step h = 1 … H.

    Returns
    -------
    err_per_step : (H,)   metres
    """
    de = displacement_error(y_true, y_pred)   # (N, H)
    return np.mean(de, axis=0)                # (H,)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "",
) -> dict:
    """
    Compute the full suite and return a dict.

    Parameters
    ----------
    y_true, y_pred : (N, H, 2)
    label          : optional prefix for printing

    Returns
    -------
    dict with keys: mae, rmse, final_step_error, ade, fde, error_vs_horizon
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    result = {
        "mae":              mae(y_true, y_pred),
        "rmse":             rmse(y_true, y_pred),
        "final_step_error": final_step_error(y_true, y_pred),
        "ade":              ade(y_true, y_pred),
        "fde":              fde(y_true, y_pred),
        "error_vs_horizon": error_vs_horizon(y_true, y_pred),
    }

    if label:
        prefix = f"[{label}]"
        print(f"{prefix} MAE={result['mae']:.3f}m  RMSE={result['rmse']:.3f}m  "
              f"ADE={result['ade']:.3f}m  FDE={result['fde']:.3f}m")

    return result
