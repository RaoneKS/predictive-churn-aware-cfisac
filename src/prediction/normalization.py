"""
src/prediction/normalization.py

Deterministic domain normalization for trajectory positions.

Normalization equation
----------------------
  x_norm = (x - area_size / 2) / (area_size / 2)

This maps the simulation domain [0, area_size] to [-1, +1] symmetrically
about the centre.  It depends only on the known simulation parameter
`area_size`, NOT on any training-set statistics (no data leakage).

Inverse (de-normalization)
--------------------------
  x = x_norm * (area_size / 2) + area_size / 2

Units
-----
  Input  : metres (same as MobilitySimulator)
  Output : dimensionless, in [-1, +1]  (approximately; reflective boundary
            may briefly put objects slightly outside [0, area_size], but
            the normalization remains numerically stable)
"""

import numpy as np


def normalize(positions: np.ndarray, area_size: float) -> np.ndarray:
    """
    Normalize positions from metres to [-1, +1].

    Parameters
    ----------
    positions : any shape (..., 2)  in metres
    area_size : simulation area side length (m)

    Returns
    -------
    np.ndarray, same shape, values in [-1, +1]
    """
    centre = area_size / 2.0
    return (np.asarray(positions, dtype=np.float32) - centre) / centre


def denormalize(positions_norm: np.ndarray, area_size: float) -> np.ndarray:
    """
    De-normalize positions from [-1, +1] back to metres.

    Parameters
    ----------
    positions_norm : any shape (..., 2)  normalized
    area_size      : simulation area side length (m)

    Returns
    -------
    np.ndarray, same shape, values in metres
    """
    centre = area_size / 2.0
    return np.asarray(positions_norm, dtype=np.float32) * centre + centre
