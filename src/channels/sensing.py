import numpy as np


def distance_2d(ap_positions, target_positions):
    """
    Compute AP-target distances.

    Returns
    -------
    ndarray, shape (M, Q)
    """
    ap_positions = np.asarray(ap_positions, dtype=float)
    target_positions = np.asarray(target_positions, dtype=float)

    return np.linalg.norm(
        ap_positions[:, None, :] - target_positions[None, :, :],
        axis=-1,
    )


def generate_sensing_gains(
    ap_positions,
    target_positions,
    rcs=None,
    pathloss_exponent=2.0,
):
    """
    Generate bistatic sensing gains.

    G_sens[m, n, q] represents the gain for:

        AP m -> target q -> AP n

    Returns
    -------
    G_sens : ndarray, shape (M, M, Q)
    """
    ap_positions = np.asarray(ap_positions, dtype=float)
    target_positions = np.asarray(target_positions, dtype=float)

    M = len(ap_positions)
    Q = len(target_positions)

    distances = distance_2d(ap_positions, target_positions)

    if rcs is None:
        rcs = np.ones(Q, dtype=float)
    else:
        rcs = np.asarray(rcs, dtype=float)

    if len(rcs) != Q:
        raise ValueError("rcs must have one value per target.")

    G_sens = np.zeros((M, M, Q), dtype=float)

    for tx in range(M):
        for rx in range(M):
            for q in range(Q):
                d_tx = max(distances[tx, q], 1e-6)
                d_rx = max(distances[rx, q], 1e-6)

                G_sens[tx, rx, q] = (
                    rcs[q]
                    / (d_tx ** pathloss_exponent)
                    / (d_rx ** pathloss_exponent)
                )

    return G_sens
