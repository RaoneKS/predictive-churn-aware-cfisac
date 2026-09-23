import numpy as np


def distance_2d(ap_positions, user_positions):
    """
    Compute AP-user distances.

    Parameters
    ----------
    ap_positions : ndarray, shape (M, 2)
    user_positions : ndarray, shape (K, 2)

    Returns
    -------
    ndarray, shape (M, K)
    """
    ap_positions = np.asarray(ap_positions, dtype=float)
    user_positions = np.asarray(user_positions, dtype=float)

    return np.linalg.norm(
        ap_positions[:, None, :] - user_positions[None, :, :],
        axis=-1,
    )


def pathloss_gain(
    distances,
    reference_distance=1.0,
    reference_gain=1.0,
    pathloss_exponent=2.0,
):
    """
    Simple distance-based large-scale channel gain.
    """
    distances = np.maximum(np.asarray(distances, dtype=float), reference_distance)

    return reference_gain * (
        reference_distance / distances
    ) ** pathloss_exponent


def generate_comm_gains(
    ap_positions,
    user_positions,
    pathloss_exponent=2.0,
):
    """
    Generate AP-user communication channel gains.

    Returns
    -------
    G_comm : ndarray, shape (M, K)
    """
    distances = distance_2d(ap_positions, user_positions)

    G_comm = pathloss_gain(
        distances,
        pathloss_exponent=pathloss_exponent,
    )

    return G_comm


def spatial_correlation(
    ap_positions,
    user_positions,
    correlation_distance=100.0,
):
    """
    Approximate AP-specific spatial correlation between users.

    Returns
    -------
    S_mat : ndarray, shape (M, K, K)
    """
    ap_positions = np.asarray(ap_positions, dtype=float)
    user_positions = np.asarray(user_positions, dtype=float)

    M = len(ap_positions)
    K = len(user_positions)

    S_mat = np.zeros((M, K, K), dtype=float)

    for m in range(M):
        distances = np.linalg.norm(
            user_positions - ap_positions[m],
            axis=1,
        )

        for k in range(K):
            for j in range(K):
                separation = np.linalg.norm(
                    user_positions[k] - user_positions[j]
                )

                local_scale = max(
                    distances[k],
                    distances[j],
                    1e-9,
                )

                S_mat[m, k, j] = np.exp(
                    -separation
                    / max(correlation_distance, local_scale)
                )

    np.fill_diagonal(S_mat[m], 1.0)

    return S_mat


def generate_communication_features(
    ap_positions,
    user_positions,
    pathloss_exponent=2.0,
    correlation_distance=100.0,
):
    """
    Generate the communication features used by the clustering layer.

    Returns
    -------
    G_comm : ndarray, shape (M, K)
    S_mat   : ndarray, shape (M, K, K)
    """
    G_comm = generate_comm_gains(
        ap_positions,
        user_positions,
        pathloss_exponent=pathloss_exponent,
    )

    S_mat = spatial_correlation(
        ap_positions,
        user_positions,
        correlation_distance=correlation_distance,
    )

    return G_comm, S_mat
