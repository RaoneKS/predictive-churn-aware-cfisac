import numpy as np


def communication_clustering(
    G_comm,
    num_users,
    aps_per_user=3,
):
    """
    User-centric overlapping communication clustering.

    Each user receives its strongest APs independently.
    Therefore, the same AP may belong to multiple users.

    Returns
    -------
    x : ndarray, shape (M, K)
        x[m,k] = 1 if AP m serves user k.
    """
    G_comm = np.asarray(G_comm, dtype=float)

    M, K = G_comm.shape

    if num_users != K:
        raise ValueError("num_users must match G_comm.shape[1].")

    aps_per_user = min(max(1, aps_per_user), M)

    x = np.zeros((M, K), dtype=int)

    for k in range(K):
        selected = np.argsort(G_comm[:, k])[-aps_per_user:]
        x[selected, k] = 1

    return x


def sensing_clustering(
    G_sens,
    num_targets,
    tx_aps_per_target=2,
    rx_aps_per_target=2,
):
    """
    Target-centric overlapping sensing clustering.

    TX and RX clusters are selected independently.

    G_sens[m,n,q] represents the usefulness of
    AP m as TX and AP n as RX for target q.

    Returns
    -------
    y_tx : ndarray, shape (M,Q)
    y_rx : ndarray, shape (M,Q)
    """
    G_sens = np.asarray(G_sens, dtype=float)

    M, M_rx, Q = G_sens.shape

    if M != M_rx:
        raise ValueError("G_sens must have shape (M,M,Q).")

    if num_targets != Q:
        raise ValueError(
            "num_targets must match G_sens.shape[2]."
        )

    tx_aps_per_target = min(
        max(1, tx_aps_per_target),
        M,
    )

    rx_aps_per_target = min(
        max(1, rx_aps_per_target),
        M,
    )

    y_tx = np.zeros((M, Q), dtype=int)
    y_rx = np.zeros((M, Q), dtype=int)

    for q in range(Q):

        # TX score: best receiving AP for each candidate TX.
        tx_score = np.max(G_sens[:, :, q], axis=1)

        # RX score: best transmitting AP for each candidate RX.
        rx_score = np.max(G_sens[:, :, q], axis=0)

        tx_selected = np.argsort(tx_score)[-tx_aps_per_target:]
        rx_selected = np.argsort(rx_score)[-rx_aps_per_target:]

        y_tx[tx_selected, q] = 1
        y_rx[rx_selected, q] = 1

    return y_tx, y_rx


def build_overlapping_clusters(
    G_comm,
    G_sens,
    num_users,
    num_targets,
    aps_per_user=3,
    tx_aps_per_target=2,
    rx_aps_per_target=2,
):
    """
    Build communication and sensing clusters independently.

    This explicitly permits overlap between:

        communication
        sensing TX
        sensing RX

    and also permits an AP to serve multiple users/targets.
    """
    x = communication_clustering(
        G_comm,
        num_users=num_users,
        aps_per_user=aps_per_user,
    )

    y_tx, y_rx = sensing_clustering(
        G_sens,
        num_targets=num_targets,
        tx_aps_per_target=tx_aps_per_target,
        rx_aps_per_target=rx_aps_per_target,
    )

    return x, y_tx, y_rx
