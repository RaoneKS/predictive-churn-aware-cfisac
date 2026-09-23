import numpy as np

from src.channels.communication import (
    generate_communication_features,
)
from src.channels.sensing import (
    generate_sensing_gains,
)
from src.clustering.overlapping import (
    build_overlapping_clusters,
)


def cluster_for_positions(
    ap_positions,
    user_positions,
    target_positions,
    aps_per_user=3,
    tx_aps_per_target=2,
    rx_aps_per_target=2,
):
    """
    Generate communication and sensing clusters for a given
    network state.
    """
    G_comm, _ = generate_communication_features(
        ap_positions,
        user_positions,
    )

    G_sens = generate_sensing_gains(
        ap_positions,
        target_positions,
    )

    return build_overlapping_clusters(
        G_comm,
        G_sens,
        num_users=len(user_positions),
        num_targets=len(target_positions),
        aps_per_user=aps_per_user,
        tx_aps_per_target=tx_aps_per_target,
        rx_aps_per_target=rx_aps_per_target,
    )


def static_policy(
    ap_positions,
    initial_user_positions,
    initial_target_positions,
    **cluster_kwargs,
):
    """
    STATIC baseline.

    Clusters are selected once at the initial state and
    remain unchanged.
    """
    return cluster_for_positions(
        ap_positions,
        initial_user_positions,
        initial_target_positions,
        **cluster_kwargs,
    )


def reactive_policy(
    ap_positions,
    user_positions,
    target_positions,
    **cluster_kwargs,
):
    """
    REACTIVE baseline.

    Clusters are selected using the current network state.
    """
    return cluster_for_positions(
        ap_positions,
        user_positions,
        target_positions,
        **cluster_kwargs,
    )


def predictive_policy(
    ap_positions,
    predicted_user_positions,
    predicted_target_positions,
    **cluster_kwargs,
):
    """
    PREDICTIVE baseline.

    Clusters are selected using predicted future positions.
    """
    return cluster_for_positions(
        ap_positions,
        predicted_user_positions,
        predicted_target_positions,
        **cluster_kwargs,
    )


def cluster_difference(
    previous_x,
    current_x,
    previous_y_tx,
    current_y_tx,
    previous_y_rx,
    current_y_rx,
):
    """
    Count topology changes between two decisions.
    """
    return (
        np.sum(
            np.abs(
                np.asarray(current_x)
                - np.asarray(previous_x)
            )
        )
        + np.sum(
            np.abs(
                np.asarray(current_y_tx)
                - np.asarray(previous_y_tx)
            )
        )
        + np.sum(
            np.abs(
                np.asarray(current_y_rx)
                - np.asarray(previous_y_rx)
            )
        )
    )
