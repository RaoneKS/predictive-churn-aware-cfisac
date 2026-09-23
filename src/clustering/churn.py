import numpy as np


def binary_churn(previous, current):
    return np.abs(np.asarray(current) - np.asarray(previous))


def total_churn_cost(
    previous_x,
    current_x,
    previous_y_tx,
    current_y_tx,
    previous_y_rx,
    current_y_rx,
    c_comm=1.0,
    c_sensing_tx=1.0,
    c_sensing_rx=1.0,
):
    return (
        c_comm * np.sum(binary_churn(previous_x, current_x))
        + c_sensing_tx * np.sum(binary_churn(previous_y_tx, current_y_tx))
        + c_sensing_rx * np.sum(binary_churn(previous_y_rx, current_y_rx))
    )


def max_raw_churn_cost(
    num_users,
    num_targets,
    aps_per_user=3,
    tx_aps_per_target=2,
    rx_aps_per_target=2,
    num_aps=None,
    c_comm=1.0,
    c_sensing_tx=1.0,
    c_sensing_rx=1.0,
):
    """
    Maximum feasible raw weighted churn cost for one reconfiguration.

    Derivation
    ----------
    `total_churn_cost` sums |current - previous| over the binary assignment
    matrices.  The overlapping clustering in src/clustering/overlapping.py
    selects EXACTLY `aps_per_user` APs per user (argsort[-k:]) and EXACTLY
    `tx_aps_per_target` / `rx_aps_per_target` APs per target, so every column
    of x, y_tx, y_rx has a fixed, known cardinality.

    For a single column with cardinality k, the worst case is a completely
    disjoint new selection: k entries fall 1 -> 0 and k entries rise 0 -> 1,
    giving an L1 difference of 2k (NOT k).  Hence:

        max_churn = c_comm      * 2 * aps_per_user       * num_users
                  + c_sensing_tx* 2 * tx_aps_per_target  * num_targets
                  + c_sensing_rx* 2 * rx_aps_per_target  * num_targets

    This bound is attainable whenever num_aps >= 2 * k for each cluster type;
    if num_aps < 2k the disjoint selection is impossible and the bound is
    reduced to the number of distinct APs that can actually change.

    For the default configuration (20 APs, 5 users, 2 targets, 3/2/2, unit
    costs) this evaluates to 2*15 + 2*4 + 2*4 = 46.

    Note
    ----
    Counting "assignment events" one-directionally would give 23.  That is NOT
    the quantity total_churn_cost computes, so 23 would under-normalize the
    penalty by exactly a factor of two.
    """
    def _column_max(k, m):
        k = max(1, int(k))
        if m is not None:
            k = min(k, int(m))
            # Cannot swap out more APs than exist outside the current set.
            return k + min(k, int(m) - k)
        return 2 * k

    return float(
        c_comm        * _column_max(aps_per_user,      num_aps) * int(num_users)
        + c_sensing_tx * _column_max(tx_aps_per_target, num_aps) * int(num_targets)
        + c_sensing_rx * _column_max(rx_aps_per_target, num_aps) * int(num_targets)
    )
