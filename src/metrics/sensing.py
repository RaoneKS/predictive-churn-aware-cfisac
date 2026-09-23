import numpy as np


def fisher_information_from_pair(
    tx_position,
    rx_position,
    target_position,
    measurement_noise=1.0,
):
    """
    Construct a simple 2-D Fisher Information Matrix for one
    bistatic AP-pair observing one target.

    The measurement is based on the bistatic range:

        r = ||p - tx|| + ||p - rx||

    The gradient with respect to target position is used to
    construct the FIM:

        J = (1 / sigma^2) * grad(r) grad(r)^T
    """
    tx_position = np.asarray(tx_position, dtype=float)
    rx_position = np.asarray(rx_position, dtype=float)
    target_position = np.asarray(target_position, dtype=float)

    d_tx = target_position - tx_position
    d_rx = target_position - rx_position

    norm_tx = max(np.linalg.norm(d_tx), 1e-9)
    norm_rx = max(np.linalg.norm(d_rx), 1e-9)

    gradient = (
        d_tx / norm_tx
        + d_rx / norm_rx
    )

    variance = max(float(measurement_noise), 1e-12)

    return np.outer(gradient, gradient) / variance


def aggregate_fisher_information(
    ap_positions,
    target_positions,
    y_tx,
    y_rx,
    measurement_noise=1.0,
):
    """
    Aggregate bistatic Fisher information for every target.

    Parameters
    ----------
    ap_positions : ndarray, (M, 2)
    target_positions : ndarray, (Q, 2)
    y_tx : ndarray, (M, Q)
        TX sensing assignment.
    y_rx : ndarray, (M, Q)
        RX sensing assignment.

    Returns
    -------
    J_total : ndarray, (Q, 2, 2)
    """
    ap_positions = np.asarray(ap_positions, dtype=float)
    target_positions = np.asarray(target_positions, dtype=float)
    y_tx = np.asarray(y_tx, dtype=float)
    y_rx = np.asarray(y_rx, dtype=float)

    M = len(ap_positions)
    Q = len(target_positions)

    J_total = np.zeros((Q, 2, 2), dtype=float)

    for q in range(Q):
        for m in range(M):
            if y_tx[m, q] <= 0:
                continue

            for n in range(M):
                if y_rx[n, q] <= 0:
                    continue

                J_mnq = fisher_information_from_pair(
                    ap_positions[m],
                    ap_positions[n],
                    target_positions[q],
                    measurement_noise=measurement_noise,
                )

                J_total[q] += J_mnq

    return J_total


def posterior_covariance(prior_covariance, fisher_information):
    """
    Compute:

        P+ = [(P-)^-1 + J]^-1
    """
    prior_covariance = np.asarray(prior_covariance, dtype=float)
    fisher_information = np.asarray(fisher_information, dtype=float)

    information_matrix = (
        np.linalg.inv(prior_covariance)
        + fisher_information
    )

    return np.linalg.inv(information_matrix)


def information_gain(prior_covariance, posterior_covariance):
    """
    Compute:

        I = log(det(P-) / det(P+))
    """
    prior_covariance = np.asarray(prior_covariance, dtype=float)
    posterior_covariance = np.asarray(
        posterior_covariance,
        dtype=float,
    )

    sign_prior, logdet_prior = np.linalg.slogdet(
        prior_covariance
    )
    sign_post, logdet_post = np.linalg.slogdet(
        posterior_covariance
    )

    if sign_prior <= 0 or sign_post <= 0:
        raise ValueError(
            "Covariance matrices must be positive definite."
        )

    return logdet_prior - logdet_post


def tracking_error(posterior_covariance):
    """
    Trace of posterior covariance:

        tr(P+)
    """
    return float(np.trace(posterior_covariance))
