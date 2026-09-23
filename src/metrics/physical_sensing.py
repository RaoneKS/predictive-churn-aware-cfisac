"""
Phase 5A — physical communication <-> Fisher-information sensing bridge.

Opt-in bridge between the validated Phase 4A/4B physical communication
layer and the existing scalar Fisher-information sensing model.

The existing scalar sensing path and Phase 4A/4B beamforming code are
left untouched.

Model used here:
    S_m = (p_sens,m / N) I_N

and for an active bistatic (m,n,q) pair:
    SNR_mnq = p_sens,m * G_sens[m,n,q] / noise_power_sens
    variance_mnq = reference_variance / SNR_mnq

Zero sensing power means no measurement and therefore zero FIM
contribution.
"""

import numpy as np

from src.metrics.sensing import (
    fisher_information_from_pair,
    posterior_covariance,
    information_gain,
    tracking_error,
)


def comm_power_per_ap(W):
    """Per-AP communication power from W with shape (M,K,N)."""
    W = np.asarray(W, dtype=np.complex128)
    if W.ndim != 3:
        raise ValueError("W must have shape (M, K, N).")
    return np.sum(np.abs(W) ** 2, axis=(1, 2))


def isotropic_sensing_covariance(sensing_power_per_ap, num_antennas):
    """Build S_m = (p_sens,m / N) I_N."""
    p = np.maximum(np.asarray(sensing_power_per_ap, dtype=float), 0.0)
    M = len(p)
    N = int(num_antennas)
    if N < 1:
        raise ValueError("num_antennas must be >= 1.")

    S = np.zeros((M, N, N), dtype=float)
    eye_over_n = np.eye(N) / N

    for m in range(M):
        S[m] = p[m] * eye_over_n

    return S


def sensing_power_from_covariance(S):
    """Return tr(S_m) for each AP."""
    S = np.asarray(S, dtype=float)
    return np.trace(S, axis1=-2, axis2=-1)


def sensing_snr(
    sensing_power_per_ap,
    G_sens,
    y_tx,
    y_rx,
    noise_power_sens=1e-9,
):
    """
    Compute active bistatic sensing SNRs.

    Shape:
        G_sens -> (M,M,Q)
        y_tx   -> (M,Q)
        y_rx   -> (M,Q)
        output -> (M,M,Q)
    """
    p = np.maximum(np.asarray(sensing_power_per_ap, dtype=float), 0.0)
    G_sens = np.asarray(G_sens, dtype=float)
    y_tx = np.asarray(y_tx, dtype=float)
    y_rx = np.asarray(y_rx, dtype=float)

    M, M_rx, Q = G_sens.shape

    if M != M_rx:
        raise ValueError("G_sens must have shape (M, M, Q).")

    if y_tx.shape != (M, Q) or y_rx.shape != (M, Q):
        raise ValueError(
            "y_tx and y_rx must have shape (M, Q) matching G_sens."
        )

    if len(p) != M:
        raise ValueError(
            "sensing_power_per_ap must have one value per AP."
        )

    noise_power_sens = max(float(noise_power_sens), 1e-30)

    snr = np.zeros((M, M, Q), dtype=float)

    for q in range(Q):
        tx_active = np.where(y_tx[:, q] > 0)[0]
        rx_active = np.where(y_rx[:, q] > 0)[0]

        if len(tx_active) == 0 or len(rx_active) == 0:
            continue

        for m in tx_active:
            if p[m] <= 0.0:
                continue

            for n in rx_active:
                snr[m, n, q] = (
                    p[m] * G_sens[m, n, q] / noise_power_sens
                )

    return snr


def snr_to_measurement_variance(snr, reference_variance=1.0):
    """
    Convert SNR to effective measurement variance.

        variance = reference_variance / SNR   for SNR > 0
        variance = inf                         for SNR == 0
    """
    snr = np.asarray(snr, dtype=float)

    if reference_variance <= 0:
        raise ValueError(
            "reference_variance must be strictly positive."
        )

    variance = np.full(snr.shape, np.inf, dtype=float)

    active = snr > 0
    variance[active] = (
        float(reference_variance) / snr[active]
    )

    return variance


def aggregate_fisher_information_bridge(
    ap_positions,
    target_positions,
    y_tx,
    y_rx,
    sensing_power_per_ap,
    G_sens,
    noise_power_sens=1e-9,
    reference_variance=1.0,
):
    """
    Physical-power-aware Fisher-information bridge.

    Reuses the existing fisher_information_from_pair() implementation.
    """
    ap_positions = np.asarray(ap_positions, dtype=float)
    target_positions = np.asarray(target_positions, dtype=float)
    y_tx = np.asarray(y_tx, dtype=float)
    y_rx = np.asarray(y_rx, dtype=float)

    M = len(ap_positions)
    Q = len(target_positions)

    snr = sensing_snr(
        sensing_power_per_ap,
        G_sens,
        y_tx,
        y_rx,
        noise_power_sens=noise_power_sens,
    )

    variance = snr_to_measurement_variance(
        snr,
        reference_variance=reference_variance,
    )

    J_total = np.zeros((Q, 2, 2), dtype=float)

    for q in range(Q):
        for m in range(M):
            if y_tx[m, q] <= 0:
                continue

            for n in range(M):
                if y_rx[n, q] <= 0:
                    continue

                var_mnq = variance[m, n, q]

                if not np.isfinite(var_mnq):
                    continue

                J_total[q] += fisher_information_from_pair(
                    ap_positions[m],
                    ap_positions[n],
                    target_positions[q],
                    measurement_noise=var_mnq,
                )

    return J_total


def physical_tx_power_per_ap(W, sensing_power_per_ap):
    """
    Combined AP transmit power:

        P_tx,m = sum_k ||w_mk||^2 + p_sens,m
    """
    p_comm = comm_power_per_ap(W)

    p_sens = np.maximum(
        np.asarray(sensing_power_per_ap, dtype=float),
        0.0,
    )

    if len(p_sens) != len(p_comm):
        raise ValueError(
            "sensing_power_per_ap must have one value per AP, "
            "matching W's M dimension."
        )

    return p_comm + p_sens


def check_power_constraint(P_tx, P_max, tol=1e-9):
    """Check P_tx,m <= Pmax,m + tol."""
    P_tx = np.asarray(P_tx, dtype=float)
    P_max = np.broadcast_to(
        np.asarray(P_max, dtype=float),
        P_tx.shape,
    )

    violation = np.maximum(P_tx - P_max, 0.0)
    feasible = violation <= tol

    return feasible, violation


def evaluate_physical_sensing_bridge(
    ap_positions,
    target_positions,
    y_tx,
    y_rx,
    W,
    sensing_power_per_ap,
    G_sens,
    tx_power_per_ap_max,
    prior_covariances=None,
    noise_power_sens=1e-9,
    reference_variance=1.0,
):
    """Run the complete Phase 5A bridge evaluation."""
    target_positions = np.asarray(
        target_positions,
        dtype=float,
    )

    Q = len(target_positions)

    if prior_covariances is None:
        prior_covariances = [
            np.eye(2) for _ in range(Q)
        ]

    J_total = aggregate_fisher_information_bridge(
        ap_positions,
        target_positions,
        y_tx,
        y_rx,
        sensing_power_per_ap,
        G_sens,
        noise_power_sens=noise_power_sens,
        reference_variance=reference_variance,
    )

    post_covs = []
    info_gains = np.zeros(Q, dtype=float)
    track_err = np.zeros(Q, dtype=float)

    for q in range(Q):
        post_cov = posterior_covariance(
            prior_covariances[q],
            J_total[q],
        )

        post_covs.append(post_cov)

        info_gains[q] = information_gain(
            prior_covariances[q],
            post_cov,
        )

        track_err[q] = tracking_error(post_cov)

    P_comm = comm_power_per_ap(W)

    P_sens = np.maximum(
        np.asarray(sensing_power_per_ap, dtype=float),
        0.0,
    )

    P_tx = P_comm + P_sens

    feasible, violation = check_power_constraint(
        P_tx,
        tx_power_per_ap_max,
    )

    return {
        "J_total": J_total,
        "posterior_covariances": post_covs,
        "information_gain": info_gains,
        "tracking_error": track_err,
        "P_comm": P_comm,
        "P_sens": P_sens,
        "P_tx": P_tx,
        "power_feasible": feasible,
        "power_violation": violation,
    }


__all__ = [
    "comm_power_per_ap",
    "isotropic_sensing_covariance",
    "sensing_power_from_covariance",
    "sensing_snr",
    "snr_to_measurement_variance",
    "aggregate_fisher_information_bridge",
    "physical_tx_power_per_ap",
    "check_power_constraint",
    "evaluate_physical_sensing_bridge",
]
