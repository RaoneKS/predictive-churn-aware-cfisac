"""
src/beamforming/precoders.py

Phase 4A — MRT (matched-filter) and RZF beamforming for the per-AP,
per-antenna physical-layer channel produced by src.channels.physical.

Signal-model convention
------------------------
Following the convention used in vendor/cordis/cordis/algorithms/beamforming.py
(adapted here for a per-AP, association-matrix-driven cell-free setting):

    y_k = sum_m h_{m,k}^H w_{m,k} s_k   (desired signal, AP m serving user k)
        + sum_{l != k} sum_m h_{m,k}^H w_{m,l} s_l   (interference)
        + n_k

A precoding vector w_{m,k} is only nonzero if AP m serves user k, i.e. the
association matrix entry x[m, k] == 1 (see src.clustering.overlapping).
Beamforming is computed LOCALLY per AP: AP m only uses the channels of the
users IT serves, exactly as in the vendor reference's "local" precoders
(mrt / rzf in cordis/algorithms/beamforming.py). This matches a
distributed cell-free implementation where each AP has only local CSI to
its own associated users.

MRT (matched filter)
    w_{m,k} = h_{m,k} / ||h_{m,k}||          (per (m,k), unit-norm)

    Gives h_{m,k}^H w_{m,k} = ||h_{m,k}|| (real, positive) -- coherent,
    matches the vendor convention exactly (their comment: "Signal:
    h_u^H h_u = ||h_u||^2, real, positive").

RZF (regularized zero-forcing), computed per AP across the set of users
that AP serves (user-domain / K x K form, following cordis.rzf):

    Let K_m = {k : x[m,k] == 1}, H_m = rows [h_{m,k}]_{k in K_m}  (|K_m| x N)

        Gram = H_m.conj() @ H_m.T + eps * I         (|K_m| x |K_m|)
        W_m  = H_m.T @ Gram^{-1}                     (N x |K_m|)

    eps is channel-power-relative (see `_relative_regularization`),
    matching cordis._eps_scaled, so it is meaningful regardless of the
    absolute pathloss magnitude.

Both precoders are then subject to per-AP power normalization
(`normalize_per_ap_power`), which is the only place total transmit power
is fixed -- MRT/RZF above return *directionally correct but arbitrarily
scaled* weights.
"""

import numpy as np


def _relative_regularization(gram, eps_rel):
    """
    Channel-power-relative regularization constant, matching
    vendor/cordis/cordis/algorithms/beamforming.py::_eps_scaled.

        eps_abs = eps_rel * trace(gram) / n
    """
    n = gram.shape[0]
    tr = float(np.real(np.trace(gram)))
    return eps_rel * max(tr, 1e-30) / n


def mrt_weights(H, association):
    """
    Per-AP matched-filter (MRT) precoding weights.

    Parameters
    ----------
    H : ndarray, shape (M, K, N), complex
        Physical-layer channel (see src.channels.physical.generate_physical_channel).
    association : ndarray, shape (M, K)
        Binary AP-user association (x[m,k] == 1 iff AP m serves user k).

    Returns
    -------
    W : ndarray, shape (M, K, N), complex128
        W[m, k, :] == 0 for every (m, k) with association[m, k] == 0.
        For served pairs, ||W[m, k, :]|| == 1 (unit-norm matched filter,
        prior to power normalization).
    """
    H = np.asarray(H, dtype=np.complex128)
    association = np.asarray(association)

    M, K, N = H.shape
    if association.shape != (M, K):
        raise ValueError("association must have shape (M, K) matching H.")

    W = np.zeros((M, K, N), dtype=np.complex128)

    served = association.astype(bool)
    norms = np.linalg.norm(H, axis=-1)  # (M, K)

    safe_norms = np.where(norms > 1e-15, norms, 1.0)
    unit_H = H / safe_norms[:, :, None]

    W[served] = unit_H[served]

    return W


def rzf_weights(H, association, epsilon_rel=1e-2):
    """
    Per-AP regularized zero-forcing (RZF) precoding weights.

    Each AP independently designs a zero-forcing-style precoder using
    ONLY the channels of the users it serves (local CSI), following
    vendor/cordis/cordis/algorithms/beamforming.py::rzf (user-domain form).

    Parameters
    ----------
    H : ndarray, shape (M, K, N), complex
    association : ndarray, shape (M, K)
    epsilon_rel : float
        Relative regularization strength (see `_relative_regularization`).

    Returns
    -------
    W : ndarray, shape (M, K, N), complex128
        W[m, k, :] == 0 for every (m, k) with association[m, k] == 0 or
        for any AP m that serves no users.
    """
    H = np.asarray(H, dtype=np.complex128)
    association = np.asarray(association)

    M, K, N = H.shape
    if association.shape != (M, K):
        raise ValueError("association must have shape (M, K) matching H.")

    W = np.zeros((M, K, N), dtype=np.complex128)

    for m in range(M):
        served_users = np.where(association[m] > 0)[0]
        n_served = len(served_users)

        if n_served == 0:
            continue

        H_m = H[m, served_users, :]  # (n_served, N), rows = h_k

        gram = H_m.conj() @ H_m.T  # (n_served, n_served)
        eps = _relative_regularization(gram, epsilon_rel)
        gram = gram + eps * np.eye(n_served)

        # W_m = H_m.T @ inv(gram)   -> (N, n_served), columns = w_k
        W_m = H_m.T @ np.linalg.solve(gram, np.eye(n_served))

        W[m, served_users, :] = W_m.T

    return W


def normalize_per_ap_power(W, tx_power_per_ap):
    """
    Scale precoding weights so that every AP's total transmit power
    (summed over the users it serves, assuming unit-power data symbols)
    equals `tx_power_per_ap` exactly, or 0 if the AP serves no one.

    Parameters
    ----------
    W : ndarray, shape (M, K, N), complex
    tx_power_per_ap : float or ndarray, shape (M,)
        Per-AP power budget (linear units, e.g. Watts).

    Returns
    -------
    W_scaled : ndarray, shape (M, K, N), complex128
    ap_power_used : ndarray, shape (M,)
        Actual total transmit power used by each AP after scaling
        (== tx_power_per_ap[m] for any AP serving >=1 user, else 0.0).
    """
    W = np.asarray(W, dtype=np.complex128)
    M, K, N = W.shape

    tx_power_per_ap = np.broadcast_to(
        np.asarray(tx_power_per_ap, dtype=float), (M,)
    ).copy()

    current_power = np.sum(np.abs(W) ** 2, axis=(1, 2))  # (M,)

    scale = np.zeros(M, dtype=float)
    active = current_power > 1e-30
    scale[active] = np.sqrt(tx_power_per_ap[active] / current_power[active])

    W_scaled = W * scale[:, None, None]

    ap_power_used = np.sum(np.abs(W_scaled) ** 2, axis=(1, 2))

    return W_scaled, ap_power_used


__all__ = [
    "mrt_weights",
    "rzf_weights",
    "normalize_per_ap_power",
]
