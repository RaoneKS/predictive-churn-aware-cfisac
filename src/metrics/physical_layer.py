"""
src/metrics/physical_layer.py

Phase 4A — physical-layer (beamformed, complex-channel) SINR and rate.

This module ADDS a new SINR/rate computation for the per-antenna complex
channel + beamforming path (src.channels.physical + src.beamforming).
It does NOT modify or replace the existing scalar SINR/rate model in
src.metrics.system (communication_sinr / communication_rate), which
remains fully intact and is in fact reused here for the rate formula so
both paths share the exact same Shannon-rate convention.
"""

import numpy as np

from src.metrics.system import communication_rate


def physical_sinr(H, W, noise_power=1e-9):
    """
    Downlink SINR per user for a coherent joint-transmission cell-free
    system with per-AP local beamforming weights.

    cross_gain[k, l] = sum_m h_{m,k}^H w_{m,l}
                     = sum_{m,n} conj(H[m,k,n]) * W[m,l,n]

    signal_power[k]       = |cross_gain[k, k]|^2
    interference_power[k] = sum_{l != k} |cross_gain[k, l]|^2
    SINR[k]               = signal_power[k] / (interference_power[k] + noise_power)

    Because W[m, l, :] is zero for any AP m that does not serve user l
    (see src.beamforming.precoders), the sum over m above automatically
    restricts to the APs actually serving each user -- no explicit
    association matrix is needed at this stage.

    Parameters
    ----------
    H : ndarray, shape (M, K, N), complex
    W : ndarray, shape (M, K, N), complex
        Beamforming weights, already power-normalized
        (see src.beamforming.precoders.normalize_per_ap_power).
    noise_power : float
        Receiver noise power (linear units), must be > 0.

    Returns
    -------
    sinr : ndarray, shape (K,), float64
        Always finite and >= 0, including the degenerate all-zero
        association / all-zero beamforming case (SINR == 0 for every user).
    """
    H = np.asarray(H, dtype=np.complex128)
    W = np.asarray(W, dtype=np.complex128)

    if H.shape != W.shape:
        raise ValueError("H and W must have the same shape (M, K, N).")

    if noise_power <= 0:
        raise ValueError("noise_power must be strictly positive.")

    # cross_gain[k, l] = sum_m,n conj(H[m,k,n]) W[m,l,n]
    cross_gain = np.einsum("mkn,mln->kl", np.conj(H), W)

    power = np.abs(cross_gain) ** 2  # (K, K), power[k, l]

    signal_power = np.diag(power).copy()
    interference_power = np.sum(power, axis=1) - signal_power
    interference_power = np.maximum(interference_power, 0.0)

    sinr = signal_power / (interference_power + noise_power)

    return sinr


def physical_rate(sinr, bandwidth_hz=20e6, pilot_fraction=0.1):
    """
    Physical-layer Shannon-rate approximation.

    Thin wrapper around the EXISTING src.metrics.system.communication_rate
    so both the scalar path and the physical-layer path share one rate
    formula:

        R = (1 - pilot_fraction) * bandwidth_hz * log2(1 + SINR)

    Parameters
    ----------
    sinr : ndarray, shape (K,)
    bandwidth_hz : float
    pilot_fraction : float

    Returns
    -------
    ndarray, shape (K,), float64, always finite and >= 0.
    """
    return communication_rate(
        sinr,
        bandwidth_hz=bandwidth_hz,
        pilot_fraction=pilot_fraction,
    )


__all__ = [
    "physical_sinr",
    "physical_rate",
]
