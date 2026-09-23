"""
src/channels/physical.py

Phase 4A — physical-layer (per-antenna, complex-valued) communication channel.

This module ADDS a new physical-layer channel representation on top of the
existing scalar large-scale-gain model in `src/channels/communication.py`.
It does NOT replace or modify that model: `generate_comm_gains` /
`generate_communication_features` are reused here (for the large-scale
power/pathloss magnitude) and remain fully intact and importable for any
code that still wants the old scalar SINR/rate path.

Model
-----
Each AP has `num_antennas` co-located antenna elements arranged as a
uniform linear array (ULA). The per-antenna, narrowband, complex downlink
channel from AP m to user k is modeled as Rician:

    h_{m,k} = sqrt(G_comm[m,k]) *
              ( sqrt(K/(K+1)) * a(theta_{m,k})
                + sqrt(1/(K+1)) * g_{m,k} )

where
    G_comm[m,k]   : existing scalar large-scale gain (pathloss), reused
                    unmodified from src.channels.communication.
    a(theta)      : ULA steering vector, unit average element power
                    (E[|a_n|^2] = 1 for every element).
    g_{m,k}       : i.i.d. CN(0, 1) per-antenna small-scale (NLOS) fading.
    K             : Rician K-factor (linear, not dB). K -> inf reduces to
                    pure LOS/steering; K = 0 reduces to pure Rayleigh.

With this normalization, E[|h_{m,k,n}|^2] = G_comm[m,k] for every antenna
element n, so the *average* per-antenna power matches the existing scalar
gain exactly; only the addition of antennas/phase/beamforming gain is new.

Steering-vector convention
---------------------------
theta_{m,k} is the angle (radians) of user k as seen from AP m, measured
relative to the AP's boresight (broadside) direction. Antenna spacing is
expressed as a fraction of the wavelength (`antenna_spacing`, default 0.5
i.e. half-wavelength spacing), so no absolute carrier wavelength is
required:

    a_n(theta) = exp(j * 2*pi * antenna_spacing * n * sin(theta)),  n = 0..N-1

This vector satisfies ||a(theta)||_2 = sqrt(N) for every theta (unit
average element power / "steering normalization").
"""

import numpy as np

from src.channels.communication import distance_2d, generate_comm_gains


def aoa_2d(ap_positions, user_positions, ap_orientations=None):
    """
    Angle of arrival/departure of each user as seen from each AP,
    relative to the AP's boresight direction.

    Parameters
    ----------
    ap_positions : ndarray, shape (M, 2)
    user_positions : ndarray, shape (K, 2)
    ap_orientations : ndarray or None, shape (M,)
        Boresight direction of each AP's array, in radians, measured the
        same way as `np.arctan2`. Defaults to 0.0 for every AP (arrays
        all facing the same reference direction), which is sufficient for
        the minimum Phase-4A physical-layer extension.

    Returns
    -------
    theta : ndarray, shape (M, K)
        Angle (radians) of user k relative to AP m's boresight.
    """
    ap_positions = np.asarray(ap_positions, dtype=float)
    user_positions = np.asarray(user_positions, dtype=float)

    M = len(ap_positions)

    if ap_orientations is None:
        ap_orientations = np.zeros(M, dtype=float)
    else:
        ap_orientations = np.asarray(ap_orientations, dtype=float)
        if len(ap_orientations) != M:
            raise ValueError("ap_orientations must have one value per AP.")

    delta = user_positions[None, :, :] - ap_positions[:, None, :]
    raw_angle = np.arctan2(delta[..., 1], delta[..., 0])

    theta = raw_angle - ap_orientations[:, None]

    # wrap to (-pi, pi]
    theta = np.mod(theta + np.pi, 2 * np.pi) - np.pi

    return theta


def steering_vector(theta, num_antennas, antenna_spacing=0.5):
    """
    Single ULA steering vector for one angle.

    Parameters
    ----------
    theta : float, radians
    num_antennas : int
    antenna_spacing : float
        Element spacing as a fraction of the wavelength (0.5 = half-wave).

    Returns
    -------
    ndarray, shape (num_antennas,), complex128, ||a|| == sqrt(num_antennas)
    """
    n = np.arange(num_antennas, dtype=float)
    phase = 2.0 * np.pi * antenna_spacing * n * np.sin(theta)
    return np.exp(1j * phase)


def steering_matrix(theta, num_antennas, antenna_spacing=0.5):
    """
    Vectorized ULA steering vectors for an arbitrary-shaped array of angles.

    Parameters
    ----------
    theta : ndarray, any shape (...,)
    num_antennas : int
    antenna_spacing : float

    Returns
    -------
    ndarray, shape (..., num_antennas), complex128
        a[..., n] satisfies ||a[...]||_2 == sqrt(num_antennas) for every
        leading index (steering-vector normalization).
    """
    theta = np.asarray(theta, dtype=float)
    n = np.arange(num_antennas, dtype=float)

    # Broadcast: sin(theta) has shape (...,); n has shape (N,).
    # Result shape: (..., N)
    phase = 2.0 * np.pi * antenna_spacing * n[None, ...] * np.sin(theta)[..., None]

    return np.exp(1j * phase)


def generate_physical_channel(
    ap_positions,
    user_positions,
    num_antennas,
    pathloss_exponent=2.0,
    antenna_spacing=0.5,
    rician_k_factor=10.0,
    small_scale_fading=True,
    ap_orientations=None,
    seed=None,
):
    """
    Generate the complex, per-antenna physical-layer downlink channel.

    Reuses the existing scalar large-scale gain (`generate_comm_gains`)
    unmodified for the magnitude, and adds a ULA steering vector plus
    optional Rician small-scale fading for phase/spatial structure.

    Parameters
    ----------
    ap_positions : ndarray, shape (M, 2)
    user_positions : ndarray, shape (K, 2)
    num_antennas : int
        Number of antenna elements per AP (N).
    pathloss_exponent : float
        Forwarded to the existing scalar gain model.
    antenna_spacing : float
        ULA element spacing as a fraction of wavelength.
    rician_k_factor : float
        Linear Rician K-factor. Ignored if small_scale_fading is False
        (pure LOS/steering channel is used instead).
    small_scale_fading : bool
        If False, the channel is purely deterministic given the
        topology (LOS steering only) -- no RNG is used at all, which is
        the mode exercised by the deterministic-generation test.
    ap_orientations : ndarray or None, shape (M,)
    seed : int or None
        Seed for the small-scale fading RNG. Same seed -> identical
        channel realization. Ignored if small_scale_fading is False.

    Returns
    -------
    H : ndarray, shape (M, K, N), complex128
        H[m, k, :] is the channel vector from AP m's antenna array to
        user k. E[|H[m,k,n]|^2] == G_comm[m,k] for every antenna n.
    """
    ap_positions = np.asarray(ap_positions, dtype=float)
    user_positions = np.asarray(user_positions, dtype=float)

    if num_antennas < 1:
        raise ValueError("num_antennas must be >= 1.")

    M = len(ap_positions)
    K = len(user_positions)
    N = int(num_antennas)

    # Large-scale gain: EXACT reuse of the existing scalar model.
    G_comm = generate_comm_gains(
        ap_positions,
        user_positions,
        pathloss_exponent=pathloss_exponent,
    )

    theta = aoa_2d(ap_positions, user_positions, ap_orientations=ap_orientations)
    a = steering_matrix(theta, N, antenna_spacing=antenna_spacing)  # (M, K, N)

    sqrt_G = np.sqrt(G_comm)[:, :, None]  # (M, K, 1)

    if not small_scale_fading:
        H = sqrt_G * a
        return H.astype(np.complex128)

    k_lin = max(float(rician_k_factor), 0.0)
    los_weight = np.sqrt(k_lin / (k_lin + 1.0))
    nlos_weight = np.sqrt(1.0 / (k_lin + 1.0))

    rng = np.random.default_rng(seed)
    nlos_real = rng.standard_normal(size=(M, K, N))
    nlos_imag = rng.standard_normal(size=(M, K, N))
    g = (nlos_real + 1j * nlos_imag) / np.sqrt(2.0)  # CN(0, 1) per element

    H = sqrt_G * (los_weight * a + nlos_weight * g)

    return H.astype(np.complex128)


# Backward-compatibility re-export note:
# The scalar path (distance_2d, pathloss_gain, generate_comm_gains,
# spatial_correlation, generate_communication_features) remains defined in
# src/channels/communication.py and is unchanged by this module.
__all__ = [
    "aoa_2d",
    "steering_vector",
    "steering_matrix",
    "generate_physical_channel",
]
