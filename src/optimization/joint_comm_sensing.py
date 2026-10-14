"""
src/optimization/joint_comm_sensing.py

Phase 5B -- joint communication + sensing per-AP power allocation.

This module is ADDITIVE. It reuses, read-only and unmodified:

    * Phase 4A  src.beamforming.precoders      (MRT / RZF directions)
    * Phase 4A  src.metrics.physical_layer     (physical_sinr / physical_rate)
    * Phase 5A  src.metrics.physical_sensing   (power-aware Fisher bridge)
    * scalar    src.metrics.sensing            (FIM, posterior cov., info. gain)

Nothing in Phases 1-5A is modified. Phase 4B (constrained beamforming) is not
called here; its output can be used as the direction source (see `W_dir`).

===========================================================================
1. Problem formulation
===========================================================================

Notation.  M APs, K users, Q targets, N antennas per AP.
    H      (M,K,N) physical channel                     (Phase 4A)
    W_dir  (M,K,N) fixed communication beam DIRECTIONS with unit power per
                    active AP:  sum_k ||W_dir[m,k,:]||^2 = 1 if AP m serves
                    anybody, else 0.  (MRT or RZF, normalized to 1 W.)
    y_tx, y_rx (M,Q) sensing TX / RX roles                (existing clustering)
    P_max  (M,)     per-AP transmit power budget

Decision variables (per AP m):
    c_m = P_comm,m  >= 0 communication power; the beamformer is
                           W[m,k,:] = sqrt(c_m) * W_dir[m,k,:], hence
                           sum_k ||W[m,k,:]||^2 == c_m exactly.
    s_m = P_sens,m  >= 0   sensing power (isotropic covariance
                           S_m = (s_m/N) I_N, Phase 5A; tr(S_m) = s_m).

Utilities (project convention, see churn_aware.evaluate_cluster_performance):
    U_comm(c) = (1/ref_rate) * sum_k R_k(c),
                R_k = physical_rate(physical_sinr(H, W(c), noise))_k
    U_sens(s) = (1/ref_ig)   * sum_q IG_q(s),
                IG_q = log det(P_q^-) - log det(P_q^+)   (Phase 5A bridge),
                P_q^+ = ( (P_q^-)^-1 + J_q(s) )^-1

Optimization problem (P):

        maximize      alpha * U_comm(c) + beta * U_sens(s)
        subject to    c_m + s_m <= P_max,m      for every AP m
                      c_m >= 0, s_m >= 0

The single coupling between the two utilities is the per-AP budget, so
transmit power is counted exactly once: P_tx,m = P_comm,m + P_sens,m.

===========================================================================
2. Structure exploited (all verified numerically by the tests)
===========================================================================

(S1) The Phase 5A bridge maps sensing power to SNR linearly
        SNR_mnq = s_m G_sens[m,n,q] / noise_sens,
        variance_mnq = ref_var / SNR_mnq
     and the pair FIM is  g g^T / variance  (scalar model, unchanged), so
        J_q(s) = sum_m s_m A_mq,     A_mq = J_q at s = e_m   (PSD 2x2).
     The A_mq are obtained by calling the EXISTING bridge with unit power on
     one AP at a time (no re-implementation of the FIM). This linearity
     holds exactly as long as the scalar FIM's 1e-12 variance floor is inactive;
     `_JointProblem` verifies this at s = P_max (variance is decreasing in s,
     so this covers every feasible s) and RAISES otherwise.

(S2) With P_q^+ = (P_q^-^-1 + J_q)^-1,
        IG_q(s) = log det( I + P_q^- J_q(s) ),
     which is concave and nondecreasing in s (J_q is affine in s, log det is
     concave and increasing on the PSD cone). Gradient:
        dIG_q/ds_m = tr( P_q^+ A_mq ).

(S3) Reduction.  U_comm does not depend on s and U_sens is nondecreasing in
     each s_m, so for beta > 0 there is an optimum with
        s_m = P_max,m - c_m   for every AP m that can help sensing
                              ("sensing-useful": A_mq != 0 for some q),
        s_m = 0               otherwise (power spent where it produces no
                              measurement is pure waste).
     Problem (P) then becomes a BOX-constrained problem in c only:
        maximize_c   alpha * U_comm(c) + beta * U_sens(P_max - c)
        subject to   0 <= c_m <= P_max,m   (c_m = 0 if AP m serves nobody).
     Boundaries (explicit tie-breaking rules):
        beta  = 0  ->  s = 0        (communication-only)
        alpha = 0  ->  c = 0, s = P_max on sensing-useful APs
                                    (sensing-only; U_sens is then maximal
                                     over the whole feasible set, exactly)

(S3b) Power-control SCOPE (audit finding, drives the default).
     With c free on EVERY communication-active AP, maximizing the project's
     sum-rate utility is dominated by comm-only interference management: on
     the Phase 5A validation topology the comm-only optimum silences two APs
     and drives one user to ~0.2 Mbps (SINR -21 dB) while raising the sum
     rate from 380.6 to 447.1 Mbps. That gain has nothing to do with sensing
     and would make "joint" look better than "communication-only" for the
     wrong reason. The default scope therefore is
        comm_power_control = "dual_role":
           c_m is a decision variable ONLY on APs that hold both roles
           (communication-active AND sensing-useful); every other
           communication-active AP stays at its Phase 4A baseline
           c_m = P_max,m (`normalize_per_ap_power(W, P_max)` convention).
        comm_power_control = "all":
           the literal problem (P): c_m free on every comm-active AP
           (sum-rate power control; can starve users -- use with care and
           inspect min_user_rate_bps).
     The scope is identical for the beta = 0 reference and the joint
     solve, so differences between them are attributable to sensing.

(S4) U_comm is NOT concave in c (multi-user interference). It is smooth, so
     (P) is solved with a deterministic multi-start projected quasi-Newton
     method in log-power variables z_m = ln(c_m / P_max,m):
        - communication is interference-limited over many decades of power
          (SINR is invariant to a uniform scaling of c until the noise floor
          is reached), so log-power is the natural scale;
        - z_m in [ln(rho_floor), 0], rho_floor = 1e-12;
        - candidates = a fixed semi-log grid of UNIFORM splits
          c_m = rho * P_max,m on the decision APs plus the L-BFGS-B polish
          of the best `n_polish` grid points; the best candidate is returned.
     Consequently the returned objective is >= the objective of EVERY
     uniform split on the grid (including rho = 1, i.e. the baseline full-power
     communication with no sensing). Global optimality is NOT claimed.

===========================================================================
3. Documented approximations (read before interpreting results)
===========================================================================

A1  No sensing->communication interference. Inherited from Phase 5A: the
    sensing signal is assumed not to enter the users' SINR (e.g. orthogonal
    resource or perfectly cancelled). The utilities are therefore coupled
    ONLY through the per-AP power budget and the beam gain. This is
    the main modelling limitation and it makes the trade-off milder than a
    physically interfering isotropic sensing signal would.
A2  Beam DIRECTIONS are fixed (Phase 4A MRT/RZF or any unit-power directions
    supplied by the caller, e.g. rescaled Phase 4B beamformers). Only a
    nonnegative per-AP amplitude is optimized; directions are not re-designed
    for each allocation and negative amplitudes are excluded.
A3  Isotropic sensing covariance S_m = (s_m/N) I (Phase 5A). The FIM depends
    on s only through tr(S_m).
A4  Communication utility is the sum-rate (project convention). No per-user
    QoS constraint is imposed; `min_rate_bps` only yields a diagnostic.
A5  Only sensing TX APs spend sensing power (Phase 5A model); receive-only
    APs have no sensing transmit budget.
A6  Single snapshot: no mobility, prediction, churn or two-timescale logic.
A7  Local optimization (see S4).

Everything is a pure function of its inputs; there is no RNG anywhere, so
repeated calls are bit-for-bit identical.
"""

import numpy as np
from scipy.optimize import minimize

from src.beamforming.precoders import (
    mrt_weights,
    normalize_per_ap_power,
    rzf_weights,
)
from src.metrics.physical_layer import physical_rate, physical_sinr
from src.metrics.physical_sensing import (
    aggregate_fisher_information_bridge,
    check_power_constraint,
    comm_power_per_ap,
    evaluate_physical_sensing_bridge,
    sensing_snr,
    snr_to_measurement_variance,
)
from src.metrics.sensing import information_gain, posterior_covariance
from src.metrics.system import qos_violations


_RHO_GRID = (
    1.0, 0.75, 0.5, 0.25, 0.1,
    3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 1e-6,
)
_RHO_FLOOR = 1e-12
_LINEARITY_RTOL = 1e-9


def comm_beam_directions(H, association, precoder="mrt", epsilon_rel=1e-2):
    """
    Unit-power-per-active-AP communication beam directions built from the
    UNMODIFIED Phase 4A precoders.

    Returns
    -------
    W_dir : ndarray (M, K, N), complex128
        sum_k ||W_dir[m,k,:]||^2 == 1 for APs serving >= 1 user, else 0.
    """
    if precoder == "mrt":
        W = mrt_weights(H, association)
    elif precoder == "rzf":
        W = rzf_weights(H, association, epsilon_rel=epsilon_rel)
    else:
        raise ValueError("precoder must be 'mrt' or 'rzf'.")
    W_dir, _ = normalize_per_ap_power(W, 1.0)
    return W_dir


def _validate_directions(W_dir):
    """Directions must have per-AP power exactly 0 or 1. Returns active mask."""
    W_dir = np.asarray(W_dir, dtype=np.complex128)
    if W_dir.ndim != 3:
        raise ValueError("W_dir must have shape (M, K, N).")
    p = np.sum(np.abs(W_dir) ** 2, axis=(1, 2))
    is_zero = p <= 1e-12
    is_unit = np.abs(p - 1.0) <= 1e-9
    if not np.all(is_zero | is_unit):
        raise ValueError(
            "W_dir must have per-AP power 0 (AP serves nobody) or 1 "
            "(unit-power directions); use normalize_per_ap_power(W, 1.0)."
        )
    return is_unit


def sensing_fim_basis(
    ap_positions,
    target_positions,
    y_tx,
    y_rx,
    G_sens,
    noise_power_sens=1e-9,
    reference_variance=1.0,
):
    """
    FIM basis A[m, q] (2x2): the aggregate Fisher information for target q
    when ONLY AP m transmits sensing power, at 1 W. Computed by calling the
    existing Phase 5A bridge (no FIM re-implementation).

    Returns
    -------
    A : ndarray (M, Q, 2, 2)
    useful : ndarray (M,), bool
    """
    ap_positions = np.asarray(ap_positions, dtype=float)
    target_positions = np.asarray(target_positions, dtype=float)
    y_tx = np.asarray(y_tx, dtype=float)
    M = len(ap_positions)
    Q = len(target_positions)

    A = np.zeros((M, Q, 2, 2), dtype=float)
    for m in range(M):
        if not np.any(y_tx[m] > 0):
            continue
        p = np.zeros(M, dtype=float)
        p[m] = 1.0
        A[m] = aggregate_fisher_information_bridge(
            ap_positions,
            target_positions,
            y_tx,
            y_rx,
            p,
            G_sens,
            noise_power_sens=noise_power_sens,
            reference_variance=reference_variance,
        )
    useful = np.any(np.abs(A) > 0.0, axis=(1, 2, 3))
    return A, useful


def uniform_split_allocation(
    W_dir,
    P_max,
    rho,
    sensing_useful,
    comm_power_control="dual_role",
):
    """
    Fixed uniform split used for baselines and solver start grid.
    """
    W_dir = np.asarray(W_dir, dtype=np.complex128)
    M = W_dir.shape[0]
    P_max = np.broadcast_to(
        np.asarray(P_max, dtype=float),
        (M,),
    ).copy()

    if not (0.0 <= rho <= 1.0):
        raise ValueError("rho must lie in [0, 1].")

    if comm_power_control not in ("dual_role", "all"):
        raise ValueError(
            "comm_power_control must be 'dual_role' or 'all'."
        )

    active = _validate_directions(W_dir)
    useful = np.asarray(sensing_useful, dtype=bool)

    if comm_power_control == "all":
        varying = active
    else:
        varying = active & useful

    c = np.where(active, P_max, 0.0)
    c = np.where(varying, rho * P_max, c)
    s = np.where(useful, P_max - c, 0.0)

    return c, s


def evaluate_joint_allocation(
    H,
    W_dir,
    P_comm_setpoint,
    sensing_power,
    ap_positions,
    target_positions,
    y_tx,
    y_rx,
    G_sens,
    P_max,
    alpha=1.0,
    beta=1.0,
    prior_covariances=None,
    noise_power=1e-9,
    noise_power_sens=1e-9,
    reference_variance=1.0,
    bandwidth_hz=20e6,
    pilot_fraction=0.1,
    ref_rate_bps=1e7,
    ref_info_gain=1.0,
    min_rate_bps=None,
    x=None,
    enforce_qos=False,
):
    """
    Evaluate an allocation (c, s) through the existing physical paths.
    """
    H = np.asarray(H, dtype=np.complex128)
    W_dir = np.asarray(W_dir, dtype=np.complex128)

    if H.shape != W_dir.shape:
        raise ValueError(
            "H and W_dir must have the same shape (M, K, N)."
        )

    M = H.shape[0]

    c = np.asarray(P_comm_setpoint, dtype=float)
    s = np.asarray(sensing_power, dtype=float)

    if c.shape != (M,) or s.shape != (M,):
        raise ValueError(
            "P_comm_setpoint and sensing_power must have shape (M,)."
        )

    if np.any(c < 0.0) or np.any(s < 0.0):
        raise ValueError(
            "Communication and sensing powers must be >= 0."
        )

    if alpha < 0.0 or beta < 0.0:
        raise ValueError("alpha and beta must be >= 0.")

    P_max = np.broadcast_to(
        np.asarray(P_max, dtype=float),
        (M,),
    ).copy()

    Q = len(np.asarray(target_positions))

    W = np.sqrt(c)[:, None, None] * W_dir

    sinr = physical_sinr(
        H,
        W,
        noise_power,
    )

    rates = physical_rate(
        sinr,
        bandwidth_hz=bandwidth_hz,
        pilot_fraction=pilot_fraction,
    )

    comm_raw = float(np.sum(rates))
    comm_norm = comm_raw / float(ref_rate_bps)

    bridge = evaluate_physical_sensing_bridge(
        ap_positions=ap_positions,
        target_positions=target_positions,
        y_tx=y_tx,
        y_rx=y_rx,
        W=W,
        sensing_power_per_ap=s,
        G_sens=G_sens,
        tx_power_per_ap_max=P_max,
        prior_covariances=prior_covariances,
        noise_power_sens=noise_power_sens,
        reference_variance=reference_variance,
    )

    ig = np.asarray(
        bridge["information_gain"],
        dtype=float,
    )
    te = np.asarray(
        bridge["tracking_error"],
        dtype=float,
    )

    sens_raw = float(np.sum(ig))
    sens_norm = sens_raw / float(ref_info_gain)

    P_comm = np.asarray(
        bridge["P_comm"],
        dtype=float,
    )
    P_sens = np.asarray(
        bridge["P_sens"],
        dtype=float,
    )
    P_tx = np.asarray(
        bridge["P_tx"],
        dtype=float,
    )

    power_ok, violation = check_power_constraint(
        P_tx,
        P_max,
    )

    objective = (
        float(alpha) * comm_norm
        + float(beta) * sens_norm
    )

    finite = bool(
        np.isfinite(objective)
        and np.all(np.isfinite(sinr))
        and np.all(np.isfinite(ig))
        and np.all(np.isfinite(te))
        and np.all(np.isfinite(P_tx))
    )

    feasible = bool(
        np.all(power_ok)
        and np.all(P_sens >= 0.0)
        and np.all(P_comm >= 0.0)
        and finite
    )

    out = {
        "W": W,
        "P_comm": P_comm,
        "rho_comm": np.divide(
            P_comm,
            P_max,
            out=np.zeros(M),
            where=P_max > 0,
        ),
        "P_sens": P_sens,
        "P_tx": P_tx,
        "P_max": P_max,
        "P_slack": P_max - P_tx,
        "power_feasible_per_ap": np.asarray(
            power_ok,
            dtype=bool,
        ),
        "power_violation": np.asarray(
            violation,
            dtype=float,
        ),
        "max_power_violation": float(
            np.max(violation)
        ),
        "feasible": feasible,
        "sinr": np.asarray(
            sinr,
            dtype=float,
        ),
        "rates_bps": np.asarray(
            rates,
            dtype=float,
        ),
        "comm_utility_raw": comm_raw,
        "comm_utility_norm": float(comm_norm),
        "min_user_rate_bps": (
            float(np.min(rates))
            if len(rates)
            else 0.0
        ),
        "information_gain": ig,
        "tracking_error": te,
        "mean_tracking_error": (
            float(np.mean(te))
            if Q
            else 0.0
        ),
        "sensing_utility_raw": sens_raw,
        "sensing_utility_norm": float(sens_norm),
        "J_total": np.asarray(
            bridge["J_total"],
            dtype=float,
        ),
        "objective": float(objective),
        "alpha": float(alpha),
        "beta": float(beta),
        "ref_rate_bps": float(ref_rate_bps),
        "ref_info_gain": float(ref_info_gain),
    }

    if min_rate_bps is not None:
        out["qos_violation_total_bps"] = float(
            np.sum(
                qos_violations(
                    rates,
                    float(min_rate_bps),
                )
            )
        )

    return out


class _JointProblem:
    """
    Internal evaluator for the reduced problem in log-power variables.
    """

    def __init__(
        self,
        H,
        W_dir,
        ap_positions,
        target_positions,
        y_tx,
        y_rx,
        G_sens,
        P_max,
        alpha,
        beta,
        prior_covariances,
        noise_power,
        noise_power_sens,
        reference_variance,
        bandwidth_hz,
        pilot_fraction,
        ref_rate_bps,
        ref_info_gain,
        comm_power_control="dual_role",
    ):
        if comm_power_control not in ("dual_role", "all"):
            raise ValueError(
                "comm_power_control must be 'dual_role' or 'all'."
            )

        self.scope = comm_power_control
        self.H = np.asarray(
            H,
            dtype=np.complex128,
        )
        self.W_dir = np.asarray(
            W_dir,
            dtype=np.complex128,
        )

        if self.H.shape != self.W_dir.shape:
            raise ValueError(
                "H and W_dir must have the same shape (M, K, N)."
            )

        self.M, self.K, self.N = self.H.shape

        self.comm_active = _validate_directions(
            self.W_dir
        )

        self.P_max = np.broadcast_to(
            np.asarray(P_max, dtype=float),
            (self.M,),
        ).copy()

        if np.any(self.P_max <= 0.0) or not np.all(
            np.isfinite(self.P_max)
        ):
            raise ValueError(
                "P_max must be finite and strictly positive "
                "for every AP."
            )

        if alpha < 0.0 or beta < 0.0:
            raise ValueError(
                "alpha and beta must be >= 0."
            )

        if alpha == 0.0 and beta == 0.0:
            raise ValueError(
                "alpha and beta cannot both be zero."
            )

        if noise_power <= 0.0:
            raise ValueError(
                "noise_power must be strictly positive."
            )

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.noise_power = float(noise_power)
        self.bandwidth_hz = float(bandwidth_hz)
        self.pilot_fraction = float(pilot_fraction)
        self.ref_rate = float(ref_rate_bps)
        self.ref_ig = float(ref_info_gain)

        self.Cr = float(
            physical_rate(
                np.array([1.0]),
                bandwidth_hz=self.bandwidth_hz,
                pilot_fraction=self.pilot_fraction,
            )[0]
        )

        self.g = np.einsum(
            "mkn,mln->mkl",
            np.conj(self.H),
            self.W_dir,
        )

        target_positions = np.asarray(
            target_positions,
            dtype=float,
        )
        self.Q = len(target_positions)

        if prior_covariances is None:
            prior_covariances = [
                np.eye(2)
                for _ in range(self.Q)
            ]

        self.prior = [
            np.asarray(p, dtype=float)
            for p in prior_covariances
        ]

        self.A, self.sens_useful = sensing_fim_basis(
            ap_positions,
            target_positions,
            y_tx,
            y_rx,
            G_sens,
            noise_power_sens=noise_power_sens,
            reference_variance=reference_variance,
        )

        self._check_fim_linearity(
            ap_positions,
            target_positions,
            y_tx,
            y_rx,
            G_sens,
            noise_power_sens,
            reference_variance,
        )

        self.comm_on = self.alpha > 0.0
        self.sens_on = self.beta > 0.0

        self.dual_role = (
            self.comm_active
            & self.sens_useful
        )

        if not self.comm_on:
            var_mask = np.zeros(
                self.M,
                dtype=bool,
            )
        elif self.scope == "all":
            var_mask = self.comm_active.copy()
        else:
            var_mask = self.dual_role.copy()

        self.var_idx = np.flatnonzero(
            var_mask
        )

        self.held_idx = np.flatnonzero(
            self.comm_active & ~var_mask
        ) if self.comm_on else np.zeros(
            0,
            dtype=int,
        )

    def _check_fim_linearity(
        self,
        ap_positions,
        target_positions,
        y_tx,
        y_rx,
        G_sens,
        noise_power_sens,
        reference_variance,
    ):
        p_full = np.where(
            self.sens_useful,
            self.P_max,
            0.0,
        )

        # The scalar FIM implementation applies a hard measurement-variance
        # floor of 1e-12. When that floor is active, J(s) is no longer
        # affine in sensing power, so the Phase 5B reduction is invalid.
        snr_full = sensing_snr(
            p_full,
            G_sens,
            y_tx,
            y_rx,
            noise_power_sens=noise_power_sens,
        )
        variance_full = snr_to_measurement_variance(
            snr_full,
            reference_variance=reference_variance,
        )
        finite_variances = variance_full[np.isfinite(variance_full)]

        if (
            finite_variances.size
            and np.any(finite_variances < 1e-12)
        ):
            raise ValueError(
                "FIM is not linear in sensing power over [0, P_max] "
                "because the scalar FIM's 1e-12 measurement-variance "
                "floor is active. Lower P_max or raise noise_power_sens."
            )

        J_auth = aggregate_fisher_information_bridge(
            ap_positions,
            target_positions,
            y_tx,
            y_rx,
            p_full,
            G_sens,
            noise_power_sens=noise_power_sens,
            reference_variance=reference_variance,
        )

        J_lin = np.einsum(
            "m,mqij->qij",
            p_full,
            self.A,
        )

        err = (
            np.max(np.abs(J_auth - J_lin))
            if J_auth.size
            else 0.0
        )

        scale = (
            max(
                1.0,
                float(np.max(np.abs(J_auth))),
            )
            if J_auth.size
            else 1.0
        )

        if err > _LINEARITY_RTOL * scale:
            raise ValueError(
                "FIM is not linear in sensing power over [0, P_max] "
                "(the scalar FIM's 1e-12 variance floor is active: "
                "SNR too high). The Phase 5B reduction requires linearity; "
                "lower P_max or raise noise_power_sens. "
                "max|J_bridge - J_linear| = "
                f"{err:.3e}."
            )

    def c_from_z(self, z):
        c = np.zeros(self.M)

        c[self.held_idx] = (
            self.P_max[self.held_idx]
        )

        c[self.var_idx] = (
            self.P_max[self.var_idx]
            * np.exp(z)
        )

        return c

    def s_from_c(self, c):
        if not self.sens_on:
            return np.zeros(self.M)

        return np.where(
            self.sens_useful,
            np.maximum(
                self.P_max - c,
                0.0,
            ),
            0.0,
        )

    def _sens_value_grad(self, s):
        J = np.einsum(
            "m,mqij->qij",
            s,
            self.A,
        )

        post = np.empty(
            (self.Q, 2, 2)
        )

        total = 0.0

        for q in range(self.Q):
            post[q] = posterior_covariance(
                self.prior[q],
                J[q],
            )

            total += information_gain(
                self.prior[q],
                post[q],
            )

        dU_ds = np.einsum(
            "qij,mqji->m",
            post,
            self.A,
        )

        return float(total), dU_ds

    def value_and_grad(self, z):
        z = np.asarray(
            z,
            dtype=float,
        )

        c = self.c_from_z(z)
        u = np.sqrt(c)

        W = (
            u[:, None, None]
            * self.W_dir
        )

        sinr = physical_sinr(
            self.H,
            W,
            self.noise_power,
        )

        rates = physical_rate(
            sinr,
            bandwidth_hz=self.bandwidth_hz,
            pilot_fraction=self.pilot_fraction,
        )

        Uc = float(
            np.sum(rates)
        )

        F = (
            self.alpha
            * Uc
            / self.ref_rate
        )

        grad_full = np.zeros(
            self.M
        )

        if (
            self.comm_on
            and len(self.var_idx)
        ):
            cross = np.einsum(
                "mkl,m->kl",
                self.g,
                u,
            )

            P = np.abs(cross) ** 2

            S = np.diag(P).copy()

            D = (
                np.sum(P, axis=1)
                - S
                + self.noise_power
            )

            gamma = S / D

            dP = (
                u[:, None, None]
                * np.real(
                    np.conj(cross)[None]
                    * self.g
                )
            )

            dS = np.einsum(
                "mkk->mk",
                dP,
            )

            dI = (
                np.sum(
                    dP,
                    axis=2,
                )
                - dS
            )

            dgamma = (
                (
                    dS * D[None]
                    - S[None] * dI
                )
                / (D[None] ** 2)
            )

            dR = (
                self.Cr
                / np.log(2.0)
                / (1.0 + gamma)[None]
                * dgamma
            )

            grad_full += (
                self.alpha
                / self.ref_rate
                * np.sum(
                    dR,
                    axis=1,
                )
            )

        if self.sens_on:
            s = self.s_from_c(c)

            Us, dUs_ds = (
                self._sens_value_grad(s)
            )

            F += (
                self.beta
                * Us
                / self.ref_ig
            )

            grad_full += (
                self.beta
                / self.ref_ig
                * (-c * dUs_ds)
            )

        return (
            F,
            grad_full[self.var_idx],
        )


def _projected_gradient_norm(
    z,
    grad,
    lb,
    ub,
):
    """inf-norm of projected ascent step."""
    if len(z) == 0:
        return 0.0

    return float(
        np.max(
            np.abs(
                z
                - np.clip(
                    z + grad,
                    lb,
                    ub,
                )
            )
        )
    )


def optimize_joint_allocation(
    H,
    W_dir,
    ap_positions,
    target_positions,
    y_tx,
    y_rx,
    G_sens,
    P_max,
    alpha=1.0,
    beta=1.0,
    prior_covariances=None,
    noise_power=1e-9,
    noise_power_sens=1e-9,
    reference_variance=1.0,
    bandwidth_hz=20e6,
    pilot_fraction=0.1,
    ref_rate_bps=1e7,
    ref_info_gain=1.0,
    min_rate_bps=None,
    comm_power_control="dual_role",
    n_polish=3,
    max_iter=500,
    kkt_tol=1e-5,
    tracking_error_thresholds=None,
    fronthaul_capacity=None,
    comm_fronthaul_per_user=None,
    sensing_fronthaul_per_target=None,
    control_fronthaul=0.0,
    x=None,
    enforce_qos=False,
):
    """
    Solve problem (P) for per-AP communication and sensing power.
    """
    if tracking_error_thresholds is not None:
        tracking_thresholds = np.asarray(tracking_error_thresholds, dtype=float)
    else:
        tracking_thresholds = None

    if fronthaul_capacity is not None:
        fh_capacity = np.broadcast_to(np.asarray(fronthaul_capacity, dtype=float), (H.shape[0],)).copy()
        comm_fh = np.ones(H.shape[1], dtype=float) if comm_fronthaul_per_user is None else np.broadcast_to(np.asarray(comm_fronthaul_per_user, dtype=float), (H.shape[1],)).copy()
        sens_fh = np.ones(len(target_positions), dtype=float) if sensing_fronthaul_per_target is None else np.broadcast_to(np.asarray(sensing_fronthaul_per_target, dtype=float), (len(target_positions),)).copy()
        control_fh = float(control_fronthaul)

    if min_rate_bps is not None:
        min_rate = np.asarray(min_rate_bps, dtype=float)
    else:
        min_rate = None

    prob = _JointProblem(
        H,
        W_dir,
        ap_positions,
        target_positions,
        y_tx,
        y_rx,
        G_sens,
        P_max,
        alpha,
        beta,
        prior_covariances,
        noise_power,
        noise_power_sens,
        reference_variance,
        bandwidth_hz,
        pilot_fraction,
        ref_rate_bps,
        ref_info_gain,
        comm_power_control=comm_power_control,
    )

    nvar = len(
        prob.var_idx
    )

    if not prob.comm_on:
        mode = (
            "sensing_only_boundary(alpha=0)"
        )
    elif not prob.sens_on:
        mode = (
            "communication_only_boundary(beta=0)"
        )
    else:
        mode = "joint"

    solver = {
        "method":
            "uniform_split_grid + L-BFGS-B(log-power, box bounds)",
        "mode": mode,
        "comm_power_control":
            comm_power_control,
        "decision_aps":
            [
                int(i)
                for i in prob.var_idx
            ],
        "held_at_baseline_aps":
            [
                int(i)
                for i in prob.held_idx
            ],
        "n_variables":
            int(nvar),
        "rho_grid":
            list(_RHO_GRID),
        "rho_floor":
            _RHO_FLOOR,
        "reduction":
            (
                "s_m = P_max,m - P_comm,m on sensing-useful APs "
                "when beta>0; s = 0 when beta=0; "
                "c = 0 when alpha=0"
            ),
        "global_optimality_claimed":
            False,
        "grid_objectives": [],
        "starts": [],
    }

    z_best = np.zeros(
        0
    )

    iterations = 0
    kkt = 0.0


    if nvar > 0:
        solver["method"] = "uniform_split_grid + L-BFGS-B"

    if (tracking_thresholds is not None or (min_rate is not None and enforce_qos)) and nvar > 0:
        lb = np.full(
            nvar,
            np.log(_RHO_FLOOR),
        )
        ub = np.zeros(nvar)
        bounds = list(zip(lb, ub))

        def constrained_objective(z):
            f, g = prob.value_and_grad(z)
            return -f, -g

        def get_tracking_and_qos_errors(z):
            c_local = prob.c_from_z(z)
            s_local = prob.s_from_c(c_local)

            W_local = (
                np.sqrt(c_local)[:, None, None]
                * prob.W_dir
            )

            if tracking_thresholds is not None:
                bridge_local = evaluate_physical_sensing_bridge(
                    ap_positions=ap_positions,
                    target_positions=target_positions,
                    y_tx=y_tx,
                    y_rx=y_rx,
                    W=W_local,
                    sensing_power_per_ap=s_local,
                    G_sens=G_sens,
                    tx_power_per_ap_max=prob.P_max,
                    prior_covariances=prior_covariances,
                    noise_power_sens=noise_power_sens,
                    reference_variance=reference_variance,
                )
                te = np.asarray(bridge_local["tracking_error"], dtype=float)
            else:
                te = np.zeros(0)

            if min_rate is not None and enforce_qos:
                sinr_local = physical_sinr(H, W_local, noise_power)
                r_local = physical_rate(sinr_local, bandwidth_hz=bandwidth_hz, pilot_fraction=pilot_fraction)
            else:
                r_local = np.zeros(0)

            return te, r_local

        def all_constraints(z):
            te, r_local = get_tracking_and_qos_errors(z)
            c = []
            if tracking_thresholds is not None:
                c.append(tracking_thresholds - te)
            if min_rate is not None and enforce_qos:
                c.append(r_local - min_rate)
            if not c:
                return np.array([0.0])
            return np.concatenate(c)

        # Use the existing deterministic rho-grid points as SLSQP starts.
        starts = []
        for rho in _RHO_GRID:
            starts.append(
                np.full(
                    nvar,
                    np.log(rho),
                    dtype=float,
                )
            )

        candidates = []

        for start_idx, z0 in enumerate(starts):
            res = minimize(
                constrained_objective,
                z0,
                jac=True,
                method="SLSQP",
                bounds=bounds,
                constraints=[
                    {
                        "type": "ineq",
                        "fun": all_constraints,
                    }
                ],
                options={
                    "maxiter": int(max_iter),
                    "ftol": 1e-12,
                    "disp": False,
                },
            )

            z_end = np.clip(
                np.asarray(res.x, dtype=float),
                lb,
                ub,
            )

            f_end, g_end = prob.value_and_grad(z_end)
            te_end, r_end = get_tracking_and_qos_errors(z_end)

            feasible = True
            if tracking_thresholds is not None:
                feasible = feasible and bool(np.all((tracking_thresholds - te_end) >= -1e-8))
            if min_rate is not None and enforce_qos:
                feasible = feasible and bool(np.all((r_end - min_rate) >= -1e-8))

            rec = {
                "start_rho": float(_RHO_GRID[start_idx]),
                "end_objective": float(f_end),
                "iterations": int(res.nit),
                "function_evaluations": int(res.nfev),
                "status": int(res.status),
                "message": str(res.message),
                "success": bool(res.success),
                "tracking_constraint_satisfied": feasible,
            }

            solver["starts"].append(rec)

            candidates.append(
                (
                    f_end if feasible else -np.inf,
                    z_end,
                    int(res.nit),
                    rec,
                    feasible,
                )
            )

        feasible_candidates = [
            cnd for cnd in candidates
            if cnd[4]
        ]

        if feasible_candidates:
            best = max(
                range(len(feasible_candidates)),
                key=lambda j: (
                    feasible_candidates[j][0],
                    -j,
                ),
            )

            (
                f_best,
                z_best,
                iterations,
                _rec,
                _feasible,
            ) = feasible_candidates[best]

            _, g_best = prob.value_and_grad(z_best)
            kkt = _projected_gradient_norm(
                z_best,
                g_best,
                lb,
                ub,
            )

            solver["method"] = "uniform_split_grid + SLSQP (tracking/QoS constraints)"
            solver["returned_from"] = "constrained_start"
            solver["tracking_constraints_enabled"] = True
        else:
            # No feasible constrained solution was found. Keep the best
            # objective candidate so the authoritative output can report
            # the actual violation rather than silently claiming feasibility.
            best = max(
                range(len(candidates)),
                key=lambda j: (
                    -float(
                        np.max(
                            np.concatenate([
                                (get_tracking_and_qos_errors(candidates[j][1])[0] - tracking_thresholds) if tracking_thresholds is not None else np.zeros(0),
                                (min_rate - get_tracking_and_qos_errors(candidates[j][1])[1]) if min_rate is not None else np.zeros(0),
                            ])
                        )
                    ) if (tracking_thresholds is not None or (min_rate is not None and enforce_qos)) else 0.0,
                    candidates[j][0],
                    -j,
                ),
            )

            (
                _f_best,
                z_best,
                iterations,
                _rec,
                _feasible,
            ) = candidates[best]

            _, g_best = prob.value_and_grad(z_best)
            kkt = _projected_gradient_norm(
                z_best,
                g_best,
                lb,
                ub,
            )

            solver["method"] = "uniform_split_grid + SLSQP (infeasible tracking/QoS constraints)"
            solver["returned_from"] = "least_tracking_violation"
            solver["tracking_constraints_enabled"] = True
            solver["tracking_constraints_feasible"] = False

    elif (tracking_thresholds is not None or (min_rate is not None and enforce_qos)) and nvar == 0:
        solver["tracking_constraints_enabled"] = True

    elif nvar > 0:

        lb = np.full(
            nvar,
            np.log(_RHO_FLOOR),
        )

        ub = np.zeros(
            nvar
        )

        bounds = list(
            zip(lb, ub)
        )

        grid = []

        for i, rho in enumerate(
            _RHO_GRID
        ):
            z0 = np.full(
                nvar,
                np.log(rho),
            )

            f0, _ = (
                prob.value_and_grad(z0)
            )

            grid.append(
                (
                    f0,
                    i,
                    z0,
                )
            )

            solver[
                "grid_objectives"
            ].append(
                {
                    "rho":
                        float(rho),
                    "objective":
                        float(f0),
                }
            )

        ranked = sorted(
            grid,
            key=lambda t: (
                -t[0],
                t[1],
            ),
        )

        candidates = [
            (
                f0,
                z0,
                0,
                None,
            )
            for (
                f0,
                _,
                z0,
            ) in grid
        ]

        for (
            f0,
            i,
            z0,
        ) in ranked[
            :max(
                0,
                int(n_polish),
            )
        ]:

            def fun(z):
                f, g = (
                    prob.value_and_grad(z)
                )
                return (
                    -f,
                    -g,
                )

            res = minimize(
                fun,
                z0,
                jac=True,
                method="L-BFGS-B",
                bounds=bounds,
                options={
                    "maxiter":
                        int(max_iter),
                    "ftol":
                        1e-12,
                    "gtol":
                        1e-9,
                    "maxcor":
                        20,
                },
            )

            z_end = np.clip(
                res.x,
                lb,
                ub,
            )

            f_end, g_end = (
                prob.value_and_grad(
                    z_end
                )
            )

            rec = {
                "start_rho":
                    float(
                        _RHO_GRID[i]
                    ),
                "start_objective":
                    float(f0),
                "end_objective":
                    float(f_end),
                "iterations":
                    int(res.nit),
                "function_evaluations":
                    int(res.nfev),
                "status":
                    int(res.status),
                "message":
                    str(res.message),
                "success":
                    bool(res.success),
                "projected_gradient_norm":
                    _projected_gradient_norm(
                        z_end,
                        g_end,
                        lb,
                        ub,
                    ),
            }

            solver[
                "starts"
            ].append(
                rec
            )

            candidates.append(
                (
                    f_end,
                    z_end,
                    int(res.nit),
                    rec,
                )
            )

        best = max(
            range(
                len(candidates)
            ),
            key=lambda j: (
                candidates[j][0],
                -j,
            ),
        )

        (
            f_best,
            z_best,
            iterations,
            _rec,
        ) = candidates[best]

        _, g_best = (
            prob.value_and_grad(
                z_best
            )
        )

        kkt = (
            _projected_gradient_norm(
                z_best,
                g_best,
                lb,
                ub,
            )
        )

        solver[
            "returned_from"
        ] = (
            "polished_start"
            if _rec is not None
            else "uniform_grid_point"
        )
    else:
        solver[
            "returned_from"
        ] = "closed_form_boundary"

    solver[
        "kkt_residual"
    ] = float(kkt)

    solver[
        "kkt_tol"
    ] = float(kkt_tol)

    converged = bool(
        kkt <= kkt_tol
    )

    c = prob.c_from_z(
        z_best
    )

    W_tmp = (
        np.sqrt(c)[:, None, None]
        * prob.W_dir
    )

    P_comm_meas = (
        comm_power_per_ap(
            W_tmp
        )
    )

    if prob.sens_on:
        s = np.where(
            prob.sens_useful,
            np.maximum(
                prob.P_max
                - P_comm_meas,
                0.0,
            ),
            0.0,
        )
    else:
        s = np.zeros(
            prob.M
        )

    out = evaluate_joint_allocation(
        H,
        W_dir,
        c,
        s,
        ap_positions,
        target_positions,
        y_tx,
        y_rx,
        G_sens,
        prob.P_max,
        alpha=alpha,
        beta=beta,
        prior_covariances=prior_covariances,
        noise_power=noise_power,
        noise_power_sens=noise_power_sens,
        reference_variance=reference_variance,
        bandwidth_hz=bandwidth_hz,
        pilot_fraction=pilot_fraction,
        ref_rate_bps=ref_rate_bps,
        ref_info_gain=ref_info_gain,
        min_rate_bps=min_rate_bps,
    )

    f_int, _ = (
        prob.value_and_grad(
            z_best
        )
    )

    solver[
        "objective_internal_vs_authoritative_abs_error"
    ] = float(
        abs(
            f_int
            - out["objective"]
        )
    )

    solver[
        "comm_power_setpoint_vs_measured_max_abs_error"
    ] = float(
        np.max(
            np.abs(
                out["P_comm"]
                - c
            )
        )
    )

    # Authoritative PDF tracking-accuracy constraint:
    # tr(P_q^+) <= epsilon_q^trk for every target q.
    if tracking_thresholds is not None:
        tracking_errors = np.asarray(
            out["tracking_error"],
            dtype=float,
        )
        tracking_margins = (
            tracking_thresholds
            - tracking_errors
        )
        tracking_violation = np.maximum(
            -tracking_margins,
            0.0,
        )

        tracking_satisfied = bool(
            np.all(
                tracking_margins >= -1e-8
            )
        )

        out["tracking_constraint_enabled"] = True
        out["tracking_error_thresholds"] = (
            tracking_thresholds.copy()
        )
        out["tracking_constraint_margin"] = (
            tracking_margins
        )
        out["tracking_constraint_violation"] = (
            tracking_violation
        )
        out["tracking_constraint_satisfied"] = (
            tracking_satisfied
        )

        # Power feasibility from evaluate_joint_allocation is not enough:
        # the PDF tracking constraint is part of the optimization feasibility.
        out["feasible"] = bool(
            out["feasible"]
            and tracking_satisfied
        )
    else:
        out["tracking_constraint_enabled"] = False
        out["tracking_constraint_satisfied"] = True


    # Authoritative PDF fronthaul-capacity constraint (evaluation only)
    if fronthaul_capacity is not None:
        if x is None:
            raise ValueError("x must be provided when fronthaul_capacity is supplied.")

        x_array = np.asarray(x, dtype=float)

        # F_m = sum_k x_mk * comm_fh_k + sum_q yR_mq * sens_fh_q + f_m^ctrl
        F_m = (
            x_array @ comm_fh
            + y_rx @ sens_fh
            + control_fh
        )

        fh_margins = fh_capacity - F_m
        fh_violation = np.maximum(-fh_margins, 0.0)
        fh_satisfied = bool(np.all(fh_margins >= -1e-8))

        out["fronthaul_constraint_enabled"] = True
        out["fronthaul_capacity"] = fh_capacity.copy()
        out["fronthaul_usage"] = F_m
        out["fronthaul_constraint_margin"] = fh_margins
        out["fronthaul_constraint_violation"] = fh_violation
        out["fronthaul_constraint_satisfied"] = fh_satisfied

        out["feasible"] = bool(out["feasible"] and fh_satisfied)
    else:
        out["fronthaul_constraint_enabled"] = False
        out["fronthaul_constraint_satisfied"] = True


    if min_rate is not None and enforce_qos:
        qos_margins = out["rates_bps"] - min_rate
        qos_satisfied = bool(np.all(qos_margins >= -1e-8))
        out["qos_constraint_satisfied"] = qos_satisfied
        out["feasible"] = bool(out["feasible"] and qos_satisfied)

    out[
        "converged"

    ] = converged

    out[
        "iterations"
    ] = int(iterations)

    out[
        "solver"
    ] = solver

    return out


__all__ = [
    "comm_beam_directions",
    "sensing_fim_basis",
    "uniform_split_allocation",
    "evaluate_joint_allocation",
    "optimize_joint_allocation",
]
