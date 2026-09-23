"""
src/beamforming/constrained.py

Phase 4B — SINR-constrained ("QoS") communication beamforming with
per-AP power constraints, for the per-AP, per-antenna physical-layer
channel produced by src.channels.physical (same H convention as
src.beamforming.precoders: shape (M, K, N), M = APs, K = users,
N = antennas/AP).

Problem being solved
---------------------
Reference (MATLAB / CVX, feasibility SOCP, no association mask, all
APs may jointly serve all users):

    vendor/matlab_isac/optimization/opt_comm_SOCP_vec.m
    vendor/matlab_isac/optimization/bisection_SINR.m

    find    F(u, :) for u = 1..U      (F[u] in C^{M*N}, a joint precoder
                                        spanning every AP's antennas)
    s.t.    SINR_u(F) >= gamma,  for every user u
            sum_u ||F[u] restricted to AP m's antennas||^2 <= P_m,
                                        for every AP m

This is a convex second-order-cone feasibility problem. `gamma` is
searched over with an outer bisection (`bisection_SINR.m`) to find the
largest uniformly-achievable SINR floor.

Solver actually used here (and why)
------------------------------------
This sandbox has no CVX/cvxpy/MOSEK/general SOCP solver available
(numpy + scipy only, no network to install one -- see requirements.txt).
Re-deriving the CVX feasibility SOCP exactly therefore is not possible
with what is available. Rather than approximate it with something
unprincipled (e.g. a fixed-direction linear program on top of the
already-frozen RZF weights), this module implements the classical
**uplink-downlink duality fixed-point algorithm** for QoS/SINR-
constrained beamforming with (weighted) per-group power constraints:

    M. Schubert & H. Boche, "Solution of the Multiuser Downlink
    Beamforming Problem with Individual SINR Constraints," IEEE Trans.
    Veh. Technol., 2004 (sum-power case; the fixed-point core used
    below).
    W. Yu & T. Lan, "Transmitter Optimization for the Multi-Antenna
    Downlink With Per-Antenna Power Constraints," IEEE Trans. Signal
    Process., 2007 (generalization to per-antenna / per-group power
    constraints via an outer dual-weight update -- this is what turns
    the per-AP constraint into a solvable problem here).

For this problem class strong duality holds, so -- when the inner and
outer fixed-point iterations converge -- this recovers the SAME optimum
the CVX SOCP would (not merely a heuristic upper/lower bound). It needs
only dense linear algebra (np.linalg.solve), so it is deterministic and
numerically stable by construction (see "Numerical stability" below).

Algorithm (informal)
---------------------
Outer loop over per-AP dual weights lambda_m >= 0 (initialized to 1):
  1. Build a per-antenna noise-weight vector Psi from lambda (each
     antenna inherits its AP's lambda_m).
  2. Inner fixed point: find virtual-uplink powers q_k such that the
     MMSE-receiver uplink SINR of every user exactly equals `gamma`
     (Yates' standard-interference-function iteration; monotone,
     provably convergent to the unique fixed point when one exists --
     and provably DIVERGING when the target SINR exceeds what is
     achievable for ANY power allocation, which is exactly the
     infeasible-SINR certificate).
  3. Downlink beamforming DIRECTIONS equal the uplink MMSE filters
     (duality theorem) -- computed via one D x D linear solve, not a
     matrix inverse, for stability.
  4. Downlink POWERS are then the unique solution of a small K x K
     linear system that makes every downlink SINR exactly `gamma`
     given those fixed directions.
  5. If every AP's resulting power is within its budget: FEASIBLE,
     done. Otherwise scale lambda_m up in proportion to how far AP m
     overshot its budget (a standard multiplicative dual update) and
     repeat.

Association / MRT / RZF
------------------------
The existing MRT/RZF precoders (src.beamforming.precoders) are
PER-AP-LOCAL: each AP only forms a beam using the channels of the
users *its own association row* says it serves, using purely local
CSI -- unmodified by this module. The SINR-constrained solver here
instead follows the vendor SOCP reference exactly: it is a GLOBAL/
JOINT design (every AP may contribute to every user's signal, subject
only to the per-AP power budget). These are two different, both
legitimate, precoding architectures; this file adds the second one
without touching the first. `mrt_weights` and `rzf_weights` are not
imported, called, or modified anywhere in this module.

Numerical stability
--------------------
- All matrix "inversions" are performed via np.linalg.solve (never
  explicit np.linalg.inv).
- The inner fixed point is checked for divergence (any q_k exceeding
  a large but finite bound, or a non-finite value) and reported as
  infeasible rather than allowed to overflow.
- The downlink K x K linear system is solved once directions are
  fixed; a singular/ill-conditioned system is caught and reported as
  infeasible (with `solver_status`) rather than raising.
- Everything here is a pure function of its numeric inputs (no RNG),
  so repeated calls with identical inputs are bit-for-bit identical
  (see TestDeterminism in tests/test_phase4b_constrained_beamforming.py).
"""

import numpy as np

_LARGE_POWER_BOUND = 1e12       # inner fixed point: q_k beyond this => diverging/infeasible
_MIN_DIAGONAL = 1e-30            # guards against divide-by-zero on degenerate channels


def _stack_global_channel(H):
    """
    Flatten the (M, K, N) per-AP channel into (K, D) global channel
    vectors (D = M*N), plus the AP index owning each of the D
    dimensions. h[k, :] = concat_m H[m, k, :].

    Returns
    -------
    h : ndarray, shape (K, D), complex128
    ap_of_dim : ndarray, shape (D,), int
    M, N : int
    """
    H = np.asarray(H, dtype=np.complex128)
    M, K, N = H.shape
    # (M, K, N) -> (K, M, N) -> (K, M*N); dimension d = m*N + n belongs to AP m.
    h = np.transpose(H, (1, 0, 2)).reshape(K, M * N)
    ap_of_dim = np.repeat(np.arange(M), N)
    return h, ap_of_dim, M, N


def _uplink_fixed_point(h, psi, gamma, max_iter=500, tol=1e-9):
    """
    Yates' standard-interference-function fixed point for the virtual
    uplink: find q (K,) >= 0 such that, for every user k,

        q_k * h_k^H (diag(psi) + sum_{l != k} q_l h_l h_l^H)^{-1} h_k = gamma_k

    Uses the Sherman-Morrison identity so each iteration needs only ONE
    D x D linear solve (R X = h^T) rather than K separate D x D
    inversions.

    Returns
    -------
    q : ndarray, shape (K,), float64
    converged : bool
    diverged : bool
    R : ndarray, shape (D, D), complex128 -- final virtual-uplink
        covariance (diag(psi) + sum_l q_l h_l h_l^H), used to recover
        beamforming directions.
    """
    K, D = h.shape
    gamma = np.asarray(gamma, dtype=float)
    q = np.ones(K, dtype=float)  # deterministic fixed starting point

    R = None
    for _ in range(max_iter):
        R = np.diag(psi).astype(np.complex128) + (h.T * q) @ h.conj()
        # X[:, k] = R^{-1} h_k  for every k at once.
        try:
            X = np.linalg.solve(R, h.T)
        except np.linalg.LinAlgError:
            return q, False, True, R

        u = np.real(np.sum(h.conj() * X.T, axis=1))  # u_k = h_k^H R^{-1} h_k
        denom = 1.0 - q * u
        if not np.all(np.isfinite(u)) or np.any(denom <= 1e-12):
            return q, False, True, R

        a = u / denom                       # a_k = h_k^H R_k^{-1} h_k  (Sherman-Morrison)
        a = np.maximum(a, _MIN_DIAGONAL)
        q_new = gamma / a

        if not np.all(np.isfinite(q_new)) or np.any(q_new > _LARGE_POWER_BOUND):
            return q_new, False, True, R

        if np.max(np.abs(q_new - q)) < tol * max(1.0, np.max(q_new)):
            return q_new, True, False, R

        q = q_new

    return q, False, False, R  # exhausted iteration budget without converging


def _solve_downlink_powers(h, w, gamma, noise_power):
    """
    Given fixed unit-norm downlink directions w (K, D) and SINR targets
    gamma (K,), solve the K x K linear system for the unique downlink
    powers p (K,) that make every user's SINR exactly gamma_k:

        p_k a_k = gamma_k * (sum_{l != k} p_l B_{kl} + noise_power)

    Returns
    -------
    p : ndarray, shape (K,), float64 or None (singular system)
    a : ndarray, shape (K,), float64  -- |h_k^H w_k|^2
    B : ndarray, shape (K, K), float64 -- |h_k^H w_l|^2, zero diagonal
    """
    K = h.shape[0]
    cross = h.conj() @ w.T           # cross[k, l] = h_k^H w_l
    power = np.abs(cross) ** 2       # (K, K)
    a = np.diag(power).copy()
    a = np.maximum(a, _MIN_DIAGONAL)
    B = power.copy()
    np.fill_diagonal(B, 0.0)

    D_diag = gamma / a
    lhs = np.eye(K) - D_diag[:, None] * B
    rhs = D_diag * noise_power

    try:
        p = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return None, a, B

    return p, a, B


def constrained_sinr_beamforming(
    H,
    gamma,
    tx_power_per_ap,
    noise_power=1e-9,
    max_outer_iter=60,
    max_inner_iter=500,
    inner_tol=1e-9,
    outer_tol=1e-3,
):
    """
    Design a GLOBAL/JOINT SINR-constrained downlink precoder meeting a
    per-user SINR target `gamma`, subject to a per-AP power budget,
    via the uplink-downlink duality fixed point described in this
    module's docstring.

    Parameters
    ----------
    H : ndarray, shape (M, K, N), complex
        Physical-layer channel (src.channels.physical.generate_physical_channel).
    gamma : float or ndarray, shape (K,)
        Target SINR (linear, not dB). A scalar is broadcast to every
        user. gamma <= 0 is trivially feasible (returns the all-zero
        precoder).
    tx_power_per_ap : float or ndarray, shape (M,)
        Per-AP transmit power budget (linear units, e.g. Watts).
    noise_power : float
        Receiver noise power (linear units), must be > 0.
    max_outer_iter : int
        Maximum per-AP dual-weight (lambda) updates.
    max_inner_iter : int
        Maximum iterations of the inner virtual-uplink power-control
        fixed point, per outer iteration.
    inner_tol : float
        Relative convergence tolerance for the inner fixed point.
    outer_tol : float
        Per-AP power constraint is considered satisfied once
        used_power <= (1 + outer_tol) * budget.

    Returns
    -------
    W : ndarray, shape (M, K, N), complex128
        Beamforming weights. All-zero if gamma <= 0 for every user, or
        if the problem is found infeasible (so callers can still
        safely compute physical_sinr on it -- SINR will be 0).
    feasible : bool
        True iff a per-AP-power-feasible design meeting `gamma` for
        every user was found.
    achieved_sinr : ndarray, shape (K,), float64
        Actual SINR delivered by W (computed the same way as
        src.metrics.physical_layer.physical_sinr). Equal to `gamma`
        (within tolerance) for every user when feasible is True.
    info : dict
        Diagnostics: solver name, outer/inner iteration counts,
        per-AP power used, per-AP power budget, and (when infeasible)
        a human-readable `reason`.
    """
    H = np.asarray(H, dtype=np.complex128)
    M, K, N = H.shape

    gamma_vec = np.broadcast_to(np.asarray(gamma, dtype=float), (K,)).copy()

    if noise_power <= 0:
        raise ValueError("noise_power must be strictly positive.")

    tx_power_per_ap = np.broadcast_to(
        np.asarray(tx_power_per_ap, dtype=float), (M,)
    ).copy()
    if np.any(tx_power_per_ap <= 0):
        raise ValueError("tx_power_per_ap must be strictly positive for every AP.")

    info = {
        "solver": "uplink_downlink_duality_fixed_point",
        "reference": "Schubert & Boche 2004; Yu & Lan 2007 (per-AP-group generalization)",
        "outer_iterations": 0,
        "inner_iterations": None,
        "ap_power_used_w": np.zeros(M),
        "ap_power_budget_w": tx_power_per_ap,
    }

    # Degenerate but well-defined case: target SINR is non-positive for
    # every user -> trivially feasible with zero transmit power.
    if np.all(gamma_vec <= 0.0):
        W = np.zeros((M, K, N), dtype=np.complex128)
        info["reason"] = "gamma <= 0 for every user: trivially feasible with zero power."
        return W, True, np.zeros(K), info

    if np.any(gamma_vec < 0.0):
        raise ValueError("gamma must be >= 0 for every user (use 0 to disable a user's constraint).")

    h, ap_of_dim, M_check, N_check = _stack_global_channel(H)
    assert (M_check, N_check) == (M, N)
    D = M * N

    lam = np.ones(M, dtype=float)  # per-AP dual weight, deterministic init

    best_p = None
    best_w = None
    best_ratio = np.inf

    for outer_it in range(1, max_outer_iter + 1):
        psi = lam[ap_of_dim]  # (D,), each antenna inherits its AP's lambda

        q, converged, diverged, R = _uplink_fixed_point(
            h, psi, gamma_vec, max_iter=max_inner_iter, tol=inner_tol
        )
        info["inner_iterations"] = info.get("inner_iterations") or []
        info["inner_iterations"].append(int(max_inner_iter if not converged and not diverged else 0))

        if diverged or R is None:
            # This gamma is not achievable for ANY power allocation under
            # this (or any less generous) lambda weighting -- a genuine
            # SINR-infeasibility certificate, independent of the AP power
            # budgets. Bisection callers rely on this to shrink `high`.
            info["outer_iterations"] = outer_it
            info["reason"] = (
                "Virtual-uplink power-control fixed point diverged: the requested "
                "SINR target is not achievable for any beamforming/power allocation "
                "(interference-limited infeasibility, independent of per-AP power)."
            )
            W = np.zeros((M, K, N), dtype=np.complex128)
            return W, False, np.zeros(K), info

        try:
            X = np.linalg.solve(R, h.T)  # X[:, k] = R^{-1} h_k, i.e. duality direction
        except np.linalg.LinAlgError:
            info["outer_iterations"] = outer_it
            info["reason"] = "Virtual-uplink covariance became singular while recovering beam directions."
            W = np.zeros((M, K, N), dtype=np.complex128)
            return W, False, np.zeros(K), info

        v = X.T  # (K, D)
        norms = np.linalg.norm(v, axis=1)
        norms = np.where(norms > 1e-15, norms, 1.0)
        w = v / norms[:, None]  # unit-norm downlink directions

        p, a, B = _solve_downlink_powers(h, w, gamma_vec, noise_power)

        if p is None or not np.all(np.isfinite(p)) or np.any(p < -1e-6):
            info["outer_iterations"] = outer_it
            info["reason"] = (
                "Downlink power-allocation linear system was singular or returned "
                "a negative power: SINR targets not simultaneously achievable with "
                "these fixed beam directions."
            )
            W = np.zeros((M, K, N), dtype=np.complex128)
            return W, False, np.zeros(K), info

        p = np.maximum(p, 0.0)

        # Per-AP power actually used: sum_k p_k * ||w_k restricted to AP m||^2
        w_power = (np.abs(w) ** 2) * p[:, None]     # (K, D)
        ap_power_used = np.zeros(M)
        np.add.at(ap_power_used, ap_of_dim, np.sum(w_power, axis=0))

        ratio = ap_power_used / tx_power_per_ap
        max_ratio = float(np.max(ratio))

        if max_ratio < best_ratio:
            best_ratio = max_ratio
            best_p, best_w = p.copy(), w.copy()

        if max_ratio <= 1.0 + outer_tol:
            info["outer_iterations"] = outer_it
            info["ap_power_used_w"] = ap_power_used
            info["reason"] = "Feasible: per-AP power constraints satisfied for the requested SINR target."

            W = np.zeros((M, K, N), dtype=np.complex128)
            w_reshaped = w.reshape(K, M, N)
            for k in range(K):
                W[:, k, :] = np.sqrt(p[k]) * w_reshaped[k]

            achieved_sinr = _achieved_sinr(H, W, noise_power)
            return W, True, achieved_sinr, info

        # Multiplicative dual update (Yu & Lan): push lambda_m up in
        # proportion to how far AP m overshot its budget.
        lam = lam * np.maximum(ratio, 1e-3)
        # Renormalize (geometric mean = 1) so lambda cannot drift to 0/inf
        # as a group -- only the *relative* weighting across APs matters.
        log_lam = np.log(np.maximum(lam, 1e-12))
        lam = np.exp(log_lam - np.mean(log_lam))

    # Outer budget exhausted without meeting every AP's power constraint.
    # Report infeasible, but keep the closest witness found for diagnostics.
    info["outer_iterations"] = max_outer_iter
    info["best_power_ratio_found"] = best_ratio
    info["reason"] = (
        f"Outer per-AP power-balancing loop did not converge within "
        f"{max_outer_iter} iterations (best max per-AP power ratio found: "
        f"{best_ratio:.4f}). Treated as infeasible (conservative)."
    )

    if best_w is not None and best_ratio <= 2.0:
        # Close-to-feasible witness: return it (still flagged infeasible)
        # so callers/tests can inspect how close the solver got.
        W = np.zeros((M, K, N), dtype=np.complex128)
        w_reshaped = best_w.reshape(K, M, N)
        for k in range(K):
            W[:, k, :] = np.sqrt(best_p[k]) * w_reshaped[k]
        achieved_sinr = _achieved_sinr(H, W, noise_power)
        return W, False, achieved_sinr, info

    W = np.zeros((M, K, N), dtype=np.complex128)
    return W, False, np.zeros(K), info


def _achieved_sinr(H, W, noise_power):
    """Local copy of the physical_sinr formula (avoids a circular import
    with src.metrics.physical_layer, which is free to import THIS module
    if ever useful -- see src.beamforming.__init__ for the public API)."""
    H = np.asarray(H, dtype=np.complex128)
    W = np.asarray(W, dtype=np.complex128)
    cross_gain = np.einsum("mkn,mln->kl", np.conj(H), W)
    power = np.abs(cross_gain) ** 2
    signal_power = np.diag(power).copy()
    interference_power = np.maximum(np.sum(power, axis=1) - signal_power, 0.0)
    return signal_power / (interference_power + noise_power)


def bisection_sinr_beamforming(
    H,
    tx_power_per_ap,
    noise_power=1e-9,
    low=0.0,
    high=None,
    tol=1e-2,
    max_bisection_iter=40,
    max_outer_iter=60,
    max_inner_iter=500,
):
    """
    Bisection over a uniform target SINR floor `gamma`, following
    vendor/matlab_isac/optimization/bisection_SINR.m: repeatedly test
    feasibility of `constrained_sinr_beamforming(H, gamma=mid, ...)`
    and shrink [low, high] until it is smaller than `tol`, keeping the
    beamformer from the LAST feasible midpoint tested.

    This answers "what is the largest SINR floor this network can
    guarantee to every user, given the per-AP power budgets?" -- the
    max-min-fair SINR design point.

    Parameters
    ----------
    H : ndarray, shape (M, K, N), complex
    tx_power_per_ap : float or ndarray, shape (M,)
    noise_power : float
    low : float
        Lower SINR bound (linear). 0.0 is always feasible.
    high : float or None
        Upper SINR bound (linear). If None, a generous analytic upper
        bound is derived from the single-user, no-interference,
        full-power case (guaranteed infeasible or exactly at the true
        optimum boundary once interference/other users are considered
        -- a safe starting point for bisection either way).
    tol : float
        Bisection stops once (high - low) < tol (linear SINR units).
    max_bisection_iter : int
        Hard cap on bisection steps (safety net; `tol` normally stops
        it first).
    max_outer_iter, max_inner_iter :
        Forwarded to `constrained_sinr_beamforming` at every midpoint.

    Returns
    -------
    W_star : ndarray, shape (M, K, N), complex128
        Beamformer at the best feasible SINR found (`gamma_star`).
        All-zero if even `low` (default 0.0) could not be certified
        feasible (should not happen for low=0.0, see
        `constrained_sinr_beamforming`'s gamma<=0 special case).
    gamma_star : float
        Largest target SINR (linear) for which feasibility was
        confirmed.
    achieved_sinr : ndarray, shape (K,)
        Per-user SINR actually delivered by W_star.
    history : list of dict
        One entry per bisection step: {mid, feasible, low, high},
        useful for the "convergence/iteration behavior" and
        "deterministic repeated runs" validation checks.
    """
    H = np.asarray(H, dtype=np.complex128)
    M, K, N = H.shape

    if high is None:
        # Single-user, full-power, no-interference upper bound:
        # sum_m P_m * max_k ||H[m,k,:]||^2 / noise_power.
        tx_power_per_ap_arr = np.broadcast_to(
            np.asarray(tx_power_per_ap, dtype=float), (M,)
        )
        per_ap_gain = np.max(np.sum(np.abs(H) ** 2, axis=2), axis=1)  # (M,)
        high = float(np.sum(tx_power_per_ap_arr * per_ap_gain) / noise_power)
        high = max(high, 1e-6)

    low = float(low)
    high = float(high)

    W_star, gamma_star, achieved_sinr = constrained_sinr_beamforming(
        H, low, tx_power_per_ap, noise_power=noise_power,
        max_outer_iter=max_outer_iter, max_inner_iter=max_inner_iter,
    )[0], low, None
    achieved_sinr = _achieved_sinr(H, W_star, noise_power)

    history = []
    for step in range(max_bisection_iter):
        if (high - low) <= tol:
            break
        mid = 0.5 * (high + low)
        W_mid, feasible_mid, sinr_mid, _ = constrained_sinr_beamforming(
            H, mid, tx_power_per_ap, noise_power=noise_power,
            max_outer_iter=max_outer_iter, max_inner_iter=max_inner_iter,
        )
        history.append({"step": step, "mid": mid, "feasible": bool(feasible_mid),
                         "low": low, "high": high})
        if feasible_mid:
            low = mid
            W_star, gamma_star, achieved_sinr = W_mid, mid, sinr_mid
        else:
            high = mid

    return W_star, gamma_star, achieved_sinr, history


__all__ = [
    "constrained_sinr_beamforming",
    "bisection_sinr_beamforming",
]
