"""
tests/test_phase4b_constrained_beamforming.py

Focused tests for Phase 4B: SINR-constrained joint beamforming with
per-AP power constraints (src.beamforming.constrained).

These tests exercise ONLY the new constrained-beamforming path. They do
not modify or depend on prediction, clustering, or churn logic, and do
not touch the existing scalar / MRT / RZF paths (only import them for
side-by-side comparison, read-only).

Covers the Phase-4B validation checklist:
  1. feasible SINR cases
  2. infeasible SINR cases
  3. single-user case
  4. multi-user/multi-AP case
  5. achieved minimum SINR versus requested target
  6. per-AP power constraint violations
  7. convergence/iteration behavior
  8. deterministic repeated runs
  9. comparison against MRT and RZF
"""

import unittest

import numpy as np

from src.beamforming.constrained import (
    constrained_sinr_beamforming,
    bisection_sinr_beamforming,
)
from src.beamforming.precoders import mrt_weights, rzf_weights, normalize_per_ap_power
from src.metrics.physical_layer import physical_sinr
from src.channels.physical import generate_physical_channel
from src.environment.topology import generate_ap_positions, generate_positions
from src.channels.communication import generate_comm_gains
from src.clustering.overlapping import communication_clustering


def _random_channel(M, K, N, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    H = (rng.standard_normal((M, K, N)) + 1j * rng.standard_normal((M, K, N)))
    return (H * scale / np.sqrt(2)).astype(np.complex128)


def _topology_channel(num_aps=6, num_users=3, num_antennas=4, seed=42, area_size=100.0):
    ap_positions = generate_ap_positions(num_aps, area_size, seed=seed)
    user_positions = generate_positions(num_users, area_size, seed=seed + 1)
    H = generate_physical_channel(
        ap_positions, user_positions, num_antennas=num_antennas,
        rician_k_factor=10.0, small_scale_fading=True, seed=seed,
    )
    G_comm = generate_comm_gains(ap_positions, user_positions)
    x = communication_clustering(G_comm, num_users=num_users, aps_per_user=2)
    return H, x


# ─────────────────────────────────────────────────────────────────────────
# 1 & 5. Feasible SINR cases + achieved SINR vs requested target
# ─────────────────────────────────────────────────────────────────────────

class TestFeasibleCases(unittest.TestCase):

    def test_modest_target_is_feasible(self):
        H = _random_channel(3, 4, 4, seed=1)
        W, feasible, sinr, info = constrained_sinr_beamforming(
            H, gamma=1.0, tx_power_per_ap=1.0, noise_power=1e-2
        )
        self.assertTrue(feasible)
        self.assertEqual(info["reason"].startswith("Feasible"), True)

    def test_achieved_sinr_matches_target_when_feasible(self):
        H = _random_channel(3, 4, 4, seed=1)
        for gamma in [0.5, 2.0, 10.0]:
            W, feasible, sinr, info = constrained_sinr_beamforming(
                H, gamma=gamma, tx_power_per_ap=1.0, noise_power=1e-3
            )
            self.assertTrue(feasible, f"gamma={gamma} expected feasible")
            np.testing.assert_allclose(sinr, gamma, rtol=1e-3)

    def test_achieved_sinr_from_returned_W_matches_reported_sinr(self):
        """The SINR reported in the return value must equal physical_sinr(H, W)."""
        H = _random_channel(4, 3, 4, seed=2)
        W, feasible, sinr, info = constrained_sinr_beamforming(
            H, gamma=3.0, tx_power_per_ap=2.0, noise_power=1e-3
        )
        self.assertTrue(feasible)
        recomputed = physical_sinr(H, W, noise_power=1e-3)
        np.testing.assert_allclose(sinr, recomputed, rtol=1e-8)

    def test_zero_target_is_trivially_feasible_with_zero_power(self):
        H = _random_channel(2, 2, 4, seed=3)
        W, feasible, sinr, info = constrained_sinr_beamforming(
            H, gamma=0.0, tx_power_per_ap=1.0, noise_power=1e-3
        )
        self.assertTrue(feasible)
        np.testing.assert_allclose(W, 0.0)


# ─────────────────────────────────────────────────────────────────────────
# 2. Infeasible SINR cases (two distinct mechanisms)
# ─────────────────────────────────────────────────────────────────────────

class TestInfeasibleCases(unittest.TestCase):

    def test_interference_limited_infeasibility(self):
        """More users than spatial DoF (K > M*N) at high target SINR: the
        target is unachievable for ANY power allocation (independent of
        the power budget)."""
        H = _random_channel(1, 5, 2, seed=4)  # D=2, K=5
        W, feasible, sinr, info = constrained_sinr_beamforming(
            H, gamma=5.0, tx_power_per_ap=1e6, noise_power=1e-9, max_outer_iter=30
        )
        self.assertFalse(feasible)
        self.assertIn("not achievable for any", info["reason"])
        np.testing.assert_allclose(W, 0.0)

    def test_power_limited_infeasibility(self):
        """Spatially achievable target, but the per-AP power budget is far
        too small to reach it."""
        H = _random_channel(2, 2, 4, seed=5)
        W, feasible, sinr, info = constrained_sinr_beamforming(
            H, gamma=1e6, tx_power_per_ap=1e-9, noise_power=1e-3, max_outer_iter=30
        )
        self.assertFalse(feasible)
        self.assertIn("did not converge", info["reason"])

    def test_infeasible_result_still_returns_well_formed_shapes(self):
        H = _random_channel(1, 5, 2, seed=4)
        W, feasible, sinr, info = constrained_sinr_beamforming(
            H, gamma=5.0, tx_power_per_ap=1e6, noise_power=1e-9, max_outer_iter=30
        )
        self.assertEqual(W.shape, H.shape)
        self.assertEqual(sinr.shape, (5,))
        self.assertTrue(np.all(np.isfinite(W)))
        self.assertTrue(np.all(np.isfinite(sinr)))

    def test_negative_gamma_rejected(self):
        H = _random_channel(2, 2, 4, seed=6)
        with self.assertRaises(ValueError):
            constrained_sinr_beamforming(H, gamma=np.array([-1.0, 1.0]), tx_power_per_ap=1.0)


# ─────────────────────────────────────────────────────────────────────────
# 3. Single-user case
# ─────────────────────────────────────────────────────────────────────────

class TestSingleUser(unittest.TestCase):

    def test_single_ap_single_user(self):
        H = _random_channel(1, 1, 4, seed=7)
        W, feasible, sinr, info = constrained_sinr_beamforming(
            H, gamma=8.0, tx_power_per_ap=1.0, noise_power=1e-2
        )
        self.assertTrue(feasible)
        np.testing.assert_allclose(sinr, [8.0], rtol=1e-3)

    def test_multi_ap_single_user_uses_available_aps(self):
        """A single user served by 3 APs should be able to reach a higher
        SINR than with 1 AP at the same per-AP power (more array gain)."""
        H_full = _random_channel(3, 1, 4, seed=8)
        H_one_ap = H_full[:1]

        _, feasible_full, sinr_full, _ = constrained_sinr_beamforming(
            H_full, gamma=5.0, tx_power_per_ap=1.0, noise_power=1e-2
        )
        max_ratio_one_ap = None
        # With only 1 AP available, an equally high target may or may not
        # be feasible; check the achievable power-limited ceiling instead
        # by comparing achieved SINR at equal, deliberately-infeasible-for-
        # one-AP gamma via bisection (monotonic, so max achievable SINR
        # with 3 APs must be >= with 1 AP under identical per-AP budget).
        _, gamma_star_full, _, _ = bisection_sinr_beamforming(
            H_full, tx_power_per_ap=1.0, noise_power=1e-2, tol=1e-2
        )
        _, gamma_star_one, _, _ = bisection_sinr_beamforming(
            H_one_ap, tx_power_per_ap=1.0, noise_power=1e-2, tol=1e-2
        )
        self.assertTrue(feasible_full)
        self.assertGreaterEqual(gamma_star_full, gamma_star_one - 1e-6)


# ─────────────────────────────────────────────────────────────────────────
# 4. Multi-user / multi-AP case
# ─────────────────────────────────────────────────────────────────────────

class TestMultiUserMultiAP(unittest.TestCase):

    def test_realistic_topology(self):
        H, _x = _topology_channel(num_aps=6, num_users=3, num_antennas=4, seed=42)
        W, feasible, sinr, info = constrained_sinr_beamforming(
            H, gamma=1.0, tx_power_per_ap=1.0, noise_power=1e-9
        )
        self.assertEqual(W.shape, (6, 3, 4))
        self.assertTrue(feasible)
        self.assertTrue(np.all(sinr >= 1.0 - 1e-3))

    def test_larger_multi_user_multi_ap(self):
        H = _random_channel(8, 6, 4, seed=9)
        W, feasible, sinr, info = constrained_sinr_beamforming(
            H, gamma=1.5, tx_power_per_ap=1.0, noise_power=1e-2
        )
        self.assertTrue(feasible)
        self.assertEqual(sinr.shape, (6,))
        np.testing.assert_allclose(sinr, 1.5, rtol=1e-3)


# ─────────────────────────────────────────────────────────────────────────
# 6. Per-AP power constraint violations
# ─────────────────────────────────────────────────────────────────────────

class TestPerAPPowerConstraint(unittest.TestCase):

    def test_feasible_solution_respects_per_ap_budget(self):
        H = _random_channel(4, 3, 4, seed=10)
        tx_power_per_ap = np.array([1.0, 0.5, 2.0, 0.1])
        W, feasible, sinr, info = constrained_sinr_beamforming(
            H, gamma=1.0, tx_power_per_ap=tx_power_per_ap, noise_power=1e-3
        )
        self.assertTrue(feasible)
        ap_power = np.sum(np.abs(W) ** 2, axis=(1, 2))
        # Allow the same outer_tol slack the solver itself uses (default 1e-3).
        self.assertTrue(np.all(ap_power <= tx_power_per_ap * 1.01))

    def test_heterogeneous_ap_budgets_are_individually_enforced(self):
        H = _random_channel(3, 2, 4, seed=11)
        tx_power_per_ap = np.array([5.0, 5.0, 1e-6])  # AP 3 essentially starved
        W, feasible, sinr, info = constrained_sinr_beamforming(
            H, gamma=0.8, tx_power_per_ap=tx_power_per_ap, noise_power=1e-3,
            max_outer_iter=40,
        )
        ap_power = np.sum(np.abs(W) ** 2, axis=(1, 2))
        # Whether or not globally feasible, AP 3's own budget must never
        # be exceeded in the returned witness.
        self.assertLessEqual(ap_power[2], tx_power_per_ap[2] * 1.01 + 1e-9)

    def test_rejects_nonpositive_power_budget(self):
        H = _random_channel(2, 2, 4, seed=12)
        with self.assertRaises(ValueError):
            constrained_sinr_beamforming(H, gamma=1.0, tx_power_per_ap=0.0)


# ─────────────────────────────────────────────────────────────────────────
# 7. Convergence / iteration behavior
# ─────────────────────────────────────────────────────────────────────────

class TestConvergenceBehavior(unittest.TestCase):

    def test_info_reports_outer_iterations(self):
        H = _random_channel(3, 3, 4, seed=13)
        W, feasible, sinr, info = constrained_sinr_beamforming(
            H, gamma=1.0, tx_power_per_ap=1.0, noise_power=1e-2
        )
        self.assertIn("outer_iterations", info)
        self.assertGreaterEqual(info["outer_iterations"], 1)
        self.assertLessEqual(info["outer_iterations"], 60)

    def test_bisection_history_is_monotonically_shrinking(self):
        H = _random_channel(3, 3, 4, seed=14)
        W, gamma_star, sinr, history = bisection_sinr_beamforming(
            H, tx_power_per_ap=1.0, noise_power=1e-2, tol=1e-2
        )
        self.assertGreater(len(history), 0)
        for i in range(1, len(history)):
            width_prev = history[i - 1]["high"] - history[i - 1]["low"]
            width_cur = history[i]["high"] - history[i]["low"]
            self.assertLessEqual(width_cur, width_prev + 1e-12)
        # final bracket width below tolerance (or iteration cap hit)
        final_width = history[-1]["high"] - history[-1]["low"]
        self.assertLessEqual(final_width, 1e-2 * 2)  # one step's worth of slack

    def test_bisection_gamma_star_is_feasible(self):
        H = _random_channel(3, 3, 4, seed=14)
        W, gamma_star, sinr, history = bisection_sinr_beamforming(
            H, tx_power_per_ap=1.0, noise_power=1e-2, tol=1e-2
        )
        self.assertTrue(np.all(sinr >= gamma_star - 1e-2))


# ─────────────────────────────────────────────────────────────────────────
# 8. Deterministic repeated runs
# ─────────────────────────────────────────────────────────────────────────

class TestDeterminism(unittest.TestCase):

    def test_repeated_calls_are_bitwise_identical(self):
        H = _random_channel(4, 3, 4, seed=15)
        W1, f1, s1, info1 = constrained_sinr_beamforming(H, 2.0, 1.0, noise_power=1e-3)
        W2, f2, s2, info2 = constrained_sinr_beamforming(H, 2.0, 1.0, noise_power=1e-3)
        self.assertEqual(f1, f2)
        np.testing.assert_array_equal(W1, W2)
        np.testing.assert_array_equal(s1, s2)
        self.assertEqual(info1["outer_iterations"], info2["outer_iterations"])

    def test_repeated_bisection_calls_are_identical(self):
        H = _random_channel(3, 3, 4, seed=16)
        W1, g1, s1, h1 = bisection_sinr_beamforming(H, 1.0, noise_power=1e-3, tol=1e-2)
        W2, g2, s2, h2 = bisection_sinr_beamforming(H, 1.0, noise_power=1e-3, tol=1e-2)
        self.assertEqual(g1, g2)
        np.testing.assert_array_equal(W1, W2)
        self.assertEqual(len(h1), len(h2))

    def test_infeasible_calls_are_also_deterministic(self):
        H = _random_channel(1, 5, 2, seed=4)
        W1, f1, s1, i1 = constrained_sinr_beamforming(H, 5.0, 1e6, noise_power=1e-9, max_outer_iter=30)
        W2, f2, s2, i2 = constrained_sinr_beamforming(H, 5.0, 1e6, noise_power=1e-9, max_outer_iter=30)
        self.assertEqual(f1, f2)
        np.testing.assert_array_equal(W1, W2)


# ─────────────────────────────────────────────────────────────────────────
# 9. Comparison against MRT and RZF
# ─────────────────────────────────────────────────────────────────────────

class TestComparisonAgainstMRTAndRZF(unittest.TestCase):

    def test_constrained_meets_target_where_mrt_rzf_may_not(self):
        """At a moderate SINR target, the constrained design must meet it
        exactly; MRT/RZF (which do not know about any target) have no
        such guarantee -- this is the whole point of Phase 4B."""
        H, x = _topology_channel(num_aps=6, num_users=3, num_antennas=4, seed=42)
        tx_power_per_ap = 1.0
        noise_power = 1e-9

        W_c, feasible, sinr_c, info = constrained_sinr_beamforming(
            H, gamma=1.0, tx_power_per_ap=tx_power_per_ap, noise_power=noise_power
        )
        self.assertTrue(feasible)
        self.assertTrue(np.all(sinr_c >= 1.0 - 1e-3))

        W_mrt = mrt_weights(H, x)
        W_mrt, _ = normalize_per_ap_power(W_mrt, tx_power_per_ap=tx_power_per_ap)
        sinr_mrt = physical_sinr(H, W_mrt, noise_power=noise_power)

        W_rzf = rzf_weights(H, x)
        W_rzf, _ = normalize_per_ap_power(W_rzf, tx_power_per_ap=tx_power_per_ap)
        sinr_rzf = physical_sinr(H, W_rzf, noise_power=noise_power)

        # All three are well-formed, same shape, comparable via the same
        # physical_sinr function.
        self.assertEqual(W_c.shape, W_mrt.shape)
        self.assertEqual(W_c.shape, W_rzf.shape)
        self.assertTrue(np.all(np.isfinite(sinr_mrt)))
        self.assertTrue(np.all(np.isfinite(sinr_rzf)))

    def test_all_three_use_same_power_budget(self):
        H, x = _topology_channel(num_aps=6, num_users=3, num_antennas=4, seed=43)
        tx_power_per_ap = 1.0

        W_c, feasible, sinr_c, info = constrained_sinr_beamforming(
            H, gamma=0.5, tx_power_per_ap=tx_power_per_ap, noise_power=1e-9
        )
        ap_power_c = np.sum(np.abs(W_c) ** 2, axis=(1, 2))

        W_mrt = mrt_weights(H, x)
        W_mrt, ap_power_mrt = normalize_per_ap_power(W_mrt, tx_power_per_ap=tx_power_per_ap)

        # Constrained design never exceeds the SAME per-AP budget MRT/RZF
        # are normalized to.
        self.assertTrue(np.all(ap_power_c <= tx_power_per_ap * 1.01))
        np.testing.assert_allclose(ap_power_mrt[ap_power_mrt > 0], tx_power_per_ap, rtol=1e-6)

    def test_mrt_and_rzf_unaffected_by_new_module(self):
        """Importing/using the new constrained module must not change
        MRT/RZF outputs at all (regression guard)."""
        H, x = _topology_channel(num_aps=6, num_users=3, num_antennas=4, seed=44)
        W_mrt_a = mrt_weights(H, x)
        W_rzf_a = rzf_weights(H, x)

        # touch the new module in between
        constrained_sinr_beamforming(H[:2, :2], gamma=0.1, tx_power_per_ap=1.0)

        W_mrt_b = mrt_weights(H, x)
        W_rzf_b = rzf_weights(H, x)
        np.testing.assert_array_equal(W_mrt_a, W_mrt_b)
        np.testing.assert_array_equal(W_rzf_a, W_rzf_b)


# ─────────────────────────────────────────────────────────────────────────
# Shape / input validation
# ─────────────────────────────────────────────────────────────────────────

class TestInputValidation(unittest.TestCase):

    def test_rejects_nonpositive_noise(self):
        H = _random_channel(2, 2, 4, seed=17)
        with self.assertRaises(ValueError):
            constrained_sinr_beamforming(H, gamma=1.0, tx_power_per_ap=1.0, noise_power=0.0)

    def test_output_shapes(self):
        H = _random_channel(5, 4, 3, seed=18)
        W, feasible, sinr, info = constrained_sinr_beamforming(
            H, gamma=1.0, tx_power_per_ap=1.0, noise_power=1e-2
        )
        self.assertEqual(W.shape, (5, 4, 3))
        self.assertEqual(sinr.shape, (4,))
        self.assertEqual(W.dtype, np.complex128)


if __name__ == "__main__":
    unittest.main()
