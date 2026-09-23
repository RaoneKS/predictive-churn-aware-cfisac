"""
tests/test_phase4a_physical_layer.py

Focused tests for Phase 4A: physical (complex, per-antenna) channel,
steering-vector model, MRT/RZF beamforming, per-AP power normalization,
and physical-layer SINR/rate.

These tests exercise ONLY the new physical-layer path
(src.channels.physical, src.beamforming.precoders,
src.metrics.physical_layer). They do not modify or depend on the
existing scalar channel/SINR path, which is covered by the pre-existing
test suite and is checked here only for a light backward-compatibility
smoke test (see TestScalarPathUnaffected).
"""

import unittest

import numpy as np

from src.channels.communication import generate_comm_gains, generate_communication_features
from src.channels.physical import (
    aoa_2d,
    steering_matrix,
    steering_vector,
    generate_physical_channel,
)
from src.beamforming.precoders import (
    mrt_weights,
    rzf_weights,
    normalize_per_ap_power,
)
from src.metrics.physical_layer import physical_sinr, physical_rate
from src.metrics.system import communication_sinr, communication_rate
from src.environment.topology import generate_ap_positions, generate_positions
from src.clustering.overlapping import communication_clustering


def _small_topology(num_aps=6, num_users=3, area_size=100.0, seed=42):
    ap_positions = generate_ap_positions(num_aps, area_size, seed=seed)
    user_positions = generate_positions(num_users, area_size, seed=seed + 1)
    return ap_positions, user_positions


def _default_association(ap_positions, user_positions, aps_per_user=2):
    G_comm = generate_comm_gains(ap_positions, user_positions)
    return communication_clustering(
        G_comm, num_users=len(user_positions), aps_per_user=aps_per_user
    )


# ─────────────────────────────────────────────────────────────────────────
# 1. Steering vector model
# ─────────────────────────────────────────────────────────────────────────

class TestSteeringVector(unittest.TestCase):

    def test_steering_vector_shape(self):
        a = steering_vector(theta=0.3, num_antennas=8)
        self.assertEqual(a.shape, (8,))
        self.assertEqual(a.dtype, np.complex128)

    def test_steering_normalization(self):
        """||a(theta)|| == sqrt(N) for arbitrary angles (unit element power)."""
        N = 6
        for theta in [-1.2, -0.3, 0.0, 0.5, 1.4]:
            a = steering_vector(theta, N)
            self.assertAlmostEqual(np.linalg.norm(a), np.sqrt(N), places=10)

    def test_steering_matrix_matches_vector(self):
        """Vectorized steering_matrix must agree with per-angle steering_vector."""
        thetas = np.array([-1.0, 0.0, 0.7, 2.0])
        N = 5
        mat = steering_matrix(thetas, N)
        self.assertEqual(mat.shape, (4, N))
        for i, theta in enumerate(thetas):
            expected = steering_vector(theta, N)
            np.testing.assert_allclose(mat[i], expected, atol=1e-12)

    def test_broadside_zero_angle_is_all_ones(self):
        """At theta=0 (broadside), all elements are in phase."""
        a = steering_vector(0.0, num_antennas=4)
        np.testing.assert_allclose(a, np.ones(4, dtype=np.complex128), atol=1e-12)

    def test_aoa_shape(self):
        ap_positions, user_positions = _small_topology()
        theta = aoa_2d(ap_positions, user_positions)
        self.assertEqual(theta.shape, (len(ap_positions), len(user_positions)))
        # angles must be finite and within (-pi, pi]
        self.assertTrue(np.all(np.isfinite(theta)))
        self.assertTrue(np.all(theta <= np.pi + 1e-9))
        self.assertTrue(np.all(theta > -np.pi - 1e-9))


# ─────────────────────────────────────────────────────────────────────────
# 2. Complex physical channel: dimensions, complex-valuedness, determinism
# ─────────────────────────────────────────────────────────────────────────

class TestPhysicalChannel(unittest.TestCase):

    def test_channel_dimensions(self):
        ap_positions, user_positions = _small_topology(num_aps=5, num_users=4)
        N = 8
        H = generate_physical_channel(ap_positions, user_positions, num_antennas=N)
        self.assertEqual(H.shape, (5, 4, N))

    def test_channel_is_complex_valued(self):
        ap_positions, user_positions = _small_topology()
        H = generate_physical_channel(ap_positions, user_positions, num_antennas=4, seed=1)
        self.assertEqual(H.dtype, np.complex128)
        # Not all entries should be purely real (would indicate a bug in
        # the steering/fading phase model).
        self.assertTrue(np.any(np.abs(H.imag) > 1e-9))

    def test_channel_average_power_matches_scalar_gain(self):
        """
        E[|H[m,k,n]|^2] should match the existing scalar G_comm[m,k]
        (checked via Monte Carlo averaging over many fading realizations).
        """
        ap_positions, user_positions = _small_topology(num_aps=2, num_users=2)
        G_comm = generate_comm_gains(ap_positions, user_positions)

        N = 4
        n_trials = 400
        power_accum = np.zeros((2, 2))
        for trial in range(n_trials):
            H = generate_physical_channel(
                ap_positions, user_positions, num_antennas=N, seed=trial
            )
            power_accum += np.mean(np.abs(H) ** 2, axis=-1)

        empirical_power = power_accum / n_trials
        np.testing.assert_allclose(empirical_power, G_comm, rtol=0.15)

    def test_deterministic_generation_same_seed(self):
        ap_positions, user_positions = _small_topology()
        H1 = generate_physical_channel(ap_positions, user_positions, num_antennas=4, seed=7)
        H2 = generate_physical_channel(ap_positions, user_positions, num_antennas=4, seed=7)
        np.testing.assert_array_equal(H1, H2)

    def test_different_seeds_differ_with_fading(self):
        ap_positions, user_positions = _small_topology()
        H1 = generate_physical_channel(ap_positions, user_positions, num_antennas=4, seed=1)
        H2 = generate_physical_channel(ap_positions, user_positions, num_antennas=4, seed=2)
        self.assertFalse(np.allclose(H1, H2))

    def test_no_fading_is_deterministic_regardless_of_seed(self):
        """With small_scale_fading=False the channel is pure LOS/steering:
        no RNG is used, so the seed must not matter at all."""
        ap_positions, user_positions = _small_topology()
        H1 = generate_physical_channel(
            ap_positions, user_positions, num_antennas=4,
            small_scale_fading=False, seed=1,
        )
        H2 = generate_physical_channel(
            ap_positions, user_positions, num_antennas=4,
            small_scale_fading=False, seed=999,
        )
        np.testing.assert_array_equal(H1, H2)

    def test_invalid_num_antennas_rejected(self):
        ap_positions, user_positions = _small_topology()
        with self.assertRaises(ValueError):
            generate_physical_channel(ap_positions, user_positions, num_antennas=0)


# ─────────────────────────────────────────────────────────────────────────
# 3. MRT beamforming
# ─────────────────────────────────────────────────────────────────────────

class TestMRTBeamforming(unittest.TestCase):

    def test_mrt_dimensions(self):
        ap_positions, user_positions = _small_topology(num_aps=6, num_users=3)
        H = generate_physical_channel(ap_positions, user_positions, num_antennas=4, seed=0)
        x = _default_association(ap_positions, user_positions)
        W = mrt_weights(H, x)
        self.assertEqual(W.shape, H.shape)

    def test_mrt_zero_for_unassociated_pairs(self):
        ap_positions, user_positions = _small_topology(num_aps=6, num_users=3)
        H = generate_physical_channel(ap_positions, user_positions, num_antennas=4, seed=0)
        x = _default_association(ap_positions, user_positions)
        W = mrt_weights(H, x)

        unassociated = x == 0
        self.assertTrue(np.all(W[unassociated] == 0))

    def test_mrt_unit_norm_for_associated_pairs(self):
        ap_positions, user_positions = _small_topology(num_aps=6, num_users=3)
        H = generate_physical_channel(ap_positions, user_positions, num_antennas=4, seed=0)
        x = _default_association(ap_positions, user_positions)
        W = mrt_weights(H, x)

        norms = np.linalg.norm(W, axis=-1)
        associated_norms = norms[x == 1]
        np.testing.assert_allclose(associated_norms, 1.0, atol=1e-10)

    def test_mrt_matches_scaled_channel_direction(self):
        """w_mrt should be parallel to h (matched filter), not just unit norm."""
        ap_positions, user_positions = _small_topology(num_aps=2, num_users=1)
        H = generate_physical_channel(ap_positions, user_positions, num_antennas=4, seed=0)
        x = np.ones((2, 1), dtype=int)
        W = mrt_weights(H, x)
        for m in range(2):
            h = H[m, 0, :]
            w = W[m, 0, :]
            cos_similarity = np.abs(np.vdot(h, w)) / (np.linalg.norm(h) * np.linalg.norm(w))
            self.assertAlmostEqual(cos_similarity, 1.0, places=8)


# ─────────────────────────────────────────────────────────────────────────
# 4. RZF beamforming
# ─────────────────────────────────────────────────────────────────────────

class TestRZFBeamforming(unittest.TestCase):

    def test_rzf_dimensions(self):
        ap_positions, user_positions = _small_topology(num_aps=6, num_users=3)
        H = generate_physical_channel(ap_positions, user_positions, num_antennas=4, seed=0)
        x = _default_association(ap_positions, user_positions)
        W = rzf_weights(H, x)
        self.assertEqual(W.shape, H.shape)

    def test_rzf_zero_for_unassociated_pairs(self):
        ap_positions, user_positions = _small_topology(num_aps=6, num_users=3)
        H = generate_physical_channel(ap_positions, user_positions, num_antennas=4, seed=0)
        x = _default_association(ap_positions, user_positions)
        W = rzf_weights(H, x)

        unassociated = x == 0
        self.assertTrue(np.all(W[unassociated] == 0))

    def test_rzf_zero_for_ap_with_no_users(self):
        ap_positions, user_positions = _small_topology(num_aps=4, num_users=2)
        H = generate_physical_channel(ap_positions, user_positions, num_antennas=4, seed=0)
        x = np.zeros((4, 2), dtype=int)
        x[0, 0] = 1  # only AP 0 serves anyone
        W = rzf_weights(H, x)
        self.assertTrue(np.all(W[1:] == 0))

    def test_rzf_reduces_interference_relative_to_mrt(self):
        """
        When one AP jointly serves multiple users with well-separated
        angles, RZF should produce lower cross-user leakage (from that
        AP) than MRT. This is the textbook justification for RZF/ZF over
        matched-filtering in the multi-user, high-array-gain regime.
        """
        N = 16
        ap_positions = np.array([[0.0, 0.0]])
        # Users at well-separated angles from the single AP.
        user_positions = np.array([[50.0, 0.0], [0.0, 50.0], [-50.0, 5.0]])
        H = generate_physical_channel(
            ap_positions, user_positions, num_antennas=N,
            small_scale_fading=False,
        )
        x = np.ones((1, 3), dtype=int)

        W_mrt = mrt_weights(H, x)
        W_rzf = rzf_weights(H, x)

        def total_leakage(W):
            leakage = 0.0
            for k in range(3):
                for l in range(3):
                    if k == l:
                        continue
                    leakage += np.abs(np.vdot(H[0, k, :], W[0, l, :])) ** 2
            return leakage

        self.assertLess(total_leakage(W_rzf), total_leakage(W_mrt))


# ─────────────────────────────────────────────────────────────────────────
# 5. Per-AP power normalization
# ─────────────────────────────────────────────────────────────────────────

class TestPowerNormalization(unittest.TestCase):

    def test_per_ap_power_constraint_active_aps(self):
        ap_positions, user_positions = _small_topology(num_aps=6, num_users=3)
        H = generate_physical_channel(ap_positions, user_positions, num_antennas=4, seed=0)
        x = _default_association(ap_positions, user_positions)
        W = mrt_weights(H, x)

        p_max = 2.5
        W_scaled, ap_power_used = normalize_per_ap_power(W, p_max)

        active_aps = np.sum(x, axis=1) > 0
        np.testing.assert_allclose(ap_power_used[active_aps], p_max, atol=1e-9)

    def test_per_ap_power_zero_for_inactive_aps(self):
        ap_positions, user_positions = _small_topology(num_aps=6, num_users=3)
        H = generate_physical_channel(ap_positions, user_positions, num_antennas=4, seed=0)
        x = _default_association(ap_positions, user_positions, aps_per_user=1)
        W = rzf_weights(H, x)

        W_scaled, ap_power_used = normalize_per_ap_power(W, 1.0)

        inactive_aps = np.sum(x, axis=1) == 0
        if np.any(inactive_aps):
            np.testing.assert_allclose(ap_power_used[inactive_aps], 0.0, atol=1e-12)

    def test_per_ap_power_never_exceeds_budget(self):
        ap_positions, user_positions = _small_topology(num_aps=8, num_users=5)
        H = generate_physical_channel(ap_positions, user_positions, num_antennas=6, seed=3)
        x = _default_association(ap_positions, user_positions, aps_per_user=3)
        W = rzf_weights(H, x)

        p_max = 1.0
        _, ap_power_used = normalize_per_ap_power(W, p_max)
        self.assertTrue(np.all(ap_power_used <= p_max + 1e-9))


# ─────────────────────────────────────────────────────────────────────────
# 6. Physical-layer SINR and rate
# ─────────────────────────────────────────────────────────────────────────

class TestPhysicalSINRAndRate(unittest.TestCase):

    def _built_system(self, precoder="mrt", num_aps=6, num_users=3, aps_per_user=2, N=4, seed=0):
        ap_positions, user_positions = _small_topology(num_aps=num_aps, num_users=num_users)
        H = generate_physical_channel(ap_positions, user_positions, num_antennas=N, seed=seed)
        x = _default_association(ap_positions, user_positions, aps_per_user=aps_per_user)

        if precoder == "mrt":
            W = mrt_weights(H, x)
        else:
            W = rzf_weights(H, x)

        W, _ = normalize_per_ap_power(W, tx_power_per_ap=1.0)
        return H, W, x

    def test_finite_sinr_mrt(self):
        H, W, x = self._built_system(precoder="mrt")
        sinr = physical_sinr(H, W, noise_power=1e-9)
        self.assertEqual(sinr.shape, (3,))
        self.assertTrue(np.all(np.isfinite(sinr)))
        self.assertTrue(np.all(sinr >= 0))

    def test_finite_sinr_rzf(self):
        H, W, x = self._built_system(precoder="rzf")
        sinr = physical_sinr(H, W, noise_power=1e-9)
        self.assertTrue(np.all(np.isfinite(sinr)))
        self.assertTrue(np.all(sinr >= 0))

    def test_finite_rate(self):
        H, W, x = self._built_system(precoder="mrt")
        sinr = physical_sinr(H, W, noise_power=1e-9)
        rate = physical_rate(sinr, bandwidth_hz=20e6, pilot_fraction=0.1)
        self.assertEqual(rate.shape, sinr.shape)
        self.assertTrue(np.all(np.isfinite(rate)))
        self.assertTrue(np.all(rate >= 0))

    def test_empty_association_gives_zero_finite_sinr_and_rate(self):
        ap_positions, user_positions = _small_topology(num_aps=4, num_users=3)
        H = generate_physical_channel(ap_positions, user_positions, num_antennas=4, seed=0)
        x_empty = np.zeros((4, 3), dtype=int)

        W = mrt_weights(H, x_empty)
        W, ap_power_used = normalize_per_ap_power(W, tx_power_per_ap=1.0)

        # No AP transmits anything.
        np.testing.assert_allclose(ap_power_used, 0.0, atol=1e-12)

        sinr = physical_sinr(H, W, noise_power=1e-9)
        rate = physical_rate(sinr)

        np.testing.assert_allclose(sinr, 0.0, atol=1e-12)
        np.testing.assert_allclose(rate, 0.0, atol=1e-9)
        self.assertTrue(np.all(np.isfinite(sinr)))
        self.assertTrue(np.all(np.isfinite(rate)))

    def test_single_user_single_ap_no_interference(self):
        """With exactly one user and one serving AP, interference must be
        exactly zero and SINR must equal signal_power / noise_power."""
        ap_positions = np.array([[0.0, 0.0]])
        user_positions = np.array([[30.0, 0.0]])
        H = generate_physical_channel(
            ap_positions, user_positions, num_antennas=4,
            small_scale_fading=False,
        )
        x = np.ones((1, 1), dtype=int)
        W = mrt_weights(H, x)
        W, _ = normalize_per_ap_power(W, tx_power_per_ap=1.0)

        sinr = physical_sinr(H, W, noise_power=1e-9)
        expected_signal = np.abs(np.vdot(H[0, 0, :], W[0, 0, :])) ** 2
        expected_sinr = expected_signal / 1e-9

        np.testing.assert_allclose(sinr, [expected_sinr], rtol=1e-6)

    def test_rejects_zero_or_negative_noise_power(self):
        H, W, x = self._built_system(precoder="mrt")
        with self.assertRaises(ValueError):
            physical_sinr(H, W, noise_power=0.0)


# ─────────────────────────────────────────────────────────────────────────
# 7. Backward compatibility: old scalar path must be completely unaffected
# ─────────────────────────────────────────────────────────────────────────

class TestScalarPathUnaffected(unittest.TestCase):

    def test_scalar_functions_still_importable_and_unchanged_signature(self):
        ap_positions, user_positions = _small_topology(num_aps=5, num_users=3)

        G_comm = generate_comm_gains(ap_positions, user_positions)
        G_comm2, S_mat = generate_communication_features(ap_positions, user_positions)

        np.testing.assert_array_equal(G_comm, G_comm2)
        self.assertEqual(S_mat.shape, (5, 3, 3))

        sinr = communication_sinr(signal_power=1.0, interference_power=0.5)
        rate = communication_rate(sinr)
        self.assertTrue(np.isfinite(sinr))
        self.assertTrue(np.isfinite(rate))


if __name__ == "__main__":
    unittest.main()
