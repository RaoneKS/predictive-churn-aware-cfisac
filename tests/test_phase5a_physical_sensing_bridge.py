"""
tests/test_phase5a_physical_sensing_bridge.py

Focused tests for Phase 5A: the physical communication <-> Fisher-
information sensing bridge (src.metrics.physical_sensing).

These tests exercise ONLY the new bridge module. They do not modify or
depend on prediction, clustering, or churn logic, and only import the
existing scalar/Phase-4A/4B paths read-only, for regression/comparison
checks (see TestScalarRegressionCompatibility and
TestPhase4ABBehaviorUnchanged).
"""

import unittest

import numpy as np

from src.metrics.physical_sensing import (
    comm_power_per_ap,
    isotropic_sensing_covariance,
    sensing_power_from_covariance,
    sensing_snr,
    snr_to_measurement_variance,
    aggregate_fisher_information_bridge,
    physical_tx_power_per_ap,
    check_power_constraint,
    evaluate_physical_sensing_bridge,
)
from src.metrics.sensing import (
    aggregate_fisher_information,
    posterior_covariance,
    information_gain,
    tracking_error,
)
from src.channels.sensing import generate_sensing_gains
from src.channels.physical import generate_physical_channel
from src.channels.communication import generate_comm_gains
from src.beamforming.precoders import mrt_weights, rzf_weights, normalize_per_ap_power
from src.beamforming.constrained import constrained_sinr_beamforming
from src.clustering.overlapping import communication_clustering, sensing_clustering
from src.environment.topology import generate_ap_positions, generate_positions


def _topology(num_aps=8, num_users=4, num_targets=2, num_antennas=4,
              area_size=200.0, seed=42):
    ap_positions = generate_ap_positions(num_aps, area_size, seed=seed)
    user_positions = generate_positions(num_users, area_size, seed=seed + 1)
    target_positions = generate_positions(num_targets, area_size, seed=seed + 2)

    G_comm = generate_comm_gains(ap_positions, user_positions)
    G_sens = generate_sensing_gains(ap_positions, target_positions)

    x = communication_clustering(G_comm, num_users=num_users, aps_per_user=3)
    y_tx, y_rx = sensing_clustering(
        G_sens, num_targets=num_targets, tx_aps_per_target=2, rx_aps_per_target=2
    )

    H = generate_physical_channel(
        ap_positions, user_positions, num_antennas=num_antennas,
        rician_k_factor=10.0, small_scale_fading=True, seed=seed,
    )
    W = mrt_weights(H, x)
    W, _ = normalize_per_ap_power(W, 1.0)

    return {
        "ap_positions": ap_positions,
        "user_positions": user_positions,
        "target_positions": target_positions,
        "G_comm": G_comm,
        "G_sens": G_sens,
        "x": x,
        "y_tx": y_tx,
        "y_rx": y_rx,
        "H": H,
        "W": W,
        "M": num_aps,
        "K": num_users,
        "Q": num_targets,
        "N": num_antennas,
    }


# ─────────────────────────────────────────────────────────────────────────
# 1. Shape correctness
# ─────────────────────────────────────────────────────────────────────────
class TestShapes(unittest.TestCase):
    def test_comm_power_per_ap_shape(self):
        t = _topology()
        p = comm_power_per_ap(t["W"])
        self.assertEqual(p.shape, (t["M"],))

    def test_isotropic_covariance_shape(self):
        t = _topology()
        p_sens = np.ones(t["M"])
        S = isotropic_sensing_covariance(p_sens, t["N"])
        self.assertEqual(S.shape, (t["M"], t["N"], t["N"]))

    def test_sensing_snr_shape(self):
        t = _topology()
        p_sens = np.ones(t["M"])
        snr = sensing_snr(p_sens, t["G_sens"], t["y_tx"], t["y_rx"])
        self.assertEqual(snr.shape, (t["M"], t["M"], t["Q"]))

    def test_fisher_information_bridge_shape(self):
        t = _topology()
        p_sens = np.ones(t["M"])
        J = aggregate_fisher_information_bridge(
            t["ap_positions"], t["target_positions"], t["y_tx"], t["y_rx"],
            p_sens, t["G_sens"],
        )
        self.assertEqual(J.shape, (t["Q"], 2, 2))

    def test_full_bridge_output_shapes(self):
        t = _topology()
        p_sens = np.ones(t["M"])
        out = evaluate_physical_sensing_bridge(
            t["ap_positions"], t["target_positions"], t["y_tx"], t["y_rx"],
            t["W"], p_sens, t["G_sens"], tx_power_per_ap_max=1.0,
        )
        self.assertEqual(out["J_total"].shape, (t["Q"], 2, 2))
        self.assertEqual(len(out["posterior_covariances"]), t["Q"])
        for pc in out["posterior_covariances"]:
            self.assertEqual(pc.shape, (2, 2))
        self.assertEqual(out["information_gain"].shape, (t["Q"],))
        self.assertEqual(out["tracking_error"].shape, (t["Q"],))
        self.assertEqual(out["P_comm"].shape, (t["M"],))
        self.assertEqual(out["P_sens"].shape, (t["M"],))
        self.assertEqual(out["P_tx"].shape, (t["M"],))
        self.assertEqual(out["power_feasible"].shape, (t["M"],))
        self.assertEqual(out["power_violation"].shape, (t["M"],))


# ─────────────────────────────────────────────────────────────────────────
# 2. Deterministic seeded execution
# ─────────────────────────────────────────────────────────────────────────
class TestDeterminism(unittest.TestCase):
    def test_repeated_calls_bit_identical(self):
        t = _topology(seed=42)
        p_sens = np.full(t["M"], 0.3)

        out1 = evaluate_physical_sensing_bridge(
            t["ap_positions"], t["target_positions"], t["y_tx"], t["y_rx"],
            t["W"], p_sens, t["G_sens"], tx_power_per_ap_max=1.0,
        )
        out2 = evaluate_physical_sensing_bridge(
            t["ap_positions"], t["target_positions"], t["y_tx"], t["y_rx"],
            t["W"], p_sens, t["G_sens"], tx_power_per_ap_max=1.0,
        )
        np.testing.assert_array_equal(out1["J_total"], out2["J_total"])
        np.testing.assert_array_equal(out1["information_gain"], out2["information_gain"])
        np.testing.assert_array_equal(out1["P_tx"], out2["P_tx"])

    def test_same_seed_same_topology(self):
        t1 = _topology(seed=42)
        t2 = _topology(seed=42)
        np.testing.assert_array_equal(t1["ap_positions"], t2["ap_positions"])
        np.testing.assert_array_equal(t1["G_sens"], t2["G_sens"])
        np.testing.assert_array_equal(np.abs(t1["H"]), np.abs(t2["H"]))


# ─────────────────────────────────────────────────────────────────────────
# 3. Zero sensing assignment gives zero sensing contribution
# ─────────────────────────────────────────────────────────────────────────
class TestZeroSensingAssignment(unittest.TestCase):
    def test_no_tx_or_rx_assigned_gives_zero_fim(self):
        t = _topology()
        y_tx_empty = np.zeros_like(t["y_tx"])
        y_rx_empty = np.zeros_like(t["y_rx"])
        p_sens = np.ones(t["M"])

        J = aggregate_fisher_information_bridge(
            t["ap_positions"], t["target_positions"], y_tx_empty, y_rx_empty,
            p_sens, t["G_sens"],
        )
        np.testing.assert_array_equal(J, np.zeros((t["Q"], 2, 2)))

    def test_no_assignment_gives_zero_information_gain(self):
        t = _topology()
        y_tx_empty = np.zeros_like(t["y_tx"])
        y_rx_empty = np.zeros_like(t["y_rx"])
        p_sens = np.ones(t["M"])

        out = evaluate_physical_sensing_bridge(
            t["ap_positions"], t["target_positions"], y_tx_empty, y_rx_empty,
            t["W"], p_sens, t["G_sens"], tx_power_per_ap_max=1.0,
        )
        np.testing.assert_allclose(out["information_gain"], np.zeros(t["Q"]), atol=1e-12)


# ─────────────────────────────────────────────────────────────────────────
# 4. Zero sensing power behaves correctly
# ─────────────────────────────────────────────────────────────────────────
class TestZeroSensingPower(unittest.TestCase):
    def test_zero_power_everywhere_gives_zero_snr(self):
        t = _topology()
        snr = sensing_snr(np.zeros(t["M"]), t["G_sens"], t["y_tx"], t["y_rx"])
        np.testing.assert_array_equal(snr, np.zeros_like(snr))

    def test_zero_power_gives_exact_zero_fim_even_with_active_assignment(self):
        t = _topology()
        # y_tx/y_rx have real (nonzero) assignments here, but power is 0.
        J = aggregate_fisher_information_bridge(
            t["ap_positions"], t["target_positions"], t["y_tx"], t["y_rx"],
            np.zeros(t["M"]), t["G_sens"],
        )
        np.testing.assert_array_equal(J, np.zeros((t["Q"], 2, 2)))

    def test_zero_snr_maps_to_infinite_variance(self):
        snr = np.array([0.0, 1.0, 2.0])
        var = snr_to_measurement_variance(snr, reference_variance=1.0)
        self.assertTrue(np.isinf(var[0]))
        self.assertAlmostEqual(var[1], 1.0)
        self.assertAlmostEqual(var[2], 0.5)

    def test_zero_power_gives_zero_information_gain_end_to_end(self):
        t = _topology()
        out = evaluate_physical_sensing_bridge(
            t["ap_positions"], t["target_positions"], t["y_tx"], t["y_rx"],
            t["W"], np.zeros(t["M"]), t["G_sens"], tx_power_per_ap_max=1.0,
        )
        np.testing.assert_allclose(out["information_gain"], np.zeros(t["Q"]), atol=1e-12)


# ─────────────────────────────────────────────────────────────────────────
# 5. Sensing power changes sensing information (monotonicity)
# ─────────────────────────────────────────────────────────────────────────
class TestSensingPowerChangesInformation(unittest.TestCase):
    def test_higher_power_gives_higher_or_equal_information_gain(self):
        t = _topology()
        low = np.full(t["M"], 0.1)
        high = np.full(t["M"], 5.0)

        out_low = evaluate_physical_sensing_bridge(
            t["ap_positions"], t["target_positions"], t["y_tx"], t["y_rx"],
            t["W"], low, t["G_sens"], tx_power_per_ap_max=10.0,
        )
        out_high = evaluate_physical_sensing_bridge(
            t["ap_positions"], t["target_positions"], t["y_tx"], t["y_rx"],
            t["W"], high, t["G_sens"], tx_power_per_ap_max=10.0,
        )
        for q in range(t["Q"]):
            self.assertGreaterEqual(
                out_high["information_gain"][q] + 1e-9,
                out_low["information_gain"][q],
            )

    def test_higher_power_gives_lower_or_equal_tracking_error(self):
        t = _topology()
        low = np.full(t["M"], 0.1)
        high = np.full(t["M"], 5.0)

        out_low = evaluate_physical_sensing_bridge(
            t["ap_positions"], t["target_positions"], t["y_tx"], t["y_rx"],
            t["W"], low, t["G_sens"], tx_power_per_ap_max=10.0,
        )
        out_high = evaluate_physical_sensing_bridge(
            t["ap_positions"], t["target_positions"], t["y_tx"], t["y_rx"],
            t["W"], high, t["G_sens"], tx_power_per_ap_max=10.0,
        )
        for q in range(t["Q"]):
            self.assertLessEqual(
                out_high["tracking_error"][q] - 1e-9,
                out_low["tracking_error"][q],
            )


# ─────────────────────────────────────────────────────────────────────────
# 6. Posterior covariance remains valid (positive-definite, symmetric)
# ─────────────────────────────────────────────────────────────────────────
class TestPosteriorCovarianceValidity(unittest.TestCase):
    def test_posterior_covariance_is_positive_definite_and_symmetric(self):
        t = _topology()
        p_sens = np.full(t["M"], 0.7)
        out = evaluate_physical_sensing_bridge(
            t["ap_positions"], t["target_positions"], t["y_tx"], t["y_rx"],
            t["W"], p_sens, t["G_sens"], tx_power_per_ap_max=1.0,
        )
        for pc in out["posterior_covariances"]:
            np.testing.assert_allclose(pc, pc.T, atol=1e-10)
            eigvals = np.linalg.eigvalsh(pc)
            self.assertTrue(np.all(eigvals > 0))

    def test_posterior_covariance_shrinks_relative_to_prior(self):
        # More information can only shrink (or keep equal) the posterior
        # relative to the prior, in the Loewner order sense of trace.
        t = _topology()
        p_sens = np.full(t["M"], 1.0)
        prior = [np.eye(2) for _ in range(t["Q"])]
        out = evaluate_physical_sensing_bridge(
            t["ap_positions"], t["target_positions"], t["y_tx"], t["y_rx"],
            t["W"], p_sens, t["G_sens"], tx_power_per_ap_max=1.0,
            prior_covariances=prior,
        )
        for q in range(t["Q"]):
            self.assertLessEqual(
                np.trace(out["posterior_covariances"][q]), np.trace(prior[q]) + 1e-9
            )


# ─────────────────────────────────────────────────────────────────────────
# 7. Information gain remains finite/nonnegative
# ─────────────────────────────────────────────────────────────────────────
class TestInformationGainFiniteNonnegative(unittest.TestCase):
    def test_information_gain_finite_and_nonnegative(self):
        t = _topology()
        p_sens = np.full(t["M"], 2.0)
        out = evaluate_physical_sensing_bridge(
            t["ap_positions"], t["target_positions"], t["y_tx"], t["y_rx"],
            t["W"], p_sens, t["G_sens"], tx_power_per_ap_max=5.0,
        )
        self.assertTrue(np.all(np.isfinite(out["information_gain"])))
        self.assertTrue(np.all(out["information_gain"] >= -1e-9))

    def test_tracking_error_finite_and_nonnegative(self):
        t = _topology()
        p_sens = np.full(t["M"], 2.0)
        out = evaluate_physical_sensing_bridge(
            t["ap_positions"], t["target_positions"], t["y_tx"], t["y_rx"],
            t["W"], p_sens, t["G_sens"], tx_power_per_ap_max=5.0,
        )
        self.assertTrue(np.all(np.isfinite(out["tracking_error"])))
        self.assertTrue(np.all(out["tracking_error"] >= 0.0))


# ─────────────────────────────────────────────────────────────────────────
# 8. Scalar FIM regression compatibility
# ─────────────────────────────────────────────────────────────────────────
class TestScalarRegressionCompatibility(unittest.TestCase):
    def test_reference_variance_one_at_unit_snr_matches_scalar_default(self):
        """
        At SNR == 1.0 (achieved by choosing sensing power and noise such
        that p_sens,m * G_sens[m,n,q] == noise_power_sens for every
        active pair), the bridge's variance is EXACTLY 1.0, matching the
        pre-Phase-5A scalar aggregate_fisher_information's default
        measurement_noise=1.0 bit-for-bit.
        """
        t = _topology(num_aps=5, num_users=2, num_targets=1, seed=7)

        # Force SNR == 1.0 for every active (m, n, q): choose sensing
        # power per AP m such that p_sens,m * G_sens[m,n,q] == noise for
        # some fixed reference gain -- simplest is to directly synthesize
        # a tiny scenario with a single tx/rx pair.
        ap_positions = np.array([[0.0, 0.0], [10.0, 0.0]])
        target_positions = np.array([[5.0, 5.0]])
        G_sens = generate_sensing_gains(ap_positions, target_positions)
        y_tx = np.array([[1], [0]])
        y_rx = np.array([[0], [1]])

        noise_power_sens = 1e-6
        g = G_sens[0, 1, 0]
        p_sens = np.array([noise_power_sens / g, 0.0])  # forces SNR == 1.0

        J_bridge = aggregate_fisher_information_bridge(
            ap_positions, target_positions, y_tx, y_rx, p_sens, G_sens,
            noise_power_sens=noise_power_sens, reference_variance=1.0,
        )
        J_scalar = aggregate_fisher_information(
            ap_positions, target_positions, y_tx, y_rx, measurement_noise=1.0,
        )
        np.testing.assert_allclose(J_bridge, J_scalar, rtol=1e-8, atol=1e-12)

    def test_scalar_aggregate_fisher_information_untouched(self):
        """The old scalar function's own behavior/signature is unaffected
        by this module simply being imported and used."""
        t = _topology()
        J1 = aggregate_fisher_information(
            t["ap_positions"], t["target_positions"], t["y_tx"], t["y_rx"],
            measurement_noise=1.0,
        )
        # Import and exercise the bridge module in between.
        _ = aggregate_fisher_information_bridge(
            t["ap_positions"], t["target_positions"], t["y_tx"], t["y_rx"],
            np.ones(t["M"]), t["G_sens"],
        )
        J2 = aggregate_fisher_information(
            t["ap_positions"], t["target_positions"], t["y_tx"], t["y_rx"],
            measurement_noise=1.0,
        )
        np.testing.assert_array_equal(J1, J2)


# ─────────────────────────────────────────────────────────────────────────
# 9. Combined communication+sensing power accounting
# ─────────────────────────────────────────────────────────────────────────
class TestCombinedPowerAccounting(unittest.TestCase):
    def test_combined_power_equals_comm_plus_sensing(self):
        t = _topology()
        p_sens = np.full(t["M"], 0.2)
        P_comm = comm_power_per_ap(t["W"])
        P_tx = physical_tx_power_per_ap(t["W"], p_sens)
        np.testing.assert_allclose(P_tx, P_comm + p_sens, rtol=1e-10)

    def test_isotropic_covariance_trace_equals_sensing_power(self):
        t = _topology()
        p_sens = np.array([0.0, 0.3, 1.5] + [0.0] * (t["M"] - 3))
        S = isotropic_sensing_covariance(p_sens, t["N"])
        recovered = sensing_power_from_covariance(S)
        np.testing.assert_allclose(recovered, p_sens, rtol=1e-10)


# ─────────────────────────────────────────────────────────────────────────
# 10. No AP power violation (when within budget)
# ─────────────────────────────────────────────────────────────────────────
class TestNoPowerViolation(unittest.TestCase):
    def test_within_budget_is_feasible(self):
        t = _topology()
        # Comm power already normalized to 1.0 W/AP; keep sensing power
        # small enough that combined power stays within a generous budget.
        p_sens = np.full(t["M"], 0.01)
        P_tx = physical_tx_power_per_ap(t["W"], p_sens)
        feasible, violation = check_power_constraint(P_tx, P_max=2.0)
        self.assertTrue(np.all(feasible))
        np.testing.assert_allclose(violation, np.zeros(t["M"]), atol=1e-9)

    def test_over_budget_is_detected(self):
        t = _topology()
        p_sens = np.full(t["M"], 5.0)  # deliberately huge
        P_tx = physical_tx_power_per_ap(t["W"], p_sens)
        feasible, violation = check_power_constraint(P_tx, P_max=1.0)
        # Every AP with any sensing power now overshoots a 1.0 W budget.
        active = p_sens > 0
        self.assertTrue(np.all(~feasible[active]))
        self.assertTrue(np.all(violation[active] > 0))


# ─────────────────────────────────────────────────────────────────────────
# 11. Dual-role AP has exact comm + sensing accounting
# ─────────────────────────────────────────────────────────────────────────
class TestDualRoleAP(unittest.TestCase):
    def test_dual_role_ap_sums_exactly_no_double_count(self):
        t = _topology()

        # Find an AP that is BOTH a communication server (comm power > 0)
        # and a sensing TX (y_tx > 0) -- a genuine dual-role AP. If none
        # exists in this topology/seed, force one explicitly.
        P_comm = comm_power_per_ap(t["W"])
        is_sensing_tx = t["y_tx"].sum(axis=1) > 0
        dual_candidates = np.where((P_comm > 1e-9) & is_sensing_tx)[0]

        p_sens = np.zeros(t["M"])
        if len(dual_candidates) > 0:
            dual_ap = int(dual_candidates[0])
        else:
            # Force AP 0 into a dual role explicitly.
            dual_ap = 0
            t["y_tx"][dual_ap, 0] = 1
        p_sens[dual_ap] = 0.37

        P_tx = physical_tx_power_per_ap(t["W"], p_sens)

        expected = P_comm[dual_ap] + 0.37
        self.assertAlmostEqual(P_tx[dual_ap], expected, places=10)

        # Every other AP's power is exactly its own comm power (no
        # cross-contamination from the dual-role AP's sensing power).
        for m in range(t["M"]):
            if m == dual_ap:
                continue
            self.assertAlmostEqual(P_tx[m], P_comm[m] + p_sens[m], places=10)

    def test_dual_role_ap_fisher_information_uses_only_its_own_power(self):
        # A dual-role AP's sensing contribution should depend only on ITS
        # OWN sensing power, not on its (separate) communication power.
        ap_positions = np.array([[0.0, 0.0], [10.0, 0.0]])
        target_positions = np.array([[5.0, 5.0]])
        G_sens = generate_sensing_gains(ap_positions, target_positions)
        y_tx = np.array([[1], [0]])
        y_rx = np.array([[0], [1]])

        J_a = aggregate_fisher_information_bridge(
            ap_positions, target_positions, y_tx, y_rx,
            np.array([1.0, 0.0]), G_sens,
        )
        J_b = aggregate_fisher_information_bridge(
            ap_positions, target_positions, y_tx, y_rx,
            np.array([2.0, 0.0]), G_sens,
        )
        # Doubling AP 0's sensing power should change (increase) J.
        self.assertGreater(np.trace(J_b[0]), np.trace(J_a[0]))


# ─────────────────────────────────────────────────────────────────────────
# 12. Phase 4A/4B behavior remains unchanged
# ─────────────────────────────────────────────────────────────────────────
class TestPhase4ABBehaviorUnchanged(unittest.TestCase):
    def test_mrt_rzf_untouched_after_bridge_use(self):
        t = _topology()
        H, x = t["H"], t["x"]

        W_mrt_before = mrt_weights(H, x)
        W_rzf_before = rzf_weights(H, x)

        # Exercise the bridge module in between.
        p_sens = np.ones(t["M"])
        _ = evaluate_physical_sensing_bridge(
            t["ap_positions"], t["target_positions"], t["y_tx"], t["y_rx"],
            t["W"], p_sens, t["G_sens"], tx_power_per_ap_max=1.0,
        )

        W_mrt_after = mrt_weights(H, x)
        W_rzf_after = rzf_weights(H, x)

        np.testing.assert_array_equal(W_mrt_before, W_mrt_after)
        np.testing.assert_array_equal(W_rzf_before, W_rzf_after)

    def test_constrained_sinr_beamforming_untouched_after_bridge_use(self):
        t = _topology(num_aps=6, num_users=3)
        H = t["H"]

        W1, feasible1, sinr1, _ = constrained_sinr_beamforming(H, 0.5, 1.0)

        p_sens = np.ones(t["M"])
        _ = evaluate_physical_sensing_bridge(
            t["ap_positions"], t["target_positions"], t["y_tx"], t["y_rx"],
            t["W"], p_sens, t["G_sens"], tx_power_per_ap_max=1.0,
        )

        W2, feasible2, sinr2, _ = constrained_sinr_beamforming(H, 0.5, 1.0)

        np.testing.assert_array_equal(W1, W2)
        self.assertEqual(feasible1, feasible2)
        np.testing.assert_array_equal(sinr1, sinr2)

    def test_scalar_channel_path_untouched(self):
        t = _topology()
        G1 = generate_comm_gains(t["ap_positions"], t["user_positions"])

        p_sens = np.ones(t["M"])
        _ = evaluate_physical_sensing_bridge(
            t["ap_positions"], t["target_positions"], t["y_tx"], t["y_rx"],
            t["W"], p_sens, t["G_sens"], tx_power_per_ap_max=1.0,
        )

        G2 = generate_comm_gains(t["ap_positions"], t["user_positions"])
        np.testing.assert_array_equal(G1, G2)


if __name__ == "__main__":
    unittest.main()
