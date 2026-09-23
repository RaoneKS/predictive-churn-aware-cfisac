"""
tests/test_phase7_uncertainty.py

Phase 7 -- formal test suite for src.optimization.uncertainty
(measured_cv_residuals, generate_scenarios).

This module was documented as implemented ("Phase 7 uncertainty: 36/36
PASS") but no corresponding test file shipped with it; this suite locks
down its documented contract (module docstring of
src/optimization/uncertainty.py) so src/optimization/risk_aware.py can
build on it safely.
"""

import unittest

import numpy as np

from src.optimization.uncertainty import (
    generate_scenarios,
    measured_cv_residuals,
)


# ===========================================================================
# measured_cv_residuals -- input validation
# ===========================================================================

class TestResidualsValidation(unittest.TestCase):
    def test_rejects_non_integer_horizon(self):
        with self.assertRaises(TypeError):
            measured_cv_residuals([np.zeros((3, 2))] * 5, horizon=2.5)

    def test_rejects_zero_horizon(self):
        with self.assertRaises(ValueError):
            measured_cv_residuals([np.zeros((3, 2))] * 5, horizon=0)

    def test_rejects_negative_horizon(self):
        with self.assertRaises(ValueError):
            measured_cv_residuals([np.zeros((3, 2))] * 5, horizon=-3)

    def test_rejects_zero_dt(self):
        with self.assertRaises(ValueError):
            measured_cv_residuals([np.zeros((3, 2))] * 5, horizon=1, dt=0.0)

    def test_rejects_negative_dt(self):
        with self.assertRaises(ValueError):
            measured_cv_residuals([np.zeros((3, 2))] * 5, horizon=1, dt=-1.0)

    def test_rejects_non_finite_dt(self):
        with self.assertRaises(ValueError):
            measured_cv_residuals([np.zeros((3, 2))] * 5, horizon=1, dt=np.inf)

    def test_rejects_wrong_shape_entry(self):
        series = [np.zeros((3, 2))] * 4 + [np.zeros((3, 3))]
        with self.assertRaises(ValueError):
            measured_cv_residuals(series, horizon=2)

    def test_rejects_non_finite_positions(self):
        series = [np.zeros((3, 2))] * 4 + [np.full((3, 2), np.nan)]
        with self.assertRaises(ValueError):
            measured_cv_residuals(series, horizon=2)

    def test_rejects_inconsistent_object_count(self):
        series = [np.zeros((3, 2)), np.zeros((4, 2)), np.zeros((3, 2))]
        with self.assertRaises(ValueError):
            measured_cv_residuals(series, horizon=1)


# ===========================================================================
# measured_cv_residuals -- edge cases (E2 / epoch-0 style empty history)
# ===========================================================================

class TestResidualsEmptyHistory(unittest.TestCase):
    def test_empty_series_returns_zero_shaped_array(self):
        out = measured_cv_residuals([], horizon=3)
        self.assertEqual(out.shape, (0, 0, 3, 2))

    def test_too_short_series_returns_zero_anchors(self):
        # T=2 gives n_anchors = T - horizon - 1 = 2 - 3 - 1 = -2 <= 0
        series = [np.zeros((4, 2)), np.ones((4, 2))]
        out = measured_cv_residuals(series, horizon=3)
        self.assertEqual(out.shape, (0, 4, 3, 2))

    def test_exact_boundary_series_returns_zero_anchors(self):
        # T = horizon + 1 gives n_anchors = 0 exactly.
        horizon = 4
        T = horizon + 1
        series = [np.zeros((2, 2)) for _ in range(T)]
        out = measured_cv_residuals(series, horizon=horizon)
        self.assertEqual(out.shape[0], 0)
        self.assertEqual(out.shape[1], 2)

    def test_one_more_than_boundary_gives_one_anchor(self):
        horizon = 4
        T = horizon + 2
        series = [np.random.default_rng(0).normal(size=(2, 2)) for _ in range(T)]
        out = measured_cv_residuals(series, horizon=horizon)
        self.assertEqual(out.shape[0], 1)


# ===========================================================================
# measured_cv_residuals -- correctness against hand-computed values
# ===========================================================================

class TestResidualsCorrectness(unittest.TestCase):
    def test_zero_residual_for_perfect_constant_velocity(self):
        # A single object moving at exact constant velocity: CV extrapolation
        # is exact, so every residual must be exactly zero.
        N, H, T = 1, 3, 6
        v = np.array([[1.0, -0.5]])
        series = [np.array([[0.0, 0.0]]) + t * v for t in range(T)]
        out = measured_cv_residuals(series, horizon=H, dt=1.0)
        self.assertTrue(np.allclose(out, 0.0))

    def test_nonzero_residual_for_accelerating_object(self):
        # Constant acceleration: CV extrapolation (based on the last step's
        # velocity) must NOT be exact for h >= 2.
        N, H, T = 1, 2, 5
        series = []
        pos = np.array([[0.0, 0.0]])
        vel = np.array([[1.0, 0.0]])
        accel = np.array([[0.5, 0.0]])
        for t in range(T):
            series.append(pos.copy())
            pos = pos + vel
            vel = vel + accel
        out = measured_cv_residuals(series, horizon=H, dt=1.0)
        self.assertFalse(np.allclose(out, 0.0))

    def test_hand_computed_single_anchor_single_step(self):
        # T=3, horizon=1 -> exactly one anchor at t0=1.
        p0 = np.array([[0.0, 0.0]])
        p1 = np.array([[2.0, 0.0]])
        p2 = np.array([[5.0, 1.0]])
        series = [p0, p1, p2]
        out = measured_cv_residuals(series, horizon=1, dt=1.0)
        # v_hat(1) = p1 - p0 = (2,0); cv_pred = p1 + 1*v_hat = (4,0)
        # residual = p2 - cv_pred = (1,1)
        self.assertEqual(out.shape, (1, 1, 1, 2))
        np.testing.assert_allclose(out[0, 0, 0, :], [1.0, 1.0])

    def test_multiple_objects_independent_residuals(self):
        p0 = np.array([[0.0, 0.0], [10.0, 10.0]])
        p1 = np.array([[1.0, 0.0], [10.0, 9.0]])
        p2 = np.array([[2.0, 0.0], [10.0, 6.0]])  # obj0 CV-consistent, obj1 not
        series = [p0, p1, p2]
        out = measured_cv_residuals(series, horizon=1, dt=1.0)
        np.testing.assert_allclose(out[0, 0, 0, :], [0.0, 0.0])
        self.assertFalse(np.allclose(out[0, 1, 0, :], [0.0, 0.0]))

    def test_dt_cancels_exactly_by_construction(self):
        # cv_pred(t0,h) = pos[t0] + h*v_hat*dt, v_hat = (pos[t0]-pos[t0-1])/dt
        # -> the dt factors cancel exactly, so residuals are dt-invariant.
        # This is a documented structural property of the CV extrapolation
        # (not a bug): it locks the behavior down explicitly.
        p0 = np.array([[0.0, 0.0]])
        p1 = np.array([[2.0, 0.0]])
        p2 = np.array([[5.0, 1.0]])
        series = [p0, p1, p2]
        out_dt1 = measured_cv_residuals(series, horizon=1, dt=1.0)
        out_dt2 = measured_cv_residuals(series, horizon=1, dt=3.5)
        np.testing.assert_allclose(out_dt1, out_dt2)


# ===========================================================================
# measured_cv_residuals -- determinism
# ===========================================================================

class TestResidualsDeterminism(unittest.TestCase):
    def test_repeated_calls_identical(self):
        rng = np.random.default_rng(0)
        series = [rng.normal(size=(4, 2)) for _ in range(10)]
        out1 = measured_cv_residuals(series, horizon=3)
        out2 = measured_cv_residuals(series, horizon=3)
        np.testing.assert_array_equal(out1, out2)

    def test_does_not_mutate_input_series(self):
        rng = np.random.default_rng(1)
        series = [rng.normal(size=(3, 2)) for _ in range(8)]
        originals = [s.copy() for s in series]
        measured_cv_residuals(series, horizon=2)
        for a, b in zip(series, originals):
            np.testing.assert_array_equal(a, b)


# ===========================================================================
# generate_scenarios -- input validation
# ===========================================================================

class TestScenariosValidation(unittest.TestCase):
    def test_rejects_wrong_ndim(self):
        with self.assertRaises(ValueError):
            generate_scenarios(np.zeros((5, 2, 2)), 3, seed=0)

    def test_rejects_wrong_last_dim(self):
        with self.assertRaises(ValueError):
            generate_scenarios(np.zeros((5, 3, 2, 3)), 3, seed=0)

    def test_rejects_non_finite_residuals(self):
        bad = np.zeros((2, 3, 2, 2))
        bad[0, 0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            generate_scenarios(bad, 3, seed=0)

    def test_rejects_non_integer_num_scenarios(self):
        with self.assertRaises(TypeError):
            generate_scenarios(np.zeros((5, 3, 2, 2)), 2.5, seed=0)

    def test_rejects_negative_num_scenarios(self):
        with self.assertRaises(ValueError):
            generate_scenarios(np.zeros((5, 3, 2, 2)), -1, seed=0)


# ===========================================================================
# generate_scenarios -- core behavior
# ===========================================================================

class TestScenariosCoreBehavior(unittest.TestCase):
    def test_output_shape(self):
        residuals = np.random.default_rng(0).normal(size=(7, 4, 3, 2))
        out = generate_scenarios(residuals, 10, seed=0)
        self.assertEqual(out.shape, (10, 4, 3, 2))

    def test_zero_num_scenarios_returns_empty_batch_with_correct_pool_shape(self):
        residuals = np.random.default_rng(0).normal(size=(7, 4, 3, 2))
        out = generate_scenarios(residuals, 0, seed=0)
        self.assertEqual(out.shape, (0, 4, 3, 2))

    def test_empty_pool_returns_zero_perturbations(self):
        residuals = np.zeros((0, 4, 3, 2))
        out = generate_scenarios(residuals, 5, seed=0)
        self.assertEqual(out.shape, (5, 4, 3, 2))
        self.assertTrue(np.allclose(out, 0.0))

    def test_every_scenario_drawn_from_the_residual_pool(self):
        residuals = np.random.default_rng(2).normal(size=(5, 3, 2, 2))
        out = generate_scenarios(residuals, 20, seed=0)
        pool = {tuple(r.ravel()) for r in residuals}
        for scenario in out:
            self.assertIn(tuple(scenario.ravel()), pool)

    def test_single_residual_pool_entry_collapses_every_scenario(self):
        residuals = np.random.default_rng(3).normal(size=(1, 3, 2, 2))
        out = generate_scenarios(residuals, 15, seed=0)
        for scenario in out:
            np.testing.assert_allclose(scenario, residuals[0])


# ===========================================================================
# generate_scenarios -- determinism / seeding
# ===========================================================================

class TestScenariosDeterminism(unittest.TestCase):
    def test_same_seed_identical_output(self):
        residuals = np.random.default_rng(5).normal(size=(9, 3, 2, 2))
        out1 = generate_scenarios(residuals, 25, seed=42)
        out2 = generate_scenarios(residuals, 25, seed=42)
        np.testing.assert_array_equal(out1, out2)

    def test_different_seeds_can_differ(self):
        residuals = np.random.default_rng(6).normal(size=(9, 3, 2, 2))
        out1 = generate_scenarios(residuals, 25, seed=1)
        out2 = generate_scenarios(residuals, 25, seed=2)
        self.assertFalse(np.array_equal(out1, out2))

    def test_does_not_mutate_input_residuals(self):
        residuals = np.random.default_rng(7).normal(size=(6, 3, 2, 2))
        original = residuals.copy()
        generate_scenarios(residuals, 12, seed=0)
        np.testing.assert_array_equal(residuals, original)

    def test_output_is_a_copy_not_a_view(self):
        residuals = np.random.default_rng(8).normal(size=(6, 3, 2, 2))
        out = generate_scenarios(residuals, 4, seed=0)
        out[0, 0, 0, 0] = 999.0
        self.assertFalse(np.any(residuals == 999.0))


# ===========================================================================
# Integration: residuals -> scenarios roundtrip plausibility
# ===========================================================================

class TestResidualsToScenariosIntegration(unittest.TestCase):
    def _random_trajectory(self, num_objects, T, seed):
        rng = np.random.default_rng(seed)
        pos = rng.uniform(0, 50, size=(num_objects, 2))
        vel = rng.uniform(-2, 2, size=(num_objects, 2))
        series = []
        for _ in range(T):
            series.append(pos.copy())
            pos = pos + vel + rng.normal(scale=0.3, size=(num_objects, 2))
        return series

    def test_generated_scenarios_reproduce_measured_residual_scale(self):
        series = self._random_trajectory(4, 30, seed=10)
        residuals = measured_cv_residuals(series, horizon=5)
        scenarios = generate_scenarios(residuals, 200, seed=0)
        # Every generated scenario perturbation must come from the measured
        # pool (with replacement) so its magnitude cannot exceed the pool's
        # maximum magnitude.
        self.assertLessEqual(
            np.abs(scenarios).max() + 1e-9,
            np.abs(residuals).max() + 1e-9,
        )

    def test_num_scenarios_much_larger_than_pool_still_well_defined(self):
        series = self._random_trajectory(3, 12, seed=11)
        residuals = measured_cv_residuals(series, horizon=4)
        self.assertGreater(residuals.shape[0], 0)
        out = generate_scenarios(residuals, 500, seed=0)
        self.assertEqual(out.shape[0], 500)

    def test_full_pipeline_zero_anchors_to_zero_scenarios(self):
        # Deliberately too-short history end-to-end.
        series = self._random_trajectory(3, 2, seed=12)
        residuals = measured_cv_residuals(series, horizon=5)
        self.assertEqual(residuals.shape[0], 0)
        out = generate_scenarios(residuals, 30, seed=0)
        self.assertTrue(np.allclose(out, 0.0))

    def test_longer_history_yields_more_anchors(self):
        short = self._random_trajectory(3, 10, seed=13)
        long_ = self._random_trajectory(3, 30, seed=13)
        r_short = measured_cv_residuals(short, horizon=4)
        r_long = measured_cv_residuals(long_, horizon=4)
        self.assertGreaterEqual(r_long.shape[0], r_short.shape[0])

    def test_pipeline_deterministic_end_to_end(self):
        series = self._random_trajectory(5, 20, seed=14)
        r1 = measured_cv_residuals(series, horizon=4)
        r2 = measured_cv_residuals(series, horizon=4)
        s1 = generate_scenarios(r1, 40, seed=99)
        s2 = generate_scenarios(r2, 40, seed=99)
        np.testing.assert_array_equal(s1, s2)


# ===========================================================================
# Additional boundary / robustness coverage (rounds out to 36)
# ===========================================================================

class TestAdditionalBoundaries(unittest.TestCase):
    def test_horizon_one_is_valid(self):
        series = [np.zeros((2, 2)), np.ones((2, 2)), 2 * np.ones((2, 2))]
        out = measured_cv_residuals(series, horizon=1)
        self.assertEqual(out.shape[2], 1)

    def test_large_horizon_relative_to_series_gives_zero_anchors(self):
        series = [np.zeros((2, 2))] * 3
        out = measured_cv_residuals(series, horizon=100)
        self.assertEqual(out.shape[0], 0)

    def test_generate_scenarios_num_scenarios_one(self):
        residuals = np.random.default_rng(20).normal(size=(4, 2, 3, 2))
        out = generate_scenarios(residuals, 1, seed=0)
        self.assertEqual(out.shape[0], 1)

    def test_residuals_accept_list_of_lists_like_arrays(self):
        # position_series entries need not already be ndarray.
        series = [[[0.0, 0.0], [1.0, 1.0]], [[1.0, 0.0], [2.0, 1.0]],
                  [[2.0, 0.0], [3.0, 1.0]]]
        out = measured_cv_residuals(series, horizon=1)
        self.assertEqual(out.shape, (1, 2, 1, 2))

    def test_generate_scenarios_accepts_list_input(self):
        residuals = [[[[0.1, 0.2]], [[0.3, 0.4]]]]  # shape (1,2,1,2) via list
        out = generate_scenarios(residuals, 3, seed=0)
        self.assertEqual(out.shape, (3, 2, 1, 2))

    def test_seed_zero_is_valid_and_deterministic(self):
        residuals = np.random.default_rng(21).normal(size=(6, 2, 2, 2))
        out1 = generate_scenarios(residuals, 10, seed=0)
        out2 = generate_scenarios(residuals, 10, seed=0)
        np.testing.assert_array_equal(out1, out2)


if __name__ == "__main__":
    unittest.main()
