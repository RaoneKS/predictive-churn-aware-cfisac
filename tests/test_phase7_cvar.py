"""
tests/test_phase7_cvar.py

Focused tests for Phase 7: empirical VaR/CVaR (src.optimization.cvar).

These tests are self-contained -- they exercise cvar.py directly against
plain numpy arrays and do not import any Phase 1-6 module (no
MobilitySimulator, no predictor, no channel models), matching the fact
that cvar.py itself has no such dependency.
"""

import unittest

import numpy as np

from src.optimization.cvar import (
    CVaRResult,
    cvar_risk_adjusted_objective,
    empirical_cvar,
    empirical_var,
    evaluate_cvar,
)


# ---------------------------------------------------------------------------
# 1. Alpha validation (Requirement 3)
# ---------------------------------------------------------------------------

class TestAlphaValidation(unittest.TestCase):
    def test_rejects_alpha_zero(self):
        with self.assertRaises(ValueError):
            empirical_var([1.0, 2.0, 3.0], alpha=0.0)

    def test_rejects_alpha_one(self):
        with self.assertRaises(ValueError):
            empirical_var([1.0, 2.0, 3.0], alpha=1.0)

    def test_rejects_negative_alpha(self):
        with self.assertRaises(ValueError):
            empirical_cvar([1.0, 2.0, 3.0], alpha=-0.1)

    def test_rejects_alpha_above_one(self):
        with self.assertRaises(ValueError):
            empirical_cvar([1.0, 2.0, 3.0], alpha=1.5)

    def test_accepts_boundary_adjacent_alphas(self):
        # Not exactly 0 or 1, but very close -- must not raise.
        empirical_var([1.0, 2.0, 3.0], alpha=1e-6)
        empirical_var([1.0, 2.0, 3.0], alpha=1.0 - 1e-6)

    def test_accepts_typical_alphas(self):
        for a in (0.5, 0.9, 0.95, 0.99):
            evaluate_cvar([1.0, 2.0, 3.0, 4.0, 5.0], alpha=a)


# ---------------------------------------------------------------------------
# 2. Zero-scenario handling (Requirement 5)
# ---------------------------------------------------------------------------

class TestZeroScenarioHandling(unittest.TestCase):
    def test_empty_losses_raises_on_var(self):
        with self.assertRaises(ValueError):
            empirical_var([], alpha=0.9)

    def test_empty_losses_raises_on_cvar(self):
        with self.assertRaises(ValueError):
            empirical_cvar([], alpha=0.9)

    def test_empty_losses_raises_on_evaluate(self):
        with self.assertRaises(ValueError):
            evaluate_cvar(np.array([]), alpha=0.9)

    def test_empty_objectives_raises_on_risk_adjustment(self):
        with self.assertRaises(ValueError):
            cvar_risk_adjusted_objective([], alpha=0.9)

    def test_non_finite_losses_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_cvar([1.0, float("nan"), 3.0], alpha=0.9)
        with self.assertRaises(ValueError):
            evaluate_cvar([1.0, float("inf"), 3.0], alpha=0.9)


# ---------------------------------------------------------------------------
# 3. Zero-uncertainty reduction to deterministic Phase 6 (Requirement 6)
# ---------------------------------------------------------------------------

class TestZeroUncertaintyReduction(unittest.TestCase):
    def test_single_scenario_var_equals_that_scenario(self):
        for a in (0.01, 0.3, 0.5, 0.9, 0.999):
            self.assertEqual(empirical_var([7.25], alpha=a), 7.25)

    def test_single_scenario_cvar_equals_that_scenario(self):
        for a in (0.01, 0.3, 0.5, 0.9, 0.999):
            self.assertEqual(empirical_cvar([7.25], alpha=a), 7.25)

    def test_single_scenario_var_equals_cvar(self):
        result = evaluate_cvar([-3.4], alpha=0.95)
        self.assertEqual(result.var, result.cvar)
        self.assertEqual(result.var, -3.4)

    def test_single_scenario_risk_adjusted_objective_is_deterministic_value(self):
        # This is the identity that lets Phase 7 CVaR reduce exactly to
        # Phase 6's plain deterministic objective when there is only one
        # (deterministic) predicted trajectory / zero uncertainty.
        g = 123456.789
        for a in (0.1, 0.5, 0.9, 0.99):
            g_adj, result = cvar_risk_adjusted_objective([g], alpha=a)
            self.assertAlmostEqual(g_adj, g)
            self.assertAlmostEqual(result.cvar, -g)

    def test_identical_scenarios_behave_like_single_scenario(self):
        # N copies of the same value is "zero uncertainty" in substance,
        # even though N > 1.
        result = evaluate_cvar([5.0] * 10, alpha=0.8)
        self.assertAlmostEqual(result.var, 5.0)
        self.assertAlmostEqual(result.cvar, 5.0)


# ---------------------------------------------------------------------------
# 4. Explicit worked-example correctness (Requirements 1, 2)
# ---------------------------------------------------------------------------

class TestWorkedExamples(unittest.TestCase):
    def test_integer_tail_mass_matches_hand_computed_tail_average(self):
        # losses = 1..10, alpha=0.8 => tail mass (1-alpha)*N = 2 is an
        # integer, so CVaR must equal the plain average of the worst 2
        # scenarios: (9 + 10) / 2 = 9.5.
        losses = list(range(1, 11))  # 1..10
        result = evaluate_cvar(losses, alpha=0.8)
        self.assertAlmostEqual(result.var, 8.0)
        self.assertAlmostEqual(result.cvar, 9.5)

    def test_fractional_tail_mass_matches_hand_computed_partial_weighting(self):
        # losses = 1..10, alpha=0.85 => tail mass (1-alpha)*N = 1.5.
        # VaR = ceil(0.85*10)=9th order stat = 9.
        # CVaR = 9 + max(10-9,0)/1.5 = 9 + 1/1.5 = 9.6666...
        losses = list(range(1, 11))
        result = evaluate_cvar(losses, alpha=0.85)
        self.assertAlmostEqual(result.var, 9.0)
        self.assertAlmostEqual(result.cvar, 9.0 + 1.0 / 1.5, places=9)

    def test_all_equal_losses_var_and_cvar_equal_the_constant(self):
        result = evaluate_cvar([4.0] * 7, alpha=0.6)
        self.assertAlmostEqual(result.var, 4.0)
        self.assertAlmostEqual(result.cvar, 4.0)

    def test_evaluate_cvar_returns_dataclass_with_expected_fields(self):
        result = evaluate_cvar([1.0, 2.0, 3.0], alpha=0.5)
        self.assertIsInstance(result, CVaRResult)
        self.assertEqual(result.num_scenarios, 3)
        self.assertEqual(result.alpha, 0.5)
        np.testing.assert_array_equal(result.losses, np.array([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# 5. Determinism (Requirement 4)
# ---------------------------------------------------------------------------

class TestDeterminism(unittest.TestCase):
    def test_repeated_calls_identical(self):
        losses = np.array([3.1, -0.2, 5.6, 2.2, -1.0, 7.7, 0.0, 4.4])
        r1 = evaluate_cvar(losses, alpha=0.75)
        r2 = evaluate_cvar(losses, alpha=0.75)
        self.assertEqual(r1.var, r2.var)
        self.assertEqual(r1.cvar, r2.cvar)

    def test_input_order_does_not_matter(self):
        losses = [3.1, -0.2, 5.6, 2.2, -1.0, 7.7, 0.0, 4.4]
        shuffled = losses[::-1]
        r1 = evaluate_cvar(losses, alpha=0.75)
        r2 = evaluate_cvar(shuffled, alpha=0.75)
        self.assertAlmostEqual(r1.var, r2.var)
        self.assertAlmostEqual(r1.cvar, r2.cvar)

    def test_input_array_not_mutated(self):
        losses = np.array([3.0, 1.0, 2.0])
        original = losses.copy()
        evaluate_cvar(losses, alpha=0.5)
        np.testing.assert_array_equal(losses, original)


# ---------------------------------------------------------------------------
# 6. Structural properties (Requirements 1, 2, 7 -- no invented distribution)
# ---------------------------------------------------------------------------

class TestStructuralProperties(unittest.TestCase):
    def test_cvar_always_at_least_var_for_losses(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            n = rng.integers(2, 50)
            losses = rng.normal(size=n)
            a = float(rng.uniform(0.05, 0.95))
            result = evaluate_cvar(losses, alpha=a)
            self.assertGreaterEqual(result.cvar + 1e-9, result.var)

    def test_cvar_alpha_increasing_in_alpha(self):
        rng = np.random.default_rng(1)
        losses = rng.normal(size=200)
        alphas = [0.5, 0.7, 0.9, 0.95, 0.99]
        cvars = [empirical_cvar(losses, a) for a in alphas]
        for a, b in zip(cvars, cvars[1:]):
            self.assertLessEqual(a, b + 1e-9)

    def test_cvar_bounded_by_max_loss(self):
        losses = [1.0, 2.0, 3.0, 100.0]
        for a in (0.1, 0.5, 0.9, 0.99):
            result = evaluate_cvar(losses, alpha=a)
            self.assertLessEqual(result.cvar, 100.0 + 1e-9)

    def test_risk_adjusted_objective_never_exceeds_empirical_mean(self):
        rng = np.random.default_rng(2)
        gains = rng.normal(loc=10.0, scale=3.0, size=50)
        g_adj, _ = cvar_risk_adjusted_objective(gains, alpha=0.9)
        self.assertLessEqual(g_adj, float(np.mean(gains)) + 1e-9)

    def test_no_distribution_fitting_only_empirical_values_used(self):
        # CVaR of a set containing one extreme outlier must be driven
        # entirely by that outlier at high alpha (not smoothed away by
        # any assumed parametric shape).
        losses = [0.0] * 9 + [1000.0]
        result = evaluate_cvar(losses, alpha=0.95)
        self.assertAlmostEqual(result.var, 1000.0)
        self.assertAlmostEqual(result.cvar, 1000.0)


if __name__ == "__main__":
    unittest.main()
