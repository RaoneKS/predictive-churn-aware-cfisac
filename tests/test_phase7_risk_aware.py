"""
tests/test_phase7_risk_aware.py

Phase 7 -- formal integration tests for src.optimization.risk_aware
(risk_aware_slow_update, run_risk_aware_two_timescale).

Covers, per the Phase 7 completion requirements:
  1. zero scenarios -> exact Phase 6 passthrough
  2. empty residual history -> deterministic reduction
  3. nonzero measured residuals -> CVaR differs from deterministic gain
  4. VaR/CVaR remain in loss domain
  5. risk_adjusted_gain remains in gain domain
  6. KEEP/RECONFIGURE uses the existing churn cost unchanged
  7. deterministic same-seed behavior
  8. different seeds can produce different scenario realizations
  9. fast-update compatibility with Phase 6
  10. per-AP power feasibility
"""

import unittest

import numpy as np

from src.clustering.churn import total_churn_cost
from src.environment.topology import generate_ap_positions, generate_positions
from src.optimization.baselines import cluster_for_positions
from src.optimization.two_timescale import fast_update, slow_update
from src.optimization.risk_aware import (
    risk_aware_slow_update,
    run_risk_aware_two_timescale,
)
from src.prediction.predictor_interface import CVPredictor
from src.simulation.mobility_simulator import MobilitySimulator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_scenario(num_aps=20, num_users=5, num_targets=2, area_size=500.0,
                    seed=42, stochastic=False):
    ap = generate_ap_positions(num_aps, area_size, seed=seed)
    us = generate_positions(num_users, area_size, seed=seed + 1)
    tg = generate_positions(num_targets, area_size, seed=seed + 2)
    mobility_config = None
    if stochastic:
        mobility_config = {
            "type": "stochastic",
            "users": {"sigma_v": 3.0, "rho_v": 0.5},
            "targets": {"sigma_v": 3.0, "rho_v": 0.5},
        }
    sim = MobilitySimulator(us, tg, area_size=area_size, dt=1.0, seed=seed,
                             mobility_config=mobility_config)
    predictor = CVPredictor()
    return ap, sim, predictor


def _advance(sim, n):
    for _ in range(n):
        sim.step()


# ---------------------------------------------------------------------------
# 1. Zero scenarios -> exact Phase 6 passthrough
# ---------------------------------------------------------------------------

class TestZeroScenariosPassthrough(unittest.TestCase):
    def test_matches_phase6_slow_update_bit_for_bit_at_epoch0(self):
        ap, sim, predictor = _make_scenario()
        x, y_tx, y_rx = cluster_for_positions(ap, sim.user_positions, sim.target_positions)

        phase6 = slow_update(sim, predictor, ap, x, y_tx, y_rx, T_slow=5)
        phase7 = risk_aware_slow_update(sim, predictor, ap, x, y_tx, y_rx,
                                         T_slow=5, num_scenarios=0)

        self.assertEqual(phase7["decision"], phase6["decision"])
        np.testing.assert_array_equal(phase7["x"], phase6["x"])
        np.testing.assert_array_equal(phase7["y_tx"], phase6["y_tx"])
        np.testing.assert_array_equal(phase7["y_rx"], phase6["y_rx"])
        self.assertEqual(
            phase7["decision_info"]["net_gain"], phase6["decision_info"]["net_gain"]
        )
        self.assertEqual(
            phase7["decision_info"]["raw_churn"], phase6["decision_info"]["raw_churn"]
        )
        self.assertIsNone(phase7["var"])
        self.assertIsNone(phase7["cvar"])
        self.assertIsNone(phase7["alpha"])
        self.assertIsNone(phase7["scenario_gains"])

    def test_matches_phase6_slow_update_bit_for_bit_after_history(self):
        ap, sim, predictor = _make_scenario()
        x, y_tx, y_rx = cluster_for_positions(ap, sim.user_positions, sim.target_positions)
        _advance(sim, 15)

        phase6 = slow_update(sim, predictor, ap, x, y_tx, y_rx, T_slow=5)
        phase7 = risk_aware_slow_update(sim, predictor, ap, x, y_tx, y_rx,
                                         T_slow=5, num_scenarios=0)

        self.assertEqual(phase7["decision"], phase6["decision"])
        self.assertEqual(
            phase7["risk_adjusted_gain"], phase6["decision_info"]["predicted_gain"]
        )


# ---------------------------------------------------------------------------
# 2. Empty residual history -> deterministic reduction
# ---------------------------------------------------------------------------

class TestEmptyResidualHistoryReduction(unittest.TestCase):
    def test_epoch0_scenarios_reduce_to_deterministic_gain(self):
        ap, sim, predictor = _make_scenario()
        x, y_tx, y_rx = cluster_for_positions(ap, sim.user_positions, sim.target_positions)

        upd = risk_aware_slow_update(sim, predictor, ap, x, y_tx, y_rx,
                                      T_slow=5, num_scenarios=25, alpha=0.9,
                                      scenario_seed=3)

        self.assertAlmostEqual(upd["risk_adjusted_gain"], upd["deterministic_gain"], places=6)
        # loss-domain var/cvar must agree exactly under identical scenarios.
        self.assertAlmostEqual(upd["var"], upd["cvar"], places=6)
        self.assertAlmostEqual(upd["var"], -upd["deterministic_gain"], places=6)

    def test_epoch0_decision_matches_phase6_decision(self):
        ap, sim, predictor = _make_scenario()
        x, y_tx, y_rx = cluster_for_positions(ap, sim.user_positions, sim.target_positions)

        phase6 = slow_update(sim, predictor, ap, x, y_tx, y_rx, T_slow=5)
        phase7 = risk_aware_slow_update(sim, predictor, ap, x, y_tx, y_rx,
                                         T_slow=5, num_scenarios=25, alpha=0.9,
                                         scenario_seed=3)
        self.assertEqual(phase7["decision"], phase6["decision"])

    def test_short_history_below_horizon_also_reduces(self):
        ap, sim, predictor = _make_scenario()
        x, y_tx, y_rx = cluster_for_positions(ap, sim.user_positions, sim.target_positions)
        _advance(sim, 2)  # fewer steps than T_slow -> still zero anchors

        upd = risk_aware_slow_update(sim, predictor, ap, x, y_tx, y_rx,
                                      T_slow=5, num_scenarios=10, alpha=0.8,
                                      scenario_seed=1)
        self.assertAlmostEqual(upd["risk_adjusted_gain"], upd["deterministic_gain"], places=6)


# ---------------------------------------------------------------------------
# 3. Nonzero measured residuals -> CVaR differs from deterministic gain
# ---------------------------------------------------------------------------

class TestNonzeroResidualsChangeRiskAdjustedGain(unittest.TestCase):
    def test_risk_adjusted_gain_differs_once_residual_history_exists(self):
        ap, sim, predictor = _make_scenario(stochastic=True)
        x, y_tx, y_rx = cluster_for_positions(ap, sim.user_positions, sim.target_positions)
        _advance(sim, 20)

        upd = risk_aware_slow_update(sim, predictor, ap, x, y_tx, y_rx,
                                      T_slow=5, num_scenarios=100, alpha=0.9,
                                      scenario_seed=7)
        self.assertNotAlmostEqual(
            upd["risk_adjusted_gain"], upd["deterministic_gain"], places=3
        )

    def test_scenario_gains_are_not_all_identical_with_real_history(self):
        ap, sim, predictor = _make_scenario(stochastic=True)
        x, y_tx, y_rx = cluster_for_positions(ap, sim.user_positions, sim.target_positions)
        _advance(sim, 20)

        upd = risk_aware_slow_update(sim, predictor, ap, x, y_tx, y_rx,
                                      T_slow=5, num_scenarios=100, alpha=0.9,
                                      scenario_seed=7)
        self.assertFalse(np.allclose(upd["scenario_gains"], upd["scenario_gains"][0]))


# ---------------------------------------------------------------------------
# 4. VaR / CVaR remain in loss domain ; 5. risk_adjusted_gain in gain domain
# ---------------------------------------------------------------------------

class TestSignDomains(unittest.TestCase):
    def test_cvar_result_losses_are_negated_gains(self):
        ap, sim, predictor = _make_scenario(stochastic=True)
        x, y_tx, y_rx = cluster_for_positions(ap, sim.user_positions, sim.target_positions)
        _advance(sim, 20)

        upd = risk_aware_slow_update(sim, predictor, ap, x, y_tx, y_rx,
                                      T_slow=5, num_scenarios=50, alpha=0.9,
                                      scenario_seed=4)
        np.testing.assert_allclose(
            upd["cvar_result"].losses, -upd["scenario_gains"], atol=1e-9
        )

    def test_cvar_at_least_var_in_loss_domain(self):
        ap, sim, predictor = _make_scenario(stochastic=True)
        x, y_tx, y_rx = cluster_for_positions(ap, sim.user_positions, sim.target_positions)
        _advance(sim, 20)

        upd = risk_aware_slow_update(sim, predictor, ap, x, y_tx, y_rx,
                                      T_slow=5, num_scenarios=50, alpha=0.9,
                                      scenario_seed=4)
        # CVaR of losses is always >= VaR of losses (structural property).
        self.assertGreaterEqual(upd["cvar"] + 1e-9, upd["var"])

    def test_risk_adjusted_gain_equals_negative_cvar(self):
        ap, sim, predictor = _make_scenario(stochastic=True)
        x, y_tx, y_rx = cluster_for_positions(ap, sim.user_positions, sim.target_positions)
        _advance(sim, 20)

        upd = risk_aware_slow_update(sim, predictor, ap, x, y_tx, y_rx,
                                      T_slow=5, num_scenarios=50, alpha=0.9,
                                      scenario_seed=4)
        self.assertAlmostEqual(upd["risk_adjusted_gain"], -upd["cvar"], places=9)

    def test_risk_adjusted_gain_never_exceeds_scenario_mean_gain(self):
        # Coherent-risk-measure property carried over from cvar.py:
        # G_CVaR_alpha <= mean_i(g_i) always (risk-averse, never optimistic).
        ap, sim, predictor = _make_scenario(stochastic=True)
        x, y_tx, y_rx = cluster_for_positions(ap, sim.user_positions, sim.target_positions)
        _advance(sim, 20)

        for alpha in (0.5, 0.7, 0.9, 0.95):
            upd = risk_aware_slow_update(sim, predictor, ap, x, y_tx, y_rx,
                                          T_slow=5, num_scenarios=80, alpha=alpha,
                                          scenario_seed=9)
            self.assertLessEqual(
                upd["risk_adjusted_gain"], np.mean(upd["scenario_gains"]) + 1e-6
            )

    def test_sign_domain_regression_against_naive_direct_cvar_on_gains(self):
        # Regression guard for the documented sign-domain bug: computing
        # CVaR DIRECTLY on gains (no sign flip) picks the best-case tail,
        # not the worst-case tail, and must NOT match this module's output.
        from src.optimization.cvar import empirical_cvar

        ap, sim, predictor = _make_scenario(stochastic=True)
        x, y_tx, y_rx = cluster_for_positions(ap, sim.user_positions, sim.target_positions)
        _advance(sim, 20)

        upd = risk_aware_slow_update(sim, predictor, ap, x, y_tx, y_rx,
                                      T_slow=5, num_scenarios=80, alpha=0.9,
                                      scenario_seed=11)
        gains = upd["scenario_gains"]
        if np.allclose(gains, gains[0]):
            self.skipTest("degenerate identical-gain draw; sign flip is unobservable")
        naive_wrong_cvar = empirical_cvar(gains, alpha=0.9)  # BUG: no sign flip
        self.assertNotAlmostEqual(upd["risk_adjusted_gain"], naive_wrong_cvar, places=3)


# ---------------------------------------------------------------------------
# 6. KEEP/RECONFIGURE uses the existing churn cost unchanged
# ---------------------------------------------------------------------------

class TestChurnCostUnchanged(unittest.TestCase):
    def test_raw_churn_matches_total_churn_cost_directly(self):
        ap, sim, predictor = _make_scenario(stochastic=True)
        x, y_tx, y_rx = cluster_for_positions(ap, sim.user_positions, sim.target_positions)
        _advance(sim, 20)

        upd = risk_aware_slow_update(sim, predictor, ap, x, y_tx, y_rx,
                                      T_slow=5, num_scenarios=30, alpha=0.9,
                                      scenario_seed=2)
        expected = total_churn_cost(
            x, upd["candidate_x"], y_tx, upd["candidate_y_tx"],
            y_rx, upd["candidate_y_rx"],
        )
        self.assertAlmostEqual(upd["decision_info"]["raw_churn"], expected, places=9)

    def test_raw_churn_identical_regardless_of_num_scenarios(self):
        ap, sim, predictor = _make_scenario(stochastic=True)
        x, y_tx, y_rx = cluster_for_positions(ap, sim.user_positions, sim.target_positions)
        _advance(sim, 20)

        det = risk_aware_slow_update(sim, predictor, ap, x, y_tx, y_rx,
                                      T_slow=5, num_scenarios=0)
        risky = risk_aware_slow_update(sim, predictor, ap, x, y_tx, y_rx,
                                        T_slow=5, num_scenarios=40, alpha=0.9,
                                        scenario_seed=2)
        self.assertAlmostEqual(
            det["decision_info"]["raw_churn"], risky["decision_info"]["raw_churn"],
            places=9,
        )

    def test_huge_lambda_churn_forces_keep_even_with_scenarios(self):
        ap, sim, predictor = _make_scenario(stochastic=True)
        x, y_tx, y_rx = cluster_for_positions(ap, sim.user_positions, sim.target_positions)
        _advance(sim, 20)

        upd = risk_aware_slow_update(sim, predictor, ap, x, y_tx, y_rx,
                                      T_slow=5, num_scenarios=30, alpha=0.9,
                                      scenario_seed=2,
                                      churn_kwargs={"lambda_churn": 1e12})
        self.assertEqual(upd["decision"], "KEEP")


# ---------------------------------------------------------------------------
# 7. Deterministic same-seed behavior ; 8. different seeds can differ
# ---------------------------------------------------------------------------

class TestSeeding(unittest.TestCase):
    def test_same_seed_identical_output(self):
        ap, sim, predictor = _make_scenario(stochastic=True)
        x, y_tx, y_rx = cluster_for_positions(ap, sim.user_positions, sim.target_positions)
        _advance(sim, 20)

        u1 = risk_aware_slow_update(sim, predictor, ap, x, y_tx, y_rx,
                                     T_slow=5, num_scenarios=40, alpha=0.9,
                                     scenario_seed=123)
        u2 = risk_aware_slow_update(sim, predictor, ap, x, y_tx, y_rx,
                                     T_slow=5, num_scenarios=40, alpha=0.9,
                                     scenario_seed=123)
        np.testing.assert_array_equal(u1["scenario_gains"], u2["scenario_gains"])
        self.assertEqual(u1["risk_adjusted_gain"], u2["risk_adjusted_gain"])
        self.assertEqual(u1["decision"], u2["decision"])

    def test_different_seeds_can_produce_different_scenarios(self):
        ap, sim, predictor = _make_scenario(stochastic=True)
        x, y_tx, y_rx = cluster_for_positions(ap, sim.user_positions, sim.target_positions)
        _advance(sim, 20)

        u1 = risk_aware_slow_update(sim, predictor, ap, x, y_tx, y_rx,
                                     T_slow=5, num_scenarios=40, alpha=0.9,
                                     scenario_seed=1)
        u2 = risk_aware_slow_update(sim, predictor, ap, x, y_tx, y_rx,
                                     T_slow=5, num_scenarios=40, alpha=0.9,
                                     scenario_seed=2)
        self.assertFalse(
            np.array_equal(u1["scenario_gains"], u2["scenario_gains"])
        )


# ---------------------------------------------------------------------------
# 9. fast-update compatibility with Phase 6 ; 10. per-AP power feasibility
# ---------------------------------------------------------------------------

class TestFastUpdateCompatibilityAndFeasibility(unittest.TestCase):
    def test_risk_aware_association_is_valid_fast_update_input(self):
        ap, sim, predictor = _make_scenario(stochastic=True)
        x, y_tx, y_rx = cluster_for_positions(ap, sim.user_positions, sim.target_positions)
        _advance(sim, 20)

        upd = risk_aware_slow_update(sim, predictor, ap, x, y_tx, y_rx,
                                      T_slow=5, num_scenarios=30, alpha=0.9,
                                      scenario_seed=2)

        res = fast_update(
            ap, sim.user_positions, sim.target_positions,
            upd["x"], upd["y_tx"], upd["y_rx"],
            num_antennas=4, P_max=1.0, fast_seed=0,
        )
        self.assertIn("feasible", res)
        self.assertIn("P_comm", res)
        self.assertIn("P_sens", res)

    def test_per_ap_power_feasibility(self):
        ap, sim, predictor = _make_scenario(stochastic=True)
        x, y_tx, y_rx = cluster_for_positions(ap, sim.user_positions, sim.target_positions)
        _advance(sim, 20)

        upd = risk_aware_slow_update(sim, predictor, ap, x, y_tx, y_rx,
                                      T_slow=5, num_scenarios=30, alpha=0.9,
                                      scenario_seed=2)
        P_max = 1.0
        res = fast_update(
            ap, sim.user_positions, sim.target_positions,
            upd["x"], upd["y_tx"], upd["y_rx"],
            num_antennas=4, P_max=P_max, fast_seed=0,
        )
        P_comm = np.asarray(res["P_comm"])
        P_sens = np.asarray(res["P_sens"])
        self.assertTrue(res["feasible"])
        self.assertTrue(np.all(P_comm + P_sens <= P_max + 1e-6))

    def test_run_risk_aware_two_timescale_end_to_end(self):
        ap, sim, predictor = _make_scenario(stochastic=True)
        result = run_risk_aware_two_timescale(
            sim, predictor, ap, num_antennas=4, T_total=12, T_slow=4,
            P_max=1.0, num_scenarios=20, cvar_alpha=0.9, scenario_seed=5,
            base_seed=3,
        )
        self.assertEqual(result.num_fast_updates, 12)
        self.assertEqual(result.num_slow_updates, 3)
        self.assertTrue(all(result.fast["feasible"]))
        for pc, ps in zip(result.fast["P_comm"], result.fast["P_sens"]):
            self.assertTrue(np.all(np.asarray(pc) + np.asarray(ps) <= 1.0 + 1e-6))

    def test_run_risk_aware_two_timescale_zero_scenarios_matches_phase6_driver(self):
        from src.optimization.two_timescale import run_two_timescale

        ap6, sim6, pred6 = _make_scenario()
        ap7, sim7, pred7 = _make_scenario()

        r6 = run_two_timescale(sim6, pred6, ap6, num_antennas=4, T_total=9,
                                T_slow=3, P_max=1.0, base_seed=1)
        r7 = run_risk_aware_two_timescale(sim7, pred7, ap7, num_antennas=4,
                                           T_total=9, T_slow=3, P_max=1.0,
                                           num_scenarios=0, base_seed=1)

        self.assertEqual(r6.slow["decision"], r7.slow["decision"])
        for a, b in zip(r6.fast["P_comm"], r7.fast["P_comm"]):
            np.testing.assert_allclose(a, b)


if __name__ == "__main__":
    unittest.main()
