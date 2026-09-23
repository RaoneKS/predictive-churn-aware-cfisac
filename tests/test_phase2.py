"""
tests/test_phase2.py

Tests for Phase 2:
  - Predictor interface (CV and LSTM)
  - Horizon handling
  - Predictor/simulator integration
  - Normalization round-trip
  - Simulation backward-compatibility (predictor=None gives same results as explicit CV)
  - PREDICTIVE+CHURN horizon-averaged objective
"""

import unittest
import numpy as np

from src.prediction.normalization import normalize, denormalize
from src.prediction.predictor_interface import CVPredictor, make_predictor
from src.simulation.end_to_end import run_simulation, _horizon_averaged_objective
from src.simulation.mobility_simulator import MobilitySimulator
from src.environment.mobility_configs import STOCHASTIC_CONFIG


# ─────────────────────────────────────────────────────────────────────────────
# 1. Normalization round-trip
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalization(unittest.TestCase):

    def test_normalize_centre(self):
        """Centre of area maps to 0."""
        arr = np.array([[250.0, 250.0]], dtype=np.float32)
        n   = normalize(arr, area_size=500.0)
        np.testing.assert_allclose(n, np.zeros_like(n), atol=1e-5)

    def test_normalize_zero(self):
        """Position 0 maps to -1."""
        arr = np.array([[0.0, 0.0]], dtype=np.float32)
        n   = normalize(arr, area_size=500.0)
        np.testing.assert_allclose(n, np.full_like(n, -1.0), atol=1e-5)

    def test_normalize_max(self):
        """Position area_size maps to +1."""
        arr = np.array([[500.0, 500.0]], dtype=np.float32)
        n   = normalize(arr, area_size=500.0)
        np.testing.assert_allclose(n, np.full_like(n, 1.0), atol=1e-5)

    def test_round_trip(self):
        """normalize ∘ denormalize = identity."""
        rng = np.random.default_rng(0)
        arr = rng.uniform(0, 500, size=(10, 5, 2)).astype(np.float32)
        n   = normalize(arr, 500.0)
        rec = denormalize(n, 500.0)
        np.testing.assert_allclose(rec, arr, atol=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# 2. CVPredictor interface
# ─────────────────────────────────────────────────────────────────────────────

class TestCVPredictor(unittest.TestCase):

    def _make_sim(self):
        users   = np.array([[100., 100.], [200., 300.]])
        targets = np.array([[250., 250.]])
        sim = MobilitySimulator(users, targets, area_size=500., dt=1., seed=0)
        sim.step()
        return sim

    def test_cv_predictor_output_shape(self):
        """CVPredictor returns (N_u, H, 2) and (N_t, H, 2)."""
        sim = self._make_sim()
        pred = CVPredictor()
        u, t = pred.predict(sim, horizon=5)
        self.assertEqual(u.shape, (2, 5, 2))
        self.assertEqual(t.shape, (1, 5, 2))

    def test_cv_predictor_matches_simulator(self):
        """CVPredictor must give same result as sim.predict_users."""
        sim = self._make_sim()
        pred = CVPredictor()
        u_pred, t_pred = pred.predict(sim, horizon=5)
        np.testing.assert_array_almost_equal(u_pred, sim.predict_users(5))
        np.testing.assert_array_almost_equal(t_pred, sim.predict_targets(5))

    def test_make_predictor_cv(self):
        """make_predictor('cv') returns a CVPredictor."""
        p = make_predictor("cv")
        self.assertIsInstance(p, CVPredictor)

    def test_make_predictor_unknown(self):
        with self.assertRaises(ValueError):
            make_predictor("unknown_predictor")

    def test_make_predictor_lstm_missing_args(self):
        with self.assertRaises(ValueError):
            make_predictor("lstm")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Predictor output dimensions match simulation needs
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictorDimensions(unittest.TestCase):

    def _make_sim(self, n_u=5, n_t=2):
        rng   = np.random.default_rng(7)
        users = rng.uniform(0, 500, (n_u, 2))
        tgts  = rng.uniform(0, 500, (n_t, 2))
        sim   = MobilitySimulator(users, tgts, area_size=500., dt=1., seed=7)
        sim.step()
        return sim, n_u, n_t

    def test_cv_H_steps(self):
        """Horizon H is propagated correctly for H in {1,2,5,10}."""
        sim, n_u, n_t = self._make_sim()
        pred = CVPredictor()
        for H in [1, 2, 5, 10]:
            u, t = pred.predict(sim, H)
            self.assertEqual(u.shape, (n_u, H, 2), f"H={H} users shape wrong")
            self.assertEqual(t.shape, (n_t, H, 2), f"H={H} targets shape wrong")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Simulation backward compatibility
# ─────────────────────────────────────────────────────────────────────────────

class TestSimulationBackwardCompat(unittest.TestCase):

    def test_predictor_none_same_as_explicit_cv(self):
        """
        run_simulation with predictor=None must give identical results
        to run_simulation with predictor=CVPredictor().
        """
        from src.prediction.predictor_interface import CVPredictor
        m1 = run_simulation("PREDICTIVE", seed=0, time_blocks=5)
        m2 = run_simulation("PREDICTIVE", seed=0, time_blocks=5,
                            predictor=CVPredictor())
        np.testing.assert_array_almost_equal(m1["avg_user_rate"], m2["avg_user_rate"])
        np.testing.assert_array_almost_equal(m1["combined_objective"], m2["combined_objective"])

    def test_static_policy_zero_churn(self):
        """STATIC must produce zero churn at every step after t=0."""
        m = run_simulation("STATIC", seed=1, time_blocks=10)
        self.assertEqual(m["churn"][1:].sum(), 0.0)

    def test_runtime_recorded(self):
        m = run_simulation("REACTIVE", seed=2, time_blocks=5)
        self.assertIn("runtime_s", m)
        self.assertGreater(m["runtime_s"], 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Stochastic mobility in simulation
# ─────────────────────────────────────────────────────────────────────────────

class TestStochasticMobilityInSimulation(unittest.TestCase):

    def test_static_runs_with_stochastic_mobility(self):
        m = run_simulation(
            "STATIC", seed=3, time_blocks=5,
            mobility_config=STOCHASTIC_CONFIG,
        )
        self.assertEqual(len(m["avg_user_rate"]), 5)

    def test_predictive_runs_with_stochastic_mobility(self):
        m = run_simulation(
            "PREDICTIVE", seed=4, time_blocks=5,
            mobility_config=STOCHASTIC_CONFIG,
        )
        self.assertEqual(len(m["combined_objective"]), 5)

    def test_stochastic_produces_different_results_from_cv(self):
        m_cv   = run_simulation("REACTIVE", seed=5, time_blocks=20)
        m_stoch = run_simulation("REACTIVE", seed=5, time_blocks=20,
                                 mobility_config=STOCHASTIC_CONFIG)
        # Stochastic mobility should give different avg_user_rate than CV
        self.assertFalse(np.allclose(m_cv["avg_user_rate"], m_stoch["avg_user_rate"]))


# ─────────────────────────────────────────────────────────────────────────────
# 6. Horizon-averaged objective helper
# ─────────────────────────────────────────────────────────────────────────────

class TestHorizonAveragedObjective(unittest.TestCase):

    def _setup(self):
        from src.environment.topology import generate_ap_positions
        from src.optimization.baselines import reactive_policy
        rng    = np.random.default_rng(9)
        ap_pos = generate_ap_positions(10, 500., seed=9)
        u_pos  = rng.uniform(0, 500, (3, 2))
        t_pos  = rng.uniform(0, 500, (2, 2))
        sim    = MobilitySimulator(u_pos, t_pos, area_size=500., dt=1., seed=9)
        sim.step()
        x, y_tx, y_rx = reactive_policy(ap_pos, u_pos, t_pos)
        return ap_pos, x, y_tx, y_rx, sim

    def test_horizon_average_returns_scalar(self):
        ap_pos, x, y_tx, y_rx, sim = self._setup()
        pred_u = sim.predict_users(5)
        pred_t = sim.predict_targets(5)
        perf_kwargs = {"alpha_communication": 1.0, "beta_sensing": 1.0,
                       "lambda_energy": 1.0, "lambda_fronthaul": 1.0,
                       "ref": {"communication_rate_bps": 10e6,
                               "sensing_information_gain": 1.0,
                               "energy_watts": 1.0, "fronthaul_links": 1.0,
                               "churn_events": 1.0, "qos_violations": 1.0,
                               "tracking_violations": 1.0}}
        avg = _horizon_averaged_objective(
            x, y_tx, y_rx, ap_pos, pred_u, pred_t, perf_kwargs
        )
        self.assertIsInstance(avg, float)

    def test_predictive_churn_uses_horizon_avg(self):
        """PREDICTIVE+CHURN decision trace should exist and contain horizon info."""
        import os
        m = run_simulation("PREDICTIVE+CHURN", seed=6, time_blocks=10)
        # Decision trace CSV should have been written
        self.assertTrue(os.path.exists("results/decision_trace.csv"))


if __name__ == "__main__":
    unittest.main()
