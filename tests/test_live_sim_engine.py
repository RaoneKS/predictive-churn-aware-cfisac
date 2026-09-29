import unittest
import numpy as np

from live_sim.engine import LiveCFISACEngine, LiveConfig


class TestLiveCFISACEngine(unittest.TestCase):
    def make_engine(self):
        return LiveCFISACEngine.create(LiveConfig(
            num_aps=8, num_users=2, num_targets=1, num_antennas=2,
            area_size=100.0, T_slow=2, prediction_horizon=2,
            num_scenarios=0, seed=11, base_fast_seed=3,
        ))

    def test_initial_evaluation_uses_real_physical_backend(self):
        e = self.make_engine()
        r = e.evaluate_current()
        self.assertIn("rates_bps", e.last_fast)
        self.assertIn("information_gain", e.last_fast)
        self.assertTrue(np.isfinite(r["rate_bps"]))
        self.assertTrue(np.isfinite(r["sensing_info"]))
        self.assertTrue(r["feasible"])

    def test_step_changes_mutable_state_and_computes_new_fast_result(self):
        e = self.make_engine()
        r0 = e.evaluate_current()
        p0 = e.current_positions["users"].copy()
        r1 = e.step_once()
        p1 = e.current_positions["users"].copy()
        self.assertEqual(r1["step"], 1)
        self.assertFalse(np.array_equal(p0, p1))
        self.assertEqual(len(e.records), 2)
        self.assertIn("rates_bps", e.last_fast)
        self.assertNotEqual(r0["step"], r1["step"])

    def test_slow_epoch_runs_on_schedule(self):
        e = self.make_engine()
        e.evaluate_current()
        self.assertEqual(e.slow_epoch, 0)
        e.step_once()
        self.assertEqual(e.slow_epoch, 0)
        e.step_once()
        self.assertEqual(e.slow_epoch, 1)

    def test_no_hardcoded_default_churn_reference(self):
        e = self.make_engine()
        ref = e._ref()
        self.assertNotEqual(ref["churn_events"], 46.0)
        self.assertGreater(ref["churn_events"], 0.0)

    def test_p1_mode_off(self):
        e = self.make_engine()
        e.cfg.p1_mode = False
        rec = e.step_once()
        self.assertFalse(rec.get("p1_mode", True))

    def test_p1_mode_on(self):
        e = self.make_engine()
        e.cfg.p1_mode = True
        e.cfg.p1_max_candidates = 5
        e.step_once()
        rec = e.step_once()
        self.assertTrue(rec.get("p1_mode", False))
        self.assertIn("p1_candidates", rec)
        self.assertIn("p1_feasible", rec)
        self.assertIn("p1_objective", rec)
        self.assertIn("p1_comm_util", rec)
        self.assertIn("p1_sens_util", rec)
        self.assertIn("p1_energy", rec)
        self.assertIn("p1_fronthaul", rec)
        self.assertIn("p1_fallback", rec)

    def test_p1_fallback_trigger(self):
        e = self.make_engine()
        e.cfg.p1_mode = True
        e.cfg.p1_max_candidates = 5
        e.cfg.enforce_qos = True
        e.cfg.min_rate_bps = 1e15
        e.step_once()
        rec = e.step_once()
        self.assertTrue(rec["p1_fallback"])

if __name__ == "__main__":
    unittest.main()
