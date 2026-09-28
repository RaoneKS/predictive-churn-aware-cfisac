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


if __name__ == "__main__":
    unittest.main()
