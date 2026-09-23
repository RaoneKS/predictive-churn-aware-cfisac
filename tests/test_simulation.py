import unittest
import numpy as np

from src.simulation.end_to_end import run_simulation
from src.simulation.mobility_simulator import MobilitySimulator


class TestEndToEndSimulation(unittest.TestCase):
    def setUp(self):
        self.time_blocks = 10
        self.seed = 42
        
    def test_identical_mobility_trajectory(self):
        users = np.array([[10, 10], [20, 20]])
        targets = np.array([[50, 50]])
        sim1 = MobilitySimulator(users, targets, seed=42)
        sim2 = MobilitySimulator(users, targets, seed=42)
        
        for _ in range(5):
            u1, t1 = sim1.step()
            u2, t2 = sim2.step()
            self.assertTrue(np.allclose(u1, u2))
            self.assertTrue(np.allclose(t1, t2))

    def test_static_never_changes_clusters(self):
        res = run_simulation("STATIC", time_blocks=self.time_blocks, seed=self.seed)
        # Block 0 has reconfigured=0 since we excluded t=0 from being counted as a reconfiguration
        self.assertTrue(np.all(res["reconfigurations"] == 0))
        self.assertEqual(len(res["reconfigurations"]), self.time_blocks)
        self.assertTrue(np.all(res["churn"] == 0.0))

    def test_reactive_responds_to_current_positions(self):
        res = run_simulation("REACTIVE", time_blocks=self.time_blocks, seed=self.seed)
        # Churn should be non-zero at least sometimes
        self.assertTrue(np.any(res["churn"] > 0.0))

    def test_predictive_uses_future_positions(self):
        res = run_simulation("PREDICTIVE", time_blocks=self.time_blocks, seed=self.seed)
        self.assertTrue(np.any(res["churn"] > 0.0))
        
    def test_predictive_churn_invokes_keep_reconfigure(self):
        res = run_simulation("PREDICTIVE+CHURN", time_blocks=self.time_blocks, seed=self.seed)
        total_decisions = res["keep_decisions"] + res["reconfigure_decisions"]
        self.assertEqual(total_decisions, self.time_blocks - 1)  # t=0 is excluded
        
    def test_recorded_metrics_are_finite(self):
        res = run_simulation("REACTIVE", time_blocks=2, seed=self.seed)
        for k, v in res.items():
            if isinstance(v, np.ndarray):
                self.assertTrue(np.all(np.isfinite(v)), f"Metric {k} has non-finite values.")

if __name__ == "__main__":
    unittest.main()
