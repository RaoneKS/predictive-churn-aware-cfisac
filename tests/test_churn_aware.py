import unittest
import numpy as np

from src.optimization.churn_aware import (
    evaluate_cluster_performance,
    estimate_reconfiguration_gain,
    churn_aware_decision,
)


class TestChurnAwareDecision(unittest.TestCase):
    def setUp(self):
        self.ap_positions = np.array([[0, 0], [100, 0], [0, 100], [100, 100]])
        self.user_positions = np.array([[50, 50], [20, 20]])
        self.target_positions = np.array([[60, 60]])
        
        self.x_curr = np.array([[1, 0], [0, 1], [0, 0], [0, 0]])
        self.y_tx_curr = np.array([[1], [0], [0], [0]])
        self.y_rx_curr = np.array([[1], [0], [0], [0]])

        self.x_ident = self.x_curr.copy()
        self.y_tx_ident = self.y_tx_curr.copy()
        self.y_rx_ident = self.y_rx_curr.copy()

        self.x_diff = np.array([[0, 0], [0, 0], [1, 0], [0, 1]])
        self.y_tx_diff = np.array([[0], [1], [0], [0]])
        self.y_rx_diff = np.array([[0], [1], [0], [0]])

    def test_evaluate_cluster_performance_metrics(self):
        perf = evaluate_cluster_performance(
            self.x_curr, self.y_tx_curr, self.y_rx_curr,
            self.ap_positions, self.user_positions, self.target_positions
        )
        for key, value in perf.items():
            self.assertTrue(np.all(np.isfinite(value)), f"{key} is not finite: {value}")

    def test_identical_configuration(self):
        perf = evaluate_cluster_performance(
            self.x_curr, self.y_tx_curr, self.y_rx_curr,
            self.ap_positions, self.user_positions, self.target_positions
        )
        
        decision = churn_aware_decision(
            self.x_curr, self.y_tx_curr, self.y_rx_curr,
            self.x_ident, self.y_tx_ident, self.y_rx_ident,
            current_performance=perf,
            predicted_performance=perf,
        )
        self.assertEqual(decision["norm_churn_cost"], 0.0)
        self.assertEqual(decision["net_gain"], 0.0)
        self.assertEqual(decision["decision"], "KEEP")
        self.assertAlmostEqual(decision["net_gain"], decision["predicted_gain"] - decision["norm_churn_cost"])

    def test_negative_predicted_gain(self):
        curr_perf = {"combined_objective": 100.0}
        pred_perf = {"combined_objective": 80.0}
        
        decision = churn_aware_decision(
            self.x_curr, self.y_tx_curr, self.y_rx_curr,
            self.x_diff, self.y_tx_diff, self.y_rx_diff,
            current_performance=curr_perf,
            predicted_performance=pred_perf,
        )
        self.assertEqual(decision["predicted_gain"], -20.0)
        self.assertEqual(decision["decision"], "KEEP")
        self.assertAlmostEqual(decision["net_gain"], decision["predicted_gain"] - decision["norm_churn_cost"])

    def test_high_churn_cost(self):
        curr_perf = {"combined_objective": 100.0}
        pred_perf = {"combined_objective": 101.0}
        
        decision = churn_aware_decision(
            self.x_curr, self.y_tx_curr, self.y_rx_curr,
            self.x_diff, self.y_tx_diff, self.y_rx_diff,
            current_performance=curr_perf,
            predicted_performance=pred_perf,
            c_comm=100.0,
            c_sensing_tx=100.0,
            c_sensing_rx=100.0,
            ref={"churn_events": 1.0},
        )
        self.assertEqual(decision["predicted_gain"], 1.0)
        self.assertTrue(decision["norm_churn_cost"] > 1.0)
        self.assertTrue(decision["net_gain"] < 0.0)
        self.assertEqual(decision["decision"], "KEEP")

    def test_positive_gain_reconfigure(self):
        curr_perf = {"combined_objective": 100.0}
        pred_perf = {"combined_objective": 200.0}
        
        decision = churn_aware_decision(
            self.x_curr, self.y_tx_curr, self.y_rx_curr,
            self.x_diff, self.y_tx_diff, self.y_rx_diff,
            current_performance=curr_perf,
            predicted_performance=pred_perf,
            c_comm=0.1,
            c_sensing_tx=0.1,
            c_sensing_rx=0.1,
            ref={"churn_events": 10.0},
        )
        self.assertEqual(decision["predicted_gain"], 100.0)
        self.assertTrue(decision["net_gain"] > 0.0)
        self.assertEqual(decision["decision"], "RECONFIGURE")


if __name__ == "__main__":
    unittest.main()
