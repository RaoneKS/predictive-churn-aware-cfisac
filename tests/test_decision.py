import unittest
import numpy as np

from src.optimization.churn_aware import evaluate_cluster_performance, churn_aware_decision
from src.simulation.end_to_end import run_simulation
from src.simulation.mobility_simulator import MobilitySimulator


class TestDecisionCorrectness(unittest.TestCase):
    def setUp(self):
        # Mock configs
        self.ref = {
            "communication_rate_bps": 10000000.0,
            "sensing_information_gain": 1.0,
            "energy_watts": 1.0,
            "fronthaul_links": 1.0,
            "churn_events": 1.0,
            "qos_violations": 1.0,
            "tracking_violations": 1.0
        }
        
        self.x_curr = np.array([[1, 0], [0, 1]])
        self.y_curr = np.array([[1], [0]])
        self.x_cand = np.array([[0, 1], [1, 0]])
        self.y_cand = np.array([[0], [1]])
        
        self.curr_perf = {"combined_objective": 10.0}

    def test_decision_boundaries(self):
        # 1. predicted_gain < churn_cost => KEEP
        pred_perf_low = {"combined_objective": 11.0} # Gain = 1.0
        # raw churn = 4 links changed (2 off, 2 on)
        decision1 = churn_aware_decision(
            self.x_curr, self.y_curr, self.y_curr,
            self.x_cand, self.y_cand, self.y_cand,
            self.curr_perf, pred_perf_low,
            ref=self.ref, lambda_churn=1.0
        )
        self.assertTrue(decision1["predicted_gain"] < decision1["norm_churn_cost"])
        self.assertEqual(decision1["decision"], "KEEP")
        
        # 2. predicted_gain > churn_cost => RECONFIGURE
        pred_perf_high = {"combined_objective": 20.0} # Gain = 10.0
        decision2 = churn_aware_decision(
            self.x_curr, self.y_curr, self.y_curr,
            self.x_cand, self.y_cand, self.y_cand,
            self.curr_perf, pred_perf_high,
            ref=self.ref, lambda_churn=1.0
        )
        self.assertTrue(decision2["predicted_gain"] > decision2["norm_churn_cost"])
        self.assertEqual(decision2["decision"], "RECONFIGURE")
        
    def test_churn_weight_monotonicity(self):
        # 3. Increasing churn weight cannot make a fixed positive gain more likely to reconfigure
        pred_perf = {"combined_objective": 20.0} # Gain = 10.0
        
        d_low = churn_aware_decision(
            self.x_curr, self.y_curr, self.y_curr, self.x_cand, self.y_cand, self.y_cand,
            self.curr_perf, pred_perf, ref=self.ref, lambda_churn=1.0
        ) # Churn cost = 4.0, Net = 1.0 => RECONFIGURE
        
        d_high = churn_aware_decision(
            self.x_curr, self.y_curr, self.y_curr, self.x_cand, self.y_cand, self.y_cand,
            self.curr_perf, pred_perf, ref=self.ref, lambda_churn=2.0
        ) # Churn cost = 8.0, Net = -3.0 => KEEP
        
        self.assertEqual(d_low["decision"], "RECONFIGURE")
        self.assertEqual(d_high["decision"], "KEEP")
        self.assertTrue(d_high["norm_churn_cost"] > d_low["norm_churn_cost"])
        self.assertEqual(d_high["predicted_gain"], d_low["predicted_gain"])
        
    def test_candidate_objective_monotonicity(self):
        # 4. Increasing candidate objective cannot turn RECONFIGURE into KEEP
        pred_perf1 = {"combined_objective": 20.0}
        pred_perf2 = {"combined_objective": 25.0}
        
        d1 = churn_aware_decision(
            self.x_curr, self.y_curr, self.y_curr, self.x_cand, self.y_cand, self.y_cand,
            self.curr_perf, pred_perf1, ref=self.ref, lambda_churn=1.0
        )
        
        d2 = churn_aware_decision(
            self.x_curr, self.y_curr, self.y_curr, self.x_cand, self.y_cand, self.y_cand,
            self.curr_perf, pred_perf2, ref=self.ref, lambda_churn=1.0
        )
        
        self.assertEqual(d1["decision"], "RECONFIGURE")
        self.assertEqual(d2["decision"], "RECONFIGURE")
        self.assertTrue(d2["net_gain"] > d1["net_gain"])
        self.assertEqual(d1["norm_churn_cost"], d2["norm_churn_cost"])
        
    def test_raw_metrics_unchanged_by_normalization(self):
        # 5. Raw physical metrics are unchanged by normalization
        ap_pos = np.array([[0, 0], [10, 0]])
        user_pos = np.array([[5, 0]])
        targ_pos = np.array([[5, 5]])
        
        perf_ref_1 = evaluate_cluster_performance(
            self.x_curr, self.y_curr, self.y_curr, ap_pos, user_pos, targ_pos, ref=self.ref
        )
        
        ref_mod = self.ref.copy()
        ref_mod["communication_rate_bps"] = 99999.0
        perf_ref_mod = evaluate_cluster_performance(
            self.x_curr, self.y_curr, self.y_curr, ap_pos, user_pos, targ_pos, ref=ref_mod
        )
        
        # Raw metrics must be identical
        self.assertEqual(perf_ref_1["raw_communication_utility"], perf_ref_mod["raw_communication_utility"])
        self.assertEqual(perf_ref_1["raw_energy_penalty"], perf_ref_mod["raw_energy_penalty"])
        
        # Normalized metric must change
        self.assertNotEqual(perf_ref_1["norm_communication_utility"], perf_ref_mod["norm_communication_utility"])

if __name__ == "__main__":
    unittest.main()
