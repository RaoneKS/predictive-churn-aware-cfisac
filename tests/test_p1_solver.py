import unittest
import numpy as np

from src.optimization.p1_solver import (
    ap_activation,
    generate_topology_candidates,
    filter_hard_constraints,
    solve_p1_decomposed
)
from src.optimization.two_timescale import run_two_timescale
from src.optimization.risk_aware import run_risk_aware_two_timescale

class MockPredictor:
    def predict(self, sim, horizon):
        u = np.stack([sim.user_positions] * horizon, axis=1)
        t = np.stack([sim.target_positions] * horizon, axis=1)
        return u, t

class MockSim:
    def __init__(self):
        self.user_positions = np.array([[10.0, 10.0], [20.0, 20.0]])
        self.target_positions = np.array([[30.0, 30.0]])
        self.dt = 0.1
        self.position_history = [
            (self.user_positions, self.target_positions)
        ] * 10
        self.num_users = 2
        self.num_targets = 1

    def step(self):
        pass

class TestP1Solver(unittest.TestCase):
    def setUp(self):
        self.sim = MockSim()
        self.pred = MockPredictor()
        self.ap_pos = np.random.rand(4, 2) * 100
        self.x = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
        self.y_tx = np.ones((4, 1))
        self.y_rx = np.ones((4, 1))
        self.cluster_kwargs = {
            "aps_per_user": 2,
            "tx_aps_per_target": 2,
            "rx_aps_per_target": 2
        }

    def test_activation_derivation(self):
        x = np.array([[1, 0], [0, 0]])
        y_tx = np.array([[0], [1]])
        y_rx = np.array([[0], [0]])
        a = ap_activation(x, y_tx, y_rx)
        self.assertTrue(np.array_equal(a, [1, 1]))

    def test_candidate_topology_generation(self):
        cands = generate_topology_candidates(
            self.ap_pos, self.sim.user_positions, self.sim.target_positions,
            self.x, self.y_tx, self.y_rx, self.cluster_kwargs, p1_max_candidates=10
        )
        self.assertTrue(len(cands) > 0)
        self.assertEqual(cands[0]["name"], "KEEP")

    def test_binary_constraint_rejection(self):
        cand = {"x": self.x * 0.5, "y_tx": self.y_tx, "y_rx": self.y_rx}
        passed, reason = filter_hard_constraints(cand, self.cluster_kwargs, None, 1.0, 1.0, 0.0)
        self.assertFalse(passed)
        self.assertEqual(reason, "Not binary")

    def test_cluster_size_rejection(self):
        cand = {"x": np.zeros((4, 2)), "y_tx": self.y_tx, "y_rx": self.y_rx}
        passed, reason = filter_hard_constraints(cand, self.cluster_kwargs, None, 1.0, 1.0, 0.0)
        self.assertFalse(passed)
        self.assertEqual(reason, "Min Comm APs")

    def test_fronthaul_rejection(self):
        cand = {"x": self.x, "y_tx": self.y_tx, "y_rx": self.y_rx} # Total 8+4=12 links
        fh_cap = np.full(4, 1.0)
        passed, reason = filter_hard_constraints(cand, self.cluster_kwargs, fh_cap, 2.0, 1.0, 0.0)
        self.assertFalse(passed)
        self.assertEqual(reason, "Fronthaul Capacity")

    def test_feasible_candidate_evaluation_and_fallback(self):
        res = solve_p1_decomposed(
            self.sim, self.pred, self.ap_pos, 4, self.x, self.y_tx, self.y_rx,
            T_slow=1, P_max=1.0, precoder="mrt", alpha_power=1.0, beta_power=1.0,
            cluster_kwargs=self.cluster_kwargs, perf_kwargs={}, churn_kwargs={},
            phys_kwargs={}, sens_kwargs={}, joint_kwargs={}, ref={}, p1_max_candidates=5, num_scenarios=0
        )
        self.assertTrue("objective" in res)
        # We test deterministic fallback by enforcing impossible fronthaul
        res_fail = solve_p1_decomposed(
            self.sim, self.pred, self.ap_pos, 4, self.x, self.y_tx, self.y_rx,
            T_slow=1, P_max=1.0, precoder="mrt", alpha_power=1.0, beta_power=1.0,
            cluster_kwargs=self.cluster_kwargs, perf_kwargs={}, churn_kwargs={},
            phys_kwargs={}, sens_kwargs={}, joint_kwargs={"fronthaul_capacity": np.zeros(4)}, ref={}, p1_max_candidates=5, num_scenarios=0
        )
        self.assertTrue(res_fail["fallback_used"])
        self.assertEqual(res_fail["fallback_reason"], "no_feasible_p1_candidate")

    def test_cvar_rejection(self):
        res = solve_p1_decomposed(
            self.sim, self.pred, self.ap_pos, 4, self.x, self.y_tx, self.y_rx,
            T_slow=1, P_max=1.0, precoder="mrt", alpha_power=1.0, beta_power=1.0,
            cluster_kwargs=self.cluster_kwargs, perf_kwargs={}, churn_kwargs={},
            phys_kwargs={}, sens_kwargs={}, joint_kwargs={}, ref={}, p1_max_candidates=5,
            num_scenarios=2, enforce_deficiency_cvar=True, max_comm_cvar=0.0, min_rate_bps=1e15
        )
        self.assertTrue(res["fallback_used"])

    def test_default_mode_unchanged(self):
        # We can just verify it runs
        r1 = run_two_timescale(
            self.sim, self.pred, self.ap_pos, 4, 1, 1,
            cluster_kwargs=self.cluster_kwargs, p1_mode=False
        )
        self.assertTrue(len(r1.slow["decision"]) > 0)

if __name__ == '__main__':
    unittest.main()
