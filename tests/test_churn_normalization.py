"""
tests/test_churn_normalization.py

Guards the churn normalization constant against silent drift.

reference_values.churn_events must equal the maximum feasible raw weighted
churn for one reconfiguration under the current configuration, so that
normalized churn lies in [0, 1] and lambda_churn is a pure preference weight.
"""

import unittest

import numpy as np
import yaml

from src.clustering.churn import max_raw_churn_cost, total_churn_cost
from src.environment.topology import generate_ap_positions
from src.optimization.baselines import cluster_for_positions

CONFIG_PATH = "configs/default.yaml"

# Clustering cardinalities are the defaults of
# src.optimization.baselines.cluster_for_positions.
APS_PER_USER      = 3
TX_APS_PER_TARGET = 2
RX_APS_PER_TARGET = 2


class TestChurnNormalization(unittest.TestCase):

    def setUp(self):
        with open(CONFIG_PATH) as f:
            self.cfg = yaml.safe_load(f)

    def test_derived_max_matches_hand_derivation(self):
        self.assertEqual(
            max_raw_churn_cost(5, 2, 3, 2, 2, num_aps=20),
            2 * 3 * 5 + 2 * 2 * 2 + 2 * 2 * 2,  # 30 + 8 + 8
        )
        self.assertEqual(max_raw_churn_cost(5, 2, 3, 2, 2, num_aps=20), 46.0)

    def test_config_reference_equals_derived_max(self):
        net = self.cfg["network"]
        churn = self.cfg["churn"]
        expected = max_raw_churn_cost(
            num_users=net["num_users"],
            num_targets=net["num_targets"],
            aps_per_user=APS_PER_USER,
            tx_aps_per_target=TX_APS_PER_TARGET,
            rx_aps_per_target=RX_APS_PER_TARGET,
            num_aps=net["num_aps"],
            c_comm=churn["communication_cost"],
            c_sensing_tx=churn["sensing_tx_cost"],
            c_sensing_rx=churn["sensing_rx_cost"],
        )
        self.assertAlmostEqual(
            float(self.cfg["reference_values"]["churn_events"]), expected,
            msg="churn_events reference is out of sync with the derived bound",
        )

    def test_bound_is_never_exceeded_in_practice(self):
        """Empirical check: no reconfiguration can cost more than the bound."""
        net = self.cfg["network"]
        bound = float(self.cfg["reference_values"]["churn_events"])
        ap = generate_ap_positions(net["num_aps"], net["area_size"], seed=1)
        rng = np.random.default_rng(0)
        worst = 0.0
        for _ in range(200):
            u1 = rng.uniform(0, net["area_size"], (net["num_users"], 2))
            t1 = rng.uniform(0, net["area_size"], (net["num_targets"], 2))
            u2 = rng.uniform(0, net["area_size"], (net["num_users"], 2))
            t2 = rng.uniform(0, net["area_size"], (net["num_targets"], 2))
            a = cluster_for_positions(ap, u1, t1)
            b = cluster_for_positions(ap, u2, t2)
            worst = max(worst, total_churn_cost(a[0], b[0], a[1], b[1], a[2], b[2]))
        self.assertLessEqual(worst, bound)

    def test_lambda_churn_not_weakened(self):
        """The correction must be normalization only, not penalty softening."""
        self.assertEqual(float(self.cfg["objective"]["lambda_churn"]), 1.0)


if __name__ == "__main__":
    unittest.main()
