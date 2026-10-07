import unittest
import numpy as np

from tests.test_phase5b_joint_optimization import _solve, _SC

class TestPDFCompliance(unittest.TestCase):
    def test_default_behavior_unchanged(self):
        r_plain = _solve(_SC, alpha=1.0, beta=1.0)
        self.assertFalse(r_plain["fronthaul_constraint_enabled"])
        self.assertFalse(r_plain.get("tracking_constraint_enabled", False))

    def test_feasible_qos(self):
        r = _solve(_SC, alpha=1.0, beta=1.0, min_rate_bps=1.0, enforce_qos=True)
        # Should be feasible if we set min_rate to something very small
        self.assertTrue(r["feasible"])

    def test_infeasible_qos(self):
        r = _solve(_SC, alpha=1.0, beta=1.0, min_rate_bps=1e15, enforce_qos=True)
        # Unrealistic rate
        self.assertFalse(r["feasible"])

    def test_feasible_fronthaul(self):
        x = np.ones((_SC["H"].shape[0], _SC["H"].shape[1]))
        r = _solve(_SC, alpha=1.0, beta=1.0, fronthaul_capacity=1000.0, x=x)
        self.assertTrue(r["fronthaul_constraint_enabled"])
        self.assertTrue(r["fronthaul_constraint_satisfied"])
        self.assertTrue(r["feasible"])

    def test_infeasible_fronthaul(self):
        x = np.ones((_SC["H"].shape[0], _SC["H"].shape[1]))
        r = _solve(_SC, alpha=1.0, beta=1.0, fronthaul_capacity=0.0, x=x)
        self.assertTrue(r["fronthaul_constraint_enabled"])
        self.assertFalse(r["fronthaul_constraint_satisfied"])
        self.assertFalse(r["feasible"])

    def test_missing_x(self):
        with self.assertRaises(ValueError):
            _solve(_SC, alpha=1.0, beta=1.0, fronthaul_capacity=10.0, x=None)

if __name__ == "__main__":
    unittest.main()
