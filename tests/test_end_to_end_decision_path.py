"""
tests/test_end_to_end_decision_path.py

Regression coverage for the ACTUAL executed PREDICTIVE+CHURN decision path in
src/simulation/end_to_end.py.

Motivation
----------
Prior to the Phase-3 audit, end_to_end.py imported churn_aware_decision() but
re-implemented the KEEP/RECONFIGURE rule inline.  Every existing test exercised
the unused library function, so the executed path had zero coverage.  These
tests assert that the simulation genuinely delegates to churn_aware_decision().
"""

import unittest
from unittest.mock import patch

import numpy as np

from src.environment.mobility_configs import STOCHASTIC_CONFIG
from src.optimization.churn_aware import churn_aware_decision
from src.simulation.end_to_end import run_simulation


class TestSimulationUsesChurnAwareDecision(unittest.TestCase):
    """The simulation must delegate to the tested decision function."""

    def test_predictive_churn_calls_churn_aware_decision(self):
        calls = []

        def spy(*args, **kwargs):
            result = churn_aware_decision(*args, **kwargs)
            calls.append(result)
            return result

        with patch("src.simulation.end_to_end.churn_aware_decision", side_effect=spy):
            run_simulation("PREDICTIVE+CHURN", seed=42, time_blocks=6)

        # 6 blocks => t=0 is the bootstrap, t=1..5 are decisions
        self.assertEqual(len(calls), 5)
        for c in calls:
            self.assertIn(c["decision"], ("KEEP", "RECONFIGURE"))
            self.assertIn("net_gain", c)
            self.assertIn("raw_churn", c)

    def test_other_policies_do_not_call_decision_function(self):
        for policy in ("STATIC", "REACTIVE", "PREDICTIVE"):
            with self.subTest(policy=policy):
                calls = []
                with patch("src.simulation.end_to_end.churn_aware_decision",
                           side_effect=lambda *a, **k: calls.append(1)):
                    run_simulation(policy, seed=42, time_blocks=5)
                self.assertEqual(len(calls), 0)

    def test_decision_drives_configuration_state(self):
        """A RECONFIGURE decision must change the applied configuration."""
        decisions = []

        def spy(*args, **kwargs):
            r = churn_aware_decision(*args, **kwargs)
            decisions.append(r["decision"])
            return r

        with patch("src.simulation.end_to_end.churn_aware_decision", side_effect=spy):
            m = run_simulation("PREDICTIVE+CHURN", seed=42, time_blocks=20,
                               mobility_config=STOCHASTIC_CONFIG)

        n_reconfig_decisions = decisions.count("RECONFIGURE")
        measured_reconfigs = int(m["reconfigurations"].sum())
        # Every measured reconfiguration must correspond to a RECONFIGURE
        # decision (the converse can fail only if a candidate is identical to
        # the incumbent, which cluster_difference would score as 0).
        self.assertLessEqual(measured_reconfigs, n_reconfig_decisions)
        self.assertEqual(m["keep_decisions"] + m["reconfigure_decisions"], 19)


class TestStaticRemainsZeroChurn(unittest.TestCase):
    def test_static_has_no_churn(self):
        for cfg in (None, STOCHASTIC_CONFIG):
            with self.subTest(mobility=cfg is not None):
                m = run_simulation("STATIC", seed=42, time_blocks=25,
                                   mobility_config=cfg)
                self.assertEqual(float(m["churn"].sum()), 0.0)
                self.assertEqual(int(m["reconfigurations"].sum()), 0)


if __name__ == "__main__":
    unittest.main()
