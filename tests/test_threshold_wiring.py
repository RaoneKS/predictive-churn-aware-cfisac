"""Regression tests for configured QoS and tracking threshold propagation."""

import unittest
from unittest.mock import patch

import src.simulation.end_to_end as end_to_end


class TestThresholdWiring(unittest.TestCase):
    def _run_and_capture_thresholds(self, **overrides):
        captured = []
        original_evaluator = end_to_end.evaluate_cluster_performance

        def evaluator_spy(*args, **kwargs):
            captured.append({
                "min_rate": kwargs.get("min_rate"),
                "max_tracking_error": kwargs.get("max_tracking_error"),
            })
            return original_evaluator(*args, **kwargs)

        with patch(
            "src.simulation.end_to_end.evaluate_cluster_performance",
            side_effect=evaluator_spy,
        ):
            end_to_end.run_simulation(
                "STATIC",
                seed=42,
                time_blocks=2,
                **overrides,
            )

        self.assertTrue(
            captured,
            "The simulation did not call the performance evaluator.",
        )
        return captured

    def test_configured_thresholds_reach_evaluator(self):
        calls = self._run_and_capture_thresholds()

        for call in calls:
            with self.subTest(call=call):
                self.assertEqual(call["min_rate"], 1_000_000.0)
                self.assertEqual(call["max_tracking_error"], 1.0)

    def test_explicit_threshold_overrides_are_respected(self):
        calls = self._run_and_capture_thresholds(
            min_rate=2_000_000,
            max_tracking_error=0.75,
        )

        for call in calls:
            with self.subTest(call=call):
                self.assertEqual(call["min_rate"], 2_000_000.0)
                self.assertEqual(call["max_tracking_error"], 0.75)


if __name__ == "__main__":
    unittest.main()
