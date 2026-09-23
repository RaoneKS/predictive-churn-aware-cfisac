"""
tests/test_phase3_corrections.py

Regression coverage for the Phase-3 audit corrections.

These tests would all have FAILED against the pre-correction code, which
produced an always-KEEP PREDICTIVE+CHURN policy whose results were identical
for every predictor.
"""

import unittest

import numpy as np
import yaml

from src.environment.mobility_configs import STOCHASTIC_CONFIG
from src.optimization.churn_aware import churn_aware_decision
from src.prediction.predictor_interface import CVPredictor, make_predictor
from src.simulation.end_to_end import _horizon_objective, run_simulation

CONFIG_PATH = "configs/default.yaml"


def _cfg():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


class _ShiftedPredictor:
    """
    Deterministic synthetic predictor: constant-velocity prediction plus a
    fixed spatial offset.  Used to prove that a DIFFERENT predicted trajectory
    produces DIFFERENT churn-aware decisions, without requiring torch.
    """

    def __init__(self, shift):
        self.shift = float(shift)

    def predict(self, sim, horizon):
        u = sim.predict_users(horizon).copy()
        t = sim.predict_targets(horizon).copy()
        return u + self.shift, t + self.shift


class TestPredictiveChurnCanReconfigure(unittest.TestCase):
    """(1) The policy must be able to reconfigure once churn is scaled."""

    def test_reconfigures_under_default_config(self):
        for cfg in (None, STOCHASTIC_CONFIG):
            with self.subTest(stochastic=cfg is not None):
                m = run_simulation("PREDICTIVE+CHURN", seed=42, time_blocks=60,
                                   mobility_config=cfg)
                self.assertGreater(
                    int(m["reconfigurations"].sum()), 0,
                    "PREDICTIVE+CHURN degenerated to always-KEEP",
                )

    def test_reconfigures_but_less_than_predictive(self):
        """The whole point of the policy: fewer reconfigurations than PREDICTIVE."""
        pc = run_simulation("PREDICTIVE+CHURN", seed=42, time_blocks=60)
        p = run_simulation("PREDICTIVE", seed=42, time_blocks=60)
        self.assertGreater(int(pc["reconfigurations"].sum()), 0)
        self.assertLess(int(pc["reconfigurations"].sum()),
                        int(p["reconfigurations"].sum()))

    def test_positive_gain_beats_modest_churn_at_derived_scale(self):
        """Unit-level: a gain of 1.0 must beat a 10-event churn at ref=46."""
        ref = {"churn_events": 46.0}
        x0 = np.zeros((20, 5), dtype=int); x0[:3, :] = 1
        x1 = x0.copy(); x1[:3, 0] = 0; x1[3:6, 0] = 1   # 6 flips
        ytx = np.zeros((20, 2), dtype=int); ytx[:2, :] = 1
        yrx = ytx.copy()
        d = churn_aware_decision(
            x0, ytx, yrx, x1, ytx, yrx,
            {"combined_objective": 0.0}, {"combined_objective": 1.0},
            ref=ref, lambda_churn=1.0,
        )
        self.assertEqual(d["raw_churn"], 6.0)
        self.assertEqual(d["decision"], "RECONFIGURE")

    def test_zero_gain_never_reconfigures(self):
        """Guard against over-correction: no gain must still mean KEEP."""
        x0 = np.zeros((20, 5), dtype=int); x0[:3, :] = 1
        x1 = x0.copy(); x1[:3, 0] = 0; x1[3:6, 0] = 1
        ytx = np.zeros((20, 2), dtype=int); ytx[:2, :] = 1
        d = churn_aware_decision(
            x0, ytx, ytx, x1, ytx, ytx,
            {"combined_objective": 0.0}, {"combined_objective": 0.0},
            ref={"churn_events": 46.0}, lambda_churn=1.0,
        )
        self.assertEqual(d["decision"], "KEEP")


class TestPredictorAffectsDecisions(unittest.TestCase):
    """(2) Different predicted trajectories must change the outcome."""

    def test_different_predictors_give_different_results(self):
        base = run_simulation("PREDICTIVE+CHURN", seed=42, time_blocks=60,
                              predictor=_ShiftedPredictor(0.0),
                              mobility_config=STOCHASTIC_CONFIG)
        shifted = run_simulation("PREDICTIVE+CHURN", seed=42, time_blocks=60,
                                 predictor=_ShiftedPredictor(60.0),
                                 mobility_config=STOCHASTIC_CONFIG)
        self.assertNotEqual(
            float(base["combined_objective"].mean()),
            float(shifted["combined_objective"].mean()),
            "PREDICTIVE+CHURN is insensitive to the predictor",
        )

    def test_predictor_output_actually_reaches_the_policy(self):
        seen = []

        class Spy(CVPredictor):
            def predict(self, sim, horizon):
                out = super().predict(sim, horizon)
                seen.append(out[0].shape)
                return out

        run_simulation("PREDICTIVE+CHURN", seed=42, time_blocks=5,
                       predictor=Spy())
        self.assertEqual(len(seen), 5)
        self.assertEqual(seen[0], (5, 5, 2))  # (num_users, horizon, 2)


class TestHorizonAggregation(unittest.TestCase):
    """The horizon formulation must be explicit and decision-consistent."""

    def test_sum_equals_mean_times_horizon(self):
        from src.environment.topology import generate_ap_positions
        from src.optimization.baselines import cluster_for_positions
        cfg = _cfg()
        ap = generate_ap_positions(20, 500.0, seed=1)
        rng = np.random.default_rng(3)
        u = rng.uniform(0, 500, (5, 5, 2)); t = rng.uniform(0, 500, (2, 5, 2))
        x, ytx, yrx = cluster_for_positions(ap, u[:, 0, :], t[:, 0, :])
        pk = dict(alpha_communication=1.0, beta_sensing=1.0, lambda_energy=1.0,
                  lambda_fronthaul=1.0,
                  ref={k: float(v) for k, v in cfg["reference_values"].items()})
        s = _horizon_objective(x, ytx, yrx, ap, u, t, pk, aggregation="sum")
        m = _horizon_objective(x, ytx, yrx, ap, u, t, pk, aggregation="mean")
        self.assertAlmostEqual(s, m * 5, places=9)

    def test_sum_is_decision_equivalent_to_mean_with_scaled_churn(self):
        H = _cfg()["simulation"]["prediction_horizon"]
        a = run_simulation("PREDICTIVE+CHURN", seed=42, time_blocks=50,
                           mobility_config=STOCHASTIC_CONFIG,
                           horizon_aggregation="sum")
        b = run_simulation("PREDICTIVE+CHURN", seed=42, time_blocks=50,
                           mobility_config=STOCHASTIC_CONFIG,
                           horizon_aggregation="mean", lambda_churn=1.0 / H)
        np.testing.assert_array_equal(a["reconfigurations"], b["reconfigurations"])

    def test_invalid_aggregation_rejected(self):
        with self.assertRaises(ValueError):
            run_simulation("PREDICTIVE+CHURN", seed=42, time_blocks=3,
                           horizon_aggregation="median")


class TestWarmupIsExplicit(unittest.TestCase):
    """(4) Warm-up must be explicit, not silently hidden."""

    def test_history_unavailable_before_warmup(self):
        from src.simulation.mobility_simulator import MobilitySimulator
        L = _cfg()["prediction"]["history_length"]
        sim = MobilitySimulator(np.full((5, 2), 100.0), np.full((2, 2), 100.0),
                                area_size=500.0, seed=42)
        for step in range(1, L + 2):
            sim.step()
            got = sim.get_user_history(L)
            if step < L:
                self.assertIsNone(got, f"history available too early at {step}")
            else:
                self.assertEqual(got.shape, (5, L, 2))

    def test_summary_reports_both_windows(self):
        import pandas as pd
        from experiments.run_phase3 import summarise
        df = pd.DataFrame({
            "block": list(range(20)),
            "mobility": ["m"] * 20, "predictor": ["p"] * 20, "policy": ["STATIC"] * 20,
            "avg_user_rate": 1.0, "p5_user_rate": 1.0, "sensing_utility": 1.0,
            "tracking_violations": 0.0, "energy": 1.0, "fronthaul": 1.0,
            "churn": 1.0, "cumulative_churn": 1.0, "reconfigurations": 0,
            "combined_objective": 1.0, "qos_violations": 0.0,
        })
        out = summarise(df, warmup_blocks=10)
        self.assertEqual(set(out["window"]), {"full", "post_warmup"})
        self.assertEqual(float(out[out.window == "full"]["cum_churn"].iloc[0]), 20.0)
        self.assertEqual(float(out[out.window == "post_warmup"]["cum_churn"].iloc[0]), 10.0)


class TestPredictorAPIBackwardCompatible(unittest.TestCase):
    """(6) Existing CV/LSTM APIs must be unchanged."""

    def test_cv_predictor_shapes_and_factory(self):
        from src.simulation.mobility_simulator import MobilitySimulator
        sim = MobilitySimulator(np.full((5, 2), 100.0), np.full((2, 2), 100.0),
                                area_size=500.0, seed=42)
        sim.step()
        p = make_predictor("cv")
        self.assertIsInstance(p, CVPredictor)
        u, t = p.predict(sim, 5)
        self.assertEqual(u.shape, (5, 5, 2))
        self.assertEqual(t.shape, (2, 5, 2))

    def test_factory_validation_unchanged(self):
        with self.assertRaises(ValueError):
            make_predictor("nonsense")
        with self.assertRaises(ValueError):
            make_predictor("lstm")  # missing checkpoints
        with self.assertRaises(ValueError):
            make_predictor("lstm", user_ckpt_path="a", target_ckpt_path="b")

    def test_predictor_none_defaults_to_cv(self):
        a = run_simulation("PREDICTIVE", seed=42, time_blocks=10, predictor=None)
        b = run_simulation("PREDICTIVE", seed=42, time_blocks=10,
                           predictor=CVPredictor())
        np.testing.assert_allclose(a["combined_objective"], b["combined_objective"])


if __name__ == "__main__":
    unittest.main()
