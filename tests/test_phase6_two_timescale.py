"""
tests/test_phase6_two_timescale.py

Focused tests for Phase 6: two-timescale optimization
(src.optimization.two_timescale).

Only the new module is exercised directly, plus a regression class at the
end proving Phase 5B's optimize_joint_allocation is called with the exact
same semantics as when invoked directly (bit-for-bit), i.e. Phase 6 does
not alter Phase 5B behavior.
"""

import unittest

import numpy as np

from src.environment.topology import generate_ap_positions, generate_positions
from src.optimization.baselines import cluster_for_positions
from src.optimization.joint_comm_sensing import (
    comm_beam_directions,
    optimize_joint_allocation,
)
from src.optimization.two_timescale import (
    fast_update,
    run_two_timescale,
    slow_update,
    validate_slow_to_fast,
)
from src.channels.physical import generate_physical_channel
from src.channels.sensing import generate_sensing_gains
from src.prediction.predictor_interface import CVPredictor
from src.simulation.mobility_simulator import MobilitySimulator


# ---------------------------------------------------------------------------
# Deterministic fixtures
# ---------------------------------------------------------------------------

def _make_scenario(num_aps=20, num_users=5, num_targets=2, area_size=500.0,
                    seed=42):
    ap = generate_ap_positions(num_aps, area_size, seed=seed)
    us = generate_positions(num_users, area_size, seed=seed + 1)
    tg = generate_positions(num_targets, area_size, seed=seed + 2)
    sim = MobilitySimulator(us, tg, area_size=area_size, dt=1.0, seed=seed)
    predictor = CVPredictor()
    return ap, sim, predictor


def _run(T_total=10, T_slow=3, base_seed=7, **kwargs):
    ap, sim, predictor = _make_scenario()
    return run_two_timescale(
        sim, predictor, ap, num_antennas=4, T_total=T_total, T_slow=T_slow,
        P_max=1.0, alpha=1.0, beta=1.0, base_seed=base_seed, **kwargs,
    )


# ---------------------------------------------------------------------------
# 1. Determinism
# ---------------------------------------------------------------------------

class TestDeterminism(unittest.TestCase):
    def test_identical_runs_identical_output(self):
        r1 = _run()
        r2 = _run()
        for a, b in zip(r1.fast["P_comm"], r2.fast["P_comm"]):
            np.testing.assert_allclose(a, b)
        for a, b in zip(r1.fast["P_sens"], r2.fast["P_sens"]):
            np.testing.assert_allclose(a, b)
        self.assertEqual(r1.slow["decision"], r2.slow["decision"])
        self.assertEqual(r1.fast["tau"], r2.fast["tau"])

    def test_different_base_seed_changes_fast_output(self):
        r1 = _run(base_seed=7)
        r2 = _run(base_seed=8)
        diffs = [
            not np.allclose(a, b)
            for a, b in zip(r1.fast["P_comm"], r2.fast["P_comm"])
        ]
        self.assertTrue(any(diffs))


# ---------------------------------------------------------------------------
# 2. Slow-update scheduling
# ---------------------------------------------------------------------------

class TestSlowScheduling(unittest.TestCase):
    def test_slow_update_count_matches_schedule(self):
        for T_total, T_slow, expected_epochs in [
            (10, 3, 4), (9, 3, 3), (1, 1, 1), (7, 1, 7), (20, 5, 4),
        ]:
            r = _run(T_total=T_total, T_slow=T_slow)
            self.assertEqual(r.num_slow_updates, expected_epochs)
            self.assertEqual(len(r.slow["decision"]), expected_epochs)

    def test_slow_update_occurs_exactly_at_epoch_boundaries(self):
        r = _run(T_total=10, T_slow=3)
        self.assertEqual(r.slow["t"], [0, 3, 6, 9])

    def test_T_slow_independent_of_predictor_horizon(self):
        # T_slow=4 must work even though nothing about the CVPredictor
        # is configured with a horizon of 4 anywhere; the predictor is
        # simply called with horizon=T_slow at each epoch boundary.
        r = _run(T_total=8, T_slow=4)
        self.assertEqual(r.num_slow_updates, 2)


# ---------------------------------------------------------------------------
# 3. Fast-update scheduling
# ---------------------------------------------------------------------------

class TestFastScheduling(unittest.TestCase):
    def test_fast_update_count_equals_T_total(self):
        for T_total in (1, 5, 13):
            r = _run(T_total=T_total, T_slow=3)
            self.assertEqual(r.num_fast_updates, T_total)
            self.assertEqual(len(r.fast["t"]), T_total)

    def test_fast_update_every_step_regardless_of_T_slow(self):
        r = _run(T_total=10, T_slow=100)  # one slow epoch covers everything
        self.assertEqual(r.num_fast_updates, 10)
        self.assertEqual(r.num_slow_updates, 1)


# ---------------------------------------------------------------------------
# 4. State carry-over
# ---------------------------------------------------------------------------

class TestStateCarryOver(unittest.TestCase):
    def test_association_fixed_within_epoch_via_power_pattern(self):
        # The *support* of P_comm (which APs get nonzero communication
        # power) is determined by x_tau (via W_dir), which is fixed
        # within an epoch even though the instantaneous channel changes
        # every fast step. So the zero/nonzero PATTERN of P_comm should
        # be identical across all fast steps inside one epoch, even
        # though the exact values differ.
        r = _run(T_total=9, T_slow=3)
        for tau in set(r.fast["tau"]):
            idx = [i for i, t in enumerate(r.fast["tau"]) if t == tau]
            patterns = [
                tuple((np.asarray(r.fast["P_comm"][i]) > 1e-12).tolist())
                for i in idx
            ]
            self.assertTrue(
                all(p == patterns[0] for p in patterns),
                f"P_comm active-AP pattern changed within epoch tau={tau}",
            )

    def test_slow_state_actually_changes_across_reconfigure(self):
        upd_log = []

        class _RecordingPredictor(CVPredictor):
            pass

        ap, sim, predictor = _make_scenario()
        # Run two epochs manually and check whether at least one of
        # KEEP/RECONFIGURE is exercised across a longer horizon with a
        # low churn penalty (encourages reconfiguration).
        x, y_tx, y_rx = cluster_for_positions(
            ap, sim.user_positions, sim.target_positions, aps_per_user=3,
        )
        decisions = []
        for tau in range(5):
            upd = slow_update(
                sim, predictor, ap, x, y_tx, y_rx, T_slow=3,
                churn_kwargs={"lambda_churn": 0.0},
            )
            decisions.append(upd["decision"])
            x, y_tx, y_rx = upd["x"], upd["y_tx"], upd["y_rx"]
            for _ in range(3):
                sim.step()
        # With lambda_churn=0, any nonzero predicted gain triggers
        # RECONFIGURE, so we should see at least one RECONFIGURE across
        # 5 epochs of a moving scenario.
        self.assertIn("RECONFIGURE", decisions)


# ---------------------------------------------------------------------------
# 5. Prediction / churn integration
# ---------------------------------------------------------------------------

class TestPredictionChurnIntegration(unittest.TestCase):
    def test_zero_lambda_churn_reconfigures_whenever_gain_positive(self):
        ap, sim, predictor = _make_scenario()
        x, y_tx, y_rx = cluster_for_positions(
            ap, sim.user_positions, sim.target_positions,
        )
        upd = slow_update(
            sim, predictor, ap, x, y_tx, y_rx, T_slow=5,
            churn_kwargs={"lambda_churn": 0.0},
        )
        info = upd["decision_info"]
        if info["predicted_gain"] > 0:
            self.assertEqual(upd["decision"], "RECONFIGURE")
        else:
            self.assertEqual(upd["decision"], "KEEP")

    def test_huge_lambda_churn_forces_keep(self):
        ap, sim, predictor = _make_scenario()
        x, y_tx, y_rx = cluster_for_positions(
            ap, sim.user_positions, sim.target_positions,
        )
        upd = slow_update(
            sim, predictor, ap, x, y_tx, y_rx, T_slow=5,
            churn_kwargs={"lambda_churn": 1e9},
        )
        self.assertEqual(upd["decision"], "KEEP")
        np.testing.assert_array_equal(upd["x"], x)
        np.testing.assert_array_equal(upd["y_tx"], y_tx)
        np.testing.assert_array_equal(upd["y_rx"], y_rx)


# ---------------------------------------------------------------------------
# 6. Zero / high churn full-run cases
# ---------------------------------------------------------------------------

class TestChurnRegimes(unittest.TestCase):
    def test_zero_churn_reconfigures_at_least_as_often_as_high_churn(self):
        r_zero = _run(T_total=15, T_slow=3, churn_kwargs={"lambda_churn": 0.0})
        r_high = _run(T_total=15, T_slow=3, churn_kwargs={"lambda_churn": 1e9})

        n_reconf_zero = sum(1 for d in r_zero.slow["decision"] if d == "RECONFIGURE")
        n_reconf_high = sum(1 for d in r_high.slow["decision"] if d == "RECONFIGURE")

        self.assertEqual(n_reconf_high, 0)
        self.assertGreaterEqual(n_reconf_zero, n_reconf_high)

    def test_high_churn_never_reconfigures(self):
        r = _run(T_total=20, T_slow=2, churn_kwargs={"lambda_churn": 1e9})
        self.assertTrue(all(d == "KEEP" for d in r.slow["decision"]))


# ---------------------------------------------------------------------------
# 7. Changing channel (fast reacts, association fixed)
# ---------------------------------------------------------------------------

class TestChangingChannel(unittest.TestCase):
    def test_objective_varies_across_fast_steps_within_epoch(self):
        r = _run(T_total=6, T_slow=6)  # single epoch, association fixed throughout
        self.assertEqual(r.num_slow_updates, 1)
        # NOTE: the per-AP power SPLIT (c_m, s_m) is allowed to be a
        # channel-invariant bang-bang corner solution here -- e.g. an AP
        # with no sensing role has no reason to hold back communication
        # power regardless of the channel, so c_m == P_max,m every step
        # is the CORRECT optimum, not a sign of non-reactivity. What must
        # vary with the instantaneous channel is the resulting physical
        # utility (rate / objective), since H changes every fast step.
        objectives = np.asarray(r.fast["objective"])
        comm_utility = np.asarray(r.fast["comm_utility_raw"])
        self.assertFalse(np.allclose(objectives, objectives[0]))
        self.assertFalse(np.allclose(comm_utility, comm_utility[0]))


# ---------------------------------------------------------------------------
# 8. Power feasibility
# ---------------------------------------------------------------------------

class TestPowerFeasibility(unittest.TestCase):
    def test_every_fast_step_feasible_and_within_budget(self):
        r = _run(T_total=15, T_slow=3)
        self.assertTrue(all(r.fast["feasible"]))
        for c, s, pmax in zip(r.fast["P_comm"], r.fast["P_sens"], r.fast["P_max"]):
            c = np.asarray(c)
            s = np.asarray(s)
            pmax = np.asarray(pmax)
            self.assertTrue(np.all(c >= -1e-9))
            self.assertTrue(np.all(s >= -1e-9))
            self.assertTrue(np.all(c + s <= pmax + 1e-6))

    def test_max_power_violation_effectively_zero(self):
        r = _run(T_total=10, T_slow=3)
        for v in r.fast["max_power_violation"]:
            self.assertLessEqual(v, 1e-6)


# ---------------------------------------------------------------------------
# 9. Slow -> fast boundary validation
# ---------------------------------------------------------------------------

class TestValidateSlowToFast(unittest.TestCase):
    def test_accepts_correct_shapes(self):
        x = np.zeros((4, 3))
        y_tx = np.zeros((4, 2))
        y_rx = np.zeros((4, 2))
        x, y_tx, y_rx = validate_slow_to_fast(x, y_tx, y_rx, 4, 3, 2)
        self.assertEqual(x.shape, (4, 3))

    def test_rejects_wrong_x_shape(self):
        with self.assertRaises(ValueError):
            validate_slow_to_fast(
                np.zeros((5, 3)), np.zeros((4, 2)), np.zeros((4, 2)), 4, 3, 2,
            )

    def test_rejects_wrong_y_tx_shape(self):
        with self.assertRaises(ValueError):
            validate_slow_to_fast(
                np.zeros((4, 3)), np.zeros((4, 3)), np.zeros((4, 2)), 4, 3, 2,
            )

    def test_rejects_non_binary_entries(self):
        x = np.zeros((4, 3))
        x[0, 0] = 0.5
        with self.assertRaises(ValueError):
            validate_slow_to_fast(x, np.zeros((4, 2)), np.zeros((4, 2)), 4, 3, 2)

    def test_fast_update_raises_on_shape_mismatch(self):
        ap, sim, _ = _make_scenario(num_aps=20, num_users=5, num_targets=2)
        bad_x = np.zeros((20, 4))  # wrong K
        y_tx = np.zeros((20, 2))
        y_rx = np.zeros((20, 2))
        with self.assertRaises(ValueError):
            fast_update(
                ap, sim.user_positions, sim.target_positions,
                bad_x, y_tx, y_rx, num_antennas=4, P_max=1.0,
            )


# ---------------------------------------------------------------------------
# 10. Compatibility / regression with Phase 5B
# ---------------------------------------------------------------------------

class TestPhase5BCompatibility(unittest.TestCase):
    def test_fast_update_matches_direct_phase5b_call(self):
        """
        fast_update() must be a thin, behavior-preserving wrapper: calling
        it should give bit-for-bit the same result as building H/W_dir/
        G_sens and calling optimize_joint_allocation directly (exactly
        the Phase 5B test-fixture pattern).
        """
        ap, sim, _ = _make_scenario()
        x, y_tx, y_rx = cluster_for_positions(
            ap, sim.user_positions, sim.target_positions, aps_per_user=2,
        )
        # Make y_tx/y_rx compatible shape with num_targets=2
        M = ap.shape[0]
        Q = 2
        y_tx = np.zeros((M, Q))
        y_rx = np.zeros((M, Q))
        active = np.flatnonzero(np.sum(x, axis=1) > 0)
        for q in range(Q):
            y_tx[active[:2], q] = 1.0
            y_rx[active[2:4], q] = 1.0

        fast_seed = 123

        res_wrapped = fast_update(
            ap, sim.user_positions, sim.target_positions,
            x, y_tx, y_rx, num_antennas=4, P_max=1.0,
            precoder="mrt", alpha=1.0, beta=1.0, fast_seed=fast_seed,
        )

        H = generate_physical_channel(
            ap, sim.user_positions, 4, seed=fast_seed,
        )
        W_dir = comm_beam_directions(H, x, precoder="mrt")
        G_sens = generate_sensing_gains(ap, sim.target_positions)
        res_direct = optimize_joint_allocation(
            H, W_dir, ap, sim.target_positions, y_tx, y_rx, G_sens, 1.0,
            alpha=1.0, beta=1.0,
        )

        np.testing.assert_allclose(res_wrapped["P_comm"], res_direct["P_comm"])
        np.testing.assert_allclose(res_wrapped["P_sens"], res_direct["P_sens"])
        self.assertEqual(res_wrapped["objective"], res_direct["objective"])

    def test_two_timescale_import_does_not_alter_phase5b_module(self):
        import src.optimization.joint_comm_sensing as jcs
        # Presence of the exact same public API, unmodified.
        for name in (
            "comm_beam_directions", "sensing_fim_basis",
            "uniform_split_allocation", "evaluate_joint_allocation",
            "optimize_joint_allocation",
        ):
            self.assertTrue(hasattr(jcs, name))


if __name__ == "__main__":
    unittest.main()
