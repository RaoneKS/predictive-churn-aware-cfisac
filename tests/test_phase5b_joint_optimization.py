"""
tests/test_phase5b_joint_optimization.py

Focused tests for Phase 5B: joint communication + sensing per-AP power
allocation (src.optimization.joint_comm_sensing).

Only the new module is exercised. The frozen Phase 4A (MRT/RZF, physical
SINR/rate), Phase 4B (constrained beamforming) and Phase 5A (physical
sensing bridge) code is used read-only, and regression classes at the end
prove that it is bit-for-bit unchanged by running Phase 5B.
"""

import unittest

import numpy as np

from src.beamforming.constrained import constrained_sinr_beamforming
from src.beamforming.precoders import (
    mrt_weights,
    normalize_per_ap_power,
    rzf_weights,
)
from src.channels.communication import generate_comm_gains
from src.channels.physical import generate_physical_channel
from src.channels.sensing import generate_sensing_gains
from src.clustering.overlapping import communication_clustering
from src.environment.topology import generate_ap_positions, generate_positions
from src.metrics.physical_layer import physical_rate, physical_sinr
from src.metrics.physical_sensing import (
    aggregate_fisher_information_bridge,
    comm_power_per_ap,
    evaluate_physical_sensing_bridge,
    physical_tx_power_per_ap,
)
from src.metrics.sensing import aggregate_fisher_information
from src.optimization.joint_comm_sensing import (
    _JointProblem,
    comm_beam_directions,
    evaluate_joint_allocation,
    optimize_joint_allocation,
    sensing_fim_basis,
    uniform_split_allocation,
)


# ---------------------------------------------------------------------------
# Deterministic fixtures
# ---------------------------------------------------------------------------

def _scenario(
    num_aps=20,
    num_users=5,
    num_targets=2,
    num_antennas=4,
    area_size=500.0,
    seed=42,
    precoder="mrt",
    p_max=1.0,
):
    """
    Same construction as the Phase 5A validation experiment: 2 APs per user
    for communication; for every target the first two comm-active APs are
    sensing TX and the next two are sensing RX (so TX APs are dual-role).
    """
    ap = generate_ap_positions(
        num_aps,
        area_size,
        seed=seed,
    )
    us = generate_positions(
        num_users,
        area_size,
        seed=seed + 1,
    )
    tg = generate_positions(
        num_targets,
        area_size,
        seed=seed + 2,
    )

    G_comm = generate_comm_gains(
        ap,
        us,
    )

    x = communication_clustering(
        G_comm,
        num_users=num_users,
        aps_per_user=2,
    )

    H = generate_physical_channel(
        ap,
        us,
        num_antennas=num_antennas,
        antenna_spacing=0.5,
        rician_k_factor=10.0,
        small_scale_fading=True,
        seed=seed,
    )

    W_dir = comm_beam_directions(
        H,
        x,
        precoder=precoder,
    )

    active = np.flatnonzero(
        np.sum(np.abs(W_dir) ** 2, axis=(1, 2)) > 1e-12
    )

    y_tx = np.zeros(
        (num_aps, num_targets)
    )
    y_rx = np.zeros(
        (num_aps, num_targets)
    )

    for q in range(num_targets):
        y_tx[active[:2], q] = 1.0
        y_rx[active[2:4], q] = 1.0

    G_sens = generate_sensing_gains(
        ap,
        tg,
        rcs=np.ones(num_targets),
        pathloss_exponent=2.0,
    )

    return dict(
        H=H,
        W_dir=W_dir,
        ap_positions=ap,
        target_positions=tg,
        y_tx=y_tx,
        y_rx=y_rx,
        G_sens=G_sens,
        P_max=p_max,
        _x=x,
        _active=active,
        _M=num_aps,
        _K=num_users,
        _Q=num_targets,
        _N=num_antennas,
    )


def _args(sc):
    """Keyword arguments accepted by optimize/evaluate."""
    return {
        k: v
        for k, v in sc.items()
        if not k.startswith("_")
    }


def _eval_args(sc):
    return {
        k: v
        for k, v in sc.items()
        if not k.startswith("_")
        and k != "W_dir"
        and k != "H"
    }


def _solve(sc, **kw):
    return optimize_joint_allocation(
        **_args(sc),
        **kw,
    )


_SC = _scenario()


# ---------------------------------------------------------------------------
# 1. Determinism
# ---------------------------------------------------------------------------

class TestDeterminism(unittest.TestCase):
    def test_repeated_execution_bit_identical(self):
        r1 = _solve(
            _SC,
            alpha=1.0,
            beta=1.0,
        )
        r2 = _solve(
            _SC,
            alpha=1.0,
            beta=1.0,
        )

        for key in (
            "W",
            "P_comm",
            "P_sens",
            "P_tx",
            "sinr",
            "rates_bps",
            "information_gain",
            "tracking_error",
            "J_total",
        ):
            self.assertTrue(
                np.array_equal(
                    r1[key],
                    r2[key],
                ),
                key,
            )

        self.assertEqual(
            r1["objective"],
            r2["objective"],
        )
        self.assertEqual(
            r1["iterations"],
            r2["iterations"],
        )
        self.assertEqual(
            r1["solver"]["kkt_residual"],
            r2["solver"]["kkt_residual"],
        )

    def test_repeated_execution_bit_identical_all_scope_and_rzf(self):
        sc = _scenario(
            precoder="rzf"
        )

        for scope in (
            "dual_role",
            "all",
        ):
            r1 = _solve(
                sc,
                alpha=1.0,
                beta=2.0,
                comm_power_control=scope,
            )
            r2 = _solve(
                sc,
                alpha=1.0,
                beta=2.0,
                comm_power_control=scope,
            )

            self.assertTrue(
                np.array_equal(
                    r1["P_comm"],
                    r2["P_comm"],
                )
            )
            self.assertTrue(
                np.array_equal(
                    r1["P_sens"],
                    r2["P_sens"],
                )
            )
            self.assertEqual(
                r1["objective"],
                r2["objective"],
            )

    def test_same_seed_same_scenario(self):
        a = _scenario(seed=7)
        b = _scenario(seed=7)

        self.assertTrue(
            np.array_equal(
                a["W_dir"],
                b["W_dir"],
            )
        )
        self.assertTrue(
            np.array_equal(
                a["G_sens"],
                b["G_sens"],
            )
        )


# ---------------------------------------------------------------------------
# 2. Output shapes
# ---------------------------------------------------------------------------

class TestOutputShapes(unittest.TestCase):
    def test_shapes(self):
        r = _solve(_SC)

        M = _SC["_M"]
        K = _SC["_K"]
        Q = _SC["_Q"]
        N = _SC["_N"]

        self.assertEqual(
            r["W"].shape,
            (M, K, N),
        )

        for key in (
            "P_comm",
            "P_sens",
            "P_tx",
            "P_max",
            "P_slack",
            "power_violation",
            "power_feasible_per_ap",
            "rho_comm",
        ):
            self.assertEqual(
                r[key].shape,
                (M,),
                key,
            )

        self.assertEqual(
            r["sinr"].shape,
            (K,),
        )
        self.assertEqual(
            r["rates_bps"].shape,
            (K,),
        )
        self.assertEqual(
            r["information_gain"].shape,
            (Q,),
        )
        self.assertEqual(
            r["tracking_error"].shape,
            (Q,),
        )
        self.assertEqual(
            r["J_total"].shape,
            (Q, 2, 2),
        )

    def test_scalar_types_and_solver_fields(self):
        r = _solve(_SC)

        self.assertIsInstance(
            r["objective"],
            float,
        )
        self.assertIsInstance(
            r["feasible"],
            bool,
        )
        self.assertIsInstance(
            r["converged"],
            bool,
        )
        self.assertIsInstance(
            r["iterations"],
            int,
        )

        for key in (
            "mode",
            "method",
            "kkt_residual",
            "grid_objectives",
            "starts",
            "decision_aps",
            "held_at_baseline_aps",
            "comm_power_control",
            "global_optimality_claimed",
        ):
            self.assertIn(
                key,
                r["solver"],
                key,
            )

        self.assertFalse(
            r["solver"]["global_optimality_claimed"]
        )


# ---------------------------------------------------------------------------
# 3. Finiteness
# ---------------------------------------------------------------------------

class TestFiniteness(unittest.TestCase):
    def test_everything_finite_across_weights_and_scopes(self):
        for scope in (
            "dual_role",
            "all",
        ):
            for a, b in (
                (1, 0),
                (1, 0.1),
                (1, 1),
                (1, 100),
                (0, 1),
            ):
                r = _solve(
                    _SC,
                    alpha=a,
                    beta=b,
                    comm_power_control=scope,
                )

                self.assertTrue(
                    np.isfinite(
                        r["objective"]
                    ),
                    (scope, a, b),
                )

                for key in (
                    "W",
                    "P_comm",
                    "P_sens",
                    "P_tx",
                    "sinr",
                    "rates_bps",
                    "information_gain",
                    "tracking_error",
                    "J_total",
                ):
                    self.assertTrue(
                        np.all(
                            np.isfinite(
                                r[key]
                            )
                        ),
                        (scope, a, b, key),
                    )


# ---------------------------------------------------------------------------
# 4. Sensing power nonnegative
# ---------------------------------------------------------------------------

class TestNonnegativeSensingPower(unittest.TestCase):
    def test_nonnegative_sensing_and_comm_power(self):
        for scope in (
            "dual_role",
            "all",
        ):
            for a, b in (
                (1, 0),
                (1, 0.5),
                (1, 5),
                (1, 500),
                (0, 1),
            ):
                r = _solve(
                    _SC,
                    alpha=a,
                    beta=b,
                    comm_power_control=scope,
                )

                self.assertTrue(
                    np.all(
                        r["P_sens"] >= 0.0
                    )
                )
                self.assertTrue(
                    np.all(
                        r["P_comm"] >= 0.0
                    )
                )

    def test_sensing_only_on_sensing_useful_aps(self):
        _, useful = sensing_fim_basis(
            _SC["ap_positions"],
            _SC["target_positions"],
            _SC["y_tx"],
            _SC["y_rx"],
            _SC["G_sens"],
        )

        r = _solve(
            _SC,
            alpha=1.0,
            beta=1.0,
        )

        self.assertTrue(
            np.all(
                r["P_sens"][~useful] == 0.0
            )
        )


# ---------------------------------------------------------------------------
# 5. Exact combined-power accounting
# ---------------------------------------------------------------------------

class TestCombinedPowerAccounting(unittest.TestCase):
    def test_ptx_is_exactly_comm_plus_sens(self):
        for a, b in (
            (1, 0.3),
            (1, 3),
            (0, 1),
            (1, 0),
        ):
            r = _solve(
                _SC,
                alpha=a,
                beta=b,
            )

            self.assertTrue(
                np.array_equal(
                    r["P_tx"],
                    r["P_comm"] + r["P_sens"],
                )
            )

            self.assertTrue(
                np.array_equal(
                    r["P_tx"],
                    physical_tx_power_per_ap(
                        r["W"],
                        r["P_sens"],
                    ),
                )
            )

    def test_pcomm_is_measured_from_the_returned_weights(self):
        r = _solve(
            _SC,
            alpha=1.0,
            beta=1.0,
        )

        self.assertTrue(
            np.array_equal(
                r["P_comm"],
                comm_power_per_ap(r["W"]),
            )
        )

        self.assertLess(
            r["solver"][
                "comm_power_setpoint_vs_measured_max_abs_error"
            ],
            1e-12,
        )

    def test_sensing_power_not_folded_into_beam_weights(self):
        useful = sensing_fim_basis(
            _SC["ap_positions"],
            _SC["target_positions"],
            _SC["y_tx"],
            _SC["y_rx"],
            _SC["G_sens"],
        )[1]

        c, s = uniform_split_allocation(
            _SC["W_dir"],
            1.0,
            0.4,
            useful,
        )

        a = evaluate_joint_allocation(
            _SC["H"],
            _SC["W_dir"],
            c,
            s,
            **_eval_args(_SC),
        )

        b = evaluate_joint_allocation(
            _SC["H"],
            _SC["W_dir"],
            c,
            0.0 * s,
            **_eval_args(_SC),
        )

        self.assertTrue(
            np.array_equal(
                a["W"],
                b["W"],
            )
        )
        self.assertTrue(
            np.array_equal(
                a["P_comm"],
                b["P_comm"],
            )
        )
        self.assertTrue(
            np.array_equal(
                a["sinr"],
                b["sinr"],
            )
        )

    def test_dual_role_ap_counted_once_at_budget(self):
        r = _solve(
            _SC,
            alpha=1.0,
            beta=1.0,
        )

        dual = r["solver"]["decision_aps"]

        self.assertGreater(
            len(dual),
            0,
        )

        for m in dual:
            self.assertAlmostEqual(
                r["P_tx"][m],
                r["P_max"][m],
                places=12,
            )
            self.assertGreater(
                r["P_comm"][m],
                0.0,
            )
            self.assertGreater(
                r["P_sens"][m],
                0.0,
            )


# ---------------------------------------------------------------------------
# 6. Per-AP power constraint
# ---------------------------------------------------------------------------

class TestPerAPPowerConstraint(unittest.TestCase):
    def test_constraint_holds_across_weights_scopes(self):
        for scope in (
            "dual_role",
            "all",
        ):
            for a, b in (
                (1, 0),
                (1, 0.01),
                (1, 1),
                (1, 50),
                (0, 1),
            ):
                r = _solve(
                    _SC,
                    alpha=a,
                    beta=b,
                    comm_power_control=scope,
                )

                self.assertTrue(
                    np.all(
                        r["P_tx"]
                        <= r["P_max"] + 1e-9
                    )
                )
                self.assertTrue(
                    r["feasible"]
                )
                self.assertTrue(
                    np.all(
                        r["power_feasible_per_ap"]
                    )
                )
                self.assertLessEqual(
                    r["max_power_violation"],
                    1e-9,
                )

    def test_heterogeneous_budgets(self):
        sc = dict(_SC)

        sc["P_max"] = np.linspace(
            0.4,
            2.5,
            _SC["_M"],
        )

        r = _solve(
            sc,
            alpha=1.0,
            beta=1.0,
        )

        self.assertTrue(
            np.array_equal(
                r["P_max"],
                sc["P_max"],
            )
        )
        self.assertTrue(
            np.all(
                r["P_tx"]
                <= sc["P_max"] + 1e-9
            )
        )
        self.assertTrue(
            r["feasible"]
        )

    def test_evaluate_detects_infeasible_allocation(self):
        M = _SC["_M"]

        c = np.where(
            np.sum(
                np.abs(
                    _SC["W_dir"]
                ) ** 2,
                axis=(1, 2),
            ) > 0.5,
            1.0,
            0.0,
        )

        s = np.zeros(M)
        s[_SC["_active"][0]] = 0.5

        res = evaluate_joint_allocation(
            _SC["H"],
            _SC["W_dir"],
            c,
            s,
            **_eval_args(_SC),
        )

        self.assertFalse(
            res["feasible"]
        )
        self.assertAlmostEqual(
            res["max_power_violation"],
            0.5,
            places=12,
        )
        self.assertFalse(
            res["power_feasible_per_ap"][
                _SC["_active"][0]
            ]
        )


# ---------------------------------------------------------------------------
# 7. Zero-sensing boundary
# ---------------------------------------------------------------------------

class TestZeroSensingBoundary(unittest.TestCase):
    def test_beta_zero_gives_no_sensing_resource(self):
        r = _solve(
            _SC,
            alpha=1.0,
            beta=0.0,
        )

        self.assertTrue(
            np.all(
                r["P_sens"] == 0.0
            )
        )
        self.assertTrue(
            np.all(
                r["J_total"] == 0.0
            )
        )
        self.assertTrue(
            np.all(
                r["information_gain"]
                == 0.0
            )
        )
        self.assertEqual(
            r["sensing_utility_raw"],
            0.0,
        )
        self.assertTrue(
            np.array_equal(
                r["P_tx"],
                r["P_comm"],
            )
        )
        self.assertTrue(
            np.allclose(
                r["tracking_error"],
                2.0,
                atol=1e-15,
            )
        )

    def test_evaluate_zero_sensing_power(self):
        M = _SC["_M"]

        c = np.where(
            np.sum(
                np.abs(
                    _SC["W_dir"]
                ) ** 2,
                axis=(1, 2),
            ) > 0.5,
            0.7,
            0.0,
        )

        res = evaluate_joint_allocation(
            _SC["H"],
            _SC["W_dir"],
            c,
            np.zeros(M),
            **_eval_args(_SC),
        )

        self.assertEqual(
            res["sensing_utility_raw"],
            0.0,
        )
        self.assertTrue(
            np.all(
                res["information_gain"]
                == 0.0
            )
        )
        self.assertTrue(
            res["feasible"]
        )


# ---------------------------------------------------------------------------
# 8. Communication-only boundary
# ---------------------------------------------------------------------------

class TestCommunicationOnlyBoundary(unittest.TestCase):
    def test_utility_matches_independent_physical_path(self):
        r = _solve(
            _SC,
            alpha=1.0,
            beta=0.0,
        )

        sinr = physical_sinr(
            _SC["H"],
            r["W"],
            1e-9,
        )

        rate = physical_rate(
            sinr
        )

        self.assertTrue(
            np.array_equal(
                r["sinr"],
                sinr,
            )
        )
        self.assertTrue(
            np.array_equal(
                r["rates_bps"],
                rate,
            )
        )
        self.assertEqual(
            r["comm_utility_raw"],
            float(np.sum(rate)),
        )
        self.assertAlmostEqual(
            r["objective"],
            r["comm_utility_norm"],
            places=12,
        )

    def test_communication_only_not_worse_than_full_power_baseline(self):
        r = _solve(
            _SC,
            alpha=1.0,
            beta=0.0,
        )

        useful = sensing_fim_basis(
            _SC["ap_positions"],
            _SC["target_positions"],
            _SC["y_tx"],
            _SC["y_rx"],
            _SC["G_sens"],
        )[1]

        c, s = uniform_split_allocation(
            _SC["W_dir"],
            1.0,
            1.0,
            useful,
        )

        base = evaluate_joint_allocation(
            _SC["H"],
            _SC["W_dir"],
            c,
            s,
            alpha=1.0,
            beta=0.0,
            **_eval_args(_SC),
        )

        self.assertGreaterEqual(
            r["comm_utility_raw"],
            base["comm_utility_raw"] - 1e-6,
        )

    def test_full_power_baseline_equals_phase4a_path(self):
        W4a, _ = normalize_per_ap_power(
            mrt_weights(
                _SC["H"],
                _SC["_x"],
            ),
            1.0,
        )

        rate4a = physical_rate(
            physical_sinr(
                _SC["H"],
                W4a,
                1e-9,
            )
        )

        c = np.where(
            np.sum(
                np.abs(
                    _SC["W_dir"]
                ) ** 2,
                axis=(1, 2),
            ) > 0.5,
            1.0,
            0.0,
        )

        res = evaluate_joint_allocation(
            _SC["H"],
            _SC["W_dir"],
            c,
            np.zeros(_SC["_M"]),
            alpha=1.0,
            beta=0.0,
            **_eval_args(_SC),
        )

        self.assertTrue(
            np.allclose(
                res["rates_bps"],
                rate4a,
                rtol=1e-12,
                atol=0.0,
            )
        )
        self.assertTrue(
            np.allclose(
                res["W"],
                W4a,
                rtol=1e-12,
                atol=1e-15,
            )
        )


# ---------------------------------------------------------------------------
# 9. Sensing-resource boundary
# ---------------------------------------------------------------------------

class TestSensingResourceBoundary(unittest.TestCase):
    def setUp(self):
        self.A, self.useful = sensing_fim_basis(
            _SC["ap_positions"],
            _SC["target_positions"],
            _SC["y_tx"],
            _SC["y_rx"],
            _SC["G_sens"],
        )

        self.r = _solve(
            _SC,
            alpha=0.0,
            beta=1.0,
        )

    def test_no_communication_power_and_full_sensing_budget(self):
        self.assertTrue(
            np.all(
                self.r["P_comm"] == 0.0
            )
        )
        self.assertTrue(
            np.all(
                self.r["W"] == 0.0
            )
        )
        self.assertTrue(
            np.all(
                self.r["rates_bps"] == 0.0
            )
        )
        self.assertTrue(
            np.array_equal(
                self.r["P_sens"][self.useful],
                self.r["P_max"][self.useful],
            )
        )
        self.assertTrue(
            np.all(
                self.r["P_sens"][~self.useful]
                == 0.0
            )
        )

    def test_matches_phase5a_bridge_at_full_sensing_power(self):
        p = np.where(
            self.useful,
            1.0,
            0.0,
        )

        bridge = evaluate_physical_sensing_bridge(
            _SC["ap_positions"],
            _SC["target_positions"],
            _SC["y_tx"],
            _SC["y_rx"],
            np.zeros_like(
                _SC["W_dir"]
            ),
            p,
            _SC["G_sens"],
            1.0,
        )

        self.assertTrue(
            np.array_equal(
                self.r["information_gain"],
                bridge["information_gain"],
            )
        )

        self.assertTrue(
            np.array_equal(
                self.r["tracking_error"],
                bridge["tracking_error"],
            )
        )

    def test_information_gain_is_maximal_over_the_feasible_set(self):
        joint = _solve(
            _SC,
            alpha=1.0,
            beta=1.0,
        )

        self.assertGreaterEqual(
            self.r["sensing_utility_raw"],
            joint["sensing_utility_raw"] - 1e-12,
        )

        rng = np.random.default_rng(0)

        for _ in range(25):
            frac = rng.uniform(
                0.0,
                1.0,
                _SC["_M"],
            )

            s = np.where(
                self.useful,
                frac,
                0.0,
            )

            res = evaluate_joint_allocation(
                _SC["H"],
                _SC["W_dir"],
                np.zeros(_SC["_M"]),
                s,
                alpha=0.0,
                beta=1.0,
                **_eval_args(_SC),
            )

            self.assertGreaterEqual(
                self.r["sensing_utility_raw"],
                res["sensing_utility_raw"] - 1e-12,
            )


# ---------------------------------------------------------------------------
# 10. Allocation changes objective
# ---------------------------------------------------------------------------

class TestAllocationChangesObjective(unittest.TestCase):
    def setUp(self):
        _, self.useful = sensing_fim_basis(
            _SC["ap_positions"],
            _SC["target_positions"],
            _SC["y_tx"],
            _SC["y_rx"],
            _SC["G_sens"],
        )

    def _uniform(
        self,
        rho,
        alpha=1.0,
        beta=1.0,
    ):
        c, s = uniform_split_allocation(
            _SC["W_dir"],
            1.0,
            rho,
            self.useful,
        )

        return evaluate_joint_allocation(
            _SC["H"],
            _SC["W_dir"],
            c,
            s,
            alpha=alpha,
            beta=beta,
            **_eval_args(_SC),
        )

    def test_different_splits_give_different_objectives(self):
        vals = [
            self._uniform(r)["objective"]
            for r in (
                1.0,
                0.75,
                0.5,
                0.25,
                0.0,
            )
        ]

        self.assertEqual(
            len(set(vals)),
            len(vals),
        )

    def test_more_sensing_share_trades_rate_for_information(self):
        hi_comm = self._uniform(1.0)
        lo_comm = self._uniform(0.1)

        self.assertGreater(
            hi_comm["comm_utility_raw"],
            lo_comm["comm_utility_raw"],
        )
        self.assertLess(
            hi_comm["sensing_utility_raw"],
            lo_comm["sensing_utility_raw"],
        )

    def test_joint_dominates_every_uniform_grid_split(self):
        r = _solve(
            _SC,
            alpha=1.0,
            beta=1.0,
        )

        for g in r["solver"]["grid_objectives"]:
            self.assertGreaterEqual(
                r["objective"],
                g["objective"] - 1e-9,
                g,
            )

        for rho in (
            1.0,
            0.5,
            0.25,
            0.0,
        ):
            self.assertGreaterEqual(
                r["objective"],
                self._uniform(rho)["objective"] - 1e-9,
            )

    def test_joint_strictly_improves_on_baseline_split(self):
        r = _solve(
            _SC,
            alpha=1.0,
            beta=1.0,
        )

        self.assertGreater(
            r["objective"],
            self._uniform(1.0)["objective"] + 0.1,
        )

    def test_weights_move_the_operating_point_monotonically_here(self):
        prev_ig = -1.0
        prev_rate = np.inf

        for b in (
            0.0,
            0.1,
            1.0,
            10.0,
            100.0,
        ):
            r = _solve(
                _SC,
                alpha=1.0,
                beta=b,
            )

            self.assertGreaterEqual(
                r["sensing_utility_raw"],
                prev_ig - 1e-9,
            )

            self.assertLessEqual(
                r["comm_utility_raw"],
                prev_rate + 1e-3,
            )

            prev_ig = r["sensing_utility_raw"]
            prev_rate = r["comm_utility_raw"]

    def test_solver_matches_dense_grid_on_the_two_decision_aps(self):
        r = _solve(
            _SC,
            alpha=1.0,
            beta=1.0,
        )

        d = r["solver"]["decision_aps"]

        self.assertEqual(
            len(d),
            2,
        )

        base_c = np.where(
            np.sum(
                np.abs(
                    _SC["W_dir"]
                ) ** 2,
                axis=(1, 2),
            ) > 0.5,
            1.0,
            0.0,
        )

        best = -np.inf

        grid = np.concatenate(
            [
                [0.0],
                np.logspace(
                    -4,
                    0,
                    41,
                ),
            ]
        )

        for r0 in grid:
            for r1 in grid:
                c = base_c.copy()
                c[d[0]] = r0
                c[d[1]] = r1

                s = np.where(
                    self.useful,
                    1.0 - c,
                    0.0,
                )

                res = evaluate_joint_allocation(
                    _SC["H"],
                    _SC["W_dir"],
                    c,
                    s,
                    alpha=1.0,
                    beta=1.0,
                    **_eval_args(_SC),
                )

                best = max(
                    best,
                    res["objective"],
                )

        self.assertGreaterEqual(
            r["objective"],
            best - 1e-6,
        )


# ---------------------------------------------------------------------------
# 11. Scope semantics
# ---------------------------------------------------------------------------

class TestPowerControlScope(unittest.TestCase):
    def test_dual_role_holds_comm_only_aps_at_baseline(self):
        r = _solve(
            _SC,
            alpha=1.0,
            beta=1.0,
        )

        held = r["solver"]["held_at_baseline_aps"]

        self.assertGreater(
            len(held),
            0,
        )

        self.assertTrue(
            np.allclose(
                r["P_comm"][held],
                r["P_max"][held],
                rtol=0.0,
                atol=1e-12,
            )
        )

        self.assertTrue(
            np.all(
                r["P_sens"][held] == 0.0
            )
        )

    def test_all_scope_optimizes_every_comm_active_ap(self):
        r = _solve(
            _SC,
            alpha=1.0,
            beta=1.0,
            comm_power_control="all",
        )

        self.assertEqual(
            sorted(
                r["solver"]["decision_aps"]
            ),
            sorted(
                int(i)
                for i in _SC["_active"]
            ),
        )

        self.assertEqual(
            r["solver"]["held_at_baseline_aps"],
            [],
        )

    def test_all_scope_starves_a_user_dual_role_does_not(self):
        dual = _solve(
            _SC,
            alpha=1.0,
            beta=1.0,
            comm_power_control="dual_role",
        )

        full = _solve(
            _SC,
            alpha=1.0,
            beta=1.0,
            comm_power_control="all",
        )

        self.assertGreater(
            dual["min_user_rate_bps"],
            1e6,
        )

        self.assertLess(
            full["min_user_rate_bps"],
            1e6,
        )

        self.assertGreater(
            full["comm_utility_raw"],
            dual["comm_utility_raw"],
        )

    def test_no_dual_role_aps_is_handled(self):
        sc = dict(_SC)

        y_tx = np.zeros_like(
            _SC["y_tx"]
        )

        idle = [
            m
            for m in range(_SC["_M"])
            if m not in set(
                _SC["_active"].tolist()
            )
        ]

        y_tx[idle[:2], :] = 1.0
        sc["y_tx"] = y_tx

        r = _solve(
            sc,
            alpha=1.0,
            beta=1.0,
        )

        self.assertEqual(
            r["solver"]["n_variables"],
            0,
        )
        self.assertEqual(
            r["solver"]["decision_aps"],
            [],
        )
        self.assertTrue(
            r["converged"]
        )
        self.assertTrue(
            r["feasible"]
        )
        self.assertTrue(
            np.all(
                r["P_sens"][idle[:2]]
                == r["P_max"][idle[:2]]
            )
        )

        base = np.where(
            np.sum(
                np.abs(
                    _SC["W_dir"]
                ) ** 2,
                axis=(1, 2),
            ) > 0.5,
            1.0,
            0.0,
        )

        self.assertTrue(
            np.allclose(
                r["P_comm"],
                base,
                rtol=0.0,
                atol=1e-12,
            )
        )


# ---------------------------------------------------------------------------
# 12. Convergence reporting
# ---------------------------------------------------------------------------

class TestConvergenceReporting(unittest.TestCase):
    def test_joint_solve_converges_and_reports_kkt(self):
        r = _solve(
            _SC,
            alpha=1.0,
            beta=1.0,
        )

        self.assertTrue(
            r["converged"]
        )

        self.assertLessEqual(
            r["solver"]["kkt_residual"],
            r["solver"]["kkt_tol"],
        )

        self.assertGreater(
            r["iterations"],
            0,
        )

        self.assertEqual(
            r["solver"]["returned_from"],
            "polished_start",
        )

        self.assertEqual(
            len(
                r["solver"]["starts"]
            ),
            3,
        )

        for st in r["solver"]["starts"]:
            self.assertGreaterEqual(
                st["end_objective"],
                st["start_objective"] - 1e-12,
            )

            for key in (
                "iterations",
                "function_evaluations",
                "status",
                "message",
            ):
                self.assertIn(
                    key,
                    st,
                )

    def test_boundary_modes_are_labelled(self):
        self.assertEqual(
            _solve(
                _SC,
                alpha=1.0,
                beta=0.0,
            )["solver"]["mode"],
            "communication_only_boundary(beta=0)",
        )

        self.assertEqual(
            _solve(
                _SC,
                alpha=0.0,
                beta=1.0,
            )["solver"]["mode"],
            "sensing_only_boundary(alpha=0)",
        )

        self.assertEqual(
            _solve(
                _SC,
                alpha=1.0,
                beta=1.0,
            )["solver"]["mode"],
            "joint",
        )

    def test_internal_evaluator_matches_authoritative_path(self):
        r = _solve(
            _SC,
            alpha=1.0,
            beta=1.0,
        )

        self.assertLess(
            r["solver"][
                "objective_internal_vs_authoritative_abs_error"
            ],
            1e-9,
        )


# ---------------------------------------------------------------------------
# 13. Mathematical structure
# ---------------------------------------------------------------------------

class TestSensingStructure(unittest.TestCase):
    def setUp(self):
        self.A, self.useful = sensing_fim_basis(
            _SC["ap_positions"],
            _SC["target_positions"],
            _SC["y_tx"],
            _SC["y_rx"],
            _SC["G_sens"],
        )

    def test_fim_is_linear_in_sensing_power(self):
        rng = np.random.default_rng(11)

        for _ in range(5):
            p = rng.uniform(
                0.0,
                1.0,
                _SC["_M"],
            )

            J = aggregate_fisher_information_bridge(
                _SC["ap_positions"],
                _SC["target_positions"],
                _SC["y_tx"],
                _SC["y_rx"],
                p,
                _SC["G_sens"],
            )

            J_lin = np.einsum(
                "m,mqij->qij",
                p,
                self.A,
            )

            self.assertLess(
                np.max(
                    np.abs(
                        J - J_lin
                    )
                ),
                1e-12,
            )

    def test_basis_is_psd_and_only_on_tx_aps(self):
        self.assertTrue(
            np.all(
                np.linalg.eigvalsh(
                    self.A
                ) >= -1e-15
            )
        )

        no_tx = ~np.any(
            _SC["y_tx"] > 0,
            axis=1,
        )

        self.assertTrue(
            np.all(
                self.A[no_tx] == 0.0
            )
        )

    def test_information_gain_is_concave_and_monotone_in_sensing_power(self):
        rng = np.random.default_rng(5)
        M = _SC["_M"]
        zero_c = np.zeros(M)

        def ig(s):
            return evaluate_joint_allocation(
                _SC["H"],
                _SC["W_dir"],
                zero_c,
                s,
                alpha=0.0,
                beta=1.0,
                **_eval_args(_SC),
            )["sensing_utility_raw"]

        for _ in range(15):
            s1 = np.where(
                self.useful,
                rng.uniform(
                    0.0,
                    1.0,
                    M,
                ),
                0.0,
            )

            s2 = np.where(
                self.useful,
                rng.uniform(
                    0.0,
                    1.0,
                    M,
                ),
                0.0,
            )

            t = rng.uniform(
                0.05,
                0.95,
            )

            self.assertGreaterEqual(
                ig(
                    t * s1
                    + (1 - t) * s2
                ),
                (
                    t * ig(s1)
                    + (1 - t) * ig(s2)
                    - 1e-12
                ),
            )

            self.assertGreaterEqual(
                ig(
                    np.maximum(
                        s1,
                        s2,
                    )
                ),
                ig(s1) - 1e-12,
            )

    def test_linearity_guard_raises_when_variance_floor_active(self):
        sc = dict(_SC)

        with self.assertRaises(
            ValueError
        ):
            optimize_joint_allocation(
                **_args(sc),
                noise_power_sens=1e-25,
            )


class TestAnalyticGradient(unittest.TestCase):
    def _fd_check(
        self,
        scope,
        alpha,
        beta,
    ):
        p = _JointProblem(
            _SC["H"],
            _SC["W_dir"],
            _SC["ap_positions"],
            _SC["target_positions"],
            _SC["y_tx"],
            _SC["y_rx"],
            _SC["G_sens"],
            1.0,
            alpha,
            beta,
            None,
            1e-9,
            1e-9,
            1.0,
            20e6,
            0.1,
            1e7,
            1.0,
            comm_power_control=scope,
        )

        rng = np.random.default_rng(3)

        for _ in range(3):
            z = rng.uniform(
                np.log(1e-5),
                -0.05,
                len(p.var_idx),
            )

            _, g = p.value_and_grad(
                z
            )

            h = 1e-4

            g_fd = np.array([
                (
                    p.value_and_grad(
                        z + h * e
                    )[0]
                    -
                    p.value_and_grad(
                        z - h * e
                    )[0]
                )
                / (
                    2 * h
                )
                for e in np.eye(
                    len(z)
                )
            ])

            self.assertLess(
                np.max(
                    np.abs(
                        g - g_fd
                    )
                ),
                1e-6,
            )

    def test_gradient_matches_central_differences_dual_role(self):
        self._fd_check(
            "dual_role",
            1.0,
            1.0,
        )

    def test_gradient_matches_central_differences_all_scope(self):
        self._fd_check(
            "all",
            1.0,
            3.0,
        )

    def test_gradient_matches_central_differences_comm_only(self):
        self._fd_check(
            "all",
            1.0,
            0.0,
        )


# ---------------------------------------------------------------------------
# 14. Direction sources and input validation
# ---------------------------------------------------------------------------

class TestDirectionsAndValidation(unittest.TestCase):
    def test_rzf_directions_supported(self):
        sc = _scenario(
            precoder="rzf"
        )

        r = _solve(
            sc,
            alpha=1.0,
            beta=1.0,
        )

        self.assertTrue(
            r["feasible"]
        )
        self.assertTrue(
            np.isfinite(
                r["objective"]
            )
        )

    def test_directions_have_unit_power_per_active_ap(self):
        for prec in (
            "mrt",
            "rzf",
        ):
            Wd = comm_beam_directions(
                _SC["H"],
                _SC["_x"],
                precoder=prec,
            )

            p = np.sum(
                np.abs(Wd) ** 2,
                axis=(1, 2),
            )

            self.assertTrue(
                np.all(
                    (np.abs(p - 1.0) < 1e-12)
                    | (p == 0.0)
                )
            )

    def test_invalid_inputs_raise(self):
        base = _args(_SC)

        with self.assertRaises(ValueError):
            optimize_joint_allocation(
                **{
                    **base,
                    "P_max": -1.0,
                }
            )

        with self.assertRaises(ValueError):
            optimize_joint_allocation(
                **base,
                alpha=0.0,
                beta=0.0,
            )

        with self.assertRaises(ValueError):
            optimize_joint_allocation(
                **base,
                alpha=-1.0,
            )

        with self.assertRaises(ValueError):
            optimize_joint_allocation(
                **base,
                comm_power_control="everything",
            )

        with self.assertRaises(ValueError):
            optimize_joint_allocation(
                **{
                    **base,
                    "W_dir": 2.0 * _SC["W_dir"],
                }
            )

        with self.assertRaises(ValueError):
            comm_beam_directions(
                _SC["H"],
                _SC["_x"],
                precoder="zf",
            )

        M = _SC["_M"]

        with self.assertRaises(ValueError):
            evaluate_joint_allocation(
                _SC["H"],
                _SC["W_dir"],
                -np.ones(M),
                np.zeros(M),
                **_eval_args(_SC),
            )

        with self.assertRaises(ValueError):
            evaluate_joint_allocation(
                _SC["H"],
                _SC["W_dir"],
                np.zeros(M + 1),
                np.zeros(M + 1),
                **_eval_args(_SC),
            )

    def test_qos_diagnostic_only(self):
        r_plain = _solve(
            _SC,
            alpha=1.0,
            beta=1.0,
        )

        r_qos = _solve(
            _SC,
            alpha=1.0,
            beta=1.0,
            min_rate_bps=1e6,
        )

        self.assertNotIn(
            "qos_violation_total_bps",
            r_plain,
        )

        self.assertIn(
            "qos_violation_total_bps",
            r_qos,
        )

        self.assertTrue(
            np.array_equal(
                r_plain["P_comm"],
                r_qos["P_comm"],
            )
        )


# ---------------------------------------------------------------------------
# 15. Regression: Phase 4A / 4B / 5A unchanged
# ---------------------------------------------------------------------------

class TestPhase4ARegression(unittest.TestCase):
    def test_mrt_rzf_bitwise_unchanged_by_phase5b(self):
        H = _SC["H"]
        x = _SC["_x"]

        mrt0 = mrt_weights(
            H,
            x,
        )

        rzf0 = rzf_weights(
            H,
            x,
            1e-2,
        )

        n0 = normalize_per_ap_power(
            mrt0,
            0.8,
        )

        _solve(
            _SC,
            alpha=1.0,
            beta=1.0,
        )

        _solve(
            _SC,
            alpha=1.0,
            beta=1.0,
            comm_power_control="all",
        )

        self.assertTrue(
            np.array_equal(
                mrt0,
                mrt_weights(
                    H,
                    x,
                ),
            )
        )

        self.assertTrue(
            np.array_equal(
                rzf0,
                rzf_weights(
                    H,
                    x,
                    1e-2,
                ),
            )
        )

        n1 = normalize_per_ap_power(
            mrt_weights(H, x),
            0.8,
        )

        self.assertTrue(
            np.array_equal(
                n0[0],
                n1[0],
            )
        )
        self.assertTrue(
            np.array_equal(
                n0[1],
                n1[1],
            )
        )

    def test_mrt_structural_invariants_still_hold(self):
        H = _SC["H"]
        x = _SC["_x"]

        W = mrt_weights(
            H,
            x,
        )

        norms = np.linalg.norm(
            W,
            axis=-1,
        )

        self.assertTrue(
            np.allclose(
                norms[x == 1],
                1.0,
                atol=1e-12,
            )
        )

        self.assertTrue(
            np.all(
                W[x == 0] == 0.0
            )
        )

    def test_physical_sinr_rate_inputs_not_mutated(self):
        H0 = _SC["H"].copy()
        W0 = _SC["W_dir"].copy()

        _solve(
            _SC,
            alpha=1.0,
            beta=1.0,
        )

        self.assertTrue(
            np.array_equal(
                H0,
                _SC["H"],
            )
        )

        self.assertTrue(
            np.array_equal(
                W0,
                _SC["W_dir"],
            )
        )


class TestPhase4BRegression(unittest.TestCase):
    def _small(self):
        ap = generate_ap_positions(
            6,
            150.0,
            seed=42,
        )

        us = generate_positions(
            3,
            150.0,
            seed=43,
        )

        H = generate_physical_channel(
            ap,
            us,
            num_antennas=4,
            rician_k_factor=10.0,
            small_scale_fading=True,
            seed=42,
        )

        return ap, H

    def test_constrained_beamforming_bitwise_unchanged_and_feasible(self):
        ap, H = self._small()

        W0, ok0, sinr0, info0 = (
            constrained_sinr_beamforming(
                H,
                2.0,
                1.0,
            )
        )

        self.assertTrue(
            ok0
        )

        _solve(
            _SC,
            alpha=1.0,
            beta=1.0,
        )

        W1, ok1, sinr1, info1 = (
            constrained_sinr_beamforming(
                H,
                2.0,
                1.0,
            )
        )

        self.assertTrue(
            np.array_equal(
                W0,
                W1,
            )
        )

        self.assertTrue(
            np.array_equal(
                sinr0,
                sinr1,
            )
        )

        self.assertEqual(
            info0["outer_iterations"],
            info1["outer_iterations"],
        )

        self.assertTrue(
            np.all(
                sinr1
                >= 2.0 * (1 - 1e-3)
            )
        )

    def test_phase4b_design_is_a_valid_direction_source(self):
        ap, H = self._small()

        W4b, ok, sinr4b, _ = (
            constrained_sinr_beamforming(
                H,
                2.0,
                1.0,
            )
        )

        self.assertTrue(
            ok
        )

        P_m = comm_power_per_ap(
            W4b
        )

        W_dir = np.zeros_like(
            W4b
        )

        used = P_m > 1e-12

        W_dir[used] = (
            W4b[used]
            / np.sqrt(
                P_m[used]
            )[:, None, None]
        )

        us = generate_positions(
            3,
            150.0,
            seed=43,
        )

        tg = generate_positions(
            1,
            150.0,
            seed=44,
        )

        y_tx = np.zeros(
            (6, 1)
        )
        y_rx = np.zeros(
            (6, 1)
        )

        y_tx[:2, 0] = 1.0
        y_rx[2:4, 0] = 1.0

        G = generate_sensing_gains(
            ap,
            tg,
        )

        res = evaluate_joint_allocation(
            H,
            W_dir,
            P_m,
            np.zeros(6),
            ap,
            tg,
            y_tx,
            y_rx,
            G,
            1.0,
            alpha=1.0,
            beta=0.0,
        )

        self.assertTrue(
            np.allclose(
                res["sinr"],
                sinr4b,
                rtol=1e-9,
            )
        )

        self.assertTrue(
            res["feasible"]
        )

        r = optimize_joint_allocation(
            H,
            W_dir,
            ap,
            tg,
            y_tx,
            y_rx,
            G,
            1.0,
            alpha=1.0,
            beta=1.0,
        )

        self.assertTrue(
            r["feasible"]
        )

        self.assertTrue(
            np.isfinite(
                r["objective"]
            )
        )


class TestPhase5ARegression(unittest.TestCase):
    def test_bridge_bitwise_unchanged_by_phase5b(self):
        c = np.where(
            np.sum(
                np.abs(
                    _SC["W_dir"]
                ) ** 2,
                axis=(1, 2),
            ) > 0.5,
            0.5,
            0.0,
        )

        W = (
            np.sqrt(c)[:, None, None]
            * _SC["W_dir"]
        )

        s = np.zeros(
            _SC["_M"]
        )

        s[
            _SC["_active"][:2]
        ] = 0.25

        kw = dict(
            ap_positions=_SC["ap_positions"],
            target_positions=_SC["target_positions"],
            y_tx=_SC["y_tx"],
            y_rx=_SC["y_rx"],
            W=W,
            sensing_power_per_ap=s,
            G_sens=_SC["G_sens"],
            tx_power_per_ap_max=1.0,
        )

        b0 = evaluate_physical_sensing_bridge(
            **kw
        )

        _solve(
            _SC,
            alpha=1.0,
            beta=1.0,
        )

        b1 = evaluate_physical_sensing_bridge(
            **kw
        )

        for key in (
            "J_total",
            "information_gain",
            "tracking_error",
            "P_comm",
            "P_sens",
            "P_tx",
        ):
            self.assertTrue(
                np.array_equal(
                    b0[key],
                    b1[key],
                ),
                key,
            )

    def test_phase5b_sensing_outputs_equal_direct_bridge_call(self):
        r = _solve(
            _SC,
            alpha=1.0,
            beta=1.0,
        )

        b = evaluate_physical_sensing_bridge(
            _SC["ap_positions"],
            _SC["target_positions"],
            _SC["y_tx"],
            _SC["y_rx"],
            r["W"],
            r["P_sens"],
            _SC["G_sens"],
            1.0,
        )

        self.assertTrue(
            np.array_equal(
                r["information_gain"],
                b["information_gain"],
            )
        )

        self.assertTrue(
            np.array_equal(
                r["tracking_error"],
                b["tracking_error"],
            )
        )

        self.assertTrue(
            np.array_equal(
                r["J_total"],
                b["J_total"],
            )
        )

        self.assertTrue(
            np.array_equal(
                r["P_tx"],
                b["P_tx"],
            )
        )

    def test_reproduces_a_phase5a_operating_point(self):
        c = np.where(
            np.sum(
                np.abs(
                    _SC["W_dir"]
                ) ** 2,
                axis=(1, 2),
            ) > 0.5,
            0.5,
            0.0,
        )

        s = np.zeros(
            _SC["_M"]
        )

        s[
            _SC["_active"][:2]
        ] = 0.25

        res = evaluate_joint_allocation(
            _SC["H"],
            _SC["W_dir"],
            c,
            s,
            alpha=1.0,
            beta=1.0,
            **_eval_args(_SC),
        )

        self.assertTrue(
            res["feasible"]
        )

        self.assertAlmostEqual(
            float(
                np.mean(
                    res["information_gain"]
                )
            ),
            0.505082,
            places=5,
        )

    def test_scalar_fim_untouched(self):
        args = (
            _SC["ap_positions"],
            _SC["target_positions"],
            _SC["y_tx"],
            _SC["y_rx"],
        )

        J0 = aggregate_fisher_information(
            *args,
            measurement_noise=1.0,
        )

        _solve(
            _SC,
            alpha=1.0,
            beta=1.0,
        )

        J1 = aggregate_fisher_information(
            *args,
            measurement_noise=1.0,
        )

        self.assertTrue(
            np.array_equal(
                J0,
                J1,
            )
        )


if __name__ == "__main__":
    unittest.main()
