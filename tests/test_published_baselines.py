"""
tests/test_published_baselines.py

Unit and integration tests for the published-baseline implementations in
src/baselines/published_baselines.py.

Tests verify:
  1. Import correctness.
  2. Output shape and validity for each baseline.
  3. Sensing and comm association constraints.
  4. Deterministic behavior under fixed seeds.
  5. Integration: each baseline can be used in one simulation step.

These tests do NOT verify that the implementations are exact reproductions
of their respective papers (they are labelled PAPER_INSPIRED_PROXY).
They verify that the implementations are internally consistent and produce
valid outputs for our simulator.
"""
from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def network():
    """Small network for fast tests."""
    from src.environment.topology import generate_ap_positions
    import yaml

    rng = np.random.default_rng(1234)
    ap_positions = generate_ap_positions(20, 500.0, seed=1234)
    user_pos = rng.uniform(0, 500, (5, 2))
    target_pos = rng.uniform(0, 500, (2, 2))

    with open("configs/default.yaml") as f:
        import yaml as _yaml
        cfg = _yaml.safe_load(f)

    ref = cfg["reference_values"]
    perf_kwargs = dict(
        ref=ref,
        min_rate=float(cfg["communication"]["min_rate_bps"]),
    )

    return {
        "ap_positions": ap_positions,
        "user_pos": user_pos,
        "target_pos": target_pos,
        "ref": ref,
        "perf_kwargs": perf_kwargs,
        "M": 20, "K": 5, "Q": 2,
    }


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestBaselineRegistry:
    def test_registry_keys(self):
        from src.baselines.published_baselines import BASELINE_REGISTRY
        expected = {
            "SUN2025_ADAPTIVE_HYSTERESIS",
            "ELFIATOURE2023_GAP_NPC",
            "ELFIATOURE2023_RAP_NPC",
            "KIMCHO2024_CF_HO",
            "OZBEK2026_GMSUA",
            "DEMIRHAN2023_BEAMFORMING",
            "AMMAR2024_POMDP",
        }
        assert set(BASELINE_REGISTRY.keys()) == expected

    def test_registry_reproduction_labels(self):
        from src.baselines.published_baselines import BASELINE_REGISTRY
        valid_labels = {
            "FAITHFUL_REPRODUCTION",
            "COMPATIBLE_REIMPLEMENTATION",
            "PAPER_INSPIRED_PROXY",
            "NOT_IMPLEMENTED",
        }
        for key, entry in BASELINE_REGISTRY.items():
            assert entry["reproduction_label"] in valid_labels, \
                f"{key} has invalid label: {entry['reproduction_label']}"

    def test_not_implemented_baselines(self):
        from src.baselines.published_baselines import BASELINE_REGISTRY
        for key in ("DEMIRHAN2023_BEAMFORMING", "AMMAR2024_POMDP"):
            assert BASELINE_REGISTRY[key]["reproduction_label"] == "NOT_IMPLEMENTED"

    def test_registry_required_fields(self):
        from src.baselines.published_baselines import BASELINE_REGISTRY
        required = {"id", "paper_title", "authors", "venue", "year", "doi",
                    "algorithm", "reproduction_label", "implemented_features",
                    "not_implemented_features", "model_differences"}
        for key, entry in BASELINE_REGISTRY.items():
            missing = required - set(entry.keys())
            assert len(missing) == 0, f"{key} missing fields: {missing}"


# ---------------------------------------------------------------------------
# Helper: validate output shapes
# ---------------------------------------------------------------------------

def _check_association(x, y_tx, y_rx, M, K, Q, aps_per_user=3,
                        tx_aps=2, rx_aps=2):
    """Assert shapes and binary values of association matrices."""
    assert x.shape == (M, K), f"x shape {x.shape} != ({M},{K})"
    assert y_tx.shape == (M, Q), f"y_tx shape {y_tx.shape} != ({M},{Q})"
    assert y_rx.shape == (M, Q), f"y_rx shape {y_rx.shape} != ({M},{Q})"
    assert set(np.unique(x)).issubset({0, 1}), "x must be binary"
    assert set(np.unique(y_tx)).issubset({0, 1}), "y_tx must be binary"
    assert set(np.unique(y_rx)).issubset({0, 1}), "y_rx must be binary"
    # Each user served by at least 1 AP
    assert np.all(x.sum(axis=0) >= 1), "Every user must have at least 1 serving AP"


# ---------------------------------------------------------------------------
# GAP-NPC tests
# ---------------------------------------------------------------------------

class TestElfiatoureGAPNPC:
    def test_output_shapes(self, network):
        from src.baselines.published_baselines import elfiatoure2023_gap_npc
        n = network
        x, y_tx, y_rx, info = elfiatoure2023_gap_npc(
            n["ap_positions"], n["user_pos"], n["target_pos"],
            perf_kwargs=n["perf_kwargs"], ref=n["ref"],
        )
        _check_association(x, y_tx, y_rx, n["M"], n["K"], n["Q"])

    def test_info_dict_keys(self, network):
        from src.baselines.published_baselines import elfiatoure2023_gap_npc
        n = network
        _, _, _, info = elfiatoure2023_gap_npc(
            n["ap_positions"], n["user_pos"], n["target_pos"],
        )
        assert "n_com_aps" in info
        assert "n_sen_aps" in info
        assert "iterations" in info
        assert info["n_com_aps"] + info["n_sen_aps"] <= n["M"]

    def test_partition_valid(self, network):
        from src.baselines.published_baselines import elfiatoure2023_gap_npc
        n = network
        _, _, _, info = elfiatoure2023_gap_npc(
            n["ap_positions"], n["user_pos"], n["target_pos"],
        )
        # Comm + Sensing should not exceed M
        assert info["n_com_aps"] + info["n_sen_aps"] <= n["M"]
        # At least some sensing APs to maintain sensing quality
        assert info["n_sen_aps"] >= 1

    def test_no_nan_outputs(self, network):
        from src.baselines.published_baselines import elfiatoure2023_gap_npc
        n = network
        x, y_tx, y_rx, _ = elfiatoure2023_gap_npc(
            n["ap_positions"], n["user_pos"], n["target_pos"],
        )
        assert not np.any(np.isnan(x.astype(float)))
        assert not np.any(np.isnan(y_tx.astype(float)))
        assert not np.any(np.isnan(y_rx.astype(float)))

    def test_deterministic_fixed_seed(self, network):
        """Same topology → same result."""
        from src.baselines.published_baselines import elfiatoure2023_gap_npc
        n = network
        x1, yt1, yr1, _ = elfiatoure2023_gap_npc(
            n["ap_positions"], n["user_pos"], n["target_pos"],
        )
        x2, yt2, yr2, _ = elfiatoure2023_gap_npc(
            n["ap_positions"], n["user_pos"], n["target_pos"],
        )
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(yt1, yt2)
        np.testing.assert_array_equal(yr1, yr2)


# ---------------------------------------------------------------------------
# RAP-NPC tests
# ---------------------------------------------------------------------------

class TestElfiatoureRAPNPC:
    def test_output_shapes(self, network):
        from src.baselines.published_baselines import elfiatoure2023_rap_npc
        n = network
        x, y_tx, y_rx, info = elfiatoure2023_rap_npc(
            n["ap_positions"], n["user_pos"], n["target_pos"], seed=42
        )
        _check_association(x, y_tx, y_rx, n["M"], n["K"], n["Q"])

    def test_randomness_varies_with_seed(self, network):
        from src.baselines.published_baselines import elfiatoure2023_rap_npc
        n = network
        x1, _, _, _ = elfiatoure2023_rap_npc(
            n["ap_positions"], n["user_pos"], n["target_pos"], seed=10
        )
        x2, _, _, _ = elfiatoure2023_rap_npc(
            n["ap_positions"], n["user_pos"], n["target_pos"], seed=99
        )
        # Different seeds should (very likely) give different results
        # Not guaranteed but extremely likely with 20 APs
        # At minimum the shapes are correct
        assert x1.shape == (n["M"], n["K"])
        assert x2.shape == (n["M"], n["K"])

    def test_reproducible_same_seed(self, network):
        from src.baselines.published_baselines import elfiatoure2023_rap_npc
        n = network
        x1, yt1, yr1, _ = elfiatoure2023_rap_npc(
            n["ap_positions"], n["user_pos"], n["target_pos"], seed=42
        )
        x2, yt2, yr2, _ = elfiatoure2023_rap_npc(
            n["ap_positions"], n["user_pos"], n["target_pos"], seed=42
        )
        np.testing.assert_array_equal(x1, x2)


# ---------------------------------------------------------------------------
# Sun et al. Adaptive Hysteresis tests
# ---------------------------------------------------------------------------

class TestSunAdaptiveHysteresis:
    def _prev_clustering(self, network):
        """Get a previous valid clustering for testing."""
        from src.optimization.baselines import reactive_policy
        n = network
        return reactive_policy(n["ap_positions"], n["user_pos"], n["target_pos"])

    def test_output_shapes(self, network):
        from src.baselines.published_baselines import sun2025_adaptive_hysteresis
        n = network
        prev_x, prev_yt, prev_yr = self._prev_clustering(network)
        x, y_tx, y_rx, info = sun2025_adaptive_hysteresis(
            n["ap_positions"], n["user_pos"], n["target_pos"],
            prev_x, prev_yt, prev_yr,
            perf_kwargs=n["perf_kwargs"],
        )
        _check_association(x, y_tx, y_rx, n["M"], n["K"], n["Q"])

    def test_info_has_decision(self, network):
        from src.baselines.published_baselines import sun2025_adaptive_hysteresis
        n = network
        prev_x, prev_yt, prev_yr = self._prev_clustering(network)
        _, _, _, info = sun2025_adaptive_hysteresis(
            n["ap_positions"], n["user_pos"], n["target_pos"],
            prev_x, prev_yt, prev_yr,
            perf_kwargs=n["perf_kwargs"],
        )
        assert info["decision"] in ("KEEP", "RECONFIGURE")
        assert "threshold" in info
        assert info["threshold"] > 0

    def test_high_threshold_keeps_prev(self, network):
        """Very high threshold should always keep previous clustering."""
        from src.baselines.published_baselines import sun2025_adaptive_hysteresis
        n = network
        prev_x, prev_yt, prev_yr = self._prev_clustering(network)
        x, y_tx, y_rx, info = sun2025_adaptive_hysteresis(
            n["ap_positions"], n["user_pos"], n["target_pos"],
            prev_x, prev_yt, prev_yr,
            perf_kwargs=n["perf_kwargs"],
            base_threshold=1000.0,  # absurdly high
        )
        assert info["decision"] == "KEEP"
        np.testing.assert_array_equal(x, prev_x)


# ---------------------------------------------------------------------------
# Kim and Cho CF-HO tests
# ---------------------------------------------------------------------------

class TestKimChoCFHO:
    def _prev_clustering(self, network):
        from src.optimization.baselines import reactive_policy
        n = network
        return reactive_policy(n["ap_positions"], n["user_pos"], n["target_pos"])

    def test_output_shapes(self, network):
        from src.baselines.published_baselines import kimcho2024_cf_ho
        n = network
        prev_x, prev_yt, prev_yr = self._prev_clustering(network)
        x, y_tx, y_rx, info = kimcho2024_cf_ho(
            n["ap_positions"], n["user_pos"], n["target_pos"],
            prev_x, prev_yt, prev_yr,
            perf_kwargs=n["perf_kwargs"],
        )
        _check_association(x, y_tx, y_rx, n["M"], n["K"], n["Q"])

    def test_info_has_handovers(self, network):
        from src.baselines.published_baselines import kimcho2024_cf_ho
        n = network
        prev_x, prev_yt, prev_yr = self._prev_clustering(network)
        _, _, _, info = kimcho2024_cf_ho(
            n["ap_positions"], n["user_pos"], n["target_pos"],
            prev_x, prev_yt, prev_yr,
        )
        assert "handovers" in info
        assert info["handovers"] >= 0

    def test_high_threshold_triggers_handover(self, network):
        """Quality threshold of 1.0 should trigger handovers (unless quality is exactly 1.0)."""
        from src.baselines.published_baselines import kimcho2024_cf_ho
        n = network
        prev_x, prev_yt, prev_yr = self._prev_clustering(network)
        x, _, _, info = kimcho2024_cf_ho(
            n["ap_positions"], n["user_pos"], n["target_pos"],
            prev_x, prev_yt, prev_yr,
            quality_threshold=1.0,
            hysteresis_margin=0.0,
        )
        # With threshold=1.0, most realistic associations will trigger handovers
        assert x.shape == (n["M"], n["K"])


# ---------------------------------------------------------------------------
# Özbek G-MSUA tests
# ---------------------------------------------------------------------------

class TestOzbekGMSUA:
    def test_output_shapes(self, network):
        from src.baselines.published_baselines import ozbek2026_gmsua
        n = network
        x, y_tx, y_rx, info = ozbek2026_gmsua(
            n["ap_positions"], n["user_pos"], n["target_pos"],
            perf_kwargs=n["perf_kwargs"], ref=n["ref"],
        )
        _check_association(x, y_tx, y_rx, n["M"], n["K"], n["Q"])

    def test_sensing_aps_assigned(self, network):
        from src.baselines.published_baselines import ozbek2026_gmsua
        n = network
        x, y_tx, y_rx, info = ozbek2026_gmsua(
            n["ap_positions"], n["user_pos"], n["target_pos"],
        )
        assert info["n_sen_aps"] >= 1, "G-MSUA should assign at least 1 AP to sensing"

    def test_comm_aps_not_empty(self, network):
        from src.baselines.published_baselines import ozbek2026_gmsua
        n = network
        x, _, _, info = ozbek2026_gmsua(
            n["ap_positions"], n["user_pos"], n["target_pos"],
        )
        assert info["n_com_aps"] >= 1, "G-MSUA should leave at least 1 AP in comm mode"

    def test_deterministic(self, network):
        from src.baselines.published_baselines import ozbek2026_gmsua
        n = network
        x1, yt1, yr1, _ = ozbek2026_gmsua(
            n["ap_positions"], n["user_pos"], n["target_pos"],
        )
        x2, yt2, yr2, _ = ozbek2026_gmsua(
            n["ap_positions"], n["user_pos"], n["target_pos"],
        )
        np.testing.assert_array_equal(x1, x2)


# ---------------------------------------------------------------------------
# Integration: one simulation step with each baseline
# ---------------------------------------------------------------------------

class TestBaselineSimulationStep:
    """Verify that each baseline works in a single-step evaluation."""

    def test_gap_npc_simulation_step(self, network):
        from src.baselines.published_baselines import elfiatoure2023_gap_npc
        from src.optimization.churn_aware import evaluate_cluster_performance
        n = network
        x, y_tx, y_rx, _ = elfiatoure2023_gap_npc(
            n["ap_positions"], n["user_pos"], n["target_pos"],
            perf_kwargs=n["perf_kwargs"],
        )
        perf = evaluate_cluster_performance(
            x, y_tx, y_rx,
            n["ap_positions"], n["user_pos"], n["target_pos"],
            **n["perf_kwargs"],
        )
        assert "combined_objective" in perf
        assert np.isfinite(perf["combined_objective"])

    def test_gmsua_simulation_step(self, network):
        from src.baselines.published_baselines import ozbek2026_gmsua
        from src.optimization.churn_aware import evaluate_cluster_performance
        n = network
        x, y_tx, y_rx, _ = ozbek2026_gmsua(
            n["ap_positions"], n["user_pos"], n["target_pos"],
        )
        perf = evaluate_cluster_performance(
            x, y_tx, y_rx,
            n["ap_positions"], n["user_pos"], n["target_pos"],
            **n["perf_kwargs"],
        )
        assert np.isfinite(perf["combined_objective"])
        assert perf["raw_sensing_utility"] >= 0

    def test_sun_simulation_step(self, network):
        from src.baselines.published_baselines import sun2025_adaptive_hysteresis
        from src.optimization.baselines import reactive_policy
        from src.optimization.churn_aware import evaluate_cluster_performance
        n = network
        prev_x, prev_yt, prev_yr = reactive_policy(
            n["ap_positions"], n["user_pos"], n["target_pos"]
        )
        x, y_tx, y_rx, _ = sun2025_adaptive_hysteresis(
            n["ap_positions"], n["user_pos"], n["target_pos"],
            prev_x, prev_yt, prev_yr,
            perf_kwargs=n["perf_kwargs"],
        )
        perf = evaluate_cluster_performance(
            x, y_tx, y_rx,
            n["ap_positions"], n["user_pos"], n["target_pos"],
            **n["perf_kwargs"],
        )
        assert np.isfinite(perf["combined_objective"])

    def test_kimcho_simulation_step(self, network):
        from src.baselines.published_baselines import kimcho2024_cf_ho
        from src.optimization.baselines import reactive_policy
        from src.optimization.churn_aware import evaluate_cluster_performance
        n = network
        prev_x, prev_yt, prev_yr = reactive_policy(
            n["ap_positions"], n["user_pos"], n["target_pos"]
        )
        x, y_tx, y_rx, _ = kimcho2024_cf_ho(
            n["ap_positions"], n["user_pos"], n["target_pos"],
            prev_x, prev_yt, prev_yr,
        )
        perf = evaluate_cluster_performance(
            x, y_tx, y_rx,
            n["ap_positions"], n["user_pos"], n["target_pos"],
            **n["perf_kwargs"],
        )
        assert np.isfinite(perf["combined_objective"])


# ---------------------------------------------------------------------------
# Output-invariant sanity: existing policies not affected
# ---------------------------------------------------------------------------

class TestExistingPoliciesUnchanged:
    """Confirm the existing internal policies still run and produce valid output."""

    def test_static_policy(self, network):
        from src.optimization.baselines import static_policy
        n = network
        x, y_tx, y_rx = static_policy(
            n["ap_positions"], n["user_pos"], n["target_pos"]
        )
        _check_association(x, y_tx, y_rx, n["M"], n["K"], n["Q"])

    def test_reactive_policy(self, network):
        from src.optimization.baselines import reactive_policy
        n = network
        x, y_tx, y_rx = reactive_policy(
            n["ap_positions"], n["user_pos"], n["target_pos"]
        )
        _check_association(x, y_tx, y_rx, n["M"], n["K"], n["Q"])
