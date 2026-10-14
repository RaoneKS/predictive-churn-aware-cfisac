# Reproducibility Guide

**Branch:** baseline-comparison
**Last updated:** 2026-10-15

## Experiment Overview

This guide explains how to reproduce the published-baseline comparison experiment
for the Predictive Churn-Aware CF-ISAC research project.

## Prerequisites

```bash
# Clone the repository
git clone https://github.com/RaoneKS/predictive-churn-aware-cfisac.git
cd predictive-churn-aware-cfisac
git checkout baseline-comparison

# Create and activate the virtual environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Smoke Test (Fast Validation)

Runs 3 seeds × 11 methods × 20 blocks ≈ ~30 seconds.

```bash
source .venv/bin/activate
python -m experiments.run_published_baseline_comparison --smoke
```

Expected output:
```
✓ Experiment complete: results/published_baseline_comparison_<TIMESTAMP>
  Figures: 35
  Old graphs intact: True
  Runs: 33 OK / 0 failed
```

## Full Experiment (10 Seeds)

Runs 10 seeds × 11 methods × 100 blocks ≈ ~5-10 minutes.

```bash
source .venv/bin/activate
python -m experiments.run_published_baseline_comparison --seeds 10
```

## Larger Campaign (30 Seeds)

```bash
source .venv/bin/activate
python -m experiments.run_published_baseline_comparison --seeds 30
```

> [!NOTE]
> Each run creates a uniquely timestamped output directory in `results/`.
> The script REFUSES to overwrite any existing directory. If the directory
> already exists, it exits with an error.

## Unit Tests

```bash
source .venv/bin/activate
# Run baseline-specific tests
python -m pytest tests/test_published_baselines.py -v

# Run all project tests
python -m pytest tests/ -v
```

## Output Structure

Every experiment run creates:

```
results/published_baseline_comparison_<YYYYMMDD_HHMMSS>/
├── figures/
│   ├── bar_avg_user_rate.png
│   ├── timeseries_avg_user_rate.png
│   ├── boxplot_avg_user_rate.png
│   ├── ... (35 total figures)
│   └── tradeoff_avg_user_rate_vs_sensing_utility.png
├── per_seed_block_metrics.csv    # raw per-block data
├── per_seed_summary.csv          # per-seed episode means
├── aggregate_summary.csv         # per-method across-seed stats
├── confidence_intervals.csv      # 95% CI per method and metric
├── statistical_tests.csv         # paired t-tests vs proposed method
├── improvement_table.csv         # % improvement vs each baseline
├── reproduction_status.json      # baseline labels and metadata
├── experiment_config.json        # exact parameters and git commit
├── seed_manifest.csv             # seed → status mapping
├── failed_runs.log               # (empty if all succeeded)
└── EXPERIMENT_REPORT.md          # human-readable summary
```

## Methods Compared

### Internal Policies (this project)

| Method | Description |
|--------|-------------|
| `STATIC` | One-time clustering at t=0 |
| `REACTIVE` | Re-cluster on current positions every step |
| `PREDICTIVE` | Re-cluster on predicted future positions |
| `PREDICTIVE+CHURN` | **Proposed method**: churn-aware predictive clustering |
| `OBJECTIVE-HYSTERESIS` | Heuristic: re-cluster only if objective gain > fixed threshold |
| `REACTIVE+CHURN` | Heuristic: reactive with explicit normalized switching cost |

### Published Baselines (paper-inspired proxies)

| Method | Paper | Reproduction Label |
|--------|-------|-------------------|
| `SUN2025_ADAPTIVE_HYSTERESIS` | Sun et al. IEEE IoT J. 2025 | `PAPER_INSPIRED_PROXY` |
| `ELFIATOURE2023_GAP_NPC` | Elfiatoure et al. arXiv:2310.09032 | `PAPER_INSPIRED_PROXY` |
| `ELFIATOURE2023_RAP_NPC` | (same paper — RAP-NPC baseline) | `PAPER_INSPIRED_PROXY` |
| `KIMCHO2024_CF_HO` | Kim and Cho 2024 | `PAPER_INSPIRED_PROXY` |
| `OZBEK2026_GMSUA` | Özbek and Cumalı Phys. Comm. 2026 | `PAPER_INSPIRED_PROXY` |

> [!IMPORTANT]
> All external baselines are **PAPER_INSPIRED_PROXY** implementations.
> They cannot be directly compared to results in the original papers because
> the native objectives (MASR, max-min SE, sensing SNR) are proxied by
> the simulator's scalar metrics.

## Historical Results

The existing historical experiment results are **never** modified by new runs:

```
results/professor_submission_corrected_20261013/   # 30 historical graphs
results/four_policy_comparison/
results/phase3/
results/prediction/
results/presentation/
results/presentation_final/
...
```

Every new run writes to a fresh, timestamped directory.

## Configuration

The default configuration is at `configs/default.yaml`:

```yaml
network:
  num_aps: 20
  num_users: 5
  num_targets: 2
  area_size: 500.0
simulation:
  time_blocks: 100
  prediction_horizon: 5
```

## Known Limitations

1. **Physical-layer channel model not used in baselines**: External baselines
   cannot use MASR or max-min SE expressions because those require the massive-MIMO
   physical channel model (large-scale fading, conjugate precoding, MMSE estimation).

2. **POMDP baseline not implemented**: The Ammar et al. (2024) POMDP handoff
   policy requires a full training procedure and cannot be reproduced without
   the original code.

3. **Beamforming baseline excluded**: The Demirhan and Alkhateeb (2023)
   beamforming design is incompatible with the mobility/association problem space.

4. **Simulation does not include physical fading for external baselines**:
   The two-timescale module (`src/optimization/two_timescale.py`) uses physical
   channel fading in the fast timescale, but external baselines are evaluated
   using the scalar channel model (same as STATIC/REACTIVE). This is documented
   in `experiment_config.json`.

## Source Code

| File | Purpose |
|------|---------|
| `src/baselines/published_baselines.py` | All external baseline algorithms |
| `src/simulation/end_to_end.py` | Internal policy simulator |
| `experiments/run_published_baseline_comparison.py` | Experiment runner |
| `tests/test_published_baselines.py` | Baseline unit tests |
| `docs/PUBLISHED_BASELINE_IMPLEMENTATION_STATUS.md` | Detailed implementation notes |
| `docs/LITERATURE_BASELINE_AUDIT.md` | Literature screening |
| `docs/PUBLISHED_BASELINE_COMPARISON_PROTOCOL.md` | Comparison protocol |
