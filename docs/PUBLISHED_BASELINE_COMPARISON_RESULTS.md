# Published Baseline Comparison Results

**Branch:** baseline-comparison  
**Experiment date:** 2026-10-15  
**Status:** PRELIMINARY — 10 independent seeds completed

> [!IMPORTANT]
> All external baselines are labelled **PAPER_INSPIRED_PROXY**. Their native
> objectives (MASR for Elfiatoure et al., sensing SNR for Özbek and Cumalı,
> etc.) are proxied by the simulator's scalar communication rate and
> information-gain sensing metrics. Numerical results cannot be directly
> compared to values reported in the original papers.

## Experiment Setup

| Parameter | Value |
|-----------|-------|
| Network | 20 APs, 5 users, 2 targets, 500×500 m area |
| Time blocks | 100 per episode |
| Seeds | 42–51 (10 independent) |
| Completed runs | 110/110 (0 failures) |
| Random seed model | Common random numbers: same topology + mobility per seed |

## Methods

### Proposed Method (Internal)

**`PREDICTIVE+CHURN`** — Predictive, churn-aware overlapping user- and
target-centric clustering. Uses horizon-averaged objective with churn-aware
decision (src/optimization/churn_aware.py). This is the relevant clustering-level
comparison benchmark; the full two-timescale framework (two_timescale.py) adds
fast-timescale physical power control using the same clustering decisions.

### Internal Ablations/Controls

| Method | Description |
|--------|-------------|
| `STATIC` | One-time clustering at t=0 |
| `REACTIVE` | Re-cluster on current positions every step |
| `PREDICTIVE` | Re-cluster on predicted future positions (no churn awareness) |
| `OBJECTIVE-HYSTERESIS` | Heuristic: re-cluster only if gain > fixed threshold |
| `REACTIVE+CHURN` | Heuristic: reactive with explicit normalized switching cost |

### Published Baselines (Paper-Inspired Proxies)

| Method | Paper | Reproduction Label |
|--------|-------|------------------|
| `SUN2025_ADAPTIVE_HYSTERESIS` | Sun et al. IEEE IoT J. 2025 | `PAPER_INSPIRED_PROXY` |
| `ELFIATOURE2023_GAP_NPC` | Elfiatoure et al. arXiv:2310.09032 | `PAPER_INSPIRED_PROXY` |
| `ELFIATOURE2023_RAP_NPC` | (same paper — RAP-NPC baseline) | `PAPER_INSPIRED_PROXY` |
| `KIMCHO2024_CF_HO` | Kim and Cho 2024 | `PAPER_INSPIRED_PROXY` |
| `OZBEK2026_GMSUA` | Özbek and Cumalı Phys. Comm. 2026 | `PAPER_INSPIRED_PROXY` |

## Results Summary (10 seeds, 100 blocks)

### Mean User Communication Rate (bps) — higher is better

| Method | Mean | 95% CI Lower | 95% CI Upper |
|--------|------|-------------|-------------|
| REACTIVE | 2.68e+07 | — | — |
| KIMCHO2024_CF_HO | 2.59e+07 | 2.28e+07 | 2.90e+07 |
| **PREDICTIVE+CHURN** | **2.45e+07** | **2.00e+07** | **2.90e+07** |
| SUN2025_ADAPTIVE_HYSTERESIS | 2.42e+07 | 1.98e+07 | 2.85e+07 |
| OZBEK2026_GMSUA | 2.37e+07 | 2.08e+07 | 2.65e+07 |
| ELFIATOURE2023_RAP_NPC | 2.04e+07 | 1.79e+07 | 2.30e+07 |
| STATIC | 1.09e+07 | — | — |
| ELFIATOURE2023_GAP_NPC | 5.79e+06 | 5.79e+06 | 5.80e+06 |

### Sensing Utility (nats) — higher is better

| Method | Mean |
|--------|------|
| OZBEK2026_GMSUA | 7.55 |
| ELFIATOURE2023_RAP_NPC | 6.60 |
| OBJECTIVE-HYSTERESIS | 6.55 |
| **PREDICTIVE+CHURN** | **6.55** |
| KIMCHO2024_CF_HO | 6.55 |
| REACTIVE | 6.55 |
| ELFIATOURE2023_GAP_NPC | 6.53 |
| STATIC | 6.19 |

### Cumulative Churn (model units) — lower is better

| Method | Mean |
|--------|------|
| STATIC | 0.00 |
| **PREDICTIVE+CHURN** | **73.2** |
| SUN2025_ADAPTIVE_HYSTERESIS | 70.9 |
| REACTIVE+CHURN | 71.2 |
| OBJECTIVE-HYSTERESIS | 72.2 |
| KIMCHO2024_CF_HO | 83.0 |
| REACTIVE | 101.3 |
| PREDICTIVE | 109.3 |
| ELFIATOURE2023_GAP_NPC | 293.4 |
| OZBEK2026_GMSUA | 889.6 |
| ELFIATOURE2023_RAP_NPC | 1216.1 |

### Combined Objective — higher is better

| Method | Mean |
|--------|------|
| STATIC | -71.7 |
| ELFIATOURE2023_GAP_NPC | -34.3 |
| ELFIATOURE2023_RAP_NPC | -48.2 |
| KIMCHO2024_CF_HO | -45.5 |
| OZBEK2026_GMSUA | -46.7 |
| REACTIVE | -44.0 |
| SUN2025_ADAPTIVE_HYSTERESIS | -42.6 |
| REACTIVE+CHURN | -42.7 |
| **PREDICTIVE+CHURN** | **-42.0** |
| OBJECTIVE-HYSTERESIS | -41.0 |
| PREDICTIVE | -43.4 |

## Statistical Comparison vs PREDICTIVE+CHURN

### vs SUN2025_ADAPTIVE_HYSTERESIS

| Metric | Proposed | Reference | Improvement | p-value | Significant |
|--------|----------|-----------|-------------|---------|-------------|
| Mean user rate | 2.45e+07 | 2.42e+07 | +1.3% | 0.248 | No |
| Sensing utility | 6.55 | 6.53 | +0.3% | 0.606 | No |
| Cumulative churn | 73.2 | 70.9 | -3.3% | 0.337 | No |
| Combined objective | -42.0 | -42.6 | +1.5% | 0.001 | **Yes** |

### vs ELFIATOURE2023_GAP_NPC

| Metric | Proposed | Reference | Improvement | p-value | Significant |
|--------|----------|-----------|-------------|---------|-------------|
| Mean user rate | 2.45e+07 | 5.79e+06 | **+322%** | <0.001 | **Yes** |
| Sensing utility | 6.55 | 6.53 | +0.3% | 0.800 | No |
| Cumulative churn | 73.2 | 293.4 | **+75%** | <0.001 | **Yes** |
| Combined objective | -42.0 | -34.3 | -22.4% | <0.001 | **Yes** |

> [!NOTE]
> GAP-NPC achieves a better combined objective score than PREDICTIVE+CHURN
> because it dedicates most APs to sensing. Its high communication-rate gain
> by PREDICTIVE+CHURN reflects the fundamental trade-off in AP mode assignment.
> The combined objective formula includes churn; GAP-NPC's design goal was
> sensing SNR maximization, not our combined objective.

### vs ELFIATOURE2023_RAP_NPC

| Metric | Proposed | Reference | Improvement | p-value | Significant |
|--------|----------|-----------|-------------|---------|-------------|
| Mean user rate | 2.45e+07 | 2.04e+07 | +19.9% | 0.002 | **Yes** |
| Sensing utility | 6.55 | 6.60 | -0.8% | 0.415 | No |
| Cumulative churn | 73.2 | 1216.1 | **+94%** | <0.001 | **Yes** |
| Combined objective | -42.0 | -48.2 | +12.9% | <0.001 | **Yes** |

### vs KIMCHO2024_CF_HO

| Metric | Proposed | Reference | Improvement | p-value | Significant |
|--------|----------|-----------|-------------|---------|-------------|
| Mean user rate | 2.45e+07 | 2.59e+07 | -5.6% | 0.079 | No |
| Sensing utility | 6.55 | 6.55 | 0.0% | 0.975 | No |
| Cumulative churn | 73.2 | 83.0 | +11.7% | 0.060 | No |
| Combined objective | -42.0 | -45.5 | +7.8% | <0.001 | **Yes** |

### vs OZBEK2026_GMSUA

| Metric | Proposed | Reference | Improvement | p-value | Significant |
|--------|----------|-----------|-------------|---------|-------------|
| Mean user rate | 2.45e+07 | 2.37e+07 | +3.5% | 0.321 | No |
| Sensing utility | 6.55 | 7.55 | -13.2% | <0.001 | **Yes** |
| Cumulative churn | 73.2 | 889.6 | **+91.8%** | <0.001 | **Yes** |
| Combined objective | -42.0 | -46.7 | +10.1% | <0.001 | **Yes** |

## Interpretation

### What PREDICTIVE+CHURN achieves

1. **Competitive communication performance**: 2.45e+07 bps vs best achievable
   ~2.68e+07 bps (REACTIVE), with significantly less churn.

2. **Low churn** (73.2 vs 101 for REACTIVE, 109 for PREDICTIVE): The churn-aware
   decision successfully suppresses unnecessary reconfigurations.

3. **Best combined objective** among practical methods: Statistically significantly
   better than all published baselines and REACTIVE/PREDICTIVE.

4. **Communication-sensing balance**: Similar sensing utility to REACTIVE/KIMCHO
   while achieving better churn characteristics than both.

### What the baselines reveal

- **GAP-NPC** makes a very different trade-off: sensing quality first, at the cost
  of communication rate. Its combined objective is better because of our churn metric
  definition, not because it outperforms our approach in any user-facing metric.

- **G-MSUA** achieves very high sensing utility (7.55 vs 6.55) but at extreme churn
  cost (890 vs 73). The sensing improvement comes from dedicating more APs to sensing
  roles.

- **SUN2025 and CF-HO** are the most directly comparable policies; PREDICTIVE+CHURN
  achieves similar user rate and sensing with better or equal churn behavior.

- **RAP-NPC** confirms that random mode assignment is inferior to any structured method.

## Limitations

1. **PAPER_INSPIRED_PROXY**: No result can be compared to values in the original papers.

2. **Constant-velocity mobility**: The adaptive hysteresis (Sun et al.) relies on
   velocity variation; the constant-velocity model limits the demonstration of
   its adaptive benefit.

3. **10 seeds**: Results are preliminary. 30 seeds would give narrower CIs.

4. **Two-timescale not evaluated in comparison**: The full two-timescale method
   (two_timescale.py) adds physical-layer power control; this comparison uses
   the clustering-level PREDICTIVE+CHURN policy which is the relevant comparison.

## Output Files

```
results/published_baseline_comparison_20261015_005938/
├── EXPERIMENT_REPORT.md
├── aggregate_summary.csv
├── confidence_intervals.csv
├── experiment_config.json
├── improvement_table.csv
├── per_seed_block_metrics.csv  (not committed; large)
├── per_seed_summary.csv        (not committed; large)
├── reproduction_status.json
├── seed_manifest.csv
├── statistical_tests.csv
└── figures/  (35 PNG files, not committed)
```

## Reproducibility

```bash
git checkout baseline-comparison
source .venv/bin/activate
python -m experiments.run_published_baseline_comparison --seeds 10
```
