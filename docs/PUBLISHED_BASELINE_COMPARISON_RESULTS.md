# Published Baseline Comparison Results

**Branch:** baseline-comparison
**Experiment date:** 2026-10-15
**Audit date:** 2026-10-15 (see `docs/AUDIT_REPORT.md`)
**Status:** CORRECTED — statistical analysis updated after audit

> [!IMPORTANT]
> All external baselines are labelled **PAPER_INSPIRED_PROXY**. Their native
> objectives (MASR, max-min SE, sensing SNR) are proxied by this simulator's
> scalar channel and information-gain metrics. Numerical results cannot be
> compared to values reported in the original papers.

> [!NOTE]
> Statistical significance claims in this document use Holm-Bonferroni
> family-wise error rate (FWER) correction within two pre-specified families
> (see `docs/AUDIT_REPORT.md` for full methodology). Both raw and adjusted
> p-values are available in `statistical_tests.csv`. Where this document
> uses the term "statistically significant," it refers to the Holm-adjusted
> significance (adjusted p < 0.05) unless otherwise stated.

## Experiment Setup

| Parameter | Value |
|-----------|-------|
| Network | 20 APs, 5 users, 2 targets, 500 × 500 m area |
| Time blocks | 100 per episode |
| Seeds | 42–51 (10 independent) |
| Completed runs | 110 / 110 (0 failures) |
| Random seed model | Common random numbers: same topology + mobility per seed |
| Proposed method | `PREDICTIVE+CHURN` |

## Methods

### Proposed Method

**`PREDICTIVE+CHURN`** — Predictive churn-aware overlapping user- and
target-centric clustering. Uses horizon-averaged objective with churn-aware
decision (`src/optimization/churn_aware.py`). This is the primary
clustering-level comparison benchmark; the full two-timescale framework
(`two_timescale.py`) adds fast-timescale physical power control using the
same clustering decisions.

### Internal Controls

| Method | Description |
|--------|-------------|
| `STATIC` | One-time clustering at t=0 |
| `REACTIVE` | Re-cluster on current positions every step |
| `PREDICTIVE` | Re-cluster on predicted future positions (no churn awareness) |
| `OBJECTIVE-HYSTERESIS` | Heuristic: re-cluster only if objective gain > fixed threshold |
| `REACTIVE+CHURN` | Heuristic: reactive with explicit normalised switching cost |

### Published Baselines (Paper-Inspired Proxies)

| Method | Paper | Reproduction Label |
|--------|-------|------------------|
| `SUN2025_ADAPTIVE_HYSTERESIS` | Sun et al. IEEE IoT J. 2025 | `PAPER_INSPIRED_PROXY` |
| `ELFIATOURE2023_GAP_NPC` | Elfiatoure et al. arXiv:2310.09032 | `PAPER_INSPIRED_PROXY` |
| `ELFIATOURE2023_RAP_NPC` | (same paper — RAP-NPC baseline) | `PAPER_INSPIRED_PROXY` |
| `KIMCHO2024_CF_HO` | Kim and Cho 2024 | `PAPER_INSPIRED_PROXY` |
| `OZBEK2026_GMSUA` | Özbek and Cumalı Phys. Comm. 2026 | `PAPER_INSPIRED_PROXY` |

## Aggregate Results (10 seeds, 100 blocks)

### Combined Objective — higher is better

> [!NOTE]
> The combined objective is:
> `alpha * (total_comm_rate / ref_comm) + beta * (sensing_gain / ref_sensing) - lambda_energy * (energy / ref_energy) - lambda_fronthaul * (fronthaul / ref_fronthaul) - lambda_churn * (churn / ref_churn_events)`
> with alpha=beta=lambda_energy=lambda_fronthaul=lambda_churn=1. The energy
> reference value (ref_energy=1 watt) is much smaller than the actual per-block
> energy consumption (~20–42 units), so the energy term dominates the objective.
> A method that uses fewer APs (lower energy) can therefore rank higher even
> with substantially lower communication throughput. This is a known design
> characteristic of this objective formulation, not a coding defect.

| Method | Mean Combined Obj. | Rank |
|--------|-------------------|------|
| ELFIATOURE2023_GAP_NPC | **−34.310** | 1 |
| PREDICTIVE+CHURN | −41.980 | 2 |
| OBJECTIVE-HYSTERESIS | −42.558 | 3 |
| SUN2025_ADAPTIVE_HYSTERESIS | −42.614 | 4 |
| REACTIVE+CHURN | −42.636 | 5 |
| REACTIVE | −45.486 | 6 |
| KIMCHO2024_CF_HO | −45.519 | 7 |
| PREDICTIVE | −46.002 | 8 |
| OZBEK2026_GMSUA | −46.690 | 9 |
| ELFIATOURE2023_RAP_NPC | −48.182 | 10 |
| STATIC | −275,179 | 11 |

**GAP-NPC ranks first** on this metric because it dedicates most APs to
sensing, incurring much lower energy cost (raw energy ≈ 24.6 vs ≈ 41.7 for
PREDICTIVE+CHURN), which outweighs the communication-rate loss under the
current objective weights. PREDICTIVE+CHURN ranks second among all methods.

### Mean User Communication Rate (bps) — higher is better

| Method | Mean | 95% CI |
|--------|------|--------|
| REACTIVE | 26.78 Mbps | — |
| KIMCHO2024_CF_HO | 25.92 Mbps | [22.82M, 29.01M] |
| PREDICTIVE | 25.76 Mbps | — |
| OBJECTIVE-HYSTERESIS | 24.67 Mbps | — |
| **PREDICTIVE+CHURN** | **24.48 Mbps** | **[19.95M, 29.01M]** |
| REACTIVE+CHURN | 24.19 Mbps | — |
| SUN2025_ADAPTIVE_HYSTERESIS | 24.16 Mbps | [19.78M, 28.55M] |
| OZBEK2026_GMSUA | 23.65 Mbps | [20.76M, 26.55M] |
| ELFIATOURE2023_RAP_NPC | 20.42 Mbps | [17.85M, 22.99M] |
| STATIC | 10.88 Mbps | — |
| ELFIATOURE2023_GAP_NPC | 5.80 Mbps | [5.80M, 5.80M] |

### Sensing Utility (nats) — higher is better

| Method | Mean |
|--------|------|
| OZBEK2026_GMSUA | **7.552** |
| ELFIATOURE2023_RAP_NPC | 6.601 |
| OBJECTIVE-HYSTERESIS | 6.555 |
| PREDICTIVE+CHURN | 6.551 |
| KIMCHO2024_CF_HO | 6.549 |
| REACTIVE | 6.549 |
| PREDICTIVE | 6.553 |
| SUN2025_ADAPTIVE_HYSTERESIS | 6.534 |
| ELFIATOURE2023_GAP_NPC | 6.529 |
| STATIC | 6.189 |

### Cumulative Churn (model units) — lower is better

| Method | Mean |
|--------|------|
| STATIC | 0.0 |
| SUN2025_ADAPTIVE_HYSTERESIS | 70.85 |
| REACTIVE+CHURN | 71.18 |
| OBJECTIVE-HYSTERESIS | 72.18 |
| **PREDICTIVE+CHURN** | **73.23** |
| KIMCHO2024_CF_HO | 82.95 |
| REACTIVE | 101.34 |
| PREDICTIVE | 109.32 |
| ELFIATOURE2023_GAP_NPC | 293.40 |
| OZBEK2026_GMSUA | 889.62 |
| ELFIATOURE2023_RAP_NPC | 1216.09 |

## Statistical Comparison vs PREDICTIVE+CHURN (Family A, Holm-adjusted)

Comparison family: 5 published baselines × 8 valid metrics = 40 tests.
Only comparisons with zero-variance paired differences are excluded. The QoS comparison against STATIC is valid and included in Family B; other QoS comparisons and the constant fronthaul comparisons are untestable.
Multiple-comparison correction: Holm-Bonferroni FWER control (alpha=0.05).
All 22 raw-significant results remained significant after Holm correction.

### vs SUN2025_ADAPTIVE_HYSTERESIS

| Metric | Proposed | Reference | Improvement | p (raw) | p (adj) | Sig* |
|--------|----------|-----------|-------------|---------|---------|------|
| avg_user_rate | 24.48M | 24.16M | +1.3% | 0.248 | 1.000 | No |
| p5_user_rate | — | — | +2.1% | 0.209 | 1.000 | No |
| sensing_utility | 6.551 | 6.534 | +0.3% | 0.606 | 1.000 | No |
| tracking_violations | — | — | −9.4% | 0.157 | 1.000 | No |
| cumulative_churn | 73.23 | 70.85 | −3.3% | 0.337 | 1.000 | No |
| combined_objective | −41.98 | −42.61 | +1.5% | 0.001 | 0.021 | **Yes** |
| energy | — | — | +1.1% | 0.002 | 0.043 | **Yes** |
| reconfigurations | — | — | −1.3% | 0.868 | 1.000 | No |

*Holm-adjusted p < 0.05

**Summary:** PREDICTIVE+CHURN does not statistically outperform SUN2025 on user
rate, sensing utility, or churn reduction at the family-corrected threshold.
The combined objective and energy penalty differences are significant, but both
are very small in absolute terms (< 2%).

### vs ELFIATOURE2023_GAP_NPC

| Metric | Proposed | Reference | Improvement | p (raw) | p (adj) | Sig* |
|--------|----------|-----------|-------------|---------|---------|------|
| avg_user_rate | 24.48M | 5.80M | +322% | <0.001 | 0.0002 | **Yes** |
| p5_user_rate | — | — | +105% | <0.001 | 0.0012 | **Yes** |
| sensing_utility | 6.551 | 6.529 | +0.3% | 0.800 | 1.000 | No |
| tracking_violations | — | — | −2.4% | 0.908 | 1.000 | No |
| cumulative_churn | 73.23 | 293.40 | +75% | <0.001 | 0.0006 | **Yes** |
| **combined_objective** | **−41.98** | **−34.31** | **−22%** | <0.001 | <0.001 | **Yes (GAP-NPC better)** |
| energy | — | — | −70% | <0.001 | <0.001 | **Yes (GAP-NPC lower)** |
| reconfigurations | — | — | +66% | <0.001 | 0.0001 | **Yes** |

**Summary:** PREDICTIVE+CHURN significantly outperforms GAP-NPC on user rate,
5th-percentile rate, and churn reduction. However, GAP-NPC significantly outperforms
PREDICTIVE+CHURN on combined objective and energy penalty, because GAP-NPC uses far
fewer APs in communication mode, reducing energy consumption substantially.

### vs ELFIATOURE2023_RAP_NPC

| Metric | Proposed | Reference | Improvement | p (raw) | p (adj) | Sig* |
|--------|----------|-----------|-------------|---------|---------|------|
| avg_user_rate | 24.48M | 20.42M | +19.9% | 0.002 | 0.038 | **Yes** |
| p5_user_rate | — | — | +5.9% | 0.239 | 1.000 | No |
| sensing_utility | 6.551 | 6.601 | −0.8% | 0.415 | 1.000 | No |
| tracking_violations | — | — | −15.8% | 0.260 | 1.000 | No |
| cumulative_churn | 73.23 | 1216.09 | +94% | <0.001 | <0.001 | **Yes** |
| combined_objective | −41.98 | −48.18 | +12.9% | <0.001 | 0.0001 | **Yes** |
| energy | — | — | +8.2% | <0.001 | 0.0075 | **Yes** |
| reconfigurations | — | — | +84% | <0.001 | <0.001 | **Yes** |

**Summary:** PREDICTIVE+CHURN significantly outperforms RAP-NPC on user rate,
churn, combined objective, and reconfiguration count.

### vs KIMCHO2024_CF_HO

| Metric | Proposed | Reference | Improvement | p (raw) | p (adj) | Sig* |
|--------|----------|-----------|-------------|---------|---------|------|
| avg_user_rate | 24.48M | 25.92M | −5.6% | 0.079 | 1.000 | No |
| p5_user_rate | — | — | −3.2% | 0.457 | 1.000 | No |
| sensing_utility | 6.551 | 6.549 | 0.0% | 0.975 | 1.000 | No |
| tracking_violations | — | — | −6.6% | 0.606 | 1.000 | No |
| cumulative_churn | 73.23 | 82.95 | +11.7% | 0.060 | 1.000 | No |
| combined_objective | −41.98 | −45.52 | +7.8% | <0.001 | 0.0009 | **Yes** |
| energy | — | — | +9.2% | <0.001 | 0.0087 | **Yes** |
| reconfigurations | — | — | +62% | <0.001 | <0.001 | **Yes** |

**Summary:** CF-HO achieves a higher mean user rate than PREDICTIVE+CHURN (25.92
vs 24.48 Mbps), and this difference is not statistically significant at the 0.05
level (unadjusted p=0.079, adjusted p=1.000). The churn reduction of 11.7% is
also non-significant after correction (p=0.060 raw, 1.000 adjusted). The combined
objective and energy improvements are significant, driven by PREDICTIVE+CHURN's
lower energy use.

### vs OZBEK2026_GMSUA

| Metric | Proposed | Reference | Improvement | p (raw) | p (adj) | Sig* |
|--------|----------|-----------|-------------|---------|---------|------|
| avg_user_rate | 24.48M | 23.65M | +3.5% | 0.321 | 1.000 | No |
| p5_user_rate | — | — | −4.9% | 0.188 | 1.000 | No |
| sensing_utility | 6.551 | 7.552 | **−13.2%** | <0.001 | <0.001 | **Yes (G-MSUA better)** |
| tracking_violations | — | — | — | <0.001 | 0.0011 | **Yes** |
| cumulative_churn | 73.23 | 889.62 | +91.8% | <0.001 | <0.001 | **Yes** |
| combined_objective | −41.98 | −46.69 | +10.1% | <0.001 | 0.0009 | **Yes** |
| energy | — | — | +10.6% | <0.001 | 0.0006 | **Yes** |
| reconfigurations | — | — | +83% | <0.001 | <0.001 | **Yes** |

**Summary:** G-MSUA achieves significantly higher sensing utility (7.55 vs 6.55).
PREDICTIVE+CHURN significantly outperforms G-MSUA on churn reduction (+91.8%),
combined objective, and energy. User rate difference is not statistically
significant.

## Zero-Variance Metrics and QoS Violations

- **`qos_violations`**: All methods except `STATIC` report zero violations
  across the 10 seeds. `STATIC` has nonzero violations because its cluster
  selection is frozen while users move. PREDICTIVE+CHURN versus `STATIC` is
  statistically testable (raw p = 0.000271; Family B Holm-adjusted p = 0.008117).
  The other QoS comparisons have zero-variance paired differences and therefore
  have NaN p-values.
- **`fronthaul`**: All methods report a constant value of 19.0 across the seeds
  in this configuration. Its paired differences have zero variance, so these
  comparisons are not testable.

Holm correction is applied only to valid tests with finite raw p-values. The
corrected families contain 40 valid tests in Family A and 41 in Family B.

## Interpretation

### Performance of PREDICTIVE+CHURN

PREDICTIVE+CHURN provides a trade-off between communication performance and
reconfiguration stability. The results across 10 random seeds show the following:

1. **Reduced cumulative churn:** PREDICTIVE+CHURN achieves a mean cumulative
   churn of 73.23, compared with 109.32 for PREDICTIVE, representing an
   approximately 33% reduction (Holm-adjusted p < 0.001 in Family B).

2. **Communication-rate trade-off:** The mean user rate decreases from
   25.76 Mbps for PREDICTIVE to 24.48 Mbps for PREDICTIVE+CHURN, a reduction
   of approximately 5%. The paired test reports unadjusted p = 0.080,
   which is not significant even before adjustment.

3. **Improved combined objective versus PREDICTIVE:** The combined objective
   improves from −46.00 to −41.98 (Holm-adjusted p < 0.001, Family B).

4. **Comparison with REACTIVE:** PREDICTIVE+CHURN accepts lower mean user
   rate (24.48 vs 26.78 Mbps) and lower churn (73.23 vs 101.34).

5. **Sensing performance:** Mean sensing utility remains similar
   (approximately 6.55 for both PREDICTIVE and PREDICTIVE+CHURN).

### Interpretation of Published Baseline Comparisons

- **GAP-NPC** ranks first on the combined objective (−34.31 vs −41.98), because
  it dedicates most APs to sensing, thereby using far less energy (raw energy
  ≈ 24.6 vs ≈ 41.7 units). Its mean user rate (5.80 Mbps) is however
  substantially lower than PREDICTIVE+CHURN (24.48 Mbps, Holm p = 0.0002).
  These results demonstrate that the combined objective's energy weighting
  can produce rankings that diverge from throughput-based rankings.

- **SUN2025_ADAPTIVE_HYSTERESIS** achieves similar user rate and sensing
  utility to PREDICTIVE+CHURN. No statistically significant difference in
  churn reduction or user rate is established after Holm correction.

- **KIMCHO2024_CF_HO** achieves a higher mean user rate (25.92 Mbps vs
  24.48 Mbps), but this difference is not statistically significant (adjusted
  p = 1.000). PREDICTIVE+CHURN has lower churn (73.23 vs 82.95), also not
  significant after adjustment.

- **ELFIATOURE2023_RAP_NPC** has substantially higher cumulative churn
  (1216.09 vs 73.23, Holm p < 0.001) and lower mean user rate (20.42 vs
  24.48 Mbps, Holm p = 0.038). PREDICTIVE+CHURN outperforms RAP-NPC on
  both metrics.

- **OZBEK2026_GMSUA** achieves significantly higher sensing utility (7.55 vs
  6.55, Holm p < 0.001) at the cost of much higher churn (889.62 vs 73.23,
  Holm p < 0.001). User rate difference is not significant.

### Overall Finding

The results support a specific conclusion: adding churn awareness to the
predictive policy reduces cumulative churn and improves the reported combined
objective relative to PREDICTIVE, while maintaining similar sensing utility
and accepting a modest and statistically non-significant reduction in mean
user rate.

The experiments do **not** establish that PREDICTIVE+CHURN has the highest
combined-objective score across all evaluated methods. GAP-NPC achieves a
higher combined objective because of its lower energy consumption, not because
of superior communication or sensing performance.

PREDICTIVE+CHURN outperforms the two Elfiatoure et al. baselines (GAP-NPC
and RAP-NPC) on user rate and churn (Holm-adjusted). It does not statistically
outperform SUN2025 or CF-HO on user rate or churn after family-corrected testing.

### Limits of Comparability

- All published-paper baselines are **PAPER_INSPIRED_PROXY** implementations.
  Their native objectives (MASR, sensing SNR, max-min SE) are proxied by this
  simulator's scalar metrics. Numerical values cannot be compared to those
  reported in the original papers.

- The POMDP (Ammar et al. 2024) and beamforming (Demirhan and Alkhateeb 2023)
  baselines are **NOT_IMPLEMENTED** (see `docs/PUBLISHED_BASELINE_IMPLEMENTATION_STATUS.md`).

- Statistical significance is assessed at 10 seeds. Wider CI and reduced power
  relative to a 30-seed campaign should be noted.

## Files

| File | Description |
|------|-------------|
| `aggregate_summary.csv` | Per-method means and CIs |
| `confidence_intervals.csv` | 95% CIs per metric per method |
| `statistical_tests.csv` | Paired t-tests + Holm-adjusted p-values |
| `improvement_table.csv` | % improvement table |
| `per_seed_summary.csv` | Per-seed episode means |
| `per_seed_block_metrics.csv` | Per-block raw data (all seeds) |
| `figures/` | 35 PNG plots |
| `EXPERIMENT_REPORT.md` | Auto-generated run summary |
| `reproduction_status.json` | Baseline reproduction labels |
| `experiment_config.json` | Full experiment configuration |

## Reproducibility

```bash
git checkout baseline-comparison
source .venv/bin/activate
# Rerun the statistical correction only (no simulation rerun):
python scripts/audit_regenerate_statistics.py
# Rerun full experiment from scratch:
python -m experiments.run_published_baseline_comparison --seeds 10
```
