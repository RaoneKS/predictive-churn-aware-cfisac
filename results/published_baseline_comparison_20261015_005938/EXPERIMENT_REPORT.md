# Published-Baseline Comparison: Experiment Report

**Generated:** 2026-10-15 01:01:33
**Git branch:** baseline-comparison
**Git commit:** 4abcd3f5f67b03deeaad1e3576fe7a607328a192

## Summary

- Seeds: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
- Completed runs: 110 / 110
- Failed runs: 0
- Time blocks per episode: 100
- Smoke test: False

## Proposed Method

**PREDICTIVE+CHURN**: The PREDICTIVE+CHURN policy implements predictive, churn-aware overlapping clustering. It uses the existing Phase-2B horizon-averaged objective and churn_aware_decision (src/optimization/churn_aware.py). The full two-timescale integration (src/optimization/two_timescale.py) further wraps this with fast-timescale physical power control but uses the same slow-timescale clustering decision. PREDICTIVE+CHURN is the relevant clustering benchmark.

## External Baselines (Reproduction Labels)

### SUN2025_ADAPTIVE_HYSTERESIS
- **Paper:** Mobility Management Framework for Cooperative Cell-Free ISAC Systems
- **Authors:** Sun et al. (IEEE Internet of Things Journal, 2025)
- **DOI:** https://doi.org/10.1109/JIOT.2025.3604762
- **Reproduction label:** `PAPER_INSPIRED_PROXY`
- **Algorithm:** Adaptive-hysteresis handover with kinematic-information-based dynamic clustering. Hysteresis threshold adapts to measured user/target velocity.
- **Model differences:** Uses simulator's scalar communication rate and information-gain sensing metrics instead of paper's spectral-efficiency and radar SINR definitions. Velocity is estimated from mobility simulator state.

### ELFIATOURE2023_GAP_NPC
- **Paper:** Cell-Free Massive MIMO for ISAC: Access Point Operation Mode Selection and Power Control
- **Authors:** Elfiatoure et al. (arXiv:2310.09032, 2023)
- **DOI:** https://arxiv.org/abs/2310.09032
- **Reproduction label:** `PAPER_INSPIRED_PROXY`
- **Algorithm:** Algorithm 1 (GAP-NPC): Greedy AP mode selection. All APs start in sensing mode. Iteratively switch the AP that most increases min-user communication utility, while maintaining minimum sensing quality. No power control (NPC): full power used.
- **Model differences:** Native MASR and max-min SE require the paper's physical-layer channel model (large-scale fading, conjugate precoding, MMSE estimation). Proxied by simulator's sensing information gain and min-user communication rate. The structural selection logic is faithfully preserved.

### ELFIATOURE2023_RAP_NPC
- **Paper:** Cell-Free Massive MIMO for ISAC: Access Point Operation Mode Selection and Power Control
- **Authors:** Elfiatoure et al. (arXiv:2310.09032, 2023)
- **DOI:** https://arxiv.org/abs/2310.09032
- **Reproduction label:** `PAPER_INSPIRED_PROXY`
- **Algorithm:** RAP-NPC baseline from the paper: Random AP mode selection with no power control. APs are randomly partitioned into comm/sensing roles, with a minimum sensing quality guard.
- **Model differences:** Same proxy metrics as GAP-NPC. Used as reference within the Elfiatoure comparison.

### KIMCHO2024_CF_HO
- **Paper:** Performance Analysis of Handover Algorithms in Scalable Cell-Free Massive MIMO Systems
- **Authors:** Kim and Cho (2024 (publication details not fully verified), 2024)
- **DOI:** https://exa.ai/library/publication/p470n3hq9mj
- **Reproduction label:** `PAPER_INSPIRED_PROXY`
- **Algorithm:** Threshold-based handover: re-associate to APs when signal quality falls below threshold, using hysteresis to avoid ping-pong. Compatible reimplementation using simulator's channel quality proxy (communication feature gains).
- **Model differences:** Uses simulator's scalar channel quality proxy. Cannot verify exact algorithm without full paper access.

### OZBEK2026_GMSUA
- **Paper:** Joint mode selection and user association for ISAC-enabled cell-free MIMO networks
- **Authors:** Özbek and Cumalı (Physical Communication, 2026)
- **DOI:** https://doi.org/10.1016/j.phycom.2026.103132
- **Reproduction label:** `PAPER_INSPIRED_PROXY`
- **Algorithm:** G-MSUA: Greedy Mode Selection and User Association. First, greedily assign APs to sensing roles to maximize sensing utility. Then, associate users to best available communication APs satisfying minimum serving-AP count constraint.
- **Model differences:** Native sensing SNR and per-user SINR constraints require the paper's physical channel model. Proxied by simulator's sensing information gain and scalar communication rate metrics.

## NOT_IMPLEMENTED Baselines

### DEMIRHAN2023_BEAMFORMING
- Beamforming design problem is fundamentally different from AP association/clustering mobility management. Cannot be meaningfully compared without implementing a full physical-layer beamforming system. Excluded from numerical comparison.

### AMMAR2024_POMDP
- Requires a trained POMDP policy. Cannot reproduce without the original code, training protocol, or significant additional implementation effort beyond the scope of this comparison.

## Fairness Conditions

All methods used the same:
- Network topology (AP positions from seed)
- Initial user/target positions (from seed)
- Mobility model and trajectory (same seed)
- Evaluation duration (time_blocks)
- Performance evaluation function (evaluate_cluster_performance)
- Reference normalization values

## Metrics

| Metric | Label | Units | Higher=Better |
|--------|-------|-------|---------------|
| avg_user_rate | Mean user rate (bps) | bps | True |
| p5_user_rate | 5th-pctile user rate (bps) | bps | True |
| sensing_utility | Sensing information gain (nat) | nats | True |
| tracking_violations | Tracking violation level | model units | False |
| qos_violations | QoS violation level | model units | False |
| energy | Energy penalty | model units | False |
| fronthaul | Fronthaul load | model units | False |
| cumulative_churn | Cumulative topology churn | model units | False |
| combined_objective | Combined objective | norm. units | True |
| reconfigurations | Reconfigurations per block | events/block | False |

## Improvement vs Published Baselines

See `improvement_table.csv` and `statistical_tests.csv` for full results.

## Figures

Total figures: 35
Located in `figures/` subdirectory.
- `bar_*.png` — across-seed mean ± 1 SD bar comparison
- `timeseries_*.png` — time-series mean ± 1 SD
- `boxplot_*.png` — seed distribution box plots
- `tradeoff_*.png` — trade-off scatter plots

## Historical Results

The original 30 graphs (in `results/professor_submission_corrected_20261013/`)
and all other historical results remain intact and were not modified.
Old graphs verified intact: True

## Limitations

All external baselines are labelled **PAPER_INSPIRED_PROXY**. Their native
objectives (MASR, max-min SE, sensing SNR) are proxied by the simulator's
scalar communication rate and information-gain sensing metrics. Numerical
values cannot be directly compared to results reported in the original papers.

## Reproducibility

```bash
git checkout baseline-comparison
source .venv/bin/activate
# Smoke test (fast):
python -m experiments.run_published_baseline_comparison --smoke
# Full experiment (default 10 seeds):
python -m experiments.run_published_baseline_comparison --seeds 10
```
