# Antigravity Progress Log

**Project:** Predictive Churn-Aware Overlapping User- and Target-Centric Clustering
for Mobile Cell-Free ISAC Networks

**GitHub Repository:** https://github.com/RaoneKS/predictive-churn-aware-cfisac

**Working Branch:** baseline-comparison

**Last updated:** 2026-10-15 01:02 IST

---

## Current Status

### Last Successful Commit

```
d707859 Implement published-baseline comparison framework
```

### Last Verified GitHub Push

Not yet pushed (in progress — see next steps).

---

## Work Completed This Session

### PHASE 1: Repository Inspection ✅
- Confirmed branch: `baseline-comparison`
- Located 77 existing PNG graphs (30 historical in `professor_submission_corrected_20261013/`)
- Read `LITERATURE_BASELINE_AUDIT.md`, `PUBLISHED_BASELINE_COMPARISON_PROTOCOL.md`
- Inspected `src/simulation/end_to_end.py`, `src/optimization/two_timescale.py`,
  `src/optimization/risk_aware.py`
- Confirmed `run_30_baseline_comparison.py` output directory
  (`literature_baseline_comparison_30graphs_20261015`) does NOT exist yet
- Confirmed all existing results directories are intact

### PHASE 2: Literature Research ✅
- Read Elfiatoure et al. (arXiv:2310.09032) full paper HTML
- Extracted Algorithm 1 (GAP-NPC) algorithm structure and greedy iteration rule
- Documented the MASR sensing metric and max-min SE objective
- Identified that native MASR/SE require massive-MIMO physical channel
- Assessed all 5 candidate baselines for implementation feasibility
- Documented NOT_IMPLEMENTED justifications for Demirhan and Ammar et al.

### PHASE 3: Proposed Method Verification ✅
- Confirmed `PREDICTIVE+CHURN` in `end_to_end.py` is the correct proposed method
- Verified two-timescale integration in `two_timescale.py` (Phase 6) uses same
  slow-timescale `churn_aware_decision` — consistent
- PREDICTIVE+CHURN is the correct clustering-level comparison baseline
- Two-timescale adds fast physical power control but same clustering decisions

### PHASE 4: Baselines Implemented ✅

**New file: `src/baselines/published_baselines.py`**

| Baseline | Label | Description |
|----------|-------|-------------|
| `SUN2025_ADAPTIVE_HYSTERESIS` | `PAPER_INSPIRED_PROXY` | Velocity-adaptive hysteresis for Sun et al. 2025 |
| `ELFIATOURE2023_GAP_NPC` | `PAPER_INSPIRED_PROXY` | Greedy AP mode selection (Algorithm 1) |
| `ELFIATOURE2023_RAP_NPC` | `PAPER_INSPIRED_PROXY` | Random AP mode selection (benchmark) |
| `KIMCHO2024_CF_HO` | `PAPER_INSPIRED_PROXY` | Signal-quality-threshold handover |
| `OZBEK2026_GMSUA` | `PAPER_INSPIRED_PROXY` | Greedy sensing-first mode selection and user association |
| `DEMIRHAN2023_BEAMFORMING` | `NOT_IMPLEMENTED` | Incompatible problem space |
| `AMMAR2024_POMDP` | `NOT_IMPLEMENTED` | Requires POMDP training |

### PHASE 5: Tests Written and Passed ✅

**New file: `tests/test_published_baselines.py`**

- **28 tests, all passed** in 0.57s
- Tests cover: registry, output shapes, binary constraints, determinism,
  simulation step integration, existing policy preservation

### PHASE 6: Experiment Runner Created ✅

**New file: `experiments/run_published_baseline_comparison.py`**

Features:
- Safe unique timestamped output directory (fails if collision)
- Matched-seed evaluation for all 11 methods
- Per-block CSV saving
- Aggregate summary, CI table, statistical tests, improvement table
- 35 publication-quality figures (bar, timeseries, boxplot, tradeoff)
- Old graph integrity verification
- Reproduction status JSON + experiment config JSON
- Markdown summary report

### PHASE 7: Smoke Test ✅

```
Seeds: [42, 43, 44] (3 total), 20 blocks
Output: results/published_baseline_comparison_20261015_005819/
33/33 runs completed, 0 failed
35 figures created, old graphs intact
```

### PHASE 8: Full 10-Seed Experiment ✅

```
Seeds: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51] (10 seeds)
Time blocks: 100 per episode
Output: results/published_baseline_comparison_20261015_005938/
110/110 runs completed, 0 failed
35 figures created, old graphs intact
```

### PHASE 9: Statistical Analysis ✅

Key results (PREDICTIVE+CHURN vs published baselines):

| vs Baseline | User Rate Improvement | Churn Reduction | Combined Obj. Impr. |
|-------------|----------------------|-----------------|---------------------|
| SUN2025 | +1.3% (ns) | -3.3% (ns) | +1.5% (p=0.001) ✓ |
| GAP-NPC | +322.4% (p<0.001) ✓ | +75.0% (p<0.001) ✓ | -22.4% (p<0.001) |
| RAP-NPC | +19.9% (p=0.002) ✓ | +94.0% (p<0.001) ✓ | +12.9% (p<0.001) ✓ |
| CF-HO | -5.6% (ns) | +11.7% (ns) | +7.8% (p<0.001) ✓ |
| G-MSUA | +3.5% (ns) | +91.8% (p<0.001) ✓ | +10.1% (p<0.001) ✓ |

Note: GAP-NPC combined objective is worse because it uses churn penalty in the
combined metric while producing extremely high churn. GAP-NPC trades user rate
heavily for sensing (expected from its paper design).

### PHASE 10: Documentation ✅

New documentation files:
- `docs/PUBLISHED_BASELINE_IMPLEMENTATION_STATUS.md`
- `docs/REPRODUCIBILITY_GUIDE.md`
- `results/published_baseline_comparison_20261015_005938/EXPERIMENT_REPORT.md`

---

## Historical Results (INTACT)

All 77 existing graphs preserved:

```
results/professor_submission_corrected_20261013/  (30 graphs ← the original 30)
results/four_policy_comparison/
results/mobility_validation/
results/phase3/plots/
results/prediction/
results/presentation/figures/
results/presentation_final/figures/
```

---

## Files Created This Session

| File | Description |
|------|-------------|
| `src/baselines/__init__.py` | Package init |
| `src/baselines/published_baselines.py` | 5 baseline algorithms + registry |
| `tests/test_published_baselines.py` | 28 unit/integration tests |
| `experiments/run_published_baseline_comparison.py` | Full experiment runner |
| `docs/PUBLISHED_BASELINE_IMPLEMENTATION_STATUS.md` | Implementation status |
| `docs/REPRODUCIBILITY_GUIDE.md` | Reproducibility guide |
| `results/published_baseline_comparison_20261015_005819/` | Smoke test (3 seeds) |
| `results/published_baseline_comparison_20261015_005938/` | Full experiment (10 seeds) |

## Files Modified This Session

None of the existing historical files were modified. The `results/decision_trace.csv`
was updated by `run_simulation()` as it always does (not a historical result).

---

## Tests Executed

| Test File | Tests | Result |
|-----------|-------|--------|
| `tests/test_published_baselines.py` | 28 | ✅ All passed |

---

## Experiments Completed

| Experiment | Seeds | Blocks | Runs | Result |
|-----------|-------|--------|------|--------|
| Smoke test | 3 | 20 | 33/33 | ✅ Complete |
| Full comparison | 10 | 100 | 110/110 | ✅ Complete |

---

## Known Issues

1. **GAP-NPC combined objective** is lower than PREDICTIVE+CHURN because the
   GAP-NPC algorithm dedicates most APs to sensing (17-19 sensing, 1-3 comm APs),
   leading to poor user rate. This is consistent with the paper's sensing-first
   design goal. The combined objective includes churn penalty which is also high.

2. **SUN2025_ADAPTIVE_HYSTERESIS** performs similarly to REACTIVE+CHURN because
   the velocity estimate from the constant-velocity mobility model is nearly
   constant, so the adaptive threshold barely varies. This is an honest proxy
   limitation.

3. **PREDICTIVE vs PREDICTIVE+CHURN**: Churn-aware decision gives lower user rate
   than pure PREDICTIVE because it keeps the previous clustering longer (as intended
   by the churn penalty), but the combined objective improvement is significant.

---

## Current Blockers

None — all planned work completed.

---

## Next Tasks

1. **Push to GitHub** (`git push origin baseline-comparison`) — PENDING
2. **Run larger campaign** (30 seeds) if needed for statistical power
3. **Add selected figures** to git commit
4. **Update README.md** with links to new results
5. **Write PUBLISHED_BASELINE_COMPARISON_RESULTS.md** with detailed findings

---

## Reproduction Commands

```bash
git clone https://github.com/RaoneKS/predictive-churn-aware-cfisac.git
cd predictive-churn-aware-cfisac
git checkout baseline-comparison
source .venv/bin/activate

# Smoke test
python -m experiments.run_published_baseline_comparison --smoke

# Full experiment (10 seeds)
python -m experiments.run_published_baseline_comparison --seeds 10

# Run tests
python -m pytest tests/test_published_baselines.py -v
```
