# Published Baseline Implementation Status

**Branch:** baseline-comparison
**Last updated:** 2026-10-15

## Overview

This document tracks the implementation status of each published-baseline method
for the CFISAC mobility management comparison study.

## Reproduction Label Definitions

| Label | Meaning |
|-------|---------|
| `FAITHFUL_REPRODUCTION` | Algorithm and assumptions reproduced with sufficient evidence |
| `COMPATIBLE_REIMPLEMENTATION` | Faithfully implemented but adapted to available simulator |
| `PAPER_INSPIRED_PROXY` | Simplified method inspired by paper; not exact reproduction |
| `NOT_IMPLEMENTED` | Not implemented; see reason |

## Implementation Status Table

| ID | Paper | Venue / Year | DOI | Label | Status |
|----|-------|-------------|-----|-------|--------|
| `SUN2025_ADAPTIVE_HYSTERESIS` | Sun et al., "Mobility Management Framework for Cooperative Cell-Free ISAC Systems" | IEEE IoT J., 2025 | [10.1109/JIOT.2025.3604762](https://doi.org/10.1109/JIOT.2025.3604762) | `PAPER_INSPIRED_PROXY` | ✅ Implemented |
| `ELFIATOURE2023_GAP_NPC` | Elfiatoure et al., "Cell-Free Massive MIMO for ISAC: AP Operation Mode Selection and Power Control" | arXiv:2310.09032, 2023 | [arxiv.org/abs/2310.09032](https://arxiv.org/abs/2310.09032) | `PAPER_INSPIRED_PROXY` | ✅ Implemented |
| `ELFIATOURE2023_RAP_NPC` | (same paper — RAP-NPC baseline) | arXiv:2310.09032, 2023 | [arxiv.org/abs/2310.09032](https://arxiv.org/abs/2310.09032) | `PAPER_INSPIRED_PROXY` | ✅ Implemented |
| `KIMCHO2024_CF_HO` | Kim and Cho, "Performance Analysis of Handover Algorithms in Scalable Cell-Free Massive MIMO Systems" | 2024 | [exa.ai link](https://exa.ai/library/publication/p470n3hq9mj) | `PAPER_INSPIRED_PROXY` | ✅ Implemented |
| `OZBEK2026_GMSUA` | Özbek and Cumalı, "Joint mode selection and user association for ISAC-enabled cell-free MIMO networks" | Physical Comm., 2026 | [10.1016/j.phycom.2026.103132](https://doi.org/10.1016/j.phycom.2026.103132) | `PAPER_INSPIRED_PROXY` | ✅ Implemented |
| `DEMIRHAN2023_BEAMFORMING` | Demirhan and Alkhateeb, "Cell-Free ISAC MIMO Systems: Joint Sensing and Communication Beamforming" | arXiv:2301.11328, 2023 | [arxiv.org/abs/2301.11328](https://arxiv.org/abs/2301.11328) | `NOT_IMPLEMENTED` | ❌ Out of scope |
| `AMMAR2024_POMDP` | Ammar et al., "Handoffs in User-Centric Cell-Free MIMO Networks: A POMDP Framework" | IEEE TWC, 2024 | [10.1109/TWC.2024.3371379](https://doi.org/10.1109/TWC.2024.3371379) | `NOT_IMPLEMENTED` | ❌ Requires training |

## Implementation Details

### SUN2025_ADAPTIVE_HYSTERESIS

**Algorithm:** Velocity-adaptive hysteresis mobility management.

**Implemented features:**
- Kinematic velocity estimation from mobility simulator state
- Velocity-adaptive hysteresis threshold on objective gain
- Suppression of reconfigurations when velocity is low (stable topology)
- Hysteresis threshold adapts: `clip(base + velocity_scale * mean_vel, min, max)`

**Not implemented:**
- Exact adaptive-hysteresis formula from the full paper (requires full-text access)
- Handover-loss model specific to the paper
- Distributed low-complexity initial access procedure
- Paper's exact channel and radar SINR model

**Model differences:**
Uses simulator's scalar communication rate and information-gain sensing metrics
instead of paper's spectral-efficiency and radar SINR definitions.

---

### ELFIATOURE2023_GAP_NPC

**Algorithm:** Greedy AP Mode Selection, No Power Control (Algorithm 1 from paper).

**Structural algorithm faithfully implemented:**
1. All APs start in sensing mode
2. Iteratively switch the AP that most increases min-user communication metric,
   while keeping sensing utility ≥ minimum sensing quality constraint
3. Continue until no improvement
4. Associate users to best comm-mode APs; targets to best sensing-mode APs

**Not implemented:**
- Closed-form MASR expressions (require massive-MIMO physical layer model)
- Closed-form SE with conjugate precoding
- Power control optimization (AO algorithm, Algorithm 3)
- SOCP formulation for power coefficients

**Model differences:**
Native MASR and max-min SE use large-scale fading model. Proxied by simulator's
sensing information gain and min-user communication rate. Structural selection logic preserved.

---

### ELFIATOURE2023_RAP_NPC

**Algorithm:** Random AP Mode Selection, No Power Control (benchmark from paper).

**Implemented:** Random fraction of APs assigned to sensing, remainder to comm.
User and target association follows same per-role optimal selection as GAP-NPC.

---

### KIMCHO2024_CF_HO

**Algorithm:** Signal-quality-threshold based handover for scalable CF networks.

**Implemented features:**
- Signal-quality-based handover trigger per user
- Hysteresis margin to prevent oscillation
- Per-user reassociation when quality degrades

**Not implemented:**
- Exact threshold formula (full paper not accessed)
- Paper's spectral-efficiency-based criterion and scalability method

---

### OZBEK2026_GMSUA

**Algorithm:** Greedy Mode Selection and User Association (G-MSUA).

**Implemented structural algorithm:**
1. Greedily assign APs to sensing roles (maximize sensing utility)
2. Maintain minimum comm AP count constraint
3. Associate users to best available comm APs

**Not implemented:**
- Native sensing SNR (requires physical channel model)
- Minimum per-user SINR constraint (requires beamforming model)

---

### DEMIRHAN2023_BEAMFORMING — NOT_IMPLEMENTED

**Reason:** Joint sensing/communication beamforming design operates in a
fundamentally different problem space (spatial beamforming) versus our
mobility/association decision problem. Cannot be meaningfully compared
without a full physical-layer beamforming system.

---

### AMMAR2024_POMDP — NOT_IMPLEMENTED

**Reason:** A learned POMDP policy requires state/action/reward/transition
specification and training data generation. Cannot be reproduced without
the original code, training protocol, or significant additional effort.

## Source Files

- Implementation: [`src/baselines/published_baselines.py`](../src/baselines/published_baselines.py)
- Tests: [`tests/test_published_baselines.py`](../tests/test_published_baselines.py)
- Experiment runner: [`experiments/run_published_baseline_comparison.py`](../experiments/run_published_baseline_comparison.py)

## Test Results

All 28 baseline tests pass:

```
tests/test_published_baselines.py::TestBaselineRegistry::test_registry_keys PASSED
tests/test_published_baselines.py::TestBaselineRegistry::test_registry_reproduction_labels PASSED
... (28 total)
============================== 28 passed in 0.57s ==============================
```
