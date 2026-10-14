# Predictive Churn-Aware Overlapping User- and Target-Centric Clustering for Mobile Cell-Free ISAC Networks

Research implementation of a mobile cell-free massive MIMO integrated sensing and communication system.

## Proposed framework

The project combines:

- Overlapping communication clustering
- User-centric communication association
- Target-centric sensing association
- Separate sensing TX/RX AP selection
- User and target mobility
- Future trajectory prediction
- Future communication/sensing channel prediction
- Churn-aware reconfiguration
- Fronthaul constraints
- Energy constraints
- Communication QoS
- Sensing information/tracking constraints
- Prediction-horizon optimization

## Algorithms

1. STATIC
2. REACTIVE
3. PREDICTIVE
4. PREDICTIVE + CHURN

## Architecture

Current state
    ↓
Mobility prediction
    ↓
Future positions
    ↓
Future channel/sensing state
    ↓
Clustering optimization
    ↓
Churn penalty
    ↓
Resource and QoS constraints
    ↓
New clusters
    ↓
Performance evaluation

## Vendor sources

The `vendor/` directory contains selected source code from existing
research repositories. It is kept separate from the original research
implementation in `src/`.

The main research contribution is the predictive, mobility-aware and
churn-aware clustering framework.

## Research documentation

The current validated results, reproducibility procedure, and paper-ready draft are maintained in:

- [Results and validation](docs/RESULTS_AND_VALIDATION.md)
- [Reproducibility guide](docs/REPRODUCIBILITY.md)
- [Research paper draft](docs/RESEARCH_PAPER_DRAFT.md)

The stored results distinguish validated evidence from future publication work. In particular, multi-seed statistical benchmarking and ablation studies are identified as the next experimental stage rather than being presented as already completed.

## External Published-Baseline Comparison

**Branch:** `baseline-comparison`

Five external published-baseline proxies have been implemented and compared against the proposed `PREDICTIVE+CHURN` method:

| Baseline | Paper | Reproduction Label |
|----------|-------|------------------|
| `SUN2025_ADAPTIVE_HYSTERESIS` | Sun et al. IEEE IoT J. 2025 | `PAPER_INSPIRED_PROXY` |
| `ELFIATOURE2023_GAP_NPC` | Elfiatoure et al. arXiv:2310.09032 | `PAPER_INSPIRED_PROXY` |
| `ELFIATOURE2023_RAP_NPC` | (same paper — RAP-NPC) | `PAPER_INSPIRED_PROXY` |
| `KIMCHO2024_CF_HO` | Kim and Cho 2024 | `PAPER_INSPIRED_PROXY` |
| `OZBEK2026_GMSUA` | Özbek and Cumalı Phys. Comm. 2026 | `PAPER_INSPIRED_PROXY` |

Documentation:

- [Implementation status](docs/PUBLISHED_BASELINE_IMPLEMENTATION_STATUS.md)
- [Comparison results](docs/PUBLISHED_BASELINE_COMPARISON_RESULTS.md)
- [Reproducibility guide](docs/REPRODUCIBILITY_GUIDE.md)
- [Progress log](docs/ANTIGRAVITY_PROGRESS.md)

To reproduce:

```bash
git checkout baseline-comparison
source .venv/bin/activate
python -m pytest tests/test_published_baselines.py -v
python -m experiments.run_published_baseline_comparison --seeds 10
```

> **Note:** All external baselines are labelled `PAPER_INSPIRED_PROXY`. Their native
> objectives are proxied by the simulator's scalar metrics; numerical values cannot be
> compared to values reported in the original papers.
