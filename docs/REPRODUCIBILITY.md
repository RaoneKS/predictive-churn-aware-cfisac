# Reproducibility Guide

## Repository

Repository: `RaoneKS/predictive-churn-aware-cfisac`

Primary research branch: `pdf-compliance-extension`

## Python environment

Use the project's virtual environment when available:

```bash
cd ~/predictive-churn-aware-cfisac
source .venv/bin/activate
PYTHONPATH=. pytest -q
```

The validated regression is:

```
375 passed, 7 subtests passed
```

Focused P1 validation:

```bash
PYTHONPATH=. pytest -q tests/test_p1_solver.py
```

Expected result:

```
8 passed
```

## Validation artifact

The stored Phase-7 result is:

```
results/phase7/phase7_uncertainty_cvar_validation.json
```

It contains the scenario definition, deterministic baseline, uncertainty levels, CVaR-alpha sweep, feasibility checks, and failure list.

## Live simulator

Install the dashboard dependencies:

```bash
python -m pip install -r tools/requirements_live_simulator.txt
```

Launch:

```bash
streamlit run tools/cfisac_live_simulator.py
```

The live simulator uses the existing project physics and optimization modules. It is an observability/demo layer rather than a second implementation of the CF-ISAC equations.

## Professional dashboard

The additive control dashboard can be installed with:

```bash
cd ~/predictive-churn-aware-cfisac
bash cfisac_live_control/install_live_dashboard.sh
streamlit run tools/cfisac_live_dashboard.py
```

The dashboard documentation explicitly prohibits synthetic fallback physics.

## Recommended validation scenario

The repository's documented presentation scenario is:

- M = 20 APs
- K = 5 users
- Q = 2 sensing targets
- N = 4 antennas/AP
- area = 500
- Pmax = 1
- T_total = 30
- T_slow = 5
- mobility seed = 42
- fast seed = 7
- empirical scenarios = 100
- CVaR confidence = 0.90

For a publication, rerun the complete experiment suite in a clean environment and archive the generated raw outputs alongside the commit SHA. The current JSON is a validation record, not a substitute for an independently reproduced multi-seed study.
