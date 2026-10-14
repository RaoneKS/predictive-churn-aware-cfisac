# Professional Mobile Cell-Free ISAC Live Control Dashboard

This is a front end for the existing `predictive-churn-aware-cfisac` project.
It intentionally contains no substitute mobility, channel, clustering, beamforming,
sensing, or CVaR equations.

## Install

From the project root:

```bash
python3 -m pip install streamlit plotly pandas numpy pyyaml
```

Copy:

```text
cfisac_runtime_adapter.py   -> tools/cfisac_runtime_adapter.py
cfisac_live_dashboard.py    -> tools/cfisac_live_dashboard.py
```

Then run:

```bash
cd ~/predictive-churn-aware-cfisac
streamlit run tools/cfisac_live_dashboard.py
```

## What the dashboard executes

The trace follows the same ordering as the validated risk-aware driver:

1. `src.optimization.risk_aware.risk_aware_slow_update(...)` at every slow epoch.
2. `src.optimization.two_timescale.fast_update(...)` at every simulation step.
3. `sim.step()` between blocks.

The dashboard keeps extra copies of the actual positions, associations, predictor
trajectories and decision telemetry so they can be visualized.

## Professional, user-friendly view

The main screen is written for a mixed audience: plain-language labels explain what the controller is doing, while an expandable research-details section preserves the exact technical configuration.

The UI provides:

- AP / UE / sensing-target network map
- active communication and sensing links
- predicted user/target trajectories
- candidate associations at slow decision points
- KEEP / RECONFIGURE state
- current and predicted objective
- predicted gain, churn cost, net gain
- VaR / CVaR telemetry
- sum-rate history
- communication/sensing/total power history
- association membership snapshots
- trace CSV export
- runtime provenance including seeds and configuration

## No fallback policy

If the local project API cannot be resolved, the dashboard stops and reports the
missing constructor/factory. It does not generate synthetic physics to make the UI
look functional.

When needed, point the adapter at the project's existing factories:

```bash
export CFISAC_SIM_FACTORY='your.module:existing_simulator_factory'
export CFISAC_AP_FACTORY='your.module:existing_topology_factory'
export CFISAC_PREDICTOR_FACTORY='your.module:existing_predictor_factory'
```

The factory calls must refer to existing project components.

## Recommended presentation scenario

Use the already validated parameter family:

```text
M = 20 APs
K = 5 users
Q = 2 sensing targets
N = 4 antennas/AP
area = 500
Pmax = 1
T_total = 30
T_slow = 5
mobility seed = 42
fast seed = 7
empirical scenarios = 100
CVaR confidence = 0.90
```

Start with MRT, then switch to RZF to demonstrate that the same runtime can use
the existing precoder option.

### Local simulator compatibility

The included adapter supports the current project `MobilitySimulator` constructor, which requires explicit initial `user_positions` and `target_positions`. The adapter creates only these deterministic initial conditions from the selected mobility seed, then passes them into the existing simulator. It does not replace the project's mobility evolution, history handling, prediction, clustering, beamforming, sensing, or risk calculations.
