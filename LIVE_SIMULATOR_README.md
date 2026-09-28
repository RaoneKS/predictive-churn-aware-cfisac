# Live CF-ISAC Simulator

This simulator is an additive observability/demo layer. It does **not** replace the validated scientific modules under `src/` and it does not replay a precomputed 30-step trace.

## Run

From the project root:

```bash
python -m pip install -r tools/requirements_live_simulator.txt
streamlit run tools/cfisac_live_simulator.py
```

## Runtime path

Each `Advance 1 timestep` action executes:

```text
MobilitySimulator.step()
        ↓
real predictor (CV or saved LSTM; predictor API handles warm-up)
        ↓
cluster_for_positions / overlapping clustering
        ↓
risk_aware_slow_update at the configured slow epoch
        ↓
fast_update
        ↓
physical Rician channel + MRT/RZF beam directions
        ↓
joint communication/sensing power allocation
        ↓
SINR/rates + sensing information + feasibility
        ↓
mutable dashboard state
```

CVaR scenarios reuse the existing measured-residual machinery in `src/optimization/uncertainty.py`; no fitted synthetic error distribution is introduced.

## Interaction

- Advance exactly one timestep or five.
- Evaluate the current physical state without advancing mobility.
- Change a selected user/target position before the next timestep.
- Change scenario, predictor, mobility, beamformer, power weights, churn penalty and CVaR settings and reset.
- Inspect active communication/sensing associations and prediction trajectories.
- Export the computed runtime history as CSV.

The legacy `tools/cfisac_live_dashboard.py` is left untouched.
