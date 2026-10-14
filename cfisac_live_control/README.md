# CF-ISAC Professional Live Control Dashboard v3

This package is an additive visualization layer for the existing Mobile Cell-Free ISAC project.
It **does not implement alternate mobility, channel, clustering, beamforming, sensing, or risk physics**.

## Design references used

The interface direction was informed by:

- Plotly Dash Manufacturing SPC Dashboard — compact quick-stat cards, operator controls, monitoring/trend panels.
  https://github.com/plotly/dash-sample-apps/tree/main/apps/dash-manufacture-spc-dashboard
- Streamlit layout guidance — persistent sidebar controls, columns, tabs, expanders, and containers.
  https://docs.streamlit.io/develop/concepts/design/layouts-and-containers
- Streamlit D3 Network — topology visualization with explicit nodes/links and interactive network exploration.
  https://github.com/QuentinFuxa/streamlit-d3-network
- Plotly Science & Engineering examples — engineering/scientific dashboard composition.
  https://plotly.com/examples/science-engineering/

The implementation is intentionally adapted to the CF-ISAC project's own runtime and terminology rather than copying another application's logic.

## Install

From the repository root:

```bash
cp cfisac_live_control/cfisac_live_dashboard.py tools/cfisac_live_dashboard.py
cp cfisac_live_control/cfisac_runtime_adapter.py tools/cfisac_runtime_adapter.py
streamlit run tools/cfisac_live_dashboard.py
```

## Runtime contract

The default adapter constructs the project's real `src.simulation.mobility_simulator.MobilitySimulator` and supplies its required initial user/target position arrays. All subsequent mobility evolution, history, prediction input, clustering, two-timescale optimization, physical-layer calculations, and risk-aware logic remain in the project modules.

Factory overrides remain supported:

```bash
export CFISAC_SIM_FACTORY=module:function
export CFISAC_AP_FACTORY=module:function
export CFISAC_PREDICTOR_FACTORY=module:function
```

## Dashboard structure

- **Network overview** — current topology, active/candidate links, predicted trajectories, and plain-language controller explanation.
- **Performance** — throughput, power allocation, cluster-change benefit, and VaR/CVaR trend.
- **Associations** — current AP memberships plus a step-by-step controller log.
- **Research details** — exact runtime configuration and provenance.

No fallback physics is included.
