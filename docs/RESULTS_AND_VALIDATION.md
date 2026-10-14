# Results and Validation

## Scope

This document records the validated engineering and research results currently committed to the `pdf-compliance-extension` branch. Numerical values below are taken from the repository validation artifact `results/phase7/phase7_uncertainty_cvar_validation.json`; they are not newly fabricated or estimated.

## Validation environment

| Parameter | Value |
|---|---:|
| APs | 20 |
| Users | 5 |
| Sensing targets | 2 |
| Antennas / AP | 4 |
| Area | 500 |
| Total steps | 30 |
| Slow-timescale period | 5 |
| Maximum AP power | 1.0 |
| Mobility seed | 42 |
| Fast-update seed | 7 |
| Default uncertainty scenarios | 100 |
| CVaR confidence | 0.90 |

## Software validation

The latest completed regression reported:

- Full suite: **375 passed, 7 subtests passed**
- Focused P1 solver suite: **8 passed**
- P1 tests cover activation derivation, candidate generation, binary constraints, cluster-size constraints, fronthaul rejection, feasible/fallback behavior, deficiency-CVaR rejection, and default-mode behavior.
- GitHub branch head: `370136b`
- Branch relation to `main`: 7 commits ahead, 0 behind.

## Phase-6 deterministic baseline

The stored validation artifact reports:

- 30 fast updates and 6 slow updates.
- 3 reconfigurations and 3 KEEP decisions.
- Raw churn total: **82**.
- Mean predicted gain: **12.8219 M**.
- Mean communication sum-rate: **327.609 Mbit/s**.
- Mean sensing information gain: **15.1661**.
- All fast steps power-feasible.
- Maximum power violation: **4.44e-16** (numerical zero at solver precision).

## Phase-7 uncertainty-aware result

For the default medium-uncertainty setting (`sigma_v=3.0`, CVaR confidence 0.90):

- 30 fast updates and 6 slow updates.
- 1 reconfiguration and 5 KEEP decisions.
- Raw churn total: **42**.
- Mean communication sum-rate: **299.306 Mbit/s**.
- Mean sensing information gain: **14.9868**.
- Mean VaR: **19.4765 M**.
- Mean CVaR: **20.4955 M**.
- All fast steps power-feasible.
- Maximum power violation: **4.44e-16**.

Compared with the deterministic baseline, the risk-aware controller substantially reduces topology churn (82 to 42) while maintaining power feasibility. The stored artifact also records a lower mean communication rate, so the result should be presented as a **risk/churn trade-off**, not as an unconditional throughput improvement.

## Uncertainty sweep

| Level | sigma_v | Reconfigs | Raw churn | Mean sum-rate (Mbit/s) | Mean CVaR |
|---|---:|---:|---:|---:|---:|
| Low | 0.5 | 1 | 50 | 321.921 | 14.367 M |
| Medium | 3.0 | 1 | 42 | 299.306 | 20.495 M |
| High | 8.0 | 3 | 68 | 293.077 | 24.002 M |

The stored results show that increasing uncertainty changes both the risk measure and the controller's reconfiguration behavior. The high-uncertainty case produces more reconfiguration than the medium case.

## CVaR confidence sweep

| alpha | Reconfigs | Raw churn | Mean VaR | Mean CVaR |
|---:|---:|---:|---:|---:|
| 0.50 | 2 | 40 | 17.700 M | 20.491 M |
| 0.70 | 1 | 42 | 15.280 M | 18.754 M |
| 0.80 | 1 | 42 | 18.157 M | 19.683 M |
| 0.90 | 1 | 42 | 19.477 M | 20.495 M |
| 0.95 | 1 | 42 | 20.207 M | 21.020 M |
| 0.99 | 1 | 42 | 21.223 M | 21.223 M |

The validation artifact marks CVaR as never below VaR and reports no failures. It also records that the expected monotonicity check is mostly satisfied rather than universally monotonic, so the paper should preserve that wording.

## Engineering conclusion

The repository currently supports a validated predictive, churn-aware, two-timescale mobile CF-ISAC workflow with:

1. Mobility and prediction.
2. Overlapping communication and sensing associations.
3. Churn-aware slow-timescale reconfiguration.
4. Fast physical-layer adaptation.
5. Fronthaul and QoS hard constraints.
6. Deficiency-CVaR risk constraints.
7. Live simulator and dashboard integration.
8. Regression coverage for the P1 formulation and fallback behavior.

The numerical results support a research claim about **reducing unnecessary cluster churn under uncertainty while retaining feasibility**, with an explicit throughput/risk trade-off that should be discussed rather than hidden.
