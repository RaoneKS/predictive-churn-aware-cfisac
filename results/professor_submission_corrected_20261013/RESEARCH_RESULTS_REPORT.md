# Churn-Aware Optimization for Cell-Free ISAC Systems

## Simulation Results and Experimental Report

**Experiment date:** 13 October 2026
**Evaluation:** 10 random seeds (42–51), four policies, 100 time blocks per run
**Completed simulations:** 40
**Scope:** Simulation-based evaluation; no physical hardware power measurements are claimed.

## Abstract

This report evaluates STATIC, REACTIVE, PREDICTIVE, and PREDICTIVE+CHURN in a
cell-free integrated sensing and communication (CFISAC) simulation. The
corrected experiment uses the configured communication minimum-rate threshold
and tracking-error threshold when evaluating performance. The results cover
communication rate, lower-tail user rate, sensing utility, tracking and QoS
violation magnitudes, an energy-model metric, fronthaul, cumulative churn,
reconfiguration count, and the combined objective.

PREDICTIVE+CHURN has fewer reconfigurations and lower cumulative churn than
REACTIVE in these runs, while its mean communication rates are lower. The
results support a stability-versus-throughput trade-off in the tested
simulation; they do not establish universal superiority.

## 1. Problem Statement

Dynamic cell-free ISAC systems coordinate communication and sensing while
user and target associations change. Reconfiguration can incur control
overhead and instability. This experiment investigates whether churn-aware
predictive decisions can reduce reconfiguration activity while retaining
useful communication and sensing performance.

The conclusions are limited to the modeled scenarios, selected parameters,
policy implementations, and metrics in the repository.

## 2. Policies Compared

- **STATIC:** fixed associations; reference policy for dynamic control.
- **REACTIVE:** decisions respond to current conditions.
- **PREDICTIVE:** decisions incorporate prediction.
- **PREDICTIVE+CHURN:** predictive decisions with a churn-aware penalty.

These are the policy labels used by the simulator. The source code remains the
authority for the exact decision rules.

## 3. Experimental Design

The corrected experiment evaluates seeds 42 through 51 for all four policies:
40 policy-seed simulations and 4,000 block-level records. Each run contains
100 time blocks.

The configuration is `configs/default.yaml`, with configured minimum
communication rate **1000000 bps** and
tracking-error threshold **1.000**.
The configuration SHA-256 is
`53912d3ecbb4dfc2cbae5eecb85d3cadb8fd7c7168d307e0e72a798eac9ad4b5`.

The threshold-wiring correction passes the configured communication minimum
rate and tracking threshold into performance evaluation, while preserving
explicit `min_rate` and `max_tracking_error` overrides. This makes the reported
evaluation consistent with the configured thresholds. It does not by itself
prove that the policy guarantees hard feasibility for every possible scenario.

The `simulation_results/` directory contains the aggregate and per-seed
summaries, block-level data, metadata, and per-seed decision traces. Results
are simulation-based, not measurements from a physical testbed.

## 4. Principal Results

The following values are **mean ± sample standard deviation across ten seeds**.
All values are computed from the corrected aggregate summary.

| Metric | STATIC | REACTIVE | PREDICTIVE | PREDICTIVE+CHURN |
|---|---:|---:|---:|---:|
| Average user rate (Mbps) | 10.876 ± 1.973 | 26.779 ± 4.744 | 25.755 ± 4.970 | 24.478 ± 6.329 |
| 5th-percentile user rate (Mbps) | 3.297 ± 0.601 | 13.659 ± 1.836 | 12.428 ± 2.088 | 11.891 ± 2.672 |
| Sensing utility | 6.189 ± 0.342 | 6.549 ± 0.186 | 6.553 ± 0.166 | 6.551 ± 0.176 |
| Tracking violation magnitude | 0.034 ± 0.025 | 0.012 ± 0.007 | 0.013 ± 0.006 | 0.012 ± 0.005 |
| QoS violation magnitude | 275125.242 ± 150878.264 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| Energy-model metric | 46.440 ± 5.749 | 46.369 ± 2.969 | 46.373 ± 2.615 | 41.726 ± 4.005 |
| Cumulative churn | 0.000 ± 0.000 | 204.600 ± 21.603 | 218.000 ± 23.419 | 148.000 ± 29.725 |
| Reconfigurations | 0.000 ± 0.000 | 60.000 ± 6.307 | 61.300 ± 4.739 | 15.700 ± 4.347 |
| Combined objective | -275179.089 ± 150877.832 | -45.486 ± 2.151 | -46.002 ± 2.006 | -41.980 ± 2.127 |

QoS and tracking values are evaluator violation **magnitudes**, not counts of
violating users, targets, or blocks. A zero mean QoS violation magnitude for a
policy in these runs is an observed result, not a universal feasibility
guarantee. The energy value is a simulation-model metric, not instrumented
electrical power.

### 4.1 Churn and reconfiguration trade-off

Compared with REACTIVE, PREDICTIVE+CHURN has, based on the across-seed means:

- **73.8% fewer reconfigurations**
  (60.0 to 15.7).
- **27.7% lower cumulative churn**
  (204.6 to 148.0).
- **10.0% lower energy-model metric**
  (46.369 to 41.726).
- **8.6% lower average user rate**
  (26.779 to 24.478 Mbps).
- **12.9% lower 5th-percentile user rate**
  (13.659 to 11.891 Mbps).

These are descriptive comparisons of means, not evidence of statistical
significance. They do not guarantee the same improvement for every seed or
operating condition.

### 4.2 Interpretation

The churn-aware policy reduces modeled reconfiguration activity and churn in
this experiment, while retaining similar mean sensing utility. Its average and
5th-percentile communication rates are below those of REACTIVE. Its combined
objective is less negative in this implementation, but this result depends on
the objective's weights, scaling, and normalization. No single policy wins
every metric.

## 5. Figures

The `figures/` directory contains 30 figures regenerated from the corrected
CSV results. They cover policy-level comparisons, time-series behaviour,
across-seed distributions, rate/energy/churn trade-offs, violation magnitudes,
and rate variation across seeds. The figure manifest records the corrected
source-results directory and configuration hash.

## 6. Research Contribution and Novelty

The contribution evaluated here is the integration and simulation of
churn-aware decision-making into a predictive CFISAC control workflow, with
comparisons across communication, sensing, energy-model, churn, and
reconfiguration metrics.

This report does not claim that the idea is globally novel. A defensible
novelty claim requires a documented literature review and direct comparison
with relevant prior methods. The current results provide evidence for a
design trade-off in the tested setup, not proof of a new theoretical result.

## 7. Limitations and Further Work

1. The evaluation uses ten seeds; more seeds and additional scenarios would
   improve evidence about generalization.
2. The configuration does not cover every mobility pattern, network size, or
   deployment condition.
3. Energy is a simulation metric, not instrumented hardware power.
4. The combined objective depends on selected weights and normalization.
5. QoS and tracking metrics are violation magnitudes, not counts.
6. The observed results do not establish statistical significance or hard
   constraint guarantees across all scenarios.
7. Further work should include parameter sweeps, ablations, additional
   baselines, confidence intervals, and independent reproduction.

## 8. Reproducibility and Audit Trail

The package includes corrected aggregate and per-seed summaries, 4,000
block-level records, ten per-seed decision traces, experiment metadata, and
the figure manifest. The experiment used seeds 42–51 and 100 blocks per run.
The metadata records the configuration hash, configured thresholds, source
revision, and run details.

For reproduction, use the repository's documented environment, preserve the
seed list and configuration, and run the corrected multiseed experiment
script. Compare regenerated CSVs with the included outputs before drawing
conclusions.

## 9. Conclusion

Across these ten-seed simulation runs, PREDICTIVE+CHURN reduces reconfigurations
and cumulative churn relative to REACTIVE, and lowers the energy-model metric.
It also has lower average and lower-tail communication rates. The defensible
conclusion is a measurable stability-versus-throughput trade-off in this
simulation. Broader experiments and statistical analysis are needed before
generalizing these findings to other settings or real deployments.
