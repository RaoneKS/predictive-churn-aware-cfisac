# Predictive Churn-Aware Overlapping User- and Target-Centric Clustering for Mobile Cell-Free ISAC Networks

## Abstract

Mobile cell-free integrated sensing and communication (ISAC) networks must continuously adapt access-point associations as users and sensing targets move. Frequent cluster changes, however, create signaling and fronthaul overhead and can make mobility-aware optimization unstable. This work develops a predictive churn-aware clustering framework for mobile cell-free ISAC networks. The framework jointly considers user-centric communication association, target-centric sensing transmitter/receiver association, mobility prediction, future channel/sensing state, fronthaul limits, communication quality-of-service, sensing requirements, energy constraints, and a churn penalty. A two-timescale controller performs slow topology decisions and fast physical-layer adaptation. An uncertainty-aware extension evaluates candidate configurations with deficiency-CVaR constraints derived from the project's measured-residual scenario machinery.

For the validated 20-AP, 5-user, 2-target scenario, the stored deterministic baseline produced 82 total churn events over 30 fast updates, while the default medium-uncertainty risk-aware controller produced 42. Both configurations remained power-feasible at every fast step. The uncertainty sweep demonstrates a measurable trade-off between risk, throughput, and reconfiguration behavior. These results indicate that predictive risk-aware clustering can reduce unnecessary topology changes while preserving hard feasibility, although the current results should be extended with multi-seed statistical evaluation before making broad generalization claims.

**Keywords:** cell-free massive MIMO, ISAC, mobility prediction, clustering, churn, CVaR, risk-aware optimization, two-timescale control.

## 1. Introduction

Cell-free massive MIMO removes the conventional cell boundary by allowing geographically distributed access points (APs) to cooperate. When sensing is integrated with communication, the association problem becomes more complex because APs may simultaneously participate in communication and sensing transmitter/receiver roles. Mobility further changes the useful topology over time.

A purely reactive controller responds after the channel or position has changed. Such a controller can unnecessarily reconfigure clusters, increasing control overhead and fronthaul activity. The research problem addressed here is therefore:

> How can a mobile cell-free ISAC controller exploit predicted future states and explicitly price topology churn while remaining robust to prediction uncertainty?

The proposed answer is a predictive, overlapping user- and target-centric controller with separate sensing TX/RX associations and two-timescale optimization.

## 2. System and Optimization Framework

The implementation represents three binary association structures:

- (x_{m,k}): AP (m) serves communication user (k).
- (y^{tx}_{m,q}): AP (m) participates as a sensing transmitter for target (q).
- (y^{rx}_{m,q}): AP (m) participates as a sensing receiver for target (q).

The AP activation variable is derived from the union of active communication and sensing associations:

[
a_m = mathbb{1}left(
sum_k x_{m,k}
+sum_q y^{tx}_{m,q}
+sum_q y^{rx}_{m,q} > 0
ight).
]

The controller evaluates a candidate topology against hard constraints including binary association structure, minimum cluster sizes, fronthaul capacity, communication QoS, and uncertainty-aware deficiency-CVaR limits where enabled.

A generic churn term is represented as the difference between the previous and candidate association states:

[
C_{mathrm{churn}} =
C_x(x^{prev},x)
+C_{tx}(y^{tx,prev},y^{tx})
+C_{rx}(y^{rx,prev},y^{rx}).
]

The slow-timescale decision balances predicted system benefit against the cost of changing topology. The fast-timescale stage then adapts the physical-layer allocation and beamforming for the selected topology.

For uncertainty-aware operation, candidate performance is evaluated across empirical scenarios generated from the project's existing residual machinery. VaR and CVaR are used to quantify tail risk, while hard deficiency-CVaR constraints can reject candidates that violate configured communication/sensing reliability requirements.

## 3. Predictive Two-Timescale Control

At a slow epoch, the mobility predictor produces future user and target trajectories. Candidate overlapping cluster configurations are generated and filtered through hard constraints. A risk-aware controller then selects KEEP or RECONFIGURE.

Between slow epochs, the selected topology is retained while the fast controller adapts power allocation and physical-layer variables to the current state.

This separation is important: mobility-driven topology decisions occur less frequently than physical-layer adaptation, reducing unnecessary reconfiguration.

## 4. Implementation and Validation

The repository implements:

- mobility simulation and prediction;
- overlapping communication clustering;
- separate sensing TX/RX association;
- churn-aware reconfiguration;
- fronthaul and QoS constraints;
- deficiency-CVaR constraints;
- two-timescale optimization;
- physical Rician-channel and beamforming evaluation;
- live simulation and visualization;
- regression tests for P1 candidate generation and constraint handling.

The latest software regression contains 375 passing tests and 7 passing subtests. The focused P1 regression contains 8 passing tests.

## 5. Experimental Configuration

The stored validation scenario contains 20 APs, 5 users, 2 sensing targets, 4 antennas per AP, an area size of 500, 30 fast simulation steps, and a slow decision period of 5. The maximum AP power is 1.0. The mobility seed is 42 and the fast-update seed is 7. The default uncertainty-aware configuration uses 100 empirical scenarios and CVaR confidence 0.90.

## 6. Results

### 6.1 Deterministic baseline

The stored Phase-7 validation artifact reports 3 reconfigurations, 3 KEEP decisions, and 82 total churn events for the deterministic baseline. Mean communication sum-rate is 327.609 Mbit/s, mean sensing information gain is 15.1661, and every fast step satisfies the power constraint.

### 6.2 Medium-uncertainty risk-aware controller

At uncertainty level (sigma_v=3.0), the risk-aware controller produces 1 reconfiguration, 5 KEEP decisions, and 42 total churn events. Mean communication sum-rate is 299.306 Mbit/s and mean sensing information gain is 14.9868. Mean VaR is 19.4765 M and mean CVaR is 20.4955 M. All fast steps remain power-feasible.

The comparison shows a clear reduction in topology churn, but also a lower communication sum-rate. Therefore the correct interpretation is a risk/churn trade-off rather than an unconditional throughput gain.

### 6.3 Uncertainty sweep

The low-, medium-, and high-uncertainty cases have total churn values of 50, 42, and 68 respectively. Mean sum-rate decreases from 321.921 Mbit/s at low uncertainty to 293.077 Mbit/s at high uncertainty, while mean CVaR rises from 14.367 M to 24.002 M.

### 6.4 CVaR confidence sweep

Across the stored alpha sweep from 0.50 to 0.99, the artifact reports CVaR values from 20.491 M to 21.223 M. The repository validation marks the CVaR-never-below-VaR check as passing and records no experiment failures. The alpha-monotonicity diagnostic is reported as mostly satisfied, so this observation should not be overstated.

## 7. Discussion

The most direct evidence for the proposed mechanism is the reduction in topology churn from 82 to 42 in the deterministic-versus-medium-uncertainty comparison. This is consistent with the intended purpose of explicitly accounting for reconfiguration cost.

The price is a reduction in mean communication sum-rate from 327.609 to 299.306 Mbit/s in the stored comparison. This demonstrates why churn should be treated as a multi-objective systems consideration rather than optimized in isolation.

The uncertainty sweep also shows that prediction uncertainty affects controller behavior. At high uncertainty, the controller produces more reconfigurations and lower mean sum-rate than the medium-uncertainty case. This supports the need for risk-aware decision rules rather than relying only on point predictions.

## 8. Limitations and Future Work

The current validation is a strong software-engineering and algorithmic validation, but it is not yet sufficient for a broad empirical publication claim. The following should be added for a final paper submission:

1. independent multi-seed experiments;
2. confidence intervals or standard deviations;
3. comparisons against all four repository modes: STATIC, REACTIVE, PREDICTIVE, and PREDICTIVE+CHURN;
4. explicit fronthaul-overhead measurements;
5. prediction-error sensitivity using independently generated trajectories;
6. runtime/complexity profiling;
7. ablation studies for churn penalty and CVaR constraints;
8. publication-quality figures generated directly from archived raw experiment outputs.

## 9. Conclusion

This work implements a predictive churn-aware clustering architecture for mobile cell-free ISAC. The validated results show that explicitly accounting for topology churn can substantially reduce reconfiguration events while retaining hard power feasibility. The uncertainty-aware extension exposes a meaningful risk-throughput trade-off. The current repository therefore provides a reproducible research implementation and a validated foundation for the next experimental stage: multi-seed statistical evaluation and publication-quality benchmarking.
