# Professor Presentation Guide

## Figure 1 — Phase 5B Communication–Sensing Tradeoff

Information gain is:
I = ln(det(P_prior) / det(P_posterior))

Therefore it is dimensionless and conventionally reported in nats.

Key values:
- Communication-only: 380.650 Mbps, information gain 0.0000 nats
- β=0.01: 380.643 Mbps, information gain 0.5013 nats
- β=1: 376.264 Mbps, information gain 1.9824 nats
- β=100: 345.324 Mbps, information gain 2.6358 nats

What to say:
"By changing β we control how much of the shared transmit-power budget is devoted to sensing versus communication. The curve shows the resulting communication–sensing tradeoff."

Do not claim RZF universally outperforms MRT.

## Figure 2 — Phase 6 Reconfiguration and Churn

T_slow=1:
- Reconfigurations: 8
- Raw churn: 136.0

T_slow=5:
- Reconfigurations: 3
- Raw churn: 82.0

What to say:
"Separating slow association decisions from fast physical-layer updates reduces reconfiguration events and churn in this validated scenario."

## Figure 3 — Phase 6 Communication Performance

T_slow=1:
- Mean communication sum rate: 323.642 Mbps
- Mean sensing information gain: 15.4319 nats

T_slow=5:
- Mean communication sum rate: 327.609 Mbps
- Mean sensing information gain: 15.1661 nats

Say "comparable/no loss in communication in this scenario", not "universally better".

## Figure 4 — Phase 7 Uncertainty versus CVaR

CVaR is reported in combined-objective units because the provided validation JSON does not define an SI physical unit for that objective.

CVaR:
- Low uncertainty: 14.367 million
- Medium uncertainty: 20.495 million
- High uncertainty: 24.002 million

What to say:
"As prediction uncertainty increases, empirical tail risk rises. This motivates incorporating uncertainty directly into the slow-timescale optimization."

Do not claim deterministic CVaR is lower/higher because no deterministic CVaR reference is reported.

## Figure 5 — Phase 7 Risk-Aware versus Deterministic

Medium uncertainty:

Deterministic:
- Reconfigurations: 3
- Raw churn: 82.0
- Mean communication rate: 327.609 Mbps

Risk-aware:
- Reconfigurations: 1
- Raw churn: 42.0
- Mean communication rate: 299.306 Mbps

Interpret as an operating-point tradeoff, not a universal winner.

## Figure 6 — VaR and CVaR versus α

Use this as a supporting/backup figure.

Important:
- CVaR is at or above VaR for the reported points.
- The α=0.5 to α=0.7 change coincides with a different selected decision, so do not interpret the dip as a fixed-decision property of CVaR.

## Main presentation order

1. Phase 5B tradeoff
2. Phase 6 reconfiguration/churn
3. Phase 7 uncertainty/CVaR
4. Phase 7 risk-aware operating point
5. Alpha sweep as backup

## Important scope

These are single-scenario validation results. Avoid universal claims.
