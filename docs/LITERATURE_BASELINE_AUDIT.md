# Literature Baseline Audit for CFISAC Mobility Comparison

Status: initial source-backed screening; implementation and reproduction status must be verified per method.

## Purpose

The research question is whether predictive, churn-aware AP association and sensing-role selection improves the communication/sensing/mobility trade-off against meaningful published approaches. Existing STATIC, REACTIVE, PREDICTIVE, and PREDICTIVE+CHURN results remain internal baselines; they do not satisfy the external-literature comparison requirement by themselves.

## Candidate A — Mobility management for cooperative cell-free ISAC

**Sun et al., “Mobility Management Framework for Cooperative Cell-Free ISAC Systems,” IEEE Internet of Things Journal, vol. 12, no. 23, pp. 49784–49800, 2025. DOI: 10.1109/JIOT.2025.3604762.**

- Publisher DOI: https://doi.org/10.1109/JIOT.2025.3604762
- Method described in available abstract/indexing: distributed low-complexity initial access; dynamic adaptive-hysteresis handover; kinematic-information-based dynamic clustering update.
- Relevance: very high; the paper directly studies mobility-aware C&S performance and dynamic clustering in cooperative CF-ISAC.
- Comparability limitation: the available abstract is not enough to claim a faithful reproduction. The paper's exact channel, sensing, handover-loss, clustering, and objective definitions must be extracted from the full paper before implementation. If exact reproduction is infeasible, implement only a clearly labelled hysteresis/kinematic baseline inspired by the described rule, not the paper's complete method.

## Candidate B — AP mode selection and power control for cell-free ISAC

**Elfiatoure et al., “Cell-Free Massive MIMO for ISAC: Access Point Operation Mode Selection and Power Control,” arXiv:2310.09032 (2023).**

- Paper: https://arxiv.org/abs/2310.09032
- HTML: https://arxiv.org/html/2310.09032
- Method: greedy AP mode selection with alternating optimization for communication/sensing power control; reported comparisons include greedy/random AP mode selection without power control.
- Relevance: high for AP sensing/communication role selection and C&S trade-offs.
- Comparability limitation: its sensing metric is MASR and its communication objective is max-min spectral efficiency under its own channel/power model. Do not compare its reported numerical values directly with this repository's information-gain and scalar-rate metrics. A fair local comparison needs a documented adapter and common evaluation metrics, and should preserve the method's native constraints when possible.

## Candidate C — POMDP/DRL handoff in user-centric cell-free MIMO

**Ammar et al., “Handoffs in User-Centric Cell-Free MIMO Networks: A POMDP Framework,” IEEE Transactions on Wireless Communications, 2024, 23(8), 10319–10335. DOI: 10.1109/TWC.2024.3371379.**

- DOI: https://doi.org/10.1109/TWC.2024.3371379
- Relevance: high for mobility and handoff, but it is not automatically a CF-ISAC joint sensing baseline.
- Comparability limitation: a learned/POMDP policy requires a faithful state, action, reward, and training protocol. If the original code and assumptions cannot be reproduced, label any simplified rule as a related heuristic, not as a reproduction of this method.

## Candidate D — Handover algorithms in scalable cell-free massive MIMO

**Kim and Cho, “Performance Analysis of Handover Algorithms in Scalable Cell-Free Massive MIMO Systems,” 2024.**

- Indexed publication record: https://exa.ai/library/publication/p470n3hq9mj
- Relevance: useful for mobility-aware spectral-efficiency metrics and handover/clustering baselines.
- Comparability limitation: confirm the exact published algorithm names, equations, and publication metadata from the original paper before citing numerical or algorithmic claims. The index summary alone is not a sufficient implementation specification.

## Candidate E — Joint mode selection and user association for ISAC-enabled cell-free MIMO

**Özbek and Cumalı, “Joint mode selection and user association for ISAC-enabled cell-free MIMO networks,” Physical Communication, DOI: 10.1016/j.phycom.2026.103132 (2026).**

- Publisher DOI: https://doi.org/10.1016/j.phycom.2026.103132
- Method described in available abstract: greedy mode selection and user association (G-MSUA), maximizing sensing SNR under minimum communication SINR and minimum serving-AP constraints.
- Relevance: high for joint AP mode and user association.
- Comparability limitation: its objective and constraints differ from the current churn-aware objective. Inspect the full algorithm and exact assumptions before claiming a reproduction; evaluate all methods with common output metrics and retain each method's feasibility conditions.

## Benchmark policy

1. Preserve all existing committed and uncommitted results; never delete unfavorable outcomes.
2. Use paired scenario/mobility seeds and identical simulation durations wherever the method assumptions permit it. Record any unavoidable model mismatch explicitly.
3. Report mean and fifth-percentile user rate, sensing metrics, QoS/tracking violations, energy and fronthaul (with definitions), churn/reconfiguration counts, runtime, and feasibility.
4. Keep native metrics (e.g., MASR, spectral efficiency, sensing SNR) alongside common evaluation metrics; never imply that differently defined native objectives are numerically interchangeable.
5. Distinguish faithful reproduction, compatible reimplementation, and heuristic inspired by a paper.
6. Do not generate a graph unless its underlying CSV rows come from a completed run and the method/seed/configuration metadata are recorded.
7. The target is 30 scientifically defensible comparison graphs, with reusable CSV data and explanatory captions; graph count must not be met by duplicating plots or relabelling sensitivity plots as external baselines.

## Current status

This document is a literature screening, not evidence that any external baseline has been implemented or reproduced. Before running the full 30-figure suite, implement and validate the strongest compatible algorithms in this repository's simulation, document model boundaries, and execute matched-seed experiments.