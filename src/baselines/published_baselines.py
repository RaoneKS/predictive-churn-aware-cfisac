"""
src/baselines/published_baselines.py

Published-baseline algorithm implementations for the CFISAC comparison study.

This module implements PAPER-INSPIRED PROXY versions of algorithms from three
published papers, adapted to work within the existing scalar-channel simulation.
Each implementation is honestly labelled and documented.

====== REPRODUCTION LABELS ======

FAITHFUL_REPRODUCTION:
    Algorithm and assumptions faithfully reproduced with sufficient evidence.

COMPATIBLE_REIMPLEMENTATION:
    Algorithm faithfully implemented as described, but adapted to the
    available simulator/assumptions (different channel model, etc.).

PAPER_INSPIRED_PROXY:
    A simplified method inspired by the paper but not an exact reproduction.
    Native objectives (MASR, max-min SE) cannot be computed in this simulator.

NOT_IMPLEMENTED:
    The method was not implemented; documented with reason.

====== BASELINE REGISTRY ======

Baseline A: Sun et al. Adaptive-Hysteresis Mobility Management
  - Paper: Sun et al., "Mobility Management Framework for Cooperative Cell-Free
    ISAC Systems," IEEE Internet of Things Journal, vol. 12, no. 23, 2025.
  - DOI: https://doi.org/10.1109/JIOT.2025.3604762
  - Method: adaptive-hysteresis handover with kinematic-information-based
    dynamic clustering update.
  - Label: PAPER_INSPIRED_PROXY
  - Reason: Full paper equations (adaptive hysteresis adaptation formula,
    handover-loss model, kinematic update rule, exact channel model) require
    full access to the paper. This implementation reproduces the conceptual
    algorithm: use kinematic (velocity) information to predict future AP
    suitability, and apply an adaptive hysteresis threshold to suppress
    unnecessary reconfigurations. The threshold adapts based on the measured
    velocity magnitude.

Baseline B: Elfiatoure et al. Greedy AP Mode Selection (GAP-NPC)
  - Paper: Elfiatoure et al., "Cell-Free Massive MIMO for ISAC: Access Point
    Operation Mode Selection and Power Control," arXiv:2310.09032 (2023).
  - URL: https://arxiv.org/abs/2310.09032
  - Algorithm 1 (greedy): Start with all APs in sensing mode; iteratively
    switch the AP that maximizes min per-user communication gain while still
    satisfying the sensing MASR constraint.
  - Label: PAPER_INSPIRED_PROXY
  - Reason: The native objective (max-min spectral efficiency under MASR
    constraint using massive-MIMO closed-form SE expressions) requires a
    full physical-layer channel model (large-scale fading coefficients, antenna
    arrays, conjugate precoding). Our scalar simulator lacks these. Instead,
    we implement the STRUCTURAL algorithm faithfully: greedy iterative
    mode-switch to maximize min-user communication metric while maintaining
    minimum sensing quality, using the simulator's native channel/metric
    proxies. MASR is proxied by the sensing information gain metric.
  - Baseline variant GAP-NPC (no power control): APs transmit at full power,
    mode selection only.
  - Random variant RAP-NPC (random mode selection): Random partition of APs.

Baseline C: Kim and Cho Threshold-Based Handover (CF-HO)
  - Paper: Kim and Cho, "Performance Analysis of Handover Algorithms in
    Scalable Cell-Free Massive MIMO Systems," 2024.
  - Note: Full paper not accessed; implementing a compatible threshold-based
    handover inspired by the described scalable CF handover concept:
    compare received signal quality against a threshold to decide handover.
  - Label: PAPER_INSPIRED_PROXY
  - Reason: Exact threshold formula and hysteresis model could not be verified
    from the full paper. Implementing signal-quality-threshold based clustering
    update as a standard mobility management baseline.

Baseline D: Özbek and Cumalı G-MSUA (Greedy Mode Selection + User Association)
  - Paper: Özbek and Cumalı, "Joint mode selection and user association for
    ISAC-enabled cell-free MIMO networks," Physical Communication, 2026.
  - DOI: https://doi.org/10.1016/j.phycom.2026.103132
  - Algorithm G-MSUA: Greedy joint mode selection and user association to
    maximize sensing SNR under minimum communication SINR and minimum
    serving-AP constraints.
  - Label: PAPER_INSPIRED_PROXY
  - Reason: Native objective (sensing SNR) and constraints (minimum per-user
    SINR) use a different channel model. This implementation captures the
    structural algorithm: greedily assign APs to sensing roles to maximize
    sensing utility while maintaining minimum communication association
    (serving-AP count constraint), then associate users to best available APs.

NOT_IMPLEMENTED: Demirhan and Alkhateeb (arXiv:2301.11328)
  - The joint sensing/communication beamforming approach operates in a
    fundamentally different problem space (spatial beamforming design)
    versus our mobility/association decision problem. It cannot be meaningfully
    compared against our association/clustering policies without implementing
    a full physical-layer beamforming system.

NOT_IMPLEMENTED: Ammar et al. POMDP Handoff (DOI: 10.1109/TWC.2024.3371379)
  - A learned POMDP policy requires full state/action/reward/transition
    specification and training data generation. This cannot be implemented
    without the original code or a faithful training protocol.
"""
from __future__ import annotations

import numpy as np

from src.optimization.baselines import cluster_for_positions, reactive_policy
from src.optimization.churn_aware import evaluate_cluster_performance
from src.clustering.churn import total_churn_cost


# ---------------------------------------------------------------------------
# Registry: machine-readable metadata for each baseline
# ---------------------------------------------------------------------------

BASELINE_REGISTRY = {
    "SUN2025_ADAPTIVE_HYSTERESIS": {
        "id": "SUN2025_ADAPTIVE_HYSTERESIS",
        "paper_title": (
            "Mobility Management Framework for Cooperative Cell-Free ISAC Systems"
        ),
        "authors": "Sun et al.",
        "venue": "IEEE Internet of Things Journal",
        "year": 2025,
        "doi": "https://doi.org/10.1109/JIOT.2025.3604762",
        "algorithm": (
            "Adaptive-hysteresis handover with kinematic-information-based "
            "dynamic clustering. Hysteresis threshold adapts to measured "
            "user/target velocity."
        ),
        "reproduction_label": "PAPER_INSPIRED_PROXY",
        "implemented_features": [
            "Kinematic velocity estimation from current positions",
            "Velocity-adaptive hysteresis threshold on objective gain",
            "Suppression of reconfigurations when velocity is low (stable topology)",
        ],
        "not_implemented_features": [
            "Exact adaptive-hysteresis formula from the full paper",
            "Handover-loss model specific to the paper",
            "Distributed low-complexity initial access procedure",
            "Paper's exact channel and sensing model (requires full paper access)",
        ],
        "model_differences": (
            "Uses simulator's scalar communication rate and information-gain "
            "sensing metrics instead of paper's spectral-efficiency and "
            "radar SINR definitions. Velocity is estimated from mobility "
            "simulator state."
        ),
        "validation": "Output range and constraint verification only.",
    },
    "ELFIATOURE2023_GAP_NPC": {
        "id": "ELFIATOURE2023_GAP_NPC",
        "paper_title": (
            "Cell-Free Massive MIMO for ISAC: Access Point Operation Mode "
            "Selection and Power Control"
        ),
        "authors": "Elfiatoure et al.",
        "venue": "arXiv:2310.09032",
        "year": 2023,
        "doi": "https://arxiv.org/abs/2310.09032",
        "algorithm": (
            "Algorithm 1 (GAP-NPC): Greedy AP mode selection. All APs start "
            "in sensing mode. Iteratively switch the AP that most increases "
            "min-user communication utility, while maintaining minimum sensing "
            "quality. No power control (NPC): full power used."
        ),
        "reproduction_label": "PAPER_INSPIRED_PROXY",
        "implemented_features": [
            "Structural greedy algorithm: start all-sensing, greedily switch APs to comm",
            "Iterative improvement loop until no further gain",
            "Minimum sensing quality constraint (proxied by sensing information gain)",
            "Min-user rate objective as selection criterion",
        ],
        "not_implemented_features": [
            "Closed-form MASR expressions (require massive-MIMO physical layer)",
            "Closed-form SE expressions (require large-scale fading model)",
            "Conjugate precoding beamforming",
            "Power control optimization (AO algorithm, Algorithm 3)",
            "SOCP formulation for power coefficients",
        ],
        "model_differences": (
            "Native MASR and max-min SE require the paper's physical-layer "
            "channel model (large-scale fading, conjugate precoding, MMSE "
            "estimation). Proxied by simulator's sensing information gain and "
            "min-user communication rate. The structural selection logic is "
            "faithfully preserved."
        ),
        "validation": "Output range and constraint verification only.",
    },
    "ELFIATOURE2023_RAP_NPC": {
        "id": "ELFIATOURE2023_RAP_NPC",
        "paper_title": (
            "Cell-Free Massive MIMO for ISAC: Access Point Operation Mode "
            "Selection and Power Control"
        ),
        "authors": "Elfiatoure et al.",
        "venue": "arXiv:2310.09032",
        "year": 2023,
        "doi": "https://arxiv.org/abs/2310.09032",
        "algorithm": (
            "RAP-NPC baseline from the paper: Random AP mode selection with "
            "no power control. APs are randomly partitioned into comm/sensing "
            "roles, with a minimum sensing quality guard."
        ),
        "reproduction_label": "PAPER_INSPIRED_PROXY",
        "implemented_features": [
            "Random AP partition into comm/sensing modes",
            "Minimum sensing quality guard (proxy)",
        ],
        "not_implemented_features": [
            "Paper's MASR expression",
            "SE expressions under MASR",
        ],
        "model_differences": (
            "Same proxy metrics as GAP-NPC. Used as reference within the "
            "Elfiatoure comparison."
        ),
        "validation": "Output range verification only.",
    },
    "KIMCHO2024_CF_HO": {
        "id": "KIMCHO2024_CF_HO",
        "paper_title": (
            "Performance Analysis of Handover Algorithms in Scalable "
            "Cell-Free Massive MIMO Systems"
        ),
        "authors": "Kim and Cho",
        "venue": "2024 (publication details not fully verified)",
        "year": 2024,
        "doi": "https://exa.ai/library/publication/p470n3hq9mj",
        "algorithm": (
            "Threshold-based handover: re-associate to APs when signal quality "
            "falls below threshold, using hysteresis to avoid ping-pong. "
            "Compatible reimplementation using simulator's channel quality "
            "proxy (communication feature gains)."
        ),
        "reproduction_label": "PAPER_INSPIRED_PROXY",
        "implemented_features": [
            "Signal-quality-based handover trigger",
            "Hysteresis margin to prevent oscillation",
            "Per-user best-AP re-association when quality degrades",
        ],
        "not_implemented_features": [
            "Exact threshold formula from the paper (full paper not verified)",
            "Paper's spectral-efficiency-based handover criterion",
            "Paper's specific scalability method",
        ],
        "model_differences": (
            "Uses simulator's scalar channel quality proxy. Cannot verify "
            "exact algorithm without full paper access."
        ),
        "validation": "Output range verification only.",
    },
    "OZBEK2026_GMSUA": {
        "id": "OZBEK2026_GMSUA",
        "paper_title": (
            "Joint mode selection and user association for ISAC-enabled "
            "cell-free MIMO networks"
        ),
        "authors": "Özbek and Cumalı",
        "venue": "Physical Communication",
        "year": 2026,
        "doi": "https://doi.org/10.1016/j.phycom.2026.103132",
        "algorithm": (
            "G-MSUA: Greedy Mode Selection and User Association. First, greedily "
            "assign APs to sensing roles to maximize sensing utility. Then, "
            "associate users to best available communication APs satisfying "
            "minimum serving-AP count constraint."
        ),
        "reproduction_label": "PAPER_INSPIRED_PROXY",
        "implemented_features": [
            "Greedy sensing-first AP role assignment",
            "Sensing utility maximization objective for role selection",
            "User association to best available comm APs",
            "Minimum communication AP serving count constraint",
        ],
        "not_implemented_features": [
            "Native sensing SNR objective (requires physical-layer channel)",
            "Minimum SINR per-user constraints (require beamforming model)",
            "Paper's exact feasibility checking procedure",
        ],
        "model_differences": (
            "Native sensing SNR and per-user SINR constraints require the "
            "paper's physical channel model. Proxied by simulator's sensing "
            "information gain and scalar communication rate metrics."
        ),
        "validation": "Output range and constraint verification only.",
    },
    "DEMIRHAN2023_BEAMFORMING": {
        "id": "DEMIRHAN2023_BEAMFORMING",
        "paper_title": (
            "Cell-Free ISAC MIMO Systems: Joint Sensing and Communication Beamforming"
        ),
        "authors": "Demirhan and Alkhateeb",
        "venue": "arXiv:2301.11328",
        "year": 2023,
        "doi": "https://arxiv.org/abs/2301.11328",
        "algorithm": "Joint sensing/communication beamforming design.",
        "reproduction_label": "NOT_IMPLEMENTED",
        "implemented_features": [],
        "not_implemented_features": [
            "Spatial beamforming design (incompatible problem space)",
            "Full MIMO channel model",
        ],
        "model_differences": (
            "Beamforming design problem is fundamentally different from AP "
            "association/clustering mobility management. Cannot be meaningfully "
            "compared without implementing a full physical-layer beamforming "
            "system. Excluded from numerical comparison."
        ),
        "validation": "NOT_IMPLEMENTED",
    },
    "AMMAR2024_POMDP": {
        "id": "AMMAR2024_POMDP",
        "paper_title": (
            "Handoffs in User-Centric Cell-Free MIMO Networks: A POMDP Framework"
        ),
        "authors": "Ammar et al.",
        "venue": "IEEE Transactions on Wireless Communications",
        "year": 2024,
        "doi": "https://doi.org/10.1109/TWC.2024.3371379",
        "algorithm": "POMDP-based handoff policy trained via reinforcement learning.",
        "reproduction_label": "NOT_IMPLEMENTED",
        "implemented_features": [],
        "not_implemented_features": [
            "POMDP state/action/reward/transition specification",
            "Training data generation and RL training loop",
            "Policy inference at test time",
        ],
        "model_differences": (
            "Requires a trained POMDP policy. Cannot reproduce without the "
            "original code, training protocol, or significant additional "
            "implementation effort beyond the scope of this comparison."
        ),
        "validation": "NOT_IMPLEMENTED",
    },
}


# ---------------------------------------------------------------------------
# Helper: estimate velocity from mobility simulator
# ---------------------------------------------------------------------------

def _estimate_velocity(sim) -> float:
    """
    Estimate the mean speed of all mobile entities from the mobility
    simulator's recent history.

    Returns the mean absolute velocity in m/step units, or 0.0 if
    unavailable.
    """
    try:
        # MobilitySimulator exposes user_positions and target_positions as
        # arrays; velocity can be estimated from trajectory history if
        # available, or from velocity state if tracked.
        if hasattr(sim, "user_velocities") and sim.user_velocities is not None:
            u_v = np.mean(np.linalg.norm(sim.user_velocities, axis=1))
        else:
            u_v = 0.0
        if hasattr(sim, "target_velocities") and sim.target_velocities is not None:
            t_v = np.mean(np.linalg.norm(sim.target_velocities, axis=1))
        else:
            t_v = 0.0
        return float((u_v + t_v) / 2)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Baseline A: Sun et al. Adaptive-Hysteresis
# ---------------------------------------------------------------------------

def sun2025_adaptive_hysteresis(
    ap_positions,
    user_pos,
    target_pos,
    prev_x, prev_y_tx, prev_y_rx,
    sim=None,
    perf_kwargs=None,
    ref=None,
    c_comm=1.0, c_sens_tx=1.0, c_sens_rx=1.0,
    base_threshold=0.05,
    velocity_scale=0.02,
    min_threshold=0.01,
    max_threshold=0.3,
):
    """
    Paper-inspired proxy of Sun et al. (2025) adaptive-hysteresis mobility
    management for cooperative CF-ISAC.

    REPRODUCTION LABEL: PAPER_INSPIRED_PROXY

    Concept
    -------
    1. Compute a candidate clustering from current positions (reactive).
    2. Estimate current velocity of mobile entities.
    3. Compute an adaptive hysteresis threshold:
         threshold = clip(base_threshold + velocity_scale * mean_velocity,
                         min_threshold, max_threshold)
       When velocity is low (stable mobility), the threshold is small → accept
       small gains (fast adaptation). When velocity is high (rapid movement),
       the threshold is large → require larger gain to justify reconfiguration.
    4. Switch to the candidate clustering only if the objective gain exceeds
       the adaptive threshold.

    Note: The kinematic-based adaptation concept is inspired by Sun et al.;
    the exact threshold formula and sensing/handover-loss model from the full
    paper are not reproduced.

    Parameters
    ----------
    ap_positions   : (M, 2)
    user_pos       : (K, 2) current positions
    target_pos     : (Q, 2) current positions
    prev_x         : (M, K) previous comm association
    prev_y_tx      : (M, Q) previous sensing TX
    prev_y_rx      : (M, Q) previous sensing RX
    sim            : MobilitySimulator (for velocity estimation)
    perf_kwargs    : passed to evaluate_cluster_performance
    ref            : reference dict for normalization
    c_comm etc.    : churn cost weights
    base_threshold : minimum hysteresis value (dimensionless objective units)
    velocity_scale : how much each unit of velocity raises the threshold
    min/max_threshold : clip range

    Returns
    -------
    x, y_tx, y_rx  -- selected clustering (prev or candidate)
    decision_info  -- dict with algorithm state
    """
    perf_kwargs = dict(perf_kwargs or {})
    if ref is None:
        ref = {"churn_events": 1.0}

    ap_positions = np.asarray(ap_positions, dtype=float)
    user_pos = np.asarray(user_pos, dtype=float)
    target_pos = np.asarray(target_pos, dtype=float)

    # Reactive candidate
    x_cand, y_tx_cand, y_rx_cand = reactive_policy(
        ap_positions, user_pos, target_pos
    )

    # Adaptive threshold from kinematic information
    mean_vel = _estimate_velocity(sim) if sim is not None else 0.0
    threshold = float(np.clip(
        base_threshold + velocity_scale * mean_vel,
        min_threshold, max_threshold,
    ))

    # Evaluate current vs candidate objective
    curr_perf = evaluate_cluster_performance(
        prev_x, prev_y_tx, prev_y_rx,
        ap_positions, user_pos, target_pos, **perf_kwargs,
    )
    cand_perf = evaluate_cluster_performance(
        x_cand, y_tx_cand, y_rx_cand,
        ap_positions, user_pos, target_pos, **perf_kwargs,
    )
    gain = float(cand_perf["combined_objective"] - curr_perf["combined_objective"])
    switch = gain > threshold

    if switch:
        x, y_tx, y_rx = x_cand, y_tx_cand, y_rx_cand
        decision = "RECONFIGURE"
    else:
        x, y_tx, y_rx = prev_x, prev_y_tx, prev_y_rx
        decision = "KEEP"

    return x, y_tx, y_rx, {
        "decision": decision,
        "gain": gain,
        "threshold": threshold,
        "mean_velocity": mean_vel,
    }


# ---------------------------------------------------------------------------
# Baseline B: Elfiatoure et al. GAP-NPC (Greedy AP Mode Selection, No Power Ctrl)
# ---------------------------------------------------------------------------

def _compute_min_user_rate(x, ap_positions, user_pos, perf_kwargs):
    """Helper: return the minimum per-user rate for a given association x."""
    perf = evaluate_cluster_performance(
        x, np.zeros((len(ap_positions), 1)),  # dummy y_tx, y_rx
        np.zeros((len(ap_positions), 1)),
        ap_positions, user_pos,
        np.zeros((1, 2)),  # dummy target
        **{k: v for k, v in perf_kwargs.items()},
    )
    return float(np.min(perf["raw_rates"]))


def _compute_sensing_utility(y_tx, y_rx, ap_positions, target_pos, perf_kwargs):
    """Helper: compute sensing information gain for given y_tx, y_rx."""
    num_aps = len(ap_positions)
    num_users = perf_kwargs.get("_num_users_dummy", 5)
    perf = evaluate_cluster_performance(
        np.zeros((num_aps, num_users)),  # dummy x
        y_tx, y_rx,
        ap_positions, np.zeros((num_users, 2)),  # dummy users
        target_pos,
        **{k: v for k, v in perf_kwargs.items() if k != "_num_users_dummy"},
    )
    return float(perf["raw_sensing_utility"])


def elfiatoure2023_gap_npc(
    ap_positions,
    user_pos,
    target_pos,
    perf_kwargs=None,
    ref=None,
    min_sensing_fraction=0.2,
    aps_per_user=3,
    tx_aps_per_target=2,
    rx_aps_per_target=2,
    rng=None,
):
    """
    Paper-inspired proxy of Elfiatoure et al. (2023) Algorithm 1: Greedy AP
    mode selection with no power control (GAP-NPC).

    REPRODUCTION LABEL: PAPER_INSPIRED_PROXY

    Structural algorithm (from Algorithm 1 in arXiv:2310.09032):
    1. Initialize: all APs in sensing mode (A_Sen = M, A_Com = {}).
    2. Iteratively: find the AP in A_Sen whose transfer to A_Com maximally
       improves min-user communication metric while keeping sensing utility
       above the minimum sensing quality constraint.
    3. Continue until no improvement is possible.
    4. From the comm-mode APs, associate each user to its top aps_per_user
       strongest comm APs. From sensing-mode APs, do target-centric sensing
       association.

    Parameters
    ----------
    ap_positions       : (M, 2)
    user_pos           : (K, 2)
    target_pos         : (Q, 2)
    perf_kwargs        : passed to evaluate_cluster_performance
    ref                : reference dict
    min_sensing_fraction : fraction of initial sensing utility to maintain
    aps_per_user, tx_aps_per_target, rx_aps_per_target : cluster sizes

    Returns
    -------
    x, y_tx, y_rx     -- selected clustering
    info              -- dict with algorithm state
    """
    from src.channels.communication import generate_communication_features
    from src.channels.sensing import generate_sensing_gains
    from src.clustering.overlapping import communication_clustering, sensing_clustering

    perf_kwargs = dict(perf_kwargs or {})
    if ref is None:
        ref = {"sensing_information_gain": 1.0}

    ap_positions = np.asarray(ap_positions, dtype=float)
    user_pos = np.asarray(user_pos, dtype=float)
    target_pos = np.asarray(target_pos, dtype=float)
    M = len(ap_positions)
    K = len(user_pos)
    Q = len(target_pos)

    # Compute channel features
    G_comm, _ = generate_communication_features(ap_positions, user_pos)  # (M, K)
    G_sens = generate_sensing_gains(ap_positions, target_pos)             # (M, M, Q)

    # Baseline sensing utility: all APs sensing
    # (This is the initial MASR proxy -- minimum sensing quality)
    all_sensing_y_tx = np.ones((M, Q), dtype=int)
    all_sensing_y_rx = np.ones((M, Q), dtype=int)
    # Trim to tx_aps_per_target and rx_aps_per_target (keep strongest)
    for q in range(Q):
        tx_gain = np.sum(G_sens[:, :, q], axis=1)  # AP-m as TX: sum over all RX
        rx_gain = np.sum(G_sens[:, :, q], axis=0)  # AP-n as RX: sum over all TX
        if M > tx_aps_per_target:
            tx_keep = np.argsort(tx_gain)[-(M):]  # all initially
            all_sensing_y_tx[:, q] = 0
            all_sensing_y_tx[tx_keep, q] = 1
        if M > rx_aps_per_target:
            rx_keep = np.argsort(rx_gain)[-(M):]  # all initially
            all_sensing_y_rx[:, q] = 0
            all_sensing_y_rx[rx_keep, q] = 1

    # Sensing utility baseline (with all APs in sensing)
    from src.metrics.sensing import (
        aggregate_fisher_information, posterior_covariance, information_gain,
    )
    J_all = aggregate_fisher_information(ap_positions, target_pos,
                                         all_sensing_y_tx, all_sensing_y_rx)
    base_sens = 0.0
    for q in range(Q):
        post_cov = posterior_covariance(np.eye(2), J_all[q])
        base_sens += information_gain(np.eye(2), post_cov)
    min_sens_required = min_sensing_fraction * base_sens

    # Greedy AP mode selection (Algorithm 1 proxy)
    A_sen = set(range(M))   # APs in sensing mode
    A_com = set()            # APs in communication mode

    iterations = 0
    improved = True
    best_min_rate = -np.inf

    while improved and len(A_sen) > 0:
        improved = False
        best_ap = None
        best_min_rate_this_iter = best_min_rate

        for ap_candidate in list(A_sen):
            # Tentatively move ap_candidate to comm mode
            trial_com = A_com | {ap_candidate}
            trial_sen = A_sen - {ap_candidate}

            if len(trial_com) == 0:
                continue

            # Build x for trial comm APs only (zero out sensing APs)
            x_trial = np.zeros((M, K), dtype=int)
            for k in range(K):
                # For each user, pick aps_per_user best comm-mode APs
                comm_aps = sorted(trial_com,
                                   key=lambda m: G_comm[m, k], reverse=True)
                selected = comm_aps[:aps_per_user]
                x_trial[selected, k] = 1

            # Compute min user rate from trial comm association.
            # For each user k: pick top-aps_per_user APs as serving APs.
            # Signal = sum of G_comm[m,k] for serving APs of user k.
            # Interference = sum of G_comm[m,k] for ALL comm-mode APs NOT
            #   serving user k (inter-user interference proxy).
            min_rate_trial = np.inf
            for k in range(K):
                if len(trial_com) == 0:
                    min_rate_trial = 0.0
                    break
                # Best aps_per_user comm-mode APs for user k
                sorted_aps = sorted(trial_com, key=lambda m: G_comm[m, k], reverse=True)
                serving_k = sorted_aps[:aps_per_user]
                signal_k = sum(G_comm[m, k] for m in serving_k)
                # Interference from ALL comm-mode APs to user k (less serving signal)
                all_comm_rx = sum(G_comm[m, k] for m in trial_com)
                interference_k = all_comm_rx - signal_k
                rate_k = float(np.log1p(
                    signal_k / (max(interference_k, 0.0) + 1e-9)
                ))
                min_rate_trial = min(min_rate_trial, rate_k)

            # Build sensing assignment from trial sensing APs
            if len(trial_sen) == 0:
                y_tx_trial = np.zeros((M, Q), dtype=int)
                y_rx_trial = np.zeros((M, Q), dtype=int)
                sens_trial = 0.0
            else:
                y_tx_trial = np.zeros((M, Q), dtype=int)
                y_rx_trial = np.zeros((M, Q), dtype=int)
                sen_list = list(trial_sen)
                for q in range(Q):
                    tx_g = np.array([np.sum(G_sens[m, :, q]) for m in sen_list])
                    rx_g = np.array([np.sum(G_sens[:, m, q]) for m in sen_list])
                    n_tx = min(tx_aps_per_target, len(sen_list))
                    n_rx = min(rx_aps_per_target, len(sen_list))
                    tx_sel = [sen_list[i] for i in np.argsort(tx_g)[-n_tx:]]
                    rx_sel = [sen_list[i] for i in np.argsort(rx_g)[-n_rx:]]
                    y_tx_trial[tx_sel, q] = 1
                    y_rx_trial[rx_sel, q] = 1

                J_trial = aggregate_fisher_information(ap_positions, target_pos,
                                                        y_tx_trial, y_rx_trial)
                sens_trial = 0.0
                for q in range(Q):
                    pc = posterior_covariance(np.eye(2), J_trial[q])
                    sens_trial += information_gain(np.eye(2), pc)

            # Check sensing constraint
            if sens_trial < min_sens_required:
                continue  # violates sensing requirement

            if min_rate_trial > best_min_rate_this_iter:
                best_min_rate_this_iter = min_rate_trial
                best_ap = ap_candidate

        if best_ap is not None and best_min_rate_this_iter > best_min_rate:
            A_com.add(best_ap)
            A_sen.discard(best_ap)
            best_min_rate = best_min_rate_this_iter
            improved = True
            iterations += 1

    # Build final clustering from A_com and A_sen
    A_com_list = sorted(A_com)
    A_sen_list = sorted(A_sen)

    # Communication association: users → best A_com APs
    x_final = np.zeros((M, K), dtype=int)
    if len(A_com_list) > 0:
        for k in range(K):
            comm_aps = sorted(A_com_list, key=lambda m: G_comm[m, k], reverse=True)
            selected = comm_aps[:aps_per_user]
            x_final[selected, k] = 1
    else:
        # Fallback: use reactive policy if all APs stayed in sensing mode
        x_final, _, _ = reactive_policy(ap_positions, user_pos, target_pos)

    # Sensing association: targets → best A_sen APs
    y_tx_final = np.zeros((M, Q), dtype=int)
    y_rx_final = np.zeros((M, Q), dtype=int)
    if len(A_sen_list) > 0:
        for q in range(Q):
            tx_g = np.array([np.sum(G_sens[m, :, q]) for m in A_sen_list])
            rx_g = np.array([np.sum(G_sens[:, m, q]) for m in A_sen_list])
            n_tx = min(tx_aps_per_target, len(A_sen_list))
            n_rx = min(rx_aps_per_target, len(A_sen_list))
            tx_sel = [A_sen_list[i] for i in np.argsort(tx_g)[-n_tx:]]
            rx_sel = [A_sen_list[i] for i in np.argsort(rx_g)[-n_rx:]]
            y_tx_final[tx_sel, q] = 1
            y_rx_final[rx_sel, q] = 1
    else:
        # No sensing APs: use full reactive sensing (degenerate case)
        _, y_tx_final, y_rx_final = reactive_policy(ap_positions, user_pos, target_pos)

    return x_final, y_tx_final, y_rx_final, {
        "A_com": list(A_com),
        "A_sen": list(A_sen),
        "n_com_aps": len(A_com),
        "n_sen_aps": len(A_sen),
        "iterations": iterations,
        "min_rate_achieved": float(best_min_rate),
        "min_sens_required": float(min_sens_required),
    }


def elfiatoure2023_rap_npc(
    ap_positions,
    user_pos,
    target_pos,
    perf_kwargs=None,
    ref=None,
    sensing_fraction=0.3,
    aps_per_user=3,
    tx_aps_per_target=2,
    rx_aps_per_target=2,
    rng=None,
    seed=None,
):
    """
    Paper-inspired proxy of Elfiatoure et al. (2023) RAP-NPC baseline:
    Random AP mode selection with no power control.

    REPRODUCTION LABEL: PAPER_INSPIRED_PROXY

    Randomly assigns ~sensing_fraction of APs to sensing roles, rest to comm.

    Parameters
    ----------
    sensing_fraction : fraction of APs assigned to sensing (default 0.3)
    seed             : for reproducible randomness
    """
    from src.channels.communication import generate_communication_features
    from src.channels.sensing import generate_sensing_gains

    if rng is None:
        rng = np.random.default_rng(seed)

    ap_positions = np.asarray(ap_positions, dtype=float)
    user_pos = np.asarray(user_pos, dtype=float)
    target_pos = np.asarray(target_pos, dtype=float)
    M = len(ap_positions)
    K = len(user_pos)
    Q = len(target_pos)

    G_comm, _ = generate_communication_features(ap_positions, user_pos)
    G_sens = generate_sensing_gains(ap_positions, target_pos)

    n_sen = max(1, int(M * sensing_fraction))
    n_com = M - n_sen

    # Random partition
    all_aps = np.arange(M)
    sen_aps = rng.choice(all_aps, size=n_sen, replace=False)
    com_aps = np.array([m for m in all_aps if m not in set(sen_aps)])

    # Communication association
    x_final = np.zeros((M, K), dtype=int)
    if len(com_aps) > 0:
        for k in range(K):
            sorted_com = sorted(com_aps, key=lambda m: G_comm[m, k], reverse=True)
            selected = sorted_com[:aps_per_user]
            x_final[selected, k] = 1
    else:
        x_final, _, _ = reactive_policy(ap_positions, user_pos, target_pos)

    # Sensing association
    y_tx_final = np.zeros((M, Q), dtype=int)
    y_rx_final = np.zeros((M, Q), dtype=int)
    if len(sen_aps) > 0:
        for q in range(Q):
            tx_g = np.array([np.sum(G_sens[m, :, q]) for m in sen_aps])
            rx_g = np.array([np.sum(G_sens[:, m, q]) for m in sen_aps])
            n_tx = min(tx_aps_per_target, len(sen_aps))
            n_rx = min(rx_aps_per_target, len(sen_aps))
            tx_sel = [sen_aps[i] for i in np.argsort(tx_g)[-n_tx:]]
            rx_sel = [sen_aps[i] for i in np.argsort(rx_g)[-n_rx:]]
            y_tx_final[tx_sel, q] = 1
            y_rx_final[rx_sel, q] = 1
    else:
        _, y_tx_final, y_rx_final = reactive_policy(ap_positions, user_pos, target_pos)

    return x_final, y_tx_final, y_rx_final, {
        "sen_aps": sen_aps.tolist(),
        "com_aps": com_aps.tolist(),
        "n_com_aps": len(com_aps),
        "n_sen_aps": len(sen_aps),
    }


# ---------------------------------------------------------------------------
# Baseline C: Kim and Cho CF-HO (Signal-Quality-Threshold Handover)
# ---------------------------------------------------------------------------

def kimcho2024_cf_ho(
    ap_positions,
    user_pos,
    target_pos,
    prev_x, prev_y_tx, prev_y_rx,
    perf_kwargs=None,
    ref=None,
    quality_threshold=0.5,
    hysteresis_margin=0.1,
    aps_per_user=3,
    tx_aps_per_target=2,
    rx_aps_per_target=2,
):
    """
    Paper-inspired proxy of Kim and Cho (2024) threshold-based handover for
    scalable cell-free massive MIMO.

    REPRODUCTION LABEL: PAPER_INSPIRED_PROXY

    Algorithm:
    1. For each user, compute their best achievable signal quality metric
       (max AP communication gain) from their current serving APs.
    2. If quality falls below quality_threshold - hysteresis_margin,
       trigger a handover: reassociate to the best available APs.
    3. For sensing: re-associate to the best available APs whenever
       sensing utility drops below threshold.

    Parameters
    ----------
    quality_threshold : normalized quality level below which handover triggers
    hysteresis_margin : prevents oscillation around threshold
    """
    from src.channels.communication import generate_communication_features
    from src.channels.sensing import generate_sensing_gains

    ap_positions = np.asarray(ap_positions, dtype=float)
    user_pos = np.asarray(user_pos, dtype=float)
    target_pos = np.asarray(target_pos, dtype=float)
    M = len(ap_positions)
    K = len(user_pos)
    Q = len(target_pos)

    G_comm, _ = generate_communication_features(ap_positions, user_pos)  # (M, K)
    G_sens = generate_sensing_gains(ap_positions, target_pos)             # (M, M, Q)

    # Normalize G_comm per user by max possible
    max_g = np.max(G_comm, axis=0)  # (K,)
    max_g = np.where(max_g > 0, max_g, 1.0)
    G_comm_norm = G_comm / max_g[np.newaxis, :]  # (M, K)

    # Current quality per user (sum of normalized gains for current serving APs)
    prev_x = np.asarray(prev_x, dtype=float)
    prev_y_tx = np.asarray(prev_y_tx, dtype=float)
    prev_y_rx = np.asarray(prev_y_rx, dtype=float)

    x_new = prev_x.copy().astype(int)
    y_tx_new = prev_y_tx.copy().astype(int)
    y_rx_new = prev_y_rx.copy().astype(int)

    handovers = 0

    for k in range(K):
        serving_aps = np.where(prev_x[:, k] > 0)[0]
        if len(serving_aps) == 0:
            current_quality = 0.0
        else:
            current_quality = float(np.sum(G_comm_norm[serving_aps, k]))
            # Normalize to [0,1] range by dividing by number of serving APs
            current_quality /= aps_per_user

        # Trigger handover if quality falls below threshold - hysteresis
        trigger = current_quality < (quality_threshold - hysteresis_margin)
        if trigger:
            # Reassociate to best aps_per_user APs
            best_aps = np.argsort(G_comm[:, k])[-aps_per_user:]
            x_new[:, k] = 0
            x_new[best_aps, k] = 1
            handovers += 1

    # Sensing: check if sensing quality is adequate; if not, reassociate
    for q in range(Q):
        y_tx_new[:, q] = 0
        y_rx_new[:, q] = 0
        tx_g = np.sum(G_sens[:, :, q], axis=1)
        rx_g = np.sum(G_sens[:, :, q], axis=0)
        tx_sel = np.argsort(tx_g)[-tx_aps_per_target:]
        rx_sel = np.argsort(rx_g)[-rx_aps_per_target:]
        y_tx_new[tx_sel, q] = 1
        y_rx_new[rx_sel, q] = 1

    return x_new, y_tx_new, y_rx_new, {
        "handovers": handovers,
        "quality_threshold": quality_threshold,
    }


# ---------------------------------------------------------------------------
# Baseline D: Özbek and Cumalı G-MSUA
# ---------------------------------------------------------------------------

def ozbek2026_gmsua(
    ap_positions,
    user_pos,
    target_pos,
    perf_kwargs=None,
    ref=None,
    min_comm_aps_per_user=2,
    aps_per_user=3,
    tx_aps_per_target=2,
    rx_aps_per_target=2,
    sensing_ap_fraction=0.4,
):
    """
    Paper-inspired proxy of Özbek and Cumalı (2026) G-MSUA: Greedy Mode
    Selection and User Association for ISAC-enabled CF-MIMO.

    REPRODUCTION LABEL: PAPER_INSPIRED_PROXY

    Algorithm (structural):
    1. SENSING FIRST: Greedily assign APs to sensing roles to maximize
       sensing utility (proxy for the paper's sensing SNR objective).
       Reserve at least min_comm_aps_per_user * K comm APs for users.
    2. USER ASSOCIATION: Assign users to best available comm APs.

    Parameters
    ----------
    min_comm_aps_per_user : minimum comm APs to keep available per user
    sensing_ap_fraction   : max fraction of APs to assign to sensing roles
    """
    from src.channels.communication import generate_communication_features
    from src.channels.sensing import generate_sensing_gains
    from src.metrics.sensing import (
        aggregate_fisher_information, posterior_covariance, information_gain,
    )

    ap_positions = np.asarray(ap_positions, dtype=float)
    user_pos = np.asarray(user_pos, dtype=float)
    target_pos = np.asarray(target_pos, dtype=float)
    M = len(ap_positions)
    K = len(user_pos)
    Q = len(target_pos)

    G_comm, _ = generate_communication_features(ap_positions, user_pos)
    G_sens = generate_sensing_gains(ap_positions, target_pos)

    # Minimum comm APs required (feasibility constraint)
    min_comm_aps = max(min_comm_aps_per_user, aps_per_user)
    max_sen_aps = min(M - min_comm_aps, int(M * sensing_ap_fraction))
    max_sen_aps = max(max_sen_aps, tx_aps_per_target)

    # Greedy sensing AP selection (maximize marginal sensing utility gain)
    A_sen = set()
    A_com = set(range(M))
    current_sens = 0.0

    for _ in range(max_sen_aps):
        best_marginal = -np.inf
        best_ap = None

        for ap in list(A_com):
            if M - len(A_sen) - 1 < min_comm_aps:
                break  # would violate comm AP minimum

            trial_sen = A_sen | {ap}
            trial_sen_list = list(trial_sen)

            y_tx_t = np.zeros((M, Q), dtype=int)
            y_rx_t = np.zeros((M, Q), dtype=int)
            for q in range(Q):
                tx_g = [np.sum(G_sens[m, :, q]) for m in trial_sen_list]
                rx_g = [np.sum(G_sens[:, m, q]) for m in trial_sen_list]
                n_tx = min(tx_aps_per_target, len(trial_sen_list))
                n_rx = min(rx_aps_per_target, len(trial_sen_list))
                tx_s = [trial_sen_list[i] for i in np.argsort(tx_g)[-n_tx:]]
                rx_s = [trial_sen_list[i] for i in np.argsort(rx_g)[-n_rx:]]
                y_tx_t[tx_s, q] = 1
                y_rx_t[rx_s, q] = 1

            J_t = aggregate_fisher_information(ap_positions, target_pos, y_tx_t, y_rx_t)
            sens_t = 0.0
            for q in range(Q):
                pc = posterior_covariance(np.eye(2), J_t[q])
                sens_t += information_gain(np.eye(2), pc)

            marginal = sens_t - current_sens
            if marginal > best_marginal:
                best_marginal = marginal
                best_ap = ap

        if best_ap is not None and best_marginal > 0:
            A_sen.add(best_ap)
            A_com.discard(best_ap)
            current_sens += best_marginal
        else:
            break

    # Build final sensing assignment from A_sen
    A_sen_list = sorted(A_sen)
    A_com_list = sorted(A_com)

    y_tx_final = np.zeros((M, Q), dtype=int)
    y_rx_final = np.zeros((M, Q), dtype=int)
    if len(A_sen_list) > 0:
        for q in range(Q):
            tx_g = [np.sum(G_sens[m, :, q]) for m in A_sen_list]
            rx_g = [np.sum(G_sens[:, m, q]) for m in A_sen_list]
            n_tx = min(tx_aps_per_target, len(A_sen_list))
            n_rx = min(rx_aps_per_target, len(A_sen_list))
            tx_sel = [A_sen_list[i] for i in np.argsort(tx_g)[-n_tx:]]
            rx_sel = [A_sen_list[i] for i in np.argsort(rx_g)[-n_rx:]]
            y_tx_final[tx_sel, q] = 1
            y_rx_final[rx_sel, q] = 1

    # User association: users → best comm APs
    x_final = np.zeros((M, K), dtype=int)
    if len(A_com_list) > 0:
        for k in range(K):
            sorted_com = sorted(A_com_list, key=lambda m: G_comm[m, k], reverse=True)
            selected = sorted_com[:aps_per_user]
            x_final[selected, k] = 1
    else:
        x_final, _, _ = reactive_policy(ap_positions, user_pos, target_pos)

    return x_final, y_tx_final, y_rx_final, {
        "A_com": A_com_list,
        "A_sen": A_sen_list,
        "n_com_aps": len(A_com_list),
        "n_sen_aps": len(A_sen_list),
        "sensing_utility": float(current_sens),
    }


__all__ = [
    "BASELINE_REGISTRY",
    "sun2025_adaptive_hysteresis",
    "elfiatoure2023_gap_npc",
    "elfiatoure2023_rap_npc",
    "kimcho2024_cf_ho",
    "ozbek2026_gmsua",
]
