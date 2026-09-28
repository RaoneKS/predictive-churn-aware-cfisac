"""
Deficiency-based Conditional Value-at-Risk (CVaR).

This module implements the exact deficiency definitions from the PDF:
    L_k^C = max(R_k^min - R_k, 0)
    L_q^S = max(trace(P_q^+) - epsilon_q^trk, 0)

COMPLIANCE NOTE:
Deficiency CVaR is currently implemented as an EVALUATION/REPORTING ONLY
metric. It is NOT enforced as a hard optimization constraint in the
current two-timescale predictive simulator. The simulator optimizes the
gain-based Phase-7 risk-aware objective instead.
"""

import numpy as np
from typing import Dict, List, Optional
from src.optimization.cvar import evaluate_cvar, CVaRResult

def communication_deficiency_samples(
    rates_scenarios: np.ndarray,
    min_rate_bps: float
) -> np.ndarray:
    """
    Computes per-user communication deficiency samples.

    Parameters
    ----------
    rates_scenarios : ndarray (N_scenarios, K)
        The simulated communication rates (in bps) for each user across scenarios.
    min_rate_bps : float
        The minimum target rate.

    Returns
    -------
    ndarray (N_scenarios, K)
        L_k^C = max(R_k^min - R_k, 0)
    """
    return np.maximum(min_rate_bps - rates_scenarios, 0.0)

def sensing_deficiency_samples(
    traces_scenarios: np.ndarray,
    epsilon_trk: np.ndarray
) -> np.ndarray:
    """
    Computes per-target sensing deficiency samples.

    Parameters
    ----------
    traces_scenarios : ndarray (N_scenarios, Q)
        The trace of the posterior covariance matrix (tr(P_q^+)) across scenarios.
    epsilon_trk : ndarray (Q,)
        The maximum allowable tracking error variance for each target.

    Returns
    -------
    ndarray (N_scenarios, Q)
        L_q^S = max(tr(P_q^+) - epsilon_q^trk, 0)
    """
    return np.maximum(traces_scenarios - epsilon_trk, 0.0)

def evaluate_deficiency_cvar(
    deficiency_samples: np.ndarray,
    alpha: float
) -> List[CVaRResult]:
    """
    Computes VaR and CVaR for each entity (user or target) given their deficiency samples.

    Parameters
    ----------
    deficiency_samples : ndarray (N_scenarios, M_entities)
        The deficiency samples (either communication or sensing).
    alpha : float
        Confidence level for CVaR (0 < alpha < 1).

    Returns
    -------
    list of CVaRResult
        A list of length M_entities containing the VaR/CVaR results for each entity.
    """
    results = []
    num_entities = deficiency_samples.shape[1]
    for i in range(num_entities):
        losses = deficiency_samples[:, i]
        results.append(evaluate_cvar(losses, alpha))
    return results

def aggregate_deficiency_results(
    rates_scenarios: np.ndarray,
    min_rate_bps: float,
    traces_scenarios: np.ndarray,
    epsilon_trk: np.ndarray,
    alpha_comm: float = 0.9,
    alpha_sens: float = 0.9
) -> Dict[str, List[CVaRResult]]:
    """
    Aggregates communication and sensing deficiency CVaR results.

    Returns
    -------
    dict
        Contains 'communication' and 'sensing' keys mapping to lists of CVaRResult.
    """
    comm_def = communication_deficiency_samples(rates_scenarios, min_rate_bps)
    sens_def = sensing_deficiency_samples(traces_scenarios, epsilon_trk)

    return {
        "communication": evaluate_deficiency_cvar(comm_def, alpha_comm),
        "sensing": evaluate_deficiency_cvar(sens_def, alpha_sens)
    }

def evaluate_deficiency_constraints(
    rates_scenarios: np.ndarray,
    traces_scenarios: np.ndarray,
    min_rate_bps: float,
    epsilon_trk: np.ndarray,
    max_comm_cvar: Optional[float] = None,
    max_sensing_cvar: Optional[float] = None,
    alpha_comm: float = 0.9,
    alpha_sens: float = 0.9
) -> Dict[str, Any]:
    """
    Evaluates deficiency CVaR limits for communication and sensing.

    Parameters
    ----------
    ... (same parameters plus max constraints)

    Returns
    -------
    dict
        Contains constraint status, margins, maximum CVaR values, etc.
    """
    aggregated = aggregate_deficiency_results(
        rates_scenarios, min_rate_bps,
        traces_scenarios, epsilon_trk,
        alpha_comm, alpha_sens
    )

    comm_cvars = [res.cvar for res in aggregated["communication"]]
    sens_cvars = [res.cvar for res in aggregated["sensing"]]

    max_c_cvar = float(np.max(comm_cvars)) if comm_cvars else 0.0
    max_s_cvar = float(np.max(sens_cvars)) if sens_cvars else 0.0

    comm_satisfied = True
    comm_violation = 0.0
    if max_comm_cvar is not None:
        comm_violation = max(0.0, max_c_cvar - max_comm_cvar)
        comm_satisfied = bool(max_c_cvar <= max_comm_cvar + 1e-8)

    sens_satisfied = True
    sens_violation = 0.0
    if max_sensing_cvar is not None:
        sens_violation = max(0.0, max_s_cvar - max_sensing_cvar)
        sens_satisfied = bool(max_s_cvar <= max_sensing_cvar + 1e-15)

    return {
        "communication": aggregated["communication"],
        "sensing": aggregated["sensing"],
        "max_comm_cvar": max_c_cvar,
        "max_sensing_cvar": max_s_cvar,
        "comm_satisfied": comm_satisfied,
        "sensing_satisfied": sens_satisfied,
        "comm_violation": comm_violation,
        "sensing_violation": sens_violation,
        "satisfied": comm_satisfied and sens_satisfied,
        "violation_amount": max(comm_violation, sens_violation)
    }
