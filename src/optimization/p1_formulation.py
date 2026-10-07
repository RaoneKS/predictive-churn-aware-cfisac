"""
Formal PDF P1 Formulation and Interface.

This module provides the formal mathematical representation of the mixed-integer
predictive optimization problem (P1) defined in the PDF.

===========================================================================
1. Formal Mathematical Formulation (P1)
===========================================================================
Maximize over {x, y^T, y^R, a, W, S}:
    E [ sum_t ( sum_k alpha_k U_k(R_k(t)) + sum_q beta_q I_q(t)
                - lambda_E sum_m P_m^tot(t) - lambda_F sum_m F_m(t)
                - lambda_H H(t) ) ]

Subject to:
    C1: Pr{R_k(t) >= R_k^min} >= 1 - epsilon_k                 (Comm QoS)
    C2: Pr{tr[P_q^+(t)] <= epsilon_q^trk} >= 1 - delta_q       (Sensing Tracking)
    C3: F_m(t) <= C_m^FH                                       (Fronthaul Capacity)
    C4: sum_k ||w_{mk}(t)||^2 + tr[S_m(t)] <= a_m(t) P_m^max   (AP Power Budget)
    C5: ||w_{mk}(t)||^2 <= x_{mk}(t) P_m^max                   (Comm Association)
    C6: x_{mk}(t) <= a_m(t), y_{mq}^T(t) <= a_m(t), y_{mq}^R(t) <= a_m(t) (AP Activation)
    C7: 1 <= sum_m x_{mk}(t) <= B_k^max                        (Comm Cluster Size)
    C8: sum_m y_{mq}^T(t) >= N_q^{T,min}                       (Min Sensing TX APs)
    C9: sum_m y_{mq}^R(t) >= N_q^{R,min}                       (Min Sensing RX APs)
    C10: x_{mk}, y_{mq}^T, y_{mq}^R, a_m in {0, 1}             (Binary Variables)
    C11: S_m(t) >= 0                                           (PSD Covariance)

===========================================================================
2. Implemented Operational Subproblems
===========================================================================
The full mixed-integer P1 is extremely difficult to solve exactly due to
the time-coupling (H(t) churn), chance constraints, and binary variables.
We DO NOT claim to solve the full P1 exactly.

Instead, the architecture decomposes P1 into a two-timescale heuristic
(implemented in `src/optimization/two_timescale.py`):
    - SLOW timescale: Determines topology orchestrations {x, y^T, y^R, a}.
    - FAST timescale: Solves for {W, S} via deterministic continuous power
      allocations (`src/optimization/joint_comm_sensing.py`), taking topology
      as fixed.

===========================================================================
3. Enforcement vs Evaluation/Reporting
===========================================================================
IMPLEMENTED DIRECTLY / P1 FORMULATION TRACEABILITY:
    - x (Comm Association) -> Evaluated and optimized via `src/optimization/p1_solver.py`
    - y_tx (Sensing TX Association) -> Evaluated and optimized via `src/optimization/p1_solver.py`
    - y_rx (Sensing RX Association) -> Evaluated and optimized via `src/optimization/p1_solver.py`
    - a (AP Activation) -> Derived explicitly in `ap_activation()` inside `p1_solver.py`
    - W (Comm Beamforming) -> Optimized via `src/optimization/joint_comm_sensing.py`
    - S (Sensing Covariance) -> Optimized via `src/optimization/joint_comm_sensing.py` parameterization.
      NOTE: The current implementation uses the established scalar sensing-power/fixed-basis parameterization
      and does NOT optimize a completely unrestricted matrix-valued S_m.
    - C1 (Comm QoS) -> Enforced via hard QoS mode inside `joint_comm_sensing.py`
    - C2 (Sensing Tracking) -> Enforced via tracking constraint / deficiency CVaR in `p1_solver.py` and `joint_comm_sensing.py`
    - C3 (Fronthaul Capacity) -> Enforced via fronthaul feasibility filter in `p1_solver.py`
    - C4 (AP Power Budget) -> Enforced by the joint optimizer in `joint_comm_sensing.py`
    - C5 (Comm Association) -> Enforced via W_dir / topology filter in `p1_solver.py`
    - C6 (Activation Consistency) -> Enforced via activation consistency filter in `p1_solver.py`
    - C7 (Comm Cluster Size) -> Enforced via candidate filter in `p1_solver.py`
    - C8 (Min Sensing TX APs) -> Enforced via candidate filter in `p1_solver.py`
    - C9 (Min Sensing RX APs) -> Enforced via candidate filter in `p1_solver.py`
    - C10 (Binary Variables) -> Enforced via validator filter in `p1_solver.py`
    - C11 (PSD Covariance) -> Satisfied by construction via the existing sensing covariance parameterization.

"""

from typing import Dict, Any
import numpy as np

def validate_p1_constraints(
    x: np.ndarray,
    y_tx: np.ndarray,
    y_rx: np.ndarray,
    a: np.ndarray,
    P_comm: np.ndarray,
    P_sens: np.ndarray,
    P_max: np.ndarray,
    F_m: np.ndarray,
    C_m_FH: np.ndarray
) -> Dict[str, Any]:
    """
    Validates a subset of deterministic P1 constraints for a given time step.

    Returns a dictionary of constraint satisfactions (True if satisfied).
    """
    tol = 1e-8

    # C4: AP Power Budget
    c4 = bool(np.all(P_comm + P_sens <= a * P_max + tol))

    # C6: Activation Consistency
    c6_x = bool(np.all(x <= a[:, None] + tol))
    c6_ytx = bool(np.all(y_tx <= a[:, None] + tol))
    c6_yrx = bool(np.all(y_rx <= a[:, None] + tol))

    # C3: Fronthaul Capacity
    c3 = bool(np.all(F_m <= C_m_FH + tol))

    # C10: Binary constraints
    def is_binary(v): return bool(np.all((v == 0) | (v == 1)))
    c10 = is_binary(x) and is_binary(y_tx) and is_binary(y_rx) and is_binary(a)

    return {
        "C3_fronthaul_capacity": c3,
        "C4_ap_power_budget": c4,
        "C6_activation_consistency": c6_x and c6_ytx and c6_yrx,
        "C10_binary_variables": c10
    }
