import numpy as np
from typing import Dict, Any, List, Optional
from itertools import product
from src.optimization.baselines import cluster_for_positions
from src.optimization.two_timescale import fast_update
from src.optimization.churn_aware import churn_aware_decision, total_churn_cost
from src.optimization.deficiency_cvar import evaluate_deficiency_constraints

def ap_activation(x: np.ndarray, y_tx: np.ndarray, y_rx: np.ndarray) -> np.ndarray:
    """Derive AP activation boolean vector from association matrices."""
    return ((np.sum(x, axis=1) > 0) | (np.sum(y_tx, axis=1) > 0) | (np.sum(y_rx, axis=1) > 0)).astype(int)

def generate_topology_candidates(
    ap_positions: np.ndarray,
    pred_users: np.ndarray,
    pred_targets: np.ndarray,
    prev_x: np.ndarray,
    prev_y_tx: np.ndarray,
    prev_y_rx: np.ndarray,
    cluster_kwargs: Dict[str, Any],
    p1_max_candidates: int = 20,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Generate deterministic set of topology candidates.
    Contains at least:
    1. Previous topology (KEEP)
    2. Primary predicted topology
    3. Small neighborhood perturbations around primary.
    """
    rng = np.random.RandomState(seed)
    candidates = []
    
    # 1. Previous (KEEP)
    candidates.append({"x": prev_x.copy(), "y_tx": prev_y_tx.copy(), "y_rx": prev_y_rx.copy(), "name": "KEEP"})
    
    # 2. Primary predicted
    prim_x, prim_y_tx, prim_y_rx = cluster_for_positions(
        ap_positions, pred_users, pred_targets, **cluster_kwargs
    )
    
    if not (np.array_equal(prev_x, prim_x) and np.array_equal(prev_y_tx, prim_y_tx) and np.array_equal(prev_y_rx, prim_y_rx)):
        candidates.append({"x": prim_x.copy(), "y_tx": prim_y_tx.copy(), "y_rx": prim_y_rx.copy(), "name": "PRIMARY"})
    
    M, K = prim_x.shape
    Q = prim_y_tx.shape[1]
    
    # Exhaustive enumeration for tiny test cases (<=20 total combinations is safe)
    # Actually, we will just generate neighborhood flips.
    flips_to_try = []
    for m in range(M):
        for k in range(K): flips_to_try.append(('x', m, k))
        for q in range(Q):
            flips_to_try.append(('y_tx', m, q))
            flips_to_try.append(('y_rx', m, q))
            
    rng.shuffle(flips_to_try)
    
    for ftype, m, idx in flips_to_try:
        if len(candidates) >= p1_max_candidates:
            break
            
        c_x = prim_x.copy()
        c_ytx = prim_y_tx.copy()
        c_yrx = prim_y_rx.copy()
        
        if ftype == 'x': c_x[m, idx] = 1 - c_x[m, idx]
        elif ftype == 'y_tx': c_ytx[m, idx] = 1 - c_ytx[m, idx]
        elif ftype == 'y_rx': c_yrx[m, idx] = 1 - c_yrx[m, idx]
            
        is_dup = False
        for c in candidates:
            if np.array_equal(c['x'], c_x) and np.array_equal(c['y_tx'], c_ytx) and np.array_equal(c['y_rx'], c_yrx):
                is_dup = True
                break
        
        if not is_dup:
            candidates.append({
                "x": c_x, "y_tx": c_ytx, "y_rx": c_yrx, "name": f"FLIP_{ftype}_{m}_{idx}"
            })
        
    return candidates

def filter_hard_constraints(
    cand: Dict[str, Any],
    cluster_kwargs: Dict[str, Any],
    fronthaul_capacity: Optional[np.ndarray],
    comm_fronthaul_per_user: float,
    sensing_fronthaul_per_target: float,
    control_fronthaul: float
) -> tuple[bool, str]:
    """Check deterministic topology constraints."""
    x, y_tx, y_rx = cand["x"], cand["y_tx"], cand["y_rx"]
    
    # Binary variables
    for v in [x, y_tx, y_rx]:
        if not np.all(np.isin(v, (0, 1))):
            return False, "Not binary"
            
    a = ap_activation(x, y_tx, y_rx)
    
    if not (np.all(x <= a[:, None]) and np.all(y_tx <= a[:, None]) and np.all(y_rx <= a[:, None])):
        return False, "Activation consistency"
        
    min_x = cluster_kwargs.get("min_aps_per_user", 1)
    max_x = cluster_kwargs.get("aps_per_user", 100)
    min_ytx = cluster_kwargs.get("tx_aps_per_target", 1)
    min_yrx = cluster_kwargs.get("rx_aps_per_target", 1)
    
    if np.any(np.sum(x, axis=0) < min_x):
        return False, "Min Comm APs"
    if np.any(np.sum(x, axis=0) > max_x):
        return False, "Max Comm APs"
    if np.any(np.sum(y_tx, axis=0) < min_ytx):
        return False, "Min TX APs"
    if np.any(np.sum(y_rx, axis=0) < min_yrx):
        return False, "Min RX APs"
        
    if fronthaul_capacity is not None:
        c_f = np.full(x.shape[1], comm_fronthaul_per_user)
        s_f = np.full(y_rx.shape[1], sensing_fronthaul_per_target)
        fh_cost = x @ c_f + y_rx @ s_f + control_fronthaul
        if np.any(fh_cost > fronthaul_capacity + 1e-8):
            return False, "Fronthaul Capacity"
            
    return True, "Pass"

def solve_p1_decomposed(
    sim,
    predictor,
    ap_positions: np.ndarray,
    num_antennas: int,
    prev_x: np.ndarray,
    prev_y_tx: np.ndarray,
    prev_y_rx: np.ndarray,
    T_slow: int,
    P_max: float,
    precoder: str,
    alpha_power: float,
    beta_power: float,
    cluster_kwargs: Dict[str, Any],
    perf_kwargs: Dict[str, Any],
    churn_kwargs: Dict[str, Any],
    phys_kwargs: Dict[str, Any],
    sens_kwargs: Dict[str, Any],
    joint_kwargs: Dict[str, Any],
    ref: Dict[str, float],
    p1_max_candidates: int = 20,
    seed: int = 42,
    horizon_aggregation: str = "sum",
    enforce_deficiency_cvar: bool = False,
    max_comm_cvar: Optional[float] = None,
    max_sensing_cvar: Optional[float] = None,
    min_rate_bps: Optional[float] = None,
    epsilon_trk: Optional[np.ndarray] = None,
    enforce_qos: bool = False,
    num_scenarios: int = 0,
    scenario_seed: int = 0,
    residual_dt: Optional[float] = None,
) -> Dict[str, Any]:
    
    dt = float(residual_dt) if residual_dt is not None else float(sim.dt)
    pred_users_H, pred_targets_H = predictor.predict(sim, T_slow)
    pred_users_H = np.asarray(pred_users_H, dtype=float)
    pred_targets_H = np.asarray(pred_targets_H, dtype=float)
    
    from src.optimization.risk_aware import _scenario_perturbations
    user_history = [h[0] for h in sim.position_history]
    target_history = [h[1] for h in sim.position_history]
    
    if num_scenarios > 0:
        user_pert = _scenario_perturbations(user_history, T_slow, num_scenarios, scenario_seed, pred_users_H.shape[0], dt)
        target_pert = _scenario_perturbations(target_history, T_slow, num_scenarios, scenario_seed + 1, pred_targets_H.shape[0], dt)
    else:
        user_pert = np.zeros((1, pred_users_H.shape[0], T_slow, 2))
        target_pert = np.zeros((1, pred_targets_H.shape[0], T_slow, 2))
        num_scenarios = 1
        
    candidates = generate_topology_candidates(
        ap_positions, pred_users_H[:, -1, :], pred_targets_H[:, -1, :],
        prev_x, prev_y_tx, prev_y_rx, cluster_kwargs, p1_max_candidates, seed
    )
    
    num_generated = len(candidates)
    fh_cap = joint_kwargs.get("fronthaul_capacity", None)
    c_fpu = joint_kwargs.get("comm_fronthaul_per_user", 1.0)
    s_fpt = joint_kwargs.get("sensing_fronthaul_per_target", 1.0)
    ctrl_fh = joint_kwargs.get("control_fronthaul", 0.0)
    lambda_churn = churn_kwargs.get("lambda_churn", 1.0)
    lambda_energy = churn_kwargs.get("lambda_energy", 1.0)
    lambda_fronthaul = churn_kwargs.get("lambda_fronthaul", 1.0)
    
    best_cand = None
    best_obj = -np.inf
    rejection_counts = {}
    H_blocks = pred_users_H.shape[1]
    
    # Forward QoS settings to the physical evaluator only when explicitly enabled.
    joint_kwargs = dict(joint_kwargs)
    # Remove incorrect 'min_rate' if it was passed
    if "min_rate" in joint_kwargs:
        del joint_kwargs["min_rate"]

    if enforce_qos and min_rate_bps is not None:
        joint_kwargs["enforce_qos"] = True
        joint_kwargs["min_rate_bps"] = min_rate_bps

    if epsilon_trk is not None and joint_kwargs.get("tracking_error_thresholds") is None:
        joint_kwargs["tracking_error_thresholds"] = epsilon_trk
    
    for cand in candidates:
        passed, reason = filter_hard_constraints(cand, cluster_kwargs, fh_cap, c_fpu, s_fpt, ctrl_fh)
        if not passed:
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            cand["status"] = "REJECTED_HARD"
            cand["reason"] = reason
            continue
            
        x, y_tx, y_rx = cand["x"], cand["y_tx"], cand["y_rx"]
        cand["a"] = ap_activation(x, y_tx, y_rx)
        
        raw_churn = total_churn_cost(prev_x, prev_y_tx, prev_y_rx, x, y_tx, y_rx)
        norm_churn = raw_churn / float(ref.get("churn_events", 1.0))
        churn_penalty = lambda_churn * norm_churn
        
        scen_objs = []
        scen_rates = []
        scen_traces = []
        cand_feasible = True
        
        # Primary evaluation result for W, S
        cand["primary_fast_result"] = None
        
        for i in range(num_scenarios):
            if not cand_feasible: break
            pu = pred_users_H + user_pert[i]
            pt = pred_targets_H + target_pert[i]
            
            h_objs = []
            h_rates = []
            h_traces = []
            
            for h in range(H_blocks):
                res = fast_update(
                    ap_positions, pu[:, h, :], pt[:, h, :],
                    x, y_tx, y_rx, num_antennas, P_max,
                    precoder=precoder, alpha=alpha_power, beta=beta_power,
                    phys_kwargs=phys_kwargs, sens_kwargs=sens_kwargs, joint_kwargs=joint_kwargs
                )
                
                if not res.get("feasible", True):
                    cand_feasible = False
                    cand_reason = "Physical Infeasible (Power/Tracking/QoS/Fronthaul)"
                    break

                # Explicit P1 communication QoS hard constraint:
                # every served user's rate must satisfy R_k >= R_min.
                if enforce_qos and min_rate_bps is not None:
                    rates = np.asarray(res["rates_bps"], dtype=float)
                    if np.any(rates < float(min_rate_bps)):
                        cand_feasible = False
                        cand_reason = "QoS Infeasible"
                        break
                    
                # Calculate energy and fronthaul penalty properly for full P1 objective
                # Phase 5B objective is alpha * sum_rate + beta * info_gain - lambda_E * energy - lambda_F * fronthaul
                energy = res.get("energy_penalty", 0.0) # Actually we need to calculate it if not provided
                
                # Fast update provides 'objective' which is alpha*comm + beta*sens (already scaled?) 
                # Let's extract raw utilities
                r_c = np.sum(res["rates_bps"])
                n_c = r_c / float(ref.get("communication_rate_bps", 1.0))
                r_s = np.sum(res["information_gain"])
                n_s = r_s / float(ref.get("sensing_information_gain", 1.0))
                
                # Energy penalty
                if "P_tx" in res:
                    raw_energy = np.sum(res["P_tx"]) # or more sophisticated if we have it
                else:
                    raw_energy = np.sum(res["P_comm"]) + np.sum(res["P_sens"])
                    
                n_e = raw_energy / float(ref.get("energy_watts", 1.0))
                
                # Fronthaul penalty
                raw_fh = np.sum(x @ np.full(x.shape[1], c_fpu) + y_rx @ np.full(y_rx.shape[1], s_fpt) + ctrl_fh)
                n_fh = raw_fh / float(ref.get("fronthaul_links", 1.0))
                
                obj_h = (alpha_power * n_c) + (beta_power * n_s) - (lambda_energy * n_e) - (lambda_fronthaul * n_fh)
                h_objs.append(obj_h)
                h_rates.append(res["rates_bps"])
                
                # For CVaR tracing: evaluate_physical_sensing_bridge gives tracking_error = trace(P+)
                if "tracking_error" in res:
                    h_traces.append(res["tracking_error"])
                else:
                    h_traces.append(np.zeros(y_tx.shape[1]))
                
                if h == H_blocks - 1 and i == 0:
                    cand["primary_fast_result"] = res
                    cand["raw_energy"] = raw_energy
                    cand["raw_fronthaul"] = raw_fh
                    cand["comm_utility"] = r_c
                    cand["sensing_utility"] = r_s
                    
            if not cand_feasible:
                break
                
            agg = np.sum(h_objs) if horizon_aggregation == "sum" else np.mean(h_objs)
            scen_objs.append(agg - churn_penalty)
            
            scen_rates.append(np.mean(h_rates, axis=0))
            scen_traces.append(np.mean(h_traces, axis=0))
            
        if not cand_feasible:
            cand["status"] = "REJECTED_FAST"
            cand["reason"] = cand_reason
            rejection_counts[cand_reason] = rejection_counts.get(cand_reason, 0) + 1
            continue
            
        # CVaR check
        if enforce_deficiency_cvar:
            rates_scenarios = np.array(scen_rates)
            traces_scenarios = np.array(scen_traces)
            deficiency_results = evaluate_deficiency_constraints(
                rates_scenarios, traces_scenarios,
                min_rate_bps=min_rate_bps or 0.0,
                epsilon_trk=epsilon_trk if epsilon_trk is not None else np.zeros(pred_targets_H.shape[0]),
                max_comm_cvar=max_comm_cvar,
                max_sensing_cvar=max_sensing_cvar
            )
            cand["deficiency_cvar_results"] = deficiency_results
            if not deficiency_results["satisfied"]:
                cand["status"] = "REJECTED_CVAR"
                cand["reason"] = "Deficiency CVaR Limits Exceeded"
                rejection_counts[cand["reason"]] = rejection_counts.get(cand["reason"], 0) + 1
                continue
        else:
            cand["deficiency_cvar_results"] = None
            
        cand_obj = np.mean(scen_objs) # Expected P1 objective over scenarios
        cand["objective"] = cand_obj
        cand["status"] = "FEASIBLE"
        cand["raw_churn"] = raw_churn
        
        if cand_obj > best_obj:
            best_obj = cand_obj
            best_cand = cand
            
    if best_cand is None:
        best_cand = candidates[0] # KEEP
        best_cand["status"] = "FALLBACK"
        best_cand["reason"] = "no_feasible_p1_candidate"
        best_cand["objective"] = -np.inf
        # Just run fast_update once on KEEP to populate variables
        res = fast_update(
            ap_positions, pred_users_H[:, -1, :], pred_targets_H[:, -1, :],
            best_cand["x"], best_cand["y_tx"], best_cand["y_rx"], num_antennas, P_max,
            precoder=precoder, alpha=alpha_power, beta=beta_power,
            phys_kwargs=phys_kwargs, sens_kwargs=sens_kwargs, joint_kwargs=joint_kwargs
        )
        best_cand["primary_fast_result"] = res
        best_cand["a"] = ap_activation(best_cand["x"], best_cand["y_tx"], best_cand["y_rx"])
        best_cand["raw_churn"] = 0.0
        best_cand["deficiency_cvar_results"] = None
        best_cand["comm_utility"] = np.sum(res["rates_bps"])
        best_cand["sensing_utility"] = np.sum(res["information_gain"])
        best_cand["raw_energy"] = np.sum(res.get("P_tx", np.zeros_like(best_cand["a"])))
        best_cand["raw_fronthaul"] = 0.0
        
    return {
        "x": best_cand["x"],
        "y_tx": best_cand["y_tx"],
        "y_rx": best_cand["y_rx"],
        "a": best_cand["a"],
        "primary_fast_result": best_cand["primary_fast_result"],
        "objective": best_cand["objective"],
        "comm_utility": best_cand.get("comm_utility", 0.0),
        "sensing_utility": best_cand.get("sensing_utility", 0.0),
        "energy": best_cand.get("raw_energy", 0.0),
        "fronthaul": best_cand.get("raw_fronthaul", 0.0),
        "churn": best_cand.get("raw_churn", 0.0),
        "deficiency_cvar_results": best_cand.get("deficiency_cvar_results"),
        "fallback_used": best_cand["status"] == "FALLBACK",
        "fallback_reason": best_cand.get("reason", ""),
        "num_candidates_generated": num_generated,
        "rejection_counts": rejection_counts,
        "best_candidate_name": best_cand.get("name", ""),
        "candidates": candidates
    }
