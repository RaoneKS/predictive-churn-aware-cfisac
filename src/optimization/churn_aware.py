import numpy as np

from src.channels.communication import generate_communication_features
from src.channels.sensing import generate_sensing_gains
from src.clustering.churn import total_churn_cost
from src.metrics.sensing import (
    aggregate_fisher_information,
    posterior_covariance,
    information_gain,
)
from src.metrics.system import (
    communication_rate,
    communication_sinr,
    fronthaul_load,
    power_consumption,
    qos_violations,
    tracking_violations,
    total_system_cost,
)


def evaluate_cluster_performance(
    x, y_tx, y_rx,
    ap_positions, user_positions, target_positions,
    prior_covariances=None,
    transmit_power=1.0,
    noise_power=1e-9,
    comm_fronthaul_per_user=1.0,
    sensing_fronthaul_per_target=1.0,
    min_rate=0.5,
    max_tracking_error=2.0,
    alpha_communication=1.0,
    beta_sensing=1.0,
    lambda_energy=1.0,
    lambda_fronthaul=1.0,
    lambda_qos=1.0,
    lambda_tracking=1.0,
    ref=None,
):
    x = np.asarray(x, dtype=float)
    y_tx = np.asarray(y_tx, dtype=float)
    y_rx = np.asarray(y_rx, dtype=float)
    
    if ref is None:
        ref = {
            "communication_rate_bps": 1.0,
            "sensing_information_gain": 1.0,
            "energy_watts": 1.0,
            "fronthaul_links": 1.0,
            "churn_events": 1.0,
            "qos_violations": 1.0,
            "tracking_violations": 1.0
        }
    
    # 1. Communication Model (RAW)
    G_comm, _ = generate_communication_features(ap_positions, user_positions)
    signal_power = np.sum(x * G_comm, axis=0) * transmit_power
    active_aps = np.sum(x, axis=1)
    total_tx = active_aps * transmit_power
    total_received_power = np.sum(total_tx[:, None] * G_comm, axis=0)
    interference_power = np.maximum(total_received_power - signal_power, 0.0)
    
    sinr = communication_sinr(signal_power, interference_power, noise_power=noise_power)
    raw_rates = communication_rate(sinr)
    raw_comm_utility = np.sum(raw_rates)
    raw_qos_viol = np.sum(qos_violations(raw_rates, min_rate))

    # 2. Sensing Model (RAW)
    Q = len(target_positions)
    if prior_covariances is None:
        prior_covariances = [np.eye(2) for _ in range(Q)]
        
    J_total = aggregate_fisher_information(ap_positions, target_positions, y_tx, y_rx)
    post_covs = []
    raw_sens_utility = 0.0
    for q in range(Q):
        post_cov = posterior_covariance(prior_covariances[q], J_total[q])
        post_covs.append(post_cov)
        raw_sens_utility += information_gain(prior_covariances[q], post_cov)
        
    raw_track_viol = np.sum(tracking_violations(post_covs, max_tracking_error))

    # 3. System Penalties (RAW)
    raw_energy = np.sum(power_consumption(x, y_tx, transmit_power))
    raw_fronthaul = np.sum(fronthaul_load(
        x, y_rx, 
        np.full(len(user_positions), comm_fronthaul_per_user),
        np.full(Q, sensing_fronthaul_per_target)
    ))
    
    # NORMALIZED COMPONENTS
    n_comm = raw_comm_utility / float(ref["communication_rate_bps"])
    n_sens = raw_sens_utility / float(ref["sensing_information_gain"])
    n_energy = raw_energy / float(ref["energy_watts"])
    n_front = raw_fronthaul / float(ref["fronthaul_links"])
    n_qos = raw_qos_viol / float(ref["qos_violations"])
    n_track = raw_track_viol / float(ref["tracking_violations"])
    
    # Calculate costs via standard function with normalized inputs
    normalized_cost = total_system_cost(
        [n_energy], 
        [n_front], 
        churn=0.0,  # Churn evaluated explicitly later
        qos_violation=[n_qos], 
        tracking_violation=[n_track],
        lambda_energy=lambda_energy,
        lambda_fronthaul=lambda_fronthaul,
        lambda_churn=0.0,
        lambda_qos=lambda_qos,
        lambda_tracking=lambda_tracking,
    )
    
    # FINAL COMBINED OBJECTIVE = alpha * norm_comm + beta * norm_sens - normalized_cost
    combined_objective = (alpha_communication * n_comm) + (beta_sensing * n_sens) - normalized_cost

    return {
        "raw_communication_utility": float(raw_comm_utility),
        "raw_sensing_utility": float(raw_sens_utility),
        "raw_energy_penalty": float(raw_energy),
        "raw_fronthaul_penalty": float(raw_fronthaul),
        "qos_violations": float(raw_qos_viol),
        "tracking_violations": float(raw_track_viol),
        
        "norm_communication_utility": float(n_comm),
        "norm_sensing_utility": float(n_sens),
        "norm_energy_penalty": float(n_energy),
        "norm_fronthaul_penalty": float(n_front),
        
        "combined_objective": float(combined_objective),
        "raw_rates": raw_rates.tolist(),
        "raw_traces": [float(np.trace(c)) for c in post_covs]
    }


def estimate_reconfiguration_gain(current_performance, predicted_performance):
    """
    Calculate predicted objective minus current objective.
    Preserves negative gains.
    """
    return float(predicted_performance["combined_objective"] - current_performance["combined_objective"])


def churn_aware_decision(
    x_curr, y_tx_curr, y_rx_curr,
    x_pred, y_tx_pred, y_rx_pred,
    current_performance,
    predicted_performance,
    c_comm=1.0,
    c_sensing_tx=1.0,
    c_sensing_rx=1.0,
    ref=None,
    lambda_churn=1.0,
):
    if ref is None:
        ref = {"churn_events": 1.0}
        
    gain = estimate_reconfiguration_gain(current_performance, predicted_performance)
    
    raw_churn_cost = total_churn_cost(
        x_curr, x_pred,
        y_tx_curr, y_tx_pred,
        y_rx_curr, y_rx_pred,
        c_comm=c_comm,
        c_sensing_tx=c_sensing_tx,
        c_sensing_rx=c_sensing_rx,
    )
    
    norm_churn_events = raw_churn_cost / float(ref["churn_events"])
    norm_churn_cost = norm_churn_events * float(lambda_churn)
    
    net_gain = gain - norm_churn_cost
    
    # 1. If predicted_gain < churn_cost: decision == KEEP
    # 2. If predicted_gain > churn_cost: decision == RECONFIGURE
    # Because predicted_gain - churn_cost = net_gain
    if net_gain > 0.0:
        decision = "RECONFIGURE"
    else:
        decision = "KEEP"
        
    return {
        "decision": decision,
        "predicted_gain": float(gain),
        "raw_churn": float(raw_churn_cost),
        "norm_churn_cost": float(norm_churn_cost),
        "net_gain": float(net_gain),
        "current_objective": float(current_performance["combined_objective"]),
        "predicted_objective": float(predicted_performance["combined_objective"]),
    }
