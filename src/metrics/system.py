import numpy as np


def communication_rate(
    sinr,
    bandwidth_hz=20e6,
    pilot_fraction=0.1,
):
    """
    Shannon-rate approximation:

        R = (1 - tau_p/tau_c) B log2(1 + SINR)
    """
    sinr = np.maximum(np.asarray(sinr, dtype=float), 0.0)

    prelog = max(0.0, 1.0 - pilot_fraction)

    return (
        prelog
        * bandwidth_hz
        * np.log2(1.0 + sinr)
    )


def communication_sinr(
    signal_power,
    interference_power,
    noise_power=1e-9,
):
    """
    SINR = signal / (interference + noise)
    """
    signal_power = np.maximum(
        np.asarray(signal_power, dtype=float),
        0.0,
    )

    interference_power = np.maximum(
        np.asarray(interference_power, dtype=float),
        0.0,
    )

    return signal_power / (
        interference_power + noise_power
    )


def fronthaul_load(
    x,
    y_rx,
    comm_fronthaul_per_user,
    sensing_fronthaul_per_target,
    control_fronthaul=0.0,
):
    """
    Per-AP fronthaul load:

        F_m =
        sum_k x_mk fC_mk
        +
        sum_q yR_mq fS_mq
        +
        f_ctrl_m
    """
    x = np.asarray(x, dtype=float)
    y_rx = np.asarray(y_rx, dtype=float)

    f_comm = np.asarray(
        comm_fronthaul_per_user,
        dtype=float,
    )

    f_sensing = np.asarray(
        sensing_fronthaul_per_target,
        dtype=float,
    )

    comm_load = x * f_comm[None, :]
    sensing_load = y_rx * f_sensing[None, :]

    return (
        np.sum(comm_load, axis=1)
        + np.sum(sensing_load, axis=1)
        + control_fronthaul
    )


def power_consumption(
    x,
    y_tx,
    transmit_power,
    circuit_power=1.0,
    processing_power=0.1,
    sleep_power=0.05,
    pa_efficiency=0.4,
):
    """
    Simplified per-AP power model.

    An AP is considered active if it participates in
    communication or sensing TX.
    """
    x = np.asarray(x, dtype=float)
    y_tx = np.asarray(y_tx, dtype=float)

    transmit_power = np.asarray(
        transmit_power,
        dtype=float,
    )

    active = (
        (np.sum(x, axis=1) > 0)
        | (np.sum(y_tx, axis=1) > 0)
    )

    tx = np.maximum(transmit_power, 0.0)

    power = np.where(
        active,
        circuit_power
        + tx / max(pa_efficiency, 1e-12)
        + processing_power,
        sleep_power,
    )

    return power


def qos_violations(
    rates,
    minimum_rate,
):
    """
    Communication QoS deficiency:

        [R_min - R_k]^+
    """
    rates = np.asarray(rates, dtype=float)

    return np.maximum(
        minimum_rate - rates,
        0.0,
    )


def tracking_violations(
    posterior_covariances,
    threshold,
):
    """
    Sensing tracking deficiency:

        [tr(P+) - epsilon]^+
    """
    violations = []

    for covariance in posterior_covariances:
        error = np.trace(covariance)

        violations.append(
            max(error - threshold, 0.0)
        )

    return np.asarray(violations)


def total_system_cost(
    energy,
    fronthaul,
    churn,
    qos_violation=0.0,
    tracking_violation=0.0,
    lambda_energy=0.1,
    lambda_fronthaul=0.1,
    lambda_churn=0.1,
    lambda_qos=1.0,
    lambda_tracking=1.0,
):
    """
    Combine system penalties into one scalar cost.
    """
    return (
        lambda_energy * np.sum(energy)
        + lambda_fronthaul * np.sum(fronthaul)
        + lambda_churn * churn
        + lambda_qos * np.sum(qos_violation)
        + lambda_tracking * np.sum(tracking_violation)
    )
