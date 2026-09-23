"""
src/prediction/dataset.py

Generates supervised trajectory sequences from the existing MobilitySimulator.

Each sample:
  input  shape: (history_length,  2)   -- past [x, y] positions
  target shape: (prediction_horizon, 2) -- future [x, y] positions

Positions are in simulation metres (same units as mobility_simulator).
No in-place normalisation is performed here; callers may normalise externally.

Dataset is split deterministically:
  train / validation / test are contiguous, non-overlapping windows of time
  (no shuffling before splitting) to avoid temporal leakage.
"""

import numpy as np
import yaml

from src.simulation.mobility_simulator import MobilitySimulator
from src.environment.topology import generate_ap_positions


def generate_trajectory(
    num_steps: int,
    num_users: int,
    num_targets: int,
    area_size: float = 500.0,
    dt: float = 1.0,
    seed: int = 42,
    mobility_config=None,
):
    """
    Run the mobility simulator for `num_steps` blocks and collect raw
    position trajectories.

    Returns
    -------
    user_traj   : np.ndarray, shape (num_steps, num_users, 2)   metres
    target_traj : np.ndarray, shape (num_steps, num_targets, 2) metres
    """
    rng = np.random.default_rng(seed)
    init_users   = rng.uniform(0, area_size, size=(num_users,   2))
    init_targets = rng.uniform(0, area_size, size=(num_targets, 2))

    sim = MobilitySimulator(
        init_users,
        init_targets,
        area_size=area_size,
        dt=dt,
        seed=seed,
        mobility_config=mobility_config,
    )

    user_traj   = []
    target_traj = []
    for _ in range(num_steps):
        u, t = sim.step()
        user_traj.append(u.copy())
        target_traj.append(t.copy())

    return np.asarray(user_traj), np.asarray(target_traj)


def build_sliding_window_samples(
    trajectory: np.ndarray,
    history_length: int,
    prediction_horizon: int,
):
    """
    Convert a single-object trajectory (T, 2) into supervised samples.

    Parameters
    ----------
    trajectory         : (T, 2) position sequence
    history_length     : L — number of input timesteps
    prediction_horizon : H — number of target timesteps

    Returns
    -------
    X : np.ndarray, shape (N, L, 2)
    Y : np.ndarray, shape (N, H, 2)

    where N = T - L - H + 1
    """
    T = len(trajectory)
    L = history_length
    H = prediction_horizon
    assert L >= 1 and H >= 1, "history_length and prediction_horizon must be >= 1"
    assert T > L + H, f"Trajectory too short: need > {L+H}, got {T}"

    X, Y = [], []
    for i in range(T - L - H + 1):
        X.append(trajectory[i : i + L])
        Y.append(trajectory[i + L : i + L + H])

    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)


def split_samples(X, Y, train_ratio, val_ratio):
    """
    Contiguous (non-overlapping) train / validation / test split.
    Temporal order is preserved; no shuffling.

    Returns (X_train, Y_train, X_val, Y_val, X_test, Y_test)
    """
    N = len(X)
    n_train = int(N * train_ratio)
    n_val   = int(N * val_ratio)

    X_train, Y_train = X[:n_train],          Y[:n_train]
    X_val,   Y_val   = X[n_train:n_train+n_val],  Y[n_train:n_train+n_val]
    X_test,  Y_test  = X[n_train+n_val:],    Y[n_train+n_val:]

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def load_prediction_config(config_path="configs/default.yaml") -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg["prediction"]


def build_mobility_dataset(
    config_path="configs/default.yaml",
    num_steps: int = 2000,
    entity: str = "users",
    seed: int = None,
    mobility_config=None,
):
    """
    Build a prediction dataset for a specified mobility regime.

    Unlike the legacy build_dataset(), this function performs the temporal
    train/validation/test split separately for every object before
    concatenation. This prevents the split from accidentally assigning
    different objects to different dataset partitions.

    Returns the same core arrays as build_dataset().
    """
    cfg = load_prediction_config(config_path)

    with open(config_path) as f2:
        net_cfg = yaml.safe_load(f2)["network"]

    L = int(cfg["history_length"])
    H = int(cfg["prediction_horizon"])
    tr = float(cfg["train_ratio"])
    vr = float(cfg["validation_ratio"])
    s = int(seed if seed is not None else cfg["seed"])

    num_users = int(net_cfg["num_users"])
    num_targets = int(net_cfg["num_targets"])
    area_size = float(net_cfg["area_size"])

    user_traj, target_traj = generate_trajectory(
        num_steps=num_steps,
        num_users=num_users,
        num_targets=num_targets,
        area_size=area_size,
        seed=s,
        mobility_config=mobility_config,
    )

    if entity not in {"users", "targets"}:
        raise ValueError("entity must be 'users' or 'targets'")

    traj = user_traj if entity == "users" else target_traj
    num_objects = traj.shape[1]

    train_x, train_y = [], []
    val_x, val_y = [], []
    test_x, test_y = [], []

    for obj_idx in range(num_objects):
        x_obj, y_obj = build_sliding_window_samples(
            traj[:, obj_idx, :],
            L,
            H,
        )

        n = len(x_obj)
        n_train = int(n * tr)
        n_val = int(n * vr)

        train_x.append(x_obj[:n_train])
        train_y.append(y_obj[:n_train])

        val_x.append(x_obj[n_train:n_train + n_val])
        val_y.append(y_obj[n_train:n_train + n_val])

        test_x.append(x_obj[n_train + n_val:])
        test_y.append(y_obj[n_train + n_val:])

    X_train = np.concatenate(train_x, axis=0)
    Y_train = np.concatenate(train_y, axis=0)
    X_val = np.concatenate(val_x, axis=0)
    Y_val = np.concatenate(val_y, axis=0)
    X_test = np.concatenate(test_x, axis=0)
    Y_test = np.concatenate(test_y, axis=0)

    mobility_type = "constant_velocity"
    if mobility_config is not None:
        mobility_type = mobility_config.get("type", "constant_velocity")

    return {
        "X_train": X_train,
        "Y_train": Y_train,
        "X_val": X_val,
        "Y_val": Y_val,
        "X_test": X_test,
        "Y_test": Y_test,
        "user_traj": user_traj,
        "target_traj": target_traj,
        "metadata": {
            "history_length":     L,
            "prediction_horizon": H,
            "num_steps":          num_steps,
            "entity":             entity,
            "seed":               s,
            "num_objects":        num_objects,
            "mobility_type":      mobility_type,
            "area_size":          area_size,   # needed for domain normalization
        },
    }


def build_dataset(
    config_path="configs/default.yaml",
    num_steps: int = 1000,
    entity: str = "users",
    seed: int = None,
):
    """
    High-level entry point.  Generates trajectory data and returns splits.

    Parameters
    ----------
    config_path : path to YAML configuration
    num_steps   : total simulation steps to generate (overrides config)
    entity      : "users" or "targets"
    seed        : random seed (defaults to config["prediction"]["seed"])

    Returns
    -------
    dict with keys:
        X_train, Y_train, X_val, Y_val, X_test, Y_test
        metadata (history_length, prediction_horizon, num_steps, entity, seed)
    """
    cfg = load_prediction_config(config_path)
    with open(config_path) as f2:
        net_cfg = yaml.safe_load(f2)["network"]

    L  = int(cfg["history_length"])
    H  = int(cfg["prediction_horizon"])
    tr = float(cfg["train_ratio"])
    vr = float(cfg["validation_ratio"])
    s  = int(seed if seed is not None else cfg["seed"])

    num_users   = net_cfg["num_users"]
    num_targets = net_cfg["num_targets"]
    area_size   = net_cfg["area_size"]

    user_traj, target_traj = generate_trajectory(
        num_steps=num_steps,
        num_users=num_users,
        num_targets=num_targets,
        area_size=area_size,
        seed=s,
    )

    traj = user_traj if entity == "users" else target_traj   # (T, N_obj, 2)
    num_objects = traj.shape[1]

    # Build per-object samples then concatenate
    Xs, Ys = [], []
    for obj_idx in range(num_objects):
        xi, yi = build_sliding_window_samples(traj[:, obj_idx, :], L, H)
        Xs.append(xi)
        Ys.append(yi)

    X_all = np.concatenate(Xs, axis=0)
    Y_all = np.concatenate(Ys, axis=0)

    X_train, Y_train, X_val, Y_val, X_test, Y_test = split_samples(X_all, Y_all, tr, vr)

    return {
        "X_train": X_train, "Y_train": Y_train,
        "X_val":   X_val,   "Y_val":   Y_val,
        "X_test":  X_test,  "Y_test":  Y_test,
        "user_traj":   user_traj,
        "target_traj": target_traj,
        "metadata": {
            "history_length":     L,
            "prediction_horizon": H,
            "num_steps":          num_steps,
            "entity":             entity,
            "seed":               s,
            "num_objects":        num_objects,
        },
    }
