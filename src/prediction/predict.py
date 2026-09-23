"""
src/prediction/predict.py

Prediction API — bridges the trained LSTM to the CF-ISAC simulation.

Key functions
-------------
predict_trajectory(model, history, device, area_size)
    history   : (history_length, 2)  in metres
    area_size : float or None.  When supplied, input is normalized to [-1,+1]
                via domain normalization before inference and the output is
                de-normalized back to metres.  When None, raw metre values are
                passed unchanged (original behavior).
    returns   : (prediction_horizon, 2) in metres

predict_all_objects(model, histories, device, area_size)
    histories : (num_objects, history_length, 2)
    returns   : (num_objects, prediction_horizon, 2)

Output format matches MobilitySimulator.predict_users() / predict_targets()
so that the existing predictive_policy() can consume LSTM predictions without
modification.

Constant-velocity baseline wrapper
------------------------------------
predict_cv(positions, velocities, prediction_horizon, dt)
"""

import numpy as np
import torch

from src.environment.mobility import predict_constant_velocity
from src.prediction.normalization import normalize, denormalize


def predict_trajectory(
    model,
    history: np.ndarray,
    device: torch.device = None,
    area_size: float = None,
) -> np.ndarray:
    """
    Single-object LSTM prediction.

    Parameters
    ----------
    model     : trained TrajectoryLSTM
    history   : (history_length, 2) float32 positions in metres
    device    : torch device (inferred from model if None)
    area_size : float or None.  When supplied, positions are normalized to
                [-1, +1] before the model and de-normalized afterward.
                When None, raw metre values are used (original behavior).

    Returns
    -------
    np.ndarray, shape (prediction_horizon, 2)  in metres
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    h = np.asarray(history, dtype=np.float32)
    if area_size is not None:
        h = normalize(h, area_size)

    x = torch.from_numpy(h).unsqueeze(0).to(device)   # (1, L, 2)

    with torch.no_grad():
        pred = model(x)                                # (1, H, 2)

    out = pred.squeeze(0).cpu().numpy()                # (H, 2)
    if area_size is not None:
        out = denormalize(out, area_size)
    return out


def predict_all_objects(
    model,
    histories: np.ndarray,
    device: torch.device = None,
    area_size: float = None,
) -> np.ndarray:
    """
    Batch LSTM prediction for all objects.

    Parameters
    ----------
    model     : trained TrajectoryLSTM
    histories : (num_objects, history_length, 2)
    device    : torch device
    area_size : float or None — see predict_trajectory

    Returns
    -------
    np.ndarray, shape (num_objects, prediction_horizon, 2)
    Matches the layout of MobilitySimulator.predict_users() output.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    h = np.asarray(histories, dtype=np.float32)
    if area_size is not None:
        h = normalize(h, area_size)

    x = torch.from_numpy(h).to(device)

    with torch.no_grad():
        pred = model(x)   # (num_objects, H, 2)

    out = pred.cpu().numpy()
    if area_size is not None:
        out = denormalize(out, area_size)
    return out


def predict_cv(
    positions: np.ndarray,
    velocities: np.ndarray,
    prediction_horizon: int,
    dt: float = 1.0,
) -> np.ndarray:
    """
    Constant-velocity prediction wrapper with the same output shape as
    predict_all_objects.

    Parameters
    ----------
    positions  : (num_objects, 2)
    velocities : (num_objects, 2)
    prediction_horizon : H

    Returns
    -------
    np.ndarray, shape (num_objects, prediction_horizon, 2)
    """
    preds = []
    for pos, vel in zip(positions, velocities):
        future = predict_constant_velocity(pos, vel, prediction_horizon, dt)
        preds.append(future)
    return np.asarray(preds)   # (num_objects, H, 2)


def load_lstm(checkpoint_path: str, model_class, pred_config: dict, device=None):
    """
    Load a saved model checkpoint.

    Parameters
    ----------
    checkpoint_path : path to .pt file
    model_class     : TrajectoryLSTM class
    pred_config     : prediction config dict (for architecture params)
    device          : torch device

    Returns
    -------
    model (eval mode, on device)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model = model_class(
        input_size=2,
        hidden_size=int(pred_config["hidden_size"]),
        num_layers=int(pred_config["num_layers"]),
        dropout=float(pred_config["dropout"]),
        prediction_horizon=int(pred_config["prediction_horizon"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model
