"""
src/prediction/predictor_interface.py

Clean, interchangeable predictor API for CF-ISAC simulations.

Both constant-velocity (CV) and LSTM predictors expose the same interface:

    predict(sim, horizon) -> (users_pred, targets_pred)

where:
    users_pred   : (num_users,   horizon, 2)  predicted positions in metres
    targets_pred : (num_targets, horizon, 2)  predicted positions in metres

This is intentionally the same shape as MobilitySimulator.predict_users() so
that the simulation loop only needs to call predict() and never know which
predictor it is using.

Usage
-----
    from src.prediction.predictor_interface import make_predictor

    predictor = make_predictor("cv")
    predictor = make_predictor("lstm", user_ckpt=..., target_ckpt=..., area_size=...)

    pred_users, pred_targets = predictor.predict(sim, horizon)
"""

import numpy as np

from src.simulation.mobility_simulator import MobilitySimulator

# NOTE: torch is imported lazily inside LSTMPredictor (see __init__).  The CV
# predictor and every non-LSTM code path must remain importable in environments
# where torch is not installed.


class CVPredictor:
    """
    Constant-velocity predictor.

    Calls MobilitySimulator.predict_users / predict_targets — these already
    return (N, H, 2) and are the ground-truth CV predictions, using the
    simulator's current velocity state.
    """

    def predict(self, sim: MobilitySimulator, horizon: int) -> tuple:
        """
        Parameters
        ----------
        sim     : MobilitySimulator (current state)
        horizon : int  number of future steps to predict

        Returns
        -------
        users_pred   : (num_users,   horizon, 2)
        targets_pred : (num_targets, horizon, 2)
        """
        return sim.predict_users(horizon), sim.predict_targets(horizon)

    def __repr__(self):
        return "CVPredictor()"


class LSTMPredictor:
    """
    LSTM-based trajectory predictor.

    Loads trained checkpoints for users and targets.
    Uses the position history stored in MobilitySimulator.position_history.
    Falls back to CV prediction if insufficient history.

    Parameters
    ----------
    user_ckpt_path   : path to the .pt checkpoint for users
    target_ckpt_path : path to the .pt checkpoint for targets
    area_size        : float — used for domain normalization (required)
    history_length   : L — number of past steps used as input (default 10)
    device           : torch device (auto-detected if None)
    """

    def __init__(
        self,
        user_ckpt_path: str,
        target_ckpt_path: str,
        area_size: float,
        history_length: int = 10,
        device=None,
    ):
        import torch  # local import: only the LSTM path requires torch

        from src.prediction.lstm_model import TrajectoryLSTM
        from src.prediction.normalization import normalize, denormalize

        self._torch = torch

        self._normalize   = normalize
        self._denormalize = denormalize

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device         = device
        self.area_size      = float(area_size)
        self.history_length = int(history_length)

        self.user_model   = self._load(user_ckpt_path,   TrajectoryLSTM)
        self.target_model = self._load(target_ckpt_path, TrajectoryLSTM)

        self._cv = CVPredictor()

    def _load(self, path: str, cls):
        torch = self._torch
        ckpt  = torch.load(path, map_location=self.device)
        pcfg  = ckpt["pred_config"]
        model = cls(
            input_size=2,
            hidden_size=int(pcfg["hidden_size"]),
            num_layers=int(pcfg["num_layers"]),
            dropout=float(pcfg["dropout"]),
            prediction_horizon=int(pcfg["prediction_horizon"]),
        ).to(self.device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model

    def _predict_model(self, model, history_arr: np.ndarray) -> np.ndarray:
        """
        history_arr : (N_obj, L, 2)  in metres
        returns     : (N_obj, H, 2)  in metres
        """
        torch  = self._torch
        h_norm = self._normalize(history_arr.astype(np.float32), self.area_size)
        x      = torch.from_numpy(h_norm).to(self.device)
        with torch.no_grad():
            out = model(x)           # (N_obj, H, 2)  normalised
        return self._denormalize(out.cpu().numpy(), self.area_size)

    def predict(self, sim: MobilitySimulator, horizon: int) -> tuple:
        """
        Returns (users_pred, targets_pred) each shape (N_obj, horizon, 2).
        Falls back to CV if position history is insufficient.
        """
        user_hist   = sim.get_user_history(self.history_length)
        target_hist = sim.get_target_history(self.history_length)

        if user_hist is None or target_hist is None:
            # Not enough history yet — fall back to CV
            return self._cv.predict(sim, horizon)

        return (
            self._predict_model(self.user_model,   user_hist),
            self._predict_model(self.target_model, target_hist),
        )

    def __repr__(self):
        return f"LSTMPredictor(area_size={self.area_size}, L={self.history_length})"


def make_predictor(
    predictor_type: str = "cv",
    user_ckpt_path: str = None,
    target_ckpt_path: str = None,
    area_size: float = None,
    history_length: int = 10,
    device=None,
):
    """
    Factory function — returns a predictor ready to call .predict(sim, horizon).

    Parameters
    ----------
    predictor_type : "cv" or "lstm"
    user_ckpt_path   : required when predictor_type="lstm"
    target_ckpt_path : required when predictor_type="lstm"
    area_size        : required when predictor_type="lstm"
    history_length   : LSTM input window length (default 10)
    device           : torch device (auto-detected if None)
    """
    if predictor_type == "cv":
        return CVPredictor()

    if predictor_type == "lstm":
        if user_ckpt_path is None or target_ckpt_path is None:
            raise ValueError("LSTM predictor requires user_ckpt_path and target_ckpt_path")
        if area_size is None:
            raise ValueError("LSTM predictor requires area_size for domain normalization")
        return LSTMPredictor(
            user_ckpt_path=user_ckpt_path,
            target_ckpt_path=target_ckpt_path,
            area_size=area_size,
            history_length=history_length,
            device=device,
        )

    raise ValueError(f"Unknown predictor_type: {predictor_type!r}. Use 'cv' or 'lstm'.")
