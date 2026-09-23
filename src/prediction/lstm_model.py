"""
src/prediction/lstm_model.py

PyTorch LSTM model for trajectory prediction.

Architecture
------------
Input  : (batch, history_length, 2)   -- [x, y] positions at each past step
Output : (batch, prediction_horizon, 2) -- [x, y] positions at each future step

Design decisions
----------------
* Encoder-decoder style: LSTM encodes the history; a linear projection maps the
  final hidden state to all H future steps simultaneously (direct multi-step).
* Dropout is applied between stacked LSTM layers (not on the last layer output).
* Completely independent of CF-ISAC clustering / channel code.
"""

import torch
import torch.nn as nn


class TrajectoryLSTM(nn.Module):
    """
    Parameters
    ----------
    input_size         : 2  ([x, y])
    hidden_size        : LSTM hidden units  (from config)
    num_layers         : stacked LSTM depth (from config)
    dropout            : inter-layer dropout probability
    prediction_horizon : H — number of future steps to predict
    """

    def __init__(
        self,
        input_size: int = 2,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        prediction_horizon: int = 5,
    ):
        super().__init__()
        self.hidden_size        = hidden_size
        self.num_layers         = num_layers
        self.prediction_horizon = prediction_horizon

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,               # (batch, seq, features)
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Project final hidden state → H future 2-D positions
        self.output_proj = nn.Linear(hidden_size, prediction_horizon * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, history_length, 2)

        Returns
        -------
        pred : (batch, prediction_horizon, 2)
        """
        # lstm_out: (batch, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)

        # Use only the last hidden state to predict all future steps
        last_hidden = lstm_out[:, -1, :]               # (batch, hidden_size)
        flat_pred   = self.output_proj(last_hidden)    # (batch, H*2)

        batch = x.size(0)
        pred = flat_pred.view(batch, self.prediction_horizon, 2)
        return pred


def build_model_from_config(prediction_config: dict) -> TrajectoryLSTM:
    """
    Instantiate a TrajectoryLSTM using values from the 'prediction' YAML block.
    """
    return TrajectoryLSTM(
        input_size=2,
        hidden_size=int(prediction_config["hidden_size"]),
        num_layers=int(prediction_config["num_layers"]),
        dropout=float(prediction_config["dropout"]),
        prediction_horizon=int(prediction_config["prediction_horizon"]),
    )
