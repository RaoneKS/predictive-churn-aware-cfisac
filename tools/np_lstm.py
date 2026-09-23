"""
tools/np_lstm.py

AUDIT HARNESS -- NOT part of src/.

Loads a PyTorch TrajectoryLSTM checkpoint and runs its forward pass using only
numpy, so the LSTM conditions of Phase 3 can be executed in environments where
torch cannot be installed.

This is a faithful re-implementation of:
    nn.LSTM(input_size=2, hidden_size=H, num_layers=L, batch_first=True)
    -> take last timestep hidden state
    -> nn.Linear(H, horizon*2)
    -> reshape (batch, horizon, 2)

Dropout is inactive because the reference model is used in eval() mode.

It MUST be validated against torch-produced reference metrics before use
(see tools/validate_np_lstm.py).  On a machine with torch installed, prefer
src.prediction.predictor_interface.LSTMPredictor -- this module exists purely
so that the corrected Phase-3 experiment can be executed here.
"""

import pickle
import zipfile
from collections import OrderedDict

import numpy as np

# torch storage class name -> numpy dtype
_DTYPES = {
    "FloatStorage": np.dtype("<f4"),
    "DoubleStorage": np.dtype("<f8"),
    "HalfStorage": np.dtype("<f2"),
    "LongStorage": np.dtype("<i8"),
    "IntStorage": np.dtype("<i4"),
    "ShortStorage": np.dtype("<i2"),
    "CharStorage": np.dtype("<i1"),
    "ByteStorage": np.dtype("<u1"),
    "BoolStorage": np.dtype("?"),
}


class _StorageTag:
    def __init__(self, name):
        self.name = name


class _TorchZipUnpickler(pickle.Unpickler):
    """Minimal unpickler for the torch zipfile serialization format."""

    def __init__(self, file, zf, prefix):
        super().__init__(file, encoding="utf-8")
        self._zf = zf
        self._prefix = prefix

    def find_class(self, module, name):
        if module.startswith("torch") and name.endswith("Storage"):
            return _StorageTag(name)
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return self._rebuild_tensor_v2
        if module == "torch._utils" and name == "_rebuild_parameter":
            return lambda data, requires_grad, hooks: data
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        return super().find_class(module, name)

    def persistent_load(self, pid):
        # ('storage', storage_tag, key, location, numel)
        _, storage_tag, key, _location, numel = pid
        name = storage_tag.name if isinstance(storage_tag, _StorageTag) else str(storage_tag)
        dtype = _DTYPES[name]
        raw = self._zf.read(f"{self._prefix}/data/{key}")
        return np.frombuffer(raw, dtype=dtype, count=int(numel))

    @staticmethod
    def _rebuild_tensor_v2(storage, storage_offset, size, stride,
                           requires_grad=False, backward_hooks=None,
                           metadata=None):
        size = tuple(int(s) for s in size)
        if len(size) == 0:
            return np.array(storage[storage_offset])
        stride = tuple(int(s) for s in stride)
        itemsize = storage.dtype.itemsize
        return np.lib.stride_tricks.as_strided(
            storage[storage_offset:],
            shape=size,
            strides=tuple(s * itemsize for s in stride),
        ).copy()


def load_checkpoint(path):
    """Return (state_dict of numpy arrays, pred_config dict)."""
    with zipfile.ZipFile(path) as zf:
        prefix = zf.namelist()[0].split("/")[0]
        with zf.open(f"{prefix}/data.pkl") as f:
            obj = _TorchZipUnpickler(f, zf, prefix).load()
    return obj["model_state_dict"], obj["pred_config"]


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class NumpyTrajectoryLSTM:
    """Numpy forward pass matching src.prediction.lstm_model.TrajectoryLSTM."""

    def __init__(self, state_dict, pred_config):
        self.hidden_size = int(pred_config["hidden_size"])
        self.num_layers = int(pred_config["num_layers"])
        self.horizon = int(pred_config["prediction_horizon"])
        sd = {k: np.asarray(v, dtype=np.float64) for k, v in state_dict.items()}
        self.layers = []
        for l in range(self.num_layers):
            self.layers.append((
                sd[f"lstm.weight_ih_l{l}"],   # (4H, in)
                sd[f"lstm.weight_hh_l{l}"],   # (4H, H)
                sd[f"lstm.bias_ih_l{l}"],     # (4H,)
                sd[f"lstm.bias_hh_l{l}"],     # (4H,)
            ))
        self.w_out = sd["output_proj.weight"]  # (H*2, hidden)
        self.b_out = sd["output_proj.bias"]    # (H*2,)

    def forward(self, x):
        """x: (batch, seq, 2) -> (batch, horizon, 2)"""
        x = np.asarray(x, dtype=np.float64)
        B, T, _ = x.shape
        H = self.hidden_size
        inp = x
        for (w_ih, w_hh, b_ih, b_hh) in self.layers:
            h = np.zeros((B, H))
            c = np.zeros((B, H))
            outs = []
            # Pre-compute input projections for the whole sequence.
            gates_x = inp @ w_ih.T + (b_ih + b_hh)      # (B, T, 4H)
            for t in range(T):
                g = gates_x[:, t, :] + h @ w_hh.T       # (B, 4H)
                # PyTorch gate order: input, forget, cell, output
                i = _sigmoid(g[:, 0:H])
                f = _sigmoid(g[:, H:2 * H])
                gg = np.tanh(g[:, 2 * H:3 * H])
                o = _sigmoid(g[:, 3 * H:4 * H])
                c = f * c + i * gg
                h = o * np.tanh(c)
                outs.append(h)
            inp = np.stack(outs, axis=1)                # (B, T, H)
        last_hidden = inp[:, -1, :]                     # (B, H)
        flat = last_hidden @ self.w_out.T + self.b_out  # (B, horizon*2)
        return flat.reshape(B, self.horizon, 2)


class NumpyLSTMPredictor:
    """
    Drop-in replacement for src.prediction.predictor_interface.LSTMPredictor.

    Identical semantics, including the CV fallback when the simulator does not
    yet hold `history_length` steps of position history.
    """

    def __init__(self, user_ckpt_path, target_ckpt_path, area_size,
                 history_length=10):
        from src.prediction.normalization import normalize, denormalize
        self._normalize = normalize
        self._denormalize = denormalize
        self.area_size = float(area_size)
        self.history_length = int(history_length)
        self.user_model = NumpyTrajectoryLSTM(*load_checkpoint(user_ckpt_path))
        self.target_model = NumpyTrajectoryLSTM(*load_checkpoint(target_ckpt_path))

    def _run(self, model, history_arr):
        h_norm = self._normalize(np.asarray(history_arr, dtype=np.float32), self.area_size)
        out = model.forward(h_norm)
        return self._denormalize(out, self.area_size)

    def predict(self, sim, horizon):
        user_hist = sim.get_user_history(self.history_length)
        target_hist = sim.get_target_history(self.history_length)
        if user_hist is None or target_hist is None:
            # Insufficient history -- CV fallback, exactly as LSTMPredictor does.
            return sim.predict_users(horizon), sim.predict_targets(horizon)
        return (
            self._run(self.user_model, user_hist),
            self._run(self.target_model, target_hist),
        )

    def __repr__(self):
        return f"NumpyLSTMPredictor(area_size={self.area_size}, L={self.history_length})"
