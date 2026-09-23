"""
tests/test_prediction.py

Unit tests for the LSTM prediction pipeline.

Tests
-----
1.  Dataset shape: X (N, L, 2), Y (N, H, 2)
2.  Deterministic dataset generation (same seed → identical data)
3.  Model forward-pass shape: (batch, H, 2)
4.  Prediction output shape from predict_trajectory and predict_all_objects
5.  No NaN / Inf in model output
6.  Deterministic training: same seed → same final loss
7.  Metric correctness on known trajectories
8.  Train / val / test do not overlap (contiguous split)
9.  Prediction horizon is respected (output has exactly H steps)
"""

import unittest
import numpy as np
import torch

from src.prediction.dataset import (
    build_dataset,
    build_mobility_dataset,
    build_sliding_window_samples,
    generate_trajectory,
    load_prediction_config,
    split_samples,
)
from src.prediction.lstm_model import TrajectoryLSTM, build_model_from_config
from src.prediction.predict import predict_trajectory, predict_all_objects, predict_cv
from src.metrics.prediction import mae, rmse, ade, fde, compute_all_metrics


MINI_CFG = {
    "history_length": 5,
    "prediction_horizon": 3,
    "hidden_size": 16,
    "num_layers": 1,
    "dropout": 0.0,
    "learning_rate": 0.001,
    "batch_size": 8,
    "epochs": 2,
    "train_ratio": 0.70,
    "validation_ratio": 0.15,
    "test_ratio": 0.15,
    "seed": 0,
}


class TestDataset(unittest.TestCase):

    def setUp(self):
        self.ds = build_dataset(
            config_path="configs/default.yaml",
            num_steps=300,
            entity="users",
            seed=0,
        )
        self.L = self.ds["metadata"]["history_length"]
        self.H = self.ds["metadata"]["prediction_horizon"]

    # 1. Dataset shapes
    def test_input_shape(self):
        for split in ["X_train", "X_val", "X_test"]:
            self.assertEqual(self.ds[split].ndim, 3)
            self.assertEqual(self.ds[split].shape[1], self.L, f"{split} history axis")
            self.assertEqual(self.ds[split].shape[2], 2,      f"{split} coord axis")

    def test_target_shape(self):
        for split in ["Y_train", "Y_val", "Y_test"]:
            self.assertEqual(self.ds[split].ndim, 3)
            self.assertEqual(self.ds[split].shape[1], self.H, f"{split} horizon axis")
            self.assertEqual(self.ds[split].shape[2], 2,      f"{split} coord axis")

    # 2. Deterministic generation
    def test_deterministic_generation(self):
        ds1 = build_dataset(config_path="configs/default.yaml", num_steps=300, entity="users", seed=7)
        ds2 = build_dataset(config_path="configs/default.yaml", num_steps=300, entity="users", seed=7)
        np.testing.assert_array_equal(ds1["X_train"], ds2["X_train"])
        np.testing.assert_array_equal(ds1["Y_test"],  ds2["Y_test"])

    # 8. No temporal overlap between splits (legacy build_dataset: global split)
    def test_no_overlap(self):
        # Sizes must partition (N_total across all objects in dataset)
        n_train = len(self.ds["X_train"])
        n_val   = len(self.ds["X_val"])
        n_test  = len(self.ds["X_test"])
        self.assertGreater(n_train, 0)
        self.assertGreater(n_val,   0)
        self.assertGreater(n_test,  0)

        # build_dataset() splits AFTER concatenating all objects, so the
        # global partition is contiguous over the concatenated array: the
        # last training row must never reappear in val/test, and vice
        # versa. Verify by direct membership check (rows are float
        # trajectory windows, effectively unique per timestep/object).
        train_set = {tuple(row.ravel()) for row in self.ds["X_train"]}
        val_set   = {tuple(row.ravel()) for row in self.ds["X_val"]}
        test_set  = {tuple(row.ravel()) for row in self.ds["X_test"]}
        self.assertEqual(len(train_set & val_set), 0)
        self.assertEqual(len(train_set & test_set), 0)
        self.assertEqual(len(val_set & test_set), 0)


class TestPerObjectTemporalSplit(unittest.TestCase):
    """
    Regression coverage for the corrected per-object temporal split.

    build_mobility_dataset() must, for EVERY object independently, take the
    first 70% of windows as train, the next 15% as validation, and the
    final 15% as test -- and only THEN concatenate across objects. This
    guards against the original bug where per-object windows were
    concatenated first and split globally afterwards, which could let a
    single object's trajectory dominate one partition (e.g. the test set
    being mostly-or-only the last user/target).
    """

    def setUp(self):
        self.config_path = "configs/default.yaml"
        self.num_steps = 300
        self.seed = 3
        self.entity = "users"

        self.cfg = load_prediction_config(self.config_path)
        import yaml
        with open(self.config_path) as f:
            self.net_cfg = yaml.safe_load(f)["network"]

        self.ds = build_mobility_dataset(
            config_path=self.config_path,
            num_steps=self.num_steps,
            entity=self.entity,
            seed=self.seed,
        )

    def _expected_per_object_split(self):
        L = self.cfg["history_length"]
        H = self.cfg["prediction_horizon"]
        tr = self.cfg["train_ratio"]
        vr = self.cfg["validation_ratio"]

        user_traj, target_traj = generate_trajectory(
            num_steps=self.num_steps,
            num_users=self.net_cfg["num_users"],
            num_targets=self.net_cfg["num_targets"],
            area_size=self.net_cfg["area_size"],
            seed=self.seed,
        )
        traj = user_traj if self.entity == "users" else target_traj

        train_x, val_x, test_x = [], [], []
        for obj_idx in range(traj.shape[1]):
            x_obj, _ = build_sliding_window_samples(traj[:, obj_idx, :], L, H)
            n = len(x_obj)
            n_train = int(n * tr)
            n_val = int(n * vr)
            train_x.append(x_obj[:n_train])
            val_x.append(x_obj[n_train:n_train + n_val])
            test_x.append(x_obj[n_train + n_val:])

        return train_x, val_x, test_x

    def test_matches_independent_per_object_reconstruction(self):
        train_x, val_x, test_x = self._expected_per_object_split()
        np.testing.assert_array_equal(self.ds["X_train"], np.concatenate(train_x, axis=0))
        np.testing.assert_array_equal(self.ds["X_val"],   np.concatenate(val_x,   axis=0))
        np.testing.assert_array_equal(self.ds["X_test"],  np.concatenate(test_x,  axis=0))

    def test_every_object_contributes_to_test_split(self):
        # This is the direct regression check for the discovered bug: under
        # the old global-post-concat split, a single object could dominate
        # (or fully occupy) the test partition. Under the corrected
        # per-object split, every object contributes a (roughly) equal
        # share since all objects share the same trajectory length.
        _, _, test_x = self._expected_per_object_split()
        sizes = [t.shape[0] for t in test_x]
        self.assertTrue(all(s > 0 for s in sizes), "every object must contribute test windows")
        self.assertEqual(len(set(sizes)), 1, "equal-length trajectories must yield equal per-object test shares")

    def test_no_cross_split_row_overlap(self):
        train_set = {tuple(row.ravel()) for row in self.ds["X_train"]}
        val_set   = {tuple(row.ravel()) for row in self.ds["X_val"]}
        test_set  = {tuple(row.ravel()) for row in self.ds["X_test"]}
        self.assertEqual(len(train_set & val_set), 0)
        self.assertEqual(len(train_set & test_set), 0)
        self.assertEqual(len(val_set & test_set), 0)

    def test_backward_compatible_keys_present(self):
        for key in ("X_train", "Y_train", "X_val", "Y_val", "X_test", "Y_test", "metadata"):
            self.assertIn(key, self.ds)


class TestSlidingWindow(unittest.TestCase):

    def test_shapes(self):
        traj = np.random.rand(50, 2).astype(np.float32)
        L, H = 5, 3
        X, Y = build_sliding_window_samples(traj, L, H)
        expected_N = 50 - L - H + 1
        self.assertEqual(X.shape, (expected_N, L, 2))
        self.assertEqual(Y.shape, (expected_N, H, 2))

    # 9. Prediction horizon respected
    def test_horizon_respected(self):
        traj = np.random.rand(30, 2).astype(np.float32)
        for H in [1, 3, 5]:
            _, Y = build_sliding_window_samples(traj, history_length=5, prediction_horizon=H)
            self.assertEqual(Y.shape[1], H)


class TestModel(unittest.TestCase):

    def setUp(self):
        self.model = build_model_from_config(MINI_CFG)
        self.L = MINI_CFG["history_length"]
        self.H = MINI_CFG["prediction_horizon"]

    # 3. Forward-pass shape
    def test_forward_shape(self):
        batch = 8
        x = torch.zeros(batch, self.L, 2)
        out = self.model(x)
        self.assertEqual(out.shape, (batch, self.H, 2))

    # 5. No NaN / Inf
    def test_no_nan_inf(self):
        x = torch.randn(4, self.L, 2)
        out = self.model(x)
        self.assertFalse(torch.any(torch.isnan(out)).item())
        self.assertFalse(torch.any(torch.isinf(out)).item())


class TestPredictionAPI(unittest.TestCase):

    def setUp(self):
        self.model = build_model_from_config(MINI_CFG)
        self.L = MINI_CFG["history_length"]
        self.H = MINI_CFG["prediction_horizon"]

    # 4. Output shape — single object
    def test_predict_trajectory_shape(self):
        history = np.random.rand(self.L, 2).astype(np.float32)
        out = predict_trajectory(self.model, history)
        self.assertEqual(out.shape, (self.H, 2))

    # 4. Output shape — batch
    def test_predict_all_objects_shape(self):
        histories = np.random.rand(5, self.L, 2).astype(np.float32)
        out = predict_all_objects(self.model, histories)
        self.assertEqual(out.shape, (5, self.H, 2))

    # 9. Horizon respected via API
    def test_horizon_via_api(self):
        history = np.zeros((self.L, 2), dtype=np.float32)
        out = predict_trajectory(self.model, history)
        self.assertEqual(out.shape[0], self.H)

    # 5. No NaN / Inf from API
    def test_no_nan_inf_api(self):
        history = np.random.rand(self.L, 2).astype(np.float32)
        out = predict_trajectory(self.model, history)
        self.assertFalse(np.any(np.isnan(out)))
        self.assertFalse(np.any(np.isinf(out)))


class TestMetrics(unittest.TestCase):

    # 7. Metric correctness on known trajectories
    def test_perfect_prediction_gives_zero(self):
        y = np.ones((10, 5, 2))
        self.assertAlmostEqual(mae(y, y),  0.0)
        self.assertAlmostEqual(rmse(y, y), 0.0)
        self.assertAlmostEqual(ade(y, y),  0.0)
        self.assertAlmostEqual(fde(y, y),  0.0)

    def test_constant_offset_mae(self):
        y_true = np.zeros((10, 5, 2))
        y_pred = np.ones((10, 5, 2))
        self.assertAlmostEqual(mae(y_true, y_pred), 1.0)

    def test_ade_known_value(self):
        # Each prediction is 3 units away in x (L2 = 3)
        y_true = np.zeros((4, 5, 2))
        y_pred = y_true.copy()
        y_pred[:, :, 0] = 3.0
        result = ade(y_true, y_pred)
        self.assertAlmostEqual(result, 3.0, places=5)

    def test_fde_only_last_step(self):
        y_true = np.zeros((4, 5, 2))
        y_pred = np.zeros_like(y_true)
        y_pred[:, -1, 0] = 4.0   # only final step has error 4 in x
        result = fde(y_true, y_pred)
        self.assertAlmostEqual(result, 4.0, places=5)


class TestDeterministicTraining(unittest.TestCase):

    # 6. Same seed → same final training loss
    def test_deterministic_loss(self):
        import torch, numpy as np
        from torch.utils.data import DataLoader, TensorDataset
        from src.prediction.lstm_model import build_model_from_config

        def run_one():
            torch.manual_seed(0)
            np.random.seed(0)
            model = build_model_from_config(MINI_CFG)
            X = torch.zeros(16, MINI_CFG["history_length"], 2)
            Y = torch.ones(16,  MINI_CFG["prediction_horizon"], 2)
            opt = torch.optim.Adam(model.parameters(), lr=0.01)
            loss_fn = torch.nn.MSELoss()
            losses = []
            for _ in range(3):
                opt.zero_grad()
                out = model(X)
                l = loss_fn(out, Y)
                l.backward()
                opt.step()
                losses.append(l.item())
            return losses

        l1 = run_one()
        l2 = run_one()
        for a, b in zip(l1, l2):
            self.assertAlmostEqual(a, b, places=5)


if __name__ == "__main__":
    unittest.main()
