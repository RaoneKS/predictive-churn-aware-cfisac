"""
tools/validate_np_lstm.py

Validation gate for tools/np_lstm.py.

Rebuilds the exact test split used by experiments/compare_predictors_grid.py
(which is pure numpy) and recomputes ADE/FDE/MAE/RMSE and error-vs-horizon for
the stored LSTM checkpoints using the numpy forward pass.  These are compared
against results/prediction/grid_2x2_summary.csv and
grid_2x2_horizon_errors.csv, which were produced on a machine WITH torch.

If the numbers match, the numpy implementation is a faithful substitute.
"""

import sys

import numpy as np
import pandas as pd

from src.environment.mobility_configs import STOCHASTIC_CONFIG
from src.metrics.prediction import compute_all_metrics
from src.prediction.dataset import build_mobility_dataset
from tools.np_lstm import NumpyTrajectoryLSTM, load_checkpoint

CONFIG_PATH = "configs/default.yaml"
NUM_STEPS = 2000
SEED = 42

ref_summary = pd.read_csv("results/prediction/grid_2x2_summary.csv")
ref_horizon = pd.read_csv("results/prediction/grid_2x2_horizon_errors.csv")

rows = []
ok = True

for entity in ["users", "targets"]:
    for mob_name, mob_cfg, suffix in [
        ("CV mobility", None, ""),
        ("Stochastic mobility", STOCHASTIC_CONFIG, "_stochastic"),
    ]:
        ds = build_mobility_dataset(
            config_path=CONFIG_PATH, num_steps=NUM_STEPS, entity=entity,
            seed=SEED, mobility_config=mob_cfg,
        )
        X_test, Y_test = ds["X_test"], ds["Y_test"]
        area = float(ds["metadata"]["area_size"])

        ckpt = f"results/prediction/lstm_{entity}{suffix}.pt"
        model = NumpyTrajectoryLSTM(*load_checkpoint(ckpt))

        from src.prediction.normalization import normalize, denormalize
        preds = denormalize(
            model.forward(normalize(X_test.astype(np.float32), area)), area
        )

        m = compute_all_metrics(Y_test, preds, label="np")

        sel = (ref_summary.entity == entity) & (ref_summary.mobility == mob_name) \
            & (ref_summary.predictor == "LSTM")
        r = ref_summary[sel].iloc[0]

        for key, refval in [("ade", r["ADE (m)"]), ("fde", r["FDE (m)"]),
                            ("mae", r["MAE (m)"]), ("rmse", r["RMSE (m)"])]:
            got = float(m[key])
            match = np.isclose(got, float(refval), rtol=1e-3, atol=1e-3)
            ok &= match
            rows.append({
                "entity": entity, "mobility": mob_name, "metric": key.upper(),
                "torch_reference": float(refval), "numpy_reimpl": got,
                "abs_diff": abs(got - float(refval)), "match": match,
            })

        hsel = (ref_horizon.entity == entity) & (ref_horizon.mobility == mob_name) \
            & (ref_horizon.predictor == "LSTM")
        href = ref_horizon[hsel].sort_values("horizon_step")["error_m"].values
        hgot = np.asarray(m["error_vs_horizon"], dtype=float)
        hmatch = np.allclose(hgot, href, rtol=1e-3, atol=1e-3)
        ok &= hmatch
        rows.append({
            "entity": entity, "mobility": mob_name, "metric": "err_vs_horizon(max abs diff)",
            "torch_reference": float(href[-1]), "numpy_reimpl": float(hgot[-1]),
            "abs_diff": float(np.max(np.abs(hgot - href))), "match": hmatch,
        })

df = pd.DataFrame(rows)
pd.set_option("display.width", 160)
print(df.to_string(index=False))
print()
print("VALIDATION:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
