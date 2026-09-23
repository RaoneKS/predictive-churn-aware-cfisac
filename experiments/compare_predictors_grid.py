"""
Controlled 2x2 mobility x predictor experiment.

A = CV mobility + CV predictor
B = CV mobility + LSTM predictor
C = stochastic mobility + CV predictor
D = stochastic mobility + LSTM predictor

Uses build_mobility_dataset(), which performs a per-object temporal split.
The existing LSTM architecture is unchanged.
"""

import argparse
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

from src.environment.mobility import predict_constant_velocity
from src.environment.mobility_configs import STOCHASTIC_CONFIG
from src.metrics.prediction import compute_all_metrics
from src.prediction.dataset import build_mobility_dataset
from src.prediction.lstm_model import build_model_from_config
from src.prediction.normalization import normalize, denormalize
from src.prediction.predict import predict_all_objects


CONFIG_PATH = "configs/default.yaml"
RESULTS_DIR = "results/prediction"

DEFAULT_STEPS = 2000
DEFAULT_SEED = 42


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_cv_predictions(X_test, horizon, dt):
    pos = X_test[:, -1, :]
    vel = (X_test[:, -1, :] - X_test[:, -2, :]) / dt

    return np.stack(
        [
            pos + h * vel * dt
            for h in range(1, horizon + 1)
        ],
        axis=1,
    )


def train_lstm_for_dataset(
    dataset,
    pred_cfg,
    seed,
    device,
    checkpoint_path,
    area_size,
    force=False,
):
    """
    Train the existing LSTM on a dataset.

    Normalization
    -------------
    X_train, Y_train, X_val, Y_val are normalized to [-1, +1] using
    domain normalization (area_size/2 centre, area_size/2 scale).
    Test data is kept in metres; de-normalization happens at inference time
    via predict_all_objects(area_size=area_size).

    Early stopping
    --------------
    Training stops when val loss has not improved for `early_stopping_patience`
    epochs.  The model state at the best validation epoch is restored before
    returning.
    """
    if os.path.exists(checkpoint_path) and not force:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )
        model = build_model_from_config(pred_cfg).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model, pd.DataFrame()

    set_seed(seed)

    patience = int(pred_cfg.get("early_stopping_patience", 20))

    # Normalize train and val data; test data stays in metres
    X_train_n = normalize(dataset["X_train"].astype(np.float32), area_size)
    Y_train_n = normalize(dataset["Y_train"].astype(np.float32), area_size)
    X_val_n   = normalize(dataset["X_val"].astype(np.float32),   area_size)
    Y_val_n   = normalize(dataset["Y_val"].astype(np.float32),   area_size)

    X_train_t = torch.from_numpy(X_train_n)
    Y_train_t = torch.from_numpy(Y_train_n)
    X_val_t   = torch.from_numpy(X_val_n)
    Y_val_t   = torch.from_numpy(Y_val_n)

    model = build_model_from_config(pred_cfg).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(pred_cfg["learning_rate"]),
    )
    loss_fn = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(X_train_t, Y_train_t),
        batch_size=int(pred_cfg["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    epochs = int(pred_cfg["epochs"])

    history = []
    best_val   = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * xb.shape[0]

        train_loss = total / len(X_train_t)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t.to(device)), Y_val_t.to(device)).item()

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    Early stopping at epoch {epoch} (patience={patience})")
                break

    # Restore best validation checkpoint
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "pred_config":      pred_cfg,
            "mobility_type":    dataset["metadata"]["mobility_type"],
            "seed":             seed,
            "best_val_loss":    best_val,
            "area_size":        area_size,   # stored so loading code can denormalize
        },
        checkpoint_path,
    )

    history_df = pd.DataFrame(history)
    history_df.to_csv(checkpoint_path.replace(".pt", "_history.csv"), index=False)

    print(
        f"    Trained {len(history)} epochs | "
        f"best_val={best_val:.4f} (normalized MSE)"
    )
    return model, history_df


def evaluate_entity(
    entity,
    mobility_name,
    mobility_config,
    cfg,
    seed,
    num_steps,
    device,
    force,
):
    pred_cfg = cfg["prediction"]
    dt = float(cfg["simulation"]["dt"])

    dataset = build_mobility_dataset(
        config_path=CONFIG_PATH,
        num_steps=num_steps,
        entity=entity,
        seed=seed,
        mobility_config=mobility_config,
    )

    X_test = dataset["X_test"]   # metres — kept unnormalized for metric calculation
    Y_test = dataset["Y_test"]   # metres

    horizon = int(dataset["metadata"]["prediction_horizon"])

    # area_size is needed for domain normalization
    area_size = float(dataset["metadata"]["area_size"])

    # ---------------- CV predictor ----------------
    cv_preds = build_cv_predictions(X_test, horizon, dt)

    cv_metrics = compute_all_metrics(
        Y_test, cv_preds,
        label=f"CV/{mobility_name}/{entity}",
    )

    # ---------------- LSTM predictor ----------------
    suffix = "" if mobility_config is None else "_stochastic"
    checkpoint_path = os.path.join(RESULTS_DIR, f"lstm_{entity}{suffix}.pt")

    model, history = train_lstm_for_dataset(
        dataset=dataset,
        pred_cfg=pred_cfg,
        seed=seed,
        device=device,
        checkpoint_path=checkpoint_path,
        area_size=area_size,   # normalization uses area_size only, not train stats
        force=force,
    )

    # Inference: input X_test is in metres; area_size triggers normalize→infer→denormalize
    lstm_preds = predict_all_objects(
        model, X_test, device=device, area_size=area_size,
    )

    lstm_metrics = compute_all_metrics(
        Y_test, lstm_preds,
        label=f"LSTM/{mobility_name}/{entity}",
    )

    summary = []
    for predictor, metrics in [("CV", cv_metrics), ("LSTM", lstm_metrics)]:
        summary.append(
            {
                "entity":        entity,
                "mobility":      mobility_name,
                "predictor":     predictor,
                "MAE (m)":       metrics["mae"],
                "RMSE (m)":      metrics["rmse"],
                "ADE (m)":       metrics["ade"],
                "FDE (m)":       metrics["fde"],
                "seed":          seed,
                "test_samples":  len(X_test),
            }
        )

    horizon_rows = []
    for predictor, metrics in [("CV", cv_metrics), ("LSTM", lstm_metrics)]:
        for step, error in enumerate(metrics["error_vs_horizon"], start=1):
            horizon_rows.append(
                {
                    "entity":       entity,
                    "mobility":     mobility_name,
                    "predictor":    predictor,
                    "horizon_step": step,
                    "error_m":      float(error),
                    "seed":         seed,
                }
            )

    return summary, horizon_rows, history


def plot_horizon(df, entity):
    subset = df[
        df["entity"] == entity
    ]

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    for (
        mobility,
        predictor,
    ), group in subset.groupby(
        ["mobility", "predictor"]
    ):
        group = group.sort_values(
            "horizon_step"
        )

        ax.plot(
            group["horizon_step"],
            group["error_m"],
            marker="o",
            label=f"{mobility} / {predictor}",
        )

    ax.set_xlabel(
        "Prediction horizon step"
    )
    ax.set_ylabel(
        "Mean displacement error (m)"
    )
    ax.set_title(
        f"Prediction Error vs Horizon — {entity}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            RESULTS_DIR,
            f"grid_error_vs_horizon_{entity}.png",
        ),
        dpi=150,
    )

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-steps",
        type=int,
        default=DEFAULT_STEPS,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    args = parser.parse_args()

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    cfg = load_config()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 80)
    print("2x2 MOBILITY x PREDICTOR EXPERIMENT")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Steps: {args.num_steps}")

    conditions = [
        (
            "CV mobility",
            None,
        ),
        (
            "Stochastic mobility",
            STOCHASTIC_CONFIG,
        ),
    ]

    summary_rows = []
    horizon_rows = []

    for entity in [
        "users",
        "targets",
    ]:
        for (
            mobility_name,
            mobility_config,
        ) in conditions:
            print()
            print(
                f"=== {entity.upper()} / "
                f"{mobility_name} ==="
            )

            summary, horizon, _ = evaluate_entity(
                entity=entity,
                mobility_name=mobility_name,
                mobility_config=mobility_config,
                cfg=cfg,
                seed=args.seed,
                num_steps=args.num_steps,
                device=device,
                force=args.force,
            )

            summary_rows.extend(summary)
            horizon_rows.extend(horizon)

    summary_df = pd.DataFrame(
        summary_rows
    )
    horizon_df = pd.DataFrame(
        horizon_rows
    )

    summary_path = os.path.join(
        RESULTS_DIR,
        "grid_2x2_summary.csv",
    )

    horizon_path = os.path.join(
        RESULTS_DIR,
        "grid_2x2_horizon_errors.csv",
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    horizon_df.to_csv(
        horizon_path,
        index=False,
    )

    for entity in [
        "users",
        "targets",
    ]:
        plot_horizon(
            horizon_df,
            entity,
        )

    print()
    print("=" * 80)
    print("2x2 SUMMARY")
    print("=" * 80)
    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved: {summary_path}"
    )
    print(
        f"Saved: {horizon_path}"
    )


if __name__ == "__main__":
    main()
