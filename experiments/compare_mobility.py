"""
Controlled mobility/predictor comparison.

Four conditions:

A: constant-velocity mobility + CV predictor
B: constant-velocity mobility + LSTM predictor
C: stochastic mobility + CV predictor
D: stochastic mobility + LSTM predictor

The LSTM architecture is unchanged from src/prediction/lstm_model.py.

Outputs:
  results/prediction/mobility_comparison/
"""

import argparse
import copy
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

from src.metrics.prediction import compute_all_metrics
from src.prediction.dataset import build_mobility_dataset
from src.prediction.lstm_model import build_model_from_config
from src.prediction.predict import predict_all_objects
from src.environment.mobility import predict_constant_velocity
from src.environment.mobility_configs import STOCHASTIC_CONFIG


CONFIG_PATH = "configs/default.yaml"
RESULTS_DIR = "results/prediction/mobility_comparison"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def build_cv_predictions(X, horizon, dt):
    """
    Constant-velocity prediction using only the final two observed positions.
    """
    if X.shape[1] < 2:
        raise ValueError("Need at least two history samples for CV prediction.")

    pos = X[:, -1, :]
    vel = (X[:, -1, :] - X[:, -2, :]) / dt

    predictions = []

    for h in range(1, horizon + 1):
        predictions.append(pos + h * vel * dt)

    return np.stack(predictions, axis=1)


def train_lstm(
    dataset,
    prediction_cfg,
    checkpoint_path,
    seed,
    device,
    force=False,
    epochs_override=None,
):
    if os.path.exists(checkpoint_path) and not force:
        ckpt = torch.load(checkpoint_path, map_location=device)
        model = build_model_from_config(prediction_cfg).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        history_path = checkpoint_path.replace(".pt", "_history.csv")
        history = (
            pd.read_csv(history_path)
            if os.path.exists(history_path)
            else pd.DataFrame()
        )

        return model, history

    seed_everything(seed)

    model = build_model_from_config(prediction_cfg).to(device)

    batch_size = int(prediction_cfg["batch_size"])
    epochs = (
        int(epochs_override)
        if epochs_override is not None
        else int(prediction_cfg["epochs"])
    )
    lr = float(prediction_cfg["learning_rate"])

    train_x = torch.from_numpy(
        dataset["X_train"].astype(np.float32)
    )
    train_y = torch.from_numpy(
        dataset["Y_train"].astype(np.float32)
    )

    val_x = torch.from_numpy(
        dataset["X_val"].astype(np.float32)
    )
    val_y = torch.from_numpy(
        dataset["Y_val"].astype(np.float32)
    )

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        TensorDataset(val_x, val_y),
        batch_size=batch_size,
        shuffle=False,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    criterion = torch.nn.MSELoss()

    history_rows = []
    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()

        train_loss_sum = 0.0
        train_count = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)

            pred = model(xb)
            loss = criterion(pred, yb)

            loss.backward()
            optimizer.step()

            n = xb.shape[0]
            train_loss_sum += loss.item() * n
            train_count += n

        model.eval()

        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                pred = model(xb)
                loss = criterion(pred, yb)

                n = xb.shape[0]
                val_loss_sum += loss.item() * n
                val_count += n

        train_loss = train_loss_sum / max(train_count, 1)
        val_loss = val_loss_sum / max(val_count, 1)

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(
                model.state_dict()
            )

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"      epoch {epoch:3d}/{epochs} "
                f"train={train_loss:.4f} "
                f"val={val_loss:.4f}"
            )

    model.load_state_dict(best_state)
    model.eval()

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_val_loss": best_val,
            "prediction_config": prediction_cfg,
        },
        checkpoint_path,
    )

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(
        checkpoint_path.replace(".pt", "_history.csv"),
        index=False,
    )

    return model, history_df


def evaluate_model(model, X_test, device):
    return predict_all_objects(
        model,
        X_test,
        device=device,
    )


def evaluate_condition(
    mobility_name,
    mobility_config,
    entity,
    cfg,
    seed,
    num_steps,
    device,
    force,
    epochs_override,
):
    prediction_cfg = cfg["prediction"]
    simulation_cfg = cfg["simulation"]

    dataset = build_mobility_dataset(
        config_path=CONFIG_PATH,
        num_steps=num_steps,
        entity=entity,
        seed=seed,
        mobility_config=mobility_config,
    )

    X_test = dataset["X_test"]
    Y_test = dataset["Y_test"]

    L = dataset["metadata"]["history_length"]
    H = dataset["metadata"]["prediction_horizon"]
    dt = float(simulation_cfg["dt"])

    print(
        f"    {entity}: "
        f"train={len(dataset['X_train'])} "
        f"val={len(dataset['X_val'])} "
        f"test={len(dataset['X_test'])} "
        f"L={L} H={H}"
    )

    cv_preds = build_cv_predictions(
        X_test,
        horizon=H,
        dt=dt,
    )

    cv_metrics = compute_all_metrics(
        Y_test,
        cv_preds,
        label=f"CV {mobility_name} {entity}",
    )

    ckpt_name = f"{mobility_name}_{entity}.pt"
    ckpt_path = os.path.join(
        RESULTS_DIR,
        "checkpoints",
        ckpt_name,
    )

    model, history = train_lstm(
        dataset=dataset,
        prediction_cfg=prediction_cfg,
        checkpoint_path=ckpt_path,
        seed=seed,
        device=device,
        force=force,
        epochs_override=epochs_override,
    )

    lstm_preds = evaluate_model(
        model,
        X_test,
        device,
    )

    lstm_metrics = compute_all_metrics(
        Y_test,
        lstm_preds,
        label=f"LSTM {mobility_name} {entity}",
    )

    rows = []

    for predictor, metrics in [
        ("Constant-Velocity", cv_metrics),
        ("LSTM", lstm_metrics),
    ]:
        rows.append(
            {
                "mobility": mobility_name,
                "entity": entity,
                "predictor": predictor,
                "MAE_m": metrics["mae"],
                "RMSE_m": metrics["rmse"],
                "ADE_m": metrics["ade"],
                "FDE_m": metrics["fde"],
                "test_samples": len(X_test),
                "seed": seed,
            }
        )

    horizon_rows = []

    for predictor, metrics in [
        ("Constant-Velocity", cv_metrics),
        ("LSTM", lstm_metrics),
    ]:
        for h, error in enumerate(
            metrics["error_vs_horizon"],
            start=1,
        ):
            horizon_rows.append(
                {
                    "mobility": mobility_name,
                    "entity": entity,
                    "predictor": predictor,
                    "horizon_step": h,
                    "mean_displacement_error_m": float(error),
                    "seed": seed,
                }
            )

    return (
        rows,
        horizon_rows,
        dataset,
        cv_preds,
        lstm_preds,
        history,
    )


def plot_horizon_errors(horizon_df):
    for entity in sorted(horizon_df["entity"].unique()):
        sub = horizon_df[
            horizon_df["entity"] == entity
        ]

        fig, ax = plt.subplots(figsize=(8, 5))

        for (mobility, predictor), grp in sub.groupby(
            ["mobility", "predictor"]
        ):
            grp = grp.sort_values("horizon_step")

            ax.plot(
                grp["horizon_step"],
                grp["mean_displacement_error_m"],
                marker="o",
                label=f"{mobility} + {predictor}",
            )

        ax.set_xlabel("Prediction horizon step")
        ax.set_ylabel("Mean displacement error (m)")
        ax.set_title(
            f"Prediction Error vs Horizon — {entity}"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        fig.tight_layout()

        fig.savefig(
            os.path.join(
                RESULTS_DIR,
                f"horizon_error_{entity}.png",
            ),
            dpi=150,
        )

        plt.close(fig)


def plot_ade_fde(summary_df):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
    )

    for ax, metric, title in [
        (axes[0], "ADE_m", "ADE"),
        (axes[1], "FDE_m", "FDE"),
    ]:
        labels = []
        values = []

        for _, row in summary_df.iterrows():
            labels.append(
                f"{row['mobility']}\n"
                f"{row['predictor']}\n"
                f"{row['entity']}"
            )
            values.append(row[metric])

        ax.bar(
            np.arange(len(values)),
            values,
        )
        ax.set_xticks(np.arange(len(values)))
        ax.set_xticklabels(
            labels,
            rotation=55,
            ha="right",
            fontsize=7,
        )
        ax.set_ylabel("metres")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "2×2 Mobility / Predictor Comparison"
    )
    fig.tight_layout()

    fig.savefig(
        os.path.join(
            RESULTS_DIR,
            "ade_fde_comparison.png",
        ),
        dpi=150,
    )

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-steps",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override configured LSTM epochs.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain all LSTM models.",
    )

    args = parser.parse_args()

    cfg = load_config()

    os.makedirs(
        os.path.join(RESULTS_DIR, "checkpoints"),
        exist_ok=True,
    )

    seed_everything(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 80)
    print("CONTROLLED MOBILITY / PREDICTOR EXPERIMENT")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Simulation steps: {args.num_steps}")
    print()

    conditions = [
        ("constant_velocity", None),
        ("stochastic", STOCHASTIC_CONFIG),
    ]

    summary_rows = []
    horizon_rows = []

    for mobility_name, mobility_config in conditions:
        print()
        print("=" * 70)
        print(f"MOBILITY: {mobility_name}")
        print("=" * 70)

        for entity in ["users", "targets"]:
            rows, h_rows, *_ = evaluate_condition(
                mobility_name=mobility_name,
                mobility_config=mobility_config,
                entity=entity,
                cfg=cfg,
                seed=args.seed,
                num_steps=args.num_steps,
                device=device,
                force=args.force,
                epochs_override=args.epochs,
            )

            summary_rows.extend(rows)
            horizon_rows.extend(h_rows)

    summary_df = pd.DataFrame(summary_rows)
    horizon_df = pd.DataFrame(horizon_rows)

    summary_path = os.path.join(
        RESULTS_DIR,
        "summary.csv",
    )
    horizon_path = os.path.join(
        RESULTS_DIR,
        "horizon_errors.csv",
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    horizon_df.to_csv(
        horizon_path,
        index=False,
    )

    plot_horizon_errors(horizon_df)
    plot_ade_fde(summary_df)

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(
        summary_df.to_string(index=False)
    )
    print("=" * 80)
    print(f"Saved: {summary_path}")
    print(f"Saved: {horizon_path}")


if __name__ == "__main__":
    main()
