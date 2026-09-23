"""
experiments/compare_predictors.py

Compare constant-velocity vs LSTM trajectory prediction on the same test set.

Evaluated separately for:
  - communication users
  - sensing targets

Metrics (all in metres):
  MAE, RMSE, ADE, FDE, error vs horizon

Plots saved to results/prediction/
"""

import os
import yaml
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.prediction.dataset import build_dataset, generate_trajectory
from src.prediction.lstm_model import TrajectoryLSTM, build_model_from_config
from src.prediction.predict import predict_cv, predict_all_objects, load_lstm
from src.metrics.prediction import compute_all_metrics, error_vs_horizon
from src.simulation.mobility_simulator import MobilitySimulator
from src.environment.mobility import predict_constant_velocity


CONFIG_PATH   = "configs/default.yaml"
RESULTS_DIR   = "results/prediction"
CKPT_USERS    = "results/prediction/lstm_users.pt"
CKPT_TARGETS  = "results/prediction/lstm_targets.pt"
NUM_STEPS     = 2000
SEED          = 42


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_cv_predictions_for_test(ds, entity, cfg, net_cfg):
    """
    Reproduce constant-velocity predictions for the test split.

    The CV predictor uses only the last two positions to estimate velocity,
    matching what MobilitySimulator.predict_users() does (current pos + velocity).
    Here we back-estimate velocity from the last two history steps.
    """
    L = int(cfg["prediction"]["history_length"])
    H = int(cfg["prediction"]["prediction_horizon"])
    dt = net_cfg["simulation"]["dt"] if "simulation" in net_cfg else 1.0

    X_test = ds["X_test"]   # (N, L, 2)
    Y_test = ds["Y_test"]   # (N, H, 2)

    cv_preds = []
    for i in range(len(X_test)):
        hist = X_test[i]                        # (L, 2)
        pos  = hist[-1]                          # current position
        vel  = (hist[-1] - hist[-2]) / dt       # estimated from last two steps
        future = predict_constant_velocity(pos, vel, H, dt)   # (H, 2)
        cv_preds.append(future)

    return np.asarray(cv_preds), Y_test         # (N, H, 2), (N, H, 2)


def plot_training_history(entity: str):
    history_path = os.path.join(RESULTS_DIR, f"train_history_{entity}.csv")
    if not os.path.exists(history_path):
        print(f"  [skip] no history file for {entity}")
        return
    df = pd.read_csv(history_path)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["epoch"], df["train_loss"], label="train MSE")
    ax.plot(df["epoch"], df["val_loss"],   label="val MSE",   linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (m²)")
    ax.set_title(f"LSTM Training History — {entity}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, f"training_loss_{entity}.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_error_vs_horizon(h_cv, h_lstm, entity: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    H = len(h_cv)
    steps = np.arange(1, H + 1)
    ax.plot(steps, h_cv,   marker="o", label="Constant Velocity")
    ax.plot(steps, h_lstm, marker="s", label="LSTM")
    ax.set_xlabel("Prediction horizon step")
    ax.set_ylabel("Mean displacement error (m)")
    ax.set_title(f"Error vs Horizon — {entity}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, f"error_vs_horizon_{entity}.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_ade_fde_comparison(rows: list):
    """rows: list of dicts with keys predictor, entity, ade, fde"""
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ["ade", "fde"]):
        for entity in df["entity"].unique():
            sub = df[df["entity"] == entity]
            ax.bar(
                [f"{r['predictor']}\n({entity})" for _, r in sub.iterrows()],
                sub[metric],
                label=entity,
            )
        ax.set_title(metric.upper())
        ax.set_ylabel("metres")
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("ADE / FDE Comparison")
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "ade_fde_comparison.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_sample_trajectories(ds, cv_preds, lstm_preds, entity: str, n_samples: int = 3):
    X_test = ds["X_test"]   # (N, L, 2)
    Y_test = ds["Y_test"]   # (N, H, 2)
    n_samples = min(n_samples, len(X_test))

    fig, axes = plt.subplots(1, n_samples, figsize=(5 * n_samples, 4))
    if n_samples == 1:
        axes = [axes]

    for idx, ax in enumerate(axes):
        hist  = X_test[idx]       # (L, 2)
        truth = Y_test[idx]       # (H, 2)
        cv    = cv_preds[idx]     # (H, 2)
        lstm  = lstm_preds[idx]   # (H, 2)

        ax.plot(hist[:, 0],   hist[:, 1],   "k.-",  label="History",    alpha=0.6)
        ax.plot(truth[:, 0],  truth[:, 1],  "g-o",  label="Ground Truth")
        ax.plot(cv[:, 0],     cv[:, 1],     "b--s", label="Const. Vel.")
        ax.plot(lstm[:, 0],   lstm[:, 1],   "r--^", label="LSTM")
        ax.set_title(f"Sample {idx+1}")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Trajectory Comparison — {entity}")
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, f"trajectory_comparison_{entity}.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out}")


def evaluate_entity(entity: str, cfg: dict, ckpt_path: str, ade_fde_rows: list):
    print(f"\n{'='*60}")
    print(f"  Entity: {entity.upper()}")
    print(f"{'='*60}")

    pred_cfg = cfg["prediction"]
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Dataset (same seed as training) ──────────────────────────────────────
    ds = build_dataset(
        config_path=CONFIG_PATH,
        num_steps=NUM_STEPS,
        entity=entity,
        seed=SEED,
    )
    L = ds["metadata"]["history_length"]
    H = ds["metadata"]["prediction_horizon"]
    N_test = len(ds["X_test"])
    print(f"  Test set: {N_test} samples  (L={L}, H={H})")

    # ── Constant-velocity predictions ─────────────────────────────────────────
    cv_preds, Y_test = get_cv_predictions_for_test(ds, entity, cfg, cfg)
    cv_metrics       = compute_all_metrics(Y_test, cv_preds, label=f"CV  {entity}")

    # ── LSTM predictions ──────────────────────────────────────────────────────
    if not os.path.exists(ckpt_path):
        print(f"  [WARNING] Checkpoint not found: {ckpt_path}. Skipping LSTM.")
        return cv_metrics, None, None, None

    model      = load_lstm(ckpt_path, TrajectoryLSTM, pred_cfg, device=device)
    X_test_np  = ds["X_test"]                    # (N, L, 2)
    lstm_preds = []
    for i in range(len(X_test_np)):
        histories = X_test_np[i : i + 1]         # (1, L, 2)
        out = predict_all_objects(model, histories, device=device)
        lstm_preds.append(out[0])
    lstm_preds = np.asarray(lstm_preds)           # (N, H, 2)

    lstm_metrics = compute_all_metrics(Y_test, lstm_preds, label=f"LSTM {entity}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_training_history(entity)
    plot_error_vs_horizon(
        cv_metrics["error_vs_horizon"],
        lstm_metrics["error_vs_horizon"],
        entity,
    )
    plot_sample_trajectories(ds, cv_preds, lstm_preds, entity)

    for predictor, m in [("CV", cv_metrics), ("LSTM", lstm_metrics)]:
        ade_fde_rows.append({"predictor": predictor, "entity": entity,
                             "ade": m["ade"], "fde": m["fde"]})

    return cv_metrics, lstm_metrics, cv_preds, lstm_preds


def main():
    cfg = load_config()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    ade_fde_rows = []
    summary_rows = []

    for entity, ckpt in [("users", CKPT_USERS), ("targets", CKPT_TARGETS)]:
        cv_m, lstm_m, cv_preds, lstm_preds = evaluate_entity(entity, cfg, ckpt, ade_fde_rows)

        for predictor, m in [("Constant-Velocity", cv_m), ("LSTM", lstm_m)]:
            if m is None:
                continue
            summary_rows.append({
                "entity": entity, "predictor": predictor,
                "MAE (m)":  m["mae"],
                "RMSE (m)": m["rmse"],
                "ADE (m)":  m["ade"],
                "FDE (m)":  m["fde"],
            })

    # ── ADE/FDE bar chart ─────────────────────────────────────────────────────
    if ade_fde_rows:
        plot_ade_fde_comparison(ade_fde_rows)

    # ── Summary table ─────────────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "predictor_comparison.csv"), index=False)

    print("\n" + "="*70)
    print("PREDICTOR COMPARISON SUMMARY")
    print("="*70)
    print(summary_df.to_string(index=False))
    print("="*70)
    print("\nNote: all errors are in metres.  Lower is better.")
    print("These results characterise predictor accuracy independently.")
    print("LSTM has NOT been integrated into the four-policy simulation yet.")


if __name__ == "__main__":
    main()
