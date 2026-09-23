"""
src/prediction/train_lstm.py

Train the TrajectoryLSTM using MSE loss.

Guarantees
----------
* Deterministic: torch seed + numpy seed set from config["prediction"]["seed"].
* Train / validation / test split is contiguous (no temporal leakage).
* Training uses only training set; validation loss is evaluated at each epoch
  without gradient updates.
* Model checkpoint saved to results/prediction/lstm_<entity>.pt
* Training history saved to results/prediction/train_history_<entity>.csv
"""

import os
import time
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.prediction.dataset import build_dataset
from src.prediction.lstm_model import build_model_from_config


def set_seeds(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(
    config_path: str = "configs/default.yaml",
    entity: str = "users",
    num_steps: int = 2000,
    checkpoint_dir: str = "results/prediction",
    verbose: bool = True,
    mobility_config: dict = None,
    ckpt_suffix: str = "",
):
    """
    Full training pipeline.

    Parameters
    ----------
    config_path    : YAML config
    entity         : "users" or "targets"
    num_steps      : total simulation steps for trajectory generation
    checkpoint_dir : directory for saved checkpoint and history
    verbose        : print epoch progress
    mobility_config: dict or None — passed to build_dataset for mobility type
    ckpt_suffix    : string appended before .pt in checkpoint filename

    Returns
    -------
    dict with model, train_losses, val_losses, test_loss, history_df, dataset, device
    """
    # ── Load config ──────────────────────────────────────────────────────────
    with open(config_path) as f:
        full_cfg = yaml.safe_load(f)
    pred_cfg = full_cfg["prediction"]

    seed         = int(pred_cfg["seed"])
    lr           = float(pred_cfg["learning_rate"])
    batch_size   = int(pred_cfg["batch_size"])
    epochs       = int(pred_cfg["epochs"])
    H            = int(pred_cfg["prediction_horizon"])

    set_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Dataset ───────────────────────────────────────────────────────────────
    # build_dataset does not yet support mobility_config — use build_mobility_dataset
    # when mobility_config is supplied, otherwise fall back to build_dataset
    if mobility_config is not None:
        from src.prediction.dataset import build_mobility_dataset
        ds = build_mobility_dataset(
            config_path=config_path,
            num_steps=num_steps,
            entity=entity,
            seed=seed,
            mobility_config=mobility_config,
        )
    else:
        ds = build_dataset(config_path=config_path, num_steps=num_steps,
                           entity=entity, seed=seed)

    def to_tensor(arr):
        return torch.from_numpy(arr).float().to(device)

    X_train, Y_train = to_tensor(ds["X_train"]), to_tensor(ds["Y_train"])
    X_val,   Y_val   = to_tensor(ds["X_val"]),   to_tensor(ds["Y_val"])
    X_test,  Y_test  = to_tensor(ds["X_test"]),  to_tensor(ds["Y_test"])

    train_loader = DataLoader(
        TensorDataset(X_train, Y_train),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model_from_config(pred_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()

    # ── Training loop ─────────────────────────────────────────────────────────
    train_losses = []
    val_losses   = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = loss_fn(pred, Y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)

        train_loss = running_loss / len(X_train)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = loss_fn(val_pred, Y_val).item()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"Epoch {epoch:3d}/{epochs}  train_MSE={train_loss:.4f}  val_MSE={val_loss:.4f}")

    # ── Test evaluation ───────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test)
        test_loss = loss_fn(test_pred, Y_test).item()

    if verbose:
        print(f"\nTest MSE: {test_loss:.4f}")

    # ── Save checkpoint + history ─────────────────────────────────────────────
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"lstm_{entity}{ckpt_suffix}.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "pred_config":      pred_cfg,
        "entity":           entity,
        "train_loss":       train_losses[-1],
        "val_loss":         val_losses[-1],
        "test_loss":        test_loss,
    }, ckpt_path)

    history_df = pd.DataFrame({
        "epoch":      list(range(1, epochs + 1)),
        "train_loss": train_losses,
        "val_loss":   val_losses,
    })
    history_df.to_csv(
        os.path.join(checkpoint_dir, f"train_history_{entity}{ckpt_suffix}.csv"),
        index=False,
    )

    if verbose:
        print(f"Checkpoint saved: {ckpt_path}")

    return {
        "model":        model,
        "train_losses": train_losses,
        "val_losses":   val_losses,
        "test_loss":    test_loss,
        "history_df":   history_df,
        "dataset":      ds,
        "device":       device,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity",    default="users",    choices=["users", "targets"])
    parser.add_argument("--steps",     default=2000, type=int)
    parser.add_argument("--config",    default="configs/default.yaml")
    args = parser.parse_args()
    train(config_path=args.config, entity=args.entity, num_steps=args.steps)
