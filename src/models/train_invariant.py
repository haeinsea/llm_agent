from __future__ import annotations

import json
from pathlib import Path

import argparse
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.temporal_backbone import build_temporal_model, temporal_model_display_name
from src.models.train_tcn import (
    best_threshold_by_f1,
    ensure_dir,
    flattened_to_tensor,
    load_yaml,
    phase_recall,
    print_window_stats,
    read_windows,
    set_seed,
    transform_with_imputer_scaler,
)
from src.utils.device import torch_device_info


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
CONFIG_DIR = PROJECT_ROOT / "configs"
MODEL_PREFIX = "invariant_tcn"


def phase_to_env(phase: str) -> int:
    phase = str(phase)
    if phase in {"normal", "pre"}:
        return 0
    if phase == "transition":
        return 1
    return 2


def _coral_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.size(0) < 2 or y.size(0) < 2:
        return x.new_tensor(0.0)
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    cov_x = (x.T @ x) / max(x.size(0) - 1, 1)
    cov_y = (y.T @ y) / max(y.size(0) - 1, 1)
    mean_gap = (x.mean(dim=0) - y.mean(dim=0)).pow(2).mean()
    cov_gap = (cov_x - cov_y).pow(2).mean()
    return mean_gap + cov_gap


def invariant_penalty(features: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
    unique_envs = [int(v) for v in torch.unique(env_ids).tolist()]
    losses = []
    for i, env_i in enumerate(unique_envs):
        feat_i = features[env_ids == env_i]
        for env_j in unique_envs[i + 1:]:
            feat_j = features[env_ids == env_j]
            if feat_i.size(0) < 2 or feat_j.size(0) < 2:
                continue
            losses.append(_coral_loss(feat_i, feat_j))
    if not losses:
        return features.new_tensor(0.0)
    return torch.stack(losses).mean()


def main(seed: int | None = None) -> None:
    ensure_dir(MODEL_DIR)
    cfg = load_yaml(
        CONFIG_DIR / "train_invariant.yaml",
        default={
            "seed": 42,
            "architecture": "modern_tcn",
            "window_size": 50,
            "stride": 1,
            "channels": [64, 64, 64],
            "dilations": [1, 2, 4],
            "kernel_size": 3,
            "dropout": 0.1,
            "expansion_ratio": 2,
            "pool": "avg",
            "batch_size": 128,
            "inference_batch_size": 512,
            "epochs": 15,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "penalty_weight": 0.1,
            "domain_count": 3,
        },
    )

    seed = int(cfg.get("seed", 42) if seed is None else seed)
    set_seed(seed)
    print("\n" + "=" * 80, flush=True)
    print(f"[START] train_invariant seed={seed}", flush=True)
    print("  model     : Invariant temporal baseline", flush=True)
    print("  config    : configs/train_invariant.yaml", flush=True)
    print("=" * 80, flush=True)

    train_df = read_windows("te_train_windows.csv")
    val_df = read_windows("te_val_windows.csv")
    print_window_stats("train", train_df)
    print_window_stats("val", val_df)

    X_train = flattened_to_tensor(train_df)
    y_train = train_df["y"].to_numpy(dtype=np.float32)
    env_train = train_df["phase"].map(phase_to_env).to_numpy(dtype=np.int64)
    X_val = flattened_to_tensor(val_df)
    y_val = val_df["y"].to_numpy(dtype=np.float32)
    val_phases = val_df["phase"].to_numpy()
    X_train, X_val, imputer, scaler = transform_with_imputer_scaler(X_train, X_val)

    batch_size = int(cfg.get("batch_size", 128))
    channels = tuple(cfg.get("channels", [64, 64, 64]))
    dilations = tuple(cfg.get("dilations", [1, 2, 4]))
    kernel_size = int(cfg.get("kernel_size", 3))
    dropout = float(cfg.get("dropout", 0.1))
    expansion_ratio = int(cfg.get("expansion_ratio", 2))
    pool = str(cfg.get("pool", "avg"))
    epochs = int(cfg.get("epochs", 15))
    lr = float(cfg.get("lr", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 1e-4))
    penalty_weight = float(cfg.get("penalty_weight", 0.1))
    architecture = str(cfg.get("architecture", "modern_tcn"))
    temporal_name = temporal_model_display_name(architecture)

    device_info = torch_device_info(prefer_mps=True)
    device = device_info["selected_device"]

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
            torch.tensor(env_train, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_xb = torch.tensor(X_val, dtype=torch.float32).to(device)

    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    pos_weight = torch.tensor([max(1.0, neg / max(pos, 1.0))], dtype=torch.float32).to(device)

    model = build_temporal_model(
        n_features=X_train.shape[1],
        cfg={
            "architecture": architecture,
            "channels": channels,
            "dilations": dilations,
            "kernel_size": kernel_size,
            "dropout": dropout,
            "expansion_ratio": expansion_ratio,
            "pool": pool,
        },
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_f1 = -np.inf
    best_recall = -np.inf
    best_post_shift_recall = -np.inf
    best_threshold = 0.5
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        penalty_sum = 0.0
        n_seen = 0
        for xb, yb, envb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            envb = envb.to(device)
            optimizer.zero_grad()
            logits, features = model(xb, return_features=True)
            cls_loss = criterion(logits, yb)
            normal_mask = yb < 0.5
            penalty = invariant_penalty(features[normal_mask], envb[normal_mask]) if bool(normal_mask.any()) else features.new_tensor(0.0)
            loss = cls_loss + penalty_weight * penalty
            loss.backward()
            optimizer.step()
            bs = xb.size(0)
            loss_sum += float(cls_loss.item()) * bs
            penalty_sum += float(penalty.item()) * bs
            n_seen += bs

        train_loss = loss_sum / max(n_seen, 1)
        train_penalty = penalty_sum / max(n_seen, 1)
        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(model(val_xb)).cpu().numpy()
        val_threshold, metrics = best_threshold_by_f1(y_val, val_probs)
        post_shift_recall = phase_recall(y_val, val_probs, val_phases, "post_shift", val_threshold)
        transition_recall = phase_recall(y_val, val_probs, val_phases, "transition", val_threshold)
        selection_score = float(metrics["f1"]) + 0.10 * float(metrics["recall"]) + 0.35 * float(post_shift_recall)
        best_score = best_f1 + 0.10 * best_recall + 0.35 * best_post_shift_recall
        if selection_score > best_score or (selection_score == best_score and metrics["f1"] > best_f1):
            best_f1 = float(metrics["f1"])
            best_recall = float(metrics["recall"])
            best_post_shift_recall = float(post_shift_recall)
            best_threshold = float(val_threshold)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        history.append(
            {
                "epoch": epoch,
                "loss": train_loss,
                "invariant_penalty": train_penalty,
                "f1": metrics["f1"],
                "recall": metrics["recall"],
                "auc": metrics["auc"],
                "post_shift_recall": post_shift_recall,
                "transition_recall": transition_recall,
                "threshold": float(val_threshold),
            }
        )

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    torch.save(model.state_dict(), MODEL_DIR / f"{MODEL_PREFIX}_model_seed{seed}.pt")
    joblib.dump(imputer, MODEL_DIR / f"{MODEL_PREFIX}_imputer_seed{seed}.pkl")
    joblib.dump(scaler, MODEL_DIR / f"{MODEL_PREFIX}_scaler_seed{seed}.pkl")
    with open(MODEL_DIR / f"{MODEL_PREFIX}_meta_seed{seed}.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": seed,
                "architecture": architecture,
                "display_name": f"{temporal_name} + Invariant",
                "n_features": int(X_train.shape[1]),
                "window_size": int(X_train.shape[2]),
                "channels": list(channels),
                "dilations": list(dilations),
                "kernel_size": kernel_size,
                "dropout": dropout,
                "expansion_ratio": expansion_ratio,
                "pool": pool,
                "batch_size": batch_size,
                "epochs": epochs,
                "lr": lr,
                "weight_decay": weight_decay,
                "best_val_threshold": float(best_threshold),
                "penalty_weight": penalty_weight,
                "domain_count": int(cfg.get("domain_count", 3)),
                "env_definition": {"normal_or_pre": 0, "transition": 1, "post_shift": 2},
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    pd.DataFrame(history).to_csv(MODEL_DIR / f"{MODEL_PREFIX}_history_seed{seed}.csv", index=False)
    print(f"[DONE] train_invariant seed={seed}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    main(seed=args.seed)
