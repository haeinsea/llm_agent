from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import re

from src.utils.io import read_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
CONFIG_DIR = PROJECT_ROOT / "configs"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path, default: dict) -> dict:
    return read_yaml(path, default=default)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_windows(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing windows file: {path}")
    return pd.read_csv(path)


WINDOW_COL_PATTERN = re.compile(r"^(?P<base>.+)_t(?P<lag>\d+|-?\d+)$")


def infer_window_feature_structure(columns: List[str]) -> Tuple[List[str], int]:
    feat_cols = [c for c in columns if WINDOW_COL_PATTERN.match(c)]
    feats: Dict[str, List[int]] = {}

    for c in feat_cols:
        m = WINDOW_COL_PATTERN.match(c)
        assert m is not None
        base = m.group("base")
        lag = int(m.group("lag").replace("-", ""))
        feats.setdefault(base, []).append(lag)

    if not feats:
        raise ValueError("No valid window feature columns found.")

    feature_names = sorted(feats.keys())
    max_lag = max(max(v) for v in feats.values())
    win = max_lag + 1
    return feature_names, win


def flattened_to_tensor(df: pd.DataFrame) -> np.ndarray:
    flat_cols = [c for c in df.columns if WINDOW_COL_PATTERN.match(c)]
    feature_names, win = infer_window_feature_structure(flat_cols)
    n_feat = len(feature_names)

    X = np.zeros((len(df), n_feat, win), dtype=np.float32)
    feat_to_idx = {f: i for i, f in enumerate(feature_names)}

    for c in flat_cols:
        m = WINDOW_COL_PATTERN.match(c)
        assert m is not None
        base = m.group("base")
        lag = int(m.group("lag").replace("-", ""))
        feat_idx = feat_to_idx[base]
        X[:, feat_idx, lag] = df[c].to_numpy(dtype=np.float32)

    return X

class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNClassifier(nn.Module):
    def __init__(self, n_features, channels=(64, 64, 64), kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        n_in = n_features
        for i, ch in enumerate(channels):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            layers.append(
                TemporalBlock(
                    n_inputs=n_in,
                    n_outputs=ch,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation,
                    padding=padding,
                    dropout=dropout,
                )
            )
            n_in = ch
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(channels[-1], 1)

    def forward(self, x):
        h = self.tcn(x)
        h_last = h[:, :, -1]
        logits = self.head(h_last).squeeze(-1)
        return logits


def transform_with_imputer_scaler(
    X_train: np.ndarray,
    X_val: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, SimpleImputer, StandardScaler]:
    """
    X shape: (B, F, T)
    imputer/scaler는 feature 축 기준으로 모든 time step에 동일하게 적용
    """
    Btr, F, T = X_train.shape
    Bva = X_val.shape[0]

    Xtr2 = X_train.transpose(0, 2, 1).reshape(-1, F)
    Xva2 = X_val.transpose(0, 2, 1).reshape(-1, F)

    imputer = SimpleImputer(strategy="median")
    Xtr_imp = imputer.fit_transform(Xtr2)
    Xva_imp = imputer.transform(Xva2)

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr_imp)
    Xva_s = scaler.transform(Xva_imp)

    Xtr_out = Xtr_s.reshape(Btr, T, F).transpose(0, 2, 1).astype(np.float32)
    Xva_out = Xva_s.reshape(Bva, T, F).transpose(0, 2, 1).astype(np.float32)
    return Xtr_out, Xva_out, imputer, scaler


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    preds = (probs >= threshold).astype(int)

    out = {
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
    }

    try:
        if len(np.unique(y_true)) < 2:
            out["auc"] = None
        else:
            out["auc"] = float(roc_auc_score(y_true, probs))
    except Exception:
        out["auc"] = None

    return out


def print_window_stats(name: str, df: pd.DataFrame) -> None:
    print(f"\n[{name}] windows={len(df):,}")
    if len(df) == 0:
        return
    print(f"  pos_ratio    : {df['y'].mean():.4f}")
    print(f"  y_counts     : {df['y'].value_counts(dropna=False).to_dict()}")
    if "phase" in df.columns:
        print(f"  phase_counts : {df['phase'].value_counts(dropna=False).to_dict()}")


def main(seed: int | None = None) -> None:
    ensure_dir(MODEL_DIR)

    cfg = load_yaml(
        CONFIG_DIR / "train_tcn.yaml",
        default={
            "seed": 42,
            "window_size": 50,
            "stride": 1,
            "channels": [64, 64, 64],
            "kernel_size": 3,
            "dropout": 0.1,
            "batch_size": 128,
            "epochs": 15,
            "lr": 1e-3,
            "weight_decay": 1e-4,
        },
    )

    seed = int(cfg.get("seed", 42) if seed is None else seed)
    set_seed(seed)

    train_df = read_windows("te_train_windows.csv")
    val_df = read_windows("te_val_windows.csv")

    print_window_stats("train", train_df)
    print_window_stats("val", val_df)

    if len(train_df) == 0:
        raise ValueError("No training windows found. Check split/window generation.")
    if len(val_df) == 0:
        raise ValueError("No validation windows found. Check split/window generation.")

    X_train = flattened_to_tensor(train_df)
    y_train = train_df["y"].to_numpy(dtype=np.float32)

    X_val = flattened_to_tensor(val_df)
    y_val = val_df["y"].to_numpy(dtype=np.float32)

    X_train, X_val, imputer, scaler = transform_with_imputer_scaler(X_train, X_val)

    batch_size = int(cfg.get("batch_size", 128))
    channels = tuple(cfg.get("channels", [64, 64, 64]))
    kernel_size = int(cfg.get("kernel_size", 3))
    dropout = float(cfg.get("dropout", 0.1))
    epochs = int(cfg.get("epochs", 15))
    lr = float(cfg.get("lr", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 1e-4))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_xb = torch.tensor(X_val, dtype=torch.float32).to(device)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    n_pos = float((y_train == 1).sum())
    n_neg = float((y_train == 0).sum())
    pos_weight_value = n_neg / max(n_pos, 1.0)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)

    model = TCNClassifier(
        n_features=X_train.shape[1],
        channels=channels,
        kernel_size=kernel_size,
        dropout=dropout,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_auc = -np.inf
    best_f1 = -np.inf
    history = []

    print(f"\n[train setup]")
    print(f"  device       : {device}")
    print(f"  X_train      : {X_train.shape}")
    print(f"  X_val        : {X_val.shape}")
    print(f"  pos_weight   : {pos_weight_value:.4f}")
    print(f"  channels     : {channels}")
    print(f"  kernel_size  : {kernel_size}")
    print(f"  dropout      : {dropout}")
    print(f"  batch_size   : {batch_size}")
    print(f"  epochs       : {epochs}")
    print(f"  lr           : {lr}")
    print(f"  weight_decay : {weight_decay}")

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        n_seen = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)  # logits only
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            bs = xb.size(0)
            loss_sum += float(loss.item()) * bs
            n_seen += bs

        train_loss = loss_sum / max(n_seen, 1)

        model.eval()
        with torch.no_grad():
            val_logits = model(val_xb)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()

        metrics = compute_metrics(y_val, val_probs, threshold=0.5)
        prob_min = float(np.min(val_probs))
        prob_max = float(np.max(val_probs))
        n_pred_pos = int((val_probs >= 0.5).sum())

        auc_for_select = metrics["auc"] if metrics["auc"] is not None else -np.inf
        improved = False
        if auc_for_select > best_auc:
            improved = True
        elif auc_for_select == best_auc and metrics["f1"] > best_f1:
            improved = True

        if improved:
            best_auc = auc_for_select
            best_f1 = metrics["f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        row = {
            "epoch": epoch,
            "loss": train_loss,
            "f1": metrics["f1"],
            "recall": metrics["recall"],
            "auc": metrics["auc"],
            "prob_min": prob_min,
            "prob_max": prob_max,
            "n_pred_pos": n_pred_pos,
        }
        history.append(row)

        print(
            f"[Epoch {epoch:02d}] "
            f"loss={train_loss:.4f} "
            f"f1={metrics['f1']:.4f} "
            f"recall={metrics['recall']:.4f} "
            f"auc={metrics['auc']} "
            f"prob_min={prob_min:.4f} "
            f"prob_max={prob_max:.4f} "
            f"n_pred_pos={n_pred_pos}"
        )

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # best model restore
    model.load_state_dict(best_state)

    # save
    torch.save(model.state_dict(), MODEL_DIR / f"tcn_model_seed{seed}.pt")
    joblib.dump(imputer, MODEL_DIR / f"tcn_imputer_seed{seed}.pkl")
    joblib.dump(scaler, MODEL_DIR / f"tcn_scaler_seed{seed}.pkl")

    meta = {
        "seed": seed,
        "n_features": int(X_train.shape[1]),
        "window_size": int(X_train.shape[2]),
        "channels": list(channels),
        "kernel_size": kernel_size,
        "dropout": dropout,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "label_definition": {
            "normal": 0,
            "pre": 0,
            "transition": 1,
            "post_shift": 1,
        },
        "train_windows": int(len(train_df)),
        "val_windows": int(len(val_df)),
        "train_positive_ratio": float(np.mean(y_train)),
        "val_positive_ratio": float(np.mean(y_val)),
        "pos_weight": float(pos_weight_value),
    }
    with open(MODEL_DIR / f"tcn_meta_seed{seed}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(MODEL_DIR / f"tcn_history_seed{seed}.csv", index=False)

    print("\nTCN training completed.")
    print(f"Saved: {MODEL_DIR / f'tcn_model_seed{seed}.pt'}")
    print(f"Saved: {MODEL_DIR / f'tcn_imputer_seed{seed}.pkl'}")
    print(f"Saved: {MODEL_DIR / f'tcn_scaler_seed{seed}.pkl'}")
    print(f"Saved: {MODEL_DIR / f'tcn_meta_seed{seed}.json'}")
    print(f"Saved: {MODEL_DIR / f'tcn_history_seed{seed}.csv'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    main(seed=args.seed)
