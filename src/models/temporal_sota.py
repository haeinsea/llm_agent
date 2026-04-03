from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None

from src.models.temporal_backbone import build_temporal_model
from src.models.train_tcn import flattened_to_tensor
from src.utils.device import synchronize_torch_device


def load_temporal_artifact(model_dir: Path, model_prefix: str, seed: int, device: str):
    if torch is None:
        raise ImportError("PyTorch is required for temporal baseline loading.")

    imp_path = model_dir / f"{model_prefix}_imputer_seed{seed}.pkl"
    scaler_path = model_dir / f"{model_prefix}_scaler_seed{seed}.pkl"
    meta_path = model_dir / f"{model_prefix}_meta_seed{seed}.json"
    state_path = model_dir / f"{model_prefix}_model_seed{seed}.pt"

    imputer = joblib.load(imp_path) if imp_path.exists() else None
    scaler = joblib.load(scaler_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    model = build_temporal_model(
        n_features=int(meta["n_features"]),
        cfg={
            "architecture": meta.get("architecture", "modern_tcn"),
            "channels": tuple(meta.get("channels", [64, 64, 64])),
            "dilations": tuple(meta.get("dilations", [1, 2, 4])),
            "kernel_size": int(meta.get("kernel_size", 3)),
            "dropout": float(meta.get("dropout", 0.1)),
            "expansion_ratio": int(meta.get("expansion_ratio", 2)),
            "pool": str(meta.get("pool", "avg")),
        },
    ).to(device)
    state = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state)
    return model, imputer, scaler, meta


def transform_windows(df_win, imputer, scaler) -> np.ndarray:
    X = flattened_to_tensor(df_win)
    B, F, T = X.shape
    X2 = X.transpose(0, 2, 1).reshape(-1, F)
    if imputer is not None:
        X2 = imputer.transform(X2)
    X2 = scaler.transform(X2)
    return X2.reshape(B, T, F).transpose(0, 2, 1).astype(np.float32)


def infer_temporal_probs(model, Xs: np.ndarray, device: str, infer_batch_size: int) -> np.ndarray:
    if torch is None:
        raise ImportError("PyTorch is required for temporal baseline inference.")
    model.eval()
    probs = []
    synchronize_torch_device(device)
    with torch.no_grad():
        for start in range(0, len(Xs), infer_batch_size):
            stop = min(start + infer_batch_size, len(Xs))
            xb = torch.tensor(Xs[start:stop], dtype=torch.float32, device=device)
            logits = model(xb)
            probs.append(torch.sigmoid(logits).detach().cpu().numpy())
    synchronize_torch_device(device)
    return np.concatenate(probs, axis=0)


def _binary_entropy_from_logits(logits):
    probs = torch.sigmoid(logits)
    probs = probs.clamp(min=1e-6, max=1.0 - 1e-6)
    return -(probs * probs.log() + (1.0 - probs) * (1.0 - probs).log()).mean()


def _configure_tent_modules(model) -> list:
    if nn is None:
        return []
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.eval()
    params = []
    for param in model.parameters():
        param.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.train()
            if module.weight is not None:
                module.weight.requires_grad_(True)
                params.append(module.weight)
            if module.bias is not None:
                module.bias.requires_grad_(True)
                params.append(module.bias)
        elif isinstance(module, nn.Dropout):
            module.eval()
    if not params:
        for name, param in model.named_parameters():
            if name.startswith("head."):
                param.requires_grad_(True)
                params.append(param)
    return params


def infer_tta_online_probs(
    model,
    Xs: np.ndarray,
    device: str,
    infer_batch_size: int,
    lr: float,
    steps: int,
) -> np.ndarray:
    if torch is None:
        raise ImportError("PyTorch is required for TTA inference.")
    params = _configure_tent_modules(model)
    optimizer = torch.optim.Adam(params, lr=float(lr)) if params else None
    probs = []
    synchronize_torch_device(device)
    for start in range(0, len(Xs), infer_batch_size):
        stop = min(start + infer_batch_size, len(Xs))
        xb = torch.tensor(Xs[start:stop], dtype=torch.float32, device=device)
        if optimizer is not None:
            for _ in range(max(int(steps), 1)):
                optimizer.zero_grad()
                logits = model(xb)
                loss = _binary_entropy_from_logits(logits)
                loss.backward()
                optimizer.step()
        with torch.no_grad():
            logits = model(xb)
            probs.append(torch.sigmoid(logits).detach().cpu().numpy())
    synchronize_torch_device(device)
    return np.concatenate(probs, axis=0)
