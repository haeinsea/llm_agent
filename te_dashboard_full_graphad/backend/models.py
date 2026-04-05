from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

try:
    import torch
except Exception:
    torch = None

from src.models.graphad import infer_graphad, load_graphad_artifact
from src.models.temporal_backbone import build_temporal_model
from src.routing.selective_llm_eval import LLMProbabilityRunner, apply_mode
from src.utils.device import get_torch_device, synchronize_torch_device
from src.utils.experiment import get_seed_list
from src.utils.io import read_csv, read_json, read_yaml
from src.utils.routing import build_routing_features

from .config import (
    CONFIG_DIR,
    FEATURE_COLUMNS_PATH,
    GRAPHAD_ARTIFACT_PATH,
    GRAYZONE_GRID_PATH,
    METRIC_DIR,
    MODELS_DIR,
    SELECTED_Q_PATH,
    THRESHOLDS_PATH,
)


SEEDS = get_seed_list()
_CACHE: dict[str, Any] = {}


@dataclass
class RoutingContext:
    selected_q: float
    tau: float
    margin: float
    entropy_threshold: float
    discrepancy_threshold: float
    routing_cfg: dict[str, Any]
    llm_runner: LLMProbabilityRunner
    device: str


def load_feature_cols() -> list[str]:
    if "feature_cols" in _CACHE:
        return _CACHE["feature_cols"]
    with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
        cols = json.load(f)
    _CACHE["feature_cols"] = list(cols)
    return _CACHE["feature_cols"]


def align_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = load_feature_cols()
    aligned = pd.DataFrame(index=df.index)
    for col in feature_cols:
        if col in df.columns:
            aligned[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            aligned[col] = np.nan
    return aligned


def _load_compat_joblib(path: Path) -> Any:
    obj = joblib.load(path)
    if hasattr(obj, "_fit_dtype") and not hasattr(obj, "_fill_dtype"):
        obj._fill_dtype = obj._fit_dtype
    return obj


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _cached(key: str, loader):
    if key not in _CACHE:
        _CACHE[key] = loader()
    return _CACHE[key]


def _label_to_binary(y: pd.Series | None) -> pd.Series:
    if y is None:
        return pd.Series(np.nan, index=pd.RangeIndex(0))
    numeric = pd.to_numeric(y, errors="coerce")
    return (numeric.fillna(0) != 0).astype(int)


def _infer_phase_info(y_bin: pd.Series | None, n_rows: int, transition_len: int = 50) -> tuple[list[str], list[int]]:
    if y_bin is None or len(y_bin) != n_rows or y_bin.isna().all():
        return ["uploaded"] * n_rows, [10**9] * n_rows

    y_arr = y_bin.to_numpy(dtype=float)
    anomaly_idx = np.flatnonzero(y_arr >= 0.5)
    if len(anomaly_idx) == 0:
        return ["normal"] * n_rows, [10**9] * n_rows

    onset = int(anomaly_idx[0])
    phases: list[str] = []
    onset_steps: list[int] = []
    for idx, label in enumerate(y_arr):
        onset_steps.append(onset)
        if label < 0.5:
            phases.append("pre" if idx < onset else "normal")
        elif idx < onset + int(transition_len):
            phases.append("transition")
        else:
            phases.append("post_shift")
    return phases, onset_steps


def build_dashboard_metadata(
    feature_df: pd.DataFrame,
    display_index: pd.Series | None = None,
    y_true: pd.Series | None = None,
) -> pd.DataFrame:
    n_rows = len(feature_df)
    if display_index is not None:
        display_numeric = pd.to_numeric(display_index, errors="coerce")
        fallback = pd.Series(np.arange(n_rows, dtype=int), index=display_numeric.index)
        display_vals = display_numeric.where(display_numeric.notna(), fallback).astype(int).to_numpy()
    else:
        display_vals = np.arange(n_rows, dtype=int)
    y_bin = _label_to_binary(y_true) if y_true is not None else None
    phases, onset_steps = _infer_phase_info(y_bin if y_true is not None else None, n_rows=n_rows)

    meta = pd.DataFrame(
        {
            "source_file": ["dashboard_upload.csv"] * n_rows,
            "domain_tag": ["dashboard"] * n_rows,
            "split_group": ["dashboard_upload"] * n_rows,
            "run_id": [1] * n_rows,
            "fault_id": [0] * n_rows,
            "sample_idx": np.arange(n_rows, dtype=int),
            "display_index": display_vals,
            "y_true": y_bin.to_numpy(dtype=float) if y_true is not None else np.full(n_rows, np.nan),
            "phase": phases,
            "onset_step": onset_steps,
            "transition_len": [50] * n_rows,
        }
    )
    return meta


def _load_rf_artifact(seed: int) -> tuple[Any, Any]:
    return _cached(
        f"rf:{seed}",
        lambda: (
            joblib.load(MODELS_DIR / f"rf_model_seed{seed}.pkl"),
            _load_compat_joblib(MODELS_DIR / f"rf_imputer_seed{seed}.pkl"),
        ),
    )


def _load_xgb_artifact(seed: int) -> tuple[Any, Any]:
    return _cached(
        f"xgb:{seed}",
        lambda: (
            joblib.load(MODELS_DIR / f"xgb_model_seed{seed}.pkl"),
            _load_compat_joblib(MODELS_DIR / f"xgb_imputer_seed{seed}.pkl"),
        ),
    )


def _load_tcn_artifact(seed: int, device: str) -> tuple[Any, Any, Any, dict[str, Any]]:
    key = f"tcn:{seed}:{device}"

    def _loader():
        meta = _load_json(MODELS_DIR / f"tcn_meta_seed{seed}.json")
        imputer = _load_compat_joblib(MODELS_DIR / f"tcn_imputer_seed{seed}.pkl")
        scaler = joblib.load(MODELS_DIR / f"tcn_scaler_seed{seed}.pkl")
        if torch is None:
            raise ImportError("PyTorch is required for ModernTCN inference.")
        model = build_temporal_model(n_features=int(meta.get("n_features", len(load_feature_cols()))), cfg=meta)
        state = torch.load(MODELS_DIR / f"tcn_model_seed{seed}.pt", map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        return model, imputer, scaler, meta

    return _cached(key, _loader)


def _build_left_padded_windows(feature_df: pd.DataFrame, window_size: int) -> np.ndarray:
    values = feature_df.to_numpy(dtype=np.float32)
    n_rows, n_features = values.shape
    windows = np.zeros((n_rows, n_features, window_size), dtype=np.float32)
    for end_idx in range(n_rows):
        start_idx = max(0, end_idx - window_size + 1)
        seq = values[start_idx : end_idx + 1]
        if len(seq) == 0:
            continue
        if len(seq) < window_size:
            pad = np.repeat(seq[:1], window_size - len(seq), axis=0)
            seq = np.vstack([pad, seq])
        windows[end_idx] = seq.T
    return windows


def _transform_windows(windows: np.ndarray, imputer: Any, scaler: Any) -> np.ndarray:
    n_rows, n_features, window_size = windows.shape
    flat = windows.transpose(0, 2, 1).reshape(-1, n_features)
    if imputer is not None:
        flat = imputer.transform(flat)
    if scaler is not None:
        flat = scaler.transform(flat)
    return flat.reshape(n_rows, window_size, n_features).transpose(0, 2, 1).astype(np.float32)


def predict_rf_xgb_mean(feature_df: pd.DataFrame) -> pd.DataFrame:
    X_raw = feature_df.to_numpy(dtype=float)
    out = pd.DataFrame(index=feature_df.index)

    rf_seed_cols: list[str] = []
    for seed in SEEDS:
        model, imputer = _load_rf_artifact(seed)
        probs = model.predict_proba(imputer.transform(X_raw))[:, 1]
        col = f"p_rf_seed{seed}"
        out[col] = probs.astype(float)
        rf_seed_cols.append(col)
    out["p_rf"] = out[rf_seed_cols].mean(axis=1)

    xgb_seed_cols: list[str] = []
    for seed in SEEDS:
        model, imputer = _load_xgb_artifact(seed)
        probs = model.predict_proba(imputer.transform(X_raw))[:, 1]
        col = f"p_xgb_seed{seed}"
        out[col] = probs.astype(float)
        xgb_seed_cols.append(col)
    out["p_xgb"] = out[xgb_seed_cols].mean(axis=1)
    return out


def predict_tcn_mean(feature_df: pd.DataFrame) -> pd.DataFrame:
    device = get_torch_device(prefer_mps=True)
    first_meta = _load_json(MODELS_DIR / f"tcn_meta_seed{SEEDS[0]}.json")
    window_size = int(first_meta.get("window_size", 50))
    infer_batch_size = int(read_yaml(CONFIG_DIR / "train_tcn.yaml", default={}).get("inference_batch_size", 512))
    windows = _build_left_padded_windows(feature_df, window_size=window_size)

    out = pd.DataFrame(index=feature_df.index)
    for seed in SEEDS:
        model, imputer, scaler, _ = _load_tcn_artifact(seed, device=device)
        Xs = _transform_windows(windows, imputer=imputer, scaler=scaler)
        if torch is None:
            raise ImportError("PyTorch is required for ModernTCN inference.")
        probs_chunks = []
        with torch.no_grad():
            for start in range(0, len(Xs), infer_batch_size):
                xb = torch.tensor(Xs[start : start + infer_batch_size], dtype=torch.float32, device=device)
                logits = model(xb)
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                probs_chunks.append(probs)
        synchronize_torch_device(device)
        out[f"p_tcn_seed{seed}"] = np.concatenate(probs_chunks).astype(float)
    out["p_tcn"] = out[[f"p_tcn_seed{seed}" for seed in SEEDS]].mean(axis=1)
    return out


def infer_graphad_features(meta_df: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    artifact = _cached("graphad_artifact", lambda: load_graphad_artifact(GRAPHAD_ARTIFACT_PATH))
    graph_input = pd.concat(
        [
            meta_df[["source_file", "fault_id", "run_id", "sample_idx"]].copy(),
            feature_df.reset_index(drop=True),
        ],
        axis=1,
    )
    return infer_graphad(graph_input, artifact)


def load_routing_context() -> RoutingContext:
    if "routing_context" in _CACHE:
        return _CACHE["routing_context"]

    routing_cfg = read_yaml(CONFIG_DIR / "routing.yaml", default={})
    llm_cfg = dict(routing_cfg.get("llm", {}) or {})
    # The dashboard should keep working even if the network or API key is unavailable.
    # In a healthy environment it will still use the real OpenAI path.
    llm_cfg["allow_stub_fallback"] = True
    routing_cfg["llm"] = llm_cfg
    selected_q = float(read_json(SELECTED_Q_PATH).get("selected_q", 0.4))
    thresholds = read_json(THRESHOLDS_PATH)
    tau = float(thresholds.get("tau", 0.5))
    gray_grid = read_csv(GRAYZONE_GRID_PATH)
    gray_row = gray_grid.iloc[(gray_grid["q"] - selected_q).abs().argsort()[:1]]
    margin = float(gray_row["gray_margin_mean"].iloc[0])
    entropy_threshold = float(gray_row["entropy_threshold_mean"].iloc[0])
    discrepancy_threshold = float(gray_row["discrepancy_threshold_mean"].iloc[0])
    device = get_torch_device(prefer_mps=True)
    ctx = RoutingContext(
        selected_q=selected_q,
        tau=tau,
        margin=margin,
        entropy_threshold=entropy_threshold,
        discrepancy_threshold=discrepancy_threshold,
        routing_cfg=routing_cfg,
        llm_runner=LLMProbabilityRunner(routing_cfg),
        device=device,
    )
    _CACHE["routing_context"] = ctx
    return ctx


def score_uploaded_te(
    raw_df: pd.DataFrame,
    display_index: pd.Series | None = None,
    y_true: pd.Series | None = None,
) -> tuple[pd.DataFrame, RoutingContext]:
    feature_df = align_feature_frame(raw_df)
    meta_df = build_dashboard_metadata(feature_df, display_index=display_index, y_true=y_true)

    score_df = pd.concat(
        [
            meta_df,
            predict_rf_xgb_mean(feature_df),
            predict_tcn_mean(feature_df),
        ],
        axis=1,
    )
    score_df["p_ensemble"] = score_df[["p_rf", "p_xgb", "p_tcn"]].mean(axis=1)

    graphad_df = infer_graphad_features(meta_df, feature_df)
    score_df = pd.concat([score_df, graphad_df], axis=1)

    routing_ctx = load_routing_context()
    routing_features = build_routing_features(score_df, routing_ctx.routing_cfg)
    score_df = pd.concat([score_df, routing_features], axis=1)

    routed = apply_mode(
        df=score_df,
        tau=routing_ctx.tau,
        margin=routing_ctx.margin,
        entropy_threshold=routing_ctx.entropy_threshold,
        discrepancy_threshold=routing_ctx.discrepancy_threshold,
        mode="selective",
        llm_runner=routing_ctx.llm_runner,
        progress_label="dashboard mode=selective",
    )
    routed["selected_q"] = routing_ctx.selected_q
    routed["tau"] = routing_ctx.tau
    routed["gray_margin"] = routing_ctx.margin
    routed["display_index"] = meta_df["display_index"].values
    return routed, routing_ctx
