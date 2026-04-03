from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


DEFAULT_GROUP_COLS = ("source_file", "fault_id", "run_id")


def _available_group_cols(df: pd.DataFrame, group_cols: Sequence[str] | None = None) -> list[str]:
    cols = list(group_cols or DEFAULT_GROUP_COLS)
    return [col for col in cols if col in df.columns]


def _binary_entropy(prob: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(np.asarray(prob, dtype=float), eps, 1.0 - eps)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def build_routing_features(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    rf_col: str = "p_rf",
    xgb_col: str = "p_xgb",
    tcn_col: str = "p_tcn",
    group_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    eps = float(cfg.get("temporal_weight_eps", 1e-8))
    sigmoid_gain = float(cfg.get("sigmoid_gain", 5.0))

    out = pd.DataFrame(index=df.index)
    temporal = df[tcn_col].to_numpy(dtype=float)
    groups = _available_group_cols(df, group_cols=group_cols)
    if groups:
        t_min = df.groupby(groups)[tcn_col].transform("min").to_numpy(dtype=float)
        t_max = df.groupby(groups)[tcn_col].transform("max").to_numpy(dtype=float)
    else:
        t_min = np.full(len(df), float(np.nanmin(temporal)))
        t_max = np.full(len(df), float(np.nanmax(temporal)))

    denom = np.maximum(t_max - t_min, eps)
    temporal_norm = np.clip((temporal - t_min) / denom, 0.0, 1.0)
    temporal_weight = 1.0 / (1.0 + np.exp(-sigmoid_gain * (temporal_norm - 0.5)))

    rf_score = df[rf_col].to_numpy(dtype=float)
    xgb_score = df[xgb_col].to_numpy(dtype=float)
    temporal_component = temporal_weight * temporal
    rf_component = (1.0 - temporal_weight) * rf_score
    xgb_component = (1.0 - temporal_weight) * xgb_score

    score_matrix = np.column_stack([rf_score, xgb_score, temporal])
    ensemble_mean = score_matrix.mean(axis=1)

    out["p_temporal_norm"] = temporal_norm
    out["temporal_weight"] = temporal_weight
    out["rf_component"] = rf_component
    out["xgb_component"] = xgb_component
    out["temporal_component"] = temporal_component
    out["ensemble_mean"] = ensemble_mean
    out["ensemble_entropy"] = _binary_entropy(ensemble_mean)
    out["model_discrepancy"] = score_matrix.std(axis=1)
    out["p_utar_base"] = np.maximum.reduce([rf_component, xgb_component, temporal_component])
    return out


def compute_base_routing_score(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    rf_col: str = "p_rf",
    xgb_col: str = "p_xgb",
    tcn_col: str = "p_tcn",
    group_cols: Sequence[str] | None = None,
) -> pd.Series:
    return build_routing_features(
        df=df,
        cfg=cfg,
        rf_col=rf_col,
        xgb_col=xgb_col,
        tcn_col=tcn_col,
        group_cols=group_cols,
    )["p_utar_base"]
