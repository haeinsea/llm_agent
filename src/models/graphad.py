from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.utils.io import read_json, write_json


GRAPHAD_SMOOTH_FEATURES = [
    "graphad_score",
    "graphad_top1_sensor",
    "graphad_top1_score",
    "graphad_top1_z",
    "graphad_top1_trend",
    "graphad_top1_fluct",
    "graphad_top1_neighbors",
    "graphad_top2_sensor",
    "graphad_top2_score",
    "graphad_top1_gap",
    "graphad_topk_mean",
    "graphad_topk_sensors",
    "graphad_topk_scores",
    "graphad_topology",
]

GRAPHAD_RAW_FEATURES = [
    "graphad_raw_score",
    "graphad_raw_top1_sensor",
    "graphad_raw_top1_score",
    "graphad_raw_top1_z",
    "graphad_raw_top1_trend",
    "graphad_raw_top1_fluct",
    "graphad_raw_top1_neighbors",
    "graphad_raw_top2_sensor",
    "graphad_raw_top2_score",
    "graphad_raw_top1_gap",
    "graphad_raw_topk_mean",
    "graphad_raw_topk_sensors",
    "graphad_raw_topk_scores",
    "graphad_raw_topology",
]

GRAPHAD_FEATURES = GRAPHAD_SMOOTH_FEATURES + GRAPHAD_RAW_FEATURES

GROUP_COL_CANDIDATES = ["source_file", "fault_id", "run_id"]


def _group_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in GROUP_COL_CANDIDATES if c in df.columns]
    if not cols:
        raise KeyError(f"GraphAD requires at least one grouping key from {GROUP_COL_CANDIDATES}.")
    return cols


def _as_feature_frame(df: pd.DataFrame, feature_cols: list[str], fill_values: dict[str, float] | None = None) -> pd.DataFrame:
    feat = df[feature_cols].copy()
    for col in feature_cols:
        feat[col] = pd.to_numeric(feat[col], errors="coerce")
    if fill_values:
        feat = feat.fillna(fill_values)
    else:
        feat = feat.fillna(0.0)
    return feat


def _robust_scale(values: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    median = np.nanmedian(values, axis=0)
    mad = np.nanmedian(np.abs(values - median), axis=0)
    scale = 1.4826 * np.maximum(mad, eps)
    return median, scale


def _group_dynamics(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_cols = _group_cols(df)
    grouped = df.groupby(group_cols, sort=False)[feature_cols]
    prev1 = grouped.shift(1)
    prev2 = grouped.shift(2)
    delta = df[feature_cols] - prev1
    curvature = df[feature_cols] - 2.0 * prev1 + prev2
    return delta.fillna(0.0), curvature.fillna(0.0)


def fit_graphad(train_df: pd.DataFrame, feature_cols: list[str], cfg: dict) -> dict:
    eps = float(cfg.get("eps", 1.0e-6))
    corr_threshold = float(cfg.get("corr_threshold", 0.70))
    alpha = float(cfg.get("alpha", 0.30))
    top_k = int(cfg.get("top_k", 5))
    lambda_z = float(cfg.get("lambda_z", 0.40))
    lambda_tr = float(cfg.get("lambda_tr", 0.30))
    lambda_fl = float(cfg.get("lambda_fl", 0.30))
    score_clip = float(cfg.get("score_clip", 12.0))
    normal_only = bool(cfg.get("normal_only", True))

    fit_df = train_df.copy()
    if normal_only and "y" in fit_df.columns:
        fit_df = fit_df[fit_df["y"] == 0].copy()
    if fit_df.empty:
        raise ValueError("GraphAD fit dataset is empty after filtering.")

    feature_df = _as_feature_frame(fit_df, feature_cols)
    delta_df, curvature_df = _group_dynamics(pd.concat([fit_df[_group_cols(fit_df)], feature_df], axis=1), feature_cols)

    median, scale = _robust_scale(feature_df.to_numpy(dtype=float), eps=eps)
    delta_median, delta_scale = _robust_scale(delta_df.to_numpy(dtype=float), eps=eps)
    fluct_median, fluct_scale = _robust_scale(np.abs(curvature_df.to_numpy(dtype=float)), eps=eps)

    corr = feature_df.corr(method="pearson").fillna(0.0).to_numpy(dtype=float)
    corr_abs = np.abs(corr)
    adjacency = (corr_abs >= corr_threshold).astype(float)
    np.fill_diagonal(adjacency, 0.0)
    degree = adjacency.sum(axis=1)

    artifact = {
        "feature_cols": list(feature_cols),
        "corr_threshold": corr_threshold,
        "alpha": alpha,
        "top_k": top_k,
        "lambda_z": lambda_z,
        "lambda_tr": lambda_tr,
        "lambda_fl": lambda_fl,
        "score_clip": score_clip,
        "eps": eps,
        "normal_only": normal_only,
        "median": {col: float(median[idx]) for idx, col in enumerate(feature_cols)},
        "scale": {col: float(scale[idx]) for idx, col in enumerate(feature_cols)},
        "delta_median": {col: float(delta_median[idx]) for idx, col in enumerate(feature_cols)},
        "delta_scale": {col: float(delta_scale[idx]) for idx, col in enumerate(feature_cols)},
        "fluct_median": {col: float(fluct_median[idx]) for idx, col in enumerate(feature_cols)},
        "fluct_scale": {col: float(fluct_scale[idx]) for idx, col in enumerate(feature_cols)},
        "adjacency": {
            col: [feature_cols[j] for j in np.flatnonzero(adjacency[idx])]
            for idx, col in enumerate(feature_cols)
        },
        "summary": {
            "n_rows": int(len(feature_df)),
            "n_features": int(len(feature_cols)),
            "edge_count": int(adjacency.sum() / 2.0),
            "mean_degree": float(np.mean(degree)),
            "median_degree": float(np.median(degree)),
            "max_degree": int(degree.max()) if len(degree) else 0,
        },
    }
    return artifact


def save_graphad_artifact(path: Path, artifact: dict) -> None:
    write_json(path, artifact)


def load_graphad_artifact(path: Path) -> dict:
    return read_json(path)


def _artifact_vectors(artifact: dict, feature_cols: list[str]) -> dict[str, np.ndarray]:
    adjacency = artifact.get("adjacency", {})
    adjacency_matrix = np.zeros((len(feature_cols), len(feature_cols)), dtype=float)
    for i, feature in enumerate(feature_cols):
        for neighbor in adjacency.get(feature, []):
            if neighbor in feature_cols:
                adjacency_matrix[i, feature_cols.index(neighbor)] = 1.0

    degree = adjacency_matrix.sum(axis=1)
    adjacency_norm = np.zeros_like(adjacency_matrix)
    nonzero = degree > 0
    adjacency_norm[nonzero] = adjacency_matrix[nonzero] / degree[nonzero, None]

    return {
        "median": np.asarray([float(artifact["median"][c]) for c in feature_cols], dtype=float),
        "scale": np.asarray([float(artifact["scale"][c]) for c in feature_cols], dtype=float),
        "delta_median": np.asarray([float(artifact["delta_median"][c]) for c in feature_cols], dtype=float),
        "delta_scale": np.asarray([float(artifact["delta_scale"][c]) for c in feature_cols], dtype=float),
        "fluct_median": np.asarray([float(artifact["fluct_median"][c]) for c in feature_cols], dtype=float),
        "fluct_scale": np.asarray([float(artifact["fluct_scale"][c]) for c in feature_cols], dtype=float),
        "adjacency_norm": adjacency_norm,
        "degree_zero": degree == 0,
    }


def _neighbor_summary(adjacency: dict[str, list[str]], sensors: list[str]) -> str:
    top_set = set(sensors)
    edges = []
    for sensor in sensors:
        neighbors = [neighbor for neighbor in adjacency.get(sensor, []) if neighbor in top_set]
        if neighbors:
            edges.append(f"{sensor}->{','.join(neighbors[:3])}")
    return "; ".join(edges) if edges else "none"


def graphad_score_matrices(df: pd.DataFrame, artifact: dict) -> dict[str, pd.DataFrame]:
    feature_cols = list(artifact["feature_cols"])
    fill_values = {col: float(artifact["median"][col]) for col in feature_cols}
    feat = _as_feature_frame(df, feature_cols, fill_values=fill_values)
    group_meta = df[_group_cols(df)].copy()
    delta_df, curvature_df = _group_dynamics(pd.concat([group_meta, feat], axis=1), feature_cols)

    vectors = _artifact_vectors(artifact, feature_cols)
    feat_np = feat.to_numpy(dtype=float)
    delta_np = delta_df.to_numpy(dtype=float)
    fluct_np = np.abs(curvature_df.to_numpy(dtype=float))

    z = np.abs((feat_np - vectors["median"]) / vectors["scale"])
    trend = np.abs((delta_np - vectors["delta_median"]) / vectors["delta_scale"])
    fluct = np.abs((fluct_np - vectors["fluct_median"]) / vectors["fluct_scale"])

    score_clip = float(artifact.get("score_clip", 12.0))
    z = np.clip(z, 0.0, score_clip)
    trend = np.clip(trend, 0.0, score_clip)
    fluct = np.clip(fluct, 0.0, score_clip)

    raw_score = (
        float(artifact.get("lambda_z", 0.40)) * z
        + float(artifact.get("lambda_tr", 0.30)) * trend
        + float(artifact.get("lambda_fl", 0.30)) * fluct
    )

    adjacency_norm = vectors["adjacency_norm"]
    neighbor_mean = raw_score @ adjacency_norm.T
    if np.any(vectors["degree_zero"]):
        neighbor_mean[:, vectors["degree_zero"]] = raw_score[:, vectors["degree_zero"]]
    alpha = float(artifact.get("alpha", 0.30))
    smooth_score = (1.0 - alpha) * raw_score + alpha * neighbor_mean

    return {
        "z": pd.DataFrame(z, index=df.index, columns=feature_cols),
        "trend": pd.DataFrame(trend, index=df.index, columns=feature_cols),
        "fluct": pd.DataFrame(fluct, index=df.index, columns=feature_cols),
        "raw": pd.DataFrame(raw_score, index=df.index, columns=feature_cols),
        "smooth": pd.DataFrame(smooth_score, index=df.index, columns=feature_cols),
    }


def infer_graphad(df: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    feature_cols = list(artifact["feature_cols"])
    score_mats = graphad_score_matrices(df, artifact)
    z = score_mats["z"].to_numpy(dtype=float)
    trend = score_mats["trend"].to_numpy(dtype=float)
    fluct = score_mats["fluct"].to_numpy(dtype=float)
    raw_score = score_mats["raw"].to_numpy(dtype=float)
    smooth_score = score_mats["smooth"].to_numpy(dtype=float)
    top_k = int(artifact.get("top_k", 5))
    top_order = np.argsort(-smooth_score, axis=1)[:, :top_k]
    adjacency = artifact.get("adjacency", {})

    rows = []
    for i, indices in enumerate(top_order):
        sensors = [feature_cols[idx] for idx in indices]
        scores = [float(smooth_score[i, idx]) for idx in indices]
        top1_idx = int(indices[0])
        top2_idx = int(indices[1]) if len(indices) > 1 else top1_idx
        top1_sensor = feature_cols[top1_idx]
        top2_sensor = feature_cols[top2_idx]
        raw_order = np.argsort(-raw_score[i])[:top_k]
        raw_sensors = [feature_cols[idx] for idx in raw_order]
        raw_scores = [float(raw_score[i, idx]) for idx in raw_order]
        raw_top1_idx = int(raw_order[0])
        raw_top2_idx = int(raw_order[1]) if len(raw_order) > 1 else raw_top1_idx
        raw_top1_sensor = feature_cols[raw_top1_idx]
        raw_top2_sensor = feature_cols[raw_top2_idx]
        rows.append(
            {
                "graphad_score": float(scores[0]),
                "graphad_top1_sensor": top1_sensor,
                "graphad_top1_score": float(smooth_score[i, top1_idx]),
                "graphad_top1_z": float(z[i, top1_idx]),
                "graphad_top1_trend": float(trend[i, top1_idx]),
                "graphad_top1_fluct": float(fluct[i, top1_idx]),
                "graphad_top1_neighbors": "|".join(adjacency.get(top1_sensor, [])[:5]),
                "graphad_top2_sensor": top2_sensor,
                "graphad_top2_score": float(smooth_score[i, top2_idx]),
                "graphad_top1_gap": float(smooth_score[i, top1_idx] - smooth_score[i, top2_idx]),
                "graphad_topk_mean": float(np.mean(scores)),
                "graphad_topk_sensors": json.dumps(sensors, ensure_ascii=True, separators=(",", ":")),
                "graphad_topk_scores": json.dumps([round(score, 6) for score in scores], ensure_ascii=True, separators=(",", ":")),
                "graphad_topology": _neighbor_summary(adjacency, sensors),
                "graphad_raw_score": float(raw_score[i, raw_top1_idx]),
                "graphad_raw_top1_sensor": raw_top1_sensor,
                "graphad_raw_top1_score": float(raw_score[i, raw_top1_idx]),
                "graphad_raw_top1_z": float(z[i, raw_top1_idx]),
                "graphad_raw_top1_trend": float(trend[i, raw_top1_idx]),
                "graphad_raw_top1_fluct": float(fluct[i, raw_top1_idx]),
                "graphad_raw_top1_neighbors": "|".join(adjacency.get(raw_top1_sensor, [])[:5]),
                "graphad_raw_top2_sensor": raw_top2_sensor,
                "graphad_raw_top2_score": float(raw_score[i, raw_top2_idx]),
                "graphad_raw_top1_gap": float(raw_score[i, raw_top1_idx] - raw_score[i, raw_top2_idx]),
                "graphad_raw_topk_mean": float(np.mean(raw_scores)),
                "graphad_raw_topk_sensors": json.dumps(raw_sensors, ensure_ascii=True, separators=(",", ":")),
                "graphad_raw_topk_scores": json.dumps(
                    [round(float(raw_score[i, idx]), 6) for idx in raw_order],
                    ensure_ascii=True,
                    separators=(",", ":"),
                ),
                "graphad_raw_topology": _neighbor_summary(adjacency, raw_sensors),
            }
        )

    return pd.DataFrame(rows, index=df.index)


def copy_graphad_columns(df: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    for col in GRAPHAD_FEATURES:
        if col in source.columns:
            df[col] = source[col]
    return df


def graphad_feature_columns(columns: Iterable[str]) -> list[str]:
    existing = set(columns)
    return [col for col in GRAPHAD_FEATURES if col in existing]
