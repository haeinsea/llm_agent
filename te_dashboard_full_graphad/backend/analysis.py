from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .graph_utils import build_sample_subgraph, graph_to_json, load_process_graph


def compute_meta_features_simple(x_row: pd.Series) -> dict[str, float]:
    vals = x_row.to_numpy(dtype=float)
    return {
        "feat_mean": float(np.nanmean(vals)),
        "feat_std": float(np.nanstd(vals)),
        "feat_max": float(np.nanmax(vals)),
        "feat_min": float(np.nanmin(vals)),
        "feat_range": float(np.nanmax(vals) - np.nanmin(vals)),
        "feat_energy": float(np.nansum(vals**2)),
    }


def _parse_jsonish_list(value: Any) -> list[Any]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def _direction_from_baseline(var: str, feature_row: pd.Series, feature_median: pd.Series) -> str:
    baseline = float(feature_median.get(var, np.nan))
    current = float(feature_row.get(var, np.nan))
    if np.isnan(current) or np.isnan(baseline):
        return "unknown"
    return "increase" if current >= baseline else "decrease"


def graphad_topk_list(feature_row: pd.Series, graphad_row: pd.Series, feature_median: pd.Series) -> list[dict[str, Any]]:
    _, var_to_proc = load_process_graph()
    sensors = _parse_jsonish_list(graphad_row.get("graphad_topk_sensors"))
    scores = _parse_jsonish_list(graphad_row.get("graphad_topk_scores"))
    topk: list[dict[str, Any]] = []
    for idx, sensor in enumerate(sensors):
        score = float(scores[idx]) if idx < len(scores) else float("nan")
        direction = _direction_from_baseline(str(sensor), feature_row, feature_median)
        topk.append(
            {
                "var": str(sensor),
                "score": float(score),
                "direction": direction,
                "process": var_to_proc.get(str(sensor), "Unknown"),
            }
        )
    return topk


def build_graph_contexts(topk_list: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    g, _ = load_process_graph()
    top_vars = [str(item["var"]) for item in topk_list]
    sub_g = build_sample_subgraph(g, top_vars, hops=1)
    return graph_to_json(g), graph_to_json(sub_g)


def build_routing_paths(topk_list: list[dict[str, Any]]) -> list[list[str]]:
    path = []
    for item in topk_list:
        process = str(item.get("process", "Unknown") or "Unknown")
        sensor = str(item.get("var", "")).strip()
        if not sensor:
            continue
        path.append(f"{process}:{sensor}")
    return [path] if path else []


def build_llm_structured_inputs(
    feature_row: pd.Series,
    score_row: pd.Series,
    topk_list: list[dict[str, Any]],
    process_graph_context: dict[str, Any],
    subgraph_context: dict[str, Any],
) -> dict[str, Any]:
    meta = compute_meta_features_simple(feature_row)
    return {
        "anomaly_scores": {
            "utar_base_score": float(score_row.get("p_utar_base", 0.0)),
            "rf_prob": float(score_row.get("p_rf", 0.0)),
            "xgb_prob": float(score_row.get("p_xgb", 0.0)),
            "tcn_prob": float(score_row.get("p_tcn", 0.0)),
            "ensemble_mean": float(score_row.get("ensemble_mean", score_row.get("p_ensemble", 0.0))),
            "ensemble_entropy": float(score_row.get("ensemble_entropy", 0.0)),
            "model_discrepancy": float(score_row.get("model_discrepancy", 0.0)),
            "graphad_score": float(score_row.get("graphad_score", 0.0)),
            "final_prob": float(score_row.get("p_final", 0.0)),
            "tau": float(score_row.get("tau", 0.5)),
            "selected_q": float(score_row.get("selected_q", 0.4)),
            "gray_margin": float(score_row.get("gray_margin", 0.0)),
            "gray_zone": int(score_row.get("gray_zone", 0)),
            "llm_called": int(score_row.get("llm_called", 0)),
            "llm_decision": None if pd.isna(score_row.get("llm_decision")) else str(score_row.get("llm_decision")),
            "decision_source": str(score_row.get("decision_source", "utar_base")),
            "final_decision": str(score_row.get("final_decision", "normal")),
        },
        "graphad_topk_list": topk_list,
        "process_graph_context": process_graph_context,
        "meta_features": meta,
        "subgraph_context": subgraph_context,
    }
