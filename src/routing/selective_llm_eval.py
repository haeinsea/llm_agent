from __future__ import annotations

from pathlib import Path
from typing import Callable
from functools import lru_cache
import time
import hashlib

import json
import os
import traceback
import urllib.request
import urllib.error
import socket

import numpy as np
import pandas as pd

from src.models.graphad import graphad_feature_columns
from src.models.temporal_backbone import temporal_model_display_name
from src.utils.env import load_dotenv
from src.utils.experiment import get_llm_seed_policy, get_representative_seed, get_seed_list
from src.utils.io import read_csv, read_json, read_yaml, write_csv
from src.utils.metrics import binary_metrics, instability_score, prr, worst_case_recall
from src.utils.runtime import get_base_runtime_stat, load_base_runtime_summary
from src.utils.routing import build_routing_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PRED_DIR = OUTPUT_DIR / "predictions"
METRIC_DIR = OUTPUT_DIR / "metrics"
SEEDS = get_seed_list()
REPRESENTATIVE_SEED = get_representative_seed()
DEFAULT_Q = 0.80
KEY_COLS = ["source_file", "domain_tag", "split_group", "run_id", "fault_id", "sample_idx", "y_true", "phase", "onset_step", "transition_len"]
TEMPORAL_MODEL_LABEL = temporal_model_display_name(read_yaml(CONFIG_DIR / "train_tcn.yaml", default={}).get("architecture", "modern_tcn"))
BASE_RUNTIME_SUMMARY_PATH = METRIC_DIR / "base_inference_runtime_summary.json"
SELECTED_Q_PATH = METRIC_DIR / "selected_q.json"
BASE_STACK_COMPONENT = "UTAR Base Stack"
GRAPHAD_RAW_CONTEXT_MAP = {
    "graphad_score": "graphad_raw_score",
    "graphad_top1_sensor": "graphad_raw_top1_sensor",
    "graphad_top1_score": "graphad_raw_top1_score",
    "graphad_top1_z": "graphad_raw_top1_z",
    "graphad_top1_trend": "graphad_raw_top1_trend",
    "graphad_top1_fluct": "graphad_raw_top1_fluct",
    "graphad_top1_neighbors": "graphad_raw_top1_neighbors",
    "graphad_top2_sensor": "graphad_raw_top2_sensor",
    "graphad_top2_score": "graphad_raw_top2_score",
    "graphad_top1_gap": "graphad_raw_top1_gap",
    "graphad_topk_mean": "graphad_raw_topk_mean",
    "graphad_topk_sensors": "graphad_raw_topk_sensors",
    "graphad_topk_scores": "graphad_raw_topk_scores",
    "graphad_topology": "graphad_raw_topology",
}


def _safe_float(row: pd.Series, key: str, default: float = np.nan) -> float:
    value = row[key] if key in row else default
    if pd.isna(value):
        return default
    return float(value)


def _safe_text(row: pd.Series, key: str, default: str = "n/a") -> str:
    value = row[key] if key in row else default
    if pd.isna(value):
        return default
    return str(value)


def read_selected_q(default_q: float = DEFAULT_Q) -> float:
    if not SELECTED_Q_PATH.exists():
        return float(default_q)
    payload = read_json(SELECTED_Q_PATH)
    try:
        return float(payload.get("selected_q", default_q))
    except Exception:
        return float(default_q)


def _categorize_level(value: float, low_thr: float, high_thr: float) -> str:
    if value <= low_thr:
        return "low"
    if value >= high_thr:
        return "high"
    return "medium"


def _is_present_text(value: str) -> bool:
    text = str(value).strip().lower()
    return text not in {"", "n/a", "none", "nan", "[]"}


@lru_cache(maxsize=1)
def _prompt_reference_state() -> dict:
    cfg = read_yaml(CONFIG_DIR / "routing.yaml", default={})
    tau_payload = read_json(METRIC_DIR / "thresholds.json") if (METRIC_DIR / "thresholds.json").exists() else {}
    rep_seed_key = f"seed{REPRESENTATIVE_SEED}"
    tau = float(tau_payload.get("per_seed", {}).get(rep_seed_key, {}).get("tau", tau_payload.get("tau", 0.5)))
    selected_q = read_selected_q(DEFAULT_Q)
    gray_grid = read_csv(METRIC_DIR / "grayzone_grid.csv") if (METRIC_DIR / "grayzone_grid.csv").exists() else pd.DataFrame()
    if not gray_grid.empty and "q" in gray_grid.columns:
        row = gray_grid.iloc[(gray_grid["q"] - selected_q).abs().argsort()[:1]]
        margin = float(row["gray_margin_mean"].iloc[0])
    else:
        margin = 0.05

    base_path = PRED_DIR / "base_test_main_predictions.csv"
    if not base_path.exists():
        return {
            "tau": tau,
            "margin": margin,
            "examples": [],
            "entropy_low": 0.33,
            "entropy_high": 0.66,
            "discrepancy_low": 0.03,
            "discrepancy_high": 0.10,
            "graphad_low": 2.0,
            "graphad_high": 5.0,
            "gap_low": 0.10,
            "gap_high": 0.50,
        }

    df = read_csv(base_path)
    rep_rf_col = f"p_rf_seed{REPRESENTATIVE_SEED}"
    rep_xgb_col = f"p_xgb_seed{REPRESENTATIVE_SEED}"
    rep_tcn_col = f"p_tcn_seed{REPRESENTATIVE_SEED}"
    if {rep_rf_col, rep_xgb_col, rep_tcn_col}.issubset(df.columns):
        ref = df[KEY_COLS].copy()
        ref["p_rf"] = df[rep_rf_col]
        ref["p_xgb"] = df[rep_xgb_col]
        ref["p_tcn"] = df[rep_tcn_col]
        for col in graphad_feature_columns(df.columns):
            ref[col] = df[col]
        routing = build_routing_features(ref, cfg)
        ref = pd.concat([ref, routing], axis=1)
    else:
        needed_cols = KEY_COLS + ["p_rf", "p_xgb", "p_tcn", "p_utar_base", "ensemble_entropy", "model_discrepancy"]
        for col in graphad_feature_columns(df.columns):
            if col not in needed_cols:
                needed_cols.append(col)
        ref = df[[col for col in needed_cols if col in df.columns]].copy()
        if "ensemble_mean" not in ref.columns:
            ref["ensemble_mean"] = ref[["p_rf", "p_xgb", "p_tcn"]].mean(axis=1)
        if "temporal_weight" not in ref.columns:
            routing = build_routing_features(ref[["source_file", "fault_id", "run_id", "p_rf", "p_xgb", "p_tcn"]], cfg)
            ref["temporal_weight"] = routing["temporal_weight"]

    entropy_low = float(ref["ensemble_entropy"].quantile(0.35))
    entropy_high = float(ref["ensemble_entropy"].quantile(0.75))
    discrepancy_low = float(ref["model_discrepancy"].quantile(0.35))
    discrepancy_high = float(ref["model_discrepancy"].quantile(0.75))
    graphad_low = float(ref["graphad_score"].quantile(0.35)) if "graphad_score" in ref.columns else 2.0
    graphad_high = float(ref["graphad_score"].quantile(0.75)) if "graphad_score" in ref.columns else 5.0
    gap_low = float(ref["graphad_top1_gap"].quantile(0.35)) if "graphad_top1_gap" in ref.columns else 0.10
    gap_high = float(ref["graphad_top1_gap"].quantile(0.75)) if "graphad_top1_gap" in ref.columns else 0.50

    state = {
        "tau": tau,
        "margin": margin,
        "entropy_low": entropy_low,
        "entropy_high": entropy_high,
        "discrepancy_low": discrepancy_low,
        "discrepancy_high": discrepancy_high,
        "graphad_low": graphad_low,
        "graphad_high": graphad_high,
        "gap_low": gap_low,
        "gap_high": gap_high,
    }

    def pick(mask: pd.Series, order_cols: list[str], ascending: list[bool]) -> pd.Series:
        cand = ref[mask].copy()
        if cand.empty:
            cand = ref.copy()
        cand = cand.sort_values(order_cols, ascending=ascending)
        return cand.iloc[0]

    ex1 = pick(
        (ref["y_true"] == 0)
        & (ref["p_utar_base"] < tau - margin)
        & (ref["ensemble_entropy"] <= entropy_low)
        & (ref["model_discrepancy"] <= discrepancy_low)
        & (ref["graphad_score"] <= graphad_low),
        ["p_utar_base", "ensemble_entropy", "model_discrepancy", "graphad_score"],
        [True, True, True, True],
    )
    ex2 = pick(
        (ref["y_true"] == 1)
        & ((ref["p_utar_base"] - tau).abs() <= margin)
        & (ref["ensemble_entropy"] >= entropy_high),
        ["ensemble_entropy", "graphad_score", "model_discrepancy"],
        [False, False, False],
    )
    ex3_mask = (
        (ref["y_true"] == 1)
        & (ref["p_utar_base"] < tau)
        & (ref["graphad_score"] >= graphad_high)
    )
    ex3 = pick(
        ex3_mask,
        ["graphad_score", "p_utar_base", "graphad_top1_gap"],
        [False, False, False],
    )
    state["examples"] = [ex1, ex2, ex3]
    return state


def _derived_prompt_context(row: pd.Series) -> dict[str, str]:
    state = _prompt_reference_state()
    tau = float(state["tau"])
    margin = float(state["margin"])
    p_utar = _safe_float(row, "p_utar_base", 0.0)
    if abs(p_utar - tau) <= margin:
        utar_side = "near_tau"
    elif p_utar > tau:
        utar_side = "above_tau"
    else:
        utar_side = "below_tau"

    scores = {
        "rf": _safe_float(row, "p_rf", 0.0),
        "xgb": _safe_float(row, "p_xgb", 0.0),
        TEMPORAL_MODEL_LABEL: _safe_float(row, "p_tcn", 0.0),
    }
    detector_anomaly_votes = int(sum(value >= 0.5 for value in scores.values()))
    max_detector_name, max_detector_value = max(scores.items(), key=lambda kv: kv[1])

    entropy_level = _categorize_level(_safe_float(row, "ensemble_entropy", 0.0), state["entropy_low"], state["entropy_high"])
    discrepancy_level = _categorize_level(_safe_float(row, "model_discrepancy", 0.0), state["discrepancy_low"], state["discrepancy_high"])
    graphad_support = _categorize_level(_safe_float(row, "graphad_score", 0.0), state["graphad_low"], state["graphad_high"])

    gap = _safe_float(row, "graphad_top1_gap", 0.0)
    if gap <= state["gap_low"]:
        graphad_concentration = "diffuse"
    elif gap >= state["gap_high"]:
        graphad_concentration = "concentrated"
    else:
        graphad_concentration = "moderate"

    coherence_score = 0
    if _is_present_text(_safe_text(row, "graphad_top1_neighbors")):
        coherence_score += 1
    if _is_present_text(_safe_text(row, "graphad_topology")):
        coherence_score += 1
    if gap >= state["gap_high"]:
        coherence_score += 1
    if coherence_score >= 3:
        sensor_coherence = "strong"
    elif coherence_score == 2:
        sensor_coherence = "moderate"
    else:
        sensor_coherence = "weak"

    return {
        "utar_side": utar_side,
        "detector_anomaly_votes": str(detector_anomaly_votes),
        "max_detector": f"{max_detector_name}({max_detector_value:.4f})",
        "entropy_level": entropy_level,
        "discrepancy_level": discrepancy_level,
        "graphad_support": graphad_support,
        "graphad_concentration": graphad_concentration,
        "sensor_coherence": sensor_coherence,
    }


def _format_example_block(example_idx: int, row: pd.Series, decision: str) -> str:
    derived = _derived_prompt_context(row)
    graphad_score = _safe_float(row, "graphad_score", 0.0)
    graphad_top1_score = _safe_float(row, "graphad_top1_score", 0.0)
    graphad_top2_score = _safe_float(row, "graphad_top2_score", 0.0)
    graphad_top1_gap = _safe_float(row, "graphad_top1_gap", 0.0)
    graphad_topk_mean = _safe_float(row, "graphad_topk_mean", 0.0)
    return (
        f"[Few-shot Example {example_idx}]\n"
        "Case:\n"
        f"utar_side={derived['utar_side']}\n"
        f"detector_anomaly_votes={derived['detector_anomaly_votes']}\n"
        f"max_detector={derived['max_detector']}\n"
        f"entropy_level={derived['entropy_level']}\n"
        f"discrepancy_level={derived['discrepancy_level']}\n"
        f"graphad_support={derived['graphad_support']}\n"
        f"graphad_concentration={derived['graphad_concentration']}\n"
        f"sensor_coherence={derived['sensor_coherence']}\n"
        f"rf={_safe_float(row, 'p_rf', 0.0):.6f}\n"
        f"xgb={_safe_float(row, 'p_xgb', 0.0):.6f}\n"
        f"{TEMPORAL_MODEL_LABEL}={_safe_float(row, 'p_tcn', 0.0):.6f}\n"
        f"utar_base={_safe_float(row, 'p_utar_base', 0.0):.6f}\n"
        f"ensemble_mean={_safe_float(row, 'ensemble_mean', _safe_float(row, 'p_ensemble', 0.0)):.6f}\n"
        f"temporal_weight={_safe_float(row, 'temporal_weight', 0.0):.6f}\n"
        f"ensemble_entropy={_safe_float(row, 'ensemble_entropy', 0.0):.6f}\n"
        f"model_discrepancy={_safe_float(row, 'model_discrepancy', 0.0):.6f}\n"
        f"graphad_score={graphad_score:.6f}\n"
        f"top1_sensor={_safe_text(row, 'graphad_top1_sensor')}\n"
        f"top1_score={graphad_top1_score:.6f}\n"
        f"top1_z={_safe_float(row, 'graphad_top1_z', 0.0):.6f}\n"
        f"top1_trend={_safe_float(row, 'graphad_top1_trend', 0.0):.6f}\n"
        f"top1_fluct={_safe_float(row, 'graphad_top1_fluct', 0.0):.6f}\n"
        f"top1_neighbors={_safe_text(row, 'graphad_top1_neighbors')}\n"
        f"top2_sensor={_safe_text(row, 'graphad_top2_sensor')}\n"
        f"top2_score={graphad_top2_score:.6f}\n"
        f"top1_gap={graphad_top1_gap:.6f}\n"
        f"topk_mean={graphad_topk_mean:.6f}\n"
        f"candidate_sensors={_safe_text(row, 'graphad_topk_sensors')}\n"
        f"candidate_scores={_safe_text(row, 'graphad_topk_scores')}\n"
        f"candidate_topology={_safe_text(row, 'graphad_topology')}\n"
        "Output:\n"
        f'{{"decision": "{decision}"}}\n'
    )


def build_llm_prompt(row: pd.Series) -> str:
    derived = _derived_prompt_context(row)
    examples = _prompt_reference_state()["examples"]
    graphad_score = _safe_float(row, "graphad_score", 0.0)
    graphad_top1_score = _safe_float(row, "graphad_top1_score", 0.0)
    graphad_top2_score = _safe_float(row, "graphad_top2_score", 0.0)
    graphad_top1_gap = _safe_float(row, "graphad_top1_gap", 0.0)
    graphad_topk_mean = _safe_float(row, "graphad_topk_mean", 0.0)
    return (
        "[System Role]\n"
        "You are an expert Tennessee Eastman process engineer.\n"
        "Your task is to make the final routing-time decision for one ambiguous sample escalated near the UTAR decision boundary.\n"
        "[Output Rule]\n"
        'Return JSON only with:\n{"decision": "normal"}\nor\n{"decision": "anomaly"}\n\n'
        "Do not output any explanation, reasoning, confidence, or extra keys.\n"
        "[Task Context]\n"
        "This sample was escalated because it is near the UTAR boundary and is not a straightforward case.\n"
        "Your job is to resolve this ambiguity using base detector evidence and GraphAD+ structural evidence.\n"
        "[Decision Objective]\n"
        "At this routing stage, missing a true anomaly is more costly than flagging a borderline anomaly.\n"
        'Do not default to "normal" for boundary samples.\n'
        "Use base detectors as the primary signal and GraphAD+ as supporting structural evidence.\n"
        "[Decision Policy]\n"
        'Output {"decision": "anomaly"} when one strong anomaly signal or multiple moderate anomaly signals are present.\n\n'
        "Treat the following as strong anomaly evidence:\n"
        "- utar_side is near_tau or above_tau and detector_anomaly_votes >= 1\n"
        "- entropy_level is high\n"
        "- discrepancy_level is high\n"
        "- graphad_support is strong\n"
        "- graphad_concentration is concentrated\n"
        "- sensor_coherence is strong\n\n"
        'Output {"decision": "anomaly"} if ANY of the following holds:\n'
        "1. One strong anomaly signal is present and the rest of the evidence is not clearly normal.\n"
        "2. Two or more moderate anomaly signals appear together.\n"
        "3. Detector-side evidence is mixed, but GraphAD+ structural evidence is clearly anomaly-oriented.\n"
        "4. GraphAD+ evidence is moderate or strong and coherent on top sensors for a boundary sample.\n"
        "5. The overall evidence is ambiguous but leans toward anomaly rather than normal.\n\n"
        'Output {"decision": "normal"} only when the combined evidence is more convincingly normal than anomaly.\n'
        "[Guardrails]\n"
        '- Do not require perfect detector agreement for "anomaly".\n'
        "- Do not require both detector disagreement and GraphAD+ evidence if one side is already strongly anomaly-oriented.\n"
        '- For near-boundary cases, plausible anomaly evidence is sufficient for "anomaly".\n'
        '- Choose "normal" only when anomaly evidence is weak overall and GraphAD+ support is not meaningful.\n'
        f"{_format_example_block(1, examples[0], 'normal') if len(examples) > 0 else ''}"
        f"{_format_example_block(2, examples[1], 'anomaly') if len(examples) > 1 else ''}"
        f"{_format_example_block(3, examples[2], 'anomaly') if len(examples) > 2 else ''}"
        "[Derived Context]\n"
        f"utar_side={derived['utar_side']}\n"
        f"detector_anomaly_votes={derived['detector_anomaly_votes']}\n"
        f"max_detector={derived['max_detector']}\n"
        f"entropy_level={derived['entropy_level']}\n"
        f"discrepancy_level={derived['discrepancy_level']}\n"
        f"graphad_support={derived['graphad_support']}\n"
        f"graphad_concentration={derived['graphad_concentration']}\n"
        f"sensor_coherence={derived['sensor_coherence']}\n"
        "[Input Context / Detection]\n"
        f"rf={_safe_float(row, 'p_rf', 0.0):.6f}\n"
        f"xgb={_safe_float(row, 'p_xgb', 0.0):.6f}\n"
        f"{TEMPORAL_MODEL_LABEL}={_safe_float(row, 'p_tcn', 0.0):.6f}\n"
        f"utar_base={_safe_float(row, 'p_utar_base', 0.0):.6f}\n"
        f"ensemble_mean={_safe_float(row, 'ensemble_mean', _safe_float(row, 'p_ensemble', 0.0)):.6f}\n"
        f"temporal_weight={_safe_float(row, 'temporal_weight', 0.0):.6f}\n"
        f"ensemble_entropy={_safe_float(row, 'ensemble_entropy', 0.0):.6f}\n"
        f"model_discrepancy={_safe_float(row, 'model_discrepancy', 0.0):.6f}\n"
        "[Input Context / GraphAD+]\n"
        f"graphad_score={graphad_score:.6f}\n"
        f"top1_sensor={_safe_text(row, 'graphad_top1_sensor')}\n"
        f"top1_score={graphad_top1_score:.6f}\n"
        f"top1_z={_safe_float(row, 'graphad_top1_z', 0.0):.6f}\n"
        f"top1_trend={_safe_float(row, 'graphad_top1_trend', 0.0):.6f}\n"
        f"top1_fluct={_safe_float(row, 'graphad_top1_fluct', 0.0):.6f}\n"
        f"top1_neighbors={_safe_text(row, 'graphad_top1_neighbors')}\n"
        f"top2_sensor={_safe_text(row, 'graphad_top2_sensor')}\n"
        f"top2_score={graphad_top2_score:.6f}\n"
        f"top1_gap={graphad_top1_gap:.6f}\n"
        f"topk_mean={graphad_topk_mean:.6f}\n"
        f"candidate_sensors={_safe_text(row, 'graphad_topk_sensors')}\n"
        f"candidate_scores={_safe_text(row, 'graphad_topk_scores')}\n"
        f"candidate_topology={_safe_text(row, 'graphad_topology')}\n"
    )


def _graphad_adjust_scalar(row: pd.Series) -> float:
    score = _safe_float(row, "graphad_score", 0.0)
    gap = _safe_float(row, "graphad_top1_gap", 0.0)
    return float(0.05 * np.tanh(score / 6.0) + 0.03 * np.tanh(gap / 3.0))


def _graphad_adjust_vector(df: pd.DataFrame) -> np.ndarray:
    if "graphad_score" not in df.columns:
        return np.zeros(len(df), dtype=float)
    score = df["graphad_score"].fillna(0.0).to_numpy(dtype=float)
    gap = df["graphad_top1_gap"].fillna(0.0).to_numpy(dtype=float) if "graphad_top1_gap" in df.columns else np.zeros(len(df), dtype=float)
    return 0.05 * np.tanh(score / 6.0) + 0.03 * np.tanh(gap / 3.0)


def zero_usage() -> dict[str, float]:
    return {
        "prompt_tokens": 0.0,
        "completion_tokens": 0.0,
        "total_tokens": 0.0,
        "total_latency_ms": 0.0,
        "billed_prompt_tokens": 0.0,
        "billed_completion_tokens": 0.0,
        "billed_total_tokens": 0.0,
        "billed_total_latency_ms": 0.0,
        "billed_api_call": 0.0,
        "cache_hit": 0.0,
    }


def _cached_usage(usage: dict[str, float]) -> dict[str, float]:
    cached = dict(usage)
    cached["billed_prompt_tokens"] = 0.0
    cached["billed_completion_tokens"] = 0.0
    cached["billed_total_tokens"] = 0.0
    cached["billed_total_latency_ms"] = 0.0
    cached["billed_api_call"] = 0.0
    cached["cache_hit"] = 1.0
    return cached


def _decision_to_probability(decision: str) -> float:
    normalized = str(decision).strip().lower()
    if normalized == "anomaly":
        return 1.0
    if normalized == "normal":
        return 0.0
    raise ValueError(f"Unsupported LLM decision: {decision}")


def llm_stub_probability(row: pd.Series) -> float:
    base = float(row["p_utar_base"])
    disagreement = float(row["model_discrepancy"]) if "model_discrepancy" in row else float(np.std([row["p_rf"], row["p_xgb"], row["p_tcn"]]))
    entropy = float(row["ensemble_entropy"]) if "ensemble_entropy" in row else float(
        -(np.clip(np.mean([row["p_rf"], row["p_xgb"], row["p_tcn"]]), 1e-8, 1 - 1e-8) * np.log2(np.clip(np.mean([row["p_rf"], row["p_xgb"], row["p_tcn"]]), 1e-8, 1 - 1e-8))
          + (1 - np.clip(np.mean([row["p_rf"], row["p_xgb"], row["p_tcn"]]), 1e-8, 1 - 1e-8)) * np.log2(np.clip(1 - np.mean([row["p_rf"], row["p_xgb"], row["p_tcn"]]), 1e-8, 1 - 1e-8)))
    )
    vote = float(np.mean([(row["p_rf"] >= 0.5), (row["p_xgb"] >= 0.5), (row["p_tcn"] >= 0.5)]))
    adjust = 0.08 + 0.18 * disagreement + 0.12 * max(0.0, entropy - 0.85)
    if vote >= 2 / 3:
        base = min(1.0, base + adjust)
    else:
        base = max(0.0, base - adjust)
    direction = 1.0 if base >= 0.5 else -1.0
    base = float(np.clip(base + direction * _graphad_adjust_scalar(row), 0.0, 1.0))
    return float(base)


def llm_stub_probability_batch(df: pd.DataFrame) -> np.ndarray:
    base = df["p_utar_base"].to_numpy(dtype=float)
    scores = df[["p_rf", "p_xgb", "p_tcn"]].to_numpy(dtype=float)
    disagreement = df["model_discrepancy"].to_numpy(dtype=float) if "model_discrepancy" in df.columns else scores.std(axis=1)
    entropy = (
        df["ensemble_entropy"].to_numpy(dtype=float)
        if "ensemble_entropy" in df.columns
        else -(np.clip(scores.mean(axis=1), 1e-8, 1 - 1e-8) * np.log2(np.clip(scores.mean(axis=1), 1e-8, 1 - 1e-8))
               + np.clip(1 - scores.mean(axis=1), 1e-8, 1 - 1e-8) * np.log2(np.clip(1 - scores.mean(axis=1), 1e-8, 1 - 1e-8)))
    )
    vote = (scores >= 0.5).mean(axis=1)
    out = base.copy()
    adjust = 0.08 + 0.18 * disagreement + 0.12 * np.maximum(0.0, entropy - 0.85)
    pos = vote >= (2 / 3)
    out[pos] = np.minimum(1.0, out[pos] + adjust[pos])
    out[~pos] = np.maximum(0.0, out[~pos] - adjust[~pos])
    direction = np.where(out >= 0.5, 1.0, -1.0)
    out = np.clip(out + direction * _graphad_adjust_vector(df), 0.0, 1.0)
    return out


class LLMProbabilityRunner:
    def __init__(self, cfg: dict, force_stub: bool = False):
        self.llm_cfg = cfg.get("llm", {})
        load_dotenv(PROJECT_ROOT / ".env")
        self.api_key = os.getenv(str(self.llm_cfg.get("api_env_key", "OPENAI_API_KEY")), "")
        self.model = os.getenv(str(self.llm_cfg.get("model_env_key", "OPENAI_MODEL")), str(self.llm_cfg.get("model", "gpt-4o-mini")))
        self.use_openai = (not force_stub) and bool(self.llm_cfg.get("enabled", False)) and str(self.llm_cfg.get("mode", "stub")).lower() == "openai" and self.api_key
        self.temperature = float(self.llm_cfg.get("temperature", 0.0))
        self.timeout_sec = int(self.llm_cfg.get("timeout_sec", 30))
        self.max_retries = int(self.llm_cfg.get("max_retries", 4))
        self.retry_backoff_sec = float(self.llm_cfg.get("retry_backoff_sec", 2.0))
        self.request_pause_sec = float(self.llm_cfg.get("request_pause_sec", 0.0))
        self.progress_every = int(self.llm_cfg.get("progress_every", 50))
        self.allow_fallback = bool(self.llm_cfg.get("allow_stub_fallback", False))
        self.error_log_path = METRIC_DIR / "selective_llm_errors.log"
        self.persist_cache = bool(self.llm_cfg.get("persist_cache", True))
        self.cache_path = METRIC_DIR / str(self.llm_cfg.get("persist_cache_filename", "selective_llm_response_cache.jsonl"))
        self.disabled = False
        self.response_cache: dict[str, tuple[float, dict[str, float]]] = {}
        if self.persist_cache:
            self._load_persistent_cache()

    @property
    def is_stub(self) -> bool:
        return not self.use_openai or self.disabled

    def _cache_key(self, prompt: str) -> str:
        payload = f"{self.model}\n{prompt}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _normalize_usage(self, usage: dict[str, float]) -> dict[str, float]:
        prompt_tokens = float(usage.get("prompt_tokens", 0.0) or 0.0)
        completion_tokens = float(usage.get("completion_tokens", 0.0) or 0.0)
        total_tokens = float(usage.get("total_tokens", 0.0) or (prompt_tokens + completion_tokens))
        total_latency_ms = float(usage.get("total_latency_ms", 0.0) or 0.0)
        billed_prompt_tokens = float(usage.get("billed_prompt_tokens", prompt_tokens) or 0.0)
        billed_completion_tokens = float(usage.get("billed_completion_tokens", completion_tokens) or 0.0)
        billed_total_tokens = float(usage.get("billed_total_tokens", billed_prompt_tokens + billed_completion_tokens) or 0.0)
        billed_total_latency_ms = float(usage.get("billed_total_latency_ms", total_latency_ms) or 0.0)
        billed_api_call = float(usage.get("billed_api_call", 1.0 if billed_total_tokens > 0 else 0.0) or 0.0)
        cache_hit = float(usage.get("cache_hit", 0.0) or 0.0)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "total_latency_ms": total_latency_ms,
            "billed_prompt_tokens": billed_prompt_tokens,
            "billed_completion_tokens": billed_completion_tokens,
            "billed_total_tokens": billed_total_tokens,
            "billed_total_latency_ms": billed_total_latency_ms,
            "billed_api_call": billed_api_call,
            "cache_hit": cache_hit,
        }

    def _load_persistent_cache(self) -> None:
        if not self.cache_path.exists():
            return
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if payload.get("model") != self.model:
                        continue
                    key = str(payload.get("key", "")).strip()
                    if not key:
                        continue
                    prob = float(payload.get("probability", 0.0))
                    usage = self._normalize_usage(payload.get("usage", {}))
                    self.response_cache[key] = (prob, usage)
        except Exception:
            # If the cache file is malformed, keep the current run usable.
            pass

    def _append_persistent_cache(self, key: str, prob: float, usage: dict[str, float]) -> None:
        if not self.persist_cache:
            return
        record = {
            "key": key,
            "model": self.model,
            "probability": float(prob),
            "usage": self._normalize_usage(usage),
        }
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    def _call_openai(self, row: pd.Series) -> tuple[float, dict[str, float]]:
        prompt = build_llm_prompt(row)
        payload = {
            "model": self.model,
            "input": prompt,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "decision_output",
                    "schema": {
                        "type": "object",
                        "properties": {"decision": {"type": "string", "enum": ["normal", "anomaly"]}},
                        "required": ["decision"],
                        "additionalProperties": False,
                    },
                }
            },
            "temperature": self.temperature,
        }
        req = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            if self.request_pause_sec > 0:
                time.sleep(self.request_pause_sec)
            started = time.perf_counter()
            print(
                f"[openai] request start attempt={attempt + 1}/{self.max_retries + 1} "
                f"sample_idx={int(row['sample_idx'])} run_id={int(row['run_id'])} phase={row['phase']}",
                flush=True,
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                text = self._extract_output_text(body)
                parsed = json.loads(text) if isinstance(text, str) else text
                prob = _decision_to_probability(str(parsed.get("decision")))
                usage = body.get("usage") or {}
                print(
                    f"[openai] request success attempt={attempt + 1}/{self.max_retries + 1} "
                    f"sample_idx={int(row['sample_idx'])} latency_ms={elapsed_ms:.1f} "
                    f"input_tokens={usage.get('input_tokens', 0)} output_tokens={usage.get('output_tokens', 0)}",
                    flush=True,
                )
                prompt_tokens = float(usage.get("input_tokens", 0.0) or 0.0)
                completion_tokens = float(usage.get("output_tokens", 0.0) or 0.0)
                total_tokens = float(usage.get("total_tokens", 0.0) or (prompt_tokens + completion_tokens))
                return float(np.clip(prob, 0.0, 1.0)), {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "total_latency_ms": float(elapsed_ms),
                    "billed_prompt_tokens": prompt_tokens,
                    "billed_completion_tokens": completion_tokens,
                    "billed_total_tokens": total_tokens,
                    "billed_total_latency_ms": float(elapsed_ms),
                    "billed_api_call": 1.0,
                    "cache_hit": 0.0,
                }
            except urllib.error.HTTPError as exc:
                status = getattr(exc, "code", None)
                last_exc = RuntimeError(f"HTTPError status={status}: {self._safe_http_error_body(exc)}")
                print(
                    f"[openai] request http_error attempt={attempt + 1}/{self.max_retries + 1} "
                    f"sample_idx={int(row['sample_idx'])} status={status}",
                    flush=True,
                )
                if attempt >= self.max_retries or status not in {408, 409, 429, 500, 502, 503, 504}:
                    raise last_exc
            except (urllib.error.URLError, TimeoutError, socket.timeout, json.JSONDecodeError, ValueError) as exc:
                last_exc = exc
                print(
                    f"[openai] request retryable_error attempt={attempt + 1}/{self.max_retries + 1} "
                    f"sample_idx={int(row['sample_idx'])} error={type(exc).__name__}: {exc}",
                    flush=True,
                )
                if attempt >= self.max_retries:
                    raise
            time.sleep(self.retry_backoff_sec * (2 ** attempt))
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("OpenAI call failed without a captured exception.")

    def _safe_http_error_body(self, exc: urllib.error.HTTPError) -> str:
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = "<unavailable>"
        return body[:1000]

    def _extract_output_text(self, body: dict) -> str:
        if "output_text" in body and body["output_text"]:
            return str(body["output_text"])
        for output_item in body.get("output", []):
            for content_item in output_item.get("content", []):
                if "text" in content_item and content_item["text"]:
                    return str(content_item["text"])
        raise ValueError(f"Could not locate JSON text in Responses API payload: keys={list(body.keys())}")

    def _log_error(self, row: pd.Series, exc: Exception) -> None:
        payload = {
            "model": self.model,
            "sample_idx": int(row["sample_idx"]),
            "fault_id": int(row["fault_id"]),
            "run_id": int(row["run_id"]),
            "phase": str(row["phase"]),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "timeout_sec": self.timeout_sec,
            "max_retries": self.max_retries,
            "traceback": traceback.format_exc(),
        }
        with open(self.error_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def probability_with_usage(self, row: pd.Series) -> tuple[float, dict[str, float]]:
        if self.is_stub:
            return llm_stub_probability(row), zero_usage()
        prompt = build_llm_prompt(row)
        cache_key = self._cache_key(prompt)
        if cache_key in self.response_cache:
            cached_prob, cached_usage = self.response_cache[cache_key]
            return cached_prob, _cached_usage(cached_usage)
        try:
            result = self._call_openai(row)
            self.response_cache[cache_key] = result
            self._append_persistent_cache(cache_key, result[0], result[1])
            return result
        except Exception as exc:
            self._log_error(row, exc)
            if self.allow_fallback:
                self.disabled = True
                return llm_stub_probability(row), zero_usage()
            raise RuntimeError(
                "OpenAI call failed during selective_llm_eval. "
                f"See {self.error_log_path} for details. "
                "Set llm.allow_stub_fallback: true only if you intentionally want silent fallback."
            ) from exc

    def apply(self, df: pd.DataFrame, progress_label: str = "") -> tuple[np.ndarray, pd.DataFrame]:
        if len(df) == 0:
            usage_df = pd.DataFrame(
                columns=[
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                    "total_latency_ms",
                    "billed_prompt_tokens",
                    "billed_completion_tokens",
                    "billed_total_tokens",
                    "billed_total_latency_ms",
                    "billed_api_call",
                    "cache_hit",
                ],
                index=df.index,
            )
            return np.array([], dtype=float), usage_df
        if self.is_stub:
            probs = llm_stub_probability_batch(df)
            usage_df = pd.DataFrame([zero_usage()] * len(df), index=df.index)
            return probs, usage_df

        label = progress_label.strip() or "llm"
        print(f"[{label}] starting {len(df):,} OpenAI calls", flush=True)
        probs = []
        usages = []
        for idx, (_, row) in enumerate(df.iterrows(), start=1):
            prob, usage = self.probability_with_usage(row)
            probs.append(prob)
            usages.append(usage)
            if self.progress_every > 0 and (idx == 1 or idx % self.progress_every == 0 or idx == len(df)):
                print(
                    f"[{label}] completed {idx:,}/{len(df):,} calls "
                    f"(sample_idx={int(row['sample_idx'])}, run_id={int(row['run_id'])})",
                    flush=True,
                )
        usage_df = pd.DataFrame(usages, index=df.index)
        return np.asarray(probs, dtype=float), usage_df


def build_llm_runner(cfg: dict, force_stub: bool = False) -> LLMProbabilityRunner:
    return LLMProbabilityRunner(cfg, force_stub=force_stub)


def build_llm_probability_fn(cfg: dict) -> Callable[[pd.Series], float]:
    llm_cfg = cfg.get("llm", {})
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv(str(llm_cfg.get("api_env_key", "OPENAI_API_KEY")), "")
    model = os.getenv(str(llm_cfg.get("model_env_key", "OPENAI_MODEL")), str(llm_cfg.get("model", "gpt-4o-mini")))
    use_openai = bool(llm_cfg.get("enabled", False)) and str(llm_cfg.get("mode", "stub")).lower() == "openai" and api_key
    if not use_openai:
        return llm_stub_probability

    temperature = float(llm_cfg.get("temperature", 0.0))
    timeout_sec = int(llm_cfg.get("timeout_sec", 30))
    state = {"disabled": False}

    def openai_probability(row: pd.Series) -> float:
        if state["disabled"]:
            return llm_stub_probability(row)
        prompt = build_llm_prompt(row)
        payload = {
            "model": model,
            "input": prompt,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "decision_output",
                    "schema": {
                        "type": "object",
                        "properties": {"decision": {"type": "string", "enum": ["normal", "anomaly"]}},
                        "required": ["decision"],
                        "additionalProperties": False,
                    },
                }
            },
            "temperature": temperature,
        }
        req = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            text = body.get("output", [{}])[0].get("content", [{}])[0].get("text", "{}")
            prob = _decision_to_probability(str(json.loads(text).get("decision")))
            return float(np.clip(prob, 0.0, 1.0))
        except Exception:
            state["disabled"] = True
            return llm_stub_probability(row)

    return openai_probability


def build_base_view(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = df[KEY_COLS].copy()
    out["p_rf"] = df["p_rf"]
    out["p_xgb"] = df["p_xgb"]
    out["p_tcn"] = df["p_tcn"]
    out["p_ensemble"] = df["p_ensemble"] if "p_ensemble" in df.columns else out[["p_rf", "p_xgb", "p_tcn"]].mean(axis=1)
    for col in graphad_feature_columns(df.columns):
        out[col] = df[col]
    routing = build_routing_features(out, cfg)
    return pd.concat([out, routing], axis=1)


def _without_graphad_context(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in graphad_feature_columns(out.columns):
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = 0.0
        else:
            out[col] = "n/a"
    return out


def _with_raw_graphad_context(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for smooth_col, raw_col in GRAPHAD_RAW_CONTEXT_MAP.items():
        if raw_col not in out.columns:
            continue
        out[smooth_col] = out[raw_col]
    return out


def get_seed_view(df: pd.DataFrame, seed: int, cfg: dict) -> pd.DataFrame:
    out = df[KEY_COLS].copy()
    out["p_rf"] = df[f"p_rf_seed{seed}"]
    out["p_xgb"] = df[f"p_xgb_seed{seed}"]
    out["p_tcn"] = df[f"p_tcn_seed{seed}"]
    out["p_ensemble"] = out[["p_rf", "p_xgb", "p_tcn"]].mean(axis=1)
    for col in graphad_feature_columns(df.columns):
        out[col] = df[col]
    routing = build_routing_features(out, cfg)
    return pd.concat([out, routing], axis=1)


def _active_modes(dataset_name: str) -> list[str]:
    if dataset_name == "main":
        return ["selective", "selective_no_graph", "no_llm", "selective_no_filter", "ensemble_only"]
    if dataset_name == "cost":
        return ["selective", "selective_no_filter", "no_llm", "full_llm"]
    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


def _shortcut_mask(df: pd.DataFrame, entropy_threshold: float, discrepancy_threshold: float) -> pd.Series:
    return (
        (df["gray_zone"] == 1)
        & (df["ensemble_entropy"] <= float(entropy_threshold))
    ).astype(int)


def apply_mode(
    df: pd.DataFrame,
    tau: float,
    margin: float,
    entropy_threshold: float,
    discrepancy_threshold: float,
    mode: str,
    llm_runner: LLMProbabilityRunner,
    progress_label: str = "",
) -> pd.DataFrame:
    out = df.copy()
    out["gray_zone"] = (np.abs(out["p_utar_base"] - tau) <= margin).astype(int)
    out["shortcut_filter"] = _shortcut_mask(out, entropy_threshold=entropy_threshold, discrepancy_threshold=discrepancy_threshold)
    out["xgb_shortcut"] = out["shortcut_filter"]
    for col in [
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "llm_latency_ms",
        "billed_prompt_tokens",
        "billed_completion_tokens",
        "billed_total_tokens",
        "billed_llm_latency_ms",
        "billed_api_call",
        "cache_hit",
    ]:
        out[col] = 0.0
    out["llm_decision"] = pd.Series(pd.NA, index=out.index, dtype="object")

    if mode == "ensemble_only":
        out["gray_zone"] = 0
        out["shortcut_filter"] = 0
        out["xgb_shortcut"] = 0
        out["llm_called"] = 0
        out["p_llm"] = np.nan
        out["p_final"] = out["p_ensemble"]
    elif mode == "selective":
        out["llm_called"] = ((out["gray_zone"] == 1) & (out["shortcut_filter"] == 0)).astype(int)
        out["p_llm"] = np.nan
        if out["llm_called"].sum() > 0:
            llm_idx = out.index[out["llm_called"] == 1]
            probs, usage_df = llm_runner.apply(out.loc[llm_idx], progress_label=progress_label or f"{mode}")
            out.loc[llm_idx, "p_llm"] = probs
            out.loc[llm_idx, "llm_decision"] = np.where(probs >= 0.5, "anomaly", "normal")
            out.loc[llm_idx, "prompt_tokens"] = usage_df["prompt_tokens"]
            out.loc[llm_idx, "completion_tokens"] = usage_df["completion_tokens"]
            out.loc[llm_idx, "total_tokens"] = usage_df["total_tokens"]
            out.loc[llm_idx, "llm_latency_ms"] = usage_df["total_latency_ms"]
            out.loc[llm_idx, "billed_prompt_tokens"] = usage_df["billed_prompt_tokens"]
            out.loc[llm_idx, "billed_completion_tokens"] = usage_df["billed_completion_tokens"]
            out.loc[llm_idx, "billed_total_tokens"] = usage_df["billed_total_tokens"]
            out.loc[llm_idx, "billed_llm_latency_ms"] = usage_df["billed_total_latency_ms"]
            out.loc[llm_idx, "billed_api_call"] = usage_df["billed_api_call"]
            out.loc[llm_idx, "cache_hit"] = usage_df["cache_hit"]
        out["p_final"] = np.where(out["llm_called"] == 1, out["p_llm"], out["p_utar_base"])
    elif mode == "selective_no_graph":
        out["llm_called"] = ((out["gray_zone"] == 1) & (out["shortcut_filter"] == 0)).astype(int)
        out["p_llm"] = np.nan
        if out["llm_called"].sum() > 0:
            llm_idx = out.index[out["llm_called"] == 1]
            llm_input = _with_raw_graphad_context(out.loc[llm_idx])
            probs, usage_df = llm_runner.apply(llm_input, progress_label=progress_label or f"{mode}")
            out.loc[llm_idx, "p_llm"] = probs
            out.loc[llm_idx, "llm_decision"] = np.where(probs >= 0.5, "anomaly", "normal")
            out.loc[llm_idx, "prompt_tokens"] = usage_df["prompt_tokens"]
            out.loc[llm_idx, "completion_tokens"] = usage_df["completion_tokens"]
            out.loc[llm_idx, "total_tokens"] = usage_df["total_tokens"]
            out.loc[llm_idx, "llm_latency_ms"] = usage_df["total_latency_ms"]
            out.loc[llm_idx, "billed_prompt_tokens"] = usage_df["billed_prompt_tokens"]
            out.loc[llm_idx, "billed_completion_tokens"] = usage_df["billed_completion_tokens"]
            out.loc[llm_idx, "billed_total_tokens"] = usage_df["billed_total_tokens"]
            out.loc[llm_idx, "billed_llm_latency_ms"] = usage_df["billed_total_latency_ms"]
            out.loc[llm_idx, "billed_api_call"] = usage_df["billed_api_call"]
            out.loc[llm_idx, "cache_hit"] = usage_df["cache_hit"]
        out["p_final"] = np.where(out["llm_called"] == 1, out["p_llm"], out["p_utar_base"])
    elif mode == "selective_no_filter":
        out["shortcut_filter"] = 0
        out["xgb_shortcut"] = 0
        out["llm_called"] = out["gray_zone"].astype(int)
        out["p_llm"] = np.nan
        if out["llm_called"].sum() > 0:
            llm_idx = out.index[out["llm_called"] == 1]
            probs, usage_df = llm_runner.apply(out.loc[llm_idx], progress_label=progress_label or f"{mode}")
            out.loc[llm_idx, "p_llm"] = probs
            out.loc[llm_idx, "llm_decision"] = np.where(probs >= 0.5, "anomaly", "normal")
            out.loc[llm_idx, "prompt_tokens"] = usage_df["prompt_tokens"]
            out.loc[llm_idx, "completion_tokens"] = usage_df["completion_tokens"]
            out.loc[llm_idx, "total_tokens"] = usage_df["total_tokens"]
            out.loc[llm_idx, "llm_latency_ms"] = usage_df["total_latency_ms"]
            out.loc[llm_idx, "billed_prompt_tokens"] = usage_df["billed_prompt_tokens"]
            out.loc[llm_idx, "billed_completion_tokens"] = usage_df["billed_completion_tokens"]
            out.loc[llm_idx, "billed_total_tokens"] = usage_df["billed_total_tokens"]
            out.loc[llm_idx, "billed_llm_latency_ms"] = usage_df["billed_total_latency_ms"]
            out.loc[llm_idx, "billed_api_call"] = usage_df["billed_api_call"]
            out.loc[llm_idx, "cache_hit"] = usage_df["cache_hit"]
        out["p_final"] = np.where(out["llm_called"] == 1, out["p_llm"], out["p_utar_base"])
    elif mode == "no_llm":
        out["llm_called"] = 0
        out["p_llm"] = np.nan
        out["p_final"] = out["p_utar_base"]
    elif mode == "full_llm":
        out["gray_zone"] = 1
        out["shortcut_filter"] = 0
        out["xgb_shortcut"] = 0
        out["llm_called"] = 1
        probs, usage_df = llm_runner.apply(out, progress_label=progress_label or f"{mode}")
        out["p_llm"] = probs
        out["llm_decision"] = np.where(probs >= 0.5, "anomaly", "normal")
        out["prompt_tokens"] = usage_df["prompt_tokens"]
        out["completion_tokens"] = usage_df["completion_tokens"]
        out["total_tokens"] = usage_df["total_tokens"]
        out["llm_latency_ms"] = usage_df["total_latency_ms"]
        out["billed_prompt_tokens"] = usage_df["billed_prompt_tokens"]
        out["billed_completion_tokens"] = usage_df["billed_completion_tokens"]
        out["billed_total_tokens"] = usage_df["billed_total_tokens"]
        out["billed_llm_latency_ms"] = usage_df["billed_total_latency_ms"]
        out["billed_api_call"] = usage_df["billed_api_call"]
        out["cache_hit"] = usage_df["cache_hit"]
        out["p_final"] = out["p_llm"]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    out["decision_source"] = np.where(
        out["llm_called"] == 1,
        "llm",
        np.where((out["gray_zone"] == 1) & (out["shortcut_filter"] == 1), "entropy_shortcut", np.where(mode == "ensemble_only", "ensemble", "direct")),
    )
    out["final_decision"] = np.where(out["p_final"] >= tau, "anomaly", "normal")
    return out


def _load_base_runtime_lookup() -> dict:
    return load_base_runtime_summary(BASE_RUNTIME_SUMMARY_PATH)


def _dataset_base_latency_ms(runtime_lookup: dict, dataset_name: str) -> float:
    return get_base_runtime_stat(runtime_lookup, split=dataset_name, component=BASE_STACK_COMPONENT, field="total_latency_ms", default=0.0)


def cost_summary(df: pd.DataFrame, llm_cfg: dict) -> dict:
    n_calls = int(df["llm_called"].sum())
    in_cost = float(llm_cfg.get("input_cost_per_1m", 0.15))
    out_cost = float(llm_cfg.get("output_cost_per_1m", 0.60))
    avg_prompt = int(llm_cfg.get("avg_prompt_tokens", 700))
    avg_completion = int(llm_cfg.get("avg_completion_tokens", 120))
    latency_ms = float(llm_cfg.get("latency_ms_per_call", 900))
    has_actual_usage = {"prompt_tokens", "completion_tokens", "llm_latency_ms"}.issubset(df.columns) and (
        df["prompt_tokens"].sum() > 0 or df["completion_tokens"].sum() > 0 or df["llm_latency_ms"].sum() > 0
    )
    has_billed_usage = {"billed_prompt_tokens", "billed_completion_tokens", "billed_llm_latency_ms", "billed_api_call", "cache_hit"}.issubset(df.columns)
    if has_actual_usage:
        prompt_tokens = float(df["prompt_tokens"].sum())
        completion_tokens = float(df["completion_tokens"].sum())
        total_tokens = float(df["total_tokens"].sum()) if "total_tokens" in df.columns else prompt_tokens + completion_tokens
        total_latency_ms = float(df["llm_latency_ms"].sum())
    else:
        prompt_tokens = float(n_calls * avg_prompt)
        completion_tokens = float(n_calls * avg_completion)
        total_tokens = float(prompt_tokens + completion_tokens)
        total_latency_ms = float(n_calls * latency_ms)
    cost_usd = (prompt_tokens / 1_000_000) * in_cost + (completion_tokens / 1_000_000) * out_cost
    if has_billed_usage:
        billed_prompt_tokens = float(df["billed_prompt_tokens"].sum())
        billed_completion_tokens = float(df["billed_completion_tokens"].sum())
        billed_total_tokens = float(df["billed_total_tokens"].sum()) if "billed_total_tokens" in df.columns else billed_prompt_tokens + billed_completion_tokens
        billed_total_latency_ms = float(df["billed_llm_latency_ms"].sum())
        billed_llm_calls = int(df["billed_api_call"].sum())
        cache_hits = int(df["cache_hit"].sum())
    else:
        billed_prompt_tokens = prompt_tokens
        billed_completion_tokens = completion_tokens
        billed_total_tokens = total_tokens
        billed_total_latency_ms = total_latency_ms
        billed_llm_calls = n_calls
        cache_hits = 0
    billed_cost_usd = (billed_prompt_tokens / 1_000_000) * in_cost + (billed_completion_tokens / 1_000_000) * out_cost
    return {
        "llm_calls": n_calls,
        "billed_llm_calls": billed_llm_calls,
        "cache_hits": cache_hits,
        "cache_hit_rate": float(cache_hits / n_calls) if n_calls else 0.0,
        "prompt_tokens": int(round(prompt_tokens)),
        "completion_tokens": int(round(completion_tokens)),
        "total_tokens": int(round(total_tokens)),
        "cost_usd": float(cost_usd),
        "avg_cost_per_call_usd": float(cost_usd / n_calls) if n_calls else 0.0,
        "billed_prompt_tokens": int(round(billed_prompt_tokens)),
        "billed_completion_tokens": int(round(billed_completion_tokens)),
        "billed_total_tokens": int(round(billed_total_tokens)),
        "billed_cost_usd": float(billed_cost_usd),
        "billed_avg_cost_per_call_usd": float(billed_cost_usd / billed_llm_calls) if billed_llm_calls else 0.0,
        "llm_only_latency_ms": float(total_latency_ms),
        "llm_only_avg_latency_ms_per_sample": float(total_latency_ms / len(df)) if len(df) else 0.0,
        "billed_llm_only_latency_ms": float(billed_total_latency_ms),
        "billed_llm_only_avg_latency_ms_per_sample": float(billed_total_latency_ms / len(df)) if len(df) else 0.0,
        "uses_actual_api_usage": bool(has_actual_usage or billed_llm_calls > 0 or cache_hits > 0),
    }


def evaluate_frame(df: pd.DataFrame, tau: float, ref_recall: float) -> dict:
    m = binary_metrics(df["y_true"], df["p_final"], tau=tau)
    m["gray_ratio"] = float(df["gray_zone"].mean())
    m["llm_call_rate"] = float(df["llm_called"].mean())
    m["shortcut_filter_rate"] = float(df["shortcut_filter"].mean()) if "shortcut_filter" in df.columns else 0.0
    m["xgb_shortcut_rate"] = m["shortcut_filter_rate"]
    m["instability"] = instability_score(df["p_final"], df["run_id"], df["phase"])
    m["worst_case_recall"] = worst_case_recall(df["y_true"], df["p_final"], df["run_id"], tau=tau, window=50)
    m["prr"] = prr(ref_recall, m["recall"])
    m["ensemble_entropy_mean"] = float(df["ensemble_entropy"].mean()) if "ensemble_entropy" in df.columns else np.nan
    m["model_discrepancy_mean"] = float(df["model_discrepancy"].mean()) if "model_discrepancy" in df.columns else np.nan
    return m


def run_mode(
    base_df: pd.DataFrame,
    tau: float,
    margin: float,
    entropy_threshold: float,
    discrepancy_threshold: float,
    cfg: dict,
    mode: str,
    llm_runner: LLMProbabilityRunner,
    ref_recall: float | None,
    base_latency_ms: float,
    routing_feature_latency_ms: float,
    progress_label: str = "",
) -> tuple[pd.DataFrame, dict]:
    llm_cfg = cfg.get("llm", {})
    started_mode = time.perf_counter()
    out = apply_mode(
        base_df,
        tau=tau,
        margin=margin,
        entropy_threshold=entropy_threshold,
        discrepancy_threshold=discrepancy_threshold,
        mode=mode,
        llm_runner=llm_runner,
        progress_label=progress_label,
    )
    mode_wall_latency_ms = (time.perf_counter() - started_mode) * 1000.0
    metrics = evaluate_frame(out, tau=tau, ref_recall=ref_recall if ref_recall is not None else np.nan)
    metrics.update(cost_summary(out, llm_cfg))
    llm_only_latency_ms = float(metrics.get("llm_only_latency_ms", 0.0))
    routing_overhead_ms = max(mode_wall_latency_ms - llm_only_latency_ms, 0.0)
    total_latency_ms = float(base_latency_ms) + float(routing_feature_latency_ms) + float(routing_overhead_ms) + float(llm_only_latency_ms)
    metrics["base_latency_ms"] = float(base_latency_ms)
    metrics["routing_feature_latency_ms"] = float(routing_feature_latency_ms)
    metrics["routing_overhead_ms"] = float(routing_overhead_ms)
    metrics["mode_wall_latency_ms"] = float(mode_wall_latency_ms)
    metrics["total_latency_ms"] = float(total_latency_ms)
    metrics["avg_latency_ms_per_sample"] = float(total_latency_ms / len(out)) if len(out) else 0.0
    metrics["tau"] = tau
    metrics["gray_margin"] = margin
    metrics["entropy_threshold"] = entropy_threshold
    metrics["discrepancy_threshold"] = discrepancy_threshold
    metrics["mode"] = mode
    return out, metrics


def aggregate_seed_predictions(seed_outputs: list[pd.DataFrame]) -> pd.DataFrame:
    if len(seed_outputs) == 1:
        return seed_outputs[0].copy()
    merged = seed_outputs[0][KEY_COLS].copy()
    numeric_cols = [
        "p_rf",
        "p_xgb",
        "p_tcn",
        "p_ensemble",
        "p_temporal_norm",
        "temporal_weight",
        "rf_component",
        "xgb_component",
        "temporal_component",
        "ensemble_mean",
        "ensemble_entropy",
        "model_discrepancy",
        "p_utar_base",
        "p_final",
        "gray_zone",
        "shortcut_filter",
        "xgb_shortcut",
        "llm_called",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "llm_latency_ms",
        "billed_prompt_tokens",
        "billed_completion_tokens",
        "billed_total_tokens",
        "billed_llm_latency_ms",
        "billed_api_call",
        "cache_hit",
    ]
    for col in numeric_cols:
        if all(col in df.columns for df in seed_outputs):
            merged[col] = np.mean([df[col].to_numpy(dtype=float) for df in seed_outputs], axis=0)
    for col in graphad_feature_columns(seed_outputs[0].columns):
        merged[col] = seed_outputs[0][col].values
    merged["decision_source"] = seed_outputs[0]["decision_source"].values
    merged["gray_zone"] = (merged["gray_zone"] >= 0.5).astype(int)
    merged["shortcut_filter"] = (merged["shortcut_filter"] >= 0.5).astype(int)
    merged["xgb_shortcut"] = (merged["xgb_shortcut"] >= 0.5).astype(int)
    merged["llm_called"] = (merged["llm_called"] >= 0.5).astype(int)
    return merged


def summarize_rows(rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    summary_df = pd.DataFrame(rows).sort_values(["dataset", "q", "mode", "seed"]).reset_index(drop=True)
    metric_cols = [
        "f1",
        "recall",
        "precision",
        "roc_auc",
        "prr",
        "gray_ratio",
        "llm_call_rate",
        "worst_case_recall",
        "instability",
        "cost_usd",
        "billed_cost_usd",
        "base_latency_ms",
        "routing_feature_latency_ms",
        "routing_overhead_ms",
        "mode_wall_latency_ms",
        "llm_only_latency_ms",
        "llm_only_avg_latency_ms_per_sample",
        "billed_llm_only_latency_ms",
        "billed_llm_only_avg_latency_ms_per_sample",
        "total_latency_ms",
        "avg_latency_ms_per_sample",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "billed_prompt_tokens",
        "billed_completion_tokens",
        "billed_total_tokens",
        "llm_calls",
        "billed_llm_calls",
        "cache_hits",
        "cache_hit_rate",
        "uses_actual_api_usage",
        "xgb_shortcut_rate",
        "shortcut_filter_rate",
        "tau",
        "gray_margin",
        "entropy_threshold",
        "discrepancy_threshold",
        "ensemble_entropy_mean",
        "model_discrepancy_mean",
    ]
    summary_rows = []
    for (dataset, q, mode), group in summary_df.groupby(["dataset", "q", "mode"], dropna=False):
        row = {"dataset": dataset, "q": float(q), "mode": mode, "n_seeds": int(len(group))}
        for col in metric_cols:
            row[f"{col}_mean"] = float(group[col].mean()) if col in group.columns else np.nan
            row[f"{col}_std"] = float(group[col].std(ddof=1)) if col in group.columns and len(group) > 1 else 0.0
        summary_rows.append(row)
    summary_seed = pd.DataFrame(summary_rows).sort_values(["dataset", "q", "mode"]).reset_index(drop=True)
    return summary_df, summary_seed


def _reference_recall_map(rows: list[dict], dataset_name: str) -> dict[str, dict[int, float]]:
    df = pd.DataFrame(rows)
    ref: dict[str, dict[int, float]] = {}
    if df.empty:
        return ref
    sub = df[df["dataset"] == dataset_name]
    for _, row in sub.iterrows():
        ref.setdefault(str(row["mode"]), {})[int(row["seed"])] = float(row["recall"])
    return ref


def _gray_row_for_q(gray_grid: pd.DataFrame, q: float) -> pd.Series:
    row = gray_grid[np.isclose(gray_grid["q"], q)]
    if row.empty:
        raise KeyError(f"Gray-zone summary not found for q={q:.2f}")
    return row.iloc[0]


def evaluate_dataset_modes(
    dataset_name: str,
    base_pred: pd.DataFrame,
    tau: float,
    q: float,
    gray_row: pd.Series,
    cfg: dict,
    active_modes: list[str],
    llm_runner: LLMProbabilityRunner,
    base_runtime_lookup: dict | None = None,
    ref_recall_by_mode: dict[str, dict[int, float]] | None = None,
    seeds: list[int] | None = None,
    mode_seeds: dict[str, list[int]] | None = None,
) -> tuple[list[dict], dict[str, list[pd.DataFrame]]]:
    margin = float(gray_row["gray_margin_mean"])
    entropy_threshold = float(gray_row["entropy_threshold_mean"]) if "entropy_threshold_mean" in gray_row.index else 1.0
    discrepancy_threshold = float(gray_row["discrepancy_threshold_mean"]) if "discrepancy_threshold_mean" in gray_row.index else 1.0
    rows: list[dict] = []
    mode_outputs: dict[str, list[pd.DataFrame]] = {mode: [] for mode in active_modes}
    dataset_base_latency_ms = _dataset_base_latency_ms(base_runtime_lookup or {}, dataset_name)

    default_seeds = list(SEEDS if seeds is None else seeds)
    seed_view_cache: dict[int, tuple[pd.DataFrame, float]] = {}
    for mode in active_modes:
        mode_seed_list = list(mode_seeds.get(mode, default_seeds) if mode_seeds is not None else default_seeds)
        for seed in mode_seed_list:
            if seed not in seed_view_cache:
                started_seed_view = time.perf_counter()
                seed_df = get_seed_view(base_pred, seed=seed, cfg=cfg)
                routing_feature_latency_ms = (time.perf_counter() - started_seed_view) * 1000.0
                seed_view_cache[seed] = (seed_df, routing_feature_latency_ms)
            seed_df, routing_feature_latency_ms = seed_view_cache[seed]
            ref_recall = None if ref_recall_by_mode is None else ref_recall_by_mode.get(mode, {}).get(seed)
            out, metrics = run_mode(
                seed_df,
                tau=tau,
                margin=margin,
                entropy_threshold=entropy_threshold,
                discrepancy_threshold=discrepancy_threshold,
                cfg=cfg,
                mode=mode,
                llm_runner=llm_runner,
                ref_recall=ref_recall,
                base_latency_ms=dataset_base_latency_ms,
                routing_feature_latency_ms=routing_feature_latency_ms,
                progress_label=f"{dataset_name} mode={mode} q={q:.2f} seed={seed}",
            )
            if ref_recall is None and dataset_name == "val":
                metrics["prr"] = 1.0
            rows.append({"dataset": dataset_name, "q": float(q), "seed": seed, **metrics})
            mode_outputs[mode].append(out)

    return rows, mode_outputs


def collect_dataset_modes(
    dataset_name: str,
    base_pred: pd.DataFrame,
    q: float,
    tau: float,
    margin: float,
    cfg: dict,
    llm_runner: LLMProbabilityRunner,
) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    raise NotImplementedError("collect_dataset_modes is superseded by evaluate_dataset_modes.")


def _gray_margin_for_q(gray_grid: pd.DataFrame, q: float) -> float:
    row = gray_grid[np.isclose(gray_grid["q"], q)]
    if row.empty:
        raise KeyError(f"Gray-zone summary not found for q={q:.2f}")
    return float(row.iloc[0]["gray_margin_mean"])


def _prewarm_q_sweep_llm_cache(
    base_pred: pd.DataFrame,
    tau: float,
    gray_row: pd.Series,
    cfg: dict,
    llm_runner: LLMProbabilityRunner,
    seeds: list[int],
) -> None:
    if llm_runner.is_stub or not seeds:
        return

    margin = float(gray_row["gray_margin_mean"])
    entropy_threshold = float(gray_row["entropy_threshold_mean"]) if "entropy_threshold_mean" in gray_row.index else 1.0
    discrepancy_threshold = float(gray_row["discrepancy_threshold_mean"]) if "discrepancy_threshold_mean" in gray_row.index else 1.0
    q = float(gray_row["q"]) if "q" in gray_row.index else float("nan")

    # Warm the widest gray-zone once so the smaller-q selective runs can reuse
    # prompt-level cache hits instead of issuing duplicate OpenAI calls.
    for seed in seeds:
        seed_df = get_seed_view(base_pred, seed=seed, cfg=cfg)
        apply_mode(
            seed_df,
            tau=tau,
            margin=margin,
            entropy_threshold=entropy_threshold,
            discrepancy_threshold=discrepancy_threshold,
            mode="selective_no_filter",
            llm_runner=llm_runner,
            progress_label=f"main q-sweep prewarm q={q:.2f} seed={seed}",
        )


def _read_existing_summary_rows() -> list[dict]:
    path = METRIC_DIR / "selective_llm_seed_metrics.csv"
    if not path.exists():
        return []
    return read_csv(path).to_dict("records")


def _write_merged_summary(new_rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    existing_rows = _read_existing_summary_rows()
    merged_rows = existing_rows + new_rows
    if not merged_rows:
        empty = pd.DataFrame()
        write_csv(METRIC_DIR / "selective_llm_seed_metrics.csv", empty)
        write_csv(METRIC_DIR / "selective_llm_summary.csv", empty)
        return empty, empty

    summary_df = pd.DataFrame(merged_rows)
    dedup_cols = ["dataset", "q", "mode"]
    if "seed" in summary_df.columns:
        dedup_cols.append("seed")
    summary_df = summary_df.drop_duplicates(subset=dedup_cols, keep="last").sort_values(["dataset", "q", "mode"]).reset_index(drop=True)
    summary_rows = summary_df.to_dict("records")
    summary_df, summary_seed = summarize_rows(summary_rows)
    write_csv(METRIC_DIR / "selective_llm_seed_metrics.csv", summary_df)
    write_csv(METRIC_DIR / "selective_llm_summary.csv", summary_seed)
    return summary_df, summary_seed


def _merge_prediction_store(path: Path, new_df: pd.DataFrame) -> None:
    if path.exists():
        existing_df = read_csv(path)
        merged_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        merged_df = new_df.copy()
    dedup_cols = [col for col in ["q", "seed", "source_file", "fault_id", "run_id", "sample_idx"] if col in merged_df.columns]
    if dedup_cols:
        merged_df = merged_df.drop_duplicates(subset=dedup_cols, keep="last")
    sort_cols = [col for col in ["q", "seed", "source_file", "fault_id", "run_id", "sample_idx"] if col in merged_df.columns]
    if sort_cols:
        merged_df = merged_df.sort_values(sort_cols).reset_index(drop=True)
    write_csv(path, merged_df)


def _q_file_tag(q: float) -> str:
    return f"q{int(round(float(q) * 100)):03d}"


def run_main_eval() -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = read_yaml(CONFIG_DIR / "routing.yaml", default={})
    tau_info = read_json(METRIC_DIR / "thresholds.json")
    gray_grid = read_csv(METRIC_DIR / "grayzone_grid.csv").sort_values("q").reset_index(drop=True)
    llm_runner_val = build_llm_runner(cfg, force_stub=True)
    llm_runner_live = build_llm_runner(cfg)
    base_runtime_lookup = _load_base_runtime_lookup()

    pred_val = read_csv(PRED_DIR / "base_val_predictions.csv")
    pred_main = read_csv(PRED_DIR / "base_test_main_predictions.csv")

    tau = float(tau_info["tau"])
    default_q = read_selected_q(DEFAULT_Q)
    summary_rows: list[dict] = []
    q_sweep_selective_parts = []
    q_sweep_no_llm_parts = []
    llm_seed_modes = {"selective", "selective_no_graph", "selective_no_filter"}
    main_llm_seed_policy = get_llm_seed_policy("main")
    q_sweep_llm_seeds = list(SEEDS) if main_llm_seed_policy == "all" else [REPRESENTATIVE_SEED]
    main_mode_seeds = {
        mode: (list(SEEDS) if mode in llm_seed_modes and main_llm_seed_policy == "all" else [REPRESENTATIVE_SEED] if mode in llm_seed_modes else list(SEEDS))
        for mode in _active_modes("main")
    }

    main_modes = _active_modes("main")
    gray_row = _gray_row_for_q(gray_grid, default_q)
    val_rows, val_mode_outputs = evaluate_dataset_modes(
        dataset_name="val",
        base_pred=pred_val,
        tau=tau,
        q=default_q,
        gray_row=gray_row,
        cfg=cfg,
        active_modes=main_modes,
        llm_runner=llm_runner_val,
        base_runtime_lookup=base_runtime_lookup,
        ref_recall_by_mode=None,
        seeds=SEEDS,
        mode_seeds=main_mode_seeds,
    )
    summary_rows.extend(val_rows)
    val_ref = _reference_recall_map(val_rows, "val")
    val_outputs = {mode: aggregate_seed_predictions(parts) for mode, parts in val_mode_outputs.items() if parts}

    if not gray_grid.empty:
        q_sweep_anchor = float(gray_grid["q"].max())
        anchor_gray_row = _gray_row_for_q(gray_grid, q_sweep_anchor)
        _prewarm_q_sweep_llm_cache(
            base_pred=pred_main,
            tau=tau,
            gray_row=anchor_gray_row,
            cfg=cfg,
            llm_runner=llm_runner_live,
            seeds=q_sweep_llm_seeds,
        )

    for q in sorted(gray_grid["q"].tolist(), reverse=True):
        gray_row = _gray_row_for_q(gray_grid, float(q))
        active_modes = main_modes if np.isclose(q, default_q) else ["selective", "no_llm"]
        mode_seeds = main_mode_seeds if np.isclose(q, default_q) else {
            "selective": (list(SEEDS) if main_llm_seed_policy == "all" else [REPRESENTATIVE_SEED]),
            "no_llm": list(SEEDS),
        }
        main_rows, main_mode_outputs = evaluate_dataset_modes(
            dataset_name="main",
            base_pred=pred_main,
            tau=tau,
            q=float(q),
            gray_row=gray_row,
            cfg=cfg,
            active_modes=active_modes,
            llm_runner=llm_runner_live,
            base_runtime_lookup=base_runtime_lookup,
            ref_recall_by_mode=val_ref,
            seeds=SEEDS,
            mode_seeds=mode_seeds,
        )
        summary_rows.extend(main_rows)
        main_outputs = {mode: aggregate_seed_predictions(parts) for mode, parts in main_mode_outputs.items() if parts}
        if np.isclose(q, default_q):
            for mode, out_df in main_outputs.items():
                write_csv(PRED_DIR / f"utar_test_main_{mode}.csv", out_df)
        if "selective" in main_mode_outputs:
            for seed, sel_df in zip(mode_seeds["selective"], main_mode_outputs["selective"]):
                sel_part = sel_df.copy()
                sel_part["q"] = float(q)
                sel_part["seed"] = int(seed)
                q_sweep_selective_parts.append(sel_part)
        if "no_llm" in main_mode_outputs:
            for seed, no_llm_df in zip(mode_seeds["no_llm"], main_mode_outputs["no_llm"]):
                no_llm_part = no_llm_df.copy()
                no_llm_part["q"] = float(q)
                no_llm_part["seed"] = int(seed)
                q_sweep_no_llm_parts.append(no_llm_part)

    for mode, out_df in val_outputs.items():
        write_csv(PRED_DIR / f"utar_val_{mode}.csv", out_df)
    write_csv(PRED_DIR / "utar_q_sweep.csv", pd.concat(q_sweep_selective_parts, ignore_index=True))
    write_csv(PRED_DIR / "utar_q_sweep_no_llm.csv", pd.concat(q_sweep_no_llm_parts, ignore_index=True))

    return _write_merged_summary(summary_rows)


def run_main_eval_selected_q_only(*, modes: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = read_yaml(CONFIG_DIR / "routing.yaml", default={})
    tau_info = read_json(METRIC_DIR / "thresholds.json")
    gray_grid = read_csv(METRIC_DIR / "grayzone_grid.csv").sort_values("q").reset_index(drop=True)
    llm_runner_val = build_llm_runner(cfg, force_stub=True)
    llm_runner_live = build_llm_runner(cfg)
    base_runtime_lookup = _load_base_runtime_lookup()

    pred_val = read_csv(PRED_DIR / "base_val_predictions.csv")
    pred_main = read_csv(PRED_DIR / "base_test_main_predictions.csv")

    tau = float(tau_info["tau"])
    selected_q = read_selected_q(DEFAULT_Q)
    summary_rows: list[dict] = []
    llm_seed_modes = {"selective", "selective_no_graph", "selective_no_filter"}
    main_llm_seed_policy = get_llm_seed_policy("main")
    all_main_modes = _active_modes("main")
    if modes is None:
        main_modes = all_main_modes
    else:
        unknown = [mode for mode in modes if mode not in all_main_modes]
        if unknown:
            raise ValueError(f"Unsupported main modes requested: {unknown}")
        main_modes = list(dict.fromkeys(modes))
    main_mode_seeds = {
        mode: (list(SEEDS) if mode in llm_seed_modes and main_llm_seed_policy == "all" else [REPRESENTATIVE_SEED] if mode in llm_seed_modes else list(SEEDS))
        for mode in main_modes
    }
    gray_row = _gray_row_for_q(gray_grid, selected_q)
    val_rows, _ = evaluate_dataset_modes(
        dataset_name="val",
        base_pred=pred_val,
        tau=tau,
        q=selected_q,
        gray_row=gray_row,
        cfg=cfg,
        active_modes=main_modes,
        llm_runner=llm_runner_val,
        base_runtime_lookup=base_runtime_lookup,
        ref_recall_by_mode=None,
        seeds=SEEDS,
        mode_seeds=main_mode_seeds,
    )
    summary_rows.extend(val_rows)
    val_ref = _reference_recall_map(val_rows, "val")

    main_rows, main_mode_outputs = evaluate_dataset_modes(
        dataset_name="main",
        base_pred=pred_main,
        tau=tau,
        q=selected_q,
        gray_row=gray_row,
        cfg=cfg,
        active_modes=main_modes,
        llm_runner=llm_runner_live,
        base_runtime_lookup=base_runtime_lookup,
        ref_recall_by_mode=val_ref,
        seeds=SEEDS,
        mode_seeds=main_mode_seeds,
    )
    summary_rows.extend(main_rows)
    main_outputs = {mode: aggregate_seed_predictions(parts) for mode, parts in main_mode_outputs.items() if parts}
    q_tag = _q_file_tag(selected_q)
    for mode, out_df in main_outputs.items():
        write_csv(PRED_DIR / f"utar_test_main_{mode}_{q_tag}.csv", out_df)

    return _write_merged_summary(summary_rows)


def run_main_eval_q_values_only(q_values: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = read_yaml(CONFIG_DIR / "routing.yaml", default={})
    tau_info = read_json(METRIC_DIR / "thresholds.json")
    gray_grid = read_csv(METRIC_DIR / "grayzone_grid.csv").sort_values("q").reset_index(drop=True)
    llm_runner_val = build_llm_runner(cfg, force_stub=True)
    llm_runner_live = build_llm_runner(cfg)
    base_runtime_lookup = _load_base_runtime_lookup()

    pred_val = read_csv(PRED_DIR / "base_val_predictions.csv")
    pred_main = read_csv(PRED_DIR / "base_test_main_predictions.csv")

    tau = float(tau_info["tau"])
    default_q = read_selected_q(DEFAULT_Q)
    requested_qs = sorted({float(q) for q in q_values}, reverse=True)
    available_qs = {float(q) for q in gray_grid["q"].tolist()}
    missing_qs = [q for q in requested_qs if q not in available_qs]
    if missing_qs:
        missing_fmt = ", ".join(f"{q:.2f}" for q in missing_qs)
        raise ValueError(f"Requested q values are missing from grayzone_grid.csv: {missing_fmt}. Run fit_grayzone first.")

    summary_rows: list[dict] = []
    q_sweep_selective_parts = []
    q_sweep_no_llm_parts = []
    active_modes = ["selective", "no_llm"]
    llm_seed_modes = {"selective"}
    main_llm_seed_policy = get_llm_seed_policy("main")
    mode_seeds = {
        mode: (list(SEEDS) if mode in llm_seed_modes and main_llm_seed_policy == "all" else [REPRESENTATIVE_SEED] if mode in llm_seed_modes else list(SEEDS))
        for mode in active_modes
    }

    val_gray_row = _gray_row_for_q(gray_grid, default_q)
    val_rows, _ = evaluate_dataset_modes(
        dataset_name="val",
        base_pred=pred_val,
        tau=tau,
        q=default_q,
        gray_row=val_gray_row,
        cfg=cfg,
        active_modes=active_modes,
        llm_runner=llm_runner_val,
        base_runtime_lookup=base_runtime_lookup,
        ref_recall_by_mode=None,
        seeds=SEEDS,
        mode_seeds=mode_seeds,
    )
    val_ref = _reference_recall_map(val_rows, "val")

    anchor_q = requested_qs[0]
    anchor_gray_row = _gray_row_for_q(gray_grid, anchor_q)
    _prewarm_q_sweep_llm_cache(
        base_pred=pred_main,
        tau=tau,
        gray_row=anchor_gray_row,
        cfg=cfg,
        llm_runner=llm_runner_live,
        seeds=mode_seeds["selective"],
    )

    for q in requested_qs:
        gray_row = _gray_row_for_q(gray_grid, q)
        main_rows, main_mode_outputs = evaluate_dataset_modes(
            dataset_name="main",
            base_pred=pred_main,
            tau=tau,
            q=q,
            gray_row=gray_row,
            cfg=cfg,
            active_modes=active_modes,
            llm_runner=llm_runner_live,
            base_runtime_lookup=base_runtime_lookup,
            ref_recall_by_mode=val_ref,
            seeds=SEEDS,
            mode_seeds=mode_seeds,
        )
        summary_rows.extend(main_rows)
        for seed, sel_df in zip(mode_seeds["selective"], main_mode_outputs["selective"]):
            sel_part = sel_df.copy()
            sel_part["q"] = float(q)
            sel_part["seed"] = int(seed)
            q_sweep_selective_parts.append(sel_part)
        for seed, no_llm_df in zip(mode_seeds["no_llm"], main_mode_outputs["no_llm"]):
            no_llm_part = no_llm_df.copy()
            no_llm_part["q"] = float(q)
            no_llm_part["seed"] = int(seed)
            q_sweep_no_llm_parts.append(no_llm_part)

    if q_sweep_selective_parts:
        _merge_prediction_store(PRED_DIR / "utar_q_sweep.csv", pd.concat(q_sweep_selective_parts, ignore_index=True))
    if q_sweep_no_llm_parts:
        _merge_prediction_store(PRED_DIR / "utar_q_sweep_no_llm.csv", pd.concat(q_sweep_no_llm_parts, ignore_index=True))

    return _write_merged_summary(summary_rows)


def run_cost_eval() -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = read_yaml(CONFIG_DIR / "routing.yaml", default={})
    tau_info = read_json(METRIC_DIR / "thresholds.json")
    gray_grid = read_csv(METRIC_DIR / "grayzone_grid.csv").sort_values("q").reset_index(drop=True)
    llm_runner_live = build_llm_runner(cfg)
    llm_runner_val = build_llm_runner(cfg, force_stub=True)
    base_runtime_lookup = _load_base_runtime_lookup()
    pred_val = read_csv(PRED_DIR / "base_val_predictions.csv")
    pred_cost = read_csv(PRED_DIR / "base_test_cost_predictions.csv")

    tau = float(tau_info["tau"])
    summary_rows: list[dict] = []
    q = read_selected_q(DEFAULT_Q)
    gray_row = _gray_row_for_q(gray_grid, q)
    cost_modes = _active_modes("cost")
    cost_llm_seed_policy = get_llm_seed_policy("cost")
    llm_seed_modes = {"selective", "selective_no_filter", "full_llm"}
    cost_mode_seeds = {
        mode: (list(SEEDS) if mode in llm_seed_modes and cost_llm_seed_policy == "all" else [REPRESENTATIVE_SEED] if mode in llm_seed_modes else list(SEEDS))
        for mode in cost_modes
    }
    val_rows, _ = evaluate_dataset_modes(
        dataset_name="val",
        base_pred=pred_val,
        tau=tau,
        q=q,
        gray_row=gray_row,
        cfg=cfg,
        active_modes=cost_modes,
        llm_runner=llm_runner_val,
        base_runtime_lookup=base_runtime_lookup,
        ref_recall_by_mode=None,
        seeds=SEEDS,
        mode_seeds=cost_mode_seeds,
    )
    val_ref = _reference_recall_map(val_rows, "val")
    cost_rows, cost_mode_outputs = evaluate_dataset_modes(
        dataset_name="cost",
        base_pred=pred_cost,
        tau=tau,
        q=q,
        gray_row=gray_row,
        cfg=cfg,
        active_modes=cost_modes,
        llm_runner=llm_runner_live,
        base_runtime_lookup=base_runtime_lookup,
        ref_recall_by_mode=val_ref,
        seeds=SEEDS,
        mode_seeds=cost_mode_seeds,
    )
    summary_rows.extend(cost_rows)
    mode_outputs = {mode: aggregate_seed_predictions(parts) for mode, parts in cost_mode_outputs.items() if parts}
    for mode, out_df in mode_outputs.items():
        write_csv(PRED_DIR / f"utar_test_cost_{mode}.csv", out_df)
    return _write_merged_summary(summary_rows)


def main() -> None:
    _, main_summary = run_main_eval()
    print("[selective_llm_eval] main evaluation completed")
    print(main_summary[main_summary["dataset"].isin(["val", "main"])].to_string(index=False))
    _, cost_summary = run_cost_eval()
    print("[selective_llm_eval] cost evaluation completed with selected q")
    print(cost_summary[cost_summary["dataset"] == "cost"].to_string(index=False))


if __name__ == "__main__":
    main()
