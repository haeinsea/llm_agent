from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ppt_full.formulas import (
    apply_llm_calibration,
    mixing_weight,
    prompt_policy_raw_score,
    soft_fusion,
    u_total,
)
from src.utils.io import read_json, read_yaml
from src.utils.routing import build_routing_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_THRESHOLDS_PATH = PROJECT_ROOT / "outputs" / "metrics" / "thresholds.json"
DEFAULT_GRAYZONE_PATH = PROJECT_ROOT / "outputs" / "metrics" / "grayzone_defaults.json"
DEFAULT_ROUTING_CONFIG_PATH = PROJECT_ROOT / "configs" / "routing.yaml"


def _resolve_path(path_like: str | Path, default_path: Path) -> Path:
    path = Path(path_like) if path_like else default_path
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def sigmoid(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-arr))


def indicator_gray_zone(s_utar: np.ndarray, tau: float, m_q: float) -> np.ndarray:
    score = np.asarray(s_utar, dtype=float)
    margin = max(float(m_q), 1e-8)
    return (np.abs(score - float(tau)) <= margin).astype(float)


def boundary_weight(s_utar: np.ndarray, tau: float, m_q: float) -> np.ndarray:
    score = np.asarray(s_utar, dtype=float)
    half_margin = max(float(m_q) / 2.0, 1e-8)
    normalized = np.abs(score - float(tau)) / half_margin
    return sigmoid(normalized - 1.0)


def conditional_soft_fusion(
    s_base: np.ndarray,
    s_llm: np.ndarray,
    s_utar: np.ndarray,
    tau: float,
    m_q: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = np.asarray(s_base, dtype=float)
    llm = np.asarray(s_llm, dtype=float)
    utar = np.asarray(s_utar, dtype=float)
    indicator = indicator_gray_zone(utar, tau=tau, m_q=m_q)
    weight = boundary_weight(utar, tau=tau, m_q=m_q)
    fused = (1.0 - indicator * weight) * base + (indicator * weight) * llm
    return fused, indicator, weight


def _resolve_tau(cfg: dict[str, Any], *, seed: int | None = None) -> float:
    fusion_cfg = cfg.get("boundary_soft_fusion", {})
    if "tau" in fusion_cfg and fusion_cfg["tau"] is not None:
        return float(fusion_cfg["tau"])
    payload = read_json(_resolve_path(fusion_cfg.get("thresholds_path", DEFAULT_THRESHOLDS_PATH), DEFAULT_THRESHOLDS_PATH))
    if seed is not None:
        per_seed = payload.get("per_seed", {})
        seed_key = f"seed{int(seed)}"
        if seed_key in per_seed and "tau" in per_seed[seed_key]:
            return float(per_seed[seed_key]["tau"])
    if "tau_mean_from_seeds" in payload:
        return float(payload["tau_mean_from_seeds"])
    return float(payload.get("tau", 0.5))


def _resolve_gray_margin(cfg: dict[str, Any], *, seed: int | None = None) -> float:
    fusion_cfg = cfg.get("boundary_soft_fusion", {})
    if "m_q" in fusion_cfg and fusion_cfg["m_q"] is not None:
        return float(fusion_cfg["m_q"])
    payload = read_json(_resolve_path(fusion_cfg.get("grayzone_defaults_path", DEFAULT_GRAYZONE_PATH), DEFAULT_GRAYZONE_PATH))
    default_q = float(fusion_cfg.get("default_q", payload.get("default_q", 0.8)))
    q_key = f"{default_q:.2f}"
    if seed is not None:
        per_seed = payload.get("per_seed", {})
        seed_key = f"seed{int(seed)}"
        if seed_key in per_seed and q_key in per_seed[seed_key]:
            return float(per_seed[seed_key][q_key]["gray_margin"])

    margins = []
    for seed_payload in payload.get("per_seed", {}).values():
        if q_key in seed_payload and "gray_margin" in seed_payload[q_key]:
            margins.append(float(seed_payload[q_key]["gray_margin"]))
    if margins:
        return float(np.mean(margins))
    return float(fusion_cfg.get("fallback_m_q", 0.1))


def _resolve_routing_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    fusion_cfg = cfg.get("boundary_soft_fusion", {})
    routing_path = _resolve_path(fusion_cfg.get("routing_config_path", DEFAULT_ROUTING_CONFIG_PATH), DEFAULT_ROUTING_CONFIG_PATH)
    return read_yaml(routing_path, default={}) or {}


def _resolve_llm_score(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    llm_col: str | None = None,
    tcn_col: str = "p_tcn",
) -> tuple[np.ndarray, np.ndarray | None]:
    fusion_cfg = cfg.get("boundary_soft_fusion", {})
    llm_source = str(fusion_cfg.get("llm_score_source", "prompt_policy_stub")).strip().lower()
    if llm_col and llm_col in df.columns:
        raw = df[llm_col].to_numpy(dtype=float)
    elif llm_source == "prompt_policy_stub":
        entropy_col = "ppt_h_avg" if "ppt_h_avg" in df.columns else "ensemble_entropy"
        raw = prompt_policy_raw_score(df, local_col="ppt_local_avg", tcn_col=tcn_col, entropy_col=entropy_col, cfg=cfg)
    else:
        raw = df["ppt_local_avg"].to_numpy(dtype=float)

    calibrated = None
    if bool(fusion_cfg.get("calibrate_llm_score", False)):
        calibrated = apply_llm_calibration(raw, fusion_cfg.get("calibration", {}))
    return np.asarray(raw, dtype=float), None if calibrated is None else np.asarray(calibrated, dtype=float)


def attach_conditional_soft_fusion_features(
    df: pd.DataFrame,
    *,
    rf_col: str = "p_rf",
    xgb_col: str = "p_xgb",
    tcn_col: str = "p_tcn",
    llm_col: str | None = None,
    cfg: dict[str, Any] | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    cfg = cfg or {}
    out = df.copy()
    routing_cfg = _resolve_routing_cfg(cfg)
    routing_df = build_routing_features(out, routing_cfg, rf_col=rf_col, xgb_col=xgb_col, tcn_col=tcn_col)
    for col in routing_df.columns:
        out[col] = routing_df[col]

    score_matrix = out[[rf_col, xgb_col, tcn_col]].to_numpy(dtype=float)
    u_val, d_ens, h_avg = u_total(score_matrix)
    out["ppt_local_avg"] = score_matrix.mean(axis=1)
    out["p_local"] = out["ppt_local_avg"]
    out["ppt_d_ens"] = d_ens
    out["ppt_h_avg"] = h_avg
    out["ppt_u_total"] = u_val
    tau = _resolve_tau(cfg, seed=seed)
    m_q = _resolve_gray_margin(cfg, seed=seed)
    s_utar = out["p_utar_base"].to_numpy(dtype=float)
    s_base = out["ppt_local_avg"].to_numpy(dtype=float)
    s_llm_raw, s_llm_calibrated = _resolve_llm_score(out, cfg, llm_col=llm_col, tcn_col=tcn_col)
    s_llm = s_llm_calibrated if s_llm_calibrated is not None else s_llm_raw
    fused, indicator, weight = conditional_soft_fusion(s_base, s_llm, s_utar, tau=tau, m_q=m_q)

    ref_cfg = cfg.get("uncertainty_reference", {})
    if bool(ref_cfg.get("enabled", True)):
        ref_tau = float(ref_cfg.get("tau", 0.2))
        ref_k = float(ref_cfg.get("k", 15.0))
        ref_weight = mixing_weight(u_val, tau=ref_tau, k=ref_k)
        ref_calibrated = apply_llm_calibration(s_llm_raw, ref_cfg.get("calibration", {}))
        ref_fused = soft_fusion(s_base, ref_calibrated, ref_weight)
        out["ppt_uncertainty_tau"] = ref_tau
        out["ppt_uncertainty_k"] = ref_k
        out["ppt_uncertainty_weight"] = ref_weight
        out["ppt_uncertainty_llm_calibrated"] = ref_calibrated
        out["ppt_uncertainty_soft_fusion"] = ref_fused

    out["ppt_tau"] = tau
    out["ppt_m_q"] = m_q
    out["ppt_s_base"] = s_base
    out["ppt_s_utar"] = s_utar
    out["ppt_llm_raw"] = s_llm_raw
    out["p_llm_raw"] = s_llm_raw
    if s_llm_calibrated is not None:
        out["ppt_llm_calibrated"] = s_llm_calibrated
        out["p_llm_calibrated"] = s_llm_calibrated
    out["ppt_s_llm"] = s_llm
    out["ppt_gray_zone_indicator"] = indicator
    out["ppt_boundary_weight"] = weight
    out["ppt_conditional_soft_fusion"] = fused
    out["p_soft_fusion"] = fused
    return out
