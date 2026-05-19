from __future__ import annotations

import numpy as np
import pandas as pd


def binary_entropy(prob: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(prob, dtype=float), eps, 1.0 - eps)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def discrepancy_std3(scores: np.ndarray) -> np.ndarray:
    arr = np.asarray(scores, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected score matrix of shape [n_samples, 3], got {arr.shape}.")
    return arr.std(axis=1)


def mean_binary_entropy(scores: np.ndarray) -> np.ndarray:
    arr = np.asarray(scores, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected score matrix of shape [n_samples, n_models], got {arr.shape}.")
    ent = binary_entropy(arr)
    return ent.mean(axis=1)


def u_total(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d_ens = discrepancy_std3(scores)
    h_avg = mean_binary_entropy(scores)
    return d_ens * h_avg, d_ens, h_avg


def mixing_weight(u_total_value: np.ndarray, tau: float, k: float) -> np.ndarray:
    u = np.asarray(u_total_value, dtype=float)
    return 1.0 / (1.0 + np.exp(-float(k) * (u - float(tau))))


def platt_like_calibration(
    raw_score: np.ndarray,
    *,
    raw_min: float = 0.0,
    raw_max: float = 1.0,
    a: float = 4.0,
    b: float = -2.0,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> np.ndarray:
    raw = np.asarray(raw_score, dtype=float)
    denom = max(float(raw_max) - float(raw_min), 1e-6)
    scaled = np.clip((raw - float(raw_min)) / denom, 0.0, 1.0)
    calibrated = 1.0 / (1.0 + np.exp(-(float(a) * scaled + float(b))))
    return np.clip(calibrated, float(clip_min), float(clip_max))


def scale_clip_calibration(
    raw_score: np.ndarray,
    *,
    factor: float = 1.0,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> np.ndarray:
    raw = np.asarray(raw_score, dtype=float)
    calibrated = raw * float(factor)
    return np.clip(calibrated, float(clip_min), float(clip_max))


def apply_llm_calibration(raw_score: np.ndarray, cfg: dict | None = None) -> np.ndarray:
    cfg = cfg or {}
    method = str(cfg.get("method", "platt_like")).strip().lower()
    if method == "scale_clip":
        return scale_clip_calibration(
            raw_score,
            factor=float(cfg.get("factor", 1.0)),
            clip_min=float(cfg.get("clip_min", 0.0)),
            clip_max=float(cfg.get("clip_max", 1.0)),
        )
    return platt_like_calibration(
        raw_score,
        raw_min=float(cfg.get("raw_min", 0.0)),
        raw_max=float(cfg.get("raw_max", 1.0)),
        a=float(cfg.get("a", 4.0)),
        b=float(cfg.get("b", -2.0)),
        clip_min=float(cfg.get("clip_min", 0.0)),
        clip_max=float(cfg.get("clip_max", 1.0)),
    )


def soft_fusion(local_avg: np.ndarray, llm_score: np.ndarray, weight: np.ndarray) -> np.ndarray:
    local_avg = np.asarray(local_avg, dtype=float)
    llm_score = np.asarray(llm_score, dtype=float)
    weight = np.asarray(weight, dtype=float)
    return (1.0 - weight) * local_avg + weight * llm_score


def derive_graphad_concentration(
    df: pd.DataFrame,
    *,
    concentration_col: str = "graphad_concentration",
    gap_col: str = "graphad_top1_gap",
    cfg: dict | None = None,
) -> pd.Series:
    if concentration_col in df.columns:
        return df[concentration_col].astype(str).str.strip().str.lower()
    cfg = cfg or {}
    prompt_cfg = cfg.get("prompt_policy", {})
    gap_thr = float(prompt_cfg.get("graphad_gap_concentrated_threshold", 0.5))
    if gap_col not in df.columns:
        return pd.Series("diffuse", index=df.index, dtype="object")
    gap = df[gap_col].fillna(0.0).to_numpy(dtype=float)
    labels = np.where(gap >= gap_thr, "concentrated", "diffuse")
    return pd.Series(labels, index=df.index, dtype="object")


def prompt_policy_raw_score(
    df: pd.DataFrame,
    *,
    local_col: str = "ppt_local_avg",
    tcn_col: str = "p_tcn",
    entropy_col: str = "ppt_h_avg",
    graphad_z_col: str = "graphad_top1_z",
    concentration_col: str = "graphad_concentration",
    cfg: dict | None = None,
) -> np.ndarray:
    cfg = cfg or {}
    prompt_cfg = cfg.get("prompt_policy", {})
    llm_cfg = cfg.get("llm_raw", {})
    method = str(llm_cfg.get("method", "ppt_policy_stub")).strip().lower()

    local = df[local_col].to_numpy(dtype=float) if local_col in df.columns else np.zeros(len(df), dtype=float)
    raw = local.copy()

    tcn_thr = float(prompt_cfg.get("modern_tcn_priority_threshold", 0.90))
    tcn_gain = float(llm_cfg.get("modern_tcn_gain", 0.03))
    tcn_floor = float(llm_cfg.get("modern_tcn_floor_score", 0.95))
    if tcn_col in df.columns:
        tcn = df[tcn_col].to_numpy(dtype=float)
        tcn_mask = tcn >= tcn_thr
        if method == "ppt_prompt_rules_stub":
            tcn_score = np.maximum(np.clip(tcn[tcn_mask], 0.0, 1.0), tcn_floor)
            raw[tcn_mask] = np.maximum(raw[tcn_mask], np.clip(tcn_score, 0.0, 1.0))
        else:
            raw[tcn_mask] = np.maximum(raw[tcn_mask], np.clip(tcn[tcn_mask] + tcn_gain, 0.0, 1.0))

    graphad_z_thr = float(prompt_cfg.get("graphad_z_threshold", 2.0))
    conc_values = {str(v).strip().lower() for v in prompt_cfg.get("graphad_concentration_values", ["concentrated", "high"])}
    graph_base = float(llm_cfg.get("graphad_base_score", 0.78))
    graph_gain = float(llm_cfg.get("graphad_gain", 0.12))
    if graphad_z_col in df.columns:
        z = df[graphad_z_col].fillna(0.0).to_numpy(dtype=float)
        concentration = derive_graphad_concentration(df, concentration_col=concentration_col, cfg=cfg)
        graph_mask = (z >= graphad_z_thr) & concentration.isin(conc_values).to_numpy(dtype=bool)
        graph_score = np.clip(graph_base + graph_gain * np.tanh(np.maximum(z - graphad_z_thr, 0.0)), 0.0, 1.0)
        raw[graph_mask] = np.maximum(raw[graph_mask], graph_score[graph_mask])

    ent_thr = float(prompt_cfg.get("entropy_anomaly_threshold", 0.9))
    ent_base = float(llm_cfg.get("entropy_base_score", 0.62))
    ent_gain = float(llm_cfg.get("entropy_gain", 0.22))
    if entropy_col in df.columns:
        ent = df[entropy_col].fillna(0.0).to_numpy(dtype=float)
        ent_mask = ent >= ent_thr
        ent_scale = np.clip((ent - ent_thr) / max(1.0 - ent_thr, 1e-6), 0.0, 1.0)
        ent_score = np.clip(ent_base + ent_gain * ent_scale, 0.0, 1.0)
        if method == "ppt_prompt_rules_stub":
            raw[ent_mask] = np.maximum(raw[ent_mask], ent_score[ent_mask])
        else:
            raw[ent_mask] = np.maximum(raw[ent_mask], np.maximum(local[ent_mask], ent_score[ent_mask]))

    return np.clip(raw, 0.0, 1.0)


def attach_ppt_full_features(
    df: pd.DataFrame,
    *,
    rf_col: str = "p_rf",
    xgb_col: str = "p_xgb",
    tcn_col: str = "p_tcn",
    llm_col: str | None = None,
    cfg: dict | None = None,
) -> pd.DataFrame:
    cfg = cfg or {}
    out = df.copy()
    score_matrix = out[[rf_col, xgb_col, tcn_col]].to_numpy(dtype=float)
    u_val, d_ens, h_avg = u_total(score_matrix)
    out["ppt_local_avg"] = score_matrix.mean(axis=1)
    out["p_local"] = out["ppt_local_avg"]
    out["ppt_d_ens"] = d_ens
    out["ppt_h_avg"] = h_avg
    out["ppt_u_total"] = u_val
    out["graphad_concentration"] = derive_graphad_concentration(out, cfg=cfg)

    mixing_cfg = cfg.get("mixing_weight", {})
    tau = float(mixing_cfg.get("tau", 0.2))
    k = float(mixing_cfg.get("k", 15.0))
    w = mixing_weight(u_val, tau=tau, k=k)
    out["ppt_w"] = w

    raw_llm = None
    if llm_col and llm_col in out.columns:
        raw_llm = out[llm_col].to_numpy(dtype=float)
    elif str(cfg.get("llm_raw", {}).get("method", "ppt_policy_stub")).strip().lower() == "ppt_policy_stub":
        raw_llm = prompt_policy_raw_score(out, local_col="ppt_local_avg", tcn_col=tcn_col, entropy_col="ppt_h_avg", cfg=cfg)

    if raw_llm is not None:
        out["ppt_llm_raw"] = raw_llm
        out["p_llm_raw"] = raw_llm
        cal_cfg = cfg.get("calibration", {})
        llm_cal = apply_llm_calibration(raw_llm, cal_cfg)
        out["ppt_llm_calibrated"] = llm_cal
        out["p_llm_calibrated"] = llm_cal
        if bool(cfg.get("soft_fusion", {}).get("enabled", True)):
            fused = soft_fusion(out["ppt_local_avg"].to_numpy(dtype=float), llm_cal, w)
            out["ppt_soft_fusion"] = fused
            out["p_soft_fusion"] = fused
    return out


def build_policy_flags(
    df: pd.DataFrame,
    *,
    tcn_col: str = "p_tcn",
    entropy_col: str = "ppt_h_avg",
    graphad_z_col: str = "graphad_top1_z",
    graphad_concentration_col: str = "graphad_concentration",
    cfg: dict | None = None,
) -> pd.DataFrame:
    cfg = cfg or {}
    prompt_cfg = cfg.get("prompt_policy", {})
    tcn_thr = float(prompt_cfg.get("modern_tcn_priority_threshold", 0.90))
    graphad_z_thr = float(prompt_cfg.get("graphad_z_threshold", 2.0))
    ent_thr = float(prompt_cfg.get("entropy_anomaly_threshold", 0.9))
    concentration_values = {str(v).strip().lower() for v in prompt_cfg.get("graphad_concentration_values", ["concentrated", "high"])}

    out = pd.DataFrame(index=df.index)
    out["ppt_auc_priority_flag"] = df[tcn_col].to_numpy(dtype=float) >= tcn_thr
    out["ppt_entropy_priority_flag"] = df[entropy_col].to_numpy(dtype=float) >= ent_thr if entropy_col in df.columns else False

    concentration = derive_graphad_concentration(df, concentration_col=graphad_concentration_col, cfg=cfg)
    if graphad_z_col in df.columns:
        out["ppt_prr_priority_flag"] = (df[graphad_z_col].fillna(0.0).to_numpy(dtype=float) >= graphad_z_thr) & concentration.isin(concentration_values)
    else:
        out["ppt_prr_priority_flag"] = False
    out["ppt_priority_vote_count"] = out[["ppt_auc_priority_flag", "ppt_entropy_priority_flag", "ppt_prr_priority_flag"]].sum(axis=1)
    out["ppt_any_priority_flag"] = out["ppt_priority_vote_count"] > 0
    hit_labels = []
    for _, row in out[["ppt_auc_priority_flag", "ppt_prr_priority_flag", "ppt_entropy_priority_flag"]].iterrows():
        labels = []
        if bool(row["ppt_auc_priority_flag"]):
            labels.append("auc")
        if bool(row["ppt_prr_priority_flag"]):
            labels.append("prr")
        if bool(row["ppt_entropy_priority_flag"]):
            labels.append("f1")
        hit_labels.append("|".join(labels) if labels else "none")
    out["ppt_prompt_rule_hits"] = hit_labels
    return out
