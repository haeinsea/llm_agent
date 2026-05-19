from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from src.models.graphad import infer_graphad, load_graphad_artifact
from src.models.temporal_backbone import build_temporal_model
from src.models.temporal_sota import infer_temporal_probs, infer_tta_online_probs
from src.models.train_invariant import invariant_penalty, phase_to_env
from src.models.train_tcn import (
    best_threshold_by_f1,
    flattened_to_tensor,
    phase_recall,
    read_windows,
    set_seed,
    transform_with_imputer_scaler,
)
from src.ppt_full.conditional_soft_fusion import attach_conditional_soft_fusion_features
from src.ppt_full.formulas import attach_ppt_full_features, build_policy_flags
from src.utils.device import synchronize_torch_device, torch_device_info
from src.utils.experiment import get_seed_list
from src.utils.io import ensure_dir, read_csv, read_json, read_yaml, write_csv, write_json
from src.utils.metrics import prr as performance_retention_rate

try:
    from xgboost import XGBClassifier
except Exception as exc:  # pragma: no cover
    raise ImportError("xgboost is required for the PPT full model suite.") from exc


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
META_DIR = DATA_DIR / "meta"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_GRAPHAD_ARTIFACT = OUTPUT_DIR / "models" / "graphad_artifact.json"

SEEDS = get_seed_list()


def _load_feature_cols() -> list[str]:
    with open(META_DIR / "feature_columns.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _read_rows(name: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / name)


def _compute_metrics(y_true: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict[str, float | None]:
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)
    y_hat = (p >= threshold).astype(int)
    out = {
        "f1": float(f1_score(y_true, y_hat, zero_division=0)),
        "recall": float(recall_score(y_true, y_hat, zero_division=0)),
        "precision": float(precision_score(y_true, y_hat, zero_division=0)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, p))
    except Exception:
        out["roc_auc"] = None
    return out


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _maybe_int(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return None
        return int(value)
    return value


def train_rf_family(params: dict[str, Any], output_root: Path) -> None:
    model_dir = output_root / "models"
    metrics_dir = output_root / "metrics"
    ensure_dir(model_dir)
    ensure_dir(metrics_dir)
    feature_cols = _load_feature_cols()
    train_df = _read_rows("te_train_rows.csv")
    val_df = _read_rows("te_val_rows.csv")
    X_train = train_df[feature_cols].to_numpy()
    y_train = train_df["y"].to_numpy().astype(int)
    X_val = val_df[feature_cols].to_numpy()
    y_val = val_df["y"].to_numpy().astype(int)
    for seed in SEEDS:
        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train)
        X_val_imp = imputer.transform(X_val)
        model = RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=_maybe_int(params.get("max_depth")),
            min_samples_split=int(params.get("min_samples_split", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            max_features=params.get("max_features", "sqrt"),
            bootstrap=bool(params.get("bootstrap", True)),
            random_state=int(seed),
            n_jobs=-1,
            class_weight="balanced",
        )
        model.fit(X_train_imp, y_train)
        p_val = model.predict_proba(X_val_imp)[:, 1]
        joblib.dump(model, model_dir / f"rf_model_seed{seed}.pkl")
        joblib.dump(imputer, model_dir / f"rf_imputer_seed{seed}.pkl")
        _save_json(metrics_dir / f"rf_val_metrics_seed{seed}.json", _compute_metrics(y_val, p_val))


def train_xgb_family(params: dict[str, Any], output_root: Path) -> None:
    model_dir = output_root / "models"
    metrics_dir = output_root / "metrics"
    ensure_dir(model_dir)
    ensure_dir(metrics_dir)
    feature_cols = _load_feature_cols()
    train_df = _read_rows("te_train_rows.csv")
    val_df = _read_rows("te_val_rows.csv")
    X_train = train_df[feature_cols].to_numpy()
    y_train = train_df["y"].to_numpy().astype(int)
    X_val = val_df[feature_cols].to_numpy()
    y_val = val_df["y"].to_numpy().astype(int)
    pos = max(int(y_train.sum()), 1)
    neg = max(int((1 - y_train).sum()), 1)
    scale_pos_weight = neg / pos
    for seed in SEEDS:
        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train)
        X_val_imp = imputer.transform(X_val)
        model = XGBClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=_maybe_int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            min_child_weight=float(params.get("min_child_weight", 1.0)),
            gamma=float(params.get("gamma", 0.0)),
            reg_alpha=float(params.get("reg_alpha", 0.0)),
            reg_lambda=float(params["reg_lambda"]),
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=int(seed),
            scale_pos_weight=scale_pos_weight,
            n_jobs=4,
        )
        model.fit(X_train_imp, y_train)
        p_val = model.predict_proba(X_val_imp)[:, 1]
        joblib.dump(model, model_dir / f"xgb_model_seed{seed}.pkl")
        joblib.dump(imputer, model_dir / f"xgb_imputer_seed{seed}.pkl")
        _save_json(metrics_dir / f"xgb_val_metrics_seed{seed}.json", _compute_metrics(y_val, p_val))


def _train_temporal_family(
    params: dict[str, Any],
    output_root: Path,
    *,
    prefix: str,
    variant: str,
) -> None:
    model_dir = output_root / "models"
    metrics_dir = output_root / "metrics"
    ensure_dir(model_dir)
    ensure_dir(metrics_dir)
    train_df = read_windows("te_train_windows.csv")
    val_df = read_windows("te_val_windows.csv")
    device = torch_device_info(prefer_mps=True)["selected_device"]
    for seed in SEEDS:
        set_seed(int(seed))
        X_train = flattened_to_tensor(train_df)
        y_train = train_df["y"].to_numpy(dtype=np.float32)
        X_val = flattened_to_tensor(val_df)
        y_val = val_df["y"].to_numpy(dtype=np.float32)
        val_phases = val_df["phase"].to_numpy()
        env_train = train_df["phase"].map(phase_to_env).to_numpy(dtype=np.int64) if variant == "invariant" else None
        X_train, X_val, imputer, scaler = transform_with_imputer_scaler(X_train, X_val)

        batch_size = int(params.get("batch_size", 64))
        infer_batch_size = int(params.get("inference_batch_size", 2048))
        epochs = int(params.get("epochs", 6))
        lr = float(params.get("lr", 5e-4))
        weight_decay = float(params.get("weight_decay", 1e-4))
        n_pos = float((y_train == 1).sum())
        n_neg = float((y_train == 0).sum())
        pos_weight = torch.tensor([max(1.0, n_neg / max(n_pos, 1.0))], dtype=torch.float32).to(device)

        if variant == "invariant":
            train_ds = TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32),
                torch.tensor(env_train, dtype=torch.long),
            )
        else:
            train_ds = TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32),
            )
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
        model = build_temporal_model(
            n_features=X_train.shape[1],
            cfg={
                "architecture": str(params.get("architecture", "modern_tcn")),
                "channels": tuple(int(v) for v in params.get("channels", [64, 96, 128])),
                "dilations": tuple(int(v) for v in params.get("dilations", [1, 2, 4])),
                "kernel_size": int(params.get("kernel_size", 3)),
                "dropout": float(params.get("dropout", 0.1)),
                "expansion_ratio": int(params.get("expansion_ratio", 2)),
                "pool": str(params.get("pool", "avg")),
            },
        ).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_state = None
        best_score = -np.inf
        best_threshold = 0.5
        for _ in range(epochs):
            model.train()
            for batch in loader:
                optimizer.zero_grad()
                if variant == "invariant":
                    xb, yb, envb = batch
                    xb = xb.to(device)
                    yb = yb.to(device)
                    envb = envb.to(device)
                    logits, features = model(xb, return_features=True)
                    cls_loss = criterion(logits, yb)
                    normal_mask = yb < 0.5
                    penalty = invariant_penalty(features[normal_mask], envb[normal_mask]) if bool(normal_mask.any()) else features.new_tensor(0.0)
                    loss = cls_loss + float(params.get("penalty_weight", 0.1)) * penalty
                else:
                    xb, yb = batch
                    xb = xb.to(device)
                    yb = yb.to(device)
                    logits = model(xb)
                    loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            model.eval()
            if variant == "adaptable":
                probs = infer_tta_online_probs(
                    model,
                    X_val,
                    device=device,
                    infer_batch_size=infer_batch_size,
                    lr=float(params.get("adaptation_lr", 1e-4)),
                    steps=int(params.get("adaptation_steps", 1)),
                )
            else:
                with torch.no_grad():
                    probs = infer_temporal_probs(model, X_val, device=device, infer_batch_size=infer_batch_size)
            threshold, metrics = best_threshold_by_f1(y_val, probs)
            post_shift_recall = phase_recall(y_val, probs, val_phases, "post_shift", threshold)
            score = float(metrics["f1"]) + 0.10 * float(metrics["recall"]) + 0.35 * float(post_shift_recall)
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        assert best_state is not None
        model.load_state_dict(best_state)
        torch.save(model.state_dict(), model_dir / f"{prefix}_model_seed{seed}.pt")
        joblib.dump(imputer, model_dir / f"{prefix}_imputer_seed{seed}.pkl")
        joblib.dump(scaler, model_dir / f"{prefix}_scaler_seed{seed}.pkl")
        meta = {
            "seed": int(seed),
            "architecture": str(params.get("architecture", "modern_tcn")),
            "n_features": int(X_train.shape[1]),
            "window_size": int(X_train.shape[2]),
            "channels": [int(v) for v in params.get("channels", [64, 96, 128])],
            "dilations": [int(v) for v in params.get("dilations", [1, 2, 4])],
            "kernel_size": int(params.get("kernel_size", 3)),
            "dropout": float(params.get("dropout", 0.1)),
            "expansion_ratio": int(params.get("expansion_ratio", 2)),
            "pool": str(params.get("pool", "avg")),
            "batch_size": batch_size,
            "inference_batch_size": infer_batch_size,
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "best_val_threshold": best_threshold,
        }
        if variant == "adaptable":
            meta["adaptation_lr"] = float(params.get("adaptation_lr", 1e-4))
            meta["adaptation_steps"] = int(params.get("adaptation_steps", 1))
        if variant == "invariant":
            meta["penalty_weight"] = float(params.get("penalty_weight", 0.1))
        _save_json(model_dir / f"{prefix}_meta_seed{seed}.json", meta)
        _save_json(metrics_dir / f"{prefix}_val_metrics_seed{seed}.json", {"selection_score": best_score, "best_val_threshold": best_threshold})


def train_tcn_family(params: dict[str, Any], output_root: Path) -> None:
    _train_temporal_family(params, output_root, prefix="tcn", variant="base")


def train_adaptable_family(params: dict[str, Any], output_root: Path) -> None:
    _train_temporal_family(params, output_root, prefix="adaptable_tcn", variant="adaptable")


def train_invariant_family(params: dict[str, Any], output_root: Path) -> None:
    _train_temporal_family(params, output_root, prefix="invariant_tcn", variant="invariant")


def _load_temporal_artifact(model_dir: Path, prefix: str, seed: int, device: str):
    imputer = joblib.load(model_dir / f"{prefix}_imputer_seed{seed}.pkl")
    scaler = joblib.load(model_dir / f"{prefix}_scaler_seed{seed}.pkl")
    meta = read_json(model_dir / f"{prefix}_meta_seed{seed}.json")
    model = build_temporal_model(
        n_features=int(meta["n_features"]),
        cfg={
            "architecture": meta.get("architecture", "modern_tcn"),
            "channels": tuple(meta.get("channels", [64, 96, 128])),
            "dilations": tuple(meta.get("dilations", [1, 2, 4])),
            "kernel_size": int(meta.get("kernel_size", 3)),
            "dropout": float(meta.get("dropout", 0.1)),
            "expansion_ratio": int(meta.get("expansion_ratio", 2)),
            "pool": str(meta.get("pool", "avg")),
        },
    ).to(device)
    state = torch.load(model_dir / f"{prefix}_model_seed{seed}.pt", map_location="cpu")
    model.load_state_dict(state)
    return model, imputer, scaler, meta


def _transform_windows(df_win: pd.DataFrame, imputer, scaler) -> np.ndarray:
    X = flattened_to_tensor(df_win)
    bsz, n_feat, win = X.shape
    X2 = X.transpose(0, 2, 1).reshape(-1, n_feat)
    X2 = imputer.transform(X2)
    X2 = scaler.transform(X2)
    return X2.reshape(bsz, win, n_feat).transpose(0, 2, 1).astype(np.float32)


def _predict_static(df_rows: pd.DataFrame, feature_cols: list[str], model_dir: Path, prefix: str) -> pd.DataFrame:
    out = df_rows[
        ["source_file", "domain_tag", "split_group", "run_id", "fault_id", "sample_idx", "y", "phase", "onset_step", "transition_len"]
    ].rename(columns={"y": "y_true"}).copy()
    X_raw = df_rows[feature_cols].to_numpy()
    cols = []
    for seed in SEEDS:
        model = joblib.load(model_dir / f"{prefix}_model_seed{seed}.pkl")
        imputer = joblib.load(model_dir / f"{prefix}_imputer_seed{seed}.pkl")
        col = f"p_{prefix}_seed{seed}"
        out[col] = model.predict_proba(imputer.transform(X_raw))[:, 1]
        cols.append(col)
    out[f"p_{prefix}"] = out[cols].mean(axis=1)
    return out


def _predict_temporal(df_win: pd.DataFrame, model_dir: Path, prefix: str, *, inference_kind: str) -> pd.DataFrame:
    device = torch_device_info(prefer_mps=True)["selected_device"]
    out = df_win[
        ["source_file", "domain_tag", "split_group", "run_id", "fault_id", "sample_idx", "y", "phase", "onset_step", "transition_len"]
    ].rename(columns={"y": "y_true"}).copy()
    cols = []
    for seed in SEEDS:
        model, imputer, scaler, meta = _load_temporal_artifact(model_dir, prefix, seed, device)
        Xs = _transform_windows(df_win, imputer, scaler)
        if inference_kind == "tta":
            probs = infer_tta_online_probs(
                model,
                Xs,
                device=device,
                infer_batch_size=int(meta.get("inference_batch_size", 512)),
                lr=float(meta.get("adaptation_lr", 1e-4)),
                steps=int(meta.get("adaptation_steps", 1)),
            )
        else:
            probs = infer_temporal_probs(model, Xs, device=device, infer_batch_size=int(meta.get("inference_batch_size", 512)))
        col = f"p_{prefix.replace('_tcn', '')}_seed{seed}"
        out[col] = probs
        cols.append(col)
        synchronize_torch_device(device)
    out[f"p_{prefix.replace('_tcn', '')}"] = out[cols].mean(axis=1)
    return out


def _align_temporal_to_rows(row_df: pd.DataFrame, pred_df: pd.DataFrame, out_split_group: str, out_prefix: str) -> pd.DataFrame:
    keep_cols = ["source_file", "run_id", "fault_id", "sample_idx"] + [c for c in pred_df.columns if c.startswith(f"p_{out_prefix}")]
    pred_sub = pred_df[keep_cols].drop_duplicates(subset=["source_file", "fault_id", "run_id", "sample_idx"])
    merged = row_df.merge(pred_sub, on=["source_file", "fault_id", "run_id", "sample_idx"], how="left")
    merged["split_group"] = out_split_group
    return merged


def _attach_graphad(df_rows: pd.DataFrame, graphad_artifact_path: Path) -> pd.DataFrame:
    if not graphad_artifact_path.exists():
        return pd.DataFrame(index=df_rows.index)
    artifact = load_graphad_artifact(graphad_artifact_path)
    return infer_graphad(df_rows, artifact)


def _drop_context_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["domain_tag", "split_group", "y_true", "phase", "onset_step", "transition_len"], errors="ignore")


def _merge_prediction_blocks(blocks: list[pd.DataFrame]) -> pd.DataFrame:
    merged = blocks[0].copy()
    for block in blocks[1:]:
        merged = merged.merge(_drop_context_cols(block), on=["source_file", "run_id", "fault_id", "sample_idx"], how="left")
    return merged


def _apply_ppt_bundle(df: pd.DataFrame, ppt_cfg: dict) -> pd.DataFrame:
    fusion_method = str(ppt_cfg.get("fusion", {}).get("method", "uncertainty_soft_fusion")).strip().lower()
    if fusion_method == "conditional_grayzone_soft_fusion":
        out = attach_conditional_soft_fusion_features(df, cfg=ppt_cfg)
        flags = build_policy_flags(out, entropy_col="ppt_h_avg", cfg=ppt_cfg)
        for col in flags.columns:
            out[col] = flags[col]
        decision_thr = float(
            ppt_cfg.get("boundary_soft_fusion", {}).get(
                "decision_threshold",
                ppt_cfg.get("soft_fusion", {}).get("decision_threshold", 0.5),
            )
        )
        if "p_llm_raw" in out.columns:
            out["ppt_prompt_decision"] = np.where(out["p_llm_raw"] >= decision_thr, "anomaly", "normal")
        prompt_spec = ppt_cfg.get("prompt_spec", {})
        if prompt_spec:
            out["ppt_prompt_spec_name"] = str(prompt_spec.get("name", "ppt_complete"))
    else:
        out = attach_ppt_full_features(df, cfg=ppt_cfg)
        flags = build_policy_flags(out, cfg=ppt_cfg)
        for col in flags.columns:
            out[col] = flags[col]
        decision_thr = float(ppt_cfg.get("soft_fusion", {}).get("decision_threshold", 0.5))
        if "p_llm_raw" in out.columns:
            out["ppt_prompt_decision"] = np.where(out["p_llm_raw"] >= decision_thr, "anomaly", "normal")
    for seed in SEEDS:
        rf_col = f"p_rf_seed{seed}"
        xgb_col = f"p_xgb_seed{seed}"
        tcn_col = f"p_tcn_seed{seed}"
        if not all(col in out.columns for col in [rf_col, xgb_col, tcn_col]):
            continue
        if fusion_method == "conditional_grayzone_soft_fusion":
            seeded = attach_conditional_soft_fusion_features(out, rf_col=rf_col, xgb_col=xgb_col, tcn_col=tcn_col, cfg=ppt_cfg, seed=seed)
            rename_map = {
                "p_local": f"p_local_seed{seed}",
                "p_utar_base": f"p_utar_base_seed{seed}",
                "ensemble_entropy": f"ensemble_entropy_seed{seed}",
                "model_discrepancy": f"model_discrepancy_seed{seed}",
                "ppt_d_ens": f"ppt_d_ens_seed{seed}",
                "ppt_h_avg": f"ppt_h_avg_seed{seed}",
                "ppt_u_total": f"ppt_u_total_seed{seed}",
                "ppt_uncertainty_weight": f"ppt_uncertainty_weight_seed{seed}",
                "ppt_uncertainty_soft_fusion": f"ppt_uncertainty_soft_fusion_seed{seed}",
                "ppt_tau": f"ppt_tau_seed{seed}",
                "ppt_m_q": f"ppt_m_q_seed{seed}",
                "ppt_gray_zone_indicator": f"ppt_gray_zone_indicator_seed{seed}",
                "ppt_boundary_weight": f"ppt_boundary_weight_seed{seed}",
                "p_llm_raw": f"p_llm_raw_seed{seed}",
                "p_llm_calibrated": f"p_llm_calibrated_seed{seed}",
                "ppt_s_llm": f"p_llm_seed{seed}",
                "p_soft_fusion": f"p_soft_fusion_seed{seed}",
            }
        else:
            seeded = attach_ppt_full_features(out, rf_col=rf_col, xgb_col=xgb_col, tcn_col=tcn_col, cfg=ppt_cfg)
            rename_map = {
                "p_local": f"p_local_seed{seed}",
                "ppt_d_ens": f"ppt_d_ens_seed{seed}",
                "ppt_h_avg": f"ppt_h_avg_seed{seed}",
                "ppt_u_total": f"ppt_u_total_seed{seed}",
                "ppt_w": f"ppt_w_seed{seed}",
                "p_llm_raw": f"p_llm_raw_seed{seed}",
                "p_llm_calibrated": f"p_llm_calibrated_seed{seed}",
                "p_soft_fusion": f"p_soft_fusion_seed{seed}",
            }
        for src_col, dst_col in rename_map.items():
            if src_col in seeded.columns:
                out[dst_col] = seeded[src_col].to_numpy()
    return out


def generate_suite_predictions(output_root: Path, *, ppt_cfg: dict | None = None) -> dict[str, Path]:
    model_dir = output_root / "models"
    pred_dir = output_root / "predictions"
    ensure_dir(pred_dir)
    ppt_cfg = ppt_cfg or {}
    feature_cols = _load_feature_cols()

    row_val = _read_rows("te_val_rows.csv")
    row_main = _read_rows("te_test_main_rows.csv")
    row_cost = _read_rows("te_test_cost_rows.csv")
    win_val = read_windows("te_val_windows.csv")
    win_full = read_windows("te_test_full_windows_tcn.csv")

    rf_val = _predict_static(row_val, feature_cols, model_dir, "rf")
    xgb_val = _predict_static(row_val, feature_cols, model_dir, "xgb")
    tcn_val = _predict_temporal(win_val, model_dir, "tcn", inference_kind="standard")
    adaptable_val = _predict_temporal(win_val, model_dir, "adaptable_tcn", inference_kind="tta")
    invariant_val = _predict_temporal(win_val, model_dir, "invariant_tcn", inference_kind="standard")

    val_df = _merge_prediction_blocks([rf_val, xgb_val, tcn_val, adaptable_val, invariant_val])
    graphad_val = _attach_graphad(row_val, DEFAULT_GRAPHAD_ARTIFACT)
    for col in graphad_val.columns:
        val_df[col] = graphad_val[col]
    val_df = _apply_ppt_bundle(val_df, ppt_cfg)
    write_csv(pred_dir / "ppt_full_val_predictions.csv", val_df)

    rf_main = _predict_static(row_main, feature_cols, model_dir, "rf")
    xgb_main = _predict_static(row_main, feature_cols, model_dir, "xgb")
    tcn_full = _predict_temporal(win_full, model_dir, "tcn", inference_kind="standard")
    adaptable_full = _predict_temporal(win_full, model_dir, "adaptable_tcn", inference_kind="tta")
    invariant_full = _predict_temporal(win_full, model_dir, "invariant_tcn", inference_kind="standard")

    main_df = _merge_prediction_blocks(
        [
            rf_main,
            xgb_main,
            _align_temporal_to_rows(row_main.rename(columns={"y": "y_true"}), tcn_full, "test_main", "tcn"),
            _align_temporal_to_rows(row_main.rename(columns={"y": "y_true"}), adaptable_full, "test_main", "adaptable"),
            _align_temporal_to_rows(row_main.rename(columns={"y": "y_true"}), invariant_full, "test_main", "invariant"),
        ]
    )
    graphad_main = _attach_graphad(row_main, DEFAULT_GRAPHAD_ARTIFACT)
    for col in graphad_main.columns:
        main_df[col] = graphad_main[col]
    main_df = _apply_ppt_bundle(main_df, ppt_cfg)
    write_csv(pred_dir / "ppt_full_test_main_predictions.csv", main_df)

    rf_cost = _predict_static(row_cost, feature_cols, model_dir, "rf")
    xgb_cost = _predict_static(row_cost, feature_cols, model_dir, "xgb")
    cost_df = _merge_prediction_blocks(
        [
            rf_cost,
            xgb_cost,
            _align_temporal_to_rows(row_cost.rename(columns={"y": "y_true"}), tcn_full, "test_cost", "tcn"),
            _align_temporal_to_rows(row_cost.rename(columns={"y": "y_true"}), adaptable_full, "test_cost", "adaptable"),
            _align_temporal_to_rows(row_cost.rename(columns={"y": "y_true"}), invariant_full, "test_cost", "invariant"),
        ]
    )
    graphad_cost = _attach_graphad(row_cost, DEFAULT_GRAPHAD_ARTIFACT)
    for col in graphad_cost.columns:
        cost_df[col] = graphad_cost[col]
    cost_df = _apply_ppt_bundle(cost_df, ppt_cfg)
    write_csv(pred_dir / "ppt_full_test_cost_predictions.csv", cost_df)
    return {
        "val": pred_dir / "ppt_full_val_predictions.csv",
        "main": pred_dir / "ppt_full_test_main_predictions.csv",
        "cost": pred_dir / "ppt_full_test_cost_predictions.csv",
    }


def evaluate_suite_predictions(output_root: Path) -> dict[str, Path]:
    pred_dir = output_root / "predictions"
    metrics_dir = output_root / "metrics"
    ensure_dir(metrics_dir)
    datasets = {
        "val": read_csv(pred_dir / "ppt_full_val_predictions.csv"),
        "main": read_csv(pred_dir / "ppt_full_test_main_predictions.csv"),
        "cost": read_csv(pred_dir / "ppt_full_test_cost_predictions.csv"),
    }
    model_prefixes = ["rf", "xgb", "tcn", "adaptable", "invariant", "local", "soft_fusion"]
    ensemble_cols = {
        "rf": "p_rf",
        "xgb": "p_xgb",
        "tcn": "p_tcn",
        "adaptable": "p_adaptable",
        "invariant": "p_invariant",
        "local": "p_local",
        "llm_calibrated": "p_llm_calibrated",
        "soft_fusion": "p_soft_fusion",
    }
    threshold_rows = []
    seed_rows = []
    thresholds: dict[str, dict[int, float]] = {prefix: {} for prefix in model_prefixes}
    val_recall_by_model: dict[str, dict[int, float]] = {prefix: {} for prefix in model_prefixes}
    y_val = datasets["val"]["y_true"].to_numpy(dtype=int)
    for prefix in model_prefixes:
        for seed in SEEDS:
            col = f"p_{prefix}_seed{seed}"
            if col not in datasets["val"].columns:
                continue
            tau, metrics = best_threshold_by_f1(y_val, datasets["val"][col].to_numpy(dtype=float))
            thresholds[prefix][seed] = float(tau)
            val_recall_by_model[prefix][seed] = float(metrics["recall"])
            threshold_rows.append({"model": prefix, "seed": int(seed), "threshold": float(tau), **metrics})

    for split_name, df in datasets.items():
        y_true = df["y_true"].to_numpy(dtype=int)
        for prefix in model_prefixes:
            for seed, tau in thresholds[prefix].items():
                col = f"p_{prefix}_seed{seed}"
                metrics = _compute_metrics(y_true, df[col].to_numpy(dtype=float), tau)
                seed_rows.append(
                    {
                        "split": split_name,
                        "model": prefix,
                        "seed": int(seed),
                        "threshold": tau,
                        "f1": float(metrics["f1"]),
                        "recall": float(metrics["recall"]),
                        "precision": float(metrics["precision"]),
                        "auc": float(metrics["roc_auc"]) if metrics["roc_auc"] is not None else np.nan,
                        "prr": 1.0 if split_name == "val" else performance_retention_rate(val_recall_by_model[prefix][seed], float(metrics["recall"])),
                    }
                )
    threshold_df = pd.DataFrame(threshold_rows)
    seed_df = pd.DataFrame(seed_rows)
    summary_df = seed_df.groupby(["split", "model"]).agg(
        n_seeds=("seed", "count"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
        auc_mean=("auc", "mean"),
        auc_std=("auc", "std"),
        prr_mean=("prr", "mean"),
        prr_std=("prr", "std"),
        threshold_mean=("threshold", "mean"),
        threshold_std=("threshold", "std"),
    ).reset_index()

    ensemble_threshold_rows = []
    ensemble_metric_rows = []
    ensemble_thresholds: dict[str, float] = {}
    ensemble_val_recall: dict[str, float] = {}
    for label, col in ensemble_cols.items():
        if col not in datasets["val"].columns:
            continue
        tau, metrics = best_threshold_by_f1(y_val, datasets["val"][col].to_numpy(dtype=float))
        ensemble_thresholds[label] = float(tau)
        ensemble_val_recall[label] = float(metrics["recall"])
        ensemble_threshold_rows.append({"model": label, "threshold": float(tau), **metrics})

    for split_name, df in datasets.items():
        y_true = df["y_true"].to_numpy(dtype=int)
        for label, col in ensemble_cols.items():
            if label not in ensemble_thresholds or col not in df.columns:
                continue
            tau = ensemble_thresholds[label]
            metrics = _compute_metrics(y_true, df[col].to_numpy(dtype=float), tau)
            ensemble_metric_rows.append(
                {
                    "split": split_name,
                    "model": label,
                    "threshold": tau,
                    "f1": float(metrics["f1"]),
                    "recall": float(metrics["recall"]),
                    "precision": float(metrics["precision"]),
                    "auc": float(metrics["roc_auc"]) if metrics["roc_auc"] is not None else np.nan,
                    "prr": 1.0 if split_name == "val" else performance_retention_rate(ensemble_val_recall[label], float(metrics["recall"])),
                }
            )

    ensemble_threshold_df = pd.DataFrame(ensemble_threshold_rows)
    ensemble_metric_df = pd.DataFrame(ensemble_metric_rows)
    ensemble_summary_df = ensemble_metric_df.groupby(["split", "model"]).agg(
        f1_mean=("f1", "mean"),
        recall_mean=("recall", "mean"),
        precision_mean=("precision", "mean"),
        auc_mean=("auc", "mean"),
        prr_mean=("prr", "mean"),
        threshold_mean=("threshold", "mean"),
    ).reset_index()

    write_csv(metrics_dir / "ppt_full_thresholds.csv", threshold_df)
    write_csv(metrics_dir / "ppt_full_seed_metrics.csv", seed_df)
    write_csv(metrics_dir / "ppt_full_summary.csv", summary_df)
    write_csv(metrics_dir / "ppt_full_ensemble_thresholds.csv", ensemble_threshold_df)
    write_csv(metrics_dir / "ppt_full_ensemble_metrics.csv", ensemble_metric_df)
    write_csv(metrics_dir / "ppt_full_ensemble_summary.csv", ensemble_summary_df)
    return {
        "thresholds": metrics_dir / "ppt_full_thresholds.csv",
        "seed_metrics": metrics_dir / "ppt_full_seed_metrics.csv",
        "summary": metrics_dir / "ppt_full_summary.csv",
        "ensemble_thresholds": metrics_dir / "ppt_full_ensemble_thresholds.csv",
        "ensemble_metrics": metrics_dir / "ppt_full_ensemble_metrics.csv",
        "ensemble_summary": metrics_dir / "ppt_full_ensemble_summary.csv",
    }
