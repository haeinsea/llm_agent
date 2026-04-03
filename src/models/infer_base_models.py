from __future__ import annotations

import json
from pathlib import Path
from typing import List
import time

import joblib
import numpy as np
import pandas as pd

try:
    import torch
except Exception:
    torch = None

import re

from src.models.graphad import GRAPHAD_FEATURES, infer_graphad, load_graphad_artifact
from src.models.temporal_backbone import build_temporal_model, temporal_model_display_name
from src.utils.device import get_torch_device, synchronize_torch_device, torch_device_info
from src.utils.experiment import ensemble_component_label, get_seed_list
from src.utils.io import ensure_dir, read_yaml, write_csv, write_json
from src.utils.routing import build_routing_features

WINDOW_COL_PATTERN = re.compile(r"^(?P<base>.+)_t(?P<lag>\d+|-?\d+)$")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
META_DIR = DATA_DIR / "meta"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
PRED_DIR = OUTPUT_DIR / "predictions"
METRIC_DIR = OUTPUT_DIR / "metrics"
CONFIG_DIR = PROJECT_ROOT / "configs"

SEEDS = get_seed_list()
TEMPORAL_MODEL_NAME = temporal_model_display_name(read_yaml(CONFIG_DIR / "train_tcn.yaml", default={}).get("architecture", "modern_tcn"))
RF_COMPONENT = ensemble_component_label("RF")
XGB_COMPONENT = ensemble_component_label("XGB")
TEMPORAL_COMPONENT = ensemble_component_label(TEMPORAL_MODEL_NAME)
GRAPHAD_COMPONENT = "GraphAD+"
AVG_ENSEMBLE_STACK_COMPONENT = "Avg. Ensemble Stack"
BASE_STACK_COMPONENT = "UTAR Base Stack"


def load_feature_cols() -> List[str]:
    with open(META_DIR / "feature_columns.json", "r", encoding="utf-8") as f:
        return json.load(f)


def infer_window_feature_structure(columns):
    feat_cols = [c for c in columns if WINDOW_COL_PATTERN.match(c)]
    feats = {}
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


def read_rows(name: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / name)


def read_windows(name: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / name)


def read_graphad_artifact() -> dict:
    path = MODEL_DIR / "graphad_artifact.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing GraphAD artifact: {path}. Run `python -m src.models.train_graphad` first."
        )
    return load_graphad_artifact(path)


def load_compat_joblib(path: Path):
    obj = joblib.load(path)
    if hasattr(obj, "_fit_dtype") and not hasattr(obj, "_fill_dtype"):
        obj._fill_dtype = obj._fit_dtype
    return obj


def _runtime_record(
    split: str,
    component: str,
    total_latency_ms: float,
    n_samples: int,
    device: str,
    measured_on_split: str | None = None,
    notes: str = "",
    seed: int | None = None,
) -> dict:
    return {
        "split": split,
        "component": component,
        "n_samples": int(n_samples),
        "total_latency_ms": float(total_latency_ms),
        "avg_latency_ms_per_sample": float(total_latency_ms / n_samples) if n_samples else 0.0,
        "device": str(device),
        "measured_on_split": str(measured_on_split or split),
        "notes": str(notes),
        "seed": int(seed) if seed is not None else np.nan,
    }


def _combine_devices(records: list[dict]) -> str:
    devices = sorted({str(rec["device"]) for rec in records if rec.get("device")})
    if not devices:
        return "cpu"
    if len(devices) == 1:
        return devices[0]
    return "+".join(devices)


def _find_runtime_records(records: list[dict], split: str, component: str) -> list[dict]:
    return [rec for rec in records if rec["split"] == split and rec["component"] == component]


def _allocate_runtime_from_source(
    records: list[dict],
    source_split: str,
    target_split: str,
    component: str,
    n_samples: int,
) -> None:
    source_records = _find_runtime_records(records, source_split, component)
    if not source_records:
        raise KeyError(f"Runtime record not found: split={source_split}, component={component}")
    seeded = [rec for rec in source_records if not pd.isna(rec.get("seed", np.nan))]
    if seeded:
        for source in seeded:
            records.append(
                _runtime_record(
                    split=target_split,
                    component=component,
                    total_latency_ms=float(source["avg_latency_ms_per_sample"]) * float(n_samples),
                    n_samples=n_samples,
                    device=str(source["device"]),
                    measured_on_split=source_split,
                    notes=f"allocated from {source_split} per-sample average",
                    seed=int(source["seed"]),
                )
            )
        return
    source = source_records[0]
    records.append(
        _runtime_record(
            split=target_split,
            component=component,
            total_latency_ms=float(source["avg_latency_ms_per_sample"]) * float(n_samples),
            n_samples=n_samples,
            device=str(source["device"]),
            measured_on_split=source_split,
            notes=f"allocated from {source_split} per-sample average",
        )
    )


def _append_stack_records(records: list[dict], split: str, n_samples: int, *, include_graphad: bool) -> None:
    shared_parts = _find_runtime_records(records, split, GRAPHAD_COMPONENT)[:1] if include_graphad else []
    per_seed_parts: dict[int, list[dict]] = {}
    for component in [RF_COMPONENT, XGB_COMPONENT, TEMPORAL_COMPONENT]:
        for rec in _find_runtime_records(records, split, component):
            if pd.isna(rec.get("seed", np.nan)):
                continue
            per_seed_parts.setdefault(int(rec["seed"]), []).append(rec)

    stack_component = BASE_STACK_COMPONENT if include_graphad else AVG_ENSEMBLE_STACK_COMPONENT
    stack_notes = "rf+xgb+temporal+graphad" if include_graphad else "rf+xgb+temporal"
    for seed, parts in sorted(per_seed_parts.items()):
        if len(parts) < 3:
            continue
        full_parts = parts + shared_parts
        records.append(
            _runtime_record(
                split=split,
                component=stack_component,
                total_latency_ms=sum(float(part["total_latency_ms"]) for part in full_parts),
                n_samples=n_samples,
                device=_combine_devices(full_parts),
                measured_on_split=split,
                notes=stack_notes,
                seed=seed,
            )
        )


def _write_runtime_artifacts(records: list[dict], device_info: dict) -> None:
    ensure_dir(METRIC_DIR)
    df = pd.DataFrame(records)
    if "seed" in df.columns:
        df = df.sort_values(["split", "component", "seed"], na_position="last").reset_index(drop=True)
    write_csv(METRIC_DIR / "base_inference_runtime.csv", df)
    summary: dict[str, dict[str, dict]] = {}
    for (split, component), group in df.groupby(["split", "component"], sort=False):
        total_vals = group["total_latency_ms"].astype(float)
        avg_vals = group["avg_latency_ms_per_sample"].astype(float)
        measured_on = sorted({str(v) for v in group["measured_on_split"].astype(str).tolist() if str(v)})
        notes = sorted({str(v) for v in group["notes"].astype(str).tolist() if str(v) and str(v) != "nan"})
        seeds = sorted(int(v) for v in group["seed"].dropna().astype(int).tolist()) if "seed" in group.columns else []
        summary.setdefault(str(split), {})[str(component)] = {
            "n_samples": int(group["n_samples"].iloc[0]),
            "n_measurements": int(len(group)),
            "seeds": seeds,
            "total_latency_ms": float(total_vals.mean()),
            "total_latency_ms_std": float(total_vals.std(ddof=1)) if len(group) > 1 else 0.0,
            "avg_latency_ms_per_sample": float(avg_vals.mean()),
            "avg_latency_ms_per_sample_std": float(avg_vals.std(ddof=1)) if len(group) > 1 else 0.0,
            "device": _combine_devices(group.to_dict("records")),
            "measured_on_split": ",".join(measured_on),
            "notes": "; ".join(notes),
        }
    write_json(
        METRIC_DIR / "base_inference_runtime_summary.json",
        {
            "device_info": device_info,
            "records": summary,
        },
    )


def base_predict_rows(df_rows: pd.DataFrame, feature_cols: List[str], split_name: str) -> tuple[pd.DataFrame, list[dict]]:
    out = df_rows[
        ["source_file", "domain_tag", "split_group", "run_id", "fault_id", "sample_idx", "y", "phase", "onset_step", "transition_len"]
    ].copy()
    out = out.rename(columns={"y": "y_true"})

    X_raw = df_rows[feature_cols].to_numpy()
    records: list[dict] = []

    rf_cols = []
    for seed in SEEDS:
        started_rf = time.perf_counter()
        rf = joblib.load(MODEL_DIR / f"rf_model_seed{seed}.pkl")
        rf_imp = load_compat_joblib(MODEL_DIR / f"rf_imputer_seed{seed}.pkl")
        X_rf = rf_imp.transform(X_raw)
        col = f"p_rf_seed{seed}"
        out[col] = rf.predict_proba(X_rf)[:, 1]
        rf_cols.append(col)
        rf_ms = (time.perf_counter() - started_rf) * 1000.0
        records.append(_runtime_record(split_name, RF_COMPONENT, rf_ms, len(df_rows), "cpu", seed=seed))

    xgb_cols = []
    for seed in SEEDS:
        started_xgb = time.perf_counter()
        xgb = joblib.load(MODEL_DIR / f"xgb_model_seed{seed}.pkl")
        xgb_imp = load_compat_joblib(MODEL_DIR / f"xgb_imputer_seed{seed}.pkl")
        X_xgb = xgb_imp.transform(X_raw)
        col = f"p_xgb_seed{seed}"
        out[col] = xgb.predict_proba(X_xgb)[:, 1]
        xgb_cols.append(col)
        xgb_ms = (time.perf_counter() - started_xgb) * 1000.0
        records.append(_runtime_record(split_name, XGB_COMPONENT, xgb_ms, len(df_rows), "cpu", seed=seed))

    out["p_rf"] = out[rf_cols].mean(axis=1)
    out["p_xgb"] = out[xgb_cols].mean(axis=1)
    return out, records


def tcn_predict_windows(
    df_win: pd.DataFrame,
    split_name: str,
    device: str,
    infer_batch_size: int,
) -> tuple[pd.DataFrame, list[dict]]:
    if torch is None:
        raise ImportError("PyTorch is required for temporal-model inference.")

    out = df_win[
        ["source_file", "domain_tag", "split_group", "run_id", "fault_id", "sample_idx", "y", "phase", "onset_step", "transition_len"]
    ].copy()
    out = out.rename(columns={"y": "y_true"})

    X = flattened_to_tensor(df_win)
    tcn_cols = []
    records: list[dict] = []
    for seed in SEEDS:
        started_seed = time.perf_counter()
        # 학습 코드에서 imputer + scaler를 둘 다 저장한 경우 대응
        imp_path = MODEL_DIR / f"tcn_imputer_seed{seed}.pkl"
        scaler_path = MODEL_DIR / f"tcn_scaler_seed{seed}.pkl"

        imputer = load_compat_joblib(imp_path) if imp_path.exists() else None
        scaler = joblib.load(scaler_path)

        with open(MODEL_DIR / f"tcn_meta_seed{seed}.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        B, F, T = X.shape
        X2 = X.transpose(0, 2, 1).reshape(-1, F)

        if imputer is not None:
            X2 = imputer.transform(X2)
        X2 = scaler.transform(X2)
        Xs = X2.reshape(B, T, F).transpose(0, 2, 1)

        model = build_temporal_model(
            n_features=int(meta["n_features"]),
            cfg={
                "architecture": meta.get("architecture", "tcn"),
                "channels": tuple(meta.get("channels", [64, 64, 64])),
                "dilations": tuple(meta.get("dilations", [1, 2, 4])),
                "kernel_size": int(meta.get("kernel_size", 3)),
                "dropout": float(meta.get("dropout", 0.1)),
                "expansion_ratio": int(meta.get("expansion_ratio", 2)),
                "pool": str(meta.get("pool", "avg")),
            },
        ).to(device)
        state = torch.load(MODEL_DIR / f"tcn_model_seed{seed}.pt", map_location="cpu")
        model.load_state_dict(state)
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
        p = np.concatenate(probs, axis=0)

        col = f"p_tcn_seed{seed}"
        out[col] = p
        tcn_cols.append(col)
        total_ms = (time.perf_counter() - started_seed) * 1000.0
        records.append(_runtime_record(split_name, TEMPORAL_COMPONENT, total_ms, len(df_win), device, seed=seed))

    out["p_tcn"] = out[tcn_cols].mean(axis=1)
    return out, records


def align_tcn_to_rows(
    row_pred: pd.DataFrame,
    tcn_pred: pd.DataFrame,
    out_split_group: str,
) -> pd.DataFrame:
    # tcn_pred는 full-run test에서 왔으므로 split_group은 제거하고 key align
    tcn_keep = [
        "source_file",
        "run_id",
        "fault_id",
        "sample_idx",
    ] + [c for c in tcn_pred.columns if c.startswith("p_tcn_seed")] + ["p_tcn"]

    tcn_sub = tcn_pred[tcn_keep].drop_duplicates(
        subset=["source_file", "fault_id", "run_id", "sample_idx"]
    )

    merged = row_pred.merge(
        tcn_sub,
        on=["source_file", "fault_id", "run_id", "sample_idx"],
        how="left",
    )

    merged["split_group"] = out_split_group
    merged["p_ensemble"] = merged[["p_rf", "p_xgb", "p_tcn"]].mean(axis=1)
    return merged


def align_graphad_to_rows(
    row_pred: pd.DataFrame,
    graphad_pred: pd.DataFrame,
    out_split_group: str,
) -> pd.DataFrame:
    graphad_keep = ["source_file", "run_id", "fault_id", "sample_idx"] + [c for c in GRAPHAD_FEATURES if c in graphad_pred.columns]
    graphad_sub = graphad_pred[graphad_keep].drop_duplicates(
        subset=["source_file", "fault_id", "run_id", "sample_idx"]
    )
    merged = row_pred.merge(
        graphad_sub,
        on=["source_file", "fault_id", "run_id", "sample_idx"],
        how="left",
    )
    merged["split_group"] = out_split_group
    return merged


def merge_graphad_same_rows(row_pred: pd.DataFrame, graphad_pred: pd.DataFrame) -> pd.DataFrame:
    key = ["source_file", "domain_tag", "split_group", "run_id", "fault_id", "sample_idx", "y_true", "phase", "onset_step", "transition_len"]
    keep_cols = key + [c for c in GRAPHAD_FEATURES if c in graphad_pred.columns]
    return row_pred.merge(graphad_pred[keep_cols], on=key, how="inner")


def merge_base_predictions_same_rows(row_pred: pd.DataFrame, tcn_pred: pd.DataFrame) -> pd.DataFrame:
    key = ["source_file", "domain_tag", "split_group", "run_id", "fault_id", "sample_idx", "y_true", "phase", "onset_step", "transition_len"]
    keep_cols = key + [c for c in tcn_pred.columns if c.startswith("p_tcn_seed")] + ["p_tcn"]
    merged = row_pred.merge(tcn_pred[keep_cols], on=key, how="inner")
    merged["p_ensemble"] = merged[["p_rf", "p_xgb", "p_tcn"]].mean(axis=1)
    return merged


def save_pred(df: pd.DataFrame, filename: str) -> None:
    ensure_dir(PRED_DIR)
    df.to_csv(PRED_DIR / filename, index=False)
    print(f"Saved predictions: {filename} ({len(df):,})")


def attach_routing_columns(df: pd.DataFrame, routing_cfg: dict) -> pd.DataFrame:
    routing_features = build_routing_features(df, routing_cfg)
    for col in ["p_utar_base", "ensemble_entropy", "model_discrepancy"]:
        df[col] = routing_features[col].to_numpy()
    return df


def main() -> None:
    ensure_dir(PRED_DIR)
    ensure_dir(METRIC_DIR)
    feature_cols = load_feature_cols()
    graphad_artifact = read_graphad_artifact()
    routing_cfg = read_yaml(CONFIG_DIR / "routing.yaml", default={})
    device_info = torch_device_info(prefer_mps=True)
    device = device_info["selected_device"]
    tcn_cfg = read_yaml(CONFIG_DIR / "train_tcn.yaml", default={})
    infer_batch_size = int(tcn_cfg.get("inference_batch_size", max(256, int(tcn_cfg.get("batch_size", 128)) * 4)))
    runtime_records: list[dict] = []
    print(f"[base inference] temporal device={device} infer_batch_size={infer_batch_size}")
    print(
        "[base inference] "
        f"cuda_available={device_info['cuda_available']} "
        f"mps_built={device_info['mps_built']} "
        f"mps_available={device_info['mps_available']}"
    )

    # val: 기존처럼 같은 row/window에서 바로 merge 가능
    row_val = read_rows("te_val_rows.csv")
    win_val = read_windows("te_val_windows.csv")
    pred_val_rows, val_row_runtime = base_predict_rows(row_val, feature_cols, split_name="val")
    runtime_records.extend(val_row_runtime)
    pred_val_tcn, val_tcn_runtime = tcn_predict_windows(win_val, split_name="val", device=device, infer_batch_size=infer_batch_size)
    runtime_records.extend(val_tcn_runtime)
    pred_val = merge_base_predictions_same_rows(
        pred_val_rows,
        pred_val_tcn,
    )
    started_graphad_val = time.perf_counter()
    pred_graphad_val = pd.concat(
        [
            row_val[
                ["source_file", "domain_tag", "split_group", "run_id", "fault_id", "sample_idx", "y", "phase", "onset_step", "transition_len"]
            ].rename(columns={"y": "y_true"}),
            infer_graphad(row_val, graphad_artifact),
        ],
        axis=1,
    )
    graphad_val_ms = (time.perf_counter() - started_graphad_val) * 1000.0
    runtime_records.append(_runtime_record("val", GRAPHAD_COMPONENT, graphad_val_ms, len(row_val), "cpu"))
    pred_val = merge_graphad_same_rows(pred_val, pred_graphad_val)
    pred_val = attach_routing_columns(pred_val, routing_cfg)
    _append_stack_records(runtime_records, "val", len(pred_val), include_graphad=False)
    _append_stack_records(runtime_records, "val", len(pred_val), include_graphad=True)
    save_pred(pred_val, "base_val_predictions.csv")

    # test: temporal backbone과 GraphAD는 full-run contiguous test에서 추론
    win_test_full = read_windows("te_test_full_windows_tcn.csv")
    row_test_full = read_rows("te_test_full_rows_tcn.csv")
    pred_tcn_test_full, test_full_tcn_runtime = tcn_predict_windows(
        win_test_full,
        split_name="test_full",
        device=device,
        infer_batch_size=infer_batch_size,
    )
    runtime_records.extend(test_full_tcn_runtime)
    started_graphad_full = time.perf_counter()
    pred_graphad_test_full = pd.concat(
        [
            row_test_full[
                ["source_file", "domain_tag", "split_group", "run_id", "fault_id", "sample_idx", "y", "phase", "onset_step", "transition_len"]
            ].rename(columns={"y": "y_true"}),
            infer_graphad(row_test_full, graphad_artifact),
        ],
        axis=1,
    )
    graphad_full_ms = (time.perf_counter() - started_graphad_full) * 1000.0
    runtime_records.append(_runtime_record("test_full", GRAPHAD_COMPONENT, graphad_full_ms, len(row_test_full), "cpu"))

    # 공통 4000 rows
    row_main = read_rows("te_test_main_rows.csv")
    pred_main_rows, main_row_runtime = base_predict_rows(row_main, feature_cols, split_name="main")
    runtime_records.extend(main_row_runtime)
    _allocate_runtime_from_source(runtime_records, "test_full", "main", TEMPORAL_COMPONENT, len(row_main))
    _allocate_runtime_from_source(runtime_records, "test_full", "main", GRAPHAD_COMPONENT, len(row_main))
    pred_main = align_tcn_to_rows(pred_main_rows, pred_tcn_test_full, out_split_group="test_main")
    pred_main = align_graphad_to_rows(pred_main, pred_graphad_test_full, out_split_group="test_main")
    pred_main = attach_routing_columns(pred_main, routing_cfg)
    _append_stack_records(runtime_records, "main", len(pred_main), include_graphad=False)
    _append_stack_records(runtime_records, "main", len(pred_main), include_graphad=True)
    save_pred(pred_main, "base_test_main_predictions.csv")

    # 공통 cost rows
    row_cost = read_rows("te_test_cost_rows.csv")
    pred_cost_rows, cost_row_runtime = base_predict_rows(row_cost, feature_cols, split_name="cost")
    runtime_records.extend(cost_row_runtime)
    _allocate_runtime_from_source(runtime_records, "test_full", "cost", TEMPORAL_COMPONENT, len(row_cost))
    _allocate_runtime_from_source(runtime_records, "test_full", "cost", GRAPHAD_COMPONENT, len(row_cost))
    pred_cost = align_tcn_to_rows(pred_cost_rows, pred_tcn_test_full, out_split_group="test_cost")
    pred_cost = align_graphad_to_rows(pred_cost, pred_graphad_test_full, out_split_group="test_cost")
    pred_cost = attach_routing_columns(pred_cost, routing_cfg)
    _append_stack_records(runtime_records, "cost", len(pred_cost), include_graphad=False)
    _append_stack_records(runtime_records, "cost", len(pred_cost), include_graphad=True)
    save_pred(pred_cost, "base_test_cost_predictions.csv")

    _write_runtime_artifacts(runtime_records, device_info)

    # 참고용: temporal model alignment coverage 출력
    main_cov = float(pred_main["p_tcn"].notna().mean()) if len(pred_main) else 0.0
    cost_cov = float(pred_cost["p_tcn"].notna().mean()) if len(pred_cost) else 0.0
    print(f"{TEMPORAL_MODEL_NAME} coverage on test_main rows: {main_cov:.4f}")
    print(f"{TEMPORAL_MODEL_NAME} coverage on test_cost rows: {cost_cov:.4f}")
    print("Base inference completed.")


if __name__ == "__main__":
    main()
