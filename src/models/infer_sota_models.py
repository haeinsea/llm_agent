from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models.temporal_sota import infer_temporal_probs, infer_tta_online_probs, load_temporal_artifact, transform_windows
from src.utils.device import torch_device_info
from src.utils.experiment import get_seed_list
from src.utils.io import ensure_dir, read_csv, read_yaml, write_csv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
PRED_DIR = OUTPUT_DIR / "predictions"
CONFIG_DIR = PROJECT_ROOT / "configs"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SEEDS = get_seed_list()
KEY_COLS = ["source_file", "domain_tag", "split_group", "run_id", "fault_id", "sample_idx", "y_true", "phase", "onset_step", "transition_len"]


def _load_windows(name: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / name)


def _window_key_frame(df_win: pd.DataFrame) -> pd.DataFrame:
    return df_win[
        ["source_file", "domain_tag", "split_group", "run_id", "fault_id", "sample_idx", "y", "phase", "onset_step", "transition_len"]
    ].rename(columns={"y": "y_true"}).copy()


def _align_temporal_to_rows(row_df: pd.DataFrame, pred_df: pd.DataFrame, out_split_group: str, prefix: str) -> pd.DataFrame:
    keep_cols = ["source_file", "run_id", "fault_id", "sample_idx"] + [c for c in pred_df.columns if c.startswith(f"p_{prefix}")]
    pred_sub = pred_df[keep_cols].drop_duplicates(subset=["source_file", "fault_id", "run_id", "sample_idx"])
    merged = row_df.merge(pred_sub, on=["source_file", "fault_id", "run_id", "sample_idx"], how="left")
    merged["split_group"] = out_split_group
    return merged


def _require_all_seed_artifacts(model_prefix: str) -> None:
    missing = []
    for seed in SEEDS:
        expected = [
            MODEL_DIR / f"{model_prefix}_model_seed{seed}.pt",
            MODEL_DIR / f"{model_prefix}_scaler_seed{seed}.pkl",
            MODEL_DIR / f"{model_prefix}_meta_seed{seed}.json",
        ]
        if not all(path.exists() for path in expected):
            missing.append(seed)
    if missing:
        raise FileNotFoundError(
            f"Missing {model_prefix} artifacts for seeds {missing}. "
            "Run `python -m src.models.train_sota_models` first so the full configured seed list is available."
        )


def _predict_windows(
    df_win: pd.DataFrame,
    model_prefix: str,
    output_prefix: str,
    inference_kind: str,
    infer_batch_size: int,
    tta_cfg: dict | None = None,
) -> pd.DataFrame:
    device = torch_device_info(prefer_mps=True)["selected_device"]
    out = _window_key_frame(df_win)
    avg_cols = []
    for seed in SEEDS:
        model, imputer, scaler, _ = load_temporal_artifact(MODEL_DIR, model_prefix=model_prefix, seed=seed, device=device)
        Xs = transform_windows(df_win, imputer=imputer, scaler=scaler)
        if inference_kind == "tta":
            if tta_cfg is None:
                raise ValueError("tta_cfg is required for TTA inference.")
            probs = infer_tta_online_probs(
                model,
                Xs,
                device=device,
                infer_batch_size=infer_batch_size,
                lr=float(tta_cfg.get("adaptation_lr", 1e-4)),
                steps=int(tta_cfg.get("adaptation_steps", 1)),
            )
        else:
            probs = infer_temporal_probs(model, Xs, device=device, infer_batch_size=infer_batch_size)
        col = f"p_{output_prefix}_seed{seed}"
        out[col] = probs
        avg_cols.append(col)
    out[f"p_{output_prefix}"] = out[avg_cols].mean(axis=1)
    return out


def main() -> None:
    ensure_dir(PRED_DIR)
    tta_cfg = read_yaml(CONFIG_DIR / "train_adaptable.yaml", default={})
    inv_cfg = read_yaml(CONFIG_DIR / "train_invariant.yaml", default={})
    infer_batch_size = int(max(tta_cfg.get("inference_batch_size", 512), inv_cfg.get("inference_batch_size", 512)))
    _require_all_seed_artifacts("adaptable_tcn")
    _require_all_seed_artifacts("invariant_tcn")

    base_val = read_csv(PRED_DIR / "base_val_predictions.csv")
    base_main = read_csv(PRED_DIR / "base_test_main_predictions.csv")
    base_cost = read_csv(PRED_DIR / "base_test_cost_predictions.csv")

    val_win = _load_windows("te_val_windows.csv")
    full_test_win = _load_windows("te_test_full_windows_tcn.csv")

    adaptable_val = _predict_windows(val_win, model_prefix="adaptable_tcn", output_prefix="adaptable", inference_kind="tta", infer_batch_size=infer_batch_size, tta_cfg=tta_cfg)
    invariant_val = _predict_windows(val_win, model_prefix="invariant_tcn", output_prefix="invariant", inference_kind="standard", infer_batch_size=infer_batch_size)
    val_df = base_val[KEY_COLS + [c for c in base_val.columns if c.startswith("p_tcn_seed")] + ["p_tcn"]].copy()
    val_df = val_df.merge(
        adaptable_val[["source_file", "fault_id", "run_id", "sample_idx"] + [c for c in adaptable_val.columns if c.startswith("p_adaptable")]],
        on=["source_file", "fault_id", "run_id", "sample_idx"],
        how="left",
    )
    val_df = val_df.merge(
        invariant_val[["source_file", "fault_id", "run_id", "sample_idx"] + [c for c in invariant_val.columns if c.startswith("p_invariant")]],
        on=["source_file", "fault_id", "run_id", "sample_idx"],
        how="left",
    )
    write_csv(PRED_DIR / "sota_val_predictions.csv", val_df)

    adaptable_full = _predict_windows(full_test_win, model_prefix="adaptable_tcn", output_prefix="adaptable", inference_kind="tta", infer_batch_size=infer_batch_size, tta_cfg=tta_cfg)
    invariant_full = _predict_windows(full_test_win, model_prefix="invariant_tcn", output_prefix="invariant", inference_kind="standard", infer_batch_size=infer_batch_size)

    adaptable_main = _align_temporal_to_rows(base_main[KEY_COLS].copy(), adaptable_full, out_split_group="test_main", prefix="adaptable")
    invariant_main = _align_temporal_to_rows(base_main[KEY_COLS].copy(), invariant_full, out_split_group="test_main", prefix="invariant")
    main_df = base_main[KEY_COLS + [c for c in base_main.columns if c.startswith("p_tcn_seed")] + ["p_tcn"]].copy()
    main_df = main_df.merge(
        adaptable_main[["source_file", "fault_id", "run_id", "sample_idx"] + [c for c in adaptable_main.columns if c.startswith("p_adaptable")]],
        on=["source_file", "fault_id", "run_id", "sample_idx"],
        how="left",
    )
    main_df = main_df.merge(
        invariant_main[["source_file", "fault_id", "run_id", "sample_idx"] + [c for c in invariant_main.columns if c.startswith("p_invariant")]],
        on=["source_file", "fault_id", "run_id", "sample_idx"],
        how="left",
    )
    write_csv(PRED_DIR / "sota_test_main_predictions.csv", main_df)

    adaptable_cost = _align_temporal_to_rows(base_cost[KEY_COLS].copy(), adaptable_full, out_split_group="test_cost", prefix="adaptable")
    invariant_cost = _align_temporal_to_rows(base_cost[KEY_COLS].copy(), invariant_full, out_split_group="test_cost", prefix="invariant")
    cost_df = base_cost[KEY_COLS + [c for c in base_cost.columns if c.startswith("p_tcn_seed")] + ["p_tcn"]].copy()
    cost_df = cost_df.merge(
        adaptable_cost[["source_file", "fault_id", "run_id", "sample_idx"] + [c for c in adaptable_cost.columns if c.startswith("p_adaptable")]],
        on=["source_file", "fault_id", "run_id", "sample_idx"],
        how="left",
    )
    cost_df = cost_df.merge(
        invariant_cost[["source_file", "fault_id", "run_id", "sample_idx"] + [c for c in invariant_cost.columns if c.startswith("p_invariant")]],
        on=["source_file", "fault_id", "run_id", "sample_idx"],
        how="left",
    )
    write_csv(PRED_DIR / "sota_test_cost_predictions.csv", cost_df)
    print("Saved SOTA baseline predictions to outputs/predictions")


if __name__ == "__main__":
    main()
