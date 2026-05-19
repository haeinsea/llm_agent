from __future__ import annotations

import argparse
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.temporal_backbone import build_temporal_model
from src.models.temporal_sota import infer_temporal_probs
from src.models.train_tcn import (
    best_threshold_by_f1,
    flattened_to_tensor,
    phase_recall,
    read_windows,
    set_seed,
    transform_with_imputer_scaler,
)
from src.tuning.common import (
    build_trial_space,
    maybe_sample_frame,
    probability_focus_metrics,
    read_search_cfg,
    safe_jsonable,
    weighted_objective,
    write_search_outputs,
)
from src.utils.device import torch_device_info


def _train_single_trial(
    train_df,
    val_df,
    params: dict,
    seed: int,
    *,
    entropy_floor: float,
    gray_margin: float,
) -> dict[str, float]:
    set_seed(seed)
    X_train = flattened_to_tensor(train_df)
    y_train = train_df["y"].to_numpy(dtype=np.float32)
    X_val = flattened_to_tensor(val_df)
    y_val = val_df["y"].to_numpy(dtype=np.float32)
    val_phases = val_df["phase"].to_numpy()
    X_train, X_val, _, _ = transform_with_imputer_scaler(X_train, X_val)

    device = torch_device_info(prefer_mps=True)["selected_device"]
    batch_size = int(params.get("batch_size", 64))
    infer_batch_size = int(params.get("inference_batch_size", 2048))
    epochs = int(params.get("epochs", 25))
    lr = float(params.get("lr", 5e-4))
    weight_decay = float(params.get("weight_decay", 1e-4))

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    n_pos = float((y_train == 1).sum())
    n_neg = float((y_train == 0).sum())
    raw_pos_weight = n_neg / max(n_pos, 1.0)
    pos_weight = torch.tensor([max(1.0, raw_pos_weight)], dtype=torch.float32).to(device)

    model = build_temporal_model(
        n_features=X_train.shape[1],
        cfg={
            "architecture": "modern_tcn",
            "channels": tuple(params.get("channels", [64, 96, 128])),
            "dilations": tuple(params.get("dilations", [1, 2, 4])),
            "kernel_size": int(params.get("kernel_size", 3)),
            "dropout": float(params.get("dropout", 0.1)),
            "expansion_ratio": int(params.get("expansion_ratio", 2)),
            "pool": str(params.get("pool", "avg")),
        },
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_f1 = -np.inf
    best_recall = -np.inf
    best_post_shift_recall = -np.inf
    best_metrics: dict[str, float] | None = None
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            probs = infer_temporal_probs(model, X_val, device=device, infer_batch_size=infer_batch_size)
        threshold, metrics = best_threshold_by_f1(y_val, probs)
        post_shift_recall = phase_recall(y_val, probs, val_phases, "post_shift", threshold)
        transition_recall = phase_recall(y_val, probs, val_phases, "transition", threshold)
        selection_score = float(metrics["f1"]) + 0.10 * float(metrics["recall"]) + 0.35 * float(post_shift_recall)
        best_selection_score = best_f1 + 0.10 * best_recall + 0.35 * best_post_shift_recall
        if selection_score > best_selection_score or (selection_score == best_selection_score and metrics["f1"] > best_f1):
            best_f1 = float(metrics["f1"])
            best_recall = float(metrics["recall"])
            best_post_shift_recall = float(post_shift_recall)
            focus_metrics = probability_focus_metrics(
                y_val,
                probs,
                tau=threshold,
                entropy_floor=entropy_floor,
                gray_margin=gray_margin,
            )
            best_metrics = {
                "f1": float(metrics["f1"]),
                "recall": float(metrics["recall"]),
                "auc": float(metrics["auc"]) if metrics["auc"] is not None else float("nan"),
                "post_shift_recall": float(post_shift_recall),
                "transition_recall": float(transition_recall),
                "threshold": float(threshold),
                "high_entropy_recall": float(focus_metrics["high_entropy_recall"]),
                "grayzone_recall": float(focus_metrics["grayzone_recall"]),
                "mean_entropy": float(focus_metrics["mean_entropy"]),
                "grayzone_share": float(focus_metrics["grayzone_share"]),
            }

    assert best_metrics is not None
    return best_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="search_tcn.yaml", help="Search config file name under configs/.")
    parser.add_argument("--output-prefix", default="tcn", help="Prefix for saved trial/best files.")
    parser.add_argument("--output-dir", default=None, help="Optional alternative output directory.")
    return parser.parse_args()


def run_search(*, config_name: str = "search_tcn.yaml", output_prefix: str = "tcn", output_dir: str | None = None) -> tuple[pd.DataFrame, dict]:
    print("\n" + "=" * 80, flush=True)
    print("[START] optimize_modern_tcn", flush=True)
    print("  objective : weighted validation objective", flush=True)
    print(f"  config    : configs/{config_name}", flush=True)
    print("=" * 80, flush=True)
    search = read_search_cfg(config_name)
    train_df = read_windows("te_train_windows.csv")
    val_df = read_windows("te_val_windows.csv")
    train_df = maybe_sample_frame(
        train_df,
        frac=search.get("train_sample_frac"),
        n_rows=search.get("train_sample_n"),
        seed=int(search.get("train_sample_seed", 42)),
    )
    seeds = [int(seed) for seed in search.get("search_seeds", [0])]
    weights = search.get("score_weights", {"f1": 1.0, "post_shift_recall": 0.35, "recall": 0.10, "auc": 0.10})
    entropy_floor = float(search.get("entropy_floor", 0.9))
    gray_margin = float(search.get("gray_margin", 0.1))
    trials = []
    trial_space = build_trial_space(search, exclude_keys={"search_seeds", "selection_metric", "score_weights"})
    print(f"[optimize_modern_tcn] train_windows={len(train_df):,} val_windows={len(val_df):,} seeds={seeds} trials={len(trial_space):,}", flush=True)
    print(f"[optimize_modern_tcn] score_weights={weights}", flush=True)

    for idx, params in enumerate(trial_space, start=1):
        print(f"[optimize_modern_tcn] trial {idx}/{len(trial_space)} params={params}", flush=True)
        seed_rows = []
        for seed in seeds:
            seed_rows.append(
                _train_single_trial(
                    train_df,
                    val_df,
                    params=params,
                    seed=seed,
                    entropy_floor=entropy_floor,
                    gray_margin=gray_margin,
                )
            )
        row = deepcopy(params)
        metric_keys = [
            "f1",
            "recall",
            "auc",
            "post_shift_recall",
            "transition_recall",
            "threshold",
            "high_entropy_recall",
            "grayzone_recall",
            "mean_entropy",
            "grayzone_share",
        ]
        for key in metric_keys:
            vals = np.asarray([seed_row[key] for seed_row in seed_rows], dtype=float)
            row[f"{key}_mean"] = float(np.nanmean(vals))
            row[f"{key}_std"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
        row["objective"] = weighted_objective(
            {metric: row.get(f"{metric}_mean", 0.0) for metric in weights},
            weights,
        )
        trials.append(row)

    trials_df = pd.DataFrame(trials).sort_values(
        ["objective", "auc_mean", "post_shift_recall_mean", "f1_mean"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    best_row = {key: safe_jsonable(value) for key, value in trials_df.iloc[0].to_dict().items()}
    best_row["best_params"] = {
        key: best_row[key]
        for key in ["channels", "dilations", "kernel_size", "dropout", "expansion_ratio", "pool", "batch_size", "epochs", "lr", "weight_decay"]
        if key in best_row
    }
    best_row["score_weights"] = weights
    best_row["search_config"] = config_name
    write_search_outputs(output_prefix, trials_df, best_row, output_dir=output_dir)
    print("[DONE] optimize_modern_tcn", flush=True)
    print(trials_df.head(10).to_string(index=False))
    return trials_df, best_row


def main() -> None:
    args = parse_args()
    run_search(config_name=args.config, output_prefix=args.output_prefix, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
