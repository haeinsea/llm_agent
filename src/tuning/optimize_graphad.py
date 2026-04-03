from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.models.graphad import fit_graphad, infer_graphad
from src.tuning.common import (
    best_threshold,
    load_feature_cols,
    param_product,
    read_rows,
    read_search_cfg,
    safe_jsonable,
    weighted_objective,
    write_search_outputs,
)


GROUP_COLS = ["source_file", "fault_id", "run_id"]


def _parse_sensor_list(value: object) -> set[str]:
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return {str(item) for item in parsed}
        except Exception:
            pass
    return set()


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 1.0
    return float(len(left & right) / len(union))


def _best_f1(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    tau, f1 = best_threshold(y_true, probs)
    return float(tau), float(f1)


def _run_consistency(eval_df: pd.DataFrame, pred_df: pd.DataFrame) -> float:
    merged = pd.concat(
        [
            eval_df[GROUP_COLS + ["sample_idx", "y", "phase"]].reset_index(drop=True),
            pred_df[["graphad_top1_sensor", "graphad_topk_sensors"]].reset_index(drop=True),
        ],
        axis=1,
    )
    if "phase" in merged.columns:
        merged = merged[merged["phase"].isin(["transition", "post_shift"])]
    merged = merged[merged["y"] == 1].copy()
    if merged.empty:
        return 0.0

    scores = []
    for _, group in merged.groupby(GROUP_COLS, sort=False):
        group = group.sort_values("sample_idx")
        if len(group) < 2:
            continue
        top1 = group["graphad_top1_sensor"].astype(str).tolist()
        topk = [_parse_sensor_list(v) for v in group["graphad_topk_sensors"]]
        top1_same = [float(left == right) for left, right in zip(top1[:-1], top1[1:])]
        topk_same = [_jaccard(left, right) for left, right in zip(topk[:-1], topk[1:])]
        scores.append(0.5 * float(np.mean(top1_same)) + 0.5 * float(np.mean(topk_same)))
    return float(np.mean(scores)) if scores else 0.0


def _noise_consistency(
    eval_df: pd.DataFrame,
    feature_cols: list[str],
    artifact: dict,
    *,
    bootstrap_repeats: int,
    noise_std: float,
    eval_samples: int,
) -> float:
    candidate = eval_df.copy()
    if "phase" in candidate.columns:
        candidate = candidate[candidate["phase"].isin(["transition", "post_shift"])]
    candidate = candidate[candidate["y"] == 1].copy()
    if candidate.empty:
        candidate = eval_df.sample(min(len(eval_df), eval_samples), random_state=42)
    if len(candidate) > eval_samples:
        candidate = candidate.sample(eval_samples, random_state=42)
    if candidate.empty:
        return 0.0

    base_pred = infer_graphad(candidate, artifact)
    scale = np.asarray([float(artifact["scale"][col]) for col in feature_cols], dtype=float)
    scale = np.where(scale > 0.0, scale, 1.0)
    rng = np.random.default_rng(42)
    scores = []

    for _ in range(int(bootstrap_repeats)):
        noisy = candidate.copy()
        noise = rng.normal(0.0, float(noise_std), size=(len(noisy), len(feature_cols))) * scale
        noisy.loc[:, feature_cols] = noisy[feature_cols].to_numpy(dtype=float) + noise
        perturbed = infer_graphad(noisy, artifact)

        top1_same = np.mean(
            base_pred["graphad_top1_sensor"].astype(str).to_numpy()
            == perturbed["graphad_top1_sensor"].astype(str).to_numpy()
        )
        topk_same = np.mean(
            [
                _jaccard(_parse_sensor_list(left), _parse_sensor_list(right))
                for left, right in zip(base_pred["graphad_topk_sensors"], perturbed["graphad_topk_sensors"])
            ]
        )
        base_score = base_pred["graphad_score"].to_numpy(dtype=float)
        pert_score = perturbed["graphad_score"].to_numpy(dtype=float)
        if len(base_score) > 1 and np.nanstd(base_score) > 0 and np.nanstd(pert_score) > 0:
            corr = float(np.corrcoef(base_score, pert_score)[0, 1])
            corr = float(np.clip(np.nan_to_num(corr, nan=0.0), -1.0, 1.0))
            corr = 0.5 * (corr + 1.0)
        else:
            corr = 0.0
        scores.append((float(top1_same) + float(topk_same) + corr) / 3.0)
    return float(np.mean(scores)) if scores else 0.0


def _normalize_lambdas(params: dict) -> dict:
    out = deepcopy(params)
    weights = np.asarray(
        [
            float(out.get("lambda_z", 0.4)),
            float(out.get("lambda_tr", 0.3)),
            float(out.get("lambda_fl", 0.3)),
        ],
        dtype=float,
    )
    total = float(weights.sum())
    if total <= 0:
        weights = np.asarray([0.4, 0.3, 0.3], dtype=float)
        total = float(weights.sum())
    weights = weights / total
    out["lambda_z"] = float(weights[0])
    out["lambda_tr"] = float(weights[1])
    out["lambda_fl"] = float(weights[2])
    return out


def main() -> None:
    print("\n" + "=" * 80, flush=True)
    print("[START] optimize_graphad", flush=True)
    print("  objective : AUC + F1 + run_consistency + noise_consistency + top1_gap", flush=True)
    print("  config    : configs/search_graphad.yaml", flush=True)
    print("=" * 80, flush=True)
    search = read_search_cfg("search_graphad.yaml")
    feature_cols = load_feature_cols()
    train_df = read_rows("te_train_rows.csv")
    val_df = read_rows("te_val_rows.csv")

    weights = search.get(
        "score_weights",
        {
            "auc": 0.40,
            "f1": 0.20,
            "run_consistency": 0.20,
            "noise_consistency": 0.15,
            "top1_gap": 0.05,
        },
    )
    bootstrap_repeats = int(search.get("bootstrap_repeats", 5))
    noise_std = float(search.get("noise_std", 0.01))
    eval_samples = int(search.get("eval_samples", 32))
    trials = []

    exclude = {"selection_metric", "bootstrap_repeats", "noise_std", "eval_samples", "score_weights"}
    trial_space = param_product(search, exclude_keys=exclude)
    print(f"[optimize_graphad] train_rows={len(train_df):,} val_rows={len(val_df):,} trials={len(trial_space):,}", flush=True)
    for idx, params in enumerate(trial_space, start=1):
        print(f"[optimize_graphad] trial {idx}/{len(trial_space)} params={params}", flush=True)
        cfg = _normalize_lambdas(params)
        artifact = fit_graphad(train_df=train_df, feature_cols=feature_cols, cfg=cfg)
        pred = infer_graphad(val_df, artifact)

        y_true = val_df["y"].to_numpy(dtype=int)
        score = pred["graphad_score"].to_numpy(dtype=float)
        tau, f1 = _best_f1(y_true, score)
        try:
            auc = float(roc_auc_score(y_true, score))
        except Exception:
            auc = float("nan")

        anomaly_mask = y_true == 1
        normal_mask = y_true == 0
        top1_gap_anom = float(pred.loc[anomaly_mask, "graphad_top1_gap"].mean()) if anomaly_mask.any() else 0.0
        top1_gap_norm = float(pred.loc[normal_mask, "graphad_top1_gap"].mean()) if normal_mask.any() else 0.0
        top1_gap = top1_gap_anom - top1_gap_norm

        run_consistency = _run_consistency(val_df, pred)
        noise_consistency = _noise_consistency(
            val_df,
            feature_cols,
            artifact,
            bootstrap_repeats=bootstrap_repeats,
            noise_std=noise_std,
            eval_samples=eval_samples,
        )

        row = deepcopy(cfg)
        row["threshold"] = float(tau)
        row["f1_mean"] = float(f1)
        row["f1_std"] = 0.0
        row["auc_mean"] = float(auc)
        row["auc_std"] = 0.0
        row["run_consistency_mean"] = float(run_consistency)
        row["run_consistency_std"] = 0.0
        row["noise_consistency_mean"] = float(noise_consistency)
        row["noise_consistency_std"] = 0.0
        row["top1_gap_mean"] = float(top1_gap)
        row["top1_gap_std"] = 0.0
        row["objective"] = weighted_objective(
            {
                "auc": np.nan_to_num(row["auc_mean"], nan=0.0),
                "f1": row["f1_mean"],
                "run_consistency": row["run_consistency_mean"],
                "noise_consistency": row["noise_consistency_mean"],
                "top1_gap": row["top1_gap_mean"],
            },
            weights,
        )
        trials.append(row)

    trials_df = pd.DataFrame(trials).sort_values(
        ["objective", "auc_mean", "run_consistency_mean", "noise_consistency_mean"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    best_row = {key: safe_jsonable(value) for key, value in trials_df.iloc[0].to_dict().items()}
    best_row["best_params"] = {
        key: best_row[key]
        for key in ["corr_threshold", "alpha", "top_k", "lambda_z", "lambda_tr", "lambda_fl"]
        if key in best_row
    }
    write_search_outputs("graphad", trials_df, best_row)
    print("[DONE] optimize_graphad", flush=True)
    print(trials_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
