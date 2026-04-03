from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.experiment import get_seed_list

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PRED_DIR = OUTPUT_DIR / "predictions"
EVAL_DIR = OUTPUT_DIR / "evaluation"

SEEDS = get_seed_list()
MODEL_PREFIXES = ["rf", "xgb", "tcn"]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_prediction_csv(filename: str) -> pd.DataFrame:
    path = PRED_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")
    return pd.read_csv(path)


def compute_binary_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs).astype(float)

    # NaN 제외 전 coverage 계산
    coverage = float(np.isfinite(probs).mean())

    mask = np.isfinite(probs)
    y_true = y_true[mask]
    probs = probs[mask]

    if len(y_true) == 0:
        return {
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "positive_rate_pred": np.nan,
            "positive_rate_true": np.nan,
            "n_samples": 0,
            "threshold": float(threshold),
            "auc": np.nan,
            "coverage": coverage,
        }

    preds = (probs >= threshold).astype(int)

    out = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "positive_rate_pred": float(preds.mean()),
        "positive_rate_true": float(y_true.mean()),
        "n_samples": int(len(y_true)),
        "threshold": float(threshold),
        "coverage": coverage,
    }

    try:
        if len(np.unique(y_true)) < 2:
            out["auc"] = np.nan
        else:
            out["auc"] = float(roc_auc_score(y_true, probs))
    except Exception:
        out["auc"] = np.nan

    return out


def find_best_threshold_by_f1(
    y_true: np.ndarray,
    probs: np.ndarray,
    grid: np.ndarray | None = None,
) -> Tuple[float, Dict[str, float]]:
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs).astype(float)

    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)

    best_threshold = 0.5
    best_metrics = compute_binary_metrics(y_true, probs, threshold=0.5)
    best_f1 = best_metrics["f1"]
    best_recall = best_metrics["recall"]
    best_precision = best_metrics["precision"]

    for th in grid:
        m = compute_binary_metrics(y_true, probs, threshold=float(th))

        # 1순위: F1 최대
        # 동률이면 recall 큰 것
        # 또 동률이면 precision 큰 것
        improved = False
        if m["f1"] > best_f1:
            improved = True
        elif np.isclose(m["f1"], best_f1):
            if m["recall"] > best_recall:
                improved = True
            elif np.isclose(m["recall"], best_recall) and m["precision"] > best_precision:
                improved = True

        if improved:
            best_threshold = float(th)
            best_metrics = m
            best_f1 = m["f1"]
            best_recall = m["recall"]
            best_precision = m["precision"]

    return best_threshold, best_metrics


def summarize_seedwise(seed_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["accuracy", "precision", "recall", "f1", "auc", "positive_rate_pred", "threshold", "coverage"]
    rows = []
    for (split_name, model_name), g in seed_df.groupby(["split", "model"], dropna=False):
        row = {
            "split": split_name,
            "model": model_name,
            "n_seeds": int(len(g)),
            "positive_rate_true": float(g["positive_rate_true"].iloc[0]),
            "n_samples": int(g["n_samples"].iloc[0]),
        }
        for m in metric_cols:
            row[f"{m}_mean"] = float(g[m].mean())
            row[f"{m}_std"] = float(g[m].std(ddof=1)) if len(g) > 1 else 0.0
        rows.append(row)

    return pd.DataFrame(rows)


def format_mean_std_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in summary_df.iterrows():
        rows.append(
            {
                "split": r["split"],
                "model": r["model"],
                "Threshold": f"{r['threshold_mean']:.4f} ± {r['threshold_std']:.4f}",
                "Accuracy": f"{r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f}",
                "Precision": f"{r['precision_mean']:.4f} ± {r['precision_std']:.4f}",
                "Recall": f"{r['recall_mean']:.4f} ± {r['recall_std']:.4f}",
                "F1": f"{r['f1_mean']:.4f} ± {r['f1_std']:.4f}",
                "AUC": f"{r['auc_mean']:.4f} ± {r['auc_std']:.4f}" if pd.notna(r["auc_mean"]) else "NaN",
                "n_samples": int(r["n_samples"]),
                "positive_rate_true": float(r["positive_rate_true"]),
                "Coverage": f"{r['coverage_mean']:.4f} ± {r['coverage_std']:.4f}",
            }
        )
    return pd.DataFrame(rows)


def collect_best_thresholds(val_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    y_val = val_df["y_true"].to_numpy().astype(int)

    rows = []
    threshold_dict: Dict[str, Dict[str, float]] = {}

    for model_prefix in MODEL_PREFIXES:
        model_name = model_prefix.upper()
        threshold_dict[model_name] = {}

        for seed in SEEDS:
            col = f"p_{model_prefix}_seed{seed}"
            if col not in val_df.columns:
                continue

            best_th, best_metrics = find_best_threshold_by_f1(y_val, val_df[col].to_numpy())
            threshold_dict[model_name][f"seed{seed}"] = float(best_th)

            rows.append(
                {
                    "split": "val",
                    "model": model_name,
                    "seed": seed,
                    **best_metrics,
                }
            )

    return pd.DataFrame(rows), threshold_dict


def evaluate_on_split_with_thresholds(
    df: pd.DataFrame,
    split_name: str,
    threshold_dict: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    y_true = df["y_true"].to_numpy().astype(int)
    rows = []

    for model_prefix in MODEL_PREFIXES:
        model_name = model_prefix.upper()

        for seed in SEEDS:
            col = f"p_{model_prefix}_seed{seed}"
            if col not in df.columns:
                continue

            threshold = threshold_dict[model_name][f"seed{seed}"]
            metrics = compute_binary_metrics(y_true, df[col].to_numpy(), threshold=threshold)

            rows.append(
                {
                    "split": split_name,
                    "model": model_name,
                    "seed": seed,
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def evaluate_avgprob_with_avg_thresholds(
    df: pd.DataFrame,
    split_name: str,
    threshold_dict: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    y_true = df["y_true"].to_numpy().astype(int)

    rows = []

    avg_thresholds = {
        "RF_avgprob": float(np.mean([threshold_dict["RF"][f"seed{s}"] for s in SEEDS])),
        "XGB_avgprob": float(np.mean([threshold_dict["XGB"][f"seed{s}"] for s in SEEDS])),
        "TCN_avgprob": float(np.mean([threshold_dict["TCN"][f"seed{s}"] for s in SEEDS])),
    }

    avg_cols = {
        "RF_avgprob": "p_rf",
        "XGB_avgprob": "p_xgb",
        "TCN_avgprob": "p_tcn",
        "Ensemble_avgprob": "p_ensemble",
    }

    ensemble_threshold = float(np.mean([
        avg_thresholds["RF_avgprob"],
        avg_thresholds["XGB_avgprob"],
        avg_thresholds["TCN_avgprob"],
    ]))

    for model_name, col in avg_cols.items():
        if col not in df.columns:
            continue

        if model_name == "Ensemble_avgprob":
            threshold = ensemble_threshold
        else:
            threshold = avg_thresholds[model_name]

        metrics = compute_binary_metrics(y_true, df[col].to_numpy(), threshold=threshold)
        rows.append(
            {
                "split": split_name,
                "model": model_name,
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    ensure_dir(EVAL_DIR)

    val_df = load_prediction_csv("base_val_predictions.csv")
    test_main_df = load_prediction_csv("base_test_main_predictions.csv")
    test_cost_df = load_prediction_csv("base_test_cost_predictions.csv")

    # 1) val 기준 seed별 best threshold 찾기
    val_best_df, threshold_dict = collect_best_thresholds(val_df)

    with open(EVAL_DIR / "best_thresholds_by_seed.json", "w", encoding="utf-8") as f:
        json.dump(threshold_dict, f, indent=2, ensure_ascii=False)

    val_best_df.to_csv(EVAL_DIR / "val_best_threshold_metrics.csv", index=False)

    # 2) 그 threshold를 test_main / test_cost에 적용
    test_main_seed_df = evaluate_on_split_with_thresholds(
        test_main_df,
        split_name="test_main",
        threshold_dict=threshold_dict,
    )
    test_cost_seed_df = evaluate_on_split_with_thresholds(
        test_cost_df,
        split_name="test_cost",
        threshold_dict=threshold_dict,
    )

    # 3) seed별 결과 저장
    test_main_seed_df.to_csv(EVAL_DIR / "test_main_seed_metrics_thresholded.csv", index=False)
    test_cost_seed_df.to_csv(EVAL_DIR / "test_cost_seed_metrics_thresholded.csv", index=False)

    all_seed_df = pd.concat([val_best_df, test_main_seed_df, test_cost_seed_df], ignore_index=True)
    all_seed_df.to_csv(EVAL_DIR / "all_seed_metrics_thresholded.csv", index=False)

    # 4) mean ± std 요약
    summary_df = summarize_seedwise(all_seed_df)
    summary_df.to_csv(EVAL_DIR / "all_seed_summary_thresholded.csv", index=False)

    mean_std_df = format_mean_std_table(summary_df)
    mean_std_df.to_csv(EVAL_DIR / "all_mean_std_table_thresholded.csv", index=False)

    # split별도 저장
    for split_name in ["val", "test_main", "test_cost"]:
        sub = summary_df[summary_df["split"] == split_name].copy()
        if len(sub):
            sub.to_csv(EVAL_DIR / f"{split_name}_seed_summary_thresholded.csv", index=False)

        sub2 = mean_std_df[mean_std_df["split"] == split_name].copy()
        if len(sub2):
            sub2.to_csv(EVAL_DIR / f"{split_name}_mean_std_table_thresholded.csv", index=False)

    # 5) 평균확률(p_rf, p_xgb, p_tcn, p_ensemble) 기준 성능
    val_avgprob_df = evaluate_avgprob_with_avg_thresholds(val_df, "val", threshold_dict)
    test_main_avgprob_df = evaluate_avgprob_with_avg_thresholds(test_main_df, "test_main", threshold_dict)
    test_cost_avgprob_df = evaluate_avgprob_with_avg_thresholds(test_cost_df, "test_cost", threshold_dict)

    all_avgprob_df = pd.concat(
        [val_avgprob_df, test_main_avgprob_df, test_cost_avgprob_df],
        ignore_index=True,
    )
    all_avgprob_df.to_csv(EVAL_DIR / "all_avgprob_metrics_thresholded.csv", index=False)

    # 6) manifest
    manifest = {
        "selection_rule": "best threshold on validation by maximum F1",
        "seeds": SEEDS,
        "models": ["RF", "XGB", "TCN"],
        "files_written": [
            "best_thresholds_by_seed.json",
            "val_best_threshold_metrics.csv",
            "test_main_seed_metrics_thresholded.csv",
            "test_cost_seed_metrics_thresholded.csv",
            "all_seed_metrics_thresholded.csv",
            "all_seed_summary_thresholded.csv",
            "all_mean_std_table_thresholded.csv",
            "all_avgprob_metrics_thresholded.csv",
        ],
    }
    with open(EVAL_DIR / "evaluation_manifest_thresholded.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("Saved thresholded evaluation summaries to:", EVAL_DIR)
    print("- best_thresholds_by_seed.json")
    print("- all_seed_metrics_thresholded.csv")
    print("- all_seed_summary_thresholded.csv")
    print("- all_mean_std_table_thresholded.csv")
    print("- all_avgprob_metrics_thresholded.csv")


if __name__ == "__main__":
    main()
