from __future__ import annotations

import argparse
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from src.tuning.common import (
    best_threshold,
    build_trial_space,
    load_feature_cols,
    maybe_sample_frame,
    probability_focus_metrics,
    read_rows,
    read_search_cfg,
    safe_jsonable,
    weighted_objective,
    write_search_outputs,
)


def _trial_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    tau, best_f1 = best_threshold(y_true, probs)
    preds = (probs >= tau).astype(int)
    out = {
        "threshold": float(tau),
        "f1": float(best_f1),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
    }
    try:
        out["auc"] = float(roc_auc_score(y_true, probs))
    except Exception:
        out["auc"] = float("nan")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="search_rf.yaml", help="Search config file name under configs/.")
    parser.add_argument("--output-prefix", default="rf", help="Prefix for saved trial/best files.")
    parser.add_argument("--output-dir", default=None, help="Optional alternative output directory.")
    return parser.parse_args()


def run_search(*, config_name: str = "search_rf.yaml", output_prefix: str = "rf", output_dir: str | None = None) -> tuple[pd.DataFrame, dict]:
    print("\n" + "=" * 80, flush=True)
    print("[START] optimize_rf", flush=True)
    print("  objective : weighted validation objective", flush=True)
    print(f"  config    : configs/{config_name}", flush=True)
    print("=" * 80, flush=True)
    search = read_search_cfg(config_name)
    feature_cols = load_feature_cols()
    train_df = read_rows("te_train_rows.csv")
    val_df = read_rows("te_val_rows.csv")
    train_df = maybe_sample_frame(
        train_df,
        frac=search.get("train_sample_frac"),
        n_rows=search.get("train_sample_n"),
        seed=int(search.get("train_sample_seed", 42)),
    )

    X_train = train_df[feature_cols].to_numpy()
    y_train = train_df["y"].to_numpy().astype(int)
    X_val = val_df[feature_cols].to_numpy()
    y_val = val_df["y"].to_numpy().astype(int)

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)

    weights = search.get("score_weights", {"f1": 1.0, "recall": 0.2})
    entropy_floor = float(search.get("entropy_floor", 0.9))
    gray_margin = float(search.get("gray_margin", 0.1))
    seeds = [int(seed) for seed in search.get("search_seeds", [0])]
    trials = []
    trial_space = build_trial_space(search, exclude_keys={"search_seeds", "selection_metric", "score_weights"})
    print(f"[optimize_rf] train_rows={len(train_df):,} val_rows={len(val_df):,} seeds={seeds} trials={len(trial_space):,}", flush=True)
    print(f"[optimize_rf] score_weights={weights}", flush=True)

    for idx, params in enumerate(trial_space, start=1):
        print(f"[optimize_rf] trial {idx}/{len(trial_space)} params={params}", flush=True)
        seed_rows = []
        for seed in seeds:
            model = RandomForestClassifier(
                n_estimators=int(params.get("n_estimators", 300)),
                max_depth=params.get("max_depth"),
                min_samples_split=int(params.get("min_samples_split", 2)),
                min_samples_leaf=int(params.get("min_samples_leaf", 1)),
                max_features=params.get("max_features", "sqrt"),
                bootstrap=bool(params.get("bootstrap", True)),
                random_state=seed,
                n_jobs=-1,
                class_weight="balanced",
            )
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_val)[:, 1]
            row_metrics = _trial_metrics(y_val, probs)
            row_metrics.update(
                probability_focus_metrics(
                    y_val,
                    probs,
                    tau=row_metrics["threshold"],
                    entropy_floor=entropy_floor,
                    gray_margin=gray_margin,
                )
            )
            seed_rows.append(row_metrics)

        row = deepcopy(params)
        metric_keys = [
            "threshold",
            "f1",
            "recall",
            "precision",
            "auc",
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
        ["objective", "auc_mean", "f1_mean", "recall_mean"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    best_row = {key: safe_jsonable(value) for key, value in trials_df.iloc[0].to_dict().items()}
    best_row["best_params"] = {
        key: best_row[key]
        for key in ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_features", "bootstrap"]
        if key in best_row
    }
    best_row["score_weights"] = weights
    best_row["search_config"] = config_name
    write_search_outputs(output_prefix, trials_df, best_row, output_dir=output_dir)
    print("[DONE] optimize_rf", flush=True)
    print(trials_df.head(10).to_string(index=False))
    return trials_df, best_row


def main() -> None:
    args = parse_args()
    run_search(config_name=args.config, output_prefix=args.output_prefix, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
