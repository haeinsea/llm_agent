from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

from src.tuning.common import (
    load_feature_cols,
    maybe_sample_frame,
    probability_focus_metrics,
    read_rows,
    read_search_cfg,
    safe_jsonable,
    weighted_objective,
    write_search_outputs,
)
from src.tuning.optimize_adaptable import _trial_metrics as adaptable_trial_metrics
from src.tuning.optimize_invariant import _trial_metrics as invariant_trial_metrics
from src.tuning.optimize_modern_tcn import _train_single_trial as tcn_trial_metrics
from src.tuning.optimize_rf import _trial_metrics as rf_trial_metrics
from src.tuning.optimize_xgb import _trial_metrics as xgb_trial_metrics

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["rf", "xgb", "tcn", "adaptable", "invariant"])
    parser.add_argument("--screening-trials", required=True)
    parser.add_argument("--config", required=True, help="Search config file name under configs/.")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--seeds", default="0,1,2")
    return parser.parse_args()


def _parse_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return value
        try:
            return ast.literal_eval(stripped)
        except Exception:
            return value
    if isinstance(value, np.generic):
        return value.item()
    return value


def _normalize_params(model: str, row: dict[str, Any]) -> dict[str, Any]:
    params = {key: _parse_scalar(value) for key, value in row.items()}
    drop_keys = [key for key in params if key.endswith("_mean") or key.endswith("_std") or key == "objective"]
    for key in drop_keys:
        params.pop(key, None)

    if model == "rf":
        for key in ["n_estimators", "min_samples_split", "min_samples_leaf"]:
            params[key] = int(params[key])
        if params.get("max_depth") is not None:
            params["max_depth"] = int(params["max_depth"])
        params["bootstrap"] = bool(params["bootstrap"])
    elif model == "xgb":
        for key in ["n_estimators", "max_depth"]:
            params[key] = int(params[key])
        for key in [
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "min_child_weight",
            "gamma",
            "reg_alpha",
            "reg_lambda",
        ]:
            params[key] = float(params[key])
    elif model == "tcn":
        for key in ["kernel_size", "expansion_ratio", "batch_size", "inference_batch_size", "epochs"]:
            params[key] = int(params[key])
        for key in ["dropout", "lr", "weight_decay"]:
            params[key] = float(params[key])
    elif model == "adaptable":
        for key in ["kernel_size", "expansion_ratio", "batch_size", "inference_batch_size", "epochs", "adaptation_steps"]:
            params[key] = int(params[key])
        for key in ["dropout", "lr", "weight_decay", "adaptation_lr"]:
            params[key] = float(params[key])
    elif model == "invariant":
        for key in ["kernel_size", "expansion_ratio", "batch_size", "inference_batch_size", "epochs"]:
            params[key] = int(params[key])
        for key in ["dropout", "lr", "weight_decay", "penalty_weight"]:
            params[key] = float(params[key])
    return params


def _rf_search(params_list: list[dict[str, Any]], search: dict[str, Any], seeds: list[int]) -> tuple[pd.DataFrame, dict[str, Any]]:
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
    trials = []

    for idx, params in enumerate(params_list, start=1):
        print(f"[reconfirm_rf] candidate {idx}/{len(params_list)} params={params}", flush=True)
        seed_rows = []
        for seed in seeds:
            model = RandomForestClassifier(
                n_estimators=int(params["n_estimators"]),
                max_depth=params["max_depth"],
                min_samples_split=int(params["min_samples_split"]),
                min_samples_leaf=int(params["min_samples_leaf"]),
                max_features=params["max_features"],
                bootstrap=bool(params["bootstrap"]),
                random_state=seed,
                n_jobs=-1,
                class_weight="balanced",
            )
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_val)[:, 1]
            row_metrics = rf_trial_metrics(y_val, probs)
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
        row["screening_rank"] = idx
        for key in ["threshold", "f1", "recall", "precision", "auc", "high_entropy_recall", "grayzone_recall", "mean_entropy", "grayzone_share"]:
            vals = np.asarray([seed_row[key] for seed_row in seed_rows], dtype=float)
            row[f"{key}_mean"] = float(np.nanmean(vals))
            row[f"{key}_std"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
        row["objective"] = weighted_objective({metric: row.get(f"{metric}_mean", 0.0) for metric in weights}, weights)
        trials.append(row)

    trials_df = pd.DataFrame(trials).sort_values(
        ["objective", "auc_mean", "f1_mean", "recall_mean"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    best_row = {key: safe_jsonable(value) for key, value in trials_df.iloc[0].to_dict().items()}
    best_row["best_params"] = {key: best_row[key] for key in ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_features", "bootstrap"]}
    best_row["score_weights"] = weights
    best_row["search_config"] = search.get("_config_name")
    return trials_df, best_row


def _xgb_search(params_list: list[dict[str, Any]], search: dict[str, Any], seeds: list[int]) -> tuple[pd.DataFrame, dict[str, Any]]:
    if XGBClassifier is None:
        raise ImportError("xgboost is required for reconfirm_topk.py")
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

    pos = max(int(y_train.sum()), 1)
    neg = max(int((1 - y_train).sum()), 1)
    scale_pos_weight = neg / pos
    weights = search.get("score_weights", {"f1": 1.0, "recall": 0.2})
    entropy_floor = float(search.get("entropy_floor", 0.9))
    gray_margin = float(search.get("gray_margin", 0.1))
    trials = []

    for idx, params in enumerate(params_list, start=1):
        print(f"[reconfirm_xgb] candidate {idx}/{len(params_list)} params={params}", flush=True)
        seed_rows = []
        for seed in seeds:
            model = XGBClassifier(
                n_estimators=int(params["n_estimators"]),
                max_depth=int(params["max_depth"]),
                learning_rate=float(params["learning_rate"]),
                subsample=float(params["subsample"]),
                colsample_bytree=float(params["colsample_bytree"]),
                min_child_weight=float(params["min_child_weight"]),
                gamma=float(params["gamma"]),
                reg_alpha=float(params["reg_alpha"]),
                reg_lambda=float(params["reg_lambda"]),
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=seed,
                scale_pos_weight=scale_pos_weight,
                n_jobs=4,
            )
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_val)[:, 1]
            row_metrics = xgb_trial_metrics(y_val, probs)
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
        row["screening_rank"] = idx
        for key in ["threshold", "f1", "recall", "precision", "auc", "high_entropy_recall", "grayzone_recall", "mean_entropy", "grayzone_share"]:
            vals = np.asarray([seed_row[key] for seed_row in seed_rows], dtype=float)
            row[f"{key}_mean"] = float(np.nanmean(vals))
            row[f"{key}_std"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
        row["objective"] = weighted_objective({metric: row.get(f"{metric}_mean", 0.0) for metric in weights}, weights)
        trials.append(row)

    trials_df = pd.DataFrame(trials).sort_values(
        ["objective", "auc_mean", "f1_mean", "recall_mean"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    best_row = {key: safe_jsonable(value) for key, value in trials_df.iloc[0].to_dict().items()}
    best_row["best_params"] = {
        key: best_row[key]
        for key in ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "min_child_weight", "gamma", "reg_alpha", "reg_lambda"]
    }
    best_row["score_weights"] = weights
    best_row["search_config"] = search.get("_config_name")
    return trials_df, best_row


def _temporal_search(
    *,
    params_list: list[dict[str, Any]],
    search: dict[str, Any],
    seeds: list[int],
    model: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from src.tuning.common import read_windows

    train_df = read_windows("te_train_windows.csv")
    val_df = read_windows("te_val_windows.csv")
    train_df = maybe_sample_frame(
        train_df,
        frac=search.get("train_sample_frac"),
        n_rows=search.get("train_sample_n"),
        seed=int(search.get("train_sample_seed", 42)),
    )
    weights = search.get(
        "score_weights",
        {"auc": 0.45, "post_shift_recall": 0.20, "f1": 0.15, "recall": 0.10, "high_entropy_recall": 0.05, "grayzone_recall": 0.05},
    )
    entropy_floor = float(search.get("entropy_floor", 0.9))
    gray_margin = float(search.get("gray_margin", 0.1))
    trials = []

    for idx, params in enumerate(params_list, start=1):
        print(f"[reconfirm_{model}] candidate {idx}/{len(params_list)} params={params}", flush=True)
        seed_rows = []
        for seed in seeds:
            if model == "tcn":
                seed_rows.append(
                    tcn_trial_metrics(
                        train_df,
                        val_df,
                        params=params,
                        seed=seed,
                        entropy_floor=entropy_floor,
                        gray_margin=gray_margin,
                    )
                )
            elif model == "adaptable":
                seed_rows.append(
                    adaptable_trial_metrics(
                        train_df,
                        val_df,
                        params=params,
                        seed=seed,
                        entropy_floor=entropy_floor,
                        gray_margin=gray_margin,
                        weights=weights,
                    )
                )
            elif model == "invariant":
                seed_rows.append(
                    invariant_trial_metrics(
                        train_df,
                        val_df,
                        params=params,
                        seed=seed,
                        entropy_floor=entropy_floor,
                        gray_margin=gray_margin,
                        weights=weights,
                    )
                )
            else:
                raise ValueError(f"Unsupported temporal model: {model}")

        row = deepcopy(params)
        row["screening_rank"] = idx
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
        extra_keys = []
        if model == "adaptable":
            extra_keys = ["objective"]
        if model == "invariant":
            extra_keys = ["penalty_weight_value", "objective"]

        for key in metric_keys + extra_keys:
            vals = np.asarray([seed_row[key] for seed_row in seed_rows], dtype=float)
            row[f"{key}_mean"] = float(np.nanmean(vals))
            row[f"{key}_std"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0

        if model == "tcn":
            row["objective"] = weighted_objective({metric: row.get(f"{metric}_mean", 0.0) for metric in weights}, weights)
        else:
            row["objective"] = row["objective_mean"]
        trials.append(row)

    trials_df = pd.DataFrame(trials).sort_values(
        ["objective", "auc_mean", "post_shift_recall_mean", "f1_mean"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    best_row = {key: safe_jsonable(value) for key, value in trials_df.iloc[0].to_dict().items()}
    if model == "tcn":
        param_keys = ["channels", "dilations", "kernel_size", "dropout", "expansion_ratio", "pool", "batch_size", "epochs", "lr", "weight_decay"]
    elif model == "adaptable":
        param_keys = ["architecture", "channels", "dilations", "kernel_size", "dropout", "expansion_ratio", "pool", "batch_size", "inference_batch_size", "epochs", "lr", "weight_decay", "adaptation_lr", "adaptation_steps"]
    else:
        param_keys = ["architecture", "channels", "dilations", "kernel_size", "dropout", "expansion_ratio", "pool", "batch_size", "epochs", "lr", "weight_decay", "penalty_weight"]
    best_row["best_params"] = {key: best_row[key] for key in param_keys if key in best_row}
    best_row["score_weights"] = weights
    best_row["search_config"] = search.get("_config_name")
    return trials_df, best_row


def main() -> None:
    args = parse_args()
    search = read_search_cfg(args.config)
    search["_config_name"] = args.config
    seeds = [int(seed) for seed in args.seeds.split(",") if seed.strip()]

    screening_df = pd.read_csv(Path(args.screening_trials))
    top_df = screening_df.head(int(args.top_k)).copy()
    params_list = [_normalize_params(args.model, row) for row in top_df.to_dict(orient="records")]

    if args.model == "rf":
        trials_df, best_row = _rf_search(params_list, search, seeds)
    elif args.model == "xgb":
        trials_df, best_row = _xgb_search(params_list, search, seeds)
    else:
        trials_df, best_row = _temporal_search(params_list=params_list, search=search, seeds=seeds, model=args.model)

    write_search_outputs(args.output_prefix, trials_df, best_row, output_dir=args.output_dir)
    print(f"[DONE] reconfirm_{args.model}", flush=True)
    print(trials_df.to_string(index=False))


if __name__ == "__main__":
    main()
