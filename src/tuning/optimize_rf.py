from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, roc_auc_score

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


def main() -> None:
    print("\n" + "=" * 80, flush=True)
    print("[START] optimize_rf", flush=True)
    print("  objective : validation F1 + recall", flush=True)
    print("  config    : configs/search_rf.yaml", flush=True)
    print("=" * 80, flush=True)
    search = read_search_cfg("search_rf.yaml")
    feature_cols = load_feature_cols()
    train_df = read_rows("te_train_rows.csv")
    val_df = read_rows("te_val_rows.csv")

    X_train = train_df[feature_cols].to_numpy()
    y_train = train_df["y"].to_numpy().astype(int)
    X_val = val_df[feature_cols].to_numpy()
    y_val = val_df["y"].to_numpy().astype(int)

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)

    weights = search.get("score_weights", {"f1": 1.0, "recall": 0.2})
    seeds = [int(seed) for seed in search.get("search_seeds", [0])]
    trials = []
    trial_space = param_product(search, exclude_keys={"search_seeds", "selection_metric", "score_weights"})
    print(f"[optimize_rf] train_rows={len(train_df):,} val_rows={len(val_df):,} seeds={seeds} trials={len(trial_space):,}", flush=True)

    for idx, params in enumerate(trial_space, start=1):
        print(f"[optimize_rf] trial {idx}/{len(trial_space)} params={params}", flush=True)
        seed_rows = []
        for seed in seeds:
            model = RandomForestClassifier(
                n_estimators=int(params.get("n_estimators", 300)),
                max_depth=params.get("max_depth"),
                min_samples_split=int(params.get("min_samples_split", 2)),
                min_samples_leaf=int(params.get("min_samples_leaf", 1)),
                random_state=seed,
                n_jobs=-1,
                class_weight="balanced",
            )
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_val)[:, 1]
            seed_rows.append(_trial_metrics(y_val, probs))

        row = deepcopy(params)
        for key in ["threshold", "f1", "recall", "precision", "auc"]:
            vals = np.asarray([seed_row[key] for seed_row in seed_rows], dtype=float)
            row[f"{key}_mean"] = float(np.nanmean(vals))
            row[f"{key}_std"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
        row["objective"] = weighted_objective({"f1": row["f1_mean"], "recall": row["recall_mean"]}, weights)
        trials.append(row)

    trials_df = pd.DataFrame(trials).sort_values(["objective", "f1_mean", "recall_mean"], ascending=[False, False, False]).reset_index(drop=True)
    best_row = {key: safe_jsonable(value) for key, value in trials_df.iloc[0].to_dict().items()}
    best_row["best_params"] = {
        key: best_row[key]
        for key in ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"]
        if key in best_row
    }
    write_search_outputs("rf", trials_df, best_row)
    print("[DONE] optimize_rf", flush=True)
    print(trials_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
