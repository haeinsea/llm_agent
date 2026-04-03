from __future__ import annotations

import json
from pathlib import Path

import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
from sklearn.impute import SimpleImputer

from src.utils.io import read_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
META_DIR = DATA_DIR / "meta"
CONFIG_DIR = PROJECT_ROOT / "configs"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
METRIC_DIR = OUTPUT_DIR / "metrics"


def ensure_dirs() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path, default: dict) -> dict:
    return read_yaml(path, default=default)


def load_feature_cols():
    with open(META_DIR / "feature_columns.json", "r", encoding="utf-8") as f:
        return json.load(f)


def read_rows(name: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / name)


def compute_metrics(y_true, p):
    y_hat = (p >= 0.5).astype(int)
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


def main():
    ensure_dirs()
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_yaml(
        CONFIG_DIR / "train_rf.yaml",
        default={
            "random_state": 42,
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "n_jobs": -1,
        },
    )
    seed = int(args.seed) if args.seed is not None else int(cfg["random_state"])
    print("\n" + "=" * 80, flush=True)
    print(f"[START] train_rf seed={seed}", flush=True)
    print("  config    : configs/train_rf.yaml", flush=True)
    print("=" * 80, flush=True)

    feature_cols = load_feature_cols()
    train_df = read_rows("te_train_rows.csv")
    val_df = read_rows("te_val_rows.csv")

    X_train = train_df[feature_cols].to_numpy()
    y_train = train_df["y"].to_numpy().astype(int)

    X_val = val_df[feature_cols].to_numpy()
    y_val = val_df["y"].to_numpy().astype(int)

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)

    model = RandomForestClassifier(
        n_estimators=int(cfg["n_estimators"]),
        max_depth=cfg["max_depth"],
        min_samples_split=int(cfg.get("min_samples_split", 2)),
        min_samples_leaf=int(cfg["min_samples_leaf"]),
        random_state=seed,
        n_jobs=int(cfg["n_jobs"]),
        class_weight="balanced",
    )
    model.fit(X_train_imp, y_train)

    p_val = model.predict_proba(X_val_imp)[:, 1]
    metrics = compute_metrics(y_val, p_val)

    joblib.dump(model, MODEL_DIR / f"rf_model_seed{seed}.pkl")
    joblib.dump(imputer, MODEL_DIR / f"rf_imputer_seed{seed}.pkl")

    with open(METRIC_DIR / f"rf_val_metrics_seed{seed}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"[DONE] train_rf seed={seed}", flush=True)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
