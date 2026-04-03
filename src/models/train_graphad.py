from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.models.graphad import fit_graphad, save_graphad_artifact
from src.utils.io import read_yaml, write_json


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


def load_feature_cols() -> list[str]:
    with open(META_DIR / "feature_columns.json", "r", encoding="utf-8") as f:
        return json.load(f)


def read_rows(name: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / name)


def main() -> None:
    ensure_dirs()
    print("\n" + "=" * 80, flush=True)
    print("[START] train_graphad", flush=True)
    print("  config    : configs/train_graphad.yaml", flush=True)
    print("=" * 80, flush=True)
    cfg = read_yaml(
        CONFIG_DIR / "train_graphad.yaml",
        default={
            "normal_only": True,
            "corr_threshold": 0.70,
            "alpha": 0.30,
            "top_k": 5,
            "lambda_z": 0.40,
            "lambda_tr": 0.30,
            "lambda_fl": 0.30,
            "score_clip": 12.0,
            "eps": 1.0e-6,
        },
    )
    feature_cols = load_feature_cols()
    train_df = read_rows("te_train_rows.csv")
    artifact = fit_graphad(train_df=train_df, feature_cols=feature_cols, cfg=cfg)
    save_graphad_artifact(MODEL_DIR / "graphad_artifact.json", artifact)
    write_json(METRIC_DIR / "graphad_training_summary.json", artifact.get("summary", {}))
    print("[DONE] train_graphad", flush=True)
    print(json.dumps(artifact.get("summary", {}), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
