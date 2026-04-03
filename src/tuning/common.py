from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.utils.io import read_json, read_yaml, write_csv, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
META_DIR = DATA_DIR / "meta"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TUNING_DIR = OUTPUT_DIR / "tuning"


def ensure_tuning_dir() -> None:
    TUNING_DIR.mkdir(parents=True, exist_ok=True)


def load_feature_cols() -> list[str]:
    with open(META_DIR / "feature_columns.json", "r", encoding="utf-8") as f:
        return json.load(f)


def read_rows(name: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / name)


def read_windows(name: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / name)


def read_search_cfg(name: str) -> dict[str, Any]:
    return read_yaml(CONFIG_DIR / name, default={})


def write_search_outputs(prefix: str, trials_df: pd.DataFrame, best_row: dict[str, Any]) -> None:
    ensure_tuning_dir()
    write_csv(TUNING_DIR / f"{prefix}_trials.csv", trials_df)
    write_json(TUNING_DIR / f"{prefix}_best.json", best_row)


def param_product(space: dict[str, Any], exclude_keys: Iterable[str] | None = None) -> list[dict[str, Any]]:
    exclude = set(exclude_keys or [])
    keys = [key for key, value in space.items() if isinstance(value, list) and key not in exclude]
    if not keys:
        return [{}]
    values = [space[key] for key in keys]
    out = []
    for combo in itertools.product(*values):
        out.append({key: value for key, value in zip(keys, combo)})
    return out


def best_threshold(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs).astype(float)
    grid = np.linspace(0.05, 0.95, 91)
    best_tau = 0.5
    best_f1 = -np.inf
    for tau in grid:
        preds = (probs >= tau).astype(int)
        tp = int(((preds == 1) & (y_true == 1)).sum())
        fp = int(((preds == 1) & (y_true == 0)).sum())
        fn = int(((preds == 0) & (y_true == 1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
        if f1 > best_f1:
            best_f1 = f1
            best_tau = float(tau)
    return best_tau, float(best_f1)


def weighted_objective(row: dict[str, Any], weights: dict[str, float]) -> float:
    score = 0.0
    for key, weight in weights.items():
        score += float(weight) * float(row.get(key, 0.0))
    return float(score)


def safe_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    return value

