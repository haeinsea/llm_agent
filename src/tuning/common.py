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


def resolve_tuning_dir(output_dir: str | Path | None = None) -> Path:
    if output_dir is None:
        ensure_tuning_dir()
        return TUNING_DIR
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_feature_cols() -> list[str]:
    with open(META_DIR / "feature_columns.json", "r", encoding="utf-8") as f:
        return json.load(f)


def read_rows(name: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / name)


def read_windows(name: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / name)


def read_search_cfg(name: str) -> dict[str, Any]:
    return read_yaml(CONFIG_DIR / name, default={})


def write_search_outputs(
    prefix: str,
    trials_df: pd.DataFrame,
    best_row: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
) -> None:
    out_dir = resolve_tuning_dir(output_dir)
    write_csv(out_dir / f"{prefix}_trials.csv", trials_df)
    write_json(out_dir / f"{prefix}_best.json", best_row)


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


def build_trial_space(space: dict[str, Any], exclude_keys: Iterable[str] | None = None) -> list[dict[str, Any]]:
    exclude = set(exclude_keys or [])
    exclude.update({"max_trials", "trial_sample_seed"})
    trials = param_product(space, exclude_keys=exclude)
    max_trials = space.get("max_trials")
    if max_trials is None:
        return trials
    max_trials = int(max_trials)
    if max_trials <= 0 or len(trials) <= max_trials:
        return trials
    sample_seed = int(space.get("trial_sample_seed", 42))
    rng = np.random.default_rng(sample_seed)
    picked = rng.choice(len(trials), size=max_trials, replace=False)
    return [trials[int(idx)] for idx in picked]


def maybe_sample_frame(
    df: pd.DataFrame,
    *,
    frac: float | None = None,
    n_rows: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    if frac is not None:
        frac = float(frac)
        if 0.0 < frac < 1.0:
            return df.sample(frac=frac, random_state=seed).reset_index(drop=True)
    if n_rows is not None:
        n_rows = int(n_rows)
        if 0 < n_rows < len(df):
            return df.sample(n=n_rows, random_state=seed).reset_index(drop=True)
    return df.reset_index(drop=True)


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


def binary_entropy(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
    return -(probs * np.log2(probs) + (1.0 - probs) * np.log2(1.0 - probs))


def recall_at_mask(y_true: np.ndarray, preds: np.ndarray, mask: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() == 0:
        return float("nan")
    y_mask = np.asarray(y_true, dtype=int)[mask]
    if len(y_mask) == 0 or int((y_mask == 1).sum()) == 0:
        return float("nan")
    pred_mask = np.asarray(preds, dtype=int)[mask]
    tp = int(((pred_mask == 1) & (y_mask == 1)).sum())
    fn = int(((pred_mask == 0) & (y_mask == 1)).sum())
    return tp / max(tp + fn, 1)


def probability_focus_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    tau: float,
    *,
    entropy_floor: float = 0.9,
    gray_margin: float = 0.1,
) -> dict[str, float]:
    probs = np.asarray(probs, dtype=float)
    preds = (probs >= float(tau)).astype(int)
    ent = binary_entropy(probs)
    high_entropy_mask = ent >= float(entropy_floor)
    grayzone_mask = np.abs(probs - float(tau)) <= float(gray_margin)
    return {
        "high_entropy_recall": float(recall_at_mask(y_true, preds, high_entropy_mask)),
        "grayzone_recall": float(recall_at_mask(y_true, preds, grayzone_mask)),
        "mean_entropy": float(np.nanmean(ent)),
        "grayzone_share": float(np.mean(grayzone_mask)),
    }


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
