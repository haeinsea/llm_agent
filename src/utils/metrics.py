from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def binary_metrics(y_true, p, tau: float = 0.5) -> Dict[str, float | None]:
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)
    y_hat = (p >= tau).astype(int)

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


def gray_ratio(p: np.ndarray, tau: float, margin: float) -> float:
    p = np.asarray(p, dtype=float)
    return float(np.mean(np.abs(p - tau) <= margin))


def disagreement_ratio(p1: np.ndarray, p2: np.ndarray, thr: float = 0.2) -> float:
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    return float(np.mean(np.abs(p1 - p2) >= thr))


def rolling_variance(x: np.ndarray, window: int = 15) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan, dtype=float)
    if len(x) < window:
        return out
    for i in range(window - 1, len(x)):
        out[i] = np.var(x[i - window + 1:i + 1])
    return out


def instability_score(p: np.ndarray, run_ids: np.ndarray, phases: Optional[np.ndarray] = None, window: int = 15) -> float:
    p = np.asarray(p, dtype=float)
    run_ids = np.asarray(run_ids)
    if phases is not None:
        phases = np.asarray(phases)

    vals = []
    for run in np.unique(run_ids):
        m = run_ids == run
        pr = p[m]
        ph = phases[m] if phases is not None else None

        if ph is not None:
            focus = np.isin(ph, ["transition", "post_shift"])
            if focus.sum() >= window:
                pr = pr[focus]

        rv = rolling_variance(pr, window=window)
        rv = rv[~np.isnan(rv)]
        if len(rv):
            vals.append(np.mean(rv))
    if not vals:
        return np.nan
    return float(np.mean(vals))


def worst_case_recall(y_true: np.ndarray, p: np.ndarray, run_ids: np.ndarray, tau: float = 0.5, window: int = 50) -> float:
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)
    run_ids = np.asarray(run_ids)

    recalls = []
    for run in np.unique(run_ids):
        m = run_ids == run
        yr = y_true[m]
        pr = p[m]
        if len(yr) < window:
            continue
        for i in range(window - 1, len(yr)):
            ys = yr[i - window + 1:i + 1]
            ps = pr[i - window + 1:i + 1]
            if ys.sum() == 0:
                continue
            rec = recall_score(ys, (ps >= tau).astype(int), zero_division=0)
            recalls.append(rec)
    if not recalls:
        return np.nan
    return float(np.min(recalls))


def prr(base_recall: float, utar_recall: float) -> float:
    if base_recall is None or np.isnan(base_recall):
        return np.nan
    if base_recall <= 0:
        return np.nan
    return float(utar_recall / base_recall)