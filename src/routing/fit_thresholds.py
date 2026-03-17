from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import read_csv, read_yaml, write_csv, write_json
from src.utils.metrics import binary_metrics
from src.utils.routing import compute_base_routing_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PRED_DIR = OUTPUT_DIR / "predictions"
METRIC_DIR = OUTPUT_DIR / "metrics"
SEEDS = [0, 1, 2, 3, 4]


def best_tau_for_score(y_true: np.ndarray, p: np.ndarray, grid: np.ndarray, metric_name: str) -> tuple[float, dict]:
    best = None
    for tau in grid:
        m = binary_metrics(y_true, p, tau=float(tau))
        score = m[metric_name]
        row = {"tau": float(tau), **m}
        if best is None or score > best["score"]:
            best = {"score": score, "row": row}
    if best is None:
        raise RuntimeError("Threshold fitting failed.")
    return float(best["row"]["tau"]), best["row"]


def main() -> None:
    cfg = read_yaml(CONFIG_DIR / "routing.yaml", default={})
    pred_val = read_csv(PRED_DIR / "base_val_predictions.csv")

    metric_name = str(cfg.get("tau_metric", "f1"))
    tmin = float(cfg.get("threshold_grid_min", 0.05))
    tmax = float(cfg.get("threshold_grid_max", 0.95))
    tstep = float(cfg.get("threshold_grid_step", 0.01))
    grid = np.arange(tmin, tmax + 1e-12, tstep)
    y_true = pred_val["y_true"].to_numpy().astype(int)

    rows = []
    per_seed: dict[str, dict] = {}
    for seed in SEEDS:
        seed_key = f"seed{seed}"
        p_utar_seed = compute_base_routing_score(
            pred_val,
            cfg,
            rf_col=f"p_rf_seed{seed}",
            xgb_col=f"p_xgb_seed{seed}",
            tcn_col=f"p_tcn_seed{seed}",
        ).to_numpy()
        tau_seed, metrics_seed = best_tau_for_score(y_true, p_utar_seed, grid, metric_name)
        per_seed[seed_key] = {"tau": tau_seed, "val_metrics_at_tau": metrics_seed}
        rows.append({"seed": seed, "tau": tau_seed, **metrics_seed})

    pred_val["p_utar_base"] = compute_base_routing_score(pred_val, cfg)
    tau_mean, metrics_mean = best_tau_for_score(y_true, pred_val["p_utar_base"].to_numpy(), grid, metric_name)

    seed_df = pd.DataFrame(rows)
    write_csv(METRIC_DIR / "thresholds_by_seed.csv", seed_df)

    out = {
        "tau": tau_mean,
        "selected_metric": metric_name,
        "val_metrics_at_tau": metrics_mean,
        "per_seed": per_seed,
        "tau_mean_from_seeds": float(seed_df["tau"].mean()),
        "tau_std_from_seeds": float(seed_df["tau"].std(ddof=1)) if len(seed_df) > 1 else 0.0,
    }
    write_json(METRIC_DIR / "thresholds.json", out)
    print(out)


if __name__ == "__main__":
    main()
