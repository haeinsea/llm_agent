from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import read_csv, read_json, read_yaml, write_csv, write_json
from src.utils.metrics import gray_ratio
from src.utils.routing import compute_base_routing_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PRED_DIR = OUTPUT_DIR / "predictions"
METRIC_DIR = OUTPUT_DIR / "metrics"
SEEDS = [0, 1, 2, 3, 4]


def quantile_margin(errors: np.ndarray, q: float) -> float:
    errors = np.asarray(errors, dtype=float)
    return float(np.quantile(errors, q))


def main() -> None:
    cfg = read_yaml(CONFIG_DIR / "routing.yaml", default={})
    pred_val = read_csv(PRED_DIR / "base_val_predictions.csv")
    tau_info = read_json(METRIC_DIR / "thresholds.json")
    q_grid = cfg.get("q_grid", [0.60, 0.70, 0.80, 0.90])

    rows = []
    per_seed: dict[str, dict[str, dict[str, float]]] = {}
    for seed in SEEDS:
        seed_key = f"seed{seed}"
        tau_seed = float(tau_info["per_seed"][seed_key]["tau"])
        p_seed = compute_base_routing_score(
            pred_val,
            cfg,
            rf_col=f"p_rf_seed{seed}",
            xgb_col=f"p_xgb_seed{seed}",
            tcn_col=f"p_tcn_seed{seed}",
        ).to_numpy()
        y = pred_val["y_true"].to_numpy().astype(int)
        mis = ((p_seed >= tau_seed).astype(int) != y)
        errors = np.abs(p_seed[mis] - tau_seed) if mis.sum() > 0 else np.abs(p_seed - tau_seed)

        per_seed[seed_key] = {}
        for q in q_grid:
            m_q = quantile_margin(errors, q)
            row = {
                "seed": seed,
                "q": float(q),
                "tau": tau_seed,
                "gray_margin": float(m_q),
                "gray_ratio_val": gray_ratio(p_seed, tau=tau_seed, margin=m_q),
            }
            rows.append(row)
            per_seed[seed_key][f"{float(q):.2f}"] = {
                "tau": tau_seed,
                "gray_margin": float(m_q),
                "gray_ratio_val": float(row["gray_ratio_val"]),
            }

    df = pd.DataFrame(rows).sort_values(["q", "seed"]).reset_index(drop=True)
    write_csv(METRIC_DIR / "grayzone_grid_by_seed.csv", df)

    summary = (
        df.groupby("q", as_index=False)
        .agg(
            tau_mean=("tau", "mean"),
            tau_std=("tau", "std"),
            gray_margin_mean=("gray_margin", "mean"),
            gray_margin_std=("gray_margin", "std"),
            gray_ratio_val_mean=("gray_ratio_val", "mean"),
            gray_ratio_val_std=("gray_ratio_val", "std"),
        )
        .fillna(0.0)
    )
    write_csv(METRIC_DIR / "grayzone_grid.csv", summary)
    write_json(METRIC_DIR / "grayzone_defaults.json", {"default_q": 0.80, "per_seed": per_seed})
    print(summary)


if __name__ == "__main__":
    main()
