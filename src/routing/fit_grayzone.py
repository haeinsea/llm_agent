from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.experiment import get_seed_list
from src.utils.io import read_csv, read_json, read_yaml, write_csv, write_json
from src.utils.metrics import gray_ratio
from src.utils.routing import build_routing_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PRED_DIR = OUTPUT_DIR / "predictions"
METRIC_DIR = OUTPUT_DIR / "metrics"
SEEDS = get_seed_list()


def quantile_margin(errors: np.ndarray, q: float) -> float:
    errors = np.asarray(errors, dtype=float)
    return float(np.quantile(errors, q))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--q-values",
        nargs="+",
        type=float,
        help="Optional subset of q values to recompute and merge into the existing gray-zone grid.",
    )
    args = parser.parse_args()

    cfg = read_yaml(CONFIG_DIR / "routing.yaml", default={})
    pred_val = read_csv(PRED_DIR / "base_val_predictions.csv")
    tau_info = read_json(METRIC_DIR / "thresholds.json")
    configured_q_grid = cfg.get("q_grid", [0.60, 0.70, 0.80, 0.90])
    q_grid = list(dict.fromkeys(float(q) for q in (args.q_values if args.q_values is not None else configured_q_grid)))
    entropy_q = float(cfg.get("entropy_shortcut_quantile", 0.80))
    discrepancy_q = float(cfg.get("discrepancy_shortcut_quantile", 0.80))

    rows = []
    per_seed: dict[str, dict[str, dict[str, float]]] = {}
    for seed in SEEDS:
        seed_key = f"seed{seed}"
        tau_seed = float(tau_info["per_seed"][seed_key]["tau"])
        routing_seed = build_routing_features(
            pred_val,
            cfg,
            rf_col=f"p_rf_seed{seed}",
            xgb_col=f"p_xgb_seed{seed}",
            tcn_col=f"p_tcn_seed{seed}",
        )
        p_seed = routing_seed["p_utar_base"].to_numpy()
        y = pred_val["y_true"].to_numpy().astype(int)
        mis = ((p_seed >= tau_seed).astype(int) != y)
        errors = np.abs(p_seed[mis] - tau_seed) if mis.sum() > 0 else np.abs(p_seed - tau_seed)

        per_seed[seed_key] = {}
        for q in q_grid:
            m_q = quantile_margin(errors, q)
            gray_mask = np.abs(p_seed - tau_seed) <= m_q
            if gray_mask.sum() == 0:
                gray_mask = np.ones(len(pred_val), dtype=bool)
            entropy_values = routing_seed.loc[gray_mask, "ensemble_entropy"].to_numpy(dtype=float)
            entropy_threshold = float(np.quantile(entropy_values, entropy_q))
            row = {
                "seed": seed,
                "q": float(q),
                "tau": tau_seed,
                "gray_margin": float(m_q),
                "gray_ratio_val": gray_ratio(p_seed, tau=tau_seed, margin=m_q),
                "entropy_threshold": entropy_threshold,
                "discrepancy_threshold": 1.0,
            }
            rows.append(row)
            per_seed[seed_key][f"{float(q):.2f}"] = {
                "tau": tau_seed,
                "gray_margin": float(m_q),
                "gray_ratio_val": float(row["gray_ratio_val"]),
                "entropy_threshold": entropy_threshold,
                "discrepancy_threshold": 1.0,
            }

    df = pd.DataFrame(rows).sort_values(["q", "seed"]).reset_index(drop=True)
    existing_df = read_csv(METRIC_DIR / "grayzone_grid_by_seed.csv") if (METRIC_DIR / "grayzone_grid_by_seed.csv").exists() else pd.DataFrame()
    if not existing_df.empty:
        df = (
            pd.concat([existing_df, df], ignore_index=True)
            .drop_duplicates(subset=["q", "seed"], keep="last")
            .sort_values(["q", "seed"])
            .reset_index(drop=True)
        )
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
            entropy_threshold_mean=("entropy_threshold", "mean"),
            entropy_threshold_std=("entropy_threshold", "std"),
            discrepancy_threshold_mean=("discrepancy_threshold", "mean"),
            discrepancy_threshold_std=("discrepancy_threshold", "std"),
        )
        .fillna(0.0)
    )
    write_csv(METRIC_DIR / "grayzone_grid.csv", summary)
    defaults_path = METRIC_DIR / "grayzone_defaults.json"
    existing_defaults = read_json(defaults_path) if defaults_path.exists() else {}
    merged_per_seed = existing_defaults.get("per_seed", {}) if isinstance(existing_defaults.get("per_seed", {}), dict) else {}
    for seed_key, q_info in per_seed.items():
        current = merged_per_seed.get(seed_key, {})
        if not isinstance(current, dict):
            current = {}
        current.update(q_info)
        merged_per_seed[seed_key] = current
    write_json(
        defaults_path,
        {
            "default_q": existing_defaults.get("default_q", 0.80),
            "entropy_shortcut_quantile": entropy_q,
            "discrepancy_shortcut_quantile": discrepancy_q,
            "per_seed": merged_per_seed,
        },
    )
    print(summary)


if __name__ == "__main__":
    main()
