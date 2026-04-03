from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from src.routing.selective_llm_eval import build_llm_runner, get_seed_view, run_mode
from src.tuning.common import param_product, read_search_cfg, safe_jsonable, weighted_objective, write_search_outputs
from src.utils.experiment import get_seed_list
from src.utils.io import read_csv, read_yaml
from src.utils.metrics import binary_metrics


PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
METRIC_DIR = PROJECT_ROOT / "outputs" / "metrics"
PRED_DIR = PROJECT_ROOT / "outputs" / "predictions"
CONFIG_DIR = PROJECT_ROOT / "configs"
SEEDS = get_seed_list()


def _best_tau(y_true: np.ndarray, probs: np.ndarray) -> float:
    grid = np.linspace(0.05, 0.95, 91)
    best_tau = 0.5
    best_f1 = -np.inf
    for tau in grid:
        f1 = binary_metrics(y_true, probs, tau=float(tau))["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_tau = float(tau)
    return best_tau


def main() -> None:
    print("\n" + "=" * 80, flush=True)
    print("[START] optimize_routing", flush=True)
    print("  objective : validation F1 + recall - call_rate - instability", flush=True)
    print("  config    : configs/search_routing.yaml", flush=True)
    print("=" * 80, flush=True)
    search = read_search_cfg("search_routing.yaml")
    base_cfg = read_yaml(CONFIG_DIR / "routing.yaml", default={})
    pred_val = read_csv(PRED_DIR / "base_val_predictions.csv")
    llm_runner = build_llm_runner(base_cfg, force_stub=True)
    weights = search.get("score_weights", {"f1": 1.0, "recall": 0.3, "llm_call_rate": -0.2, "instability": -0.1})
    trials = []
    trial_space = param_product(search, exclude_keys={"selection_metric", "llm_mode", "score_weights"})
    print(f"[optimize_routing] val_rows={len(pred_val):,} seeds={SEEDS} trials={len(trial_space):,}", flush=True)

    for idx, params in enumerate(trial_space, start=1):
        print(f"[optimize_routing] trial {idx}/{len(trial_space)} params={params}", flush=True)
        cfg = {**base_cfg, **params}
        seed_rows = []
        for seed in SEEDS:
            seed_df = get_seed_view(pred_val, seed=seed, cfg=cfg)
            tau = _best_tau(seed_df["y_true"].to_numpy(), seed_df["p_utar_base"].to_numpy())
            mis = ((seed_df["p_utar_base"] >= tau).astype(int) != seed_df["y_true"].astype(int))
            errors = np.abs(seed_df.loc[mis, "p_utar_base"] - tau).to_numpy(dtype=float)
            if len(errors) == 0:
                errors = np.abs(seed_df["p_utar_base"].to_numpy(dtype=float) - tau)
            margin = float(np.quantile(errors, float(params["q_values"])))
            gray_mask = np.abs(seed_df["p_utar_base"].to_numpy(dtype=float) - tau) <= margin
            ent_thr = float(np.quantile(seed_df.loc[gray_mask, "ensemble_entropy"].to_numpy(dtype=float), float(params["entropy_shortcut_quantile"])))
            disc_thr = 1.0
            out, metrics = run_mode(
                base_df=seed_df,
                tau=tau,
                margin=margin,
                entropy_threshold=ent_thr,
                discrepancy_threshold=disc_thr,
                cfg=cfg,
                mode="selective",
                llm_runner=llm_runner,
                ref_recall=float(binary_metrics(seed_df["y_true"], seed_df["p_utar_base"], tau=tau)["recall"]),
                base_latency_ms=0.0,
                routing_feature_latency_ms=0.0,
                progress_label=f"routing-opt seed={seed}",
            )
            seed_rows.append(
                {
                    "f1": float(metrics["f1"]),
                    "recall": float(metrics["recall"]),
                    "llm_call_rate": float(metrics["llm_call_rate"]),
                    "instability": float(metrics["instability"]),
                    "gray_ratio": float(metrics["gray_ratio"]),
                    "tau": float(tau),
                    "gray_margin": float(margin),
                    "entropy_threshold": float(ent_thr),
                    "discrepancy_threshold": float(disc_thr),
                }
            )

        row = deepcopy(params)
        row["q"] = float(params["q_values"])
        for key in ["f1", "recall", "llm_call_rate", "instability", "gray_ratio", "tau", "gray_margin", "entropy_threshold", "discrepancy_threshold"]:
            vals = np.asarray([seed_row[key] for seed_row in seed_rows], dtype=float)
            row[f"{key}_mean"] = float(np.nanmean(vals))
            row[f"{key}_std"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
        row["objective"] = weighted_objective(
            {
                "f1": row["f1_mean"],
                "recall": row["recall_mean"],
                "llm_call_rate": row["llm_call_rate_mean"],
                "instability": row["instability_mean"],
            },
            weights,
        )
        trials.append(row)

    trials_df = pd.DataFrame(trials).sort_values(["objective", "f1_mean", "recall_mean"], ascending=[False, False, False]).reset_index(drop=True)
    best_row = {key: safe_jsonable(value) for key, value in trials_df.iloc[0].to_dict().items()}
    best_row["best_params"] = {
        "q": best_row["q"],
        "sigmoid_gain": best_row["sigmoid_gain"],
        "entropy_shortcut_quantile": best_row["entropy_shortcut_quantile"],
        "discrepancy_shortcut_quantile": best_row["discrepancy_shortcut_quantile"],
    }
    write_search_outputs("routing", trials_df, best_row)
    print("[DONE] optimize_routing", flush=True)
    print(trials_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
