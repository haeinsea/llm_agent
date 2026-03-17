from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import ensure_dir, read_csv, read_json, read_yaml, write_csv
from src.utils.metrics import binary_metrics, gray_ratio
from src.utils.routing import compute_base_routing_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PRED_DIR = OUTPUT_DIR / "predictions"
METRIC_DIR = OUTPUT_DIR / "metrics"
EVAL_DIR = OUTPUT_DIR / "evaluation"
SHIFT_DIR = OUTPUT_DIR / "shift_analysis_test_main"
APPENDIX_DIR = OUTPUT_DIR / "appendix"
DEFAULT_Q = 0.80
SEEDS = [0, 1, 2, 3, 4]


def fmt(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def pick_std(row: pd.Series, prefix: str) -> float:
    std_col = f"{prefix}_std"
    return float(row[std_col]) if std_col in row.index else 0.0


def build_appendix_a(cfg: dict, tau: float) -> None:
    gray_grid = read_csv(METRIC_DIR / "grayzone_grid.csv").sort_values("q").reset_index(drop=True)
    gray_seed = read_csv(METRIC_DIR / "grayzone_grid_by_seed.csv")
    base_cost = read_csv(PRED_DIR / "base_test_cost_predictions.csv")
    utar_summary = read_csv(METRIC_DIR / "selective_llm_summary.csv")

    rows = []
    for q in gray_grid["q"]:
        utar_row = utar_summary[(utar_summary["dataset"] == "cost") & (utar_summary["mode"] == "selective") & (np.isclose(utar_summary["q"], q))].iloc[0]
        for model_name, col_prefix in [
            ("RF (Base)", "p_rf"),
            ("XGB (Base)", "p_xgb"),
            ("TCN (Base)", "p_tcn"),
            ("Avg. Ensemble (Base)", "p_ensemble"),
        ]:
            seed_metrics = []
            for seed in SEEDS:
                if col_prefix == "p_ensemble":
                    score = base_cost[[f"p_rf_seed{seed}", f"p_xgb_seed{seed}", f"p_tcn_seed{seed}"]].mean(axis=1).to_numpy(dtype=float)
                else:
                    score = base_cost[f"{col_prefix}_seed{seed}"].to_numpy(dtype=float)
                margin = float(gray_seed[(gray_seed["seed"] == seed) & (np.isclose(gray_seed["q"], q))]["gray_margin"].iloc[0])
                m = binary_metrics(base_cost["y_true"], score, tau=tau)
                seed_metrics.append(
                    {
                        "gray_ratio": gray_ratio(score, tau=tau, margin=margin),
                        "f1": m["f1"],
                    }
                )
            metrics_df = pd.DataFrame(seed_metrics)
            rows.append(
                {
                    "q": float(q),
                    "Model": model_name,
                    "Gray Ratio": fmt(metrics_df["gray_ratio"].mean(), metrics_df["gray_ratio"].std(ddof=1)),
                    "F1": fmt(metrics_df["f1"].mean(), metrics_df["f1"].std(ddof=1)),
                }
            )

        rows.append(
            {
                "q": float(q),
                "Model": "UTAR (Proposed)",
                "Gray Ratio": fmt(utar_row["gray_ratio_mean"], pick_std(utar_row, "gray_ratio")),
                "F1": fmt(utar_row["f1_mean"], pick_std(utar_row, "f1")),
            }
        )

    write_csv(APPENDIX_DIR / "table_a1_q_sweep_base_models.csv", pd.DataFrame(rows))


def build_appendix_b() -> None:
    shift_summary = read_csv(SHIFT_DIR / "shift_summary_table.csv")
    write_csv(APPENDIX_DIR / "table_b1_distribution_shift_summary.csv", shift_summary)

    manifest = pd.DataFrame(
        [
            {"Artifact": "KDE plot", "Path": str(SHIFT_DIR / "kde_normal_vs_shift.png")},
            {"Artifact": "PCA scatter", "Path": str(SHIFT_DIR / "pca_phase_scatter.png")},
            {"Artifact": "t-SNE scatter", "Path": str(SHIFT_DIR / "tsne_phase_scatter.png")},
            {"Artifact": "Feature shift summary", "Path": str(SHIFT_DIR / "feature_shift_normal_vs_shift.csv")},
        ]
    )
    write_csv(APPENDIX_DIR / "appendix_b_artifact_manifest.csv", manifest)


def build_appendix_d() -> None:
    summary = read_csv(METRIC_DIR / "selective_llm_summary.csv")
    sel = summary[(summary["dataset"] == "cost") & (summary["mode"] == "selective") & (np.isclose(summary["q"], DEFAULT_Q))].iloc[0]
    full = summary[(summary["dataset"] == "cost") & (summary["mode"] == "full_llm") & (np.isclose(summary["q"], DEFAULT_Q))].iloc[0]
    no_llm = summary[(summary["dataset"] == "cost") & (summary["mode"] == "no_llm") & (np.isclose(summary["q"], DEFAULT_Q))].iloc[0]

    df = pd.DataFrame(
        [
            {
                "Model": "Base Model (No-LLM)",
                "Total Inference Time (s)": fmt(no_llm["total_latency_ms_mean"] / 1000.0, pick_std(no_llm, "total_latency_ms") / 1000.0),
                "Avg Latency per Sample (ms)": fmt(no_llm["avg_latency_ms_per_sample_mean"], pick_std(no_llm, "avg_latency_ms_per_sample")),
                "LLM Call Rate": fmt(no_llm["llm_call_rate_mean"], pick_std(no_llm, "llm_call_rate")),
            },
            {
                "Model": "Full-LLM",
                "Total Inference Time (s)": fmt(full["total_latency_ms_mean"] / 1000.0, pick_std(full, "total_latency_ms") / 1000.0),
                "Avg Latency per Sample (ms)": fmt(full["avg_latency_ms_per_sample_mean"], pick_std(full, "avg_latency_ms_per_sample")),
                "LLM Call Rate": fmt(full["llm_call_rate_mean"], pick_std(full, "llm_call_rate")),
            },
            {
                "Model": "Proposed UTAR",
                "Total Inference Time (s)": fmt(sel["total_latency_ms_mean"] / 1000.0, pick_std(sel, "total_latency_ms") / 1000.0),
                "Avg Latency per Sample (ms)": fmt(sel["avg_latency_ms_per_sample_mean"], pick_std(sel, "avg_latency_ms_per_sample")),
                "LLM Call Rate": fmt(sel["llm_call_rate_mean"], pick_std(sel, "llm_call_rate")),
            },
        ]
    )
    write_csv(APPENDIX_DIR / "table_d1_inference_latency.csv", df)


def _seedwise_utar_metrics(split_name: str, cfg: dict) -> pd.DataFrame:
    seed_metrics = read_csv(METRIC_DIR / "selective_llm_seed_metrics.csv")
    dataset_name = "val" if split_name == "val" else "main"
    out = seed_metrics[(seed_metrics["dataset"] == dataset_name) & (seed_metrics["mode"] == "selective")].copy()
    out = out.rename(
        columns={
            "f1": "F1",
            "recall": "Recall",
            "precision": "Precision",
            "roc_auc": "AUC",
            "gray_ratio": "Gray Ratio",
            "llm_call_rate": "LLM Call Rate",
        }
    )
    out["Method"] = "UTAR (Proposed)"
    if "seed" not in out.columns:
        out["seed"] = -1
    return out[["dataset", "Method", "seed", "F1", "Recall", "Precision", "AUC", "Gray Ratio", "LLM Call Rate"]].rename(columns={"dataset": "split"})


def build_appendix_e(cfg: dict) -> None:
    seed_main = read_csv(EVAL_DIR / "test_main_seed_metrics_thresholded.csv").copy()
    seed_main["Method"] = seed_main["model"].map({"RF": "RF (Base)", "XGB": "XGB (Base)", "TCN": "TCN (Base)"})
    seed_main = seed_main[["split", "Method", "seed", "f1", "recall", "precision", "auc", "threshold", "coverage"]].rename(
        columns={"f1": "F1", "recall": "Recall", "precision": "Precision", "auc": "AUC", "threshold": "Threshold", "coverage": "Coverage"}
    )

    utar_seed_main = _seedwise_utar_metrics("test_main", cfg)
    detail = pd.concat([seed_main, utar_seed_main], ignore_index=True, sort=False)
    write_csv(APPENDIX_DIR / "table_e1_seed_variation_detail.csv", detail)

    summary_rows = []
    for method, g in detail.groupby("Method"):
        summary_rows.append(
            {
                "Method": method,
                "F1 Mean": float(g["F1"].mean()),
                "F1 SD": float(g["F1"].std(ddof=1)) if len(g) > 1 else 0.0,
                "Recall Mean": float(g["Recall"].mean()),
                "Recall SD": float(g["Recall"].std(ddof=1)) if len(g) > 1 else 0.0,
            }
        )
    write_csv(APPENDIX_DIR / "table_e2_seed_variation_summary.csv", pd.DataFrame(summary_rows))


def main() -> None:
    ensure_dir(APPENDIX_DIR)
    cfg = read_yaml(CONFIG_DIR / "routing.yaml", default={})
    tau = float(read_json(METRIC_DIR / "thresholds.json")["tau"])

    build_appendix_a(cfg, tau)
    build_appendix_b()
    build_appendix_d()
    build_appendix_e(cfg)
    print(f"Saved TE appendix artifacts to {APPENDIX_DIR}")


if __name__ == "__main__":
    main()
