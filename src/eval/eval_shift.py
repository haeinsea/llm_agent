from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.models.temporal_backbone import temporal_model_display_name
from src.routing.selective_llm_eval import DEFAULT_Q, read_selected_q
from src.utils.experiment import ensemble_component_label, get_seed_list
from src.utils.io import read_csv, read_yaml, write_csv
from src.utils.metrics import binary_metrics, instability_score, low_tail_recall, prr
from src.utils.runtime import get_base_runtime_stat, load_base_runtime_summary


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
METRIC_DIR = OUTPUT_DIR / "metrics"
EVAL_DIR = OUTPUT_DIR / "evaluation"
PRED_DIR = OUTPUT_DIR / "predictions"
CONFIG_DIR = PROJECT_ROOT / "configs"
BASE_RUNTIME_SUMMARY_PATH = METRIC_DIR / "base_inference_runtime_summary.json"
SEEDS = get_seed_list()
RF_COMPONENT = ensemble_component_label("RF")
XGB_COMPONENT = ensemble_component_label("XGB")
GRAPHAD_COMPONENT = "GraphAD+"
AVG_ENSEMBLE_STACK_COMPONENT = "Avg. Ensemble Stack"
BASE_STACK_COMPONENT = "UTAR Base Stack"


def fmt(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def best_tau(y_true: np.ndarray, p: np.ndarray) -> float:
    grid = np.linspace(0.01, 0.99, 99)
    best_tau_val = 0.5
    best_f1 = -1.0
    for tau in grid:
        f1 = binary_metrics(y_true, p, tau=float(tau))["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_tau_val = float(tau)
    return best_tau_val


def summarize_seed_rows(rows: list[dict], method_name: str) -> dict:
    df = pd.DataFrame(rows)
    return {
        "Method": method_name,
        "Ave. F1": fmt(df["f1"].mean(), df["f1"].std(ddof=1) if len(df) > 1 else 0.0),
        "PRR": fmt(df["prr"].mean(), df["prr"].std(ddof=1) if len(df) > 1 else 0.0),
        "Worst-Case Recall (P5)": fmt(df["worst_case_recall"].mean(), df["worst_case_recall"].std(ddof=1) if len(df) > 1 else 0.0),
        "Instability": fmt(df["instability"].mean(), df["instability"].std(ddof=1) if len(df) > 1 else 0.0),
        "Recall": fmt(df["recall"].mean(), df["recall"].std(ddof=1) if len(df) > 1 else 0.0),
        "Precision": fmt(df["precision"].mean(), df["precision"].std(ddof=1) if len(df) > 1 else 0.0),
        "AUC": fmt(df["roc_auc"].mean(), df["roc_auc"].std(ddof=1) if len(df) > 1 else 0.0) if df["roc_auc"].notna().any() else "NA",
    }


def pick_std(row: pd.Series, prefix: str) -> float:
    std_col = f"{prefix}_std"
    return float(row[std_col]) if std_col in row.index else 0.0


def _base_time_seconds(runtime_summary: dict, split: str, component: str) -> str:
    total_ms = get_base_runtime_stat(runtime_summary, split=split, component=component, field="total_latency_ms", default=np.nan)
    total_std_ms = get_base_runtime_stat(runtime_summary, split=split, component=component, field="total_latency_ms_std", default=0.0)
    return fmt(total_ms / 1000.0, total_std_ms / 1000.0) if np.isfinite(total_ms) else "NA"


def _base_time_per_sample_ms(runtime_summary: dict, split: str, component: str) -> str:
    avg_ms = get_base_runtime_stat(runtime_summary, split=split, component=component, field="avg_latency_ms_per_sample", default=np.nan)
    avg_std_ms = get_base_runtime_stat(runtime_summary, split=split, component=component, field="avg_latency_ms_per_sample_std", default=0.0)
    return fmt(avg_ms, avg_std_ms) if np.isfinite(avg_ms) else "NA"


def _ensemble_time_seconds(runtime_summary: dict, split: str, temporal_component: str) -> str:
    total_ms = get_base_runtime_stat(runtime_summary, split=split, component=AVG_ENSEMBLE_STACK_COMPONENT, field="total_latency_ms", default=np.nan)
    total_std_ms = get_base_runtime_stat(runtime_summary, split=split, component=AVG_ENSEMBLE_STACK_COMPONENT, field="total_latency_ms_std", default=0.0)
    if np.isfinite(total_ms):
        return fmt(total_ms / 1000.0, total_std_ms / 1000.0)
    total_ms = (
        get_base_runtime_stat(runtime_summary, split=split, component=RF_COMPONENT, field="total_latency_ms", default=0.0)
        + get_base_runtime_stat(runtime_summary, split=split, component=XGB_COMPONENT, field="total_latency_ms", default=0.0)
        + get_base_runtime_stat(runtime_summary, split=split, component=temporal_component, field="total_latency_ms", default=0.0)
    )
    return fmt(total_ms / 1000.0, 0.0)


def _ensemble_time_per_sample_ms(runtime_summary: dict, split: str, temporal_component: str) -> str:
    total_avg_ms = get_base_runtime_stat(runtime_summary, split=split, component=AVG_ENSEMBLE_STACK_COMPONENT, field="avg_latency_ms_per_sample", default=np.nan)
    total_avg_std_ms = get_base_runtime_stat(runtime_summary, split=split, component=AVG_ENSEMBLE_STACK_COMPONENT, field="avg_latency_ms_per_sample_std", default=0.0)
    if np.isfinite(total_avg_ms):
        return fmt(total_avg_ms, total_avg_std_ms)
    total_avg_ms = (
        get_base_runtime_stat(runtime_summary, split=split, component=RF_COMPONENT, field="avg_latency_ms_per_sample", default=0.0)
        + get_base_runtime_stat(runtime_summary, split=split, component=XGB_COMPONENT, field="avg_latency_ms_per_sample", default=0.0)
        + get_base_runtime_stat(runtime_summary, split=split, component=temporal_component, field="avg_latency_ms_per_sample", default=0.0)
    )
    return fmt(total_avg_ms, 0.0)


def _prediction_low_tail(path: Path, tau: float = 0.41) -> str:
    df = read_csv(path)
    value = low_tail_recall(df["y_true"], df["p_final"], df["run_id"], tau=tau, window=50, quantile=0.05)
    return fmt(value, 0.0)


def main() -> None:
    selected_q = read_selected_q(DEFAULT_Q)
    val_pred = read_csv(OUTPUT_DIR / "predictions" / "base_val_predictions.csv")
    test_pred = read_csv(OUTPUT_DIR / "predictions" / "base_test_main_predictions.csv")
    utar = read_csv(METRIC_DIR / "selective_llm_summary.csv")
    utar_seed = read_csv(METRIC_DIR / "selective_llm_seed_metrics.csv")
    runtime_summary = load_base_runtime_summary(BASE_RUNTIME_SUMMARY_PATH)
    tcn_cfg = read_yaml(CONFIG_DIR / "train_tcn.yaml", default={})
    temporal_name = temporal_model_display_name(tcn_cfg.get("architecture", "modern_tcn"))
    temporal_label = f"{temporal_name} (Base)"
    temporal_component = ensemble_component_label(temporal_name)

    base_rows_map: dict[str, list[dict]] = {
        "RF (Base)": [],
        "XGB (Base)": [],
        temporal_label: [],
        "Avg. Ensemble (Base)": [],
    }
    for seed in SEEDS:
        for label, val_col, test_col in [
            ("RF (Base)", f"p_rf_seed{seed}", f"p_rf_seed{seed}"),
            ("XGB (Base)", f"p_xgb_seed{seed}", f"p_xgb_seed{seed}"),
            (temporal_label, f"p_tcn_seed{seed}", f"p_tcn_seed{seed}"),
            ("Avg. Ensemble (Base)", None, None),
        ]:
            if label == "Avg. Ensemble (Base)":
                p_val = val_pred[[f"p_rf_seed{seed}", f"p_xgb_seed{seed}", f"p_tcn_seed{seed}"]].mean(axis=1).to_numpy()
                p_test = test_pred[[f"p_rf_seed{seed}", f"p_xgb_seed{seed}", f"p_tcn_seed{seed}"]].mean(axis=1).to_numpy()
            else:
                p_val = val_pred[val_col].to_numpy()
                p_test = test_pred[test_col].to_numpy()

            tau = best_tau(val_pred["y_true"].to_numpy().astype(int), p_val)
            val_metrics = binary_metrics(val_pred["y_true"], p_val, tau=tau)
            test_metrics = binary_metrics(test_pred["y_true"], p_test, tau=tau)
            base_rows_map[label].append(
                {
                    **test_metrics,
                    "prr": prr(val_metrics["recall"], test_metrics["recall"]),
                    "worst_case_recall": low_tail_recall(test_pred["y_true"], p_test, test_pred["run_id"], tau=tau, window=50, quantile=0.05),
                    "instability": instability_score(p_test, test_pred["run_id"], test_pred["phase"]),
                }
            )

    rows = [summarize_seed_rows(base_rows_map[name], name) for name in ["RF (Base)", "XGB (Base)", temporal_label, "Avg. Ensemble (Base)"]]

    utar_row = utar[(utar["dataset"] == "main") & (utar["mode"] == "selective") & (np.isclose(utar["q"], selected_q))].iloc[0]
    utar_seed_main = utar_seed[(utar_seed["dataset"] == "main") & (utar_seed["mode"] == "selective") & (np.isclose(utar_seed["q"], selected_q))]
    utar_auc = (
        fmt(utar_seed_main["roc_auc"].mean(), utar_seed_main["roc_auc"].std(ddof=1))
        if utar_seed_main["roc_auc"].notna().any()
        else "NA"
    )
    rows.append(
        {
            "Method": "UTAR (Proposed)",
            "Ave. F1": fmt(utar_row["f1_mean"], pick_std(utar_row, "f1")),
            "PRR": fmt(utar_row["prr_mean"], pick_std(utar_row, "prr")),
            "Worst-Case Recall (P5)": fmt(utar_row["worst_case_recall_mean"], pick_std(utar_row, "worst_case_recall")),
            "Instability": fmt(utar_row["instability_mean"], pick_std(utar_row, "instability")),
            "Recall": fmt(utar_row["recall_mean"], pick_std(utar_row, "recall")),
            "Precision": fmt(utar_row["precision_mean"], pick_std(utar_row, "precision")),
            "AUC": utar_auc,
            "Inference Time (s)": fmt(utar_row["total_latency_ms_mean"] / 1000.0, pick_std(utar_row, "total_latency_ms") / 1000.0),
            "Inference Time / Sample (ms)": fmt(utar_row["avg_latency_ms_per_sample_mean"], pick_std(utar_row, "avg_latency_ms_per_sample")),
        }
    )
    table2 = pd.DataFrame(rows)
    time_map = {
        "RF (Base)": _base_time_seconds(runtime_summary, "main", RF_COMPONENT),
        "XGB (Base)": _base_time_seconds(runtime_summary, "main", XGB_COMPONENT),
        temporal_label: _base_time_seconds(runtime_summary, "main", temporal_component),
        "Avg. Ensemble (Base)": _ensemble_time_seconds(runtime_summary, "main", temporal_component),
    }
    per_sample_time_map = {
        "RF (Base)": _base_time_per_sample_ms(runtime_summary, "main", RF_COMPONENT),
        "XGB (Base)": _base_time_per_sample_ms(runtime_summary, "main", XGB_COMPONENT),
        temporal_label: _base_time_per_sample_ms(runtime_summary, "main", temporal_component),
        "Avg. Ensemble (Base)": _ensemble_time_per_sample_ms(runtime_summary, "main", temporal_component),
    }
    table2["Inference Time (s)"] = table2["Method"].map(time_map).fillna(table2["Inference Time (s)"] if "Inference Time (s)" in table2.columns else "NA")
    table2["Inference Time / Sample (ms)"] = table2["Method"].map(per_sample_time_map).fillna(
        table2["Inference Time / Sample (ms)"] if "Inference Time / Sample (ms)" in table2.columns else "NA"
    )
    write_csv(METRIC_DIR / "table2_robustness.csv", table2)

    ablation_rows = []
    mode_names = [
        ("selective", "Full Framework (UTAR)"),
        ("no_llm", "w/o Selective LLM"),
        ("ensemble_only", "w/o Gray-Zone (Routing)"),
        ("selective_no_graph", "w/o Graph Smoothing"),
        ("selective_no_filter", "w/o Ensemble Entropy"),
    ]
    for mode, label in mode_names:
        sub = utar[(utar["dataset"] == "main") & (utar["mode"] == mode) & (np.isclose(utar["q"], selected_q))]
        if sub.empty:
            continue
        row = sub.iloc[0]
        ablation_rows.append(
            {
                "Configuration": label,
                "Ave. F1": fmt(row["f1_mean"], pick_std(row, "f1")),
                "PRR": fmt(row["prr_mean"], pick_std(row, "prr")),
                "Worst-Case Recall (P5)": fmt(row["worst_case_recall_mean"], pick_std(row, "worst_case_recall")),
                "Instability": fmt(row["instability_mean"], pick_std(row, "instability")),
                "Inference Cost": fmt(row["cost_usd_mean"], pick_std(row, "cost_usd")),
                "Inference Time (s)": fmt(row["total_latency_ms_mean"] / 1000.0, pick_std(row, "total_latency_ms") / 1000.0),
                "Inference Time / Sample (ms)": fmt(row["avg_latency_ms_per_sample_mean"], pick_std(row, "avg_latency_ms_per_sample")),
            }
        )
    table5 = pd.DataFrame(ablation_rows)
    write_csv(METRIC_DIR / "table5_ablation.csv", table5)

    # raw seed-detail used by appendix and statistical testing if needed later
    write_csv(METRIC_DIR / "table2_utar_seed_detail.csv", utar_seed[(utar_seed["dataset"] == "main") & (utar_seed["mode"] == "selective") & (np.isclose(utar_seed["q"], selected_q))])
    print(table2)
    print(table5)


if __name__ == "__main__":
    main()
