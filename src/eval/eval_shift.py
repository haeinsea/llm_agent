from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import read_csv, write_csv
from src.utils.metrics import binary_metrics, instability_score, prr, worst_case_recall


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
METRIC_DIR = OUTPUT_DIR / "metrics"
EVAL_DIR = OUTPUT_DIR / "evaluation"


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
        "Worst-Case Recall": fmt(df["worst_case_recall"].mean(), df["worst_case_recall"].std(ddof=1) if len(df) > 1 else 0.0),
        "Instability": fmt(df["instability"].mean(), df["instability"].std(ddof=1) if len(df) > 1 else 0.0),
        "Recall": fmt(df["recall"].mean(), df["recall"].std(ddof=1) if len(df) > 1 else 0.0),
        "Precision": fmt(df["precision"].mean(), df["precision"].std(ddof=1) if len(df) > 1 else 0.0),
        "AUC": fmt(df["roc_auc"].mean(), df["roc_auc"].std(ddof=1) if len(df) > 1 else 0.0) if df["roc_auc"].notna().any() else "NA",
    }


def pick_std(row: pd.Series, prefix: str) -> float:
    std_col = f"{prefix}_std"
    return float(row[std_col]) if std_col in row.index else 0.0


def main() -> None:
    val_pred = read_csv(OUTPUT_DIR / "predictions" / "base_val_predictions.csv")
    test_pred = read_csv(OUTPUT_DIR / "predictions" / "base_test_main_predictions.csv")
    utar = read_csv(METRIC_DIR / "selective_llm_summary.csv")
    utar_seed = read_csv(METRIC_DIR / "selective_llm_seed_metrics.csv")

    base_rows_map: dict[str, list[dict]] = {
        "RF (Base)": [],
        "XGB (Base)": [],
        "TCN (Base)": [],
        "Avg. Ensemble (Base)": [],
    }
    for seed in [0, 1, 2, 3, 4]:
        for label, val_col, test_col in [
            ("RF (Base)", f"p_rf_seed{seed}", f"p_rf_seed{seed}"),
            ("XGB (Base)", f"p_xgb_seed{seed}", f"p_xgb_seed{seed}"),
            ("TCN (Base)", f"p_tcn_seed{seed}", f"p_tcn_seed{seed}"),
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
                    "worst_case_recall": worst_case_recall(test_pred["y_true"], p_test, test_pred["run_id"], tau=tau, window=50),
                    "instability": instability_score(p_test, test_pred["run_id"], test_pred["phase"]),
                }
            )

    rows = [summarize_seed_rows(base_rows_map[name], name) for name in ["RF (Base)", "XGB (Base)", "TCN (Base)", "Avg. Ensemble (Base)"]]

    utar_row = utar[(utar["dataset"] == "main") & (utar["mode"] == "selective")].iloc[0]
    utar_seed_main = utar_seed[(utar_seed["dataset"] == "main") & (utar_seed["mode"] == "selective")]
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
            "Worst-Case Recall": fmt(utar_row["worst_case_recall_mean"], pick_std(utar_row, "worst_case_recall")),
            "Instability": fmt(utar_row["instability_mean"], pick_std(utar_row, "instability")),
            "Recall": fmt(utar_row["recall_mean"], pick_std(utar_row, "recall")),
            "Precision": fmt(utar_row["precision_mean"], pick_std(utar_row, "precision")),
            "AUC": utar_auc,
        }
    )
    table2 = pd.DataFrame(rows)
    write_csv(METRIC_DIR / "table2_robustness.csv", table2)

    ablation_rows = []
    mode_names = {
        "selective": "Full Framework",
        "no_llm": "w/o Gray-Zone",
    }
    for mode, label in mode_names.items():
        row = utar[(utar["dataset"] == "main") & (utar["mode"] == mode)].iloc[0]
        ablation_rows.append(
            {
                "Method": label,
                "Ave. F1": fmt(row["f1_mean"], pick_std(row, "f1")),
                "PRR": fmt(row["prr_mean"], pick_std(row, "prr")),
                "Worst-Case Recall": fmt(row["worst_case_recall_mean"], pick_std(row, "worst_case_recall")),
                "Instability": fmt(row["instability_mean"], pick_std(row, "instability")),
                "Recall": fmt(row["recall_mean"], pick_std(row, "recall")),
                "LLM Call Rate": fmt(row["llm_call_rate_mean"], pick_std(row, "llm_call_rate")),
            }
        )
    table5 = pd.DataFrame(ablation_rows)
    write_csv(METRIC_DIR / "table5_ablation.csv", table5)

    # raw seed-detail used by appendix and statistical testing if needed later
    write_csv(METRIC_DIR / "table2_utar_seed_detail.csv", utar_seed[(utar_seed["dataset"] == "main") & (utar_seed["mode"] == "selective")])
    print(table2)
    print(table5)


if __name__ == "__main__":
    main()
