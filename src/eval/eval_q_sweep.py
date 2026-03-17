from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import read_csv, write_csv
from src.utils.metrics import binary_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PRED_DIR = OUTPUT_DIR / "predictions"
METRIC_DIR = OUTPUT_DIR / "metrics"
DEFAULT_Q = 0.80


def fmt(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def pick_std(row: pd.Series, prefix: str) -> float:
    std_col = f"{prefix}_std"
    return float(row[std_col]) if std_col in row.index else 0.0


def main() -> None:
    summary = read_csv(METRIC_DIR / "selective_llm_summary.csv")
    q_sweep_pred = read_csv(PRED_DIR / "utar_q_sweep.csv")
    q_sweep_no_llm = read_csv(PRED_DIR / "utar_q_sweep_no_llm.csv")

    sel_cost = summary[(summary["dataset"] == "cost") & (summary["mode"] == "selective")].copy()
    full_cost = summary[(summary["dataset"] == "cost") & (summary["mode"] == "full_llm")].copy()
    full_cost_map = dict(zip(full_cost["q"], full_cost["cost_usd_mean"]))

    table3_rows = []
    for _, row in sel_cost.sort_values("q").iterrows():
        saving_mean = 100.0 * (1.0 - row["cost_usd_mean"] / full_cost_map[row["q"]]) if full_cost_map.get(row["q"], 0.0) > 0 else np.nan
        table3_rows.append(
            {
                "Strategy": f"Routing (q={row['q']:.2f})",
                "q": row["q"],
                "Call Rate": fmt(row["llm_call_rate_mean"], pick_std(row, "llm_call_rate")),
                "F1-Score": fmt(row["f1_mean"], pick_std(row, "f1")),
                "Recall": fmt(row["recall_mean"], pick_std(row, "recall")),
                "PRR": fmt(row["prr_mean"], pick_std(row, "prr")),
                "Worst-Case Recall": fmt(row["worst_case_recall_mean"], pick_std(row, "worst_case_recall")),
                "Gray Ratio": fmt(row["gray_ratio_mean"], pick_std(row, "gray_ratio")),
                "Cost Saving (%)": f"{saving_mean:.4f}",
                "Inference Time (s)": fmt(row["total_latency_ms_mean"] / 1000.0, pick_std(row, "total_latency_ms") / 1000.0),
                "LLM Calls": fmt(row["llm_calls_mean"], pick_std(row, "llm_calls")),
            }
        )
    table3 = pd.DataFrame(table3_rows)
    write_csv(METRIC_DIR / "table3_q_sweep.csv", table3)

    default_df = q_sweep_pred[np.isclose(q_sweep_pred["q"], DEFAULT_Q)].copy()
    n_total = len(default_df)
    n_gray = int(default_df["gray_zone"].sum())
    n_shortcut = int(((default_df["gray_zone"] == 1) & (default_df["xgb_shortcut"] == 1)).sum())
    n_llm = int(default_df["llm_called"].sum())
    n_confident = int((default_df["gray_zone"] == 0).sum())
    table6 = pd.DataFrame(
        [
            {"Stage (Module)": "Total Test Samples", "Input Samples": fmt(n_total, 0.0), "Filtered / Decided": "-", "Remaining (To Next)": fmt(n_total, 0.0), "Efficiency (Reduction)": "-"},
            {"Stage (Module)": "Confident Zone", "Input Samples": fmt(n_total, 0.0), "Filtered / Decided": fmt(n_confident, 0.0), "Remaining (To Next)": fmt(n_gray, 0.0), "Efficiency (Reduction)": fmt(n_confident / n_total if n_total else 0.0, 0.0)},
            {"Stage (Module)": "XGB-Shortcut in Gray-Zone", "Input Samples": fmt(n_gray, 0.0), "Filtered / Decided": fmt(n_shortcut, 0.0), "Remaining (To Next)": fmt(n_llm, 0.0), "Efficiency (Reduction)": fmt(n_shortcut / n_gray if n_gray else 0.0, 0.0)},
            {"Stage (Module)": "Final LLM Calls", "Input Samples": fmt(n_total, 0.0), "Filtered / Decided": fmt(n_llm, 0.0), "Remaining (To Next)": fmt(n_llm, 0.0), "Efficiency (Reduction)": fmt(n_llm / n_total if n_total else 0.0, 0.0)},
        ]
    )
    write_csv(METRIC_DIR / "table6_flow_efficiency.csv", table6)

    gray_rows = []
    for q in sorted(q_sweep_pred["q"].unique()):
        for method_name, source_df in [("UTAR (No-LLM)", q_sweep_no_llm), ("Selective Routing", q_sweep_pred)]:
            g = source_df[np.isclose(source_df["q"], q)].copy()
            gray = g[g["gray_zone"] == 1].copy()
            if len(gray) == 0:
                md = pd.DataFrame([{"gray_ratio": float((g["gray_zone"] == 1).mean()), "acc": 0.0, "prec": 0.0, "rec": 0.0, "f1": 0.0, "auc": np.nan}])
            else:
                m = binary_metrics(gray["y_true"], gray["p_final"], tau=0.5)
                y_hat = (gray["p_final"] >= 0.5).astype(int)
                md = pd.DataFrame(
                    [
                        {
                            "gray_ratio": float((g["gray_zone"] == 1).mean()),
                            "acc": float(np.mean(y_hat == gray["y_true"])),
                            "prec": m["precision"],
                            "rec": m["recall"],
                            "f1": m["f1"],
                            "auc": m["roc_auc"],
                        }
                    ]
                )
            gray_rows.append(
                {
                    "q": float(q),
                    "Method": method_name,
                    "Gray Ratio": fmt(md["gray_ratio"].mean(), md["gray_ratio"].std(ddof=1) if len(md) > 1 else 0.0),
                    "Acc (Gray)": fmt(md["acc"].mean(), md["acc"].std(ddof=1) if len(md) > 1 else 0.0),
                    "Prec (Gray)": fmt(md["prec"].mean(), md["prec"].std(ddof=1) if len(md) > 1 else 0.0),
                    "Rec (Gray)": fmt(md["rec"].mean(), md["rec"].std(ddof=1) if len(md) > 1 else 0.0),
                    "F1 (Gray)": fmt(md["f1"].mean(), md["f1"].std(ddof=1) if len(md) > 1 else 0.0),
                    "AUC (Gray)": fmt(md["auc"].mean(), md["auc"].std(ddof=1) if len(md) > 1 else 0.0) if md["auc"].notna().any() else "NA",
                }
            )
    table8 = pd.DataFrame(gray_rows)
    write_csv(METRIC_DIR / "table8_grayzone.csv", table8)

    print(table3)
    print(table6)
    print(table8)


if __name__ == "__main__":
    main()
