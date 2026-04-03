from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import read_csv, read_json, write_csv, write_json
from src.utils.metrics import binary_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PRED_DIR = OUTPUT_DIR / "predictions"
METRIC_DIR = OUTPUT_DIR / "metrics"
DEFAULT_Q = 0.80
SELECTED_Q_PATH = METRIC_DIR / "selected_q.json"


def fmt(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def pick_std(row: pd.Series, prefix: str) -> float:
    std_col = f"{prefix}_std"
    return float(row[std_col]) if std_col in row.index else 0.0


def _per_seed_stage_counts(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    shortcut_col = "shortcut_filter" if "shortcut_filter" in df.columns else "xgb_shortcut"
    for seed, group in df.groupby("seed", dropna=False):
        n_total = len(group)
        n_gray = int(group["gray_zone"].sum())
        n_shortcut = int(((group["gray_zone"] == 1) & (group[shortcut_col] == 1)).sum())
        n_llm = int(group["llm_called"].sum())
        n_confident = int((group["gray_zone"] == 0).sum())
        rows.append(
            {
                "seed": seed,
                "n_total": n_total,
                "n_gray": n_gray,
                "n_shortcut": n_shortcut,
                "n_llm": n_llm,
                "n_confident": n_confident,
                "confident_ratio": n_confident / n_total if n_total else 0.0,
                "shortcut_ratio": n_shortcut / n_gray if n_gray else 0.0,
                "llm_ratio": n_llm / n_total if n_total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _gray_zone_seed_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed, group in df.groupby("seed", dropna=False):
        gray = group[group["gray_zone"] == 1].copy()
        if len(gray) == 0:
            rows.append({"seed": seed, "gray_ratio": float((group["gray_zone"] == 1).mean()), "acc": 0.0, "prec": 0.0, "rec": 0.0, "f1": 0.0, "auc": np.nan})
            continue
        m = binary_metrics(gray["y_true"], gray["p_final"], tau=0.5)
        y_hat = (gray["p_final"] >= 0.5).astype(int)
        rows.append(
            {
                "seed": seed,
                "gray_ratio": float((group["gray_zone"] == 1).mean()),
                "acc": float(np.mean(y_hat == gray["y_true"])),
                "prec": m["precision"],
                "rec": m["recall"],
                "f1": m["f1"],
                "auc": m["roc_auc"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--keep-selected-q",
        action="store_true",
        help="Preserve the existing selected_q.json choice if it is still present in the regenerated q-sweep table.",
    )
    args = parser.parse_args()

    summary = read_csv(METRIC_DIR / "selective_llm_summary.csv")
    q_sweep_pred = read_csv(PRED_DIR / "utar_q_sweep.csv")
    q_sweep_no_llm = read_csv(PRED_DIR / "utar_q_sweep_no_llm.csv")

    sel_main = summary[(summary["dataset"] == "main") & (summary["mode"] == "selective")].copy()

    table3_rows = []
    for _, row in sel_main.sort_values("q").iterrows():
        table3_rows.append(
            {
                "Strategy": f"Routing (q={row['q']:.2f})",
                "q": row["q"],
                "Call Rate": fmt(row["llm_call_rate_mean"], pick_std(row, "llm_call_rate")),
                "F1-Score": fmt(row["f1_mean"], pick_std(row, "f1")),
                "Recall": fmt(row["recall_mean"], pick_std(row, "recall")),
                "PRR": fmt(row["prr_mean"], pick_std(row, "prr")),
                "Worst-Case Recall (P5)": fmt(row["worst_case_recall_mean"], pick_std(row, "worst_case_recall")),
                "Gray Ratio": fmt(row["gray_ratio_mean"], pick_std(row, "gray_ratio")),
                "Cost (USD)": fmt(row["cost_usd_mean"], pick_std(row, "cost_usd")),
                "Base Time (s)": fmt(row["base_latency_ms_mean"] / 1000.0, pick_std(row, "base_latency_ms") / 1000.0),
                "Routing Time (s)": fmt((row["routing_feature_latency_ms_mean"] + row["routing_overhead_ms_mean"]) / 1000.0, ((pick_std(row, "routing_feature_latency_ms") ** 2 + pick_std(row, "routing_overhead_ms") ** 2) ** 0.5) / 1000.0),
                "LLM Time (s)": fmt(row["llm_only_latency_ms_mean"] / 1000.0, pick_std(row, "llm_only_latency_ms") / 1000.0),
                "Inference Time (s)": fmt(row["total_latency_ms_mean"] / 1000.0, pick_std(row, "total_latency_ms") / 1000.0),
                "LLM Calls": fmt(row["llm_calls_mean"], pick_std(row, "llm_calls")),
            }
        )
    table3 = pd.DataFrame(table3_rows)
    selected_row = (
        sel_main.sort_values(
            ["f1_mean", "recall_mean", "worst_case_recall_mean", "llm_call_rate_mean", "cost_usd_mean"],
            ascending=[False, False, False, True, True],
        )
        .iloc[0]
    )
    selected_q = float(selected_row["q"])
    if args.keep_selected_q and SELECTED_Q_PATH.exists():
        existing_q = float(read_json(SELECTED_Q_PATH).get("selected_q", selected_q))
        if np.isclose(table3["q"].to_numpy(dtype=float), existing_q).any():
            selected_q = existing_q
    write_csv(METRIC_DIR / "table3_q_sweep.csv", table3)
    write_json(
        SELECTED_Q_PATH,
        {
            "selected_q": selected_q,
            "selection_rule": "max_f1_then_recall_then_worst_case_recall_then_min_call_rate_then_min_cost",
            "source_dataset": "main",
            "n_samples": 4000,
        },
    )

    default_df = q_sweep_pred[np.isclose(q_sweep_pred["q"], selected_q)].copy()
    stage_df = _per_seed_stage_counts(default_df)
    n_total_mean = float(stage_df["n_total"].mean()) if not stage_df.empty else 0.0
    n_total_std = float(stage_df["n_total"].std(ddof=1)) if len(stage_df) > 1 else 0.0
    n_gray_mean = float(stage_df["n_gray"].mean()) if not stage_df.empty else 0.0
    n_gray_std = float(stage_df["n_gray"].std(ddof=1)) if len(stage_df) > 1 else 0.0
    n_shortcut_mean = float(stage_df["n_shortcut"].mean()) if not stage_df.empty else 0.0
    n_shortcut_std = float(stage_df["n_shortcut"].std(ddof=1)) if len(stage_df) > 1 else 0.0
    n_llm_mean = float(stage_df["n_llm"].mean()) if not stage_df.empty else 0.0
    n_llm_std = float(stage_df["n_llm"].std(ddof=1)) if len(stage_df) > 1 else 0.0
    n_confident_mean = float(stage_df["n_confident"].mean()) if not stage_df.empty else 0.0
    n_confident_std = float(stage_df["n_confident"].std(ddof=1)) if len(stage_df) > 1 else 0.0
    confident_ratio_mean = float(stage_df["confident_ratio"].mean()) if not stage_df.empty else 0.0
    confident_ratio_std = float(stage_df["confident_ratio"].std(ddof=1)) if len(stage_df) > 1 else 0.0
    shortcut_ratio_mean = float(stage_df["shortcut_ratio"].mean()) if not stage_df.empty else 0.0
    shortcut_ratio_std = float(stage_df["shortcut_ratio"].std(ddof=1)) if len(stage_df) > 1 else 0.0
    llm_ratio_mean = float(stage_df["llm_ratio"].mean()) if not stage_df.empty else 0.0
    llm_ratio_std = float(stage_df["llm_ratio"].std(ddof=1)) if len(stage_df) > 1 else 0.0
    table6 = pd.DataFrame(
        [
            {"Stage (Module)": "Total Test Samples", "Input Samples": fmt(n_total_mean, n_total_std), "Filtered / Decided": "-", "Remaining (To Next)": fmt(n_total_mean, n_total_std), "Efficiency (Reduction)": "-"},
            {"Stage (Module)": "Confident Zone", "Input Samples": fmt(n_total_mean, n_total_std), "Filtered / Decided": fmt(n_confident_mean, n_confident_std), "Remaining (To Next)": fmt(n_gray_mean, n_gray_std), "Efficiency (Reduction)": fmt(confident_ratio_mean, confident_ratio_std)},
            {"Stage (Module)": "Entropy Shortcut in Gray-Zone", "Input Samples": fmt(n_gray_mean, n_gray_std), "Filtered / Decided": fmt(n_shortcut_mean, n_shortcut_std), "Remaining (To Next)": fmt(n_llm_mean, n_llm_std), "Efficiency (Reduction)": fmt(shortcut_ratio_mean, shortcut_ratio_std)},
            {"Stage (Module)": "Final LLM Calls", "Input Samples": fmt(n_total_mean, n_total_std), "Filtered / Decided": fmt(n_llm_mean, n_llm_std), "Remaining (To Next)": fmt(n_llm_mean, n_llm_std), "Efficiency (Reduction)": fmt(llm_ratio_mean, llm_ratio_std)},
        ]
    )
    write_csv(METRIC_DIR / "table6_flow_efficiency.csv", table6)

    gray_rows = []
    for q in sorted(q_sweep_pred["q"].unique()):
        for method_name, source_df in [("UTAR (No-LLM)", q_sweep_no_llm), ("Selective Routing", q_sweep_pred)]:
            g = source_df[np.isclose(source_df["q"], q)].copy()
            md = _gray_zone_seed_metrics(g)
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
