from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.routing.selective_llm_eval import DEFAULT_Q, read_selected_q
from src.utils.io import read_csv, write_csv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
METRIC_DIR = OUTPUT_DIR / "metrics"


def _fmt_value(mean: float, std: float | None = None) -> str:
    if std is None:
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"


def build_table7_entropy_filter_effect(summary: pd.DataFrame) -> pd.DataFrame:
    selected_q = read_selected_q(DEFAULT_Q)
    dataset_order = ["cost", "main"]
    rows = []
    for dataset in dataset_order:
        selective = summary[(summary["dataset"] == dataset) & (summary["mode"] == "selective")]
        no_filter = summary[(summary["dataset"] == dataset) & (summary["mode"] == "selective_no_filter")]
        selective = selective[np.isclose(selective["q"], selected_q)]
        no_filter = no_filter[np.isclose(no_filter["q"], selected_q)]
        if selective.empty or no_filter.empty:
            continue
        sel = selective.iloc[0]
        raw = no_filter.iloc[0]
        call_delta = float(raw["llm_call_rate_mean"] - sel["llm_call_rate_mean"])
        cost_delta = float(raw["cost_usd_mean"] - sel["cost_usd_mean"])
        rows.append(
            {
                "Dataset": dataset,
                "Selective UTAR Call Rate": _fmt_value(sel["llm_call_rate_mean"], sel.get("llm_call_rate_std", 0.0)),
                "w/o Entropy Filter Call Rate": _fmt_value(raw["llm_call_rate_mean"], raw.get("llm_call_rate_std", 0.0)),
                "Call Rate Reduction (p)": f"{call_delta:.4f}",
                "Selective Cost (USD)": _fmt_value(sel["cost_usd_mean"], sel.get("cost_usd_std", 0.0)),
                "w/o Entropy Filter Cost (USD)": _fmt_value(raw["cost_usd_mean"], raw.get("cost_usd_std", 0.0)),
                "Cost Reduction (USD)": f"{cost_delta:.4f}",
                "Interpretation": "Entropy filter removes low-risk gray-zone samples before LLM escalation.",
            }
        )
    return pd.DataFrame(rows)


def build_table9_dashboard_modules() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Feature": "Decision Source Transparency",
                "Description": "Displays whether the final decision came from direct UTAR routing, entropy shortcut, or LLM escalation.",
                "Expected Benefit": "Improves auditability and operator trust.",
            },
            {
                "Feature": "Structural Root-Cause Ranking",
                "Description": "Lists GraphAD+ top sensors and topology-aware neighbors to support traceable maintenance decisions.",
                "Expected Benefit": "Reduces diagnosis search time and prioritizes inspections.",
            },
            {
                "Feature": "Contextual LLM Explanation",
                "Description": "Summarizes why ambiguous gray-zone samples were escalated and how the final probability was refined.",
                "Expected Benefit": "Improves actionability for non-expert operators.",
            },
            {
                "Feature": "Interactive Q&A / Drill-down",
                "Description": "Supports follow-up inspection of score traces, routing metrics, and candidate-variable context.",
                "Expected Benefit": "Turns the framework into an operator-facing DSS rather than a raw score feed.",
            },
        ]
    )


def build_table10_operational_strategy(summary: pd.DataFrame) -> pd.DataFrame:
    cost = summary[summary["dataset"] == "cost"].copy()
    selected_q = read_selected_q(DEFAULT_Q)
    rows = []

    no_llm = cost[(cost["mode"] == "no_llm") & np.isclose(cost["q"], selected_q)]
    if not no_llm.empty:
        row = no_llm.iloc[0]
        rows.append(
            {
                "Strategy": "No-LLM baseline",
                "Invocation Structure": "All samples decided by UTAR base routing only",
                "Call Volume Summary": f"Call rate {row['llm_call_rate_mean']:.4f}",
                "Inference Time (s)": f"{row['total_latency_ms_mean'] / 1000.0:.4f}",
                "Operational Significance": "Lowest cost and latency baseline, but no ambiguity escalation.",
            }
        )

    selective = cost[(cost["mode"] == "selective") & np.isclose(cost["q"], selected_q)].sort_values("q")
    if not selective.empty:
        row = selective.iloc[0]
        rows.append(
            {
                "Strategy": f"Selective Routing (q={row['q']:.2f})",
                "Invocation Structure": "Gray-zone routing with entropy filter and selective LLM escalation",
                "Call Volume Summary": f"Call rate {row['llm_call_rate_mean']:.4f}, cost ${row['cost_usd_mean']:.4f}",
                "Inference Time (s)": f"{row['total_latency_ms_mean'] / 1000.0:.4f}",
                "Operational Significance": "Selected q from the 4,000-sample routing set, then fixed for the final 400-sample comparison.",
            }
        )

    full_llm = cost[(cost["mode"] == "full_llm") & np.isclose(cost["q"], selected_q)]
    if not full_llm.empty:
        row = full_llm.iloc[0]
        rows.append(
            {
                "Strategy": "Full-LLM upper bound",
                "Invocation Structure": "Every gray-zone candidate is handled through the LLM path",
                "Call Volume Summary": f"Call rate {row['llm_call_rate_mean']:.4f}, cost ${row['cost_usd_mean']:.4f}",
                "Inference Time (s)": f"{row['total_latency_ms_mean'] / 1000.0:.4f}",
                "Operational Significance": "Upper-bound quality/cost reference for practical deployment trade-offs.",
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    summary = read_csv(METRIC_DIR / "selective_llm_summary.csv")
    write_csv(METRIC_DIR / "table7_entropy_filter_effect.csv", build_table7_entropy_filter_effect(summary))
    write_csv(METRIC_DIR / "table9_dashboard_modules.csv", build_table9_dashboard_modules())
    write_csv(METRIC_DIR / "table10_operational_strategy.csv", build_table10_operational_strategy(summary))
    print("Saved Table 7, Table 9, and Table 10 to outputs/metrics")


if __name__ == "__main__":
    main()
