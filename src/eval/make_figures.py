from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.io import ensure_dir, read_csv, read_json, read_yaml
from src.utils.routing import compute_base_routing_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PRED_DIR = OUTPUT_DIR / "predictions"
METRIC_DIR = OUTPUT_DIR / "metrics"
FIG_DIR = OUTPUT_DIR / "figures"
DEFAULT_Q = 0.80


def plot_density(ax, values: pd.Series, label: str, color: str) -> None:
    clean = values.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return
    ax.hist(clean, bins=40, density=True, histtype="step", linewidth=2, color=color, label=label)


def figure_distribution_shift(base: pd.DataFrame, tau: float, margin: float) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_density(ax, base.loc[base["phase"] == "normal", "p_utar_base"], "Normal", "#1f77b4")
    plot_density(ax, base.loc[base["phase"] == "transition", "p_utar_base"], "Transition", "#ff7f0e")
    plot_density(ax, base.loc[base["phase"] == "post_shift", "p_utar_base"], "Post-shift", "#d62728")
    ax.axvline(tau, color="black", linestyle="--", linewidth=1.5, label="tau")
    ax.axvspan(tau - margin, tau + margin, color="#b0b0b0", alpha=0.25, label="Gray-Zone")
    ax.set_xlabel("Ensemble prediction score")
    ax.set_ylabel("Density")
    ax.set_title("Figure A. Distribution overlap around the decision boundary")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figureA_distribution_shift.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_margin_concentration(base: pd.DataFrame, margin: float, tau: float) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    normal_margin = np.abs(base.loc[base["phase"] == "normal", "p_utar_base"] - tau)
    shift_margin = np.abs(base.loc[base["phase"].isin(["transition", "post_shift"]), "p_utar_base"] - tau)
    plot_density(ax, normal_margin, "Normal", "#1f77b4")
    plot_density(ax, shift_margin, "Shift", "#d62728")
    ax.axvline(margin, color="black", linestyle="--", linewidth=1.5, label="m_q")
    ax.set_xlabel("|p - tau|")
    ax.set_ylabel("Density")
    ax.set_title("Figure B. Margin concentration near the threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figureB_margin_concentration.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_model_discrepancy(base: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    panels = [
        ("IID / Normal", base["phase"] == "normal"),
        ("TDS / Shift", base["phase"].isin(["transition", "post_shift"])),
    ]
    for ax, (title, mask) in zip(axes, panels):
        sub = base.loc[mask, ["p_rf", "p_tcn"]].dropna()
        if len(sub) > 2500:
            sub = sub.sample(2500, random_state=42)
        ax.scatter(sub["p_rf"], sub["p_tcn"], s=8, alpha=0.25, color="#2a6f97")
        ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("RF score")
        ax.set_ylabel("TCN score")
    fig.suptitle("Figure C. Model discrepancy under temporal shift")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figureC_model_discrepancy.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _run_instability(sub: pd.DataFrame, score_col: str, tau: float) -> float:
    ordered = sub.sort_values("sample_idx")
    score = ordered[score_col].to_numpy(dtype=float)
    crossings = np.abs(np.diff((score >= tau).astype(int))).sum()
    return float(crossings)


def figure_stability_plot(base: pd.DataFrame, selective: pd.DataFrame, tau: float, margin: float) -> None:
    merged = base.merge(
        selective[["source_file", "fault_id", "run_id", "sample_idx", "p_final"]],
        on=["source_file", "fault_id", "run_id", "sample_idx"],
        how="inner",
    )
    candidate_keys = (
        merged[merged["phase"].isin(["transition", "post_shift"])]
        [["source_file", "fault_id", "run_id"]]
        .drop_duplicates()
        .to_dict("records")
    )

    best_key = None
    best_gain = -np.inf
    for key in candidate_keys:
        sub = merged[
            (merged["source_file"] == key["source_file"])
            & (merged["fault_id"] == key["fault_id"])
            & (merged["run_id"] == key["run_id"])
        ].copy()
        if len(sub) < 10:
            continue
        gain = _run_instability(sub, "p_utar_base", tau) - _run_instability(sub, "p_final", tau)
        if gain > best_gain:
            best_gain = gain
            best_key = key

    if best_key is None:
        if not candidate_keys:
            raise RuntimeError("Could not identify a representative run for the stability plot.")
        best_key = candidate_keys[0]

    sub = merged[
        (merged["source_file"] == best_key["source_file"])
        & (merged["fault_id"] == best_key["fault_id"])
        & (merged["run_id"] == best_key["run_id"])
    ].sort_values("sample_idx")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(sub["sample_idx"], sub["p_utar_base"], label="UTAR base routing", color="#c1121f", linewidth=1.8)
    ax.plot(sub["sample_idx"], sub["p_final"], label="Proposed UTAR", color="#1d3557", linewidth=2.2)
    ax.axhline(tau, color="black", linestyle="--", linewidth=1.2, label="tau")
    ax.axhspan(tau - margin, tau + margin, color="#bdbdbd", alpha=0.25, label="Gray-Zone")
    ax.axvline(float(sub["onset_step"].iloc[0]), color="#6a994e", linestyle=":", linewidth=1.5, label="Fault onset")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Prediction score")
    ax.set_title("Figure D. Stability comparison under temporal shift")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figureD_stability_plot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_qsweep_tradeoff(table3: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(table3["Call Rate"], table3["Worst-Case Recall"], marker="o", color="#1d3557")
    axes[0].set_xlabel("LLM call rate")
    axes[0].set_ylabel("Worst-case recall")
    axes[0].set_title("Cost-performance Pareto")

    axes[1].plot(table3["Call Rate"], table3["Cost Saving (%)"], marker="o", color="#e76f51")
    axes[1].set_xlabel("LLM call rate")
    axes[1].set_ylabel("Cost saving (%)")
    axes[1].set_title("Operational savings")

    for _, row in table3.iterrows():
        label = f"q={row['q']:.2f}"
        axes[0].annotate(label, (row["Call Rate"], row["Worst-Case Recall"]), textcoords="offset points", xytext=(5, 4), fontsize=8)
        axes[1].annotate(label, (row["Call Rate"], row["Cost Saving (%)"]), textcoords="offset points", xytext=(5, 4), fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_qsweep_tradeoff.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_qsweep_grayratio_callrate(table3: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(table3["q"], table3["Gray Ratio"], marker="o", color="#457b9d", label="Gray-Zone ratio")
    ax.plot(table3["q"], table3["Call Rate"], marker="s", color="#e76f51", label="LLM call rate")
    ax.set_xlabel("Margin-q")
    ax.set_ylabel("Ratio")
    ax.set_title("Figure 2. Gray-Zone Ratio vs LLM Call Rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure2_grayzone_vs_callrate.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_callrate_vs_f1(table3: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(table3["Call Rate"], table3["F1-Score"], marker="o", color="#2a9d8f")
    for _, row in table3.iterrows():
        ax.annotate(f"q={row['q']:.2f}", (row["Call Rate"], row["F1-Score"]), textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.set_xlabel("LLM call rate")
    ax.set_ylabel("F1-score")
    ax.set_title("Figure 3. F1-score vs LLM Call Rate")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure3_callrate_vs_f1.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dir(FIG_DIR)

    cfg = read_yaml(CONFIG_DIR / "routing.yaml", default={})
    tau = float(read_json(METRIC_DIR / "thresholds.json")["tau"])
    gray_grid = read_csv(METRIC_DIR / "grayzone_grid.csv")
    margin = float(gray_grid.loc[np.isclose(gray_grid["q"], DEFAULT_Q), "gray_margin_mean"].iloc[0])

    base = read_csv(PRED_DIR / "base_test_main_predictions.csv")
    base["p_utar_base"] = compute_base_routing_score(base, cfg)
    selective = read_csv(PRED_DIR / "utar_test_main_selective.csv")
    if "p_utar_base" not in selective.columns:
        selective["p_utar_base"] = compute_base_routing_score(selective, cfg)
    summary = read_csv(METRIC_DIR / "selective_llm_summary.csv")
    table3 = summary[(summary["dataset"] == "cost") & (summary["mode"] == "selective")].sort_values("q").copy()
    full = summary[(summary["dataset"] == "cost") & (summary["mode"] == "full_llm")].sort_values("q").copy()
    full_map = dict(zip(full["q"], full["cost_usd_mean"]))
    table3["Call Rate"] = table3["llm_call_rate_mean"]
    table3["Worst-Case Recall"] = table3["worst_case_recall_mean"]
    table3["Cost Saving (%)"] = table3.apply(
        lambda row: 100.0 * (1.0 - row["cost_usd_mean"] / full_map[row["q"]]) if full_map.get(row["q"], 0.0) > 0 else np.nan,
        axis=1,
    )
    table3["Gray Ratio"] = table3["gray_ratio_mean"]
    table3["F1-Score"] = table3["f1_mean"]

    figure_distribution_shift(base, tau=tau, margin=margin)
    figure_margin_concentration(base, margin=margin, tau=tau)
    figure_model_discrepancy(base)
    figure_stability_plot(base, selective, tau=tau, margin=margin)
    figure_qsweep_grayratio_callrate(table3)
    figure_callrate_vs_f1(table3)
    figure_qsweep_tradeoff(table3)
    print(f"Saved figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
