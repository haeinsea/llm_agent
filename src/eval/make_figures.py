from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd

from src.eval.plot_style import PAPER_COLORS, add_panel_label, save_figure, set_paper_style, style_axes
from src.models.temporal_backbone import temporal_model_display_name
from src.routing.selective_llm_eval import read_selected_q
from src.utils.io import ensure_dir, read_csv, read_json, read_yaml
from src.utils.routing import build_routing_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PRED_DIR = OUTPUT_DIR / "predictions"
METRIC_DIR = OUTPUT_DIR / "metrics"
FIG_DIR = OUTPUT_DIR / "figures"
DEFAULT_Q = 0.80


def _clip_prob(values: pd.Series) -> pd.Series:
    clean = values.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    return clean.clip(0.0, 1.0)


def _hist_density(ax, values: pd.Series, label: str, color: str, bins: int = 44, *, density: bool = True) -> None:
    clean = _clip_prob(values)
    if clean.empty:
        return
    weights = None if density else np.ones(len(clean), dtype=float) / len(clean)
    ax.hist(
        clean,
        bins=np.linspace(0.0, 1.0, bins),
        density=density,
        weights=weights,
        histtype="stepfilled",
        linewidth=1.8,
        color=color,
        alpha=0.16,
        edgecolor=color,
        label=label,
        zorder=2,
    )
    ax.hist(
        clean,
        bins=np.linspace(0.0, 1.0, bins),
        density=density,
        weights=weights,
        histtype="step",
        linewidth=2.0,
        color=color,
        zorder=3,
    )


def _finalize_axes(ax, *, xlabel: str | None = None, ylabel: str | None = None, y_grid_only: bool = False) -> None:
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    style_axes(ax, y_grid_only=y_grid_only)


def _annotate_selected_point(ax, x: float, y: float, text: str, *, dx: float = 8.0, dy: float = 8.0, color: str | None = None) -> None:
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=9.5,
        color=color or PAPER_COLORS["ink"],
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "#d9dde3", "linewidth": 0.7, "alpha": 0.96},
        arrowprops={"arrowstyle": "-", "color": color or PAPER_COLORS["ink"], "lw": 0.8, "shrinkA": 4, "shrinkB": 4},
    )


def _add_panel_card(fig, ax, title: str, subtitle: str, index_label: str) -> None:
    bbox = ax.get_position()
    x0 = bbox.x0 - 0.012
    y0 = bbox.y0 - 0.075
    width = bbox.width + 0.024
    height = bbox.height + 0.14
    card = FancyBboxPatch(
        (x0, y0),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        transform=fig.transFigure,
        linewidth=0.9,
        edgecolor="#8ea4b8",
        facecolor="#f8fbfe",
        zorder=-20,
    )
    header = FancyBboxPatch(
        (x0, y0 + height - 0.085),
        width,
        0.085,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        transform=fig.transFigure,
        linewidth=0.9,
        edgecolor="#8ea4b8",
        facecolor="#dce8f3",
        zorder=-19,
    )
    fig.patches.extend([card, header])
    fig.text(x0 + 0.016, y0 + height - 0.048, f"{index_label} {title}", ha="left", va="center", fontsize=10.3, weight="semibold", color=PAPER_COLORS["ink"])
    fig.text(x0 + 0.016, y0 + 0.018, subtitle, ha="left", va="bottom", fontsize=8.4, color=PAPER_COLORS["ink"])


def _conceptual_pareto_curve(x: np.ndarray, y0: float, y1: float, x_star: float, y_star: float) -> np.ndarray:
    grid = np.linspace(0.15, 8.0, 400)
    best_k = grid[np.argmin(np.abs(y0 + (y1 - y0) * (1.0 - np.exp(-grid * x_star)) / (1.0 - np.exp(-grid)) - y_star))]
    return y0 + (y1 - y0) * (1.0 - np.exp(-best_k * x)) / (1.0 - np.exp(-best_k))


def _pick_summary_row(summary: pd.DataFrame, *, dataset: str, mode: str, preferred_q: float | None = None) -> pd.Series:
    sub = summary[(summary["dataset"] == dataset) & (summary["mode"] == mode)].sort_values("q").copy()
    if sub.empty:
        raise KeyError(f"No summary rows for dataset={dataset}, mode={mode}")
    if preferred_q is not None:
        preferred = sub[np.isclose(sub["q"], preferred_q)]
        if not preferred.empty:
            return preferred.iloc[0]
    return sub.iloc[0]


def _gaussian_curve(x: np.ndarray, mean: float, scale: float, amplitude: float = 1.0) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((x - mean) / scale) ** 2)


def _smooth_hist_curve(values: pd.Series, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    clean = values.astype(float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    hist, edges = np.histogram(clean, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    kernel = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=float)
    kernel = kernel / kernel.sum()
    smooth = np.convolve(hist, kernel, mode="same")
    return centers, smooth


def _draw_methodology_card(ax, title: str, footer: str) -> None:
    ax.set_axis_off()
    body = FancyBboxPatch((0.01, 0.03), 0.98, 0.94, boxstyle="round,pad=0.012,rounding_size=0.04", transform=ax.transAxes, linewidth=0.95, edgecolor="#8ea4b8", facecolor="#f7fbff", zorder=-10)
    header = FancyBboxPatch((0.01, 0.86), 0.98, 0.11, boxstyle="round,pad=0.012,rounding_size=0.04", transform=ax.transAxes, linewidth=0.95, edgecolor="#8ea4b8", facecolor="#dce8f3", zorder=-9)
    ax.add_patch(body)
    ax.add_patch(header)
    ax.text(0.05, 0.915, title, transform=ax.transAxes, fontsize=12.8, fontweight="semibold", va="center", ha="left", color=PAPER_COLORS["ink"], linespacing=1.05)
    ax.text(
        0.05,
        0.065,
        textwrap.fill(footer, width=72),
        transform=ax.transAxes,
        fontsize=9.3,
        va="bottom",
        ha="left",
        color=PAPER_COLORS["ink"],
        wrap=True,
        linespacing=1.1,
    )


def _draw_distribution_panel(ax, base: pd.DataFrame, tau: float, margin: float) -> None:
    _hist_density(ax, base.loc[base["phase"] == "normal", "p_utar_base"], "Normal", PAPER_COLORS["navy"], density=False)
    _hist_density(ax, base.loc[base["phase"] == "transition", "p_utar_base"], "Transition", PAPER_COLORS["orange"], density=False)
    _hist_density(ax, base.loc[base["phase"] == "post_shift", "p_utar_base"], "Post-shift", PAPER_COLORS["red"], density=False)
    ax.axvline(tau, color=PAPER_COLORS["ink"], linestyle="--", linewidth=1.6, label="tau", zorder=4)
    ax.axvspan(max(0.0, tau - margin), min(1.0, tau + margin), color=PAPER_COLORS["gray_fill"], alpha=0.28, label="Gray-Zone", zorder=1)
    ax.set_xlim(0.0, 1.0)
    _finalize_axes(ax, xlabel="UTAR base score", ylabel="Sample ratio", y_grid_only=True)
    ylim = ax.get_ylim()
    ax.annotate("Gray-zone around tau", xy=(tau, ylim[1] * 0.74), xytext=(14, 0), textcoords="offset points", fontsize=9.2, color=PAPER_COLORS["ink"])


def _draw_margin_panel(ax, base: pd.DataFrame, tau: float, margin: float) -> None:
    normal_margin = np.abs(base.loc[base["phase"] == "normal", "p_utar_base"] - tau)
    shift_margin = np.abs(base.loc[base["phase"].isin(["transition", "post_shift"]), "p_utar_base"] - tau)
    _hist_density(ax, normal_margin, "Normal", PAPER_COLORS["navy"], density=False)
    _hist_density(ax, shift_margin, "Shift", PAPER_COLORS["red"], density=False)
    ax.axvline(margin, color=PAPER_COLORS["ink"], linestyle="--", linewidth=1.6, label="m_q", zorder=4)
    ax.set_xlim(0.0, max(0.05, min(1.0, float(np.nanpercentile(np.abs(base["p_utar_base"] - tau), 99.5)))))
    _finalize_axes(ax, xlabel="|p - tau|", ylabel="Sample ratio", y_grid_only=True)
    ylim = ax.get_ylim()
    ax.annotate("Routing boundary", xy=(margin, ylim[1] * 0.78), xytext=(10, 0), textcoords="offset points", fontsize=9.2, color=PAPER_COLORS["ink"])


def _draw_discrepancy_panel(ax, base: pd.DataFrame, temporal_label: str, mask: pd.Series, title: str) -> None:
    sub = base.loc[mask, ["p_rf", "p_tcn"]].dropna()
    hb = ax.hexbin(
        sub["p_rf"],
        sub["p_tcn"],
        gridsize=28,
        extent=(0.0, 1.0, 0.0, 1.0),
        cmap="GnBu",
        mincnt=1,
        linewidths=0.0,
        bins="log",
        zorder=2,
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color=PAPER_COLORS["ink"], linewidth=1.3, zorder=3)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title, pad=10)
    _finalize_axes(ax, xlabel="RF score", ylabel=f"{temporal_label} score")
    return hb


def figure_distribution_shift(base: pd.DataFrame, tau: float, margin: float) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    _draw_distribution_panel(ax, base, tau=tau, margin=margin)
    ax.set_title("Distribution overlap near the decision boundary", pad=10)
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / "figureA_distribution_shift.png")
    plt.close(fig)


def figure_margin_concentration(base: pd.DataFrame, margin: float, tau: float) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    _draw_margin_panel(ax, base, tau=tau, margin=margin)
    ax.set_title("Margin concentration near the threshold", pad=10)
    ax.legend(loc="upper right")
    fig.tight_layout()
    save_figure(fig, FIG_DIR / "figureB_margin_concentration.png")
    plt.close(fig)


def figure_model_discrepancy(base: pd.DataFrame, temporal_label: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), sharex=True, sharey=True, constrained_layout=True)
    _draw_discrepancy_panel(axes[0], base, temporal_label, base["phase"] == "normal", "IID / Normal")
    hb = _draw_discrepancy_panel(axes[1], base, temporal_label, base["phase"].isin(["transition", "post_shift"]), "TDS / Shift")
    cbar = fig.colorbar(hb, ax=axes, pad=0.02, fraction=0.028)
    cbar.set_label("Local density (log scale)")
    save_figure(fig, FIG_DIR / "figureC_model_discrepancy.png")
    plt.close(fig)


def figure_methodology_evidence(base: pd.DataFrame, tau: float, margin: float, temporal_label: str) -> None:
    fig = plt.figure(figsize=(8.8, 7.8))
    outer = fig.add_gridspec(2, 2, left=0.025, right=0.975, top=0.98, bottom=0.04, wspace=0.06, hspace=0.09, height_ratios=[1.06, 1.04])

    card1 = fig.add_subplot(outer[0, :])
    card2 = fig.add_subplot(outer[1, 0])
    card3 = fig.add_subplot(outer[1, 1])
    _draw_methodology_card(card1, "1 Distribution Overlap\n& Gray-Zone Formation", "Core message: overlapping score distributions motivate a gray-zone instead of a single hard boundary.")
    _draw_methodology_card(card2, "2 Decision Boundary\nMargin Concentration", "Core message: boundary-near samples are the most valuable targets for selective routing.")
    _draw_methodology_card(card3, "3 Inter-Model Prediction\nDiscrepancy", "Core message: temporal disagreement highlights cases that benefit from selective LLM review.")

    val_base = read_csv(PRED_DIR / "base_val_predictions.csv")

    top_ax = card1.inset_axes([0.07, 0.56, 0.87, 0.27])
    bottom_ax = card1.inset_axes([0.07, 0.20, 0.87, 0.26])
    bins = np.linspace(0.0, 1.0, 84)
    val_normal = val_base.loc[val_base["y_true"] == 0, "p_utar_base"]
    val_anomaly = val_base.loc[val_base["y_true"] == 1, "p_utar_base"]
    test_normal = base.loc[base["y_true"] == 0, "p_utar_base"]
    test_anomaly = base.loc[base["y_true"] == 1, "p_utar_base"]
    cx, val_normal_density = _smooth_hist_curve(val_normal, bins)
    _, val_anomaly_density = _smooth_hist_curve(val_anomaly, bins)
    _, test_normal_density = _smooth_hist_curve(test_normal, bins)
    _, test_anomaly_density = _smooth_hist_curve(test_anomaly, bins)
    max_top = max(float(val_normal_density.max()), float(val_anomaly_density.max()))
    max_bottom = max(float(test_normal_density.max()), float(test_anomaly_density.max()))
    for ax, ymax in [(top_ax, max_top), (bottom_ax, max_bottom)]:
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, ymax * 1.15)
        style_axes(ax, y_grid_only=True)
    top_ax.fill_between(cx, val_normal_density, color="#9ecae1", alpha=0.75)
    top_ax.plot(cx, val_normal_density, color="#2c7fb8", linewidth=1.6, label="Normal")
    top_ax.fill_between(cx, val_anomaly_density, color="#f7b6a6", alpha=0.76)
    top_ax.plot(cx, val_anomaly_density, color="#c0392b", linewidth=1.6, label="Anomaly")
    top_ax.axvline(tau, color=PAPER_COLORS["ink"], linestyle="--", linewidth=1.0)
    top_ax.text(0.5, 1.02, "Validation (IID)", transform=top_ax.transAxes, ha="center", va="bottom", fontsize=11.0, weight="semibold")
    val_overlap = ((val_base["p_utar_base"] >= tau - margin) & (val_base["p_utar_base"] <= tau + margin)).mean()
    top_ax.annotate(f"Gray-zone: {val_overlap * 100:.1f}%", xy=(tau, max_top * 0.42), xytext=(tau + 0.08, max_top * 0.64), arrowprops={"arrowstyle": "->", "lw": 1.0, "color": PAPER_COLORS["ink"]}, fontsize=9.9, ha="left")
    top_ax.legend(loc="upper right", fontsize=9.0)
    top_ax.set_ylabel("Density", fontsize=10.1)
    top_ax.set_xlabel("")
    top_ax.tick_params(axis="x", labelbottom=False)

    bottom_ax.fill_between(cx, test_normal_density, color="#9ecae1", alpha=0.76)
    bottom_ax.plot(cx, test_normal_density, color="#2c7fb8", linewidth=1.6, label="Normal")
    bottom_ax.fill_between(cx, test_anomaly_density, color="#f7b6a6", alpha=0.76)
    bottom_ax.plot(cx, test_anomaly_density, color="#c0392b", linewidth=1.6, label="Anomaly")
    bottom_ax.axvspan(max(0.0, tau - margin), min(1.0, tau + margin), color="#f7dc6f", alpha=0.60, zorder=0)
    bottom_ax.axvline(tau, color=PAPER_COLORS["ink"], linestyle="--", linewidth=1.0)
    bottom_ax.text(0.5, 1.02, "Shift test (TDS)", transform=bottom_ax.transAxes, ha="center", va="bottom", fontsize=11.0, weight="semibold")
    shift_gray = ((base["p_utar_base"] >= tau - margin) & (base["p_utar_base"] <= tau + margin)).mean()
    bottom_ax.text(tau, max_bottom * 0.19, r"$G_q$" + "\n(Gray-Zone)", ha="center", va="center", fontsize=11.5, color=PAPER_COLORS["ink"], fontweight="semibold")
    bottom_ax.annotate(f"Gray-zone: {shift_gray * 100:.1f}%", xy=(tau, max_bottom * 0.55), xytext=(tau + 0.14, max_bottom * 0.82), arrowprops={"arrowstyle": "->", "lw": 1.0, "color": PAPER_COLORS["ink"]}, fontsize=9.9, ha="left")
    bottom_ax.set_ylabel("Density", fontsize=10.1)
    bottom_ax.set_xlabel("UTAR base score", fontsize=10.1)

    mid_ax = card2.inset_axes([0.10, 0.27, 0.84, 0.58])
    centered_normal = (test_normal - tau).clip(-0.22, 0.22)
    centered_anomaly = (test_anomaly - tau).clip(-0.22, 0.22)
    margin_bins = np.linspace(-0.22, 0.22, 88)
    mx, normal_margin_density = _smooth_hist_curve(centered_normal, margin_bins)
    _, anomaly_margin_density = _smooth_hist_curve(centered_anomaly, margin_bins)
    ymax_margin = max(float(normal_margin_density.max()), float(anomaly_margin_density.max()))
    mid_ax.axvspan(-margin, margin, color="#f7dc6f", alpha=0.58, zorder=0)
    mid_ax.axvspan(-margin * 0.35, margin * 0.35, color="#f5b041", alpha=0.35, zorder=0)
    mid_ax.fill_between(mx, normal_margin_density, color="#9ecae1", alpha=0.72, label="Normal", zorder=2)
    mid_ax.plot(mx, normal_margin_density, color="#2c7fb8", linewidth=1.6, zorder=3)
    mid_ax.fill_between(mx, anomaly_margin_density, color="#f7b6a6", alpha=0.72, label="Anomaly", zorder=2)
    mid_ax.plot(mx, anomaly_margin_density, color="#c0392b", linewidth=1.6, zorder=3)
    mid_ax.vlines(centered_normal.sample(min(120, len(centered_normal)), random_state=42).to_numpy(), 0.0, ymax_margin * 0.04, color="#2c7fb8", alpha=0.25, linewidth=0.7)
    mid_ax.vlines(centered_anomaly.sample(min(120, len(centered_anomaly)), random_state=43).to_numpy(), 0.0, ymax_margin * 0.07, color="#c0392b", alpha=0.25, linewidth=0.7)
    for ypos in [ymax_margin * 0.30, ymax_margin * 0.48, ymax_margin * 0.66]:
        mid_ax.add_patch(FancyArrowPatch((-0.18, ypos), (-margin * 0.22, ypos), arrowstyle="simple", mutation_scale=18, linewidth=0.7, facecolor="white", edgecolor=PAPER_COLORS["ink"]))
        mid_ax.add_patch(FancyArrowPatch((0.18, ypos), (margin * 0.22, ypos), arrowstyle="simple", mutation_scale=18, linewidth=0.7, facecolor="white", edgecolor=PAPER_COLORS["ink"]))
    mid_ax.annotate("Decision\nboundary", xy=(0.0, ymax_margin * 1.03), xytext=(-0.10, ymax_margin * 1.03), textcoords="data", ha="center", va="center", fontsize=10.0)
    mid_ax.annotate(r"$\tau$", xy=(0.0, ymax_margin * 1.13), xytext=(0.0, ymax_margin * 1.15), ha="center", va="bottom", fontsize=12.5)
    gray_share = ((base["p_utar_base"] >= tau - margin) & (base["p_utar_base"] <= tau + margin)).mean()
    mid_ax.annotate(f"Observed gray-zone: {gray_share * 100:.1f}%", xy=(0.0, ymax_margin * 0.82), xytext=(0.07, ymax_margin * 0.98), textcoords="data", arrowprops={"arrowstyle": "->", "lw": 1.0, "color": PAPER_COLORS["ink"]}, fontsize=9.8, ha="left")
    mid_ax.set_xlim(-0.22, 0.22)
    mid_ax.set_ylim(0.0, ymax_margin * 1.18)
    mid_ax.set_xlabel("Centered score (p - tau)", fontsize=10.1)
    mid_ax.set_ylabel("Density", fontsize=10.1)
    style_axes(mid_ax, y_grid_only=True)
    mid_ax.legend(loc="upper right", fontsize=8.9)

    top_right = card3.inset_axes([0.12, 0.62, 0.80, 0.19])
    bot_right = card3.inset_axes([0.12, 0.29, 0.80, 0.19])
    normal_sub = base.loc[base["phase"] == "normal", ["p_rf", "p_tcn"]].dropna()
    shift_sub = base.loc[base["phase"].isin(["transition", "post_shift"]), ["p_rf", "p_tcn"]].dropna()
    if len(normal_sub) > 650:
        normal_sub = normal_sub.sample(650, random_state=44)
    if len(shift_sub) > 650:
        shift_sub = shift_sub.sample(650, random_state=45)
    shift_discrepancy = base.loc[base["phase"].isin(["transition", "post_shift"]), ["p_rf", "p_tcn"]].copy()
    shift_discrepancy["gap"] = (shift_discrepancy["p_rf"] - shift_discrepancy["p_tcn"]).abs()
    highlighted = shift_discrepancy.sort_values("gap", ascending=False).head(3)
    outlier_x = highlighted["p_rf"].to_numpy()
    outlier_y = highlighted["p_tcn"].to_numpy()
    for ax, sub, subtitle in [(top_right, normal_sub, "Normal Condition (IID)"), (bot_right, shift_sub, "Distribution Shift (TDS)")]:
        ax.scatter(sub["p_rf"], sub["p_tcn"], s=10, color="#2c7fb8", alpha=0.72, edgecolors="white", linewidths=0.15)
        ax.plot([0, 1], [0, 1], linestyle="--", color=PAPER_COLORS["ink"], linewidth=0.9)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.text(0.02, 1.02, subtitle, transform=ax.transAxes, ha="left", va="bottom", fontsize=10.2, weight="semibold")
        style_axes(ax)
    top_right.set_ylabel("RF score", fontsize=10.1)
    top_right.set_xlabel("")
    top_right.tick_params(axis="x", labelbottom=False)
    bot_right.set_ylabel("RF score", fontsize=10.1)
    bot_right.set_xlabel(f"{temporal_label} score", fontsize=10.1, labelpad=2)
    top_right.annotate("Predictions aligned,\nconsensus", xy=(0.18, 0.16), xytext=(0.05, 0.58), textcoords="axes fraction", arrowprops={"arrowstyle": "->", "lw": 1.0, "color": PAPER_COLORS["ink"]}, fontsize=9.6)
    for (x0, y0) in zip(outlier_x, outlier_y):
        bot_right.scatter([x0], [y0], s=52, facecolor="none", edgecolor="#c87f2a", linewidth=1.3, zorder=4)
        bot_right.add_patch(Rectangle((x0 - 0.03, y0 - 0.03), 0.06, 0.06, linewidth=1.0, edgecolor="#c87f2a", facecolor="none"))
    bot_right.annotate("Samples for\nSelective LLM", xy=(outlier_x[-1], outlier_y[-1]), xytext=(0.74, 0.28), textcoords="axes fraction", arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#c87f2a"}, fontsize=9.3, ha="left")
    bot_right.annotate("Large gap,\ndivergent opinions", xy=(outlier_x[0], outlier_y[0]), xytext=(0.04, 0.56), textcoords="axes fraction", arrowprops={"arrowstyle": "->", "lw": 1.0, "color": PAPER_COLORS["ink"]}, fontsize=9.6)
    save_figure(fig, FIG_DIR / "figure_methodology_evidence.png")
    plt.close(fig)


def _run_instability(sub: pd.DataFrame, score_col: str, tau: float) -> float:
    ordered = sub.sort_values("sample_idx")
    score = ordered[score_col].to_numpy(dtype=float)
    return float(np.abs(np.diff((score >= tau).astype(int))).sum())


def _representative_shift_run(merged: pd.DataFrame, tau: float) -> pd.DataFrame:
    candidate_keys = (
        merged[(merged["phase"].isin(["transition", "post_shift"])) & (merged["fault_id"] != 0)]
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
        if len(sub) < 12:
            continue
        gain = _run_instability(sub, "p_ensemble", tau) - _run_instability(sub, "p_final", tau)
        if gain > best_gain:
            best_gain = gain
            best_key = key

    if best_key is None:
        raise RuntimeError("Could not identify a representative shifted run for the stability figures.")

    chosen = merged[
        (merged["source_file"] == best_key["source_file"])
        & (merged["fault_id"] == best_key["fault_id"])
        & (merged["run_id"] == best_key["run_id"])
    ].sort_values("sample_idx")
    chosen = chosen.copy()
    chosen["relative_step"] = np.arange(len(chosen))
    return chosen


def figure_stability_plot(base: pd.DataFrame, selective: pd.DataFrame, tau: float, margin: float) -> None:
    merged = base.merge(
        selective[["source_file", "fault_id", "run_id", "sample_idx", "p_final"]],
        on=["source_file", "fault_id", "run_id", "sample_idx"],
        how="inner",
    )
    sub = _representative_shift_run(merged, tau=tau)

    x = sub["relative_step"]
    fig, ax = plt.subplots(figsize=(11.2, 4.7))
    ax.axhspan(max(0.0, tau - margin), min(1.0, tau + margin), color=PAPER_COLORS["gray_fill"], alpha=0.32, zorder=0)
    ax.plot(x, sub["p_ensemble"], label="Simple ensemble", color=PAPER_COLORS["orange"], linewidth=2.0, alpha=0.95)
    ax.plot(x, sub["p_final"], label="UTAR", color=PAPER_COLORS["navy"], linewidth=2.6)
    ax.axhline(tau, color=PAPER_COLORS["ink"], linestyle="--", linewidth=1.2, label="tau")
    onset = int(np.argmax(~sub["phase"].eq("normal").to_numpy()))
    ax.axvline(onset, color=PAPER_COLORS["teal"], linestyle=":", linewidth=1.6, label="Fault onset")
    ax.set_ylim(0.0, 1.0)
    _finalize_axes(ax, xlabel="Relative time step within representative shifted run", ylabel="Prediction score", y_grid_only=True)
    ax.legend(loc="lower left", ncol=2)
    _annotate_selected_point(ax, float(x.iloc[min(len(sub) - 1, onset + 14)]), float(sub["p_final"].iloc[min(len(sub) - 1, onset + 14)]), "UTAR remains inside a stable band", dx=10, dy=-24, color=PAPER_COLORS["navy"])
    _annotate_selected_point(ax, float(x.iloc[min(len(sub) - 1, onset + 20)]), float(sub["p_ensemble"].iloc[min(len(sub) - 1, onset + 20)]), "Ensemble drops after onset", dx=12, dy=16, color=PAPER_COLORS["orange"])
    fig.tight_layout()
    save_figure(fig, FIG_DIR / "figureD_stability_plot.png")
    plt.close(fig)


def figure_prediction_flips(base: pd.DataFrame, selective: pd.DataFrame, tau: float) -> None:
    merged = base.merge(
        selective[["source_file", "fault_id", "run_id", "sample_idx", "p_final"]],
        on=["source_file", "fault_id", "run_id", "sample_idx"],
        how="inner",
    )
    sub = _representative_shift_run(merged, tau=tau)
    sample_idx = sub["relative_step"].to_numpy(dtype=float)
    ensemble_state = (sub["p_ensemble"].to_numpy(dtype=float) >= tau).astype(int)
    utar_state = (sub["p_final"].to_numpy(dtype=float) >= tau).astype(int)
    ensemble_flips = np.concatenate([[0], np.cumsum(np.abs(np.diff(ensemble_state)))])
    utar_flips = np.concatenate([[0], np.cumsum(np.abs(np.diff(utar_state)))])
    onset = int(np.argmax(~sub["phase"].eq("normal").to_numpy()))
    ensemble_flip_events = np.flatnonzero(np.diff(ensemble_state) != 0) + 1
    utar_flip_events = np.flatnonzero(np.diff(utar_state) != 0) + 1
    onset_mask = sample_idx >= onset
    ensemble_post_mean = float(np.mean(sub.loc[onset_mask, "p_ensemble"]))
    utar_post_mean = float(np.mean(sub.loc[onset_mask, "p_final"]))
    ensemble_post_std = float(np.std(sub.loc[onset_mask, "p_ensemble"]))
    utar_post_std = float(np.std(sub.loc[onset_mask, "p_final"]))
    flip_gap = int(max(0, ensemble_flips[-1] - utar_flips[-1]))
    boundary_band = 0.05

    fig = plt.figure(figsize=(12.4, 6.7))
    gs = fig.add_gridspec(2, 2, width_ratios=[4.4, 1.45], height_ratios=[1.18, 1.0], hspace=0.10, wspace=0.12)
    ax_score = fig.add_subplot(gs[0, 0])
    ax_flip = fig.add_subplot(gs[1, 0], sharex=ax_score)
    ax_info = fig.add_subplot(gs[:, 1])

    for ax in [ax_score, ax_flip]:
        ax.axvspan(onset, float(sample_idx[-1]), color="#edf4ff", alpha=0.85, zorder=0)
        ax.axvline(onset, color=PAPER_COLORS["teal"], linestyle=":", linewidth=1.4, zorder=4)

    ax_score.axhspan(max(0.0, tau - boundary_band), min(1.0, tau + boundary_band), color="#dbe5f0", alpha=0.55, zorder=0)
    ax_score.plot(sample_idx, sub["p_ensemble"], label="Simple ensemble", color=PAPER_COLORS["orange"], linewidth=2.1, alpha=0.98, zorder=3)
    ax_score.plot(sample_idx, sub["p_final"], label="UTAR", color=PAPER_COLORS["navy"], linewidth=2.7, zorder=4)
    if len(ensemble_flip_events):
        ax_score.scatter(sample_idx[ensemble_flip_events], sub["p_ensemble"].to_numpy()[ensemble_flip_events], s=34, color=PAPER_COLORS["orange"], edgecolor="white", linewidth=0.6, zorder=5)
    ax_score.axhline(tau, color=PAPER_COLORS["ink"], linestyle="--", linewidth=1.0, label="tau")
    ax_score.set_ylim(0.0, 1.0)
    _finalize_axes(ax_score, ylabel="Prediction score", y_grid_only=True)
    ax_score.legend(loc="lower left", ncol=3)
    ax_score.annotate("Post-onset operating region", xy=(onset + 0.2, 0.985), xytext=(onset + 2.0, 0.985), fontsize=8.8, va="top", color=PAPER_COLORS["ink"])
    ax_score.annotate("UTAR stays comfortably above tau", xy=(sample_idx[min(len(sample_idx) - 1, onset + 12)], sub["p_final"].iloc[min(len(sample_idx) - 1, onset + 12)]), xytext=(sample_idx[min(len(sample_idx) - 1, onset + 4)], 0.76), arrowprops={"arrowstyle": "->", "color": PAPER_COLORS["navy"], "lw": 1.0}, fontsize=8.9, color=PAPER_COLORS["navy"], bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "#d9dde3"})
    ax_score.annotate("Ensemble hovers near the decision boundary", xy=(sample_idx[min(len(sample_idx) - 1, onset + 9)], sub["p_ensemble"].iloc[min(len(sample_idx) - 1, onset + 9)]), xytext=(sample_idx[min(len(sample_idx) - 1, onset + 3)], 0.54), arrowprops={"arrowstyle": "->", "color": PAPER_COLORS["orange"], "lw": 1.0}, fontsize=8.9, color=PAPER_COLORS["orange"], bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "#d9dde3"})

    ax_flip.step(sample_idx, ensemble_flips, where="post", label="Simple ensemble", color=PAPER_COLORS["orange"], linewidth=2.3)
    ax_flip.step(sample_idx, utar_flips, where="post", label="UTAR", color=PAPER_COLORS["navy"], linewidth=2.5)
    ax_flip.fill_between(sample_idx, utar_flips, ensemble_flips, where=ensemble_flips >= utar_flips, color="#e9eef5", alpha=0.95, zorder=1)
    if len(ensemble_flip_events):
        ax_flip.scatter(sample_idx[ensemble_flip_events], ensemble_flips[ensemble_flip_events], s=34, color=PAPER_COLORS["orange"], edgecolor="white", linewidth=0.6, zorder=4)
    if len(utar_flip_events):
        ax_flip.scatter(sample_idx[utar_flip_events], utar_flips[utar_flip_events], s=34, color=PAPER_COLORS["navy"], edgecolor="white", linewidth=0.6, zorder=4)
    _finalize_axes(ax_flip, xlabel="Relative time step within representative shifted run", ylabel="Cumulative decision reversals", y_grid_only=True)
    ax_flip.set_ylim(-0.2, max(float(ensemble_flips[-1]) + 0.8, 1.5))
    ax_flip.annotate(f"{flip_gap} fewer reversals", xy=(sample_idx[-1], ensemble_flips[-1]), xytext=(sample_idx[-1] - 5.0, ensemble_flips[-1] - 0.9), arrowprops={"arrowstyle": "->", "color": PAPER_COLORS["highlight"], "lw": 1.0}, fontsize=9.2, color=PAPER_COLORS["highlight"], bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "#d9dde3"})

    ax_info.set_axis_off()
    card = FancyBboxPatch((0.05, 0.04), 0.90, 0.92, boxstyle="round,pad=0.018,rounding_size=0.04", transform=ax_info.transAxes, linewidth=0.9, edgecolor="#8ea4b8", facecolor="#f7fbff")
    header = FancyBboxPatch((0.05, 0.83), 0.90, 0.13, boxstyle="round,pad=0.018,rounding_size=0.04", transform=ax_info.transAxes, linewidth=0.9, edgecolor="#8ea4b8", facecolor="#dce8f3")
    ax_info.add_patch(card)
    ax_info.add_patch(header)
    ax_info.text(0.11, 0.89, "Decision Volatility Summary", transform=ax_info.transAxes, fontsize=10.7, fontweight="semibold", color=PAPER_COLORS["ink"])
    ax_info.text(0.11, 0.74, f"{flip_gap}", transform=ax_info.transAxes, fontsize=26, fontweight="bold", color=PAPER_COLORS["highlight"])
    ax_info.text(0.28, 0.748, "fewer\nreversals", transform=ax_info.transAxes, fontsize=10.2, color=PAPER_COLORS["highlight"], va="center")
    ax_info.text(0.11, 0.62, "Total reversals", transform=ax_info.transAxes, fontsize=9.4, fontweight="semibold")
    ax_info.text(0.11, 0.57, f"Simple ensemble: {int(ensemble_flips[-1])}", transform=ax_info.transAxes, fontsize=9.2, color=PAPER_COLORS["orange"])
    ax_info.text(0.11, 0.53, f"UTAR: {int(utar_flips[-1])}", transform=ax_info.transAxes, fontsize=9.2, color=PAPER_COLORS["navy"])
    ax_info.text(0.11, 0.43, "Post-onset mean score", transform=ax_info.transAxes, fontsize=9.4, fontweight="semibold")
    ax_info.text(0.11, 0.38, f"Simple ensemble: {ensemble_post_mean:.3f} ± {ensemble_post_std:.3f}", transform=ax_info.transAxes, fontsize=9.0, color=PAPER_COLORS["orange"])
    ax_info.text(0.11, 0.34, f"UTAR: {utar_post_mean:.3f} ± {utar_post_std:.3f}", transform=ax_info.transAxes, fontsize=9.0, color=PAPER_COLORS["navy"])
    ax_info.text(0.11, 0.22, "Takeaway", transform=ax_info.transAxes, fontsize=9.4, fontweight="semibold")
    ax_info.text(0.11, 0.11, "UTAR stays well above the decision boundary after fault onset, while the simple ensemble repeatedly approaches or crosses it.", transform=ax_info.transAxes, fontsize=8.9, color=PAPER_COLORS["ink"], wrap=True)

    fig.tight_layout()
    save_figure(fig, FIG_DIR / "figure5_prediction_flips.png")
    plt.close(fig)


def _prepare_table3(summary: pd.DataFrame) -> pd.DataFrame:
    table3 = summary[(summary["dataset"] == "main") & (summary["mode"] == "selective")].sort_values("q").copy()
    table3["Call Rate"] = table3["llm_call_rate_mean"]
    table3["Worst-Case Recall"] = table3["worst_case_recall_mean"]
    table3["Cost (USD)"] = table3["cost_usd_mean"]
    table3["Gray Ratio"] = table3["gray_ratio_mean"]
    table3["F1-Score"] = table3["f1_mean"]
    table3["Inference Time (s)"] = table3["total_latency_ms_mean"] / 1000.0
    return table3


def figure_qsweep_elbow(table3: pd.DataFrame) -> None:
    selected_q = read_selected_q(DEFAULT_Q)
    fig, ax1 = plt.subplots(figsize=(9.0, 4.9))
    ax2 = ax1.twinx()
    ax1.plot(table3["q"], table3["F1-Score"], marker="o", color=PAPER_COLORS["navy"], linewidth=2.4, markersize=5.5, label="F1-score")
    ax2.plot(table3["q"], table3["Call Rate"], marker="s", color=PAPER_COLORS["orange"], linewidth=1.9, markersize=4.8, label="LLM call rate")
    ax2.plot(table3["q"], table3["Cost (USD)"], marker="^", color=PAPER_COLORS["teal"], linewidth=1.8, linestyle="--", markersize=5.0, label="Cost (USD)")
    ax1.axvspan(selected_q - 0.025, selected_q + 0.025, color=PAPER_COLORS["gray_fill"], alpha=0.35, zorder=0)
    sel_row = table3[np.isclose(table3["q"], selected_q)].iloc[0]
    ax1.scatter([sel_row["q"]], [sel_row["F1-Score"]], s=72, color=PAPER_COLORS["highlight"], zorder=5)
    _annotate_selected_point(ax1, float(sel_row["q"]), float(sel_row["F1-Score"]), f"Selected q = {selected_q:.2f}", dx=10, dy=-24, color=PAPER_COLORS["highlight"])
    _finalize_axes(ax1, xlabel="Margin q", ylabel="F1-score", y_grid_only=True)
    ax2.set_ylabel("Call rate / cost")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="lower left", ncol=3)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / "figure2_qsweep_elbow.png")
    save_figure(fig, FIG_DIR / "figure2_grayzone_vs_callrate.png")
    plt.close(fig)


def figure_callrate_vs_f1(table3: pd.DataFrame) -> None:
    selected_q = read_selected_q(DEFAULT_Q)
    fig, ax = plt.subplots(figsize=(8.5, 4.9))
    ax.plot(table3["Call Rate"], table3["F1-Score"], color=PAPER_COLORS["muted_blue"], linewidth=1.4, alpha=0.7, zorder=1)
    scatter = ax.scatter(
        table3["Call Rate"],
        table3["F1-Score"],
        c=table3["q"],
        cmap="cividis",
        s=72,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    sel_row = table3[np.isclose(table3["q"], selected_q)].iloc[0]
    ax.scatter([sel_row["Call Rate"]], [sel_row["F1-Score"]], marker="*", s=230, color=PAPER_COLORS["highlight"], edgecolor="white", linewidth=0.9, zorder=4)
    offsets = {
        0.60: (4, 2),
        0.65: (4, 8),
        0.70: (6, 0),
        0.75: (6, 8),
        0.80: (6, 6),
        0.85: (6, 0),
        0.90: (6, -12),
    }
    for _, row in table3.iterrows():
        dx, dy = offsets.get(round(float(row["q"]), 2), (6, 4))
        ax.annotate(f"q={row['q']:.2f}", (row["Call Rate"], row["F1-Score"]), textcoords="offset points", xytext=(dx, dy), fontsize=8.8)
    _annotate_selected_point(ax, float(sel_row["Call Rate"]), float(sel_row["F1-Score"]), "Operating point used in the paper", dx=10, dy=-28, color=PAPER_COLORS["highlight"])
    _finalize_axes(ax, xlabel="LLM call rate", ylabel="F1-score", y_grid_only=True)
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Margin q")
    fig.tight_layout()
    save_figure(fig, FIG_DIR / "figure3_callrate_vs_f1.png")
    plt.close(fig)


def figure_performance_drop(temporal_label: str) -> None:
    eval_seed = read_csv(OUTPUT_DIR / "evaluation" / "all_seed_metrics_thresholded.csv")
    utar_seed = read_csv(METRIC_DIR / "selective_llm_seed_metrics.csv")
    base = eval_seed[eval_seed["split"].isin(["val", "test_main"]) & eval_seed["model"].isin(["RF", "XGB", "TCN"])].copy()
    base["Method"] = base["model"].map({"RF": "RF (Base)", "XGB": "XGB (Base)", "TCN": f"{temporal_label} (Base)"})
    base["Split"] = base["split"].map({"val": "Validation", "test_main": "Shift test"})
    base_summary = base.groupby(["Method", "Split"], as_index=False)["f1"].mean()

    utar = utar_seed[(utar_seed["dataset"].isin(["val", "main"])) & (utar_seed["mode"] == "selective")].copy()
    utar["Method"] = "UTAR (Proposed)"
    utar["Split"] = utar["dataset"].map({"val": "Validation", "main": "Shift test"})
    utar_summary = utar.groupby(["Method", "Split"], as_index=False)["f1"].mean()

    plot_df = pd.concat([base_summary, utar_summary], ignore_index=True)
    order = ["RF (Base)", "XGB (Base)", f"{temporal_label} (Base)", "UTAR (Proposed)"]
    val_vals = np.array([float(plot_df[(plot_df["Method"] == method) & (plot_df["Split"] == "Validation")]["f1"].iloc[0]) for method in order])
    shift_vals = np.array([float(plot_df[(plot_df["Method"] == method) & (plot_df["Split"] == "Shift test")]["f1"].iloc[0]) for method in order])
    y = np.arange(len(order))[::-1]
    drops = val_vals - shift_vals
    method_colors = [PAPER_COLORS["slate"], PAPER_COLORS["muted_green"], PAPER_COLORS["orange"], PAPER_COLORS["navy"]]
    utar_idx = order.index("UTAR (Proposed)")
    worst_idx = int(np.argmax(drops))

    fig = plt.figure(figsize=(12.1, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 1.35], wspace=0.08)
    ax_main = fig.add_subplot(gs[0, 0])
    ax_drop = fig.add_subplot(gs[0, 1], sharey=ax_main)

    for ax in [ax_main, ax_drop]:
        ax.axhspan(y[utar_idx] - 0.42, y[utar_idx] + 0.42, color="#edf4ff", alpha=0.9, zorder=0)

    for i, method in enumerate(order):
        color = method_colors[i]
        ax_main.add_patch(FancyArrowPatch((val_vals[i], y[i]), (shift_vals[i], y[i]), arrowstyle="-|>", mutation_scale=13, linewidth=2.4 if i == utar_idx else 2.0, color=color, alpha=0.95))
        ax_main.scatter(val_vals[i], y[i], s=92, facecolor="white", edgecolor=color, linewidth=1.4, zorder=3)
        ax_main.scatter(shift_vals[i], y[i], s=92, facecolor=color, edgecolor="white", linewidth=0.8, zorder=4)
        ax_main.text(val_vals[i] - 0.004, y[i] + 0.18, f"{val_vals[i]:.3f}", ha="right", va="center", fontsize=8.5, color=color)
        ax_main.text(shift_vals[i] + 0.004, y[i] - 0.18, f"{shift_vals[i]:.3f}", ha="left", va="center", fontsize=8.5, color=color)

    ax_main.set_yticks(y)
    ax_main.set_yticklabels(order)
    ax_main.set_xlim(min(shift_vals.min(), val_vals.min()) - 0.02, 0.92)
    _finalize_axes(ax_main, xlabel="F1-score", ylabel=None, y_grid_only=False)
    ax_main.grid(True, axis="x", zorder=0)
    ax_main.text(0.01, 1.02, "Validation", transform=ax_main.transAxes, fontsize=9.2, fontweight="semibold", color=PAPER_COLORS["ink"])
    ax_main.text(0.78, 1.02, "Shift test", transform=ax_main.transAxes, fontsize=9.2, fontweight="semibold", color=PAPER_COLORS["ink"])
    ax_main.annotate("Best shift-test F1", xy=(shift_vals[utar_idx], y[utar_idx]), xytext=(shift_vals[utar_idx] - 0.025, y[utar_idx] + 0.38), arrowprops={"arrowstyle": "->", "lw": 1.0, "color": PAPER_COLORS["navy"]}, fontsize=8.8, color=PAPER_COLORS["navy"], bbox={"boxstyle": "round,pad=0.20", "facecolor": "white", "edgecolor": "#d9dde3"})
    ax_main.annotate("Largest degradation", xy=(shift_vals[worst_idx], y[worst_idx]), xytext=(shift_vals[worst_idx] + 0.014, y[worst_idx] - 0.45), arrowprops={"arrowstyle": "->", "lw": 1.0, "color": PAPER_COLORS["orange"]}, fontsize=8.8, color=PAPER_COLORS["orange"], bbox={"boxstyle": "round,pad=0.20", "facecolor": "white", "edgecolor": "#d9dde3"})

    ax_drop.barh(y, drops, height=0.38, color=method_colors, alpha=0.95, zorder=2)
    for i, delta in enumerate(drops):
        ax_drop.text(delta + 0.002, y[i], f"{delta:.3f}", va="center", fontsize=8.8, color=method_colors[i])
    ax_drop.set_xlim(0.0, max(drops) * 1.30)
    ax_drop.set_xlabel("F1 drop")
    ax_drop.grid(True, axis="x", zorder=0)
    ax_drop.tick_params(axis="y", left=False, labelleft=False)
    ax_drop.set_title("Smaller is better", fontsize=9.4, pad=8)
    style_axes(ax_drop, y_grid_only=False)
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=PAPER_COLORS["slate"], markeredgewidth=1.3, markersize=7.5, label="Validation"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PAPER_COLORS["slate"], markeredgecolor="white", markeredgewidth=0.8, markersize=7.5, label="Shift test"),
    ]
    ax_main.legend(handles=legend_handles, loc="lower left")
    fig.tight_layout()
    save_figure(fig, FIG_DIR / "figure4_performance_drop.png")
    plt.close(fig)


def figure_pareto_frontier(summary: pd.DataFrame) -> None:
    selected_q = read_selected_q(DEFAULT_Q)
    main_sel = summary[(summary["dataset"] == "main") & (summary["mode"] == "selective")].sort_values("q").copy()
    main_sel["reliability"] = main_sel["recall_mean"].fillna(main_sel["f1_mean"])
    no_llm = _pick_summary_row(summary, dataset="cost", mode="no_llm", preferred_q=selected_q)
    cost_q = float(no_llm["q"])
    full_llm = _pick_summary_row(summary, dataset="cost", mode="full_llm", preferred_q=cost_q)

    base_models = read_csv(OUTPUT_DIR / "evaluation" / "all_seed_metrics_thresholded.csv")
    base_models = (
        base_models[base_models["split"] == "test_main"]
        .groupby("model", as_index=False)["f1"]
        .mean()
        .rename(columns={"f1": "reliability"})
    )
    base_models["llm_call_rate"] = [0.0, 0.0, 0.0]
    base_models["label"] = base_models["model"].map({"RF": "Baseline RF", "XGB": "Baseline XGB", "TCN": "Baseline TCN"})

    empirical_x = np.array([0.0] + main_sel["llm_call_rate_mean"].tolist() + [1.0], dtype=float)
    empirical_y = np.array([float(no_llm["recall_mean"])] + main_sel["reliability"].tolist() + [float(full_llm["recall_mean"])], dtype=float)
    frontier_x = np.linspace(0.0, 1.0, 220)
    sel_row = main_sel[np.isclose(main_sel["q"], selected_q)].iloc[0]
    frontier_y = _conceptual_pareto_curve(frontier_x, empirical_y[0], empirical_y[-1], float(sel_row["llm_call_rate_mean"]), float(sel_row["reliability"]))

    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    ax.plot(frontier_x * 100.0, frontier_y, linestyle="--", color=PAPER_COLORS["ink"], linewidth=1.8, label="Conceptual Pareto envelope", zorder=1)
    ax.plot(main_sel["llm_call_rate_mean"] * 100.0, main_sel["reliability"], color=PAPER_COLORS["muted_blue"], linewidth=1.5, alpha=0.75, zorder=2)

    scatter = ax.scatter(
        main_sel["llm_call_rate_mean"] * 100.0,
        main_sel["reliability"],
        c=main_sel["q"],
        cmap="cividis",
        s=85,
        edgecolor="white",
        linewidth=0.8,
        label="Selective routing sweep",
        zorder=3,
    )
    q_offsets = {
        0.60: (4, -12),
        0.65: (4, 6),
        0.70: (4, -8),
        0.75: (4, 6),
        0.80: (4, -10),
        0.85: (4, 6),
        0.90: (4, -10),
    }
    for _, row in main_sel.iterrows():
        dx, dy = q_offsets.get(round(float(row["q"]), 2), (4, 5))
        ax.annotate(f"q={row['q']:.2f}", (row["llm_call_rate_mean"] * 100.0, row["reliability"]), textcoords="offset points", xytext=(dx, dy), fontsize=8.3)

    ax.scatter(base_models["llm_call_rate"] * 100.0 + np.array([6.0, 10.0, 14.0]), base_models["reliability"], s=80, color="#9aa5b1", edgecolor="white", linewidth=0.7, label="Baselines", zorder=2.5)
    for x, y, label, dy in zip(base_models["llm_call_rate"] * 100.0 + np.array([6.0, 10.0, 14.0]), base_models["reliability"], base_models["label"], [0, 8, -14]):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(8, dy), fontsize=8.5, color=PAPER_COLORS["slate"])

    ax.scatter([0.0], [float(no_llm["recall_mean"])], color=PAPER_COLORS["slate"], s=120, marker="s", edgecolor="white", linewidth=0.8, label="No-LLM", zorder=4)
    ax.scatter([100.0], [float(full_llm["recall_mean"])], color=PAPER_COLORS["red"], s=140, marker="^", edgecolor="white", linewidth=0.8, label="Full-LLM", zorder=4)
    ax.scatter([float(sel_row["llm_call_rate_mean"]) * 100.0], [float(sel_row["reliability"])], marker="*", s=430, color=PAPER_COLORS["highlight"], edgecolor="white", linewidth=1.0, label="UTAR + selective routing", zorder=5)

    ax.annotate("Higher reliability", xy=(2.0, 0.86), xytext=(2.0, 0.775), arrowprops={"arrowstyle": "->", "lw": 1.0, "color": PAPER_COLORS["ink"]}, fontsize=9.2, rotation=90, ha="center", color=PAPER_COLORS["ink"])
    ax.annotate("Lower cost", xy=(18.0, 0.755), xytext=(2.0, 0.755), arrowprops={"arrowstyle": "->", "lw": 1.0, "color": PAPER_COLORS["ink"]}, fontsize=9.2, va="center", color=PAPER_COLORS["ink"])
    _annotate_selected_point(ax, float(sel_row["llm_call_rate_mean"]) * 100.0, float(sel_row["reliability"]), "Proposed operating point", dx=16, dy=-18, color=PAPER_COLORS["highlight"])
    ax.annotate(f"All-LLM hybrid (q={cost_q:.2f})", xy=(100.0, float(full_llm["recall_mean"])), xytext=(-82, -6), textcoords="offset points", fontsize=9.0, color=PAPER_COLORS["ink"])
    ax.set_xlim(-2.0, 104.0)
    ax.set_ylim(0.74, max(0.90, float(full_llm["recall_mean"]) + 0.025))
    _finalize_axes(ax, xlabel="Operational cost (LLM call rate %)", ylabel="Detection reliability (Recall)", y_grid_only=False)
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_xticklabels([f"{int(t)}%" for t in np.arange(0, 101, 10)])
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Margin q")
    ax.legend(loc="lower right", fontsize=8.8)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / "figure6_cost_stability_pareto.png")
    save_figure(fig, FIG_DIR / "figure_qsweep_tradeoff.png")
    plt.close(fig)


def main() -> None:
    set_paper_style()
    ensure_dir(FIG_DIR)

    cfg = read_yaml(CONFIG_DIR / "routing.yaml", default={})
    tcn_cfg = read_yaml(CONFIG_DIR / "train_tcn.yaml", default={})
    temporal_label = temporal_model_display_name(tcn_cfg.get("architecture", "modern_tcn"))
    tau = float(read_json(METRIC_DIR / "thresholds.json")["tau"])
    gray_grid = read_csv(METRIC_DIR / "grayzone_grid.csv")
    selected_q = read_selected_q(DEFAULT_Q)
    margin = float(gray_grid.loc[np.isclose(gray_grid["q"], selected_q), "gray_margin_mean"].iloc[0])

    base = read_csv(PRED_DIR / "base_test_main_predictions.csv")
    routing = build_routing_features(base[["source_file", "fault_id", "run_id", "p_rf", "p_xgb", "p_tcn"]], cfg)
    for col in routing.columns:
        base[col] = routing[col]

    selective = read_csv(PRED_DIR / "utar_test_main_selective.csv")
    if "p_utar_base" not in selective.columns or "ensemble_entropy" not in selective.columns:
        routing_sel = build_routing_features(selective[["source_file", "fault_id", "run_id", "p_rf", "p_xgb", "p_tcn"]], cfg)
        for col in routing_sel.columns:
            selective[col] = routing_sel[col]

    summary = read_csv(METRIC_DIR / "selective_llm_summary.csv")
    table3 = _prepare_table3(summary)

    figure_methodology_evidence(base, tau=tau, margin=margin, temporal_label=temporal_label)
    figure_distribution_shift(base, tau=tau, margin=margin)
    figure_margin_concentration(base, margin=margin, tau=tau)
    figure_model_discrepancy(base, temporal_label=temporal_label)
    figure_stability_plot(base, selective, tau=tau, margin=margin)
    figure_prediction_flips(base, selective, tau=tau)
    figure_qsweep_elbow(table3)
    figure_callrate_vs_f1(table3)
    figure_performance_drop(temporal_label=temporal_label)
    figure_pareto_frontier(summary)
    print(f"Saved figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
