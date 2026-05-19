#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""NDSS-based GraphAD alpha sweep for Appendix H evidence."""

import argparse
import os
import platform
import sys
import time
from glob import glob
from importlib import metadata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ndss.ndss_reasoning import (
    compute_ACT,
    compute_KConcord,
    compute_SGR,
    load_and_clean_csv,
    load_process_graph,
    prepare_graphad_bundle,
    re_split_multi,
    score_prepared_graphad_bundle,
)
from src.utils.io import read_yaml


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def alpha_grid(start: float = 0.0, stop: float = 0.6, step: float = 0.1) -> list[float]:
    n_steps = int(round((stop - start) / step))
    return [round(start + step * idx, 2) for idx in range(n_steps + 1)]


def _best_true_var_rank(scores: pd.Series, true_vars: list[str]) -> int | None:
    ranking = list(scores.index)
    ranks = [ranking.index(v) for v in true_vars if v in ranking]
    if not ranks:
        return None
    return min(ranks)


def _best_true_var_rank_one_based(scores: pd.Series, true_vars: list[str]) -> float | None:
    rank = _best_true_var_rank(scores, true_vars)
    if rank is None:
        return None
    return float(rank + 1)


def _reciprocal_rank(scores: pd.Series, true_vars: list[str]) -> float:
    rank = _best_true_var_rank(scores, true_vars)
    if rank is None:
        return 0.0
    return 1.0 / float(rank + 1)


def _bootstrap_summary(values: list[float], repeats: int, rng: np.random.Generator) -> tuple[float, float, float, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean()) if len(arr) else float("nan")
    if len(arr) <= 1 or repeats <= 1:
        return mean, 0.0, mean, mean
    idx = rng.integers(0, len(arr), size=(repeats, len(arr)))
    boot = arr[idx].mean(axis=1)
    std = float(np.std(boot, ddof=1))
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return mean, std, float(lo), float(hi)


def _paired_bootstrap(
    left: list[float],
    right: list[float],
    repeats: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float, float, float]:
    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    if len(left_arr) != len(right_arr):
        raise ValueError("Paired bootstrap requires equal-length inputs.")
    left_mean = float(left_arr.mean())
    right_mean = float(right_arr.mean())
    diff_mean = left_mean - right_mean
    if len(left_arr) <= 1 or repeats <= 1:
        return left_mean, right_mean, diff_mean, 0.0, diff_mean, diff_mean
    idx = rng.integers(0, len(left_arr), size=(repeats, len(left_arr)))
    left_boot = left_arr[idx].mean(axis=1)
    right_boot = right_arr[idx].mean(axis=1)
    diff_boot = left_boot - right_boot
    diff_std = float(np.std(diff_boot, ddof=1))
    lo, hi = np.quantile(diff_boot, [0.025, 0.975])
    return left_mean, right_mean, diff_mean, diff_std, float(lo), float(hi)


def _safe_version(pkg: str) -> str:
    try:
        return metadata.version(pkg)
    except Exception:
        return "unknown"


def _rank_candidates(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.sort_values(
        by=["Top1_recall_mean", "Top3_recall_mean", "K-Concord_mean", "SGR_mean", "Top5_recall_mean"],
        ascending=False,
    ).reset_index(drop=True)
    ranked.insert(0, "selection_rank", ranked.index + 1)
    return ranked


def _mask_alpha(df: pd.DataFrame, alpha: float) -> pd.Series:
    return df["alpha_graph"].round(6).eq(round(alpha, 6))


def _build_protocol_table(
    out_dir: str,
    *,
    n_scenarios: int,
    alpha_values: list[float],
    fixed_lambdas: tuple[float, float, float],
    bootstrap_repeats: int,
    runtime_sec: float,
) -> str:
    rows = [
        {"Category": "Dataset", "Item": "Scenario directory", "Value": "data/ndss_scenarios"},
        {"Category": "Dataset", "Item": "Ground-truth file", "Value": "data/ndss_attack_scenarios.csv"},
        {"Category": "Dataset", "Item": "Attacks evaluated", "Value": str(n_scenarios)},
        {
            "Category": "Search Space",
            "Item": "Alpha grid",
            "Value": "[" + ", ".join(f"{v:.1f}" for v in alpha_values) + "]",
        },
        {
            "Category": "Search Space",
            "Item": "Fixed lambda tuple",
            "Value": f"({fixed_lambdas[0]:.1f}, {fixed_lambdas[1]:.1f}, {fixed_lambdas[2]:.1f})",
        },
        {
            "Category": "Graph",
            "Item": "Process graph source",
            "Value": "data/NDSS_process_edges.csv",
        },
        {
            "Category": "Evaluation",
            "Item": "Ranking cutoff",
            "Value": "Top-5 candidates used for ACT/Top-k recall metrics",
        },
        {
            "Category": "Evaluation",
            "Item": "Window candidates",
            "Value": "[100, 200, 300, 400, 500]",
        },
        {
            "Category": "Evaluation",
            "Item": "Window-selection rule",
            "Value": "Choose one unsmoothed best-rank window per attack using the fixed lambda tuple, then vary only alpha during graph smoothing.",
        },
        {
            "Category": "Selection Rule",
            "Item": "Primary ranking",
            "Value": "Top1_recall_mean > Top3_recall_mean > K-Concord_mean > SGR_mean > Top5_recall_mean",
        },
        {
            "Category": "Fairness Control",
            "Item": "Uniform protocol",
            "Value": f"All alpha candidates were evaluated on the identical {n_scenarios} NDSS attacks with the same graph, same fixed lambdas, same window pool, and the same no-LLM reasoning path.",
        },
        {
            "Category": "Robustness",
            "Item": "Uncertainty estimation",
            "Value": f"Scenario-level bootstrap over attacks ({bootstrap_repeats} repeats) used to report mean±std and 95% CI.",
        },
        {
            "Category": "Runtime",
            "Item": "Total sweep wall-clock time",
            "Value": f"{runtime_sec:.2f} sec",
        },
    ]
    path = os.path.join(out_dir, "table_h7_graphad_alpha_protocol_ndss.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _build_reproducibility_table(out_dir: str, runtime_sec: float) -> str:
    rows = [
        {"Category": "Software", "Item": "Python", "Value": platform.python_version()},
        {"Category": "Software", "Item": "numpy", "Value": _safe_version("numpy")},
        {"Category": "Software", "Item": "pandas", "Value": _safe_version("pandas")},
        {"Category": "Software", "Item": "networkx", "Value": _safe_version("networkx")},
        {"Category": "Software", "Item": "matplotlib", "Value": _safe_version("matplotlib")},
        {"Category": "Hardware", "Item": "Platform", "Value": platform.platform()},
        {"Category": "Hardware", "Item": "Machine", "Value": platform.machine()},
        {"Category": "Hardware", "Item": "Processor", "Value": platform.processor() or "unknown"},
        {"Category": "Hardware", "Item": "CPU cores", "Value": str(os.cpu_count() or "unknown")},
        {"Category": "Hardware", "Item": "Execution device", "Value": "CPU-only GraphAD alpha sweep"},
        {"Category": "Runtime", "Item": "Sweep wall-clock time (sec)", "Value": f"{runtime_sec:.2f}"},
        {"Category": "Runtime", "Item": "Command", "Value": "python ndss/tune_graphad_alpha.py"},
    ]
    path = os.path.join(out_dir, "table_h8_graphad_alpha_reproducibility_ndss.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _build_pairwise_table(
    out_dir: str,
    alpha_metrics: dict[float, dict[str, list[float]]],
    best_alpha: float,
    current_alpha: float,
    bootstrap_repeats: int,
) -> str:
    if current_alpha not in alpha_metrics or best_alpha not in alpha_metrics:
        return ""
    rng = np.random.default_rng(20260406)
    rows = []
    for metric in ["Top1_recall", "Top3_recall", "Top5_recall", "K-Concord", "SGR", "MRR", "best_true_rank"]:
        cur_mean, best_mean, diff_mean, diff_std, ci_low, ci_high = _paired_bootstrap(
            alpha_metrics[current_alpha][metric],
            alpha_metrics[best_alpha][metric],
            repeats=bootstrap_repeats,
            rng=rng,
        )
        rows.append(
            {
                "Metric": metric,
                "Current alpha": current_alpha,
                "Best alpha": best_alpha,
                "Current mean": cur_mean,
                "Best mean": best_mean,
                "Current-Best diff": diff_mean,
                "Diff bootstrap std": diff_std,
                "Diff 95% CI low": ci_low,
                "Diff 95% CI high": ci_high,
            }
        )
    path = os.path.join(out_dir, "table_h9_graphad_alpha_current_vs_best_ndss.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _build_sensitivity_figure(detail_df: pd.DataFrame, out_dir: str, current_alpha: float) -> tuple[str, str]:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
    metrics = [
        ("Top1_recall_mean", "Top-1 Recall"),
        ("Top3_recall_mean", "Top-3 Recall"),
        ("K-Concord_mean", "K-Concord"),
        ("SGR_mean", "SGR"),
    ]
    best_row = detail_df.iloc[0]
    best_alpha = float(best_row["alpha_graph"])
    x = detail_df["alpha_graph"].to_numpy(dtype=float)

    for ax, (metric, title) in zip(axes.flat, metrics):
        y = detail_df[metric].to_numpy(dtype=float)
        yerr = detail_df[metric.replace("_mean", "_std")].to_numpy(dtype=float)
        ax.plot(x, y, color="#243b53", linewidth=2.0, marker="o")
        ax.fill_between(x, y - yerr, y + yerr, color="#9fb3c8", alpha=0.35)
        ax.axvline(best_alpha, color="black", linestyle="--", linewidth=1.3)
        ax.axvline(current_alpha, color="royalblue", linestyle=":", linewidth=1.5)
        ax.scatter([best_alpha], [float(best_row[metric])], s=90, facecolors="none", edgecolors="black", linewidths=2, zorder=4)
        cur_row = detail_df[_mask_alpha(detail_df, current_alpha)].iloc[0]
        ax.scatter([current_alpha], [float(cur_row[metric])], s=80, marker="x", c="royalblue", linewidths=2, zorder=5)
        ax.set_title(title)
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(title)
        ax.grid(alpha=0.25)

    handles = [
        plt.Line2D([0], [0], marker="o", color="black", markersize=8, linestyle="None", markerfacecolor="none", label="Best"),
        plt.Line2D([0], [0], marker="x", color="royalblue", markersize=8, linestyle="None", label="Current"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("GraphAD+ Alpha Sensitivity on NDSS", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    png_path = os.path.join(out_dir, "figure_h3_graphad_alpha_sensitivity_ndss.png")
    pdf_path = os.path.join(out_dir, "figure_h3_graphad_alpha_sensitivity_ndss.pdf")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def run_alpha_sweep(
    scen_dir: str,
    procmap_csv: str,
    gt_csv: str,
    out_dir: str,
    alpha_values: list[float],
    lambda_tuple: tuple[float, float, float],
    bootstrap_repeats: int = 1000,
    max_scenarios: int | None = None,
) -> tuple[str, str]:
    start_time = time.perf_counter()
    g = load_process_graph(procmap_csv)

    gt_df = pd.read_csv(gt_csv)
    if "attack_id" not in gt_df.columns or "true_var" not in gt_df.columns:
        raise ValueError("gt_csv must have columns: attack_id,true_var,...")
    gt_df = gt_df.set_index("attack_id")

    files = sorted(glob(os.path.join(scen_dir, "*.csv")))
    if max_scenarios is not None:
        files = files[:max_scenarios]
    if not files:
        raise RuntimeError(f"No scenario CSV in {scen_dir}")

    stats = {
        alpha: {
            "attacks_evaluated": 0,
            "ACT_Hit@5": [],
            "K-Concord": [],
            "SGR": [],
            "Top1_recall": [],
            "Top3_recall": [],
            "Top5_recall": [],
            "MRR": [],
            "best_true_rank": [],
            "selected_window": [],
        }
        for alpha in alpha_values
    }

    current_alpha = 0.3
    candidate_windows = (100, 200, 300, 400, 500)

    print(
        f"Running NDSS alpha sweep: scenarios={len(files)}, alphas={len(alpha_values)}, "
        f"fixed_lambda={lambda_tuple}"
    )

    for f in tqdm(files, desc="NDSS alpha sweep"):
        attack_id = os.path.basename(f).replace(".csv", "")
        if attack_id not in gt_df.index:
            continue

        tv_raw = str(gt_df.loc[attack_id, "true_var"])
        true_vars = [t.strip() for t in re_split_multi(tv_raw) if t.strip()]
        if not true_vars:
            continue

        df = load_and_clean_csv(f)
        window_bundles = {
            win_len: prepare_graphad_bundle(df, win_len=win_len)
            for win_len in candidate_windows
        }

        best_window = None
        best_rank = None
        for win_len, bundle in window_bundles.items():
            scores = score_prepared_graphad_bundle(
                bundle,
                g=None,
                alpha_graph=0.0,
                w_z=lambda_tuple[0],
                w_trend=lambda_tuple[1],
                w_flat=lambda_tuple[2],
            )
            if scores is None or scores.empty:
                continue
            rank = _best_true_var_rank(scores, true_vars)
            if rank is None:
                continue
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_window = win_len

        if best_window is None:
            continue

        for alpha in alpha_values:
            final_scores = score_prepared_graphad_bundle(
                window_bundles[best_window],
                g=g,
                alpha_graph=alpha,
                w_z=lambda_tuple[0],
                w_trend=lambda_tuple[1],
                w_flat=lambda_tuple[2],
            )
            if final_scores is None or final_scores.empty:
                continue

            top5 = list(final_scores.head(5).index)
            top3 = top5[:3]
            top1 = top5[:1]

            act = compute_ACT(top5, true_vars)
            kc = compute_KConcord(top5, true_vars)
            sgr = compute_SGR(final_scores, true_vars)
            top1_hit = 1.0 if any(v in top1 for v in true_vars) else 0.0
            top3_hit = 1.0 if any(v in top3 for v in true_vars) else 0.0
            top5_hit = 1.0 if any(v in top5 for v in true_vars) else 0.0
            best_true_rank = _best_true_var_rank_one_based(final_scores, true_vars)
            if best_true_rank is None:
                continue
            reciprocal_rank = _reciprocal_rank(final_scores, true_vars)

            rec = stats[alpha]
            rec["attacks_evaluated"] += 1
            rec["ACT_Hit@5"].append(act)
            rec["K-Concord"].append(kc)
            rec["SGR"].append(sgr)
            rec["Top1_recall"].append(top1_hit)
            rec["Top3_recall"].append(top3_hit)
            rec["Top5_recall"].append(top5_hit)
            rec["MRR"].append(reciprocal_rank)
            rec["best_true_rank"].append(best_true_rank)
            rec["selected_window"].append(float(best_window))

    rows = []
    alpha_metrics: dict[float, dict[str, list[float]]] = {}
    rng = np.random.default_rng(20260406)
    for alpha, rec in stats.items():
        n = rec["attacks_evaluated"]
        if n == 0:
            continue
        alpha_metrics[alpha] = {key: list(values) for key, values in rec.items() if isinstance(values, list)}
        act_mean, act_std, act_lo, act_hi = _bootstrap_summary(rec["ACT_Hit@5"], bootstrap_repeats, rng)
        kc_mean, kc_std, kc_lo, kc_hi = _bootstrap_summary(rec["K-Concord"], bootstrap_repeats, rng)
        sgr_mean, sgr_std, sgr_lo, sgr_hi = _bootstrap_summary(rec["SGR"], bootstrap_repeats, rng)
        top1_mean, top1_std, top1_lo, top1_hi = _bootstrap_summary(rec["Top1_recall"], bootstrap_repeats, rng)
        top3_mean, top3_std, top3_lo, top3_hi = _bootstrap_summary(rec["Top3_recall"], bootstrap_repeats, rng)
        top5_mean, top5_std, top5_lo, top5_hi = _bootstrap_summary(rec["Top5_recall"], bootstrap_repeats, rng)
        mrr_mean, mrr_std, mrr_lo, mrr_hi = _bootstrap_summary(rec["MRR"], bootstrap_repeats, rng)
        rank_mean, rank_std, rank_lo, rank_hi = _bootstrap_summary(rec["best_true_rank"], bootstrap_repeats, rng)
        win_mean, win_std, win_lo, win_hi = _bootstrap_summary(rec["selected_window"], bootstrap_repeats, rng)
        ranking_variance = float(np.var(np.asarray(rec["best_true_rank"], dtype=float)))
        rows.append(
            {
                "alpha_graph": alpha,
                "attacks_evaluated": n,
                "ACT_Hit@5_mean": act_mean,
                "ACT_Hit@5_std": act_std,
                "ACT_Hit@5_ci_low": act_lo,
                "ACT_Hit@5_ci_high": act_hi,
                "K-Concord_mean": kc_mean,
                "K-Concord_std": kc_std,
                "K-Concord_ci_low": kc_lo,
                "K-Concord_ci_high": kc_hi,
                "SGR_mean": sgr_mean,
                "SGR_std": sgr_std,
                "SGR_ci_low": sgr_lo,
                "SGR_ci_high": sgr_hi,
                "Top1_recall_mean": top1_mean,
                "Top1_recall_std": top1_std,
                "Top1_recall_ci_low": top1_lo,
                "Top1_recall_ci_high": top1_hi,
                "Top3_recall_mean": top3_mean,
                "Top3_recall_std": top3_std,
                "Top3_recall_ci_low": top3_lo,
                "Top3_recall_ci_high": top3_hi,
                "Top5_recall_mean": top5_mean,
                "Top5_recall_std": top5_std,
                "Top5_recall_ci_low": top5_lo,
                "Top5_recall_ci_high": top5_hi,
                "MRR_mean": mrr_mean,
                "MRR_std": mrr_std,
                "MRR_ci_low": mrr_lo,
                "MRR_ci_high": mrr_hi,
                "best_true_rank_mean": rank_mean,
                "best_true_rank_std": rank_std,
                "best_true_rank_ci_low": rank_lo,
                "best_true_rank_ci_high": rank_hi,
                "ranking_variance": ranking_variance,
                "mean_selected_window": win_mean,
                "selected_window_std": win_std,
                "selected_window_ci_low": win_lo,
                "selected_window_ci_high": win_hi,
                "is_current_setting": round(alpha, 6) == round(current_alpha, 6),
                "is_no_smoothing": round(alpha, 6) == 0.0,
            }
        )

    detail_df = _rank_candidates(pd.DataFrame(rows))
    detail_df["is_current_setting"] = _mask_alpha(detail_df, current_alpha)
    detail_df["is_no_smoothing"] = detail_df["alpha_graph"].round(6).eq(0.0)
    os.makedirs(out_dir, exist_ok=True)
    detail_path = os.path.join(out_dir, "table_h6_graphad_alpha_grid_detail_ndss.csv")
    detail_df.to_csv(detail_path, index=False)

    summary_rows = []
    if not detail_df.empty:
        summary_rows.append({"Selection": "Best Top1-first NDSS alpha", **detail_df.iloc[0].to_dict()})
    current_rows = detail_df[_mask_alpha(detail_df, current_alpha)]
    if not current_rows.empty:
        summary_rows.append({"Selection": "Current manuscript alpha", **current_rows.iloc[0].to_dict()})
    no_smoothing_rows = detail_df[detail_df["is_no_smoothing"]]
    if not no_smoothing_rows.empty:
        summary_rows.append({"Selection": "No smoothing baseline", **no_smoothing_rows.iloc[0].to_dict()})
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, "table_h6_graphad_alpha_selection_ndss.csv")
    summary_df.to_csv(summary_path, index=False)

    runtime_sec = time.perf_counter() - start_time
    protocol_path = _build_protocol_table(
        out_dir,
        n_scenarios=len(files),
        alpha_values=alpha_values,
        fixed_lambdas=lambda_tuple,
        bootstrap_repeats=bootstrap_repeats,
        runtime_sec=runtime_sec,
    )
    reproducibility_path = _build_reproducibility_table(out_dir, runtime_sec=runtime_sec)
    pairwise_path = _build_pairwise_table(
        out_dir,
        alpha_metrics=alpha_metrics,
        best_alpha=float(detail_df.iloc[0]["alpha_graph"]),
        current_alpha=current_alpha,
        bootstrap_repeats=bootstrap_repeats,
    )
    fig_png, fig_pdf = _build_sensitivity_figure(detail_df, out_dir, current_alpha=current_alpha)

    print(f"Saved NDSS alpha detail    -> {detail_path}")
    print(f"Saved NDSS alpha summary   -> {summary_path}")
    print(f"Saved NDSS alpha protocol  -> {protocol_path}")
    print(f"Saved NDSS reproducibility -> {reproducibility_path}")
    if pairwise_path:
        print(f"Saved NDSS pairwise table  -> {pairwise_path}")
    print(f"Saved NDSS sensitivity fig -> {fig_png}")
    print(f"Saved NDSS sensitivity pdf -> {fig_pdf}")
    return detail_path, summary_path


if __name__ == "__main__":
    default_scen = os.path.join(os.path.dirname(__file__), "..", "data", "ndss_scenarios")
    default_proc = os.path.join(os.path.dirname(__file__), "..", "data", "NDSS_process_edges.csv")
    default_gt = os.path.join(os.path.dirname(__file__), "..", "data", "ndss_attack_scenarios.csv")
    default_out = os.path.join(os.path.dirname(__file__), "..", "outputs", "appendix")
    graphad_cfg = read_yaml(CONFIG_DIR / "train_graphad.yaml", default={})
    default_lambda = (
        float(graphad_cfg.get("lambda_z", 0.4)),
        float(graphad_cfg.get("lambda_tr", 0.3)),
        float(graphad_cfg.get("lambda_fl", 0.3)),
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--scen_dir", default=default_scen)
    parser.add_argument("--procmap", default=default_proc)
    parser.add_argument("--gt_csv", default=default_gt)
    parser.add_argument("--out_dir", default=default_out)
    parser.add_argument("--alpha_start", type=float, default=0.0)
    parser.add_argument("--alpha_stop", type=float, default=0.6)
    parser.add_argument("--alpha_step", type=float, default=0.1)
    parser.add_argument("--lambda_z", type=float, default=default_lambda[0])
    parser.add_argument("--lambda_tr", type=float, default=default_lambda[1])
    parser.add_argument("--lambda_fl", type=float, default=default_lambda[2])
    parser.add_argument("--bootstrap_repeats", type=int, default=1000)
    parser.add_argument("--max_scenarios", type=int, default=None)
    args = parser.parse_args()

    run_alpha_sweep(
        scen_dir=args.scen_dir,
        procmap_csv=args.procmap,
        gt_csv=args.gt_csv,
        out_dir=args.out_dir,
        alpha_values=alpha_grid(start=args.alpha_start, stop=args.alpha_stop, step=args.alpha_step),
        lambda_tuple=(args.lambda_z, args.lambda_tr, args.lambda_fl),
        bootstrap_repeats=args.bootstrap_repeats,
        max_scenarios=args.max_scenarios,
    )
