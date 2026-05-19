#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""NDSS-based GraphAD beta sweep for hybrid reranking evidence."""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ndss.ndss_reasoning_cost import run_ndss_reasoning
from src.utils.io import read_yaml


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def beta_grid(raw: str) -> list[float]:
    return [float(v.strip()) for v in raw.split(",") if v.strip()]


def _safe_model_name(model: str) -> str:
    return model.replace("/", "_").replace(":", "_").replace(" ", "_")


def _safe_float_tag(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def _summary_matches(
    summary_payload: dict,
    *,
    model: str,
    alpha_graph: float,
    lambda_tuple: tuple[float, float, float],
    beta: float,
) -> bool:
    run_params = summary_payload.get("run_params") or {}
    return (
        run_params.get("model") == model
        and round(float(run_params.get("alpha_graph", float("nan"))), 6) == round(alpha_graph, 6)
        and round(float(run_params.get("w_z", float("nan"))), 6) == round(lambda_tuple[0], 6)
        and round(float(run_params.get("w_trend", float("nan"))), 6) == round(lambda_tuple[1], 6)
        and round(float(run_params.get("w_flat", float("nan"))), 6) == round(lambda_tuple[2], 6)
        and round(float(run_params.get("beta", float("nan"))), 6) == round(beta, 6)
    )


def run_beta_sweep(
    scen_dir: str,
    procmap_csv: str,
    gt_csv: str,
    out_dir: str,
    llm_model: str,
    beta_values: list[float],
    alpha_graph: float,
    lambda_tuple: tuple[float, float, float],
) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "ndss_reasoning_beta_sweep.json")

    rows = []
    for beta in beta_values:
        safe_model = _safe_model_name(llm_model)
        beta_tag = _safe_float_tag(beta)
        detail_path = os.path.join(out_dir, f"ndss_reasoning_hybrid_{safe_model}_beta{beta_tag}.json")
        summary_path = os.path.join(out_dir, f"ndss_performance_summary_hybrid_{safe_model}_beta{beta_tag}.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if _summary_matches(
                cached,
                model=llm_model,
                alpha_graph=alpha_graph,
                lambda_tuple=lambda_tuple,
                beta=beta,
            ):
                result = {
                    "attacks_evaluated": cached["attacks_evaluated"],
                    "ACT_Hit@5": cached["ACT_Hit@5"],
                    "K-Concord": cached["K-Concord"],
                    "SGR": cached["SGR"],
                    "Top1_recall": cached["Top1_recall"],
                    "Top3_recall": cached["Top3_recall"],
                    "Top5_recall": cached["Top5_recall"],
                    "MRR": cached.get("MRR"),
                    "best_true_rank_mean": cached.get("best_true_rank_mean"),
                    "ranking_variance": cached.get("ranking_variance"),
                    "usage": {
                        "calls": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "cost_usd": 0.0,
                    },
                    "detail_path": detail_path,
                    "summary_path": summary_path,
                }
            else:
                result = run_ndss_reasoning(
                    scen_dir=scen_dir,
                    procmap_csv=procmap_csv,
                    gt_csv=gt_csv,
                    out_json=out_json,
                    mode="hybrid",
                    llm_model=llm_model,
                    cost_csv_path=None,
                    alpha_graph=alpha_graph,
                    w_z=lambda_tuple[0],
                    w_trend=lambda_tuple[1],
                    w_flat=lambda_tuple[2],
                    beta=beta,
                )
                if result is None:
                    continue
        else:
            result = run_ndss_reasoning(
                scen_dir=scen_dir,
                procmap_csv=procmap_csv,
                gt_csv=gt_csv,
                out_json=out_json,
                mode="hybrid",
                llm_model=llm_model,
                cost_csv_path=None,
                alpha_graph=alpha_graph,
                w_z=lambda_tuple[0],
                w_trend=lambda_tuple[1],
                w_flat=lambda_tuple[2],
                beta=beta,
            )
            if result is None:
                continue
        rows.append(
            {
                "beta": beta,
                "model": llm_model,
                "alpha_graph": alpha_graph,
                "lambda_z": lambda_tuple[0],
                "lambda_tr": lambda_tuple[1],
                "lambda_fl": lambda_tuple[2],
                "attacks_evaluated": result["attacks_evaluated"],
                "ACT_Hit@5": result["ACT_Hit@5"],
                "K-Concord": result["K-Concord"],
                "SGR": result["SGR"],
                "Top1_recall": result["Top1_recall"],
                "Top3_recall": result["Top3_recall"],
                "Top5_recall": result["Top5_recall"],
                "MRR": result["MRR"],
                "best_true_rank_mean": result["best_true_rank_mean"],
                "ranking_variance": result["ranking_variance"],
                "llm_calls": result["usage"]["calls"],
                "prompt_tokens": result["usage"]["prompt_tokens"],
                "completion_tokens": result["usage"]["completion_tokens"],
                "total_tokens": result["usage"]["prompt_tokens"] + result["usage"]["completion_tokens"],
                "cost_usd": result["usage"]["cost_usd"],
                "detail_path": result["detail_path"],
                "summary_path": result["summary_path"],
            }
        )

    detail_df = pd.DataFrame(rows).sort_values(
        by=["Top1_recall", "Top3_recall", "MRR", "K-Concord", "SGR"],
        ascending=False,
    )
    detail_path = os.path.join(out_dir, "table_h10_graphad_beta_grid_detail_ndss.csv")
    detail_df.to_csv(detail_path, index=False)

    summary_rows = []
    if not detail_df.empty:
        summary_rows.append({"Selection": "Best Top1-first NDSS beta", **detail_df.iloc[0].to_dict()})
    current_rows = detail_df[detail_df["beta"].round(6).eq(round(0.2, 6))]
    if not current_rows.empty:
        summary_rows.append({"Selection": "Current manuscript beta", **current_rows.iloc[0].to_dict()})
    zero_rows = detail_df[detail_df["beta"].round(6).eq(0.0)]
    if not zero_rows.empty:
        summary_rows.append({"Selection": "Graph-only baseline (beta=0)", **zero_rows.iloc[0].to_dict()})
    summary_path = os.path.join(out_dir, "table_h10_graphad_beta_selection_ndss.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

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
    default_alpha = float(graphad_cfg.get("alpha", 0.3))

    parser = argparse.ArgumentParser()
    parser.add_argument("--scen_dir", default=default_scen)
    parser.add_argument("--procmap", default=default_proc)
    parser.add_argument("--gt_csv", default=default_gt)
    parser.add_argument("--out_dir", default=default_out)
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--betas", default="0.0,0.2,0.4")
    parser.add_argument("--alpha_graph", type=float, default=default_alpha)
    parser.add_argument("--lambda_z", type=float, default=default_lambda[0])
    parser.add_argument("--lambda_tr", type=float, default=default_lambda[1])
    parser.add_argument("--lambda_fl", type=float, default=default_lambda[2])
    args = parser.parse_args()

    run_beta_sweep(
        scen_dir=args.scen_dir,
        procmap_csv=args.procmap,
        gt_csv=args.gt_csv,
        out_dir=args.out_dir,
        llm_model=args.model,
        beta_values=beta_grid(args.betas),
        alpha_graph=args.alpha_graph,
        lambda_tuple=(args.lambda_z, args.lambda_tr, args.lambda_fl),
    )
