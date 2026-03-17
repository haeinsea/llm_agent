from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from src.routing.selective_llm_eval import (
    DEFAULT_Q,
    METRIC_DIR,
    PRED_DIR,
    build_base_view,
    build_llm_runner,
    run_mode,
    summarize_rows,
)
from src.utils.io import read_csv, read_json, read_yaml, write_csv
from src.utils.metrics import binary_metrics, prr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
INTERMEDIATE_DIR = METRIC_DIR / "main_only_progress"
STATUS_PATH = INTERMEDIATE_DIR / "status.json"


def _build_quick_table(summary: pd.DataFrame) -> pd.DataFrame:
    sub = summary[(summary["dataset"] == "main") & (summary["mode"].isin(["no_llm", "selective"]))].copy()
    label_map = {
        "no_llm": "w/o Gray-Zone",
        "selective": "Selective UTAR",
    }
    sub["Method"] = sub["mode"].map(label_map)
    cols = [
        "Method",
        "f1_mean",
        "f1_std",
        "recall_mean",
        "recall_std",
        "precision_mean",
        "precision_std",
        "prr_mean",
        "prr_std",
        "instability_mean",
        "instability_std",
        "llm_call_rate_mean",
        "llm_call_rate_std",
        "cost_usd_mean",
        "cost_usd_std",
        "uses_actual_api_usage_mean",
    ]
    return sub[cols].sort_values("Method").reset_index(drop=True)


def _update_status(**payload: object) -> None:
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATUS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _pick_first_selective_candidate(base_df: pd.DataFrame, cfg: dict, tau: float, margin: float) -> pd.Series:
    cand = base_df.copy()
    cand["gray_zone"] = (cand["p_utar_base"].sub(tau).abs() <= margin).astype(int)
    cand["xgb_shortcut"] = (
        (cand["p_xgb"] <= float(cfg.get("xgb_shortcut_low", 0.20)))
        | (cand["p_xgb"] >= float(cfg.get("xgb_shortcut_high", 0.80)))
    ).astype(int)
    out = cand[(cand["gray_zone"] == 1) & (cand["xgb_shortcut"] == 0)].head(1).copy()
    if out.empty:
        raise RuntimeError("No selective candidate row found for warmup.")
    return out.iloc[0]


def _warmup_openai_runner(base_df: pd.DataFrame, cfg: dict, tau: float, margin: float, llm_runner) -> None:
    row = _pick_first_selective_candidate(base_df, cfg, tau, margin)
    print(
        "[main_only] warmup OpenAI call "
        f"(sample_idx={int(row['sample_idx'])}, run_id={int(row['run_id'])}, phase={row['phase']})",
        flush=True,
    )
    started = time.perf_counter()
    prob, usage = llm_runner.probability_with_usage(row)
    elapsed = time.perf_counter() - started
    print(
        f"[main_only] warmup success in {elapsed:.2f}s "
        f"(prob={prob:.4f}, input_tokens={int(usage['prompt_tokens'])}, output_tokens={int(usage['completion_tokens'])})",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Main-only selective LLM evaluation for quick TE checks.")
    parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        default=["selective", "no_llm"],
        choices=["selective", "no_llm"],
        help="Modes to run for the 4,000-row main test set.",
    )
    args = parser.parse_args()

    cfg = read_yaml(CONFIG_DIR / "routing.yaml", default={})
    tau_info = read_json(METRIC_DIR / "thresholds.json")
    gray_grid = read_csv(METRIC_DIR / "grayzone_grid.csv").sort_values("q").reset_index(drop=True)

    pred_val = read_csv(PRED_DIR / "base_val_predictions.csv")
    pred_main = read_csv(PRED_DIR / "base_test_main_predictions.csv")
    val_base = build_base_view(pred_val, cfg)
    main_base = build_base_view(pred_main, cfg)

    tau = float(tau_info["tau"])
    margin = float(gray_grid.loc[(gray_grid["q"] - DEFAULT_Q).abs().idxmin(), "gray_margin_mean"])

    llm_runner_live = build_llm_runner(cfg)
    llm_runner_live.timeout_sec = min(llm_runner_live.timeout_sec, 20)
    llm_runner_live.max_retries = min(llm_runner_live.max_retries, 1)
    llm_runner_live.retry_backoff_sec = min(llm_runner_live.retry_backoff_sec, 1.0)

    print("[main_only] starting selective_llm_eval_main_only", flush=True)
    print(f"[main_only] DEFAULT_Q={DEFAULT_Q:.2f}, modes={args.modes}", flush=True)
    _update_status(stage="starting", default_q=DEFAULT_Q, modes=args.modes, tau=tau, margin=margin)

    if "selective" in args.modes:
        _warmup_openai_runner(main_base, cfg, tau, margin, llm_runner_live)

    ref_recall = binary_metrics(val_base["y_true"], val_base["p_utar_base"], tau=tau)["recall"]
    rows: list[dict] = []
    mode_outputs: dict[str, pd.DataFrame] = {}
    started_all = time.perf_counter()

    for mode in args.modes:
        runner = llm_runner_live if mode == "selective" else build_llm_runner(cfg, force_stub=True)
        out, metrics = run_mode(main_base, tau, margin, cfg, mode, runner, progress_label=f"main mode={mode}")
        metrics["prr"] = prr(ref_recall, metrics["recall"])
        rows.append({"dataset": "main", "q": float(DEFAULT_Q), "seed": -1, **metrics})
        mode_outputs[mode] = out
        _update_status(
            stage="mode_done",
            mode=mode,
            elapsed_sec=time.perf_counter() - started_all,
            f1=metrics["f1"],
            recall=metrics["recall"],
            llm_call_rate=metrics["llm_call_rate"],
            uses_actual_api_usage=metrics["uses_actual_api_usage"],
        )

    summary_df, summary_seed = summarize_rows(rows)
    quick_table = _build_quick_table(summary_seed)

    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(INTERMEDIATE_DIR / "selective_llm_main_only_seed_metrics_partial.csv", summary_df)
    write_csv(INTERMEDIATE_DIR / "selective_llm_main_only_summary_partial.csv", summary_seed)
    write_csv(INTERMEDIATE_DIR / "selective_llm_main_only_table_partial.csv", quick_table)

    if "selective" in mode_outputs:
        write_csv(PRED_DIR / "utar_test_main_selective.csv", mode_outputs["selective"])
    if "no_llm" in mode_outputs:
        write_csv(PRED_DIR / "utar_test_main_no_llm.csv", mode_outputs["no_llm"])

    write_csv(METRIC_DIR / "selective_llm_main_only_seed_metrics.csv", summary_df)
    write_csv(METRIC_DIR / "selective_llm_main_only_summary.csv", summary_seed)
    write_csv(METRIC_DIR / "selective_llm_main_only_table.csv", quick_table)
    _update_status(stage="completed", elapsed_sec=time.perf_counter() - started_all)
    print(f"[main_only] completed in {time.perf_counter() - started_all:.1f}s", flush=True)
    print(summary_seed.to_string(index=False))
    print("\nQuick main-only table")
    print(quick_table.to_string(index=False))


if __name__ == "__main__":
    main()
