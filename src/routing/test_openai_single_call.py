from __future__ import annotations

from pathlib import Path
import time

from src.routing.selective_llm_eval import DEFAULT_Q, build_llm_runner, get_seed_view
from src.utils.io import read_csv, read_json, read_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PRED_DIR = OUTPUT_DIR / "predictions"
METRIC_DIR = OUTPUT_DIR / "metrics"


def main() -> None:
    cfg = read_yaml(CONFIG_DIR / "routing.yaml", default={})
    tau_info = read_json(METRIC_DIR / "thresholds.json")
    gray_info = read_json(METRIC_DIR / "grayzone_defaults.json")
    pred_main = read_csv(PRED_DIR / "base_test_main_predictions.csv")

    seed = 0
    seed_key = f"seed{seed}"
    tau = float(tau_info["per_seed"][seed_key]["tau"])
    margin = float(gray_info["per_seed"][seed_key][f"{float(DEFAULT_Q):.2f}"]["gray_margin"])
    seed_df = get_seed_view(pred_main, seed, cfg)
    seed_df["gray_zone"] = (seed_df["p_utar_base"].sub(tau).abs() <= margin).astype(int)
    seed_df["xgb_shortcut"] = ((seed_df["p_xgb"] <= float(cfg.get("xgb_shortcut_low", 0.20))) | (seed_df["p_xgb"] >= float(cfg.get("xgb_shortcut_high", 0.80)))).astype(int)
    cand = seed_df[(seed_df["gray_zone"] == 1) & (seed_df["xgb_shortcut"] == 0)].head(1).copy()

    if cand.empty:
        raise RuntimeError("No candidate row found for a single-call test.")

    row = cand.iloc[0]
    print(
        f"testing one call with seed={seed}, sample_idx={int(row['sample_idx'])}, run_id={int(row['run_id'])}, "
        f"phase={row['phase']}, tau={tau:.4f}, margin={margin:.4f}"
    )

    runner = build_llm_runner(cfg)
    started = time.perf_counter()
    prob, usage = runner.probability_with_usage(row)
    elapsed = time.perf_counter() - started
    print(f"probability={prob:.6f}")
    print(f"usage={usage}")
    print(f"elapsed_sec={elapsed:.2f}")


if __name__ == "__main__":
    main()
