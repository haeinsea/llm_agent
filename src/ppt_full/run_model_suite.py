from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

from src.ppt_full.model_suite import (
    evaluate_suite_predictions,
    generate_suite_predictions,
    train_adaptable_family,
    train_invariant_family,
    train_rf_family,
    train_tcn_family,
    train_xgb_family,
)
from src.tuning.optimize_adaptable import run_search as run_adaptable_search
from src.tuning.optimize_invariant import run_search as run_invariant_search
from src.tuning.optimize_modern_tcn import run_search as run_tcn_search
from src.tuning.optimize_rf import run_search as run_rf_search
from src.tuning.optimize_xgb import run_search as run_xgb_search
from src.utils.io import ensure_dir, read_json, read_yaml, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
EXPERIMENT_ROOT = OUTPUT_DIR / "experiments"
DEFAULT_EXPERIMENT = "ppt_full_reflect_all_20260411_subset_hpo"
PPT_CONFIG_NAME = "ppt_full_reflect_all.yaml"

SEARCH_CONFIG_MAP = {
    "rf": "search_rf_ppt_full_auc_wide.yaml",
    "xgb": "search_xgb_ppt_full_auc_wide.yaml",
    "tcn": "search_tcn_ppt_full_auc_wide.yaml",
    "adaptable": "search_adaptable_ppt_full_auc_wide.yaml",
    "invariant": "search_invariant_ppt_full_auc_wide.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--ppt-config-name", default=PPT_CONFIG_NAME)
    parser.add_argument("--skip-tuning", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-infer", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def _snapshot_config(src_name: str, target_dir: Path) -> None:
    src = CONFIG_DIR / src_name
    if src.exists():
        shutil.copy2(src, target_dir / src_name)


def _load_best_params(path: Path) -> dict:
    data = read_json(path)
    return data.get("best_params", data)


def _run_tuning(output_root: Path) -> dict[str, dict]:
    tuning_dir = output_root / "tuning"
    ensure_dir(tuning_dir)
    _, rf_best = run_rf_search(config_name=SEARCH_CONFIG_MAP["rf"], output_prefix="rf", output_dir=str(tuning_dir))
    _, xgb_best = run_xgb_search(config_name=SEARCH_CONFIG_MAP["xgb"], output_prefix="xgb", output_dir=str(tuning_dir))
    _, tcn_best = run_tcn_search(config_name=SEARCH_CONFIG_MAP["tcn"], output_prefix="tcn", output_dir=str(tuning_dir))
    _, adaptable_best = run_adaptable_search(config_name=SEARCH_CONFIG_MAP["adaptable"], output_prefix="adaptable", output_dir=str(tuning_dir))
    _, invariant_best = run_invariant_search(config_name=SEARCH_CONFIG_MAP["invariant"], output_prefix="invariant", output_dir=str(tuning_dir))
    return {
        "rf": rf_best,
        "xgb": xgb_best,
        "tcn": tcn_best,
        "adaptable": adaptable_best,
        "invariant": invariant_best,
    }


def _load_tuned_params(output_root: Path) -> dict[str, dict]:
    tuning_dir = output_root / "tuning"
    best_paths = {
        "rf": tuning_dir / "rf_best.json",
        "xgb": tuning_dir / "xgb_best.json",
        "tcn": tuning_dir / "tcn_best.json",
        "adaptable": tuning_dir / "adaptable_best.json",
        "invariant": tuning_dir / "invariant_best.json",
    }
    return {name: _load_best_params(path) for name, path in best_paths.items()}


def main() -> None:
    args = parse_args()
    output_root = EXPERIMENT_ROOT / args.experiment_name
    config_dir = output_root / "configs"
    ensure_dir(config_dir)
    _snapshot_config(args.ppt_config_name, config_dir)
    for cfg_name in SEARCH_CONFIG_MAP.values():
        _snapshot_config(cfg_name, config_dir)

    tuning_payload = None
    if not args.skip_tuning:
        tuning_payload = _run_tuning(output_root)

    best_params = _load_tuned_params(output_root)

    if not args.skip_train:
        train_rf_family(best_params["rf"], output_root)
        train_xgb_family(best_params["xgb"], output_root)
        train_tcn_family(best_params["tcn"], output_root)
        train_adaptable_family(best_params["adaptable"], output_root)
        train_invariant_family(best_params["invariant"], output_root)

    ppt_cfg = read_yaml(CONFIG_DIR / args.ppt_config_name, default={})

    prediction_paths = {}
    if not args.skip_infer:
        prediction_paths = generate_suite_predictions(output_root, ppt_cfg=ppt_cfg)

    metric_paths = {}
    if not args.skip_eval:
        metric_paths = evaluate_suite_predictions(output_root)

    write_json(
        output_root / "manifest.json",
        {
            "experiment_name": args.experiment_name,
            "created_at": datetime.now().isoformat(),
            "purpose": "PPT-full reflected model suite with wide validation HPO and 10-seed outputs.",
            "ppt_config": str(CONFIG_DIR / args.ppt_config_name),
            "search_configs": SEARCH_CONFIG_MAP,
            "tuning_completed_in_this_run": not args.skip_tuning,
            "tuning_payload_keys": sorted(list(tuning_payload.keys())) if tuning_payload else [],
            "prediction_paths": {k: str(v) for k, v in prediction_paths.items()},
            "metric_paths": {k: str(v) for k, v in metric_paths.items()},
        },
    )


if __name__ == "__main__":
    main()
