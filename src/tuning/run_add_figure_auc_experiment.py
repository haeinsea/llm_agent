from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.tuning.optimize_adaptable import run_search as run_adaptable_search
from src.tuning.optimize_invariant import run_search as run_invariant_search
from src.tuning.optimize_modern_tcn import run_search as run_tcn_search
from src.tuning.optimize_rf import run_search as run_rf_search
from src.tuning.optimize_xgb import run_search as run_xgb_search
from src.utils.io import ensure_dir, read_json, write_csv, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
EXPERIMENT_ROOT = OUTPUT_DIR / "experiments"
DEFAULT_EXPERIMENT = "add_figure_auc_boost_20260411"
PPT_PATH = Path("/Users/cordie/Downloads/AUC 성능 올리는 방법.pptx")

MODEL_RUNNERS = {
    "rf": (run_rf_search, "search_rf_add_figure_auc.yaml"),
    "xgb": (run_xgb_search, "search_xgb_add_figure_auc.yaml"),
    "tcn": (run_tcn_search, "search_tcn_add_figure_auc.yaml"),
    "adaptable": (run_adaptable_search, "search_adaptable_add_figure_auc.yaml"),
    "invariant": (run_invariant_search, "search_invariant_add_figure_auc.yaml"),
}

MODEL_RUNNERS_BROAD = {
    "rf": (run_rf_search, "search_rf_add_figure_auc_broad.yaml"),
    "xgb": (run_xgb_search, "search_xgb_add_figure_auc_broad.yaml"),
    "tcn": (run_tcn_search, "search_tcn_add_figure_auc_broad.yaml"),
    "adaptable": (run_adaptable_search, "search_adaptable_add_figure_auc.yaml"),
    "invariant": (run_invariant_search, "search_invariant_add_figure_auc_broad.yaml"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--profile", choices=["default", "broad"], default="default")
    parser.add_argument(
        "--only",
        nargs="*",
        choices=list(MODEL_RUNNERS.keys()),
        default=list(MODEL_RUNNERS.keys()),
        help="Optional subset of models to tune.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing tuning files in the experiment folder when present.",
    )
    return parser.parse_args()


def _copy_config_snapshot(config_name: str, config_dir: Path) -> None:
    src = CONFIG_DIR / config_name
    if src.exists():
        shutil.copy2(src, config_dir / config_name)


def _summary_row(model_name: str, config_name: str, best_row: dict) -> dict:
    return {
        "Model": model_name,
        "Search Config": config_name,
        "Objective": float(best_row.get("objective", best_row.get("objective_mean", float("nan")))),
        "AUC": float(best_row.get("auc_mean", float("nan"))),
        "F1": float(best_row.get("f1_mean", float("nan"))),
        "Recall": float(best_row.get("recall_mean", float("nan"))),
        "Post-Shift Recall": float(best_row.get("post_shift_recall_mean", float("nan"))),
        "High-Entropy Recall": float(best_row.get("high_entropy_recall_mean", float("nan"))),
        "Gray-Zone Recall": float(best_row.get("grayzone_recall_mean", float("nan"))),
        "Mean Entropy": float(best_row.get("mean_entropy_mean", float("nan"))),
        "Gray-Zone Share": float(best_row.get("grayzone_share_mean", float("nan"))),
        "Best Params": json.dumps(best_row.get("best_params", {}), ensure_ascii=False),
    }


def _build_alignment_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "PPT Topic": "Discrepancy Measure",
                "Requested Logic": "Pairwise gap 대신 RF/XGB/ModernTCN 3모델의 통계적 표준편차(D_ens) 사용",
                "Where Reflected": "src/utils/routing.py:model_discrepancy, configs/soft_fusion_add_figure.yaml",
                "Experiment Reflection": "AUC 실험 요약에 high-entropy/gray-zone 지표를 포함하고, soft-fusion 설정 스냅샷을 같이 저장",
            },
            {
                "PPT Topic": "Gray-Zone / U_total",
                "Requested Logic": "Gray-Zone을 단일 임계값이 아닌 D_ens x 평균 Shannon entropy(U_total)로 정의",
                "Where Reflected": "src/utils/routing.py:ensemble_entropy+model_discrepancy, configs/soft_fusion_add_figure.yaml",
                "Experiment Reflection": "각 모델 HPO objective에 high-entropy recall, gray-zone recall을 포함",
            },
            {
                "PPT Topic": "Mixing Weight",
                "Requested Logic": "Sigmoid 기반 mixing weight w(x)=1/(1+exp(-k(U_total-tau)))",
                "Where Reflected": "configs/routing.yaml:sigmoid_gain, configs/soft_fusion_add_figure.yaml:mixing_tau/mixing_k",
                "Experiment Reflection": "신규 실험 폴더에 routing/soft-fusion 설정 스냅샷을 함께 저장",
            },
            {
                "PPT Topic": "LLM Calibration",
                "Requested Logic": "Platt-scaling 스타일로 LLM 점수를 0~1 확률 공간에 보정",
                "Where Reflected": "configs/soft_fusion_add_figure.yaml:calibration_method/calibration_factor",
                "Experiment Reflection": "모델 HPO와 별도로 soft-fusion 파라미터 스냅샷을 남겨 이후 add_figure 연결 가능하게 보존",
            },
            {
                "PPT Topic": "Final Decision Scoring",
                "Requested Logic": "Hard switch 대신 (1-w)s_local + w calibrate(s_llm) soft-fusion 사용",
                "Where Reflected": "configs/soft_fusion_add_figure.yaml:final_scoring_formula",
                "Experiment Reflection": "이번 번들은 5모델 HPO를 중심으로 저장하되, PPT 최종 의사결정 수식 설정도 동일 실험 폴더에 함께 보존",
            },
        ]
    )


def main() -> None:
    args = parse_args()
    experiment_dir = EXPERIMENT_ROOT / args.experiment_name
    tuning_dir = experiment_dir / "tuning"
    appendix_dir = experiment_dir / "appendix"
    config_dir = experiment_dir / "configs"
    ensure_dir(tuning_dir)
    ensure_dir(appendix_dir)
    ensure_dir(config_dir)

    runner_map = MODEL_RUNNERS_BROAD if args.profile == "broad" else MODEL_RUNNERS

    summary_rows = []
    for model_name in args.only:
        runner, config_name = runner_map[model_name]
        _copy_config_snapshot(config_name, config_dir)
        best_path = tuning_dir / f"{model_name}_best.json"
        trials_path = tuning_dir / f"{model_name}_trials.csv"
        if args.skip_existing and best_path.exists() and trials_path.exists():
            best_row = read_json(best_path)
            print(f"[SKIP] reuse existing tuning outputs for {model_name}: {best_path}", flush=True)
        else:
            _, best_row = runner(config_name=config_name, output_prefix=model_name, output_dir=str(tuning_dir))
        summary_rows.append(_summary_row(model_name, config_name, best_row))

    for extra_cfg in ["routing.yaml", "soft_fusion_add_figure.yaml"]:
        _copy_config_snapshot(extra_cfg, config_dir)

    summary_df = pd.DataFrame(summary_rows).sort_values(["AUC", "Objective"], ascending=[False, False]).reset_index(drop=True)
    write_csv(appendix_dir / "table_h12_add_figure_auc_hpo_summary.csv", summary_df)
    write_csv(appendix_dir / "table_h13_add_figure_ppt_alignment.csv", _build_alignment_table())
    write_json(
        experiment_dir / "manifest.json",
        {
            "experiment_name": args.experiment_name,
            "created_at": datetime.now().isoformat(),
            "ppt_source": str(PPT_PATH),
            "notes": [
                "Keep existing AUC/tuning history intact by writing into outputs/experiments/<name>/.",
                "Prioritize AUC and ranking stability while retaining recall on high-entropy and gray-zone samples.",
                "Cover RF, XGB, ModernTCN, AdapTable-style TTA, and Invariant temporal baselines.",
                "Store PPT-derived soft-fusion/discrepancy/calibration settings alongside model HPO outputs.",
            ],
            "models": args.only,
        },
    )
    print(f"[DONE] add_figure AUC experiment saved under {experiment_dir}", flush=True)


if __name__ == "__main__":
    main()
