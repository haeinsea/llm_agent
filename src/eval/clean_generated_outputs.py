from __future__ import annotations

import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"

TARGET_DIRS = [
    OUTPUT_DIR / "appendix",
    OUTPUT_DIR / "figures",
    OUTPUT_DIR / "shift_analysis_test_main",
    OUTPUT_DIR / "shift_analysis_val",
    OUTPUT_DIR / "metrics" / "main_only_progress",
]

TARGET_GLOBS = [
    OUTPUT_DIR / "predictions" / "base_*.csv",
    OUTPUT_DIR / "predictions" / "utar*.csv",
    OUTPUT_DIR / "metrics" / "base_inference_runtime*",
    OUTPUT_DIR / "metrics" / "grayzone_*",
    OUTPUT_DIR / "metrics" / "selective_llm_*",
    OUTPUT_DIR / "metrics" / "table*.csv",
    OUTPUT_DIR / "metrics" / "thresholds*.csv",
    OUTPUT_DIR / "metrics" / "thresholds.json",
    OUTPUT_DIR / "metrics" / "graphad_training_summary.json",
    OUTPUT_DIR / "metrics" / "feature_drift_summary.csv",
    OUTPUT_DIR / "metrics" / "split_summary.csv",
    OUTPUT_DIR / "evaluation" / "*.csv",
    OUTPUT_DIR / "evaluation" / "*.json",
    OUTPUT_DIR / "tuning" / "*.csv",
    OUTPUT_DIR / "tuning" / "*.json",
]


def _remove_path(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    return 1


def main() -> None:
    removed = 0
    for directory in TARGET_DIRS:
        removed += _remove_path(directory)
        directory.mkdir(parents=True, exist_ok=True)

    for pattern in TARGET_GLOBS:
        parent = pattern.parent
        if not parent.exists():
            continue
        for path in parent.glob(pattern.name):
            removed += _remove_path(path)

    print(f"Removed {removed} generated artifact(s).")


if __name__ == "__main__":
    main()
