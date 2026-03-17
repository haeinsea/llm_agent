from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from src.utils.io import read_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
META_DIR = DATA_DIR / "meta"
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_yaml(path: Path, default: dict) -> dict:
    return read_yaml(path, default=default)


def ensure_output_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_feature_cols() -> List[str]:
    with open(META_DIR / "feature_columns.json", "r", encoding="utf-8") as f:
        return json.load(f)


def read_split_csv(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing processed split file: {path}")
    return pd.read_csv(path)


def build_windows_for_df(
    df: pd.DataFrame,
    feature_cols: List[str],
    win: int,
    stride: int,
) -> pd.DataFrame:
    rows = []
    flat_cols = []

    for lag in range(win - 1, -1, -1):
        for feat in feature_cols:
            flat_cols.append(f"{feat}_t-{lag}" if lag > 0 else f"{feat}_t0")

    meta_cols = [
        "source_file",
        "domain_tag",
        "split_group",
        "run_id",
        "fault_id",
        "sample_idx",
        "y",
        "phase",
        "onset_step",
        "transition_len",
    ]

    # 중요: fault_id까지 포함해서 run unit 분리
    df = df.sort_values(["source_file", "fault_id", "run_id", "sample_idx"]).reset_index(drop=True)

    for (source_file, fault_id, run_id), g in df.groupby(["source_file", "fault_id", "run_id"], sort=False):
        g = g.sort_values("sample_idx").reset_index(drop=True)
        X = g[feature_cols].to_numpy(dtype=np.float32)

        if len(g) < win:
            continue

        for end_idx in range(win - 1, len(g), stride):
            start_idx = end_idx - win + 1
            sample_seq = g.loc[start_idx:end_idx, "sample_idx"].to_numpy()
            if not np.all(np.diff(sample_seq) == 1):
                continue

            window = X[start_idx:end_idx + 1]
            flat = window.reshape(-1)

            end_row = g.iloc[end_idx]
            meta = {c: end_row[c] for c in meta_cols if c in g.columns}

            row = {**meta}
            row.update({c: v for c, v in zip(flat_cols, flat.tolist())})
            rows.append(row)

    return pd.DataFrame(rows)


def print_window_stats(name: str, df: pd.DataFrame) -> None:
    print(f"\n[{name}] windows={len(df):,}")
    if len(df) == 0:
        return

    y_counts = df["y"].value_counts(dropna=False).to_dict()
    pos_ratio = float(df["y"].mean()) if len(df) else 0.0
    phase_counts = df["phase"].value_counts(dropna=False).to_dict() if "phase" in df.columns else {}

    print(f"  pos_ratio    : {pos_ratio:.4f}")
    print(f"  y_counts     : {y_counts}")
    print(f"  phase_counts : {phase_counts}")

    n_units = df[["source_file", "fault_id", "run_id"]].drop_duplicates().shape[0]
    print(f"  run units    : {n_units:,}")


def save_windows(df: pd.DataFrame, filename: str) -> None:
    out_path = PROCESSED_DIR / filename
    df.to_csv(out_path, index=False)
    print(f"Saved {filename}: {len(df):,} rows")


def main() -> None:
    ensure_output_dirs()

    tcn_cfg = load_yaml(
        CONFIG_DIR / "train_tcn.yaml",
        default={"window_size": 50, "stride": 1},
    )
    win = int(tcn_cfg.get("window_size", 50))
    stride = int(tcn_cfg.get("stride", 1))

    feature_cols = load_feature_cols()

    file_map = {
        "te_train_rows_tcn.csv": "te_train_windows.csv",
        "te_val_rows.csv": "te_val_windows.csv",
        # 공통 test row가 아니라 TCN 문맥용 full-run test에서 window 생성
        "te_test_full_rows_tcn.csv": "te_test_full_windows_tcn.csv",
    }

    for in_name, out_name in file_map.items():
        df = read_split_csv(in_name)
        win_df = build_windows_for_df(df, feature_cols, win=win, stride=stride)
        save_windows(win_df, out_name)
        print_window_stats(out_name, win_df)

    print("\nWindow building completed.")


if __name__ == "__main__":
    main()
