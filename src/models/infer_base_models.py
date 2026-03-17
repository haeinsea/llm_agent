from __future__ import annotations

import json
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd

try:
    import torch
except Exception:
    torch = None
    
import re

from src.utils.io import ensure_dir

WINDOW_COL_PATTERN = re.compile(r"^(?P<base>.+)_t(?P<lag>\d+|-?\d+)$")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
META_DIR = DATA_DIR / "meta"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
PRED_DIR = OUTPUT_DIR / "predictions"

SEEDS = [0, 1, 2, 3, 4]


def load_feature_cols() -> List[str]:
    with open(META_DIR / "feature_columns.json", "r", encoding="utf-8") as f:
        return json.load(f)


def infer_window_feature_structure(columns):
    feat_cols = [c for c in columns if WINDOW_COL_PATTERN.match(c)]
    feats = {}
    for c in feat_cols:
        m = WINDOW_COL_PATTERN.match(c)
        assert m is not None
        base = m.group("base")
        lag = int(m.group("lag").replace("-", ""))
        feats.setdefault(base, []).append(lag)

    if not feats:
        raise ValueError("No valid window feature columns found.")

    feature_names = sorted(feats.keys())
    max_lag = max(max(v) for v in feats.values())
    win = max_lag + 1
    return feature_names, win


def flattened_to_tensor(df: pd.DataFrame) -> np.ndarray:
    flat_cols = [c for c in df.columns if WINDOW_COL_PATTERN.match(c)]
    feature_names, win = infer_window_feature_structure(flat_cols)
    n_feat = len(feature_names)

    X = np.zeros((len(df), n_feat, win), dtype=np.float32)
    feat_to_idx = {f: i for i, f in enumerate(feature_names)}

    for c in flat_cols:
        m = WINDOW_COL_PATTERN.match(c)
        assert m is not None
        base = m.group("base")
        lag = int(m.group("lag").replace("-", ""))
        feat_idx = feat_to_idx[base]
        X[:, feat_idx, lag] = df[c].to_numpy(dtype=np.float32)

    return X


if torch is not None:
    class Chomp1d(torch.nn.Module):
        def __init__(self, chomp_size: int):
            super().__init__()
            self.chomp_size = chomp_size

        def forward(self, x):
            if self.chomp_size == 0:
                return x
            return x[:, :, :-self.chomp_size].contiguous()


    class TemporalBlock(torch.nn.Module):
        def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation),
                Chomp1d(padding),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation),
                Chomp1d(padding),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
            )
            self.downsample = torch.nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            out = self.net(x)
            res = x if self.downsample is None else self.downsample(x)
            return self.relu(out + res)


    class TCNClassifier(torch.nn.Module):
        def __init__(self, n_features, channels=(64, 64, 64), kernel_size=3, dropout=0.1):
            super().__init__()
            layers = []
            n_in = n_features
            for i, ch in enumerate(channels):
                dilation = 2 ** i
                padding = (kernel_size - 1) * dilation
                layers.append(
                    TemporalBlock(
                        n_inputs=n_in,
                        n_outputs=ch,
                        kernel_size=kernel_size,
                        stride=1,
                        dilation=dilation,
                        padding=padding,
                        dropout=dropout,
                    )
                )
                n_in = ch
            self.tcn = torch.nn.Sequential(*layers)
            self.head = torch.nn.Linear(channels[-1], 1)

        def forward(self, x):
            h = self.tcn(x)
            h_last = h[:, :, -1]
            logits = self.head(h_last).squeeze(-1)
            return logits
else:
    class TCNClassifier:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for TCN inference.")


def read_rows(name: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / name)


def read_windows(name: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / name)


def base_predict_rows(df_rows: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    out = df_rows[
        ["source_file", "domain_tag", "split_group", "run_id", "fault_id", "sample_idx", "y", "phase", "onset_step", "transition_len"]
    ].copy()
    out = out.rename(columns={"y": "y_true"})

    X_raw = df_rows[feature_cols].to_numpy()

    rf_cols = []
    for seed in SEEDS:
        rf = joblib.load(MODEL_DIR / f"rf_model_seed{seed}.pkl")
        rf_imp = joblib.load(MODEL_DIR / f"rf_imputer_seed{seed}.pkl")
        X_rf = rf_imp.transform(X_raw)
        col = f"p_rf_seed{seed}"
        out[col] = rf.predict_proba(X_rf)[:, 1]
        rf_cols.append(col)

    xgb_cols = []
    for seed in SEEDS:
        xgb = joblib.load(MODEL_DIR / f"xgb_model_seed{seed}.pkl")
        xgb_imp = joblib.load(MODEL_DIR / f"xgb_imputer_seed{seed}.pkl")
        X_xgb = xgb_imp.transform(X_raw)
        col = f"p_xgb_seed{seed}"
        out[col] = xgb.predict_proba(X_xgb)[:, 1]
        xgb_cols.append(col)

    out["p_rf"] = out[rf_cols].mean(axis=1)
    out["p_xgb"] = out[xgb_cols].mean(axis=1)
    return out


def tcn_predict_windows(df_win: pd.DataFrame) -> pd.DataFrame:
    if torch is None:
        raise ImportError("PyTorch is required for TCN inference.")

    out = df_win[
        ["source_file", "domain_tag", "split_group", "run_id", "fault_id", "sample_idx", "y", "phase", "onset_step", "transition_len"]
    ].copy()
    out = out.rename(columns={"y": "y_true"})

    X = flattened_to_tensor(df_win)
    tcn_cols = []

    for seed in SEEDS:
        # 학습 코드에서 imputer + scaler를 둘 다 저장한 경우 대응
        imp_path = MODEL_DIR / f"tcn_imputer_seed{seed}.pkl"
        scaler_path = MODEL_DIR / f"tcn_scaler_seed{seed}.pkl"

        imputer = joblib.load(imp_path) if imp_path.exists() else None
        scaler = joblib.load(scaler_path)

        with open(MODEL_DIR / f"tcn_meta_seed{seed}.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        B, F, T = X.shape
        X2 = X.transpose(0, 2, 1).reshape(-1, F)

        if imputer is not None:
            X2 = imputer.transform(X2)
        X2 = scaler.transform(X2)
        Xs = X2.reshape(B, T, F).transpose(0, 2, 1)

        model = TCNClassifier(
            n_features=int(meta["n_features"]),
            channels=tuple(meta["channels"]),
            kernel_size=int(meta["kernel_size"]),
            dropout=float(meta["dropout"]),
        )
        state = torch.load(MODEL_DIR / f"tcn_model_seed{seed}.pt", map_location="cpu")
        model.load_state_dict(state)
        model.eval()

        with torch.no_grad():
            xb = torch.tensor(Xs, dtype=torch.float32)
            logits = model(xb)
            p = torch.sigmoid(logits).numpy()

        col = f"p_tcn_seed{seed}"
        out[col] = p
        tcn_cols.append(col)

    out["p_tcn"] = out[tcn_cols].mean(axis=1)
    return out


def align_tcn_to_rows(
    row_pred: pd.DataFrame,
    tcn_pred: pd.DataFrame,
    out_split_group: str,
) -> pd.DataFrame:
    # tcn_pred는 full-run test에서 왔으므로 split_group은 제거하고 key align
    tcn_keep = [
        "source_file",
        "run_id",
        "fault_id",
        "sample_idx",
    ] + [c for c in tcn_pred.columns if c.startswith("p_tcn_seed")] + ["p_tcn"]

    tcn_sub = tcn_pred[tcn_keep].drop_duplicates(
        subset=["source_file", "fault_id", "run_id", "sample_idx"]
    )

    merged = row_pred.merge(
        tcn_sub,
        on=["source_file", "fault_id", "run_id", "sample_idx"],
        how="left",
    )

    merged["split_group"] = out_split_group
    merged["p_ensemble"] = merged[["p_rf", "p_xgb", "p_tcn"]].mean(axis=1)
    return merged


def merge_base_predictions_same_rows(row_pred: pd.DataFrame, tcn_pred: pd.DataFrame) -> pd.DataFrame:
    key = ["source_file", "domain_tag", "split_group", "run_id", "fault_id", "sample_idx", "y_true", "phase", "onset_step", "transition_len"]
    keep_cols = key + [c for c in tcn_pred.columns if c.startswith("p_tcn_seed")] + ["p_tcn"]
    merged = row_pred.merge(tcn_pred[keep_cols], on=key, how="inner")
    merged["p_ensemble"] = merged[["p_rf", "p_xgb", "p_tcn"]].mean(axis=1)
    return merged


def save_pred(df: pd.DataFrame, filename: str) -> None:
    ensure_dir(PRED_DIR)
    df.to_csv(PRED_DIR / filename, index=False)
    print(f"Saved predictions: {filename} ({len(df):,})")


def main() -> None:
    ensure_dir(PRED_DIR)
    feature_cols = load_feature_cols()

    # val: 기존처럼 같은 row/window에서 바로 merge 가능
    row_val = read_rows("te_val_rows.csv")
    win_val = read_windows("te_val_windows.csv")
    pred_val = merge_base_predictions_same_rows(
        base_predict_rows(row_val, feature_cols),
        tcn_predict_windows(win_val),
    )
    save_pred(pred_val, "base_val_predictions.csv")

    # test: TCN은 full-run contiguous test에서 추론
    win_test_full = read_windows("te_test_full_windows_tcn.csv")
    pred_tcn_test_full = tcn_predict_windows(win_test_full)

    # 공통 4000 rows
    row_main = read_rows("te_test_main_rows.csv")
    pred_main_rows = base_predict_rows(row_main, feature_cols)
    pred_main = align_tcn_to_rows(pred_main_rows, pred_tcn_test_full, out_split_group="test_main")
    save_pred(pred_main, "base_test_main_predictions.csv")

    # 공통 500 rows
    row_cost = read_rows("te_test_cost_rows.csv")
    pred_cost_rows = base_predict_rows(row_cost, feature_cols)
    pred_cost = align_tcn_to_rows(pred_cost_rows, pred_tcn_test_full, out_split_group="test_cost")
    save_pred(pred_cost, "base_test_cost_predictions.csv")

    # 참고용: TCN alignment coverage 출력
    main_cov = float(pred_main["p_tcn"].notna().mean()) if len(pred_main) else 0.0
    cost_cov = float(pred_cost["p_tcn"].notna().mean()) if len(pred_cost) else 0.0
    print(f"TCN coverage on test_main rows: {main_cov:.4f}")
    print(f"TCN coverage on test_cost rows: {cost_cov:.4f}")
    print("Base inference completed.")


if __name__ == "__main__":
    main()
