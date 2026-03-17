from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
META_DIR = DATA_DIR / "meta"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
METRIC_DIR = OUTPUT_DIR / "metrics"


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)


def load_feature_cols() -> List[str]:
    with open(META_DIR / "feature_columns.json", "r", encoding="utf-8") as f:
        return json.load(f)


def standardized_mean_diff(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    ma, mb = np.nanmean(a), np.nanmean(b)
    सा, sb = np.nanstd(a), np.nanstd(b)
    pooled = np.sqrt((सा**2 + sb**2) / 2.0)
    if pooled < 1e-12:
        return 0.0
    return float((mb - ma) / pooled)


def plot_phase_distribution(df_val: pd.DataFrame, df_main: pd.DataFrame, df_cost: pd.DataFrame) -> None:
    phase_order = ["normal", "pre", "transition", "post_shift"]

    vc_val = df_val["phase"].value_counts().reindex(phase_order, fill_value=0)
    vc_main = df_main["phase"].value_counts().reindex(phase_order, fill_value=0)
    vc_cost = df_cost["phase"].value_counts().reindex(phase_order, fill_value=0)

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    x = np.arange(len(phase_order))
    w = 0.25

    ax.bar(x - w, vc_val.values, width=w, label="Validation")
    ax.bar(x, vc_main.values, width=w, label="Test Main")
    ax.bar(x + w, vc_cost.values, width=w, label="Test Cost")

    ax.set_xticks(x)
    ax.set_xticklabels(phase_order)
    ax.set_ylabel("Count")
    ax.set_title("Phase distribution across splits")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_split_phase_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fault_distribution(df_val: pd.DataFrame, df_main: pd.DataFrame, df_cost: pd.DataFrame) -> None:
    val_vc = df_val["fault_id"].value_counts().sort_index()
    main_vc = df_main["fault_id"].value_counts().sort_index()
    cost_vc = df_cost["fault_id"].value_counts().sort_index()

    fault_ids = sorted(set(val_vc.index).union(main_vc.index).union(cost_vc.index))
    val_y = np.array([val_vc.get(fid, 0) for fid in fault_ids])
    main_y = np.array([main_vc.get(fid, 0) for fid in fault_ids])
    cost_y = np.array([cost_vc.get(fid, 0) for fid in fault_ids])

    fig, ax = plt.subplots(figsize=(11.0, 5.5))
    x = np.arange(len(fault_ids))
    w = 0.25

    ax.bar(x - w, val_y, width=w, label="Validation")
    ax.bar(x, main_y, width=w, label="Test Main")
    ax.bar(x + w, cost_y, width=w, label="Test Cost")

    ax.set_xticks(x)
    ax.set_xticklabels(fault_ids, rotation=0)
    ax.set_xlabel("Fault ID")
    ax.set_ylabel("Count")
    ax.set_title("Fault distribution across splits")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_split_fault_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pca_shift(df_train: pd.DataFrame, df_main: pd.DataFrame, feature_cols: List[str]) -> None:
    n_train = min(1500, len(df_train))
    n_main = min(1500, len(df_main))

    train_s = df_train.sample(n=n_train, random_state=42) if len(df_train) > n_train else df_train.copy()
    main_s = df_main.sample(n=n_main, random_state=42) if len(df_main) > n_main else df_main.copy()

    X = pd.concat([train_s[feature_cols], main_s[feature_cols]], axis=0).to_numpy()
    pca = PCA(n_components=2, random_state=42)
    emb = pca.fit_transform(X)

    n1 = len(train_s)
    emb_train = emb[:n1]
    emb_main = emb[n1:]

    phase = main_s["phase"].to_numpy()
    mask_normal = phase == "normal"
    mask_transition = phase == "transition"
    mask_post = phase == "post_shift"

    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    ax.scatter(emb_train[:, 0], emb_train[:, 1], s=12, alpha=0.28, label="Train")
    ax.scatter(emb_main[mask_normal, 0], emb_main[mask_normal, 1], s=16, alpha=0.40, label="Test Main Normal")
    ax.scatter(emb_main[mask_transition, 0], emb_main[mask_transition, 1], s=18, alpha=0.55, label="Transition")
    ax.scatter(emb_main[mask_post, 0], emb_main[mask_post, 1], s=18, alpha=0.55, label="Post-shift")

    ax.set_title("PCA view: train vs test_main")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_shift_pca_train_vs_testmain.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_top_feature_drift(df_train: pd.DataFrame, df_main: pd.DataFrame, feature_cols: List[str], top_k: int = 20) -> None:
    # compare train vs test_main(post_shift + transition)
    main_shift = df_main[df_main["phase"].isin(["transition", "post_shift"])].copy()

    rows = []
    for feat in feature_cols:
        smd = standardized_mean_diff(df_train[feat].to_numpy(), main_shift[feat].to_numpy())
        rows.append({"feature": feat, "smd": smd, "abs_smd": abs(smd)})

    drift_df = pd.DataFrame(rows).sort_values("abs_smd", ascending=False).reset_index(drop=True)
    drift_df.to_csv(METRIC_DIR / "feature_drift_summary.csv", index=False)

    top = drift_df.head(top_k).sort_values("abs_smd", ascending=True)

    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    ax.barh(top["feature"], top["abs_smd"])
    ax.set_xlabel("Absolute standardized mean difference")
    ax.set_ylabel("Feature")
    ax.set_title(f"Top-{top_k} feature drifts: train vs test_main shift")
    ax.grid(axis="x", alpha=0.2)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_shift_top_feature_drift.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_split_summary(df_train: pd.DataFrame, df_val: pd.DataFrame, df_main: pd.DataFrame, df_cost: pd.DataFrame) -> None:
    rows = []
    for name, df in [
        ("train", df_train),
        ("val", df_val),
        ("test_main", df_main),
        ("test_cost", df_cost),
    ]:
        rows.append({
            "split": name,
            "n_rows": len(df),
            "positive_ratio": float(df["y"].mean()),
            "n_normal": int((df["phase"] == "normal").sum()) if "phase" in df.columns else np.nan,
            "n_transition": int((df["phase"] == "transition").sum()) if "phase" in df.columns else np.nan,
            "n_post_shift": int((df["phase"] == "post_shift").sum()) if "phase" in df.columns else np.nan,
            "n_fault_ids": int(df["fault_id"].nunique()),
            "n_runs": int(df["run_id"].nunique()),
        })
    pd.DataFrame(rows).to_csv(METRIC_DIR / "split_summary.csv", index=False)


def main() -> None:
    ensure_dirs()
    feature_cols = load_feature_cols()

    df_train = pd.read_csv(PROCESSED_DIR / "te_train_rows.csv")
    df_val = pd.read_csv(PROCESSED_DIR / "te_val_rows.csv")
    df_main = pd.read_csv(PROCESSED_DIR / "te_test_main_rows.csv")
    df_cost = pd.read_csv(PROCESSED_DIR / "te_test_cost_rows.csv")

    save_split_summary(df_train, df_val, df_main, df_cost)
    plot_phase_distribution(df_val, df_main, df_cost)
    plot_fault_distribution(df_val, df_main, df_cost)
    plot_pca_shift(df_train, df_main, feature_cols)
    plot_top_feature_drift(df_train, df_main, feature_cols, top_k=20)

    print("Shift inspection figures and summaries saved.")


if __name__ == "__main__":
    main()