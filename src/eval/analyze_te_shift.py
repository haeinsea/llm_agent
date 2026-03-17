from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import umap
    _HAS_UMAP = True
except Exception:
    _HAS_UMAP = False


# =========================
# Utils
# =========================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"phase", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def infer_feature_cols(df: pd.DataFrame) -> List[str]:
    excluded = {
        "source_file", "domain_tag", "split_group", "run_id", "fault_id",
        "sample_idx", "y", "phase", "onset_step", "transition_len", "is_faulty_file"
    }
    feat_cols = []
    for c in df.columns:
        if c in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            feat_cols.append(c)
    if not feat_cols:
        raise ValueError("No numeric feature columns found.")
    return feat_cols


def prepare_feature_matrix(df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    X = df[feature_cols].to_numpy(dtype=float)
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X


def rbf_mmd2_unbiased(X: np.ndarray, Y: np.ndarray, gamma: float | None = None) -> float:
    """
    Unbiased MMD^2 with RBF kernel.
    X: (n, d), Y: (m, d)
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    n = X.shape[0]
    m = Y.shape[0]

    if n < 2 or m < 2:
        return np.nan

    Z = np.vstack([X, Y])

    if gamma is None:
        # median heuristic
        sample_n = min(len(Z), 1000)
        idx = np.random.choice(len(Z), size=sample_n, replace=False)
        Zs = Z[idx]
        dists = np.sum((Zs[:, None, :] - Zs[None, :, :]) ** 2, axis=2)
        tri = dists[np.triu_indices_from(dists, k=1)]
        med = np.median(tri[tri > 0]) if np.any(tri > 0) else 1.0
        gamma = 1.0 / max(2.0 * med, 1e-12)

    XX = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    YY = np.sum((Y[:, None, :] - Y[None, :, :]) ** 2, axis=2)
    XY = np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2)

    Kxx = np.exp(-gamma * XX)
    Kyy = np.exp(-gamma * YY)
    Kxy = np.exp(-gamma * XY)

    term_x = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
    term_y = (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
    term_xy = Kxy.mean()

    return float(term_x + term_y - 2.0 * term_xy)


def centroid_distance(emb: np.ndarray, labels: np.ndarray, a: str, b: str) -> float:
    ea = emb[labels == a]
    eb = emb[labels == b]
    if len(ea) == 0 or len(eb) == 0:
        return np.nan
    ca = ea.mean(axis=0)
    cb = eb.mean(axis=0)
    return float(np.linalg.norm(ca - cb))


def sample_for_embedding(
    X: np.ndarray,
    labels: np.ndarray,
    max_per_group: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    keep_idx = []

    for g in np.unique(labels):
        idx = np.where(labels == g)[0]
        if len(idx) > max_per_group:
            idx = rng.choice(idx, size=max_per_group, replace=False)
        keep_idx.extend(idx.tolist())

    keep_idx = np.array(sorted(keep_idx))
    return X[keep_idx], labels[keep_idx]


# =========================
# Statistics
# =========================
def compute_feature_shift_table(
    df_ref: pd.DataFrame,
    df_cmp: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    rows = []

    for feat in feature_cols:
        x = df_ref[feat].astype(float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        y = df_cmp[feat].astype(float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()

        if len(x) == 0 or len(y) == 0:
            rows.append({
                "feature": feat,
                "mean_ref": np.nan,
                "mean_cmp": np.nan,
                "mean_abs_diff": np.nan,
                "std_ref": np.nan,
                "std_cmp": np.nan,
                "std_abs_diff": np.nan,
                "ks_stat": np.nan,
                "ks_pvalue": np.nan,
                "wasserstein": np.nan,
            })
            continue

        ks = ks_2samp(x, y)
        rows.append({
            "feature": feat,
            "mean_ref": float(np.mean(x)),
            "mean_cmp": float(np.mean(y)),
            "mean_abs_diff": float(abs(np.mean(x) - np.mean(y))),
            "std_ref": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
            "std_cmp": float(np.std(y, ddof=1)) if len(y) > 1 else 0.0,
            "std_abs_diff": float(abs(np.std(x, ddof=1) - np.std(y, ddof=1))) if len(x) > 1 and len(y) > 1 else np.nan,
            "ks_stat": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
            "wasserstein": float(wasserstein_distance(x, y)),
        })

    out = pd.DataFrame(rows)
    out["ks_reject_0.05"] = out["ks_pvalue"] < 0.05
    return out


def compute_summary_table(
    feature_shift_df: pd.DataFrame,
    X_ref: np.ndarray,
    X_cmp: np.ndarray,
    ref_name: str,
    cmp_name: str,
) -> pd.DataFrame:
    row = {
        "comparison": f"{ref_name} vs {cmp_name}",
        "n_features": int(len(feature_shift_df)),
        "avg_mean_abs_diff": float(feature_shift_df["mean_abs_diff"].mean()),
        "avg_std_abs_diff": float(feature_shift_df["std_abs_diff"].mean()),
        "avg_ks_stat": float(feature_shift_df["ks_stat"].mean()),
        "median_ks_pvalue": float(feature_shift_df["ks_pvalue"].median()),
        "ks_reject_ratio_0.05": float(feature_shift_df["ks_reject_0.05"].mean()),
        "avg_wasserstein": float(feature_shift_df["wasserstein"].mean()),
        "mmd_rbf": float(rbf_mmd2_unbiased(X_ref, X_cmp)),
    }
    return pd.DataFrame([row])


# =========================
# Plots
# =========================
def plot_kde_grid(
    df: pd.DataFrame,
    feature_cols: List[str],
    out_path: Path,
    top_k: int = 12,
    compare_mode: str = "normal_vs_shift",
) -> None:
    import seaborn as sns

    if compare_mode == "normal_vs_shift":
        plot_df = df[df["phase"].isin(["normal", "transition", "post_shift"])].copy()
        plot_df["plot_group"] = np.where(plot_df["phase"] == "normal", "normal", "shift")
    else:
        plot_df = df[df["phase"].isin(["normal", "transition", "post_shift"])].copy()
        plot_df["plot_group"] = plot_df["phase"]

    # variance 큰 feature 위주로 보기
    var_rank = plot_df[feature_cols].var(numeric_only=True).sort_values(ascending=False)
    selected = var_rank.head(min(top_k, len(var_rank))).index.tolist()

    n = len(selected)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, feat in zip(axes, selected):
        for grp in sorted(plot_df["plot_group"].dropna().unique()):
            sub = plot_df.loc[plot_df["plot_group"] == grp, feat].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
            if len(sub) < 2:
                continue
            sns.kdeplot(sub, ax=ax, label=grp, fill=False, common_norm=False)
        ax.set_title(feat)
        ax.legend()

    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_embedding(
    emb: np.ndarray,
    labels: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    plt.figure(figsize=(8, 6))
    unique = list(pd.unique(labels))
    for g in unique:
        idx = labels == g
        plt.scatter(emb[idx, 0], emb[idx, 1], s=10, alpha=0.6, label=g)
    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True, help="e.g., data/processed/te_test_main_rows.csv")
    parser.add_argument("--out_dir", type=str, default="outputs/shift_analysis")
    parser.add_argument("--max_embed_per_group", type=int, default=1500)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--top_k_kde_features", type=int, default=12)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    df = load_df(args.input_csv)
    feature_cols = infer_feature_cols(df)

    # group filters
    df_n = df[df["phase"] == "normal"].copy()
    df_t = df[df["phase"] == "transition"].copy()
    df_p = df[df["phase"] == "post_shift"].copy()
    df_s = df[df["phase"].isin(["transition", "post_shift"])].copy()

    if len(df_n) == 0 or len(df_s) == 0:
        raise ValueError("Need both normal and shift samples.")
    if len(df_t) == 0 or len(df_p) == 0:
        print("Warning: transition or post_shift group is empty. Some plots/distances may be skipped.")

    # standardized matrices for multivariate distances
    X_all = prepare_feature_matrix(df, feature_cols)
    X_n = X_all[df["phase"].to_numpy() == "normal"]
    X_t = X_all[df["phase"].to_numpy() == "transition"]
    X_p = X_all[df["phase"].to_numpy() == "post_shift"]
    X_s = X_all[np.isin(df["phase"].to_numpy(), ["transition", "post_shift"])]

    # 1) feature-wise tables
    fs_normal_shift = compute_feature_shift_table(df_n, df_s, feature_cols)
    fs_normal_transition = compute_feature_shift_table(df_n, df_t, feature_cols) if len(df_t) else pd.DataFrame()
    fs_normal_post = compute_feature_shift_table(df_n, df_p, feature_cols) if len(df_p) else pd.DataFrame()

    fs_normal_shift.to_csv(out_dir / "feature_shift_normal_vs_shift.csv", index=False)
    if len(fs_normal_transition):
        fs_normal_transition.to_csv(out_dir / "feature_shift_normal_vs_transition.csv", index=False)
    if len(fs_normal_post):
        fs_normal_post.to_csv(out_dir / "feature_shift_normal_vs_post_shift.csv", index=False)

    # 2) summary tables
    summary_rows = []
    summary_rows.append(compute_summary_table(fs_normal_shift, X_n, X_s, "normal", "shift"))

    if len(df_t):
        summary_rows.append(compute_summary_table(fs_normal_transition, X_n, X_t, "normal", "transition"))
    if len(df_p):
        summary_rows.append(compute_summary_table(fs_normal_post, X_n, X_p, "normal", "post_shift"))

    summary_df = pd.concat(summary_rows, ignore_index=True)

    # 3) embeddings
    emb_labels = df["phase"].astype(str).to_numpy()
    X_emb, y_emb = sample_for_embedding(
        X_all,
        emb_labels,
        max_per_group=args.max_embed_per_group,
        random_state=args.random_state,
    )

    # PCA
    pca = PCA(n_components=2, random_state=args.random_state)
    pca_emb = pca.fit_transform(X_emb)
    plot_embedding(
        pca_emb,
        y_emb,
        "PCA of TEP Samples under Temporal Shift",
        out_dir / "pca_phase_scatter.png",
    )

    # t-SNE
    tsne = TSNE(
        n_components=2,
        random_state=args.random_state,
        perplexity=min(30, max(5, len(X_emb) // 20)),
        init="pca",
        learning_rate="auto",
    )
    tsne_emb = tsne.fit_transform(X_emb)
    plot_embedding(
        tsne_emb,
        y_emb,
        "t-SNE of TEP Samples under Temporal Shift",
        out_dir / "tsne_phase_scatter.png",
    )

    # UMAP
    umap_emb = None
    if _HAS_UMAP:
        reducer = umap.UMAP(
            n_components=2,
            random_state=args.random_state,
            n_neighbors=15,
            min_dist=0.1,
        )
        umap_emb = reducer.fit_transform(X_emb)
        plot_embedding(
            umap_emb,
            y_emb,
            "UMAP of TEP Samples under Temporal Shift",
            out_dir / "umap_phase_scatter.png",
        )

    # 4) centroid distances
    centroid_rows = []

    for name, emb in [("PCA", pca_emb), ("tSNE", tsne_emb)]:
        centroid_rows.append({
            "embedding": name,
            "normal_vs_transition": centroid_distance(emb, y_emb, "normal", "transition"),
            "normal_vs_post_shift": centroid_distance(emb, y_emb, "normal", "post_shift"),
            "transition_vs_post_shift": centroid_distance(emb, y_emb, "transition", "post_shift"),
        })

    if umap_emb is not None:
        centroid_rows.append({
            "embedding": "UMAP",
            "normal_vs_transition": centroid_distance(umap_emb, y_emb, "normal", "transition"),
            "normal_vs_post_shift": centroid_distance(umap_emb, y_emb, "normal", "post_shift"),
            "transition_vs_post_shift": centroid_distance(umap_emb, y_emb, "transition", "post_shift"),
        })

    centroid_df = pd.DataFrame(centroid_rows)
    centroid_df.to_csv(out_dir / "embedding_centroid_distances.csv", index=False)

    # centroid distance summary merge
    # normal vs shift는 transition/post를 분리해서 보는 게 더 논문 친화적이라 summary에 평균형으로 추가
    summary_df["pca_centroid_normal_vs_transition"] = centroid_df.loc[centroid_df["embedding"] == "PCA", "normal_vs_transition"].iloc[0] if "PCA" in centroid_df["embedding"].values else np.nan
    summary_df["pca_centroid_normal_vs_post_shift"] = centroid_df.loc[centroid_df["embedding"] == "PCA", "normal_vs_post_shift"].iloc[0] if "PCA" in centroid_df["embedding"].values else np.nan
    summary_df["tsne_centroid_normal_vs_transition"] = centroid_df.loc[centroid_df["embedding"] == "tSNE", "normal_vs_transition"].iloc[0] if "tSNE" in centroid_df["embedding"].values else np.nan
    summary_df["tsne_centroid_normal_vs_post_shift"] = centroid_df.loc[centroid_df["embedding"] == "tSNE", "normal_vs_post_shift"].iloc[0] if "tSNE" in centroid_df["embedding"].values else np.nan

    if "UMAP" in centroid_df["embedding"].values:
        summary_df["umap_centroid_normal_vs_transition"] = centroid_df.loc[centroid_df["embedding"] == "UMAP", "normal_vs_transition"].iloc[0]
        summary_df["umap_centroid_normal_vs_post_shift"] = centroid_df.loc[centroid_df["embedding"] == "UMAP", "normal_vs_post_shift"].iloc[0]

    summary_df.to_csv(out_dir / "shift_summary_table.csv", index=False)

    # 5) KDE plots
    plot_kde_grid(
        df,
        feature_cols,
        out_dir / "kde_normal_vs_shift.png",
        top_k=args.top_k_kde_features,
        compare_mode="normal_vs_shift",
    )
    plot_kde_grid(
        df,
        feature_cols,
        out_dir / "kde_normal_transition_post.png",
        top_k=args.top_k_kde_features,
        compare_mode="three_phase",
    )

    # 6) save meta
    meta = {
        "input_csv": args.input_csv,
        "n_rows_total": int(len(df)),
        "n_features": int(len(feature_cols)),
        "phase_counts": df["phase"].value_counts().to_dict(),
        "feature_cols": feature_cols,
        "has_umap": _HAS_UMAP,
    }
    with open(out_dir / "analysis_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(f"Saved to: {out_dir}")
    print("\nMain outputs:")
    print("- shift_summary_table.csv")
    print("- feature_shift_normal_vs_shift.csv")
    print("- embedding_centroid_distances.csv")
    print("- kde_normal_vs_shift.png")
    print("- kde_normal_transition_post.png")
    print("- pca_phase_scatter.png")
    print("- tsne_phase_scatter.png")
    if _HAS_UMAP:
        print("- umap_phase_scatter.png")


if __name__ == "__main__":
    main()