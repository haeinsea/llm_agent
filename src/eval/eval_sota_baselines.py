from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.routing.selective_llm_eval import DEFAULT_Q, read_selected_q
from src.utils.experiment import get_seed_list
from src.utils.io import ensure_dir, read_csv, read_yaml, write_csv
from src.utils.metrics import binary_metrics, instability_score, low_tail_recall, prr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PRED_DIR = OUTPUT_DIR / "predictions"
METRIC_DIR = OUTPUT_DIR / "metrics"
APPENDIX_DIR = OUTPUT_DIR / "appendix"
CONFIG_DIR = PROJECT_ROOT / "configs"
SEEDS = get_seed_list()


def fmt(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def best_tau(y_true: np.ndarray, p: np.ndarray) -> float:
    grid = np.linspace(0.01, 0.99, 99)
    best_tau_val = 0.5
    best_f1 = -1.0
    best_recall = -1.0
    for tau in grid:
        m = binary_metrics(y_true, p, tau=float(tau))
        f1 = float(m["f1"])
        recall = float(m["recall"])
        if f1 > best_f1 or (np.isclose(f1, best_f1) and recall > best_recall):
            best_tau_val = float(tau)
            best_f1 = f1
            best_recall = recall
    return best_tau_val


def build_table_a0() -> pd.DataFrame:
    selected_q = read_selected_q(DEFAULT_Q)
    tcn_cfg = read_yaml(CONFIG_DIR / "train_tcn.yaml", default={})
    tta_cfg = read_yaml(CONFIG_DIR / "train_adaptable.yaml", default={})
    inv_cfg = read_yaml(CONFIG_DIR / "train_invariant.yaml", default={})
    graphad_cfg = read_yaml(CONFIG_DIR / "train_graphad.yaml", default={})
    gray_grid = read_csv(METRIC_DIR / "grayzone_grid.csv")
    default_candidates = gray_grid[np.isclose(gray_grid["q"], selected_q)] if not gray_grid.empty else pd.DataFrame()
    default_row = default_candidates.iloc[0] if not default_candidates.empty else None
    margin = float(default_row["gray_margin_mean"]) if default_row is not None else 0.0
    return pd.DataFrame(
        [
            {
                "Method": "Base Model(ModernTCN)",
                "Category": "Static",
                "Rationale for Selection": "Serves as the primary temporal baseline without explicit temporal-shift adaptation.",
                "Key Hyperparameters": (
                    f"Channels: {list(tcn_cfg.get('channels', [64, 96, 128]))}, "
                    f"Dilations: {list(tcn_cfg.get('dilations', [1, 2, 4]))}, "
                    f"Kernel size: {int(tcn_cfg.get('kernel_size', 3))}, "
                    f"Dropout: {float(tcn_cfg.get('dropout', 0.1)):.1f}"
                ),
            },
            {
                "Method": "AdapTable (2024)",
                "Category": "TTA",
                "Rationale for Selection": "An AdapTable-inspired test-time adaptation baseline adapted to the ModernTCN backbone for temporal shift handling.",
                "Key Hyperparameters": f"Learning rate: {float(tta_cfg.get('adaptation_lr', 1e-4)):.0e}, Update steps: {int(tta_cfg.get('adaptation_steps', 1))}",
            },
            {
                "Method": "Cao et al. (2023)",
                "Category": "Invariant",
                "Rationale for Selection": "An ICCV 2023-style shift-robust baseline approximated here with phase-aware invariant feature regularization over temporal environments.",
                "Key Hyperparameters": f"Penalty weight: {float(inv_cfg.get('penalty_weight', 0.1)):.1f}, Domain count: {int(inv_cfg.get('domain_count', 3))}",
            },
            {
                "Method": "UTAR (Ours)",
                "Category": "Hybrid",
                "Rationale for Selection": "The proposed framework integrating gray-zone isolation and selective LLM routing.",
                "Key Hyperparameters": (
                    f"Margin (q={selected_q:.2f}): {margin:.4f}, "
                    f"Sigmoid gain: {float(read_yaml(CONFIG_DIR / 'routing.yaml', default={}).get('sigmoid_gain', 5.0)):.1f}, "
                    f"Entropy/Discrepancy quantile: 0.8"
                ),
            },
        ]
    )


def evaluate_seeded_method(method: str, val_df: pd.DataFrame, test_df: pd.DataFrame, col_prefix: str) -> pd.DataFrame:
    rows = []
    y_val = val_df["y_true"].to_numpy().astype(int)
    y_test = test_df["y_true"].to_numpy().astype(int)
    missing = []
    for seed in SEEDS:
        val_col = f"{col_prefix}_seed{seed}"
        test_col = f"{col_prefix}_seed{seed}"
        if val_col not in val_df.columns or test_col not in test_df.columns:
            missing.append(seed)
            continue
        p_val = val_df[val_col].to_numpy(dtype=float)
        p_test = test_df[test_col].to_numpy(dtype=float)
        tau = best_tau(y_val, p_val)
        val_metrics = binary_metrics(y_val, p_val, tau=tau)
        test_metrics = binary_metrics(y_test, p_test, tau=tau)
        rows.append(
            {
                "Method": method,
                "seed": seed,
                "tau": tau,
                "f1_iid": float(val_metrics["f1"]),
                "f1_tds": float(test_metrics["f1"]),
                "performance_drop": float(val_metrics["f1"] - test_metrics["f1"]),
                "worst_case_recall": low_tail_recall(y_test, p_test, test_df["run_id"], tau=tau, window=50, quantile=0.05),
                "instability": instability_score(p_test, test_df["run_id"], test_df["phase"]),
                "prr": prr(float(val_metrics["recall"]), float(test_metrics["recall"])),
                "recall_tds": float(test_metrics["recall"]),
                "precision_tds": float(test_metrics["precision"]),
                "auc_tds": float(test_metrics["roc_auc"]) if test_metrics["roc_auc"] is not None else np.nan,
            }
        )
    if missing:
        raise ValueError(f"{method} is missing prediction columns for seeds {missing}. Regenerate the full 10-seed prediction files first.")
    return pd.DataFrame(rows)


def summarize_method_rows(detail_df: pd.DataFrame, method_name: str) -> dict:
    group = detail_df[detail_df["Method"] == method_name]
    if group.empty:
        raise ValueError(f"No rows found for method {method_name}.")
    if group["seed"].nunique() != len(SEEDS):
        raise ValueError(f"{method_name} has {group['seed'].nunique()} seeds, but {len(SEEDS)} are required by configs/experiment.yaml.")
    return {
        "Method": method_name,
        "F1-Score (IID)": fmt(group["f1_iid"].mean(), group["f1_iid"].std(ddof=1) if len(group) > 1 else 0.0),
        "F1-Score (TDS)": fmt(group["f1_tds"].mean(), group["f1_tds"].std(ddof=1) if len(group) > 1 else 0.0),
        "AUC (TDS)": fmt(group["auc_tds"].mean(), group["auc_tds"].std(ddof=1) if len(group) > 1 else 0.0) if group["auc_tds"].notna().any() else "NA",
        "Performance Drop(Δ)": fmt(group["performance_drop"].mean(), group["performance_drop"].std(ddof=1) if len(group) > 1 else 0.0),
        "Worst-Case Recall (P5)": fmt(group["worst_case_recall"].mean(), group["worst_case_recall"].std(ddof=1) if len(group) > 1 else 0.0),
        "Instability(Var)": fmt(group["instability"].mean(), group["instability"].std(ddof=1) if len(group) > 1 else 0.0),
    }


def _require_utar_rows_at_selected_q(utar_summary: pd.DataFrame, utar_seed: pd.DataFrame, selected_q: float) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    utar_val = utar_summary[
        (utar_summary["dataset"] == "val")
        & (utar_summary["mode"] == "selective")
        & (np.isclose(utar_summary["q"], selected_q))
    ]
    utar_main_detail = utar_seed[
        (utar_seed["dataset"] == "main")
        & (utar_seed["mode"] == "selective")
        & (np.isclose(utar_seed["q"], selected_q))
    ].copy()
    utar_val_detail = utar_seed[
        (utar_seed["dataset"] == "val")
        & (utar_seed["mode"] == "selective")
        & (np.isclose(utar_seed["q"], selected_q))
    ].copy()

    missing = []
    if utar_val.empty:
        available_val_q = sorted(utar_summary[(utar_summary["dataset"] == "val") & (utar_summary["mode"] == "selective")]["q"].unique().tolist())
        missing.append(f"val selective summary @ q={selected_q:.2f} (available val q: {available_val_q})")
    if utar_main_detail.empty:
        available_main_q = sorted(utar_seed[(utar_seed["dataset"] == "main") & (utar_seed["mode"] == "selective")]["q"].unique().tolist())
        missing.append(f"main selective seed detail @ q={selected_q:.2f} (available main q: {available_main_q})")
    if utar_val_detail.empty:
        available_val_seed_q = sorted(utar_seed[(utar_seed["dataset"] == "val") & (utar_seed["mode"] == "selective")]["q"].unique().tolist())
        missing.append(f"val selective seed detail @ q={selected_q:.2f} (available val q: {available_val_seed_q})")
    if missing:
        raise ValueError(
            "UTAR selected-q artifacts are incomplete for Appendix A1: "
            + "; ".join(missing)
            + ". Recompute them with `python -m src.routing.selective_llm_eval_main --selected-q-only --modes selective` and rerun `python -m src.eval.eval_sota_baselines`."
        )

    return utar_val.iloc[0], utar_main_detail, utar_val_detail


def main() -> None:
    ensure_dir(APPENDIX_DIR)
    ensure_dir(METRIC_DIR)
    selected_q = read_selected_q(DEFAULT_Q)
    table_a0 = build_table_a0()
    write_csv(APPENDIX_DIR / "table_a0_baseline_descriptions.csv", table_a0)

    base_val = read_csv(PRED_DIR / "base_val_predictions.csv")
    base_main = read_csv(PRED_DIR / "base_test_main_predictions.csv")
    sota_val = read_csv(PRED_DIR / "sota_val_predictions.csv")
    sota_main = read_csv(PRED_DIR / "sota_test_main_predictions.csv")

    detail_parts = [
        evaluate_seeded_method("ModernTCN", base_val, base_main, "p_tcn"),
        evaluate_seeded_method("AdapTable (2024)", sota_val, sota_main, "p_adaptable"),
        evaluate_seeded_method("Cao et al. (2023)", sota_val, sota_main, "p_invariant"),
    ]
    detail_df = pd.concat(detail_parts, ignore_index=True)

    include_utar = False
    utar_summary_path = METRIC_DIR / "selective_llm_summary.csv"
    utar_seed_path = METRIC_DIR / "selective_llm_seed_metrics.csv"
    if utar_summary_path.exists() and utar_seed_path.exists():
        utar_summary = read_csv(utar_summary_path)
        utar_seed = read_csv(utar_seed_path)
        utar_val, utar_main_detail, utar_val_detail = _require_utar_rows_at_selected_q(utar_summary, utar_seed, selected_q)
        if utar_main_detail["seed"].nunique() != len(SEEDS):
            raise ValueError(f"UTAR main selective metrics have {utar_main_detail['seed'].nunique()} seeds, but {len(SEEDS)} are required.")
        utar_val_by_seed = {int(row["seed"]): row for _, row in utar_val_detail.iterrows()}
        utar_rows = []
        for _, row in utar_main_detail.iterrows():
            seed = int(row["seed"])
            val_row = utar_val_by_seed.get(seed)
            f1_iid = float(val_row["f1"]) if val_row is not None else float(utar_val["f1_mean"])
            utar_rows.append(
                {
                    "Method": "UTAR (Ours)",
                    "seed": seed,
                    "tau": float(row["tau"]),
                    "f1_iid": f1_iid,
                    "f1_tds": float(row["f1"]),
                    "performance_drop": float(f1_iid - row["f1"]),
                    "worst_case_recall": float(row["worst_case_recall"]),
                    "instability": float(row["instability"]),
                    "prr": float(row["prr"]),
                    "recall_tds": float(row["recall"]),
                    "precision_tds": float(row["precision"]),
                    "auc_tds": float(row["roc_auc"]) if not pd.isna(row["roc_auc"]) else np.nan,
                }
            )
        detail_df = pd.concat([detail_df, pd.DataFrame(utar_rows)], ignore_index=True)
        include_utar = True
    else:
        print("UTAR selective metrics not found; writing Appendix A1 with ModernTCN and SOTA baselines only.", flush=True)
    write_csv(METRIC_DIR / "sota_seed_metrics.csv", detail_df)

    summary_rows = [
        summarize_method_rows(detail_df, "ModernTCN"),
        summarize_method_rows(detail_df, "AdapTable (2024)"),
        summarize_method_rows(detail_df, "Cao et al. (2023)"),
    ]
    if include_utar:
        summary_rows.append(summarize_method_rows(detail_df, "UTAR (Ours)"))
    summary_df = pd.DataFrame(summary_rows)
    write_csv(METRIC_DIR / "table_a1_sota_comparison.csv", summary_df)
    write_csv(APPENDIX_DIR / "table_a1_supplementary_experimental_results.csv", summary_df)
    write_csv(APPENDIX_DIR / "table_a1_supplementary_seed_detail.csv", detail_df)
    print("Saved Appendix A0/A1 baseline artifacts.")


if __name__ == "__main__":
    main()
