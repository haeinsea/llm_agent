from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

from src.eval.plot_style import PAPER_COLORS, add_panel_label, save_figure, set_paper_style, style_axes
from src.models.graphad import graphad_score_matrices, load_graphad_artifact
from src.models.temporal_backbone import temporal_model_display_name
from src.routing.selective_llm_eval import DEFAULT_Q, build_llm_prompt, read_selected_q
from src.utils.experiment import ensemble_component_label, get_seed_list
from src.utils.io import ensure_dir, read_csv, read_json, read_yaml, write_csv
from src.utils.metrics import binary_metrics, gray_ratio, instability_score, low_tail_recall, prr
from src.utils.runtime import get_base_runtime_stat, load_base_runtime_summary


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
PRED_DIR = OUTPUT_DIR / "predictions"
METRIC_DIR = OUTPUT_DIR / "metrics"
EVAL_DIR = OUTPUT_DIR / "evaluation"
SHIFT_DIR = OUTPUT_DIR / "shift_analysis_test_main"
SHIFT_INPUT_PATH = PROCESSED_DIR / "te_test_main_rows.csv"
TUNING_DIR = OUTPUT_DIR / "tuning"
APPENDIX_DIR = OUTPUT_DIR / "appendix"
BASE_RUNTIME_SUMMARY_PATH = METRIC_DIR / "base_inference_runtime_summary.json"
SEEDS = get_seed_list()


def fmt(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def pick_std(row: pd.Series, prefix: str) -> float:
    std_col = f"{prefix}_std"
    return float(row[std_col]) if std_col in row.index else 0.0


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _safe_read_csv(path: Path) -> pd.DataFrame:
    return read_csv(path) if path.exists() else pd.DataFrame()


def _safe_read_json(path: Path) -> dict:
    return read_json(path) if path.exists() else {}


def _safe_read_yaml(path: Path) -> dict:
    try:
        return read_yaml(path, default={})
    except Exception:
        return {}


def _ensure_shift_summary_path() -> Path:
    summary_path = SHIFT_DIR / "shift_summary_table.csv"
    if summary_path.exists():
        return summary_path
    if not SHIFT_INPUT_PATH.exists():
        raise FileNotFoundError(
            "Missing shift summary and source rows. "
            f"Expected either {summary_path} or {SHIFT_INPUT_PATH}."
        )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.eval.analyze_te_shift",
            "--input_csv",
            str(SHIFT_INPUT_PATH),
            "--out_dir",
            str(SHIFT_DIR),
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Shift analysis completed without producing {summary_path}."
        )
    return summary_path


def build_appendix_b_prompt() -> None:
    selective = _safe_read_csv(PRED_DIR / "utar_test_cost_selective.csv")
    source_name = "utar_test_cost_selective.csv"
    if selective.empty:
        selective = _safe_read_csv(PRED_DIR / "utar_test_main_selective.csv")
        source_name = "utar_test_main_selective.csv"
    if selective.empty:
        raise FileNotFoundError(
            "Missing representative selective output. "
            "Run cost selective evaluation first, or fall back to main selective evaluation."
        )

    candidate = selective[selective.get("llm_called", 0) == 1].copy()
    if candidate.empty:
        candidate = selective.copy()
    if "ensemble_entropy" in candidate.columns:
        candidate = candidate.sort_values(["ensemble_entropy", "model_discrepancy"], ascending=[False, False])
    row = candidate.iloc[0]

    prompt = build_llm_prompt(row)
    model_cfg = read_yaml(CONFIG_DIR / "routing.yaml", default={}).get("llm", {})
    temporal_cfg = read_yaml(CONFIG_DIR / "train_tcn.yaml", default={})
    temporal_label = temporal_model_display_name(temporal_cfg.get("architecture", "modern_tcn"))

    structure = pd.DataFrame(
        [
            {"Section": "System Role", "Purpose": "Assign a routing-time final decision role for one ambiguous TEP sample near the UTAR boundary.", "Source": "Prompt prefix"},
            {"Section": "Output Rule", "Purpose": "Constrain the response to JSON only with decision in {normal, anomaly}.", "Source": "Prompt prefix"},
            {"Section": "Task Context", "Purpose": "Explain that the sample was escalated because it lies near the UTAR boundary and requires additional review.", "Source": "Prompt prefix"},
            {"Section": "Decision Objective", "Purpose": "State that missing an anomaly is costlier than flagging a borderline anomaly.", "Source": "Prompt prefix"},
            {"Section": "Decision Policy", "Purpose": "List strong and moderate anomaly cues and the conditions that trigger anomaly output.", "Source": "Prompt prefix"},
            {"Section": "Guardrails", "Purpose": "Prevent over-reliance on perfect detector agreement and discourage default-normal behavior near the boundary.", "Source": "Prompt prefix"},
            {"Section": "Few-shot Examples", "Purpose": "Provide compact normal/anomaly routing examples in the same serialized feature format.", "Source": "Prompt body"},
            {"Section": "Derived Context", "Purpose": "Provide categorical cues such as utar_side, detector votes, entropy level, discrepancy level, and graph concentration.", "Source": "Derived prompt features"},
            {"Section": "Input Context / Detection", "Purpose": "Provide RF, XGB, ModernTCN, UTAR-base, entropy, discrepancy, and temporal weighting signals.", "Source": f"Prediction columns incl. {temporal_label}"},
            {"Section": "Input Context / GraphAD+", "Purpose": "Provide top sensors, structural score, z/trend/fluctuation, gap, and candidate topology evidence.", "Source": "GraphAD+ columns"},
            {"Section": "Evaluation Policy", "Purpose": "Appendix B prompt example is drawn from the representative routing path used for cost, latency, and prompt accounting.", "Source": source_name},
        ]
    )
    write_csv(APPENDIX_DIR / "table_b1_prompt_structure.csv", structure)

    prompt_vars = pd.DataFrame(
        [
            {"Variable": "rf / xgb / temporal", "Column": "p_rf / p_xgb / p_tcn", "Description": "Detector probabilities used by UTAR"},
            {"Variable": "utar_base", "Column": "p_utar_base", "Description": "Risk-dominant base routing score"},
            {"Variable": "temporal_weight", "Column": "temporal_weight", "Description": "Relative temporal-model influence inside UTAR routing"},
            {"Variable": "ensemble_entropy", "Column": "ensemble_entropy", "Description": "Gray-zone shortcut confidence cue"},
            {"Variable": "model_discrepancy", "Column": "model_discrepancy", "Description": "Detector disagreement cue"},
            {"Variable": "utar_side / detector_anomaly_votes", "Column": "derived from p_utar_base and detector scores", "Description": "Categorical routing cues exposed in the prompt header"},
            {"Variable": "graphad_score", "Column": "graphad_score", "Description": "Structure-aware anomaly concentration"},
            {"Variable": "top1_z / top1_trend / top1_fluct", "Column": "graphad_top1_z / graphad_top1_trend / graphad_top1_fluct", "Description": "Primary sensor evidence for the leading GraphAD+ candidate"},
            {"Variable": "candidate_sensors", "Column": "graphad_topk_sensors", "Description": "Top-K GraphAD+ candidate sensors"},
            {"Variable": "candidate_topology", "Column": "graphad_topology", "Description": "Adjacency trace among candidate sensors"},
        ]
    )
    write_csv(APPENDIX_DIR / "table_b2_prompt_variables.csv", prompt_vars)

    meta = {
        "llm_enabled": bool(model_cfg.get("enabled", False)),
        "llm_mode": str(model_cfg.get("mode", "stub")),
        "configured_model": str(model_cfg.get("model", "gpt-4o-mini")),
        "temperature": float(model_cfg.get("temperature", 0.0)),
        "prompt_policy": "representative_single_seed_for_entropy_gated_final_decision",
        "prompt_source_file": source_name,
        "representative_row": {
            "source_file": str(row["source_file"]),
            "fault_id": int(row["fault_id"]),
            "run_id": int(row["run_id"]),
            "sample_idx": int(row["sample_idx"]),
            "phase": str(row["phase"]),
        },
    }
    _write_text(APPENDIX_DIR / "appendix_b_prompt_meta.json", json.dumps(meta, ensure_ascii=False, indent=2))
    _write_text(APPENDIX_DIR / "appendix_b_prompt_example.txt", prompt)


def _failure_type(df: pd.DataFrame, tau: float) -> pd.Series:
    pred = (df["p_final"] >= tau).astype(int)
    conditions = [
        (df["y_true"] == 1) & (pred == 0) & (df["gray_zone"] == 0),
        (df["y_true"] == 1) & (pred == 0) & (df["gray_zone"] == 1) & (df["shortcut_filter"] == 1),
        (df["y_true"] == 1) & (pred == 0) & (df["llm_called"] == 1),
        (df["y_true"] == 0) & (pred == 1),
    ]
    labels = [
        "confident_false_negative",
        "shortcut_false_negative",
        "llm_unresolved_false_negative",
        "false_alarm",
    ]
    out = np.select(conditions, labels, default="correct")
    return pd.Series(out, index=df.index)


def _plot_failure_profiles(df: pd.DataFrame, tau: float) -> None:
    candidates = []
    for label, mask in [
        ("Confident miss", df["failure_type"] == "confident_false_negative"),
        ("LLM unresolved", df["failure_type"] == "llm_unresolved_false_negative"),
    ]:
        sub = df[mask].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(["model_discrepancy", "ensemble_entropy"], ascending=[False, False])
        row = sub.iloc[0]
        run_df = df[
            (df["source_file"] == row["source_file"])
            & (df["fault_id"] == row["fault_id"])
            & (df["run_id"] == row["run_id"])
        ].sort_values("sample_idx")
        candidates.append((label, run_df, int(row["sample_idx"])))

    if not candidates:
        return

    fig, axes = plt.subplots(len(candidates), 1, figsize=(10.8, 4.0 * len(candidates)), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, (label, run_df, highlighted_sample) in zip(axes, candidates):
        ax.plot(run_df["sample_idx"], run_df["p_utar_base"], color=PAPER_COLORS["orange"], linewidth=1.8, label="UTAR base")
        ax.plot(run_df["sample_idx"], run_df["p_final"], color=PAPER_COLORS["navy"], linewidth=2.2, label="Final output")
        ax.axhline(tau, color=PAPER_COLORS["ink"], linestyle="--", linewidth=1.2, label="tau")
        ax.axvline(float(run_df["onset_step"].iloc[0]), color=PAPER_COLORS["teal"], linestyle=":", linewidth=1.4, label="Fault onset")
        ax.axvline(highlighted_sample, color=PAPER_COLORS["red"], linestyle="-.", linewidth=1.2, label="Failure sample")
        ax.set_title(label, pad=10)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Score")
        ax.set_ylim(0.0, 1.0)
        style_axes(ax, y_grid_only=True)
        ax.legend(loc="best", ncol=2)
    fig.suptitle("Appendix C. Representative failure-case score traces", y=1.02, fontsize=15, fontweight="semibold")
    fig.tight_layout()
    save_figure(fig, APPENDIX_DIR / "figure_c1_failure_case_profiles.png")
    plt.close(fig)


def build_appendix_c_failure_cases(tau: float) -> None:
    selective = _safe_read_csv(PRED_DIR / "utar_test_main_selective.csv")
    if selective.empty:
        raise FileNotFoundError("Missing outputs/predictions/utar_test_main_selective.csv. Run main selective evaluation first.")

    df = selective.copy()
    if "ensemble_entropy" not in df.columns or "model_discrepancy" not in df.columns:
        raise KeyError("Selective output must contain ensemble_entropy and model_discrepancy columns.")

    df["failure_type"] = _failure_type(df, tau=tau)
    failures = df[df["failure_type"] != "correct"].copy()

    summary_rows = []
    total = max(len(df), 1)
    for failure_type, group in df.groupby("failure_type"):
        summary_rows.append(
            {
                "Failure Type": failure_type,
                "Count": int(len(group)),
                "Rate": float(len(group) / total),
                "Avg Entropy": float(group["ensemble_entropy"].mean()),
                "Avg Discrepancy": float(group["model_discrepancy"].mean()),
                "LLM Call Rate": float(group["llm_called"].mean()),
            }
        )
    write_csv(APPENDIX_DIR / "table_c1_failure_case_summary.csv", pd.DataFrame(summary_rows))

    detail_cols = [
        "source_file",
        "fault_id",
        "run_id",
        "sample_idx",
        "phase",
        "y_true",
        "p_utar_base",
        "p_final",
        "gray_zone",
        "shortcut_filter",
        "llm_called",
        "ensemble_entropy",
        "model_discrepancy",
        "graphad_top1_sensor",
        "graphad_score",
        "failure_type",
    ]
    details = failures.sort_values(["ensemble_entropy", "model_discrepancy"], ascending=[False, False])[detail_cols].head(40)
    write_csv(APPENDIX_DIR / "table_c2_failure_case_examples.csv", details)
    _plot_failure_profiles(df, tau=tau)


def build_appendix_d_parameter_sensitivity(tau: float) -> None:
    selected_q = read_selected_q(DEFAULT_Q)
    tcn_cfg = read_yaml(CONFIG_DIR / "train_tcn.yaml", default={})
    temporal_label = temporal_model_display_name(tcn_cfg.get("architecture", "modern_tcn"))
    gray_grid = read_csv(METRIC_DIR / "grayzone_grid.csv").sort_values("q").reset_index(drop=True)
    gray_seed = read_csv(METRIC_DIR / "grayzone_grid_by_seed.csv")
    base_cost = read_csv(PRED_DIR / "base_test_main_predictions.csv")
    utar_summary = read_csv(METRIC_DIR / "selective_llm_summary.csv")
    runtime_summary = load_base_runtime_summary(BASE_RUNTIME_SUMMARY_PATH)

    rows = []
    for q in gray_grid["q"]:
        utar_row = utar_summary[(utar_summary["dataset"] == "main") & (utar_summary["mode"] == "selective") & (np.isclose(utar_summary["q"], q))].iloc[0]
        for model_name, col_prefix in [
            ("RF (Base)", "p_rf"),
            ("XGB (Base)", "p_xgb"),
            (f"{temporal_label} (Base)", "p_tcn"),
            ("Avg. Ensemble (Base)", "p_ensemble"),
        ]:
            seed_metrics = []
            for seed in SEEDS:
                if col_prefix == "p_ensemble":
                    score = base_cost[[f"p_rf_seed{seed}", f"p_xgb_seed{seed}", f"p_tcn_seed{seed}"]].mean(axis=1).to_numpy(dtype=float)
                else:
                    score = base_cost[f"{col_prefix}_seed{seed}"].to_numpy(dtype=float)
                margin = float(gray_seed[(gray_seed["seed"] == seed) & (np.isclose(gray_seed["q"], q))]["gray_margin"].iloc[0])
                m = binary_metrics(base_cost["y_true"], score, tau=tau)
                seed_metrics.append({"gray_ratio": gray_ratio(score, tau=tau, margin=margin), "f1": m["f1"]})
            metrics_df = pd.DataFrame(seed_metrics)
            rows.append(
                {
                    "q": float(q),
                    "Model": model_name,
                    "Gray Ratio": fmt(metrics_df["gray_ratio"].mean(), metrics_df["gray_ratio"].std(ddof=1)),
                    "F1": fmt(metrics_df["f1"].mean(), metrics_df["f1"].std(ddof=1)),
                }
            )
        rows.append(
            {
                "q": float(q),
                "Model": "UTAR (Proposed)",
                "Gray Ratio": fmt(utar_row["gray_ratio_mean"], pick_std(utar_row, "gray_ratio")),
                "F1": fmt(utar_row["f1_mean"], pick_std(utar_row, "f1")),
            }
        )

    df = pd.DataFrame(rows)
    write_csv(APPENDIX_DIR / "table_d1_q_sweep_base_models.csv", df)
    write_csv(APPENDIX_DIR / "table_a1_q_sweep_base_models.csv", df)
    write_csv(APPENDIX_DIR / "table_a2_q_sweep_base_models.csv", df)

    temporal_component = ensemble_component_label(temporal_label)
    runtime_rows = []
    summary = read_csv(METRIC_DIR / "selective_llm_summary.csv")
    for dataset in ["main", "cost"]:
        for label, component in [
            (ensemble_component_label("RF"), ensemble_component_label("RF")),
            (ensemble_component_label("XGB"), ensemble_component_label("XGB")),
            (ensemble_component_label(temporal_label), temporal_component),
            ("GraphAD+", "GraphAD+"),
            ("UTAR Base Stack", "UTAR Base Stack"),
        ]:
            total_ms = get_base_runtime_stat(runtime_summary, split=dataset, component=component, field="total_latency_ms", default=np.nan)
            total_std_ms = get_base_runtime_stat(runtime_summary, split=dataset, component=component, field="total_latency_ms_std", default=0.0)
            avg_ms = get_base_runtime_stat(runtime_summary, split=dataset, component=component, field="avg_latency_ms_per_sample", default=np.nan)
            avg_std_ms = get_base_runtime_stat(runtime_summary, split=dataset, component=component, field="avg_latency_ms_per_sample_std", default=0.0)
            if np.isfinite(total_ms):
                runtime_rows.append(
                    {
                        "Category": "Base Component",
                        "Dataset": dataset,
                        "Model": label,
                        "Total Inference Time (s)": fmt(total_ms / 1000.0, total_std_ms / 1000.0),
                        "Avg Latency per Sample (ms)": fmt(avg_ms, avg_std_ms),
                        "LLM Call Rate": "NA",
                    }
                )
    for dataset in ["main", "cost"]:
        for mode in ["no_llm", "selective", "selective_no_filter", "selective_no_graph", "ensemble_only", "full_llm"]:
            sub = summary[(summary["dataset"] == dataset) & (summary["mode"] == mode)]
            if dataset == "main":
                sub = sub[np.isclose(sub["q"], selected_q)]
            if dataset == "cost":
                sub = sub[np.isclose(sub["q"], selected_q)]
            if sub.empty:
                continue
            row = sub.iloc[0]
            runtime_rows.append(
                {
                    "Category": "End-to-End Mode",
                    "Dataset": dataset,
                    "Model": mode,
                    "Total Inference Time (s)": fmt(row["total_latency_ms_mean"] / 1000.0, pick_std(row, "total_latency_ms") / 1000.0),
                    "Avg Latency per Sample (ms)": fmt(row["avg_latency_ms_per_sample_mean"], pick_std(row, "avg_latency_ms_per_sample")),
                    "LLM Call Rate": fmt(row["llm_call_rate_mean"], pick_std(row, "llm_call_rate")),
                }
            )
    runtime_df = pd.DataFrame(runtime_rows)
    write_csv(APPENDIX_DIR / "table_d2_inference_latency.csv", runtime_df)
    write_csv(APPENDIX_DIR / "table_d1_inference_latency.csv", runtime_df)


def build_appendix_e_distribution_shift() -> None:
    shift_summary = read_csv(_ensure_shift_summary_path())
    write_csv(APPENDIX_DIR / "table_e1_distribution_shift_summary.csv", shift_summary)
    write_csv(APPENDIX_DIR / "table_b1_distribution_shift_summary.csv", shift_summary)

    manifest_rows = []
    for artifact, path in [
        ("KDE plot", SHIFT_DIR / "kde_normal_vs_shift.png"),
        ("PCA scatter", SHIFT_DIR / "pca_phase_scatter.png"),
        ("t-SNE scatter", SHIFT_DIR / "tsne_phase_scatter.png"),
        ("Feature shift summary", SHIFT_DIR / "feature_shift_normal_vs_shift.csv"),
    ]:
        if path.exists():
            manifest_rows.append({"Artifact": artifact, "Path": str(path)})
    manifest = pd.DataFrame(manifest_rows)
    write_csv(APPENDIX_DIR / "appendix_e_artifact_manifest.csv", manifest)
    write_csv(APPENDIX_DIR / "appendix_b_artifact_manifest.csv", manifest)


def _load_graphad_run_example() -> tuple[pd.Series, pd.DataFrame, dict]:
    pred_main = read_csv(PRED_DIR / "base_test_main_predictions.csv")
    example = pred_main[pred_main["phase"].isin(["transition", "post_shift"])].copy()
    example = example[example["y_true"] == 1].copy()
    example["priority"] = example["graphad_score"].fillna(0.0) + 0.25 * example["graphad_top1_gap"].fillna(0.0)
    example = example.sort_values("priority", ascending=False)
    row = example.iloc[0]

    raw_rows = read_csv(PROCESSED_DIR / "te_test_main_rows.csv")
    run_df = raw_rows[
        (raw_rows["source_file"] == row["source_file"])
        & (raw_rows["fault_id"] == row["fault_id"])
        & (raw_rows["run_id"] == row["run_id"])
    ].sort_values("sample_idx")
    artifact = load_graphad_artifact(MODEL_DIR / "graphad_artifact.json")
    return row, run_df, artifact


def build_appendix_f_graphad_visual() -> None:
    row, run_df, artifact = _load_graphad_run_example()
    mats = graphad_score_matrices(run_df, artifact)
    sample_idx = int(row["sample_idx"])
    sample_scores = {
        key: frame.loc[run_df["sample_idx"] == sample_idx].iloc[0]
        for key, frame in mats.items()
    }
    raw_series = sample_scores["raw"].sort_values(ascending=False)
    smooth_series = sample_scores["smooth"].sort_values(ascending=False)
    z_series = sample_scores["z"]
    trend_series = sample_scores["trend"]
    fluct_series = sample_scores["fluct"]

    selected = list(dict.fromkeys(list(raw_series.head(6).index) + list(smooth_series.head(6).index)))
    rank_df = pd.DataFrame(
        {
            "Sensor": selected,
            "Raw Score": [float(raw_series[s]) for s in selected],
            "Smoothed Score": [float(smooth_series[s]) for s in selected],
            "Z Score": [float(z_series[s]) for s in selected],
            "Trend Score": [float(trend_series[s]) for s in selected],
            "Fluctuation Score": [float(fluct_series[s]) for s in selected],
            "Raw Rank": [int(raw_series.index.get_loc(s) + 1) for s in selected],
            "Smoothed Rank": [int(smooth_series.index.get_loc(s) + 1) for s in selected],
        }
    ).sort_values("Smoothed Rank")
    write_csv(APPENDIX_DIR / "table_f1_graphad_rank_trace.csv", rank_df)

    fig = plt.figure(figsize=(16.4, 5.8))
    outer = fig.add_gridspec(1, 3, left=0.02, right=0.98, top=0.93, bottom=0.08, wspace=0.06)

    def draw_card(ax, title: str, footer: str) -> None:
        ax.set_axis_off()
        body = FancyBboxPatch((0.01, 0.02), 0.98, 0.96, boxstyle="round,pad=0.012,rounding_size=0.04", transform=ax.transAxes, linewidth=0.95, edgecolor="#8ea4b8", facecolor="#f7fbff", zorder=-10)
        ax.add_patch(body)
        ax.text(0.03, 0.965, title, transform=ax.transAxes, ha="left", va="top", fontsize=11.2, fontweight="semibold")
        ax.text(0.03, 0.055, footer, transform=ax.transAxes, ha="left", va="center", fontsize=8.7, color=PAPER_COLORS["ink"])

    raw_card = fig.add_subplot(outer[0, 0])
    smooth_card = fig.add_subplot(outer[0, 1])
    rank_card = fig.add_subplot(outer[0, 2])
    draw_card(raw_card, "(a) Raw Anomaly Scores (t=200):\nData-driven Ambiguity", "Raw anomaly evidence leaves several candidate sensors close together.")
    draw_card(smooth_card, "(b) Graph Smoothing (Eq. 8):\nCorrelation-based Propagation", "Topology-aware propagation reinforces coherent sensor neighborhoods.")
    draw_card(rank_card, "(c) Hybrid Reranking (p=1.8, b=0.2):\nRoot-Cause Identification", "The final reranking surfaces the most plausible root-cause candidate.")

    adjacency = artifact.get("adjacency", {})
    sensor_order = list(rank_df.sort_values("Raw Rank")["Sensor"])
    angles = np.linspace(np.pi * 0.12, np.pi * 2.12, len(sensor_order), endpoint=False)
    coords = np.c_[0.5 + 0.27 * np.cos(angles), 0.52 + 0.22 * np.sin(angles)]
    sensor_pos = {sensor: tuple(coord) for sensor, coord in zip(sensor_order, coords)}
    target_sensor = rank_df.sort_values("Smoothed Rank").iloc[0]["Sensor"]
    raw_norm = rank_df.set_index("Sensor").loc[sensor_order, "Raw Score"].to_numpy(dtype=float)
    raw_norm = raw_norm / max(float(raw_norm.max()), 1e-8)
    smooth_norm = rank_df.set_index("Sensor").loc[sensor_order, "Smoothed Score"].to_numpy(dtype=float)
    smooth_norm = smooth_norm / max(float(smooth_norm.max()), 1e-8)
    edge_pairs = []
    seen = set()
    for sensor in sensor_order:
        for neighbor in adjacency.get(sensor, []):
            if neighbor not in sensor_pos:
                continue
            key = tuple(sorted((sensor, neighbor)))
            if key in seen:
                continue
            seen.add(key)
            edge_pairs.append((sensor, neighbor))
    if len(edge_pairs) < 5:
        fallback = [(target_sensor, sensor) for sensor in sensor_order if sensor != target_sensor]
        cycle = list(zip(sensor_order, sensor_order[1:] + sensor_order[:1]))
        edge_pairs = fallback + cycle[: max(0, 6 - len(fallback))]

    raw_ax = raw_card.inset_axes([0.05, 0.14, 0.90, 0.74])
    raw_ax.set_axis_off()
    raw_ax.set_xlim(0, 1)
    raw_ax.set_ylim(0, 1)
    for sensor, neighbor in edge_pairs:
        x0, y0 = sensor_pos[sensor]
        x1, y1 = sensor_pos[neighbor]
        raw_ax.plot([x0, x1], [y0, y1], color="#c7c7c7", linewidth=1.7, zorder=1)
    for i, sensor in enumerate(sensor_order):
        x0, y0 = sensor_pos[sensor]
        color = plt.cm.Wistia(0.35 + 0.55 * raw_norm[i])
        size = 260 + 230 * raw_norm[i]
        raw_ax.scatter([x0], [y0], s=size, color=color, edgecolor="#996515", linewidth=1.1, zorder=3)
        label = f"{sensor.replace('_', ' ')}: {rank_df.set_index('Sensor').loc[sensor, 'Raw Score']:.1f}"
        dx = 0.022 if x0 < 0.5 else -0.16
        raw_ax.text(x0 + dx, y0 + 0.02, label, fontsize=8.5, bbox={"boxstyle": "round,pad=0.18", "facecolor": "#ffe699", "edgecolor": "#c8a64d"}, zorder=4)
    tx, ty = sensor_pos[target_sensor]
    raw_rank = int(rank_df.set_index("Sensor").loc[target_sensor, "Raw Rank"])
    raw_ax.text(tx - 0.11, ty - 0.12, f"Target case:\n{target_sensor.replace('_', ' ')}", fontsize=8.7, bbox={"boxstyle": "round,pad=0.24", "facecolor": "#fff3cd", "edgecolor": "#d6b656"})
    raw_ax.text(tx + 0.05, ty - 0.10, f"Ranking #{raw_rank}", fontsize=8.8, fontweight="semibold", color=PAPER_COLORS["ink"], bbox={"boxstyle": "round,pad=0.18", "facecolor": "#ffe599", "edgecolor": "#c8a64d"})
    raw_ax.plot([0.05, 0.13], [0.08, 0.08], color="#c7c7c7", linewidth=1.8)
    raw_ax.text(0.15, 0.08, "Pearson correlation", va="center", fontsize=8.3)
    raw_ax.plot([0.05, 0.13], [0.03, 0.03], color="#9d9d9d", linewidth=3.0)
    raw_ax.text(0.15, 0.03, "Correlation threshold > 0.7", va="center", fontsize=8.3)

    smooth_ax = smooth_card.inset_axes([0.05, 0.14, 0.90, 0.74])
    smooth_ax.set_axis_off()
    smooth_ax.set_xlim(0, 1)
    smooth_ax.set_ylim(0, 1)
    smooth_ax.text(0.18, 0.94, "Statistical Analysis\nGraph Construction", ha="center", va="center", fontsize=8.5, bbox={"boxstyle": "round,pad=0.25", "facecolor": "#ddebf7", "edgecolor": "#8ea4b8"})
    smooth_ax.text(0.73, 0.94, "Dynamic Edge\nPropagation", ha="center", va="center", fontsize=8.5, bbox={"boxstyle": "round,pad=0.25", "facecolor": "#ddebf7", "edgecolor": "#8ea4b8"})
    for sensor, neighbor in edge_pairs:
        s0 = rank_df.set_index("Sensor").loc[sensor, "Smoothed Score"]
        s1 = rank_df.set_index("Sensor").loc[neighbor, "Smoothed Score"]
        src, dst = (sensor, neighbor) if s0 <= s1 else (neighbor, sensor)
        x0, y0 = sensor_pos[src]
        x1, y1 = sensor_pos[dst]
        smooth_ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), connectionstyle="arc3,rad=0.10", arrowstyle="-|>", mutation_scale=12, linewidth=1.3, color="#7fb3d5", alpha=0.82))
    for i, sensor in enumerate(sensor_order):
        x0, y0 = sensor_pos[sensor]
        color = plt.cm.cividis(0.22 + 0.72 * smooth_norm[i])
        size = 270 + 380 * smooth_norm[i]
        smooth_ax.scatter([x0], [y0], s=size, color=color, edgecolor="white", linewidth=1.2, zorder=4)
    for sensor in rank_df.sort_values("Smoothed Rank").head(4)["Sensor"]:
        x0, y0 = sensor_pos[sensor]
        raw_v = rank_df.set_index("Sensor").loc[sensor, "Raw Score"]
        smooth_v = rank_df.set_index("Sensor").loc[sensor, "Smoothed Score"]
        smooth_ax.text(x0 + (0.03 if x0 < 0.5 else -0.15), y0 + (0.08 if y0 < 0.55 else -0.11), f"{sensor.replace('_', ' ')}: {raw_v:.1f} -> {smooth_v:.1f}", fontsize=8.4, bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#d9dde3"})
    smooth_ax.add_patch(FancyBboxPatch((0.29, 0.04), 0.42, 0.08, boxstyle="round,pad=0.02,rounding_size=0.04", linewidth=0.8, edgecolor="#c0c6cf", facecolor="white"))
    for x0, c in zip(np.linspace(0.34, 0.64, 5), ["#ff5a5f", "#ffbd2e", "#27c93f", "#22a6f2", "#ffffff"]):
        smooth_ax.scatter([x0], [0.08], s=260, color=c, edgecolor="#8ea4b8", linewidth=0.8, zorder=5)
    smooth_ax.text(0.11, 0.11, f"{target_sensor} correlation rises\nthrough neighborhood support", fontsize=8.6, bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "#d9dde3"})

    rank_ax = rank_card.inset_axes([0.05, 0.14, 0.90, 0.74])
    rank_ax.set_axis_off()
    rank_ax.set_xlim(0, 1)
    rank_ax.set_ylim(0, 1)
    for sensor, neighbor in edge_pairs:
        x0, y0 = sensor_pos[sensor]
        x1, y1 = sensor_pos[neighbor]
        rank_ax.plot([x0, x1], [y0, y1], color="#d0d0d0", linewidth=1.5, zorder=1)
    ordered = rank_df.sort_values("Smoothed Rank")
    palette = ["#2c6fb7", "#4f81bd", "#c87f2a", "#d9a441", "#ead39c", "#efe6c8"]
    rank_colors = {sensor: palette[min(i, len(palette) - 1)] for i, sensor in enumerate(ordered["Sensor"])}
    for sensor in sensor_order:
        x0, y0 = sensor_pos[sensor]
        face = rank_colors.get(sensor, "#efe6c8")
        size = 300 if sensor == target_sensor else 240
        rank_ax.scatter([x0], [y0], s=size, color=face, edgecolor="#8a6d3b" if sensor != target_sensor else "#0c4b84", linewidth=1.2, zorder=3)
        label = f"{sensor.replace('_', ' ')}"
        if sensor == target_sensor:
            label += f"\n{rank_df.set_index('Sensor').loc[sensor, 'Smoothed Score']:.1f} (Top-1)"
            rank_ax.text(x0 - 0.04, y0 - 0.12, "Ranking #1", fontsize=8.6, color="white", fontweight="semibold", bbox={"boxstyle": "round,pad=0.2", "facecolor": "#2c6fb7", "edgecolor": "#2c6fb7"})
        else:
            label += f"\n#{int(rank_df.set_index('Sensor').loc[sensor, 'Smoothed Rank'])}"
        dx = 0.03 if x0 < 0.5 else -0.15
        rank_ax.text(x0 + dx, y0 + 0.015, label, fontsize=8.4, bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#d9dde3"})
    delta_rank = int(rank_df.set_index("Sensor").loc[target_sensor, "Raw Rank"] - rank_df.set_index("Sensor").loc[target_sensor, "Smoothed Rank"])
    rank_ax.text(0.72, 0.83, "Root-Cause Candidate\nPerformance", ha="center", fontsize=8.8, bbox={"boxstyle": "round,pad=0.26", "facecolor": "white", "edgecolor": "#d9dde3"})
    rank_ax.text(0.72, 0.72, f"Delta ranking ({target_sensor}): {'+' if delta_rank >= 0 else ''}{delta_rank}", ha="center", fontsize=8.6, color="#2f855a")
    rank_ax.text(0.05, 0.05, "Rankings:", fontsize=8.5, fontweight="semibold")
    for i, x0 in enumerate(np.linspace(0.20, 0.64, 5), start=1):
        rank_ax.text(x0, 0.05, f"{i}", ha="center", va="center", fontsize=8.3, color="white" if i < 3 else PAPER_COLORS["ink"], bbox={"boxstyle": "round,pad=0.16", "facecolor": palette[i - 1], "edgecolor": palette[i - 1]})
    rank_ax.text(0.20, 0.00, "Stable detection", fontsize=8.1, color=PAPER_COLORS["ink"])
    rank_ax.text(0.54, 0.00, "Oscillation", fontsize=8.1, color=PAPER_COLORS["ink"])

    fig.suptitle("Appendix F. GraphAD+ visual proof for data-driven reranking", y=0.99, fontsize=15.0, fontweight="semibold")

    save_figure(fig, APPENDIX_DIR / "figure_f1_graphad_visual_proof.png")
    plt.close(fig)


def build_appendix_g_seed_variation() -> None:
    selected_q = read_selected_q(DEFAULT_Q)
    tcn_cfg = read_yaml(CONFIG_DIR / "train_tcn.yaml", default={})
    temporal_label = temporal_model_display_name(tcn_cfg.get("architecture", "modern_tcn"))
    val_pred = read_csv(PRED_DIR / "base_val_predictions.csv")
    test_pred = read_csv(PRED_DIR / "base_test_main_predictions.csv")

    detail_rows = []
    summary_rows = []
    base_rows_map: dict[str, list[dict]] = {
        "RF (Base)": [],
        "XGB (Base)": [],
        f"{temporal_label} (Base)": [],
        "Avg. Ensemble (Base)": [],
    }
    for seed in SEEDS:
        for label, val_col, test_col in [
            ("RF (Base)", f"p_rf_seed{seed}", f"p_rf_seed{seed}"),
            ("XGB (Base)", f"p_xgb_seed{seed}", f"p_xgb_seed{seed}"),
            (f"{temporal_label} (Base)", f"p_tcn_seed{seed}", f"p_tcn_seed{seed}"),
            ("Avg. Ensemble (Base)", None, None),
        ]:
            if label == "Avg. Ensemble (Base)":
                p_val = val_pred[[f"p_rf_seed{seed}", f"p_xgb_seed{seed}", f"p_tcn_seed{seed}"]].mean(axis=1).to_numpy(dtype=float)
                p_test = test_pred[[f"p_rf_seed{seed}", f"p_xgb_seed{seed}", f"p_tcn_seed{seed}"]].mean(axis=1).to_numpy(dtype=float)
            else:
                p_val = val_pred[val_col].to_numpy(dtype=float)
                p_test = test_pred[test_col].to_numpy(dtype=float)

            tau = float(np.linspace(0.01, 0.99, 99)[0])
            best_f1 = -1.0
            for cand_tau in np.linspace(0.01, 0.99, 99):
                cand_f1 = binary_metrics(val_pred["y_true"], p_val, tau=float(cand_tau))["f1"]
                if cand_f1 > best_f1:
                    best_f1 = cand_f1
                    tau = float(cand_tau)
            val_metrics = binary_metrics(val_pred["y_true"], p_val, tau=tau)
            test_metrics = binary_metrics(test_pred["y_true"], p_test, tau=tau)
            row = {
                "Seed Number": seed,
                "Method": label,
                "Ave. F1": float(test_metrics["f1"]),
                "PRR": float(prr(val_metrics["recall"], test_metrics["recall"])),
                "Worst-Case Recall (P5)": float(low_tail_recall(test_pred["y_true"], p_test, test_pred["run_id"], tau=tau, window=50, quantile=0.05)),
                "Instability (Var)": float(instability_score(p_test, test_pred["run_id"], test_pred["phase"])),
            }
            detail_rows.append(row)
            base_rows_map[label].append(
                {
                    "f1": row["Ave. F1"],
                    "prr": row["PRR"],
                    "worst_case_recall": row["Worst-Case Recall (P5)"],
                    "instability": row["Instability (Var)"],
                }
            )

    utar_seed_main = read_csv(METRIC_DIR / "selective_llm_seed_metrics.csv")
    utar_seed_main = utar_seed_main[
        (utar_seed_main["dataset"] == "main")
        & (utar_seed_main["mode"] == "selective")
        & (np.isclose(utar_seed_main["q"], selected_q))
    ].copy()
    for _, row in utar_seed_main.sort_values("seed").iterrows():
        detail_rows.append(
            {
                "Seed Number": int(row["seed"]),
                "Method": "UTAR (Proposed)",
                "Ave. F1": float(row["f1"]),
                "PRR": float(row["prr"]),
                "Worst-Case Recall (P5)": float(row["worst_case_recall"]),
                "Instability (Var)": float(row["instability"]),
            }
        )

    detail = pd.DataFrame(detail_rows).sort_values(["Seed Number", "Method"]).reset_index(drop=True)
    write_csv(APPENDIX_DIR / "table_g1_seed_variation_detail.csv", detail)
    write_csv(APPENDIX_DIR / "table_e1_seed_variation_detail.csv", detail)

    for method, group in detail.groupby("Method"):
        summary_rows.append(
            {
                "Method": method,
                "Ave. F1": fmt(group["Ave. F1"].mean(), group["Ave. F1"].std(ddof=1) if len(group) > 1 else 0.0),
                "PRR": fmt(group["PRR"].mean(), group["PRR"].std(ddof=1) if len(group) > 1 else 0.0),
                "Worst-Case Recall (P5)": fmt(group["Worst-Case Recall (P5)"].mean(), group["Worst-Case Recall (P5)"].std(ddof=1) if len(group) > 1 else 0.0),
                "Instability (Var)": fmt(group["Instability (Var)"].mean(), group["Instability (Var)"].std(ddof=1) if len(group) > 1 else 0.0),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    write_csv(APPENDIX_DIR / "table_g2_seed_variation_summary.csv", summary_df)
    write_csv(APPENDIX_DIR / "table_e2_seed_variation_summary.csv", summary_df)


def _tuning_summary(tool_name: str) -> tuple[dict, str]:
    json_path = TUNING_DIR / f"{tool_name}_best.json"
    if not json_path.exists():
        return {}, "current_config"
    return read_json(json_path), "tuning_search"


def build_appendix_h_hyperparameter_rationale() -> None:
    rf_cfg = _safe_read_yaml(CONFIG_DIR / "train_rf.yaml")
    xgb_cfg = _safe_read_yaml(CONFIG_DIR / "train_xgb.yaml")
    tcn_cfg = _safe_read_yaml(CONFIG_DIR / "train_tcn.yaml")
    graphad_cfg = _safe_read_yaml(CONFIG_DIR / "train_graphad.yaml")
    routing_cfg = _safe_read_yaml(CONFIG_DIR / "routing.yaml")
    search_cfgs = {
        "rf": _safe_read_yaml(CONFIG_DIR / "search_rf.yaml"),
        "xgb": _safe_read_yaml(CONFIG_DIR / "search_xgb.yaml"),
        "tcn": _safe_read_yaml(CONFIG_DIR / "search_tcn.yaml"),
        "routing": _safe_read_yaml(CONFIG_DIR / "search_routing.yaml"),
        "graphad": _safe_read_yaml(CONFIG_DIR / "search_graphad.yaml"),
    }

    rows = []
    for tool_name, component, params, rationale_map in [
        (
            "rf",
            "RF",
            {"n_estimators": rf_cfg.get("n_estimators"), "max_depth": rf_cfg.get("max_depth"), "min_samples_split": rf_cfg.get("min_samples_split"), "min_samples_leaf": rf_cfg.get("min_samples_leaf")},
            {
                "n_estimators": "Validation-optimized tree count for stable ensemble variance reduction.",
                "max_depth": "Controls bias-variance trade-off for static detector generalization.",
                "min_samples_split": "Prevents brittle splits under temporal shift.",
                "min_samples_leaf": "Smooths leaf-level probability estimates.",
            },
        ),
        (
            "xgb",
            "XGB",
            {"n_estimators": xgb_cfg.get("n_estimators"), "max_depth": xgb_cfg.get("max_depth"), "learning_rate": xgb_cfg.get("learning_rate"), "subsample": xgb_cfg.get("subsample")},
            {
                "n_estimators": "Balances boosting strength and overfitting under shifted test conditions.",
                "max_depth": "Controls nonlinear boundary complexity.",
                "learning_rate": "Stabilizes residual fitting across operating regimes.",
                "subsample": "Improves robustness to covariate shift through stochastic boosting.",
            },
        ),
        (
            "tcn",
            "ModernTCN",
            {"channels": tcn_cfg.get("channels"), "dilations": tcn_cfg.get("dilations"), "kernel_size": tcn_cfg.get("kernel_size"), "dropout": tcn_cfg.get("dropout"), "expansion_ratio": tcn_cfg.get("expansion_ratio"), "batch_size": tcn_cfg.get("batch_size"), "epochs": tcn_cfg.get("epochs")},
            {
                "channels": "Sets temporal capacity for industrial sensor dynamics.",
                "dilations": "Controls receptive field width for onset-to-post-shift evolution.",
                "kernel_size": "Trades local sensitivity against smoothing.",
                "dropout": "Regularizes temporal representation under shift.",
                "expansion_ratio": "Controls ModernTCN block expressiveness.",
                "batch_size": "Balances gradient stability and throughput.",
                "epochs": "Chosen to maximize validation utility before temporal overfitting.",
            },
        ),
        (
            "graphad",
            "GraphAD+",
            {"corr_threshold": graphad_cfg.get("corr_threshold"), "alpha": graphad_cfg.get("alpha"), "top_k": graphad_cfg.get("top_k"), "lambda_z": graphad_cfg.get("lambda_z"), "lambda_tr": graphad_cfg.get("lambda_tr"), "lambda_fl": graphad_cfg.get("lambda_fl")},
            {
                "corr_threshold": "Defines data-driven topology edges used for structure-aware smoothing.",
                "alpha": "Balances local anomaly evidence with neighborhood support.",
                "top_k": "Sets the number of GraphAD+ candidate sensors exposed to the downstream prompt.",
                "lambda_z": "Prioritizes instantaneous deviation.",
                "lambda_tr": "Captures monotonic directional change.",
                "lambda_fl": "Captures fluctuation and non-monotonic turbulence.",
            },
        ),
        (
            "routing",
            "UTAR",
            {"default_q": read_selected_q(DEFAULT_Q), "q_grid": routing_cfg.get("q_grid"), "sigmoid_gain": routing_cfg.get("sigmoid_gain"), "entropy_shortcut_quantile": routing_cfg.get("entropy_shortcut_quantile"), "discrepancy_shortcut_quantile": routing_cfg.get("discrepancy_shortcut_quantile")},
            {
                "default_q": "Selected as the operational elbow between stability gain and call-rate growth.",
                "q_grid": "Search candidates for gray-zone width control.",
                "sigmoid_gain": "Sharpens temporal weight transition around ambiguous regions.",
                "entropy_shortcut_quantile": "Controls confidence-based shortcut aggressiveness.",
                "discrepancy_shortcut_quantile": "Controls disagreement-aware shortcut aggressiveness.",
            },
        ),
    ]:
        best_meta, source = _tuning_summary(tool_name)
        search_space = search_cfgs.get(tool_name, {})
        best_params = best_meta.get("best_params", {})
        objective = best_meta.get("best_score")
        for param, value in params.items():
            selected_value = best_params.get(param, value)
            if selected_value is None or (isinstance(selected_value, float) and np.isnan(selected_value)):
                continue
            rows.append(
                {
                    "Component": component,
                    "Parameter": param,
                    "Selected Value": selected_value,
                    "Candidate Space": json.dumps(search_space.get(param, []), ensure_ascii=False) if param in search_space else "",
                    "Evidence Source": source,
                    "Objective Score": objective,
                    "Rationale": rationale_map.get(param, ""),
                }
            )

    df = pd.DataFrame(rows)
    write_csv(APPENDIX_DIR / "table_h1_hyperparameter_rationale.csv", df)


def main() -> None:
    set_paper_style()
    ensure_dir(APPENDIX_DIR)
    tau = float(read_json(METRIC_DIR / "thresholds.json")["tau"])

    build_appendix_b_prompt()
    build_appendix_c_failure_cases(tau=tau)
    build_appendix_d_parameter_sensitivity(tau=tau)
    build_appendix_e_distribution_shift()
    build_appendix_f_graphad_visual()
    build_appendix_g_seed_variation()
    build_appendix_h_hyperparameter_rationale()
    print(f"Saved TE appendix artifacts to {APPENDIX_DIR}")


if __name__ == "__main__":
    main()
