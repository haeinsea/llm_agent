from __future__ import annotations

from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd

from .analysis import build_graph_contexts, build_llm_structured_inputs, build_routing_paths, graphad_topk_list
from .config import BASE_DIR
from .explainer import generate_explanation, qa_on_sample
from .models import align_feature_frame, score_uploaded_te
from .preprocessing import load_uploaded_te_csv, split_features_target


app = FastAPI(title="TE UTAR Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def root():
    return FileResponse(static_dir / "index.html")


def _display_index_series(df_raw: pd.DataFrame) -> pd.Series:
    for col in ["Unnamed: 0", "index", "sample_idx", "sample", "id"]:
        if col in df_raw.columns:
            values = pd.to_numeric(df_raw[col], errors="coerce")
            fallback = pd.Series(range(len(df_raw)), index=df_raw.index, dtype=float)
            return values.where(values.notna(), fallback).astype(int)
    return pd.Series(range(len(df_raw)), index=df_raw.index, dtype=int)


def _decision_source_display(score_row: pd.Series) -> str:
    source = str(score_row.get("decision_source", "utar_base")).strip().lower()
    if source == "llm":
        return "Selective LLM"
    if source == "entropy_shortcut":
        return "Gray-Zone Shortcut"
    if source == "ensemble":
        return "Avg. Ensemble"
    return "UTAR Base"


def _selected_model_display(score_row: pd.Series) -> str:
    if int(score_row.get("llm_called", 0)) == 1:
        return "Routing LLM"
    if str(score_row.get("decision_source", "")).lower() == "entropy_shortcut":
        return "Entropy Shortcut"
    return "UTAR Base"


def analyze_uploaded_te(df_raw: pd.DataFrame) -> dict[str, Any]:
    display_index = _display_index_series(df_raw)
    X_df, y = split_features_target(df_raw.copy())
    feature_df = align_feature_frame(X_df)
    feature_median = feature_df.median(numeric_only=True)

    routed_df, routing_ctx = score_uploaded_te(X_df, display_index=display_index, y_true=y)

    rows_out: list[dict[str, Any]] = []
    process_graph_context_cache: dict[str, Any] | None = None

    for row_pos in range(len(routed_df)):
        score_row = routed_df.iloc[row_pos]
        feature_row = feature_df.iloc[row_pos]
        topk_list = graphad_topk_list(feature_row, score_row, feature_median)
        process_graph_context, subgraph_context = build_graph_contexts(topk_list)
        if process_graph_context_cache is None:
            process_graph_context_cache = process_graph_context
        routing_paths = build_routing_paths(topk_list)
        llm_struct = build_llm_structured_inputs(
            feature_row=feature_row,
            score_row=score_row,
            topk_list=topk_list,
            process_graph_context=process_graph_context,
            subgraph_context=subgraph_context,
        )
        llm_struct["paths"] = routing_paths

        predicted_is_anomaly = str(score_row.get("final_decision", "normal")) == "anomaly"
        decision_source_display = _decision_source_display(score_row)
        selected_model_display = _selected_model_display(score_row)

        rows_out.append(
            {
                "index": int(score_row["display_index"]),
                "sequence_index": int(score_row["sample_idx"]),
                "features": {c: float(feature_row[c]) if pd.notna(feature_row[c]) else None for c in feature_df.columns},
                "scores": {
                    "utar_base_score": float(score_row.get("p_utar_base", 0.0)),
                    "gtar_score": float(score_row.get("p_utar_base", 0.0)),
                    "xgb_prob": float(score_row.get("p_xgb", 0.0)),
                    "rf_prob": float(score_row.get("p_rf", 0.0)),
                    "tcn_prob": float(score_row.get("p_tcn", 0.0)),
                    "lstm_prob": float(score_row.get("p_tcn", 0.0)),
                    "ensemble_mean": float(score_row.get("ensemble_mean", score_row.get("p_ensemble", 0.0))),
                    "ensemble_entropy": float(score_row.get("ensemble_entropy", 0.0)),
                    "model_discrepancy": float(score_row.get("model_discrepancy", 0.0)),
                    "graphad_score": float(score_row.get("graphad_score", 0.0)),
                    "final_prob": float(score_row.get("p_final", 0.0)),
                    "tau": float(score_row.get("tau", routing_ctx.tau)),
                    "selected_q": float(score_row.get("selected_q", routing_ctx.selected_q)),
                    "gray_margin": float(score_row.get("gray_margin", routing_ctx.margin)),
                    "gray_zone": int(score_row.get("gray_zone", 0)),
                    "llm_called": int(score_row.get("llm_called", 0)),
                    "llm_decision": None if pd.isna(score_row.get("llm_decision")) else str(score_row.get("llm_decision")),
                    "decision_source": decision_source_display,
                    "selected_model": selected_model_display,
                },
                "graphad_topk": topk_list,
                "process_graph_context": process_graph_context_cache,
                "subgraph_context": subgraph_context,
                "llm_struct": llm_struct,
                "llm_explanation": None,
                "llm_paths": [],
                "routing_paths": routing_paths,
                "label": None if pd.isna(score_row.get("y_true")) else int(score_row.get("y_true")),
                "predicted_is_anomaly": bool(predicted_is_anomaly),
            }
        )

    return {
        "n_samples": len(feature_df),
        "columns": list(feature_df.columns),
        "selected_q": routing_ctx.selected_q,
        "tau": routing_ctx.tau,
        "gray_margin": routing_ctx.margin,
        "rows": rows_out,
    }


@app.post("/api/upload_te")
async def upload_te(file: UploadFile = File(...)) -> dict[str, Any]:
    try:
        df_raw = load_uploaded_te_csv(file.file)
        if df_raw.empty:
            raise ValueError("The uploaded CSV is empty.")
        return analyze_uploaded_te(df_raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"CSV analysis error: {exc}") from exc


@app.post("/api/explain")
async def explain(struct: dict[str, Any]) -> dict[str, Any]:
    try:
        return generate_explanation(struct)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LLM explanation error: {exc}") from exc


@app.post("/api/qa")
async def qa(payload: dict[str, Any]) -> dict[str, Any]:
    struct = payload.get("struct")
    question = payload.get("question", "")
    if not struct or not question:
        raise HTTPException(status_code=400, detail="Both 'struct' and 'question' are required.")
    try:
        answer = qa_on_sample(struct, question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LLM Q&A error: {exc}") from exc
    return {"answer": answer}
