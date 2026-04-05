from __future__ import annotations

import json
import os
import re
from typing import Any

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

from .config import OPENAI_MODEL_EXPLAIN, OPENAI_MODEL_QA


def _get_client() -> OpenAI | None:
    if OpenAI is None:
        return None
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def _strip_markdown_artifacts(text: str) -> str:
    cleaned = text.replace("```", "")
    cleaned = cleaned.replace("**", "")
    cleaned = cleaned.replace("__", "")
    cleaned = cleaned.replace("`", "")
    cleaned = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _build_explain_prompt(struct: dict[str, Any]) -> str:
    a = struct["anomaly_scores"]
    g_list = struct["graphad_topk_list"]
    pg = struct["process_graph_context"]
    meta = struct["meta_features"]
    sub = struct["subgraph_context"]

    return f"""
You are an industrial process expert specializing in anomaly diagnosis for the Tennessee Eastman Process (TEP).
Use the UTAR detection evidence and GraphAD+ structural signals below to explain whether this sample is normal or anomalous and what the most likely root causes are.

[1] UTAR Detection Evidence
- UTAR base score: {a['utar_base_score']}
- RF score: {a['rf_prob']}
- XGBoost score: {a['xgb_prob']}
- ModernTCN score: {a['tcn_prob']}
- Ensemble mean: {a['ensemble_mean']}
- Ensemble entropy: {a['ensemble_entropy']}
- Model discrepancy: {a['model_discrepancy']}
- Final routed score: {a['final_prob']}
- Tau: {a['tau']}
- Selected q: {a['selected_q']}
- Gray-zone margin: {a['gray_margin']}
- Gray-zone flag: {a['gray_zone']}
- Selective LLM called: {a['llm_called']}
- LLM routing decision: {a['llm_decision']}
- Final decision source: {a['decision_source']}
- Final anomaly decision: {a['final_decision']}

[2] GraphAD+ Top-K Variables
{json.dumps(g_list, ensure_ascii=False, indent=2)}

[3] Process Graph Structure
{json.dumps(pg, ensure_ascii=False, indent=2)}

[4] Meta Features
{json.dumps(meta, ensure_ascii=False, indent=2)}

[5] Subgraph Context
{json.dumps(sub, ensure_ascii=False, indent=2)}

Explanation instructions:
1. Explain why the sample was classified as normal or anomalous, clearly separating the roles of RF, XGBoost, ModernTCN, the UTAR base score, and the Selective LLM decision.
2. Use the GraphAD+ Top-K variables and process structure to propose 1 to 3 candidate propagation paths for the anomaly.
3. Always express process flow with arrows (→).
4. Keep the original process names and variable names unchanged.
5. Write the answer in concise, technical English bullets.
6. Use plain text only. Do not use Markdown formatting such as **bold**, headings, code fences, or backticks.

At the end, append exactly one JSON block in the following format.

[[PATH_JSON]]
{{
  "paths": [
    ["Stripper:xmeas_18", "Stripper:xmeas_19"],
    ["Reactor:xmeas_25", "Purge Gas Analysis:xmeas_31"]
  ]
}}
[[/PATH_JSON]]
"""


def generate_explanation(struct: dict[str, Any]) -> dict[str, Any]:
    client = _get_client()
    if client is None:
        return {
            "explanation": "[PLACEHOLDER] OPENAI_API_KEY is required.",
            "paths": [],
        }

    prompt = _build_explain_prompt(struct)
    resp = client.chat.completions.create(
        model=OPENAI_MODEL_EXPLAIN,
        messages=[
            {"role": "system", "content": "You are an expert industrial anomaly explainer. Always answer in English."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    full = resp.choices[0].message.content.strip()
    start_token = "[[PATH_JSON]]"
    end_token = "[[/PATH_JSON]]"
    start = full.find(start_token)
    end = full.find(end_token)
    explanation_text = full
    paths: list[list[str]] = []

    if start != -1 and end != -1 and end > start:
        json_str = full[start + len(start_token) : end].strip()
        explanation_text = (full[:start] + full[end + len(end_token) :]).strip()
        try:
            obj = json.loads(json_str)
            if isinstance(obj, dict) and isinstance(obj.get("paths"), list):
                paths = obj["paths"]
        except Exception:
            paths = []

    return {"explanation": _strip_markdown_artifacts(explanation_text), "paths": paths}


def qa_on_sample(struct: dict[str, Any], question: str) -> str:
    client = _get_client()
    if client is None:
        return "[PLACEHOLDER] OPENAI_API_KEY is required."

    ctx = json.dumps(struct, ensure_ascii=False, indent=2)
    prompt = f"""
You are an expert in Tennessee Eastman Process anomaly diagnosis.

Below is the UTAR / GraphAD+ context for the currently selected sample.

{ctx}

Question:
{question}

Answer only from the context above. Be concise, technically accurate, and write in English.
When useful, mention both the process name and the variable name, and add one short suggestion for a helpful follow-up visualization.
Use plain text only. Do not use Markdown formatting such as **bold**, headings, code fences, or backticks.
"""

    resp = client.chat.completions.create(
        model=OPENAI_MODEL_QA,
        messages=[
            {"role": "system", "content": "You are an expert industrial process anomaly explainer. Always answer in English."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return _strip_markdown_artifacts(resp.choices[0].message.content.strip())
