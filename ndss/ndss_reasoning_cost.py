#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ndss_reasoning_cost.py (Ablation 1/2/3 통합 + multi-model + token/cost logging)
---------------------------------------------------------------------------
- multi-model run (--models)
- usage 기반 토큰/비용 누적
- results/cost_summary.csv 저장 + 매 실행 후 정렬 + xlsx export
- GPT-5 계열 제약 반영:
  * max_tokens 미지원 -> max_completion_tokens 사용
  * temperature는 0.0 미지원(기본값 1만 허용) -> temperature 파라미터를 아예 보내지 않음
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

import json
import hashlib
from glob import glob
from collections import Counter

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

# ---------------- LLM 옵션 ----------------
USE_LLM = os.getenv("USE_LLM", "0") == "1"
try:
    if USE_LLM:
        from openai import OpenAI
        HAS_LLM = True
    else:
        HAS_LLM = False
except Exception:
    HAS_LLM = False
    USE_LLM = False


# ================================================================
# Token/Cost Accounting (usage 기반 정확 집계)
# ================================================================
# 가격은 "text tokens" 기준 (USD / 1M tokens)
# ⚠️ 반드시 최신 가격에 맞춰 네 환경에 맞게 유지/수정 필요
PRICE_PER_1M = {
    "gpt-4.1-nano": {"in": 0.10, "out": 0.40},
    "gpt-4o-mini":  {"in": 0.15, "out": 0.60},
    "gpt-4.1-mini": {"in": 0.40, "out": 1.60},
    "gpt-4.1":      {"in": 2.00, "out": 8.00},
    "gpt-4o":       {"in": 2.50, "out": 10.00},
    "gpt-5-nano":   {"in": 0.05, "out": 0.40},
    "gpt-5-mini":   {"in": 0.25, "out": 2.00},
    "gpt-5":        {"in": 1.25, "out": 10.00},
}

def make_usage_tracker(model_name: str) -> dict:
    return {
        "model": model_name,
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost_usd": 0.0,
    }

def add_usage(usage: dict, model_name: str, prompt_tokens: int, completion_tokens: int):
    usage["calls"] += 1
    usage["prompt_tokens"] += int(prompt_tokens)
    usage["completion_tokens"] += int(completion_tokens)

    if model_name not in PRICE_PER_1M:
        return
    p = PRICE_PER_1M[model_name]
    usage["cost_usd"] += (prompt_tokens / 1e6) * p["in"] + (completion_tokens / 1e6) * p["out"]

def _safe_model_name(model: str) -> str:
    return model.replace("/", "_").replace(":", "_").replace(" ", "_")


def _safe_float_tag(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def _default_hybrid_cache_path(model_name: str) -> str:
    safe_model = _safe_model_name(model_name)
    return os.path.join(os.path.dirname(__file__), "results", f"llm_hybrid_cache_{safe_model}.jsonl")


def _load_jsonl_cache(path: str) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if not os.path.exists(path):
        return cache
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            key = rec.get("key")
            value = rec.get("value")
            if isinstance(key, str) and isinstance(value, dict):
                cache[key] = value
    return cache


def _append_jsonl_cache(path: str, key: str, value: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")


def sort_and_export_cost_summary(cost_csv_path: str, sort_by: str = "cost_usd", ascending: bool = True):
    if not cost_csv_path or (not os.path.exists(cost_csv_path)):
        return
    df = pd.read_csv(cost_csv_path)
    if df.empty:
        return
    if sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)
    df.to_csv(cost_csv_path, index=False)

    xlsx_path = os.path.splitext(cost_csv_path)[0] + ".xlsx"
    try:
        df.to_excel(xlsx_path, index=False)
    except Exception as e:
        print(f"⚠️ Failed to write xlsx: {xlsx_path} ({e})")

    print(f"📁 Sorted & saved → {cost_csv_path}")
    print(f"📁 Exported xlsx → {xlsx_path}")


# ================================================================
# ✅ 핵심: 모델별 파라미터 제약(temperature/max_tokens) 자동 분기 래퍼
# ================================================================
def _is_gpt5(model_name: str) -> bool:
    m = (model_name or "").strip().lower().replace(" ", "")
    return m.startswith("gpt-5")

def _chat_create_json(client, model_name: str, prompt: str, max_out: int = 200):
    """
    공통 JSON 호출 래퍼
    - gpt-5* : max_completion_tokens 사용, temperature 파라미터 보내지 않음(기본값만 허용)
    - others: max_tokens 사용, temperature=0.0 고정
    - response_format=json_object 강제
    """
    kwargs = dict(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    if _is_gpt5(model_name):
        # ✅ gpt-5*: temperature=0.0 미지원 -> 아예 보내지 않음
        kwargs["max_completion_tokens"] = max_out
    else:
        kwargs["temperature"] = 0.0
        kwargs["max_tokens"] = max_out

    return client.chat.completions.create(**kwargs)


# ================================================================
# 1. NDSS 변수 스키마 (센서 + MV + Meta)
# ================================================================

NDSS_MEAS = [
    "A Feed","D Feed","E Feed","A and C Feed","Recycle Flow","Reactor Feed Rate",
    "Reactor Pressure","Reactor Level","Reactor Temperature","Purge Rate",
    "Product Sep Temp","Product Sep Level","Product Sep Pressure","Product Sep Underflow",
    "Stripper Level","Stripper Pressure","Stripper Underflow","Stripper Temp",
    "Stripper Steam Flow","Compressor Work","Reactor Coolant Temp","Separator Coolant Temp",
    "Comp A to Reactor","Comp B to Reactor","Comp C to Reactor","Comp D to Reactor",
    "Comp E to Reactor","Comp F to Reactor",
    "Comp A in Purge","Comp B in Purge","Comp C in Purge","Comp D in Purge",
    "Comp E in Purge","Comp F in Purure","Comp G in Purge","Comp H in Purge",
    "Comp D in Product","Comp E in Product","Comp F in Product",
    "Comp G in Product","Comp H in Product",
]

NDSS_ACT_BASE = [
    "D feed","E Feed","A Feed","A and C Feed","Recycle","Purge",
    "Separator","Stripper","Steam","Reactor Coolant","Condenser Coolant",
]
NDSS_META = ["Agitator"]

NDSS_ACT = [
    "D feed (MV)","E Feed (MV)","A Feed (MV)","A and C Feed (MV)","Recycle (MV)",
    "Purge (MV)","Separator (MV)","Stripper (MV)","Steam (MV)",
    "Reactor Coolant (MV)","Condenser Coolant (MV)",
]

NDSS_ALL_VARS = NDSS_MEAS + NDSS_ACT + NDSS_META


# ================================================================
# 2. NDSS CSV 로딩 & 컬럼 정규화
# ================================================================

def normalize_columns_with_mv(df: pd.DataFrame) -> pd.DataFrame:
    cols_raw = list(df.columns)
    cols_new = []
    for i, c in enumerate(cols_raw):
        base = c.split(".")[0].strip()
        if base == "Atk":
            cols_new.append("Atk")
            continue
        if i >= 41 and base in NDSS_ACT_BASE + NDSS_META:
            if base == "Agitator":
                cols_new.append("Agitator")
            else:
                cols_new.append(f"{base} (MV)")
        else:
            cols_new.append(base)
    df = df.copy()
    df.columns = cols_new
    return df

def load_and_clean_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = normalize_columns_with_mv(df)
    use_cols = [c for c in df.columns if c in NDSS_ALL_VARS or c == "Atk"]
    df = df[use_cols]
    return df


# ================================================================
# 3. Process Graph 로딩
# ================================================================

def load_process_graph(procmap_csv: str) -> nx.Graph:
    g = nx.Graph()
    if not os.path.exists(procmap_csv):
        print(f"⚠️ Process map not found: {procmap_csv}")
        return g

    df = pd.read_csv(procmap_csv)
    df.columns = [c.replace("\ufeff", "") for c in df.columns]

    required = {"src", "tgt", "corr"}
    if not required.issubset(df.columns):
        raise ValueError(f"Process map must have columns src,tgt,corr (got {set(df.columns)})")

    for _, r in df.iterrows():
        src, tgt = str(r["src"]), str(r["tgt"])
        corr = float(r["corr"])
        if src != tgt:
            g.add_edge(src, tgt, weight=abs(corr))

    print(f"📌 Loaded process graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    return g


# ================================================================
# 4. GraphAD core
# ================================================================

def flatline_score(series: pd.Series) -> float:
    s = series.values
    std = float(np.std(s))
    if std < 1e-8:
        return 1.0
    return 1.0 / (1.0 + std)

def compute_window_scores(
    win_df: pd.DataFrame,
    normal_median: pd.Series,
    normal_mad: pd.Series,
    w_z: float = 0.6,
    w_trend: float = 0.25,
    w_flat: float = 0.15,
) -> pd.Series:
    vars_used = list(win_df.columns)
    z = (win_df - normal_median[vars_used]) / (normal_mad[vars_used] + 1e-6)
    z_abs_mean = z.abs().mean(axis=0)

    t = np.arange(len(win_df))
    trends = {}
    for v in vars_used:
        y = win_df[v].values
        try:
            slope = np.polyfit(t, y, 1)[0]
        except Exception:
            slope = 0.0
        trends[v] = abs(float(slope))
    trend_series = pd.Series(trends)

    flats = {v: flatline_score(win_df[v]) for v in vars_used}
    flat_series = pd.Series(flats)

    final = w_z * z_abs_mean + w_trend * trend_series + w_flat * flat_series
    return final.sort_values(ascending=False)


# ================================================================
# 5. GraphAD: sliding window + graph smoothing
# ================================================================

def graph_smooth_scores(scores: pd.Series, g: nx.Graph, alpha: float = 0.3) -> pd.Series:
    if g is None or g.number_of_nodes() == 0:
        return scores

    smoothed = {}
    for v in scores.index:
        if v in g:
            neigh = list(g.neighbors(v))
            if neigh:
                neigh_scores = [scores[n] for n in neigh if n in scores.index]
                if neigh_scores:
                    smoothed[v] = (1 - alpha) * scores[v] + alpha * float(np.mean(neigh_scores))
                    continue
        smoothed[v] = scores[v]
    return pd.Series(smoothed).sort_values(ascending=False)

def graphad_scores_sliding(
    df: pd.DataFrame,
    win_len: int = 300,
    stride: int | None = None,
    g: nx.Graph | None = None,
    alpha_graph: float = 0.3,
    w_z: float = 0.6,
    w_trend: float = 0.25,
    w_flat: float = 0.15,
) -> pd.Series | None:
    if "Atk" not in df.columns:
        return None

    df0 = df[df["Atk"] == 0]
    df1 = df[df["Atk"] == 1]

    if len(df0) < 20 or len(df1) < 10:
        return None

    vars_used = [c for c in NDSS_ALL_VARS if c in df.columns]
    if not vars_used:
        return None

    df0 = df0[vars_used]
    df1 = df1[vars_used]

    normal_median = df0.median(axis=0)
    normal_mad = (df0 - normal_median).abs().median(axis=0) + 1e-6

    n1 = len(df1)
    if n1 < win_len:
        win_df = df1.tail(min(win_len, n1))
        scores = compute_window_scores(
            win_df,
            normal_median,
            normal_mad,
            w_z=w_z,
            w_trend=w_trend,
            w_flat=w_flat,
        )
        if g is not None:
            scores = graph_smooth_scores(scores, g, alpha_graph)
        return scores

    if stride is None:
        stride = max(win_len // 2, 50)

    agg_scores = None
    for start in range(0, n1 - win_len + 1, stride):
        end = start + win_len
        win_df = df1.iloc[start:end]
        s = compute_window_scores(
            win_df,
            normal_median,
            normal_mad,
            w_z=w_z,
            w_trend=w_trend,
            w_flat=w_flat,
        )
        if agg_scores is None:
            agg_scores = s
        else:
            agg_scores = pd.concat([agg_scores, s], axis=1).max(axis=1)

    if agg_scores is None:
        return None

    agg_scores = agg_scores.sort_values(ascending=False)

    if g is not None:
        agg_scores = graph_smooth_scores(agg_scores, g, alpha_graph)

    return agg_scores.sort_values(ascending=False)


# ================================================================
# 6. Window 길이 자동 최적화
# ================================================================

def auto_optimize_window(
    df: pd.DataFrame,
    true_vars: list[str],
    g: nx.Graph | None = None,
    candidate_windows=(100, 200, 300, 400, 500),
    w_z: float = 0.6,
    w_trend: float = 0.25,
    w_flat: float = 0.15,
) -> int:
    best_win = None
    best_score = -1e9
    for w in candidate_windows:
        scores = graphad_scores_sliding(
            df,
            win_len=w,
            g=g,
            alpha_graph=0.0,
            w_z=w_z,
            w_trend=w_trend,
            w_flat=w_flat,
        )
        if scores is None or scores.empty:
            continue
        ranks = []
        for v in true_vars:
            if v in scores.index:
                ranks.append(list(scores.index).index(v))
        if not ranks:
            continue
        best_rank = min(ranks)
        score = -best_rank
        if score > best_score:
            best_score = score
            best_win = w
    if best_win is None:
        best_win = 300
    return best_win


# ================================================================
# 7. Reasoning 품질 지표
# ================================================================

def compute_ACT(topk: list[str], true_vars: list[str]) -> float:
    return 1.0 if any(v in topk for v in true_vars) else 0.0

def compute_KConcord(topk: list[str], true_vars: list[str]) -> float:
    ranks = []
    for v in true_vars:
        if v in topk:
            ranks.append(topk.index(v))
    if not ranks:
        return 0.0
    best_rank = min(ranks)
    return 1.0 / (best_rank + 1.0)

def compute_SGR(scores: pd.Series, true_vars: list[str]) -> float:
    max_all = float(scores.max())
    if max_all <= 0:
        return 0.0
    vals = []
    for v in true_vars:
        if v in scores.index:
            vals.append(float(scores[v]) / max_all)
    if not vals:
        return 0.0
    return max(vals)


# ================================================================
# 8. LLM 보조 함수
# ================================================================

def _base_name_for_mv(var: str) -> str:
    if var.endswith(" (MV)"):
        return var[:-5]
    return var

def deduplicate_top5_by_mv_meas(top5: list[str]) -> tuple[list[str], int]:
    base_to_indices: dict[str, list[int]] = {}
    for idx, v in enumerate(top5):
        base = _base_name_for_mv(v)
        base_to_indices.setdefault(base, []).append(idx)

    remove_indices: set[int] = set()
    for base, idx_list in base_to_indices.items():
        if len(idx_list) >= 2:
            idx_list_sorted = sorted(idx_list)
            for rm_idx in idx_list_sorted[1:]:
                remove_indices.add(rm_idx)

    new_top5 = [v for i, v in enumerate(top5) if i not in remove_indices]
    removed_count = len(remove_indices)
    return new_top5, removed_count


def llm_filter_candidates(
    client,
    graph: nx.Graph,
    low_candidates: list[str],
    scores: pd.Series,
    max_edges_per_var: int = 5,
    usage: dict | None = None,
) -> list[str]:
    if (not USE_LLM) or (not HAS_LLM) or client is None:
        return low_candidates
    if not low_candidates:
        return low_candidates

    lines = []
    for v in low_candidates:
        neigh = []
        if graph is not None and v in graph.nodes:
            neigh = list(graph.neighbors(v))[:max_edges_per_var]
        sc = float(scores.get(v, 0.0))
        neigh_txt = ", ".join(neigh) if neigh else "N/A"
        lines.append(f"- {v}: score={sc:.4f}, neighbors=[{neigh_txt}]")

    vars_block = "\n".join(lines)
    cand_list_txt = ", ".join(low_candidates)

    prompt = f"""
당신은 공정 데이터 이상탐지 결과를 검토하는 전문가입니다.
아래는 GraphAD가 선택한 Top-6~10 후보 변수들입니다.

각 변수는 이상 점수(score)와 공정 그래프 상의 이웃 변수(neighbors)를 포함합니다:

{vars_block}

위 후보들 중에서, **공정 이상 원인과 직접적으로 관련성이 낮아 보이는 변수들만 제거**해 주세요.
반대로, 관련성이 있어 보이는 변수는 그대로 남겨둡니다.

다음 형식의 JSON만 출력해 주세요 (설명 없이):

{{"keep": ["변수이름1", "변수이름2", ...]}}
단, keep 리스트는 반드시 아래 후보들({cand_list_txt})의 부분집합이어야 합니다.
"""

    model_name = getattr(client, "_ndss_model_name", "gpt-4o-mini")
    resp = _chat_create_json(client, model_name, prompt, max_out=200)

    if usage is not None and hasattr(resp, "usage") and resp.usage is not None:
        pt = getattr(resp.usage, "prompt_tokens", 0) or 0
        ct = getattr(resp.usage, "completion_tokens", 0) or 0
        add_usage(usage, model_name, int(pt), int(ct))

    txt = resp.choices[0].message.content.strip()
    try:
        data = json.loads(txt)
        keep = [v for v in data.get("keep", []) if v in low_candidates]
        if not keep:
            return low_candidates
        return [v for v in low_candidates if v in keep]
    except Exception:
        return low_candidates


def llm_rerank_top10_hybrid(
    client,
    graph: nx.Graph,
    candidates: list[str],
    scores: pd.Series,
    beta: float = 0.2,
    max_edges_per_var: int = 5,
    usage: dict | None = None,
) -> list[str]:
    if beta <= 0.0:
        return candidates
    if (not USE_LLM) or (not HAS_LLM) or client is None:
        return candidates
    if not candidates:
        return candidates

    lines = []
    for v in candidates:
        neigh = []
        if graph is not None and v in graph.nodes:
            neigh = list(graph.neighbors(v))[:max_edges_per_var]
        sc = float(scores.get(v, 0.0))
        neigh_txt = ", ".join(neigh) if neigh else "N/A"
        lines.append(f"- {v}: score={sc:.4f}, neighbors=[{neigh_txt}]")
    vars_block = "\n".join(lines)

    cand_list_txt = ", ".join(candidates)

    prompt = f"""
당신은 공정 데이터 이상탐지 결과를 검토하는 전문가입니다.
아래는 GraphAD가 선택한 Top-10 후보 변수들입니다. 각 변수는 이상 점수와 공정 그래프 이웃 정보를 포함합니다.

{vars_block}

각 변수마다, 해당 변수가 이번 공정 이상(공격)의 **직접적인 원인일 가능성**을 0~1 사이의 실수로 평가해 주세요.
0은 거의 관련 없음, 1은 매우 직접적인 원인입니다.

다음 JSON 형식만 출력해 주세요 (설명 없이):

{{
  "scores": {{
    "변수이름1": 0.8,
    "변수이름2": 0.2
  }}
}}

단, 변수이름은 반드시 다음 후보 리스트({cand_list_txt}) 안에서만 사용하세요.
"""

    model_name = getattr(client, "_ndss_model_name", "gpt-4o-mini")
    cache = getattr(client, "_ndss_llm_cache", None)
    cache_path = getattr(client, "_ndss_llm_cache_path", None)
    cache_key = hashlib.sha256(f"{model_name}\n{prompt}".encode("utf-8")).hexdigest()
    cached_entry = cache.get(cache_key) if isinstance(cache, dict) else None

    if cached_entry is not None:
        llm_scores_raw = dict(cached_entry.get("scores", {}))
        if usage is not None:
            add_usage(
                usage,
                model_name,
                int(cached_entry.get("prompt_tokens", 0) or 0),
                int(cached_entry.get("completion_tokens", 0) or 0),
            )
    else:
        resp = _chat_create_json(client, model_name, prompt, max_out=200)

        pt = 0
        ct = 0
        if hasattr(resp, "usage") and resp.usage is not None:
            pt = int(getattr(resp.usage, "prompt_tokens", 0) or 0)
            ct = int(getattr(resp.usage, "completion_tokens", 0) or 0)
            if usage is not None:
                add_usage(usage, model_name, pt, ct)

        txt = resp.choices[0].message.content.strip()
        try:
            data = json.loads(txt)
            llm_scores_raw = dict(data.get("scores", {}))
        except Exception:
            return candidates

        cache_entry = {
            "scores": llm_scores_raw,
            "prompt_tokens": pt,
            "completion_tokens": ct,
        }
        if isinstance(cache, dict):
            cache[cache_key] = cache_entry
        if isinstance(cache_path, str):
            _append_jsonl_cache(cache_path, cache_key, cache_entry)

    cand_arr = np.array([float(scores.get(v, 0.0)) for v in candidates], dtype=float)
    if cand_arr.max() > cand_arr.min():
        gad_norm = (cand_arr - cand_arr.min()) / (cand_arr.max() - cand_arr.min())
    else:
        gad_norm = np.zeros_like(cand_arr)

    llm_arr = np.array([float(llm_scores_raw.get(v, 0.0)) for v in candidates], dtype=float)
    if llm_arr.max() > llm_arr.min():
        llm_norm = (llm_arr - llm_arr.min()) / (llm_arr.max() - llm_arr.min())
    else:
        llm_norm = np.zeros_like(llm_arr)

    final_score = (1.0 - beta) * gad_norm + beta * llm_norm
    idx_sorted = np.argsort(-final_score)
    return [candidates[i] for i in idx_sorted]


# ================================================================
# 9. NDSS Reasoning 메인 루프
# ================================================================

def re_split_multi(s: str):
    import re
    return re.split(r"[;,]", s)

def best_true_var_rank_from_ranking(ranking: list[str], true_vars: list[str]) -> int | None:
    ranks = [ranking.index(v) + 1 for v in true_vars if v in ranking]
    if not ranks:
        return None
    return min(ranks)


def reciprocal_rank_from_ranking(ranking: list[str], true_vars: list[str]) -> float:
    rank = best_true_var_rank_from_ranking(ranking, true_vars)
    if rank is None:
        return 0.0
    return 1.0 / float(rank)


def save_final_performance(results: list[dict], out_path: str, mode: str, run_params: dict | None = None):
    import json as _json
    import numpy as _np

    if not results:
        print("⚠️ No NDSS results to summarize.")
        return

    acts = _np.array([r["ACT"] for r in results], dtype=float)
    kcs = _np.array([r["K-Concord"] for r in results], dtype=float)
    sgris = _np.array([r["SGR"] for r in results], dtype=float)

    top1_hits = _np.array([r["top1_hit"] for r in results], dtype=float)
    top3_hits = _np.array([r["top3_hit"] for r in results], dtype=float)
    top5_hits = _np.array([r["top5_hit"] for r in results], dtype=float)
    reciprocal_ranks = _np.array([r.get("reciprocal_rank", 0.0) for r in results], dtype=float)
    best_true_ranks = _np.array([r.get("best_true_rank", _np.nan) for r in results], dtype=float)
    best_true_ranks = best_true_ranks[~_np.isnan(best_true_ranks)]

    summary = {
        "mode": mode,
        "attacks_evaluated": len(results),
        "ACT_Hit@5": float(acts.mean()),
        "K-Concord": float(kcs.mean()),
        "SGR": float(sgris.mean()),
        "Top1_recall": float(top1_hits.mean()),
        "Top3_recall": float(top3_hits.mean()),
        "Top5_recall": float(top5_hits.mean()),
        "MRR": float(reciprocal_ranks.mean()) if len(reciprocal_ranks) else float("nan"),
        "best_true_rank_mean": float(best_true_ranks.mean()) if len(best_true_ranks) else float("nan"),
        "ranking_variance": float(best_true_ranks.var()) if len(best_true_ranks) else float("nan"),
        "true_var_distribution": dict(Counter([tv for r in results for tv in r.get("true_vars", [])])),
    }
    if run_params:
        summary["run_params"] = run_params

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        _json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n📁 Saved NDSS final summary → {out_path}")


def run_ndss_reasoning(
    scen_dir: str,
    procmap_csv: str,
    gt_csv: str,
    out_json: str,
    mode: str = "no_llm",
    llm_model: str = "gpt-4o-mini",
    cost_csv_path: str | None = None,
    append_cost_csv: bool = True,
    sort_by: str = "cost_usd",
    sort_ascending: bool = True,
    alpha_graph: float = 0.3,
    w_z: float = 0.6,
    w_trend: float = 0.25,
    w_flat: float = 0.15,
    beta: float = 0.2,
):
    mode = mode.lower()
    if mode not in {"no_llm", "filter", "hybrid"}:
        raise ValueError(f"Unknown mode: {mode}")

    g = load_process_graph(procmap_csv)

    gt_df = pd.read_csv(gt_csv)
    if "attack_id" not in gt_df.columns or "true_var" not in gt_df.columns:
        raise ValueError("gt_csv must have columns: attack_id,true_var,...")
    gt_df = gt_df.set_index("attack_id")

    files = sorted(glob(os.path.join(scen_dir, "*.csv")))
    if not files:
        raise RuntimeError(f"No scenario CSV in {scen_dir}")

    client = None
    usage = make_usage_tracker(llm_model)

    if mode in {"filter", "hybrid"} and USE_LLM and HAS_LLM:
        client = OpenAI()
        setattr(client, "_ndss_model_name", llm_model)
        cache_path = _default_hybrid_cache_path(llm_model)
        setattr(client, "_ndss_llm_cache_path", cache_path)
        setattr(client, "_ndss_llm_cache", _load_jsonl_cache(cache_path))
    elif mode in {"filter", "hybrid"}:
        print("⚠️ LLM 사용 설정이지만, USE_LLM/HAS_LLM 설정으로 인해 no_llm로 강제 전환됩니다.")
        mode = "no_llm"

    print(f"\n🚀 NDSS Reasoning 시작 (mode={mode}, model={llm_model}, scenarios={len(files)})")

    results: list[dict] = []
    all_act, all_kc, all_sgr = [], [], []
    top1_hit_list, top3_hit_list, top5_hit_list = [], [], []
    true_var_counter = Counter()

    for f in tqdm(files, desc=f"NDSS reasoning ({llm_model})"):
        df = load_and_clean_csv(f)
        attack_id = os.path.basename(f).replace(".csv", "")

        if attack_id not in gt_df.index:
            continue

        tv_raw = str(gt_df.loc[attack_id, "true_var"])
        true_vars = [t.strip() for t in re_split_multi(tv_raw) if t.strip()]
        if not true_vars:
            continue

        for v in true_vars:
            true_var_counter[v] += 1

        best_w = auto_optimize_window(df, true_vars, g=None, w_z=w_z, w_trend=w_trend, w_flat=w_flat)

        scores = graphad_scores_sliding(
            df,
            win_len=best_w,
            g=g,
            alpha_graph=alpha_graph,
            w_z=w_z,
            w_trend=w_trend,
            w_flat=w_flat,
        )
        if scores is None or scores.empty:
            continue

        top10_graphad = list(scores.head(10).index)
        graphad_top5 = top10_graphad[:5]

        if mode == "no_llm":
            top5_final = graphad_top5
            final_ranking = list(scores.index)

        elif mode == "filter":
            dedup_top5, removed = deduplicate_top5_by_mv_meas(graphad_top5)
            low_candidates = top10_graphad[5:10]

            if removed > 0 and low_candidates:
                filtered_low = llm_filter_candidates(client, g, low_candidates, scores, usage=usage)
                top5_final = dedup_top5.copy()
                for v in low_candidates:
                    if len(top5_final) >= 5:
                        break
                    if v in filtered_low and v not in top5_final:
                        top5_final.append(v)
                if len(top5_final) < 5:
                    for v in low_candidates:
                        if len(top5_final) >= 5:
                            break
                        if v not in top5_final:
                            top5_final.append(v)
            else:
                top5_final = graphad_top5
            final_ranking = top5_final + [v for v in top10_graphad if v not in top5_final] + [
                v for v in list(scores.index) if v not in top10_graphad
            ]

        elif mode == "hybrid":
            ranked10 = llm_rerank_top10_hybrid(
                client, g, top10_graphad, scores, beta=beta, usage=usage
            )
            top5_ranked = ranked10[:5]
            dedup_top5, removed = deduplicate_top5_by_mv_meas(top5_ranked)

            if removed > 0:
                rest_candidates = [v for v in ranked10 if v not in dedup_top5]
                top5_final = dedup_top5.copy()
                for v in rest_candidates:
                    if len(top5_final) >= 5:
                        break
                        if v not in top5_final:
                            top5_final.append(v)
            else:
                top5_final = top5_ranked
            final_ranking = top5_final + [v for v in ranked10 if v not in top5_final] + [
                v for v in list(scores.index) if v not in ranked10
            ]

        else:
            top5_final = graphad_top5
            final_ranking = list(scores.index)

        top3_final = top5_final[:3]
        top1_final = top5_final[:1]
        best_true_rank = best_true_var_rank_from_ranking(final_ranking, true_vars)
        reciprocal_rank = reciprocal_rank_from_ranking(final_ranking, true_vars)

        act = compute_ACT(top5_final, true_vars)
        kc = compute_KConcord(top5_final, true_vars)
        sgr = compute_SGR(scores, true_vars)

        all_act.append(act)
        all_kc.append(kc)
        all_sgr.append(sgr)

        top1_hit_list.append(1.0 if any(v in top1_final for v in true_vars) else 0.0)
        top3_hit_list.append(1.0 if any(v in top3_final for v in true_vars) else 0.0)
        top5_hit_list.append(1.0 if any(v in top5_final for v in true_vars) else 0.0)

        results.append({
            "attack_id": attack_id,
            "true_vars": true_vars,
            "window": best_w,
            "top10_graphad": top10_graphad,
            "top5_final": top5_final,
            "ACT": act,
            "K-Concord": kc,
            "SGR": sgr,
            "top1_hit": top1_hit_list[-1],
            "top3_hit": top3_hit_list[-1],
            "top5_hit": top5_hit_list[-1],
            "best_true_rank": best_true_rank,
            "reciprocal_rank": reciprocal_rank,
        })

    n = len(results)
    if n == 0:
        print("⚠️ No valid NDSS scenarios processed.")
        return None

    ACT_mean = float(np.mean(all_act))
    KC_mean = float(np.mean(all_kc))
    SGR_mean = float(np.mean(all_sgr))
    top1 = float(np.mean(top1_hit_list))
    top3 = float(np.mean(top3_hit_list))
    top5 = float(np.mean(top5_hit_list))
    reciprocal_ranks = np.array([r.get("reciprocal_rank", 0.0) for r in results], dtype=float)
    rank_values = np.array([r.get("best_true_rank", np.nan) for r in results], dtype=float)
    rank_values = rank_values[~np.isnan(rank_values)]
    mrr = float(np.mean(reciprocal_ranks)) if len(reciprocal_ranks) else float("nan")
    ranking_variance = float(np.var(rank_values)) if len(rank_values) else float("nan")
    best_true_rank_mean = float(np.mean(rank_values)) if len(rank_values) else float("nan")

    print(f"\n========== NDSS {mode.upper()} FINAL PERFORMANCE ==========")
    print(f"Model             : {llm_model}")
    print(f"Attacks evaluated : {n}")
    print(f"ACT (Hit@5)      : {ACT_mean:.4f}")
    print(f"K-Concord        : {KC_mean:.4f}")
    print(f"SGR              : {SGR_mean:.4f}")
    print(f"Top-1 Recall     : {top1:.4f}")
    print(f"Top-3 Recall     : {top3:.4f}")
    print(f"Top-5 Recall     : {top5:.4f}")
    print(f"MRR              : {mrr:.4f}")
    print(f"Ranking variance : {ranking_variance:.4f}")

    print("\nLLM usage summary:")
    print(f"  calls            : {usage['calls']}")
    print(f"  prompt_tokens    : {usage['prompt_tokens']}")
    print(f"  completion_tokens: {usage['completion_tokens']}")
    print(f"  total_tokens     : {usage['prompt_tokens'] + usage['completion_tokens']}")
    if llm_model in PRICE_PER_1M:
        print(f"  cost_usd         : {usage['cost_usd']:.6f}")
    else:
        print("  cost_usd         : (price not configured for this model)")

    print("\nTrue-var distribution (top 20):")
    for v, c in true_var_counter.most_common(20):
        print(f"  {v}: {c}")

    # ============================================================
    # 결과 저장
    # ============================================================
    out_dir = os.path.dirname(out_json)
    os.makedirs(out_dir, exist_ok=True)

    safe_model = _safe_model_name(llm_model)
    beta_tag = _safe_float_tag(beta)
    detail_path = os.path.join(out_dir, f"ndss_reasoning_{mode}_{safe_model}_beta{beta_tag}.json")
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n📁 Saved NDSS detailed results → {detail_path}")

    summary_path = os.path.join(out_dir, f"ndss_performance_summary_{mode}_{safe_model}_beta{beta_tag}.json")
    save_final_performance(
        results,
        summary_path,
        mode,
        run_params={
            "model": llm_model,
            "alpha_graph": alpha_graph,
            "w_z": w_z,
            "w_trend": w_trend,
            "w_flat": w_flat,
            "beta": beta,
        },
    )

    # ============================================================
    # cost_summary.csv 저장 + 정렬 + xlsx export
    # ============================================================
    if cost_csv_path is not None:
        row = {
            "mode": mode,
            "model": llm_model,
            "attacks_evaluated": n,
            "ACT_Hit@5": ACT_mean,
            "K-Concord": KC_mean,
            "SGR": SGR_mean,
            "Top1_recall": top1,
            "Top3_recall": top3,
            "Top5_recall": top5,
            "MRR": mrr,
            "best_true_rank_mean": best_true_rank_mean,
            "ranking_variance": ranking_variance,
            "llm_calls": usage["calls"],
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["prompt_tokens"] + usage["completion_tokens"],
            "cost_usd": usage["cost_usd"],
            "avg_cost_per_attack": (usage["cost_usd"] / n) if n > 0 else 0.0,
            "alpha_graph": alpha_graph,
            "w_z": w_z,
            "w_trend": w_trend,
            "w_flat": w_flat,
            "beta": beta,
        }

        os.makedirs(os.path.dirname(cost_csv_path), exist_ok=True)
        df_row = pd.DataFrame([row])

        if append_cost_csv and os.path.exists(cost_csv_path):
            df_row.to_csv(cost_csv_path, mode="a", header=False, index=False)
        else:
            df_row.to_csv(cost_csv_path, index=False)

        sort_and_export_cost_summary(cost_csv_path, sort_by=sort_by, ascending=sort_ascending)

    return {
        "mode": mode,
        "model": llm_model,
        "attacks_evaluated": n,
        "ACT_Hit@5": ACT_mean,
        "K-Concord": KC_mean,
        "SGR": SGR_mean,
        "Top1_recall": top1,
        "Top3_recall": top3,
        "Top5_recall": top5,
        "MRR": mrr,
        "best_true_rank_mean": best_true_rank_mean,
        "ranking_variance": ranking_variance,
        "usage": usage,
        "detail_path": detail_path,
        "summary_path": summary_path,
    }


if __name__ == "__main__":
    import argparse

    default_scen = os.path.join(os.path.dirname(__file__), "..", "data", "ndss_scenarios")
    default_proc = os.path.join(os.path.dirname(__file__), "..", "data", "NDSS_process_edges.csv")
    default_gt = os.path.join(os.path.dirname(__file__), "..", "data", "ndss_attack_scenarios.csv")

    default_out = os.path.join(os.path.dirname(__file__), "..", "results", "ndss_reasoning_results.json")

    p = argparse.ArgumentParser()
    p.add_argument("--scen_dir", default=default_scen)
    p.add_argument("--procmap", default=default_proc)
    p.add_argument("--gt_csv", default=default_gt)
    p.add_argument("--out_json", default=default_out)
    p.add_argument("--mode", default="no_llm", choices=["no_llm", "filter", "hybrid"])

    p.add_argument(
        "--models",
        default="",
        help="Comma-separated model list. If set, runs sequentially and writes results/cost_summary.csv",
    )
    p.add_argument(
        "--cost_csv",
        default=os.path.join(os.path.dirname(__file__), "..", "results", "cost_summary.csv"),
        help="Output CSV for cost/perf summary",
    )
    p.add_argument(
        "--sort_by",
        default="cost_usd",
        help="Sort key for cost_summary (e.g., cost_usd, ACT_Hit@5, K-Concord, SGR)",
    )
    p.add_argument(
        "--sort_desc",
        action="store_true",
        help="If set, sort descending (useful for sorting by performance metrics).",
    )
    p.add_argument("--alpha_graph", type=float, default=0.3)
    p.add_argument("--w_z", type=float, default=0.6)
    p.add_argument("--w_trend", type=float, default=0.25)
    p.add_argument("--w_flat", type=float, default=0.15)
    p.add_argument("--beta", type=float, default=0.2)

    args = p.parse_args()

    if args.models.strip():
        model_list = [m.strip() for m in args.models.split(",") if m.strip()]
        first = True
        for m in model_list:
            run_ndss_reasoning(
                scen_dir=args.scen_dir,
                procmap_csv=args.procmap,
                gt_csv=args.gt_csv,
                out_json=args.out_json,
                mode=args.mode,
                llm_model=m,
                cost_csv_path=args.cost_csv,
                append_cost_csv=(not first),
                sort_by=args.sort_by,
                sort_ascending=(not args.sort_desc),
                alpha_graph=args.alpha_graph,
                w_z=args.w_z,
                w_trend=args.w_trend,
                w_flat=args.w_flat,
                beta=args.beta,
            )
            first = False
    else:
        run_ndss_reasoning(
            scen_dir=args.scen_dir,
            procmap_csv=args.procmap,
            gt_csv=args.gt_csv,
            out_json=args.out_json,
            mode=args.mode,
            llm_model="gpt-4o-mini",
            cost_csv_path=None,
            alpha_graph=args.alpha_graph,
            w_z=args.w_z,
            w_trend=args.w_trend,
            w_flat=args.w_flat,
            beta=args.beta,
        )
