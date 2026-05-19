#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ndss_reasoning.py (Ablation 1/2/3 통합 버전)
------------------------------------------
NDSS 조작 시나리오에 대해

1) GraphAD 기반 변수별 이상 점수 계산 (고도화 버전)
   - Normal(Atk=0) 기준 median + MAD 정규화
   - trend score (윈도우 내 선형 기울기)
   - flatline score (DoS/freeze 탐지)
   - 공격 구간에서 sliding window max-aggregation
   - process graph 기반 score smoothing

2) 시나리오별 true_var(센서 + MV 세트)에 대해
   - ACT (Hit@5)
   - K-Concord
   - SGR
   - Top-1/3/5 Recall

3) Ablation 모드
   - mode = "no_llm"   : GraphAD Top-5 그대로 사용 (baseline)
   - mode = "filter"   : GraphAD Top5 고정 + Top6~10 LLM 필터링 + slot optimization
   - mode = "hybrid"   : GraphAD score + LLM relevance blending 재랭킹 + slot optimization

환경 변수/CLI
-------------
- USE_LLM=1 인 경우에만 OpenAI 클라이언트 사용 시도
- CLI --mode {no_llm,filter,hybrid}
  * 지정하지 않으면:
    - USE_LLM=1 & OpenAI 가능 → hybrid
    - 그 외 → no_llm

입출력 기본 경로 (acl_graph_reasoning_pipeline_v5 기준):
- 시나리오 CSV   : data/ndss_scenarios/*.csv
- process graph  : data/NDSS_process_edges.csv
- true_var GT    : data/ndss_attack_scenarios.csv
- 결과 JSON      : results/ndss_reasoning_{mode}.json        ✅ (이번 수정으로 저장됨)
- 요약 JSON      : results/ndss_performance_summary_{mode}.json
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

import os
import json
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
# 1. NDSS 변수 스키마 (센서 + MV + Meta)
# ================================================================

# 센서 (measurement) 41개
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

# 원본 MV/Meta 이름 (CSV 뒤쪽에 나오는 열)
NDSS_ACT_BASE = [
    "D feed","E Feed","A Feed","A and C Feed","Recycle","Purge",
    "Separator","Stripper","Steam","Reactor Coolant","Condenser Coolant",
]
NDSS_META = ["Agitator"]

# CSV 로딩 시 MV는 "XXX (MV)"로 바꿔서 NDSS_MEAS와 구분
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


def compute_window_components(
    win_df: pd.DataFrame,
    normal_median: pd.Series,
    normal_mad: pd.Series,
) -> pd.DataFrame:
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

    return pd.DataFrame(
        {
            "z_abs_mean": z_abs_mean,
            "trend": trend_series,
            "flat": flat_series,
        }
    ).reindex(vars_used)


def combine_window_components(
    components: pd.DataFrame,
    w_z: float = 0.6,
    w_trend: float = 0.25,
    w_flat: float = 0.15,
) -> pd.Series:
    final = (
        w_z * components["z_abs_mean"]
        + w_trend * components["trend"]
        + w_flat * components["flat"]
    )
    return final.sort_values(ascending=False)


def compute_window_scores(
    win_df: pd.DataFrame,
    normal_median: pd.Series,
    normal_mad: pd.Series,
    w_z: float = 0.6,
    w_trend: float = 0.25,
    w_flat: float = 0.15,
) -> pd.Series:
    components = compute_window_components(win_df, normal_median, normal_mad)
    return combine_window_components(components, w_z=w_z, w_trend=w_trend, w_flat=w_flat)


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


def prepare_graphad_bundle(
    df: pd.DataFrame,
    win_len: int = 300,
    stride: int | None = None,
) -> dict | None:
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
    if stride is None:
        stride = max(win_len // 2, 50)

    if n1 < win_len:
        windows = [df1.tail(min(win_len, n1))]
    else:
        windows = [
            df1.iloc[start:start + win_len]
            for start in range(0, n1 - win_len + 1, stride)
        ]

    if not windows:
        return None

    component_frames = [
        compute_window_components(win_df, normal_median, normal_mad).reindex(vars_used)
        for win_df in windows
    ]
    component_array = np.stack(
        [frame[["z_abs_mean", "trend", "flat"]].to_numpy(dtype=float) for frame in component_frames],
        axis=0,
    )

    return {
        "vars_used": vars_used,
        "win_len": win_len,
        "stride": stride,
        "components": component_array,
    }


def score_prepared_graphad_bundle(
    bundle: dict | None,
    g: nx.Graph | None = None,
    alpha_graph: float = 0.3,
    w_z: float = 0.6,
    w_trend: float = 0.25,
    w_flat: float = 0.15,
) -> pd.Series | None:
    if bundle is None:
        return None

    weights = np.array([w_z, w_trend, w_flat], dtype=float)
    window_scores = bundle["components"] @ weights
    agg_scores = window_scores.max(axis=0)
    scores = pd.Series(agg_scores, index=bundle["vars_used"]).sort_values(ascending=False)

    if g is not None:
        scores = graph_smooth_scores(scores, g, alpha_graph)

    return scores.sort_values(ascending=False)


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
    bundle = prepare_graphad_bundle(df, win_len=win_len, stride=stride)
    return score_prepared_graphad_bundle(
        bundle,
        g=g,
        alpha_graph=alpha_graph,
        w_z=w_z,
        w_trend=w_trend,
        w_flat=w_flat,
    )


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

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

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

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    txt = resp.choices[0].message.content.strip()

    try:
        data = json.loads(txt)
        llm_scores_raw: dict[str, float] = data.get("scores", {})
    except Exception:
        return candidates

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


def save_final_performance(
    results: list[dict],
    out_path: str,
    mode: str,
    graphad_params: dict | None = None,
):
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
    if graphad_params:
        summary["graphad_params"] = graphad_params

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
    if mode in {"filter", "hybrid"} and USE_LLM and HAS_LLM:
        client = OpenAI()
    elif mode in {"filter", "hybrid"}:
        print("⚠️ LLM 사용 설정이지만, USE_LLM/HAS_LLM 설정으로 인해 no_llm로 강제 전환됩니다.")
        mode = "no_llm"

    print(f"\n🚀 NDSS Reasoning 시작 (mode={mode}, scenarios={len(files)})")

    results: list[dict] = []
    all_act, all_kc, all_sgr = [], [], []
    top1_hit_list, top3_hit_list, top5_hit_list = [], [], []
    true_var_counter = Counter()

    for f in tqdm(files, desc="NDSS reasoning"):
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

        best_w = auto_optimize_window(
            df,
            true_vars,
            g=None,
            w_z=w_z,
            w_trend=w_trend,
            w_flat=w_flat,
        )

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
                filtered_low = llm_filter_candidates(client, g, low_candidates, scores)
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
            ranked10 = llm_rerank_top10_hybrid(client, g, top10_graphad, scores, beta=beta)
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
        return

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

    print(f"\n========== NDSS {mode.upper()} FINAL PERFORMANCE ==========")
    print(f"Attacks evaluated : {n}")
    print(f"ACT (Hit@5)      : {ACT_mean:.4f}")
    print(f"K-Concord        : {KC_mean:.4f}")
    print(f"SGR              : {SGR_mean:.4f}")
    print(f"Top-1 Recall     : {top1:.4f}")
    print(f"Top-3 Recall     : {top3:.4f}")
    print(f"Top-5 Recall     : {top5:.4f}")
    print(f"MRR              : {mrr:.4f}")
    print(f"Ranking variance : {ranking_variance:.4f}")

    print("\nTrue-var distribution (top 20):")
    for v, c in true_var_counter.most_common(20):
        print(f"  {v}: {c}")

    # ============================================================
    # ✅ (추가) 상세 결과 ndss_reasoning_{mode}.json 저장
    # ============================================================
    out_dir = os.path.dirname(out_json)
    os.makedirs(out_dir, exist_ok=True)

    detail_path = os.path.join(out_dir, f"ndss_reasoning_{mode}.json")
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n📁 Saved NDSS detailed results → {detail_path}")

    # summary JSON 저장 (mode 반영)
    summary_path = os.path.join(out_dir, f"ndss_performance_summary_{mode}.json")
    save_final_performance(
        results,
        summary_path,
        mode,
        graphad_params={
            "alpha_graph": alpha_graph,
            "w_z": w_z,
            "w_trend": w_trend,
            "w_flat": w_flat,
            "beta": beta,
        },
    )


if __name__ == "__main__":
    import argparse

    default_scen = os.path.join(os.path.dirname(__file__), "..", "data", "ndss_scenarios")
    default_proc = os.path.join(os.path.dirname(__file__), "..", "data", "NDSS_process_edges.csv")
    default_gt = os.path.join(os.path.dirname(__file__), "..", "data", "ndss_attack_scenarios.csv")

    # out_json은 "디렉토리 기준"으로만 사용 (상세 저장은 ndss_reasoning_{mode}.json로 별도 생성)
    default_out = os.path.join(os.path.dirname(__file__), "..", "results", "ndss_reasoning_results.json")

    p = argparse.ArgumentParser()
    p.add_argument("--scen_dir", default=default_scen)
    p.add_argument("--procmap", default=default_proc)
    p.add_argument("--gt_csv", default=default_gt)
    p.add_argument("--out_json", default=default_out)
    p.add_argument("--mode", default="no_llm", choices=["no_llm", "filter", "hybrid"])
    p.add_argument("--alpha_graph", type=float, default=0.3)
    p.add_argument("--w_z", type=float, default=0.6)
    p.add_argument("--w_trend", type=float, default=0.25)
    p.add_argument("--w_flat", type=float, default=0.15)
    p.add_argument("--beta", type=float, default=0.2)

    args = p.parse_args()

    run_ndss_reasoning(
        scen_dir=args.scen_dir,
        procmap_csv=args.procmap,
        gt_csv=args.gt_csv,
        out_json=args.out_json,
        mode=args.mode,
        alpha_graph=args.alpha_graph,
        w_z=args.w_z,
        w_trend=args.w_trend,
        w_flat=args.w_flat,
        beta=args.beta,
    )
