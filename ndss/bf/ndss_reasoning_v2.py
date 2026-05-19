#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ndss_reasoning.py
-----------------
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

3) (옵션) LLM을 통한 GraphAD Top-10 → Top-5 재정렬
   - USE_LLM=1 & OpenAI 설치 시 활성화
   - LLM off: GraphAD Top-5 그대로 사용
   - LLM on: Top-10 후보를 LLM이 Top-5로 재랭크 → 지표가 달라짐

기본 경로 (acl_graph_reasoning_pipeline_v5 기준):
- 시나리오 CSV   : data/ndss_scenarios/*.csv
- process graph  : data/NDSS_process_edges.csv
- true_var GT    : data/ndss_attack_scenarios.csv
- 결과 JSON      : results/ndss_reasoning_no_llm.json 또는 _llm.json
- 요약 JSON      : results/ndss_performance_summary_no_llm.json 또는 _llm.json
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import os
import csv
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
#    - 센서 / MV / Meta 컬럼 분리
# ================================================================

def normalize_columns_with_mv(df: pd.DataFrame) -> pd.DataFrame:
    """
    NDSS 원본 CSV의 중복 컬럼(A Feed, A Feed.1 등)을
    - 앞쪽(센서/조성 등) 41개는 그대로 센서
    - 뒤쪽(MV/Agitator)은 'XXX (MV)' 이름으로 바꾸어 구분
    """
    cols_raw = list(df.columns)
    cols_new = []

    for i, c in enumerate(cols_raw):
        base = c.split(".")[0].strip()
        # Atk는 그대로 유지
        if base == "Atk":
            cols_new.append("Atk")
            continue

        # 뒤쪽 영역에 등장하는 MV/Agitator는 (MV) 또는 Meta로 변환
        if i >= 41 and base in NDSS_ACT_BASE + NDSS_META:
            if base == "Agitator":
                cols_new.append("Agitator")
            else:
                cols_new.append(f"{base} (MV)")
        else:
            # 앞쪽 센서/조성/기타는 base 그대로
            cols_new.append(base)

    df = df.copy()
    df.columns = cols_new
    return df


def load_and_clean_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = normalize_columns_with_mv(df)

    # 사용할 컬럼만 남기기 (NDSS_ALL_VARS + Atk)
    use_cols = [c for c in df.columns if c in NDSS_ALL_VARS or c == "Atk"]
    df = df[use_cols]
    return df


# ================================================================
# 3. Process Graph 로딩 (NDSS_process_edges.csv)
# ================================================================

def load_process_graph(procmap_csv: str) -> nx.Graph:
    g = nx.Graph()
    if not os.path.exists(procmap_csv):
        print(f"⚠️ Process map not found: {procmap_csv}")
        return g

    df = pd.read_csv(procmap_csv)
    # BOM 제거
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
# 4. GraphAD core: window anomaly score (단일 window)
# ================================================================

def flatline_score(series: pd.Series) -> float:
    """
    DoS / freeze 공격 감지용 단순 flatline score
    std가 매우 작을수록 점수를 높게 부여.
    """
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
    """
    특정 window(win_df)에 대해 변수별 이상 점수 계산:
      score(v) = w_z * mean(|z|) + w_trend * |trend| + w_flat * flatline
    """
    vars_used = list(win_df.columns)

    # z-score (median + MAD 기반)
    z = (win_df - normal_median[vars_used]) / (normal_mad[vars_used] + 1e-6)
    z_abs_mean = z.abs().mean(axis=0)  # per-variable

    # trend score: 윈도우 내 time index 대비 1차 회귀 기울기
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

    # flatline score
    flats = {}
    for v in vars_used:
        flats[v] = flatline_score(win_df[v])
    flat_series = pd.Series(flats)

    # 가중합
    final = w_z * z_abs_mean + w_trend * trend_series + w_flat * flat_series
    return final.sort_values(ascending=False)


# ================================================================
# 5. GraphAD: sliding window + graph smoothing
# ================================================================

def graph_smooth_scores(scores: pd.Series, g: nx.Graph, alpha: float = 0.3) -> pd.Series:
    """
    process graph를 이용한 score smoothing:
      s_smooth(v) = (1-alpha)*s(v) + alpha*mean(s(neighbors(v)))
    """
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
) -> pd.Series | None:
    """
    NDSS GraphAD 고도화 버전:
    - Atk=0 normal, Atk=1 attack
    - attack 구간에서 sliding window
    - 각 window마다 compute_window_scores
    - 변수별로 max-aggregation
    - 마지막에 graph smoothing 적용
    """
    if "Atk" not in df.columns:
        return None

    df0 = df[df["Atk"] == 0]
    df1 = df[df["Atk"] == 1]

    if len(df0) < 20 or len(df1) < 10:
        return None

    # 사용할 변수
    vars_used = [c for c in NDSS_ALL_VARS if c in df.columns]
    if not vars_used:
        return None

    df0 = df0[vars_used]
    df1 = df1[vars_used]

    # Normal 기준 median + MAD
    normal_median = df0.median(axis=0)
    normal_mad = (df0 - normal_median).abs().median(axis=0) + 1e-6

    # sliding window
    n1 = len(df1)
    if n1 < win_len:
        # fallback: 마지막 win_len (또는 전체)
        win_df = df1.tail(min(win_len, n1))
        scores = compute_window_scores(win_df, normal_median, normal_mad)
        if g is not None:
            scores = graph_smooth_scores(scores, g, alpha_graph)
        return scores

    if stride is None:
        stride = max(win_len // 2, 50)

    agg_scores = None
    for start in range(0, n1 - win_len + 1, stride):
        end = start + win_len
        win_df = df1.iloc[start:end]
        s = compute_window_scores(win_df, normal_median, normal_mad)
        if agg_scores is None:
            agg_scores = s
        else:
            # 변수별 max aggregation
            agg_scores = pd.concat([agg_scores, s], axis=1).max(axis=1)

    if agg_scores is None:
        return None

    agg_scores = agg_scores.sort_values(ascending=False)

    # graph smoothing
    if g is not None:
        agg_scores = graph_smooth_scores(agg_scores, g, alpha_graph)

    return agg_scores.sort_values(ascending=False)


# ================================================================
# 6. Window 길이 자동 최적화 (true_var 리스트 대응)
# ================================================================

def auto_optimize_window(
    df: pd.DataFrame,
    true_vars: list[str],
    g: nx.Graph | None = None,
    candidate_windows=(100, 200, 300, 400, 500),
) -> int:
    """
    여러 window 길이 중에서
      "true_var들 중 하나라도 높은 순위에 오도록" 하는 w 선택.
    평가 기준: best_rank = min rank(true_var); score = -best_rank (클수록 좋음)
    """
    best_win = None
    best_score = -1e9

    for w in candidate_windows:
        scores = graphad_scores_sliding(df, win_len=w, g=g, alpha_graph=0.0)  # 최적화 단계에서는 그래프 미사용
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
# 7. Reasoning 품질 지표 (multi-true_var)
# ================================================================

def compute_ACT(topk: list[str], true_vars: list[str]) -> float:
    """Hit@K: topk 리스트 안에 true_var 중 하나라도 포함되면 1"""
    return 1.0 if any(v in topk for v in true_vars) else 0.0


def compute_KConcord(topk: list[str], true_vars: list[str]) -> float:
    """
    true_var 중 가장 높은 순위의 1/(rank+1)
    (없으면 0)
    """
    ranks = []
    for v in true_vars:
        if v in topk:
            ranks.append(topk.index(v))
    if not ranks:
        return 0.0
    best_rank = min(ranks)
    return 1.0 / (best_rank + 1.0)


def compute_SGR(scores: pd.Series, true_vars: list[str]) -> float:
    """
    SGR: max_{v in true_vars} score(v)/max(score)
    (존재하는 true_var 중 최대값 기준)
    """
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
# 8. (옵션) LLM 기반 Top-10 → Top-5 재정렬
# ================================================================

def llm_rerank_top5(
    client,
    graph: nx.Graph,
    candidates: list[str],
    scores: pd.Series,
    max_edges_per_var: int = 5,
) -> list[str]:
    """
    GraphAD Top-10 후보(candidates)를 LLM이 Top-5로 재정렬.
    - 후보 변수 이름과 score, graph 이웃 정보를 prompt에 포함
    - 출력: 후보 중에서 선택된 Top-5 (없는 이름은 무시, 부족하면 원래 순서로 채움)
    """
    if not USE_LLM or not HAS_LLM or client is None:
        return candidates[:5]

    # 이웃 정보 텍스트 구성
    edge_lines = []
    for v in candidates:
        if v in graph:
            neighbors = list(graph.neighbors(v))
            # weight 기준 상위 몇 개만
            neigh_with_w = []
            for n in neighbors:
                w = graph[v][n].get("weight", 0.0)
                neigh_with_w.append((n, w))
            neigh_with_w.sort(key=lambda x: -x[1])
            neigh_with_w = neigh_with_w[:max_edges_per_var]
            for n, w in neigh_with_w:
                edge_lines.append(f"{v} ↔ {n} (corr={w:.2f})")

    cand_lines = [f"- {v}: score={float(scores.get(v, 0.0)):.4f}" for v in candidates]

    prompt = f"""
당신은 Tennessee Eastman 공정(TEP)을 잘 아는 공정 엔지니어입니다.
아래는 이상 탐지 모델(GraphAD)이 선택한 상위 10개 이상 변수 후보입니다.

[후보 변수 + 점수]
{os.linesep.join(cand_lines)}

[공정 상관 관계(그래프 edges 일부)]
{os.linesep.join(edge_lines) if edge_lines else "N/A"}

과제:
1. 위 후보 중에서 실제 공격/이상 원인과 가장 관련이 깊은 5개 변수를 선택하세요.
2. 공정 상관 관계를 고려하여, 원인에 더 가까운 변수들을 상위에 두고 순서를 정하세요.
3. 출력 형식은 다음과 같이, **후보 리스트에 있는 변수 이름만** 콤마로 구분하여 5개만 나열하세요.

예시:
A Feed, D Feed, Purge, A Feed (MV), Purge (MV)

지금 바로, 변수 이름만 5개 출력하세요.
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    text = resp.choices[0].message.content.strip()

    # 파싱: 콤마 또는 줄바꿈 기준 split
    import re
    tokens = re.split(r"[,\\n]", text)
    tokens = [t.strip() for t in tokens if t.strip()]

    # 후보에 포함된 것만, 순서대로 중복 제거
    seen = set()
    reranked = []
    for t in tokens:
        if t in candidates and t not in seen:
            reranked.append(t)
            seen.add(t)
        if len(reranked) >= 5:
            break

    # 5개가 안되면 남은 후보를 원래 순서대로 채우기
    for v in candidates:
        if len(reranked) >= 5:
            break
        if v not in seen:
            reranked.append(v)
            seen.add(v)

    return reranked[:5]


# ================================================================
# 9. NDSS Reasoning 메인 루프
# ================================================================

def re_split_multi(s: str):
    """ "A;B,C" 같은 문자열을 ; 또는 , 기준으로 split """
    import re
    return re.split(r"[;,]", s)


def save_final_performance(results: list[dict], out_path: str, mode: str):
    """
    NDSS 최종 summary를 JSON으로 저장
    """
    import json as _json
    import numpy as _np
    from collections import Counter as _Counter

    if not results:
        return

    ACT = float(_np.mean([r["ACT"] for r in results]))
    KC  = float(_np.mean([r["K-Concord"] for r in results]))
    SGR = float(_np.mean([r["SGR"] for r in results]))

    # Top-k recall
    top1 = float(_np.mean([
        1.0 if any(tv in r["top5_final"][:1] for tv in r["true_vars"]) else 0.0
        for r in results
    ]))
    top3 = float(_np.mean([
        1.0 if any(tv in r["top5_final"][:3] for tv in r["true_vars"]) else 0.0
        for r in results
    ]))
    top5 = float(_np.mean([
        1.0 if any(tv in r["top5_final"][:5] for tv in r["true_vars"]) else 0.0
        for r in results
    ]))

    # true_var 분포
    all_tvs = []
    for r in results:
        for tv in r["true_vars"]:
            all_tvs.append(tv)
    tv_counts = _Counter(all_tvs)

    summary = {
        "mode": mode,
        "attacks_evaluated": len(results),
        "ACT_Hit@5": ACT,
        "K-Concord": KC,
        "SGR": SGR,
        "Top1_recall": top1,
        "Top3_recall": top3,
        "Top5_recall": top5,
        "true_var_distribution": dict(tv_counts),
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        _json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n📁 Saved NDSS final summary → {out_path}")


def run_ndss_reasoning(
    scen_dir: str,
    procmap_csv: str,
    gt_csv: str,
    out_json: str,
):

    # 1) Process graph
    g = load_process_graph(procmap_csv)

    # 2) Ground truth (attack_id, true_var)
    gt_df = pd.read_csv(gt_csv)
    if "attack_id" not in gt_df.columns or "true_var" not in gt_df.columns:
        raise ValueError("gt_csv must have columns: attack_id,true_var,...")
    gt_df = gt_df.set_index("attack_id")

    # 3) NDSS files
    scen_dir = os.path.abspath(scen_dir)
    files = sorted(glob(os.path.join(scen_dir, "*.csv")))
    if not files:
        raise RuntimeError(f"No scenario CSV in {scen_dir}")

    client = OpenAI() if USE_LLM and HAS_LLM else None
    mode = "LLM" if USE_LLM and HAS_LLM else "NO-LLM"

    results = []

    all_act = []
    all_kc = []
    all_sgr = []
    top1_hit = []
    top3_hit = []
    top5_hit = []

    true_var_counter = Counter()

    print(f"\n🚀 NDSS Reasoning 시작 (mode={mode}, scenarios={len(files)})")

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

        # (1) window length 자동 선택 (graph 없이)
        best_w = auto_optimize_window(df, true_vars, g=None)

        # (2) 선택된 window 길이로 GraphAD 최종 score (graph smoothing 포함)
        scores = graphad_scores_sliding(df, win_len=best_w, g=g, alpha_graph=0.3)
        if scores is None or scores.empty:
            continue

        top10_graphad = list(scores.head(10).index)
        graphad_top5 = top10_graphad[:5]

        # (3) LLM re-rank (옵션)
        if USE_LLM and client is not None:
            top5_final = llm_rerank_top5(client, g, top10_graphad, scores)
        else:
            top5_final = graphad_top5

        top3_final = top5_final[:3]
        top1_final = top5_final[:1]

        # (4) 지표 계산 (최종 Top-5 기준)
        act = compute_ACT(top5_final, true_vars)
        kc = compute_KConcord(top5_final, true_vars)
        sgr = compute_SGR(scores, true_vars)

        all_act.append(act)
        all_kc.append(kc)
        all_sgr.append(sgr)

        top1_hit.append(1.0 if any(v in top1_final for v in true_vars) else 0.0)
        top3_hit.append(1.0 if any(v in top3_final for v in true_vars) else 0.0)
        top5_hit.append(1.0 if any(v in top5_final for v in true_vars) else 0.0)

        results.append({
            "attack_id": attack_id,
            "true_vars": true_vars,
            "window": best_w,
            "top10_graphad": top10_graphad,
            "top5_final": top5_final,
            "ACT": act,
            "K-Concord": kc,
            "SGR": sgr,
        })

    # 결과 저장
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    n = len(results)
    if n == 0:
        print("⚠️ No NDSS scenarios evaluated.")
        return

    ACT_mean = float(np.mean(all_act))
    KC_mean = float(np.mean(all_kc))
    SGR_mean = float(np.mean(all_sgr))
    top1 = float(np.mean(top1_hit))
    top3 = float(np.mean(top3_hit))
    top5 = float(np.mean(top5_hit))

    print("\n========== NDSS {} FINAL PERFORMANCE ==========".format(
        "LLM" if USE_LLM else "NO-LLM"))
    print(f"Attacks evaluated : {n}")
    print(f"ACT (Hit@5)      : {ACT_mean:.4f}")
    print(f"K-Concord        : {KC_mean:.4f}")
    print(f"SGR              : {SGR_mean:.4f}")
    print(f"Top-1 Recall     : {top1:.4f}")
    print(f"Top-3 Recall     : {top3:.4f}")
    print(f"Top-5 Recall     : {top5:.4f}")
    print("\nTrue-var distribution (top 20):")
    for v, c in true_var_counter.most_common(20):
        print(f"  {v}: {c}")

    # summary JSON 저장
    mode_key = "llm" if USE_LLM and HAS_LLM else "no_llm"
    summary_path = os.path.join(
        os.path.dirname(out_json),
        f"ndss_performance_summary_{mode_key}.json"
    )
    save_final_performance(results, summary_path, mode_key)


# ================================================================
# CLI
# ================================================================

if __name__ == "__main__":
    import argparse

    root = os.path.dirname(os.path.dirname(__file__))
    default_scen = os.path.join(root, "data", "ndss_scenarios")
    default_proc = os.path.join(root, "data", "NDSS_process_edges.csv")
    default_gt = os.path.join(root, "data", "ndss_attack_scenarios.csv")

    mode = "llm" if USE_LLM and HAS_LLM else "no_llm"
    default_out = os.path.join(root, "results", f"ndss_reasoning_{mode}.json")

    p = argparse.ArgumentParser()
    p.add_argument("--scen_dir", default=default_scen)
    p.add_argument("--procmap", default=default_proc)
    p.add_argument("--gt_csv", default=default_gt)
    p.add_argument("--out_json", default=default_out)

    args = p.parse_args()

    run_ndss_reasoning(
        scen_dir=args.scen_dir,
        procmap_csv=args.procmap,
        gt_csv=args.gt_csv,
        out_json=args.out_json,
    )
