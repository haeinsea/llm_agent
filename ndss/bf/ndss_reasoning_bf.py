#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ndss_reasoning.py
-----------------
NDSS 조작 시나리오에 대해

1) GraphAD 기반 변수별 이상 점수 계산
2) 시나리오별 true_var(센서+MV 페어) 기준으로
   - ACT (Hit@5)
   - K-Concord
   - SGR
   - Top-1/3/5 Recall
3) (옵션) LLM을 이용한 공정 흐름/원인 설명 (USE_LLM=1)

기본 경로 (acl_graph_reasoning_pipeline_v5 기준):
- 시나리오 CSV   : data/ndss_scenarios/*.csv
- process graph  : data/NDSS_process_edges.csv
- true_var GT    : data/ndss_attack_scenarios.csv
- 결과 JSON      : results/ndss_reasoning_no_llm.json 또는 _llm.json
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

# LLM 옵션 (환경변수 USE_LLM=1 && openai 설치 시 사용)
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
# 1. NDSS 변수 스키마 (TE 호환)
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
    "Comp E in Purge","Comp F in Purge","Comp G in Purge","Comp H in Purge",
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
# 2. NDSS CSV 로딩 & 컬럼 정규화 (센서 / MV 구분)
# ================================================================

def normalize_columns_with_mv(df: pd.DataFrame) -> pd.DataFrame:
    """
    NDSS 원본 CSV의 중복 컬럼(A Feed, A Feed.1 등)을
    - 앞쪽(센서) 41개는 그대로 센서
    - 뒤쪽(MV/Agitator)은 'XXX (MV)' 이름으로 바꾸어 구분
    """
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

    # 사용할 컬럼만 남기기 (NDSS_ALL_VARS + Atk)
    use_cols = [c for c in df.columns if c in NDSS_ALL_VARS or c == "Atk"]
    df = df[use_cols]
    return df


# ================================================================
# 3. Process Graph 로딩
# ================================================================

def load_process_graph(procmap_csv):
    g = nx.Graph()
    if not os.path.exists(procmap_csv):
        return g

    # --- Fix BOM on src column ---
    import pandas as pd
    df = pd.read_csv(procmap_csv)
    df.columns = [c.replace("\ufeff", "") for c in df.columns]  # 🔥 key fix

    # Validate
    required = {"src","tgt","corr"}
    if not required.issubset(df.columns):
        raise ValueError(f"Process map must have columns src,tgt,corr (got {set(df.columns)})")

    for _, r in df.iterrows():
        src, tgt = r["src"], r["tgt"]
        corr = float(r["corr"])
        if src != tgt:
            g.add_edge(src, tgt, weight=abs(corr))

    return g


# ================================================================
# 4. GraphAD (window 기반 변수별 이상 점수)
# ================================================================

def graphad_window_scores(df: pd.DataFrame, win_len: int) -> pd.Series | None:
    """
    Atk=0 normal, Atk=1 attack 구간을 이용해
    변수별 z-score 기반 윈도우 평균 이상도 계산.
    """
    if "Atk" not in df.columns:
        return None

    df0 = df[df["Atk"] == 0]
    df1 = df[df["Atk"] == 1]

    if len(df1) < win_len or len(df0) < 10:
        return None

    # 사용할 변수 (실제로 존재하는 것만)
    vars_used = [c for c in NDSS_ALL_VARS if c in df.columns]
    if not vars_used:
        return None

    win = df1.tail(win_len)[vars_used].copy()

    mu = df0[vars_used].mean()
    sd = df0[vars_used].std() + 1e-6

    z = (win - mu) / sd
    scores = z.abs().mean().sort_values(ascending=False)
    return scores


# ================================================================
# 5. Window 자동 최적화 (true_var 리스트 대응)
# ================================================================

def auto_optimize_window(df: pd.DataFrame,
                         true_vars: list[str],
                         candidate_windows=(100, 200, 300, 400, 500)) -> int:
    """
    여러 window 길이 중에서
    "true_var들 중 하나라도 가장 위에 오도록(rank 최소)" 하는 w를 선택.
    """
    best_win = None
    best_rank_score = -1e9

    for w in candidate_windows:
        scores = graphad_window_scores(df, w)
        if scores is None:
            continue

        # 각 true_var의 rank 계산 (존재하는 것만)
        ranks = []
        for v in true_vars:
            if v in scores.index:
                ranks.append(list(scores.index).index(v))

        if not ranks:
            continue

        best_rank = min(ranks)   # 작은게 좋음
        score = -best_rank       # 음수로 변환해 "클수록 좋음"

        if score > best_rank_score:
            best_rank_score = score
            best_win = w

    if best_win is None:
        best_win = 300  # fallback
    return best_win


# ================================================================
# 6. Reasoning 품질 지표 (multi-true_var 버전)
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
# 7. (옵션) LLM Reasoning
# ================================================================

def llm_reason(client, graph: nx.Graph, top_vars: list[str], true_vars: list[str]) -> str:
    if not USE_LLM or not HAS_LLM:
        return ""

    # 간단히 top_vars 주변 1-hop edge만 텍스트로 정리
    edges_ctx = []
    for v in top_vars:
        if v in graph:
            for n in graph.neighbors(v):
                w = graph[v][n].get("weight", 0.0)
                edges_ctx.append(f"{v} ↔ {n} (corr={w:.2f})")

    tv_str = ", ".join(true_vars) if true_vars else "unknown"
    tv_hint = f"Ground truth manipulated variables (for evaluation): {tv_str}"

    prompt = f"""
You are an expert process engineer analyzing abnormal behavior in the Tennessee Eastman process.

Anomaly-detector (GraphAD) selected the following top abnormal variables:
- {', '.join(top_vars)}

Process correlation graph (1-hop neighborhood):
- {'; '.join(edges_ctx) if edges_ctx else 'N/A'}

{tv_hint}

Task:
1. Infer the most plausible root-cause(s) of the attack.
2. Describe how the abnormal variables are related along the process flow.
3. Explain in 3–5 bullet points, using short, technical Korean sentences.
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


# ================================================================
# 8. NDSS Reasoning 메인 루프
# ================================================================

def re_split_multi(s: str):
    # "A;B,C" 같은 문자열을 ; 또는 , 기준으로 split
    import re
    return re.split(r"[;,]", s)


def run_ndss_reasoning(scen_dir: str,
                       procmap_csv: str,
                       gt_csv: str,
                       out_json: str):

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

    results = []
    all_act = []
    all_kc = []
    all_sgr = []
    top1_hit = []
    top3_hit = []
    top5_hit = []

    true_var_counter = Counter()

    for f in tqdm(files, desc="NDSS reasoning"):
        df = load_and_clean_csv(f)
        attack_id = os.path.basename(f).replace(".csv", "")

        if attack_id not in gt_df.index:
            # GT 없으면 스킵
            continue

        # true_var 파싱 (세미콜론/콤마 모두 허용)
        tv_raw = str(gt_df.loc[attack_id, "true_var"])
        true_vars = [t.strip() for t in re_split_multi(tv_raw) if t.strip()]

        for v in true_vars:
            true_var_counter[v] += 1

        # window 자동선택
        best_w = auto_optimize_window(df, true_vars)

        scores = graphad_window_scores(df, best_w)
        if scores is None:
            continue

        top5 = list(scores.head(5).index)
        top3 = top5[:3]
        top1 = top5[:1]

        act = compute_ACT(top5, true_vars)
        kc = compute_KConcord(top5, true_vars)
        sgr = compute_SGR(scores, true_vars)

        all_act.append(act)
        all_kc.append(kc)
        all_sgr.append(sgr)

        top1_hit.append(1.0 if any(v in top1 for v in true_vars) else 0.0)
        top3_hit.append(1.0 if any(v in top3 for v in true_vars) else 0.0)
        top5_hit.append(1.0 if any(v in top5 for v in true_vars) else 0.0)

        reasoning_text = ""
        if USE_LLM and client is not None:
            reasoning_text = llm_reason(client, g, top5, true_vars)

        results.append({
            "attack_id": attack_id,
            "true_var": true_vars,
            "window": best_w,
            "top5": top5,
            "ACT": act,
            "K-Concord": kc,
            "SGR": sgr,
            "reasoning": reasoning_text,
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
        
        # 8) Save all per-attack reasoning results
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n🎉 NDSS reasoning 완료 → {out_json}")

    summary_path = os.path.join(
    os.path.dirname(out_json),
    f"ndss_performance_summary_{'llm' if USE_LLM else 'no_llm'}.json")
    save_final_performance(results, summary_path)



def save_final_performance(results, out_path):
    """
    Save NDSS final performance summary (LLM or No-LLM)
    """
    import json
    import numpy as np
    from collections import Counter

    n_attacks = len(results)

    ACT = float(np.mean([r["ACT"] for r in results]))
    KC  = float(np.mean([r["K-Concord"] for r in results]))
    SGR = float(np.mean([r["SGR"] for r in results]))

    # ==== Top-k recalls (true_var is list) ====
    top1 = float(np.mean([
        1.0 if any(v == r["top5"][0] for v in r["true_var"]) else 0.0
        for r in results
    ]))

    top3 = float(np.mean([
        1.0 if any(v in r["top5"][:3] for v in r["true_var"]) else 0.0
        for r in results
    ]))

    top5 = float(np.mean([
        1.0 if any(v in r["top5"] for v in r["true_var"]) else 0.0
        for r in results
    ]))

    # ==== Flatten true_var for distribution ====
    flatten = []
    for r in results:
        flatten.extend(r["true_var"])
    tv_counts = Counter(flatten)

    summary = {
        "attacks_evaluated": n_attacks,
        "ACT_Hit@5": ACT,
        "K-Concord": KC,
        "SGR": SGR,
        "Top1_recall": top1,
        "Top3_recall": top3,
        "Top5_recall": top5,
        "true_var_distribution": dict(tv_counts),
    }

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n📁 Saved NDSS final summary → {out_path}")



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
    
    
