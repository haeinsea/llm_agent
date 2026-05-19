#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ndss_p0_2_report_actk.py
-----------------------
P0-2 통계 리포트 자동 생성 (Recall 제거, ACT@K만 사용):
- sensor vs actuator 분해표(1개)
- mean±std (ACT@1, ACT@3, ACT@5)
- paired t-test + Cohen's dz (paired: dz)
- CSV 저장

입력:
  results/ndss_reasoning_no_llm.json
  results/ndss_reasoning_filter.json

출력:
  results/p0_2_gt_decomposition.csv
  results/p0_2_actk_meanstd_table.csv
  results/p0_2_actk_paired_stats.csv
"""

import os
import json
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd

# scipy optional
try:
    from scipy.stats import ttest_rel
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


# ===== NDSS 센서 리스트: reasoning 코드와 동일 =====
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

def is_sensor(v: str) -> bool:
    return v in NDSS_MEAS

def is_actuator(v: str) -> bool:
    return v.endswith(" (MV)")

def gt_group(true_vars: List[str]) -> str:
    """GT true_vars 기준: sensor_only vs sensor+actuator(주로) 분해"""
    has_s = any(is_sensor(v) for v in true_vars)
    has_a = any(is_actuator(v) for v in true_vars)
    if has_s and has_a:
        return "sensor+actuator"
    if has_s:
        return "sensor_only"
    if has_a:
        return "actuator_only"
    return "other"

def mean_std(x: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float(np.mean(x)), float("nan")
    return float(np.mean(x)), float(np.std(x, ddof=1))

def cohens_dz_paired(x_new: np.ndarray, x_base: np.ndarray) -> float:
    """paired Cohen's d (dz) = mean(diff) / std(diff)"""
    diff = np.asarray(x_new, dtype=float) - np.asarray(x_base, dtype=float)
    if len(diff) < 2:
        return float("nan")
    sd = float(np.std(diff, ddof=1))
    if sd == 0.0:
        return 0.0
    return float(np.mean(diff) / sd)

def paired_t_test(x_new: np.ndarray, x_base: np.ndarray) -> Tuple[float, float]:
    """scipy 없으면 t만 계산하고 p는 nan"""
    x_new = np.asarray(x_new, dtype=float)
    x_base = np.asarray(x_base, dtype=float)
    diff = x_new - x_base
    n = len(diff)
    if n < 2:
        return float("nan"), float("nan")
    sd = float(np.std(diff, ddof=1))
    if sd == 0.0:
        # 모든 paired diff가 0이면 t-test 의미 없음(분산=0)
        return float("nan"), float("nan")
    t_stat = float(np.mean(diff) / (sd / np.sqrt(n)))
    if HAS_SCIPY:
        res = ttest_rel(x_new, x_base, nan_policy="omit")
        return float(res.statistic), float(res.pvalue)
    else:
        return t_stat, float("nan")

def load_results(path: str) -> pd.DataFrame:
    """
    reasoning 결과 json에서
    - attack_id
    - true_vars
    - top1_hit, top3_hit, top5_hit
    를 읽어 ACT@1/3/5로 사용
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    required = {"attack_id","true_vars","top1_hit","top3_hit","top5_hit"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    df["gt_group"] = df["true_vars"].apply(gt_group)

    # ACT@K (=Hit@K): 각 시나리오에서 0/1
    df["ACT@1"] = df["top1_hit"].astype(float)
    df["ACT@3"] = df["top3_hit"].astype(float)
    df["ACT@5"] = df["top5_hit"].astype(float)

    return df[["attack_id","gt_group","ACT@1","ACT@3","ACT@5"]].copy()

def build_tables(df_no: pd.DataFrame, df_fi: pd.DataFrame):
    # attack_id 기준 페어링
    m = df_no.merge(df_fi, on="attack_id", suffixes=("_no_llm","_filter"))

    # (1) GT 분해표
    decomp = m["gt_group_no_llm"].value_counts().rename_axis("gt_group").reset_index(name="N")
    mismatch = (m["gt_group_no_llm"] != m["gt_group_filter"]).sum()
    if mismatch > 0:
        print(f"⚠️ Warning: gt_group mismatch across modes for {mismatch} rows (should be 0).")

    # (2) mean±std 테이블 (ACT@1/3/5)
    groups = ["sensor_only","sensor+actuator","actuator_only","other","ALL"]
    rows = []
    for g in groups:
        mg = m if g == "ALL" else m[m["gt_group_no_llm"] == g]
        if len(mg) == 0:
            continue

        row = {"group": g, "N": len(mg)}
        for metric in ["ACT@1","ACT@3","ACT@5"]:
            mu_no, sd_no = mean_std(mg[f"{metric}_no_llm"])
            mu_fi, sd_fi = mean_std(mg[f"{metric}_filter"])
            row[f"{metric} no_llm (mean±std)"] = f"{mu_no:.3f} ± {sd_no:.3f}"
            row[f"{metric} filter (mean±std)"] = f"{mu_fi:.3f} ± {sd_fi:.3f}"
        rows.append(row)

    meanstd_table = pd.DataFrame(rows)

    # (3) paired t-test + Cohen's dz
    stats_rows = []
    for g in groups:
        mg = m if g == "ALL" else m[m["gt_group_no_llm"] == g]
        if len(mg) < 2:
            continue

        for metric in ["ACT@1","ACT@3","ACT@5"]:
            x_no = mg[f"{metric}_no_llm"].to_numpy(dtype=float)
            x_fi = mg[f"{metric}_filter"].to_numpy(dtype=float)

            diff_mean = float(np.mean(x_fi - x_no))
            t_stat, p_val = paired_t_test(x_fi, x_no)
            dz = cohens_dz_paired(x_fi, x_no)

            stats_rows.append({
                "group": g,
                "metric": metric,
                "N": len(mg),
                "mean(diff=filter-no_llm)": f"{diff_mean:.3f}",
                "t": (f"{t_stat:.3f}" if np.isfinite(t_stat) else "nan"),
                "p": (f"{p_val:.3g}" if (HAS_SCIPY and np.isfinite(p_val)) else "nan"),
                "Cohen_dz": (f"{dz:.3f}" if np.isfinite(dz) else "nan"),
            })

    paired_stats = pd.DataFrame(stats_rows)

    return decomp, meanstd_table, paired_stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_llm", default="results/ndss_reasoning_no_llm.json")
    ap.add_argument("--filter", default="results/ndss_reasoning_filter.json")
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    df_no = load_results(args.no_llm)
    df_fi = load_results(args.filter)

    decomp, meanstd_table, paired_stats = build_tables(df_no, df_fi)

    os.makedirs(args.out_dir, exist_ok=True)
    out_decomp = os.path.join(args.out_dir, "p0_2_gt_decomposition.csv")
    out_meanstd = os.path.join(args.out_dir, "p0_2_actk_meanstd_table.csv")
    out_stats = os.path.join(args.out_dir, "p0_2_actk_paired_stats.csv")

    decomp.to_csv(out_decomp, index=False)
    meanstd_table.to_csv(out_meanstd, index=False)
    paired_stats.to_csv(out_stats, index=False)

    print("\n=== [P0-2] GT Decomposition (sensor vs actuator) ===")
    print(decomp.to_string(index=False))

    print("\n=== [P0-2] Mean±Std Table (ACT@1, ACT@3, ACT@5) ===")
    print(meanstd_table.to_string(index=False))

    print("\n=== [P0-2] Paired t-test + Cohen's dz (filter vs no_llm) ===")
    if not HAS_SCIPY:
        print("⚠️ scipy가 없어 p-value는 nan일 수 있어요. (pip install scipy 권장)")
    print(paired_stats.to_string(index=False))

    print("\nSaved:")
    print(" -", out_decomp)
    print(" -", out_meanstd)
    print(" -", out_stats)

if __name__ == "__main__":
    main()
