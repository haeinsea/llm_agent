#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_ndss_process_map.py  (MV 포함 상관 기반 그래프)
---------------------------------------------------
- NDSS 286개 시나리오 전체를 사용해 공정 상관 그래프를 생성.
- 모든 시나리오의 Atk=0 (정상 구간)만 모아서 상관계수 계산 → 안정적인 공정 관계.
- 출력: src,tgt,corr  형식 CSV (NDSS_process_edges.csv)
"""

import os
import pandas as pd
import numpy as np
from glob import glob

# 41개 센서
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

# 11개 MV + Meta (Agitator)
NDSS_ACT_BASE = [
    "D feed","E Feed","A Feed","A and C Feed","Recycle","Purge",
    "Separator","Stripper","Steam","Reactor Coolant","Condenser Coolant",
]
NDSS_META = ["Agitator"]

# 실제 DataFrame에서 사용할 이름 (MV에 suffix 부여)
NDSS_ACT = [
    "D feed (MV)","E Feed (MV)","A Feed (MV)","A and C Feed (MV)","Recycle (MV)",
    "Purge (MV)","Separator (MV)","Stripper (MV)","Steam (MV)",
    "Reactor Coolant (MV)","Condenser Coolant (MV)",
]

NDSS_ALL_VARS = NDSS_MEAS + NDSS_ACT + NDSS_META


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
            # MV/Meta 영역
            if base == "Agitator":
                cols_new.append("Agitator")
            else:
                cols_new.append(f"{base} (MV)")
        else:
            cols_new.append(base)

    df = df.copy()
    df.columns = cols_new
    return df


def make_process_map(scen_dir: str, out_csv: str, corr_th: float = 0.6):
    scen_dir = os.path.abspath(scen_dir)
    files = sorted(glob(os.path.join(scen_dir, "*.csv")))
    if not files:
        raise RuntimeError(f"No CSV files in {scen_dir}")

    print(f"📂 NDSS Scenario Files Loaded: {len(files)}")
    print(f"📌 Building correlation graph from normal (Atk=0) segments")

    all_normal = []

    for f in files:
        df = pd.read_csv(f)
        df = normalize_columns_with_mv(df)

        if "Atk" not in df.columns:
            continue

        df0 = df[df["Atk"] == 0]
        if len(df0) == 0:
            continue

        # 관계 학습용 변수만 선택
        use_cols = [c for c in NDSS_ALL_VARS if c in df0.columns]
        if not use_cols:
            continue

        all_normal.append(df0[use_cols])

    if not all_normal:
        raise RuntimeError("No normal segments collected for correlation computation")

    bigdf = pd.concat(all_normal, axis=0, ignore_index=True)
    print(f"   → Normal samples used: {len(bigdf)} rows")

    # 상관계수 계산
    corr = bigdf.corr()

    rows = []
    for i in corr.index:
        for j in corr.columns:
            if i == j:
                continue
            val = corr.loc[i, j]
            if pd.isna(val):
                continue
            if abs(val) >= corr_th:
                rows.append({"src": i, "tgt": j, "corr": float(val)})

    df_edges = pd.DataFrame(rows).drop_duplicates()
    df_edges.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"🎉 NDSS process graph 생성 완료: {out_csv}")
    print(f"   → #edges: {len(df_edges)}  (|corr| >= {corr_th})")


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(__file__))
    default_scen = os.path.join(root, "data", "ndss_scenarios")
    default_out = os.path.join(root, "data", "NDSS_process_edges.csv")

    make_process_map(default_scen, default_out)
