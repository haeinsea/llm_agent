#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate_ndss_true_vars.py  (MV 복원 + true_var 확장 버전)
---------------------------------------------------------
- NDSS 조작 로그 파일 이름(TEP_test_xxx_yyy_sNN 혹은 ..._aMM)을 이용해
  각 시나리오별 ground-truth 조작 변수 목록(true_var)을 생성한다.
- 정책:
    * 센서 조작(sNN):   [해당 센서]  (+ 매칭되는 MV가 있으면 [센서, MV])
    * 액추에이터(aMM): [해당 MV]     (+ 매칭되는 센서가 있으면 [MV, 센서])
  → 대부분의 feed / flow 계열 공격은 true_var 2개(센서+밸브)로 설정됨.
    (조성 분석 등 MV가 없는 센서는 1개일 수 있음)
- 출력 형식 (CSV):
    attack_id,true_var,attack_type,description
    TEP_test_cons_m2s_a3,"A Feed (MV);A Feed",Integrity,
"""

import os
import re
import argparse
import pandas as pd
from glob import glob

# ---------------------------
# NDSS 변수 스키마
# ---------------------------

# 41개 센서(TE XMEAS(1)~XMEAS(41)와 동일 순서)
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

# 11개 액추에이터 + 1개 메타(Agitator)
# → CSV 로딩 시에는 "XXX (MV)" 형태로 rename 될 것임
NDSS_ACT_BASE = [
    "D feed","E Feed","A Feed","A and C Feed","Recycle","Purge",
    "Separator","Stripper","Steam","Reactor Coolant","Condenser Coolant",
]
NDSS_META = ["Agitator"]

# 액추에이터 id → 이름 (MV 명은 "(MV)" 포함 버전으로 만듦)
ACT_ID_TO_NAME = {
    1: "D feed (MV)",
    2: "E Feed (MV)",
    3: "A Feed (MV)",
    4: "A and C Feed (MV)",
    5: "Recycle (MV)",
    6: "Purge (MV)",
    7: "Separator (MV)",
    8: "Stripper (MV)",
    9: "Steam (MV)",
    10: "Reactor Coolant (MV)",
    11: "Condenser Coolant (MV)",
}

# 액추에이터 ↔ 센서 매핑 (공정 상 주요 대응 변수)
SENSOR_FOR_ACT = {
    "D feed (MV)": "D Feed",
    "E Feed (MV)": "E Feed",
    "A Feed (MV)": "A Feed",
    "A and C Feed (MV)": "A and C Feed",
    "Recycle (MV)": "Recycle Flow",
    "Purge (MV)": "Purge Rate",
    "Separator (MV)": "Product Sep Level",
    "Stripper (MV)": "Stripper Level",
    "Steam (MV)": "Stripper Steam Flow",
    "Reactor Coolant (MV)": "Reactor Coolant Temp",
    "Condenser Coolant (MV)": "Separator Coolant Temp",
}
ACT_FOR_SENSOR = {v: k for k, v in SENSOR_FOR_ACT.items()}


def infer_attack_type_from_name(fname: str) -> str:
    low = fname.lower()
    if "dos" in low:
        return "DoS"
    if "replay" in low:
        return "Replay"
    # cons / csum / line 등은 모두 Integrity 공격 계열
    if "cons" in low or "csum" in low or "line" in low:
        return "Integrity"
    return "Unknown"


def extract_true_vars_from_attack_id(attack_id: str):
    """
    파일명에서 마지막 토큰 (sNN or aMM)을 파싱해서
    true_vars(list[str]) 반환.
    """
    base = os.path.basename(attack_id)
    tok = base.split("_")[-1]  # 예: 's3', 'a10'
    m = re.match(r"([sa])(\d+)", tok)
    if not m:
        return []

    kind, num_str = m.group(1), m.group(2)
    idx = int(num_str)

    true_vars = []

    if kind == "s":  # sensor 조작
        # sNN → NDSS_MEAS[NN-1]
        if 1 <= idx <= len(NDSS_MEAS):
            sensor = NDSS_MEAS[idx - 1]
            true_vars.append(sensor)
            # 대응하는 MV 있으면 추가
            mv = ACT_FOR_SENSOR.get(sensor)
            if mv is not None:
                true_vars.append(mv)

    elif kind == "a":  # actuator 조작
        mv = ACT_ID_TO_NAME.get(idx)
        if mv is not None:
            true_vars.append(mv)
            sensor = SENSOR_FOR_ACT.get(mv)
            if sensor is not None:
                true_vars.append(sensor)

    # 중복 제거
    true_vars = list(dict.fromkeys(true_vars))
    return true_vars


def main(scen_dir: str, out_csv: str):
    scen_dir = os.path.abspath(scen_dir)
    files = sorted(glob(os.path.join(scen_dir, "*.csv")))

    rows = []

    for f in files:
        attack_id = os.path.basename(f).replace(".csv", "")
        attack_type = infer_attack_type_from_name(attack_id)
        true_vars = extract_true_vars_from_attack_id(attack_id)

        rows.append({
            "attack_id": attack_id,
            "true_var": ";".join(true_vars),
            "attack_type": attack_type,
            "description": "",
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"✅ Generated NDSS attack scenarios with MV 복원: {out_csv}")
    print(f"   - #scenarios: {len(df_out)}")
    print(f"   - true_var length: min={df_out['true_var'].apply(lambda s: len([x for x in s.split(';') if x])).min()} "
          f"max={df_out['true_var'].apply(lambda s: len([x for x in s.split(';') if x])).max()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    root = os.path.dirname(os.path.dirname(__file__))
    default_scen = os.path.join(root, "data", "ndss_scenarios")
    default_out = os.path.join(root, "data", "ndss_attack_scenarios.csv")

    parser.add_argument("--scen_dir", default=default_scen,
                        help=f"NDSS scenario CSV directory (default: {default_scen})")
    parser.add_argument("--out_csv", default=default_out,
                        help=f"output CSV path (default: {default_out})")

    args = parser.parse_args()
    main(args.scen_dir, args.out_csv)
