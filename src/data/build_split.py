from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.io import read_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
META_DIR = DATA_DIR / "meta"
CONFIG_DIR = PROJECT_ROOT / "configs"

RANDOM_STATE = 42
DEFAULT_TRANSITION_LEN = 50


@dataclass
class SplitConfig:
    random_state: int = 42
    transition_len: int = 50

    onset_default_faulty_train: int = 20
    onset_default_faulty_test: int = 160
    onset_default_normal: int = 10**9

    # train / val run sampling
    train_normal_n_runs: int = 100
    train_fault_n_runs_per_fault: int = 20

    val_normal_n_runs: int = 20
    val_fault_n_runs_per_fault: int = 5

    # main test run sampling before row compaction
    test_main_normal_n_runs: int = 20
    test_main_fault_n_runs_per_fault: int = 5

    # compact main test size
    test_main_target_rows: int = 4000
    test_main_normal_ratio: float = 0.30
    test_main_transition_ratio: float = 0.20
    test_main_post_ratio: float = 0.50

    # cost test from main test
    test_cost_total: int = 500
    test_cost_normal_ratio: float = 0.30
    test_cost_transition_ratio: float = 0.35
    test_cost_post_ratio: float = 0.35


def load_yaml_config() -> SplitConfig:
    cfg_path = CONFIG_DIR / "split.yaml"
    raw = read_yaml(cfg_path, default={})
    return SplitConfig(**(raw or {}))


def ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)


def _find_first_existing(columns: List[str], candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _infer_run_col(df: pd.DataFrame) -> Optional[str]:
    return _find_first_existing(
        list(df.columns),
        ["simulationRun", "simulation_run", "run", "trial", "run_id"],
    )


def _infer_sample_col(df: pd.DataFrame) -> Optional[str]:
    return _find_first_existing(
        list(df.columns),
        ["sample", "sample_idx", "time", "t", "step", "sampleNo"],
    )


def _infer_fault_col(df: pd.DataFrame) -> Optional[str]:
    return _find_first_existing(
        list(df.columns),
        ["faultNumber", "fault_number", "fault", "fault_id", "class", "label"],
    )


def _infer_feature_cols(df: pd.DataFrame, reserved_cols: List[str]) -> List[str]:
    out = []
    for c in df.columns:
        if c in reserved_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    return out


def load_te_csv(csv_path: Path, domain_tag: str, is_faulty_file: bool) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    run_col = _infer_run_col(df)
    sample_col = _infer_sample_col(df)
    fault_col = _infer_fault_col(df)

    if run_col is None:
        df["run_id"] = np.arange(len(df)) // 500 + 1
    else:
        df = df.rename(columns={run_col: "run_id"})

    if sample_col is None:
        df = df.sort_values(["run_id"]).copy()
        df["sample_idx"] = df.groupby("run_id").cumcount() + 1
    else:
        df = df.rename(columns={sample_col: "sample_idx"})

    if fault_col is None:
        df["fault_id"] = 0 if not is_faulty_file else 1
    else:
        df = df.rename(columns={fault_col: "fault_id"})
        if not is_faulty_file:
            df["fault_id"] = 0

    df["source_file"] = csv_path.name
    df["domain_tag"] = domain_tag
    df["is_faulty_file"] = int(is_faulty_file)

    reserved = ["run_id", "sample_idx", "fault_id", "source_file", "domain_tag", "is_faulty_file"]
    feature_cols = _infer_feature_cols(df, reserved_cols=reserved)
    df = df[reserved + feature_cols].copy()
    return df


def build_raw_merged() -> Tuple[pd.DataFrame, List[str]]:
    sources = [
        ("TEP_FaultFree_Training.csv", "train_domain", False),
        ("TEP_Faulty_Training.csv", "train_domain", True),
        ("TEP_FaultFree_Testing.csv", "test_domain", False),
        ("TEP_Faulty_Testing.csv", "test_domain", True),
    ]

    parts = []
    feature_union = None

    for filename, domain_tag, is_faulty in sources:
        path = RAW_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing required CSV: {path}")

        part = load_te_csv(path, domain_tag=domain_tag, is_faulty_file=is_faulty)

        feat_cols = [
            c for c in part.columns
            if c not in {"run_id", "sample_idx", "fault_id", "source_file", "domain_tag", "is_faulty_file"}
        ]

        if feature_union is None:
            feature_union = feat_cols
        else:
            if feat_cols != feature_union:
                missing = [c for c in feature_union if c not in feat_cols]
                extra = [c for c in feat_cols if c not in feature_union]
                if missing or extra:
                    raise ValueError(
                        f"Feature mismatch detected in {filename}. Missing={missing[:5]}, Extra={extra[:5]}"
                    )

        parts.append(part)

    merged = pd.concat(parts, ignore_index=True)
    assert feature_union is not None
    return merged, feature_union


def build_onset_metadata(df: pd.DataFrame, cfg: SplitConfig) -> pd.DataFrame:
    rows = []

    for source_file in sorted(df["source_file"].unique()):
        is_faulty_source = "Faulty" in source_file

        if not is_faulty_source:
            onset = cfg.onset_default_normal
        else:
            if "Training" in source_file:
                onset = cfg.onset_default_faulty_train
            elif "Testing" in source_file:
                onset = cfg.onset_default_faulty_test
            else:
                onset = cfg.onset_default_faulty_test

        fault_ids = sorted(df.loc[df["source_file"] == source_file, "fault_id"].dropna().unique().tolist())
        for fault_id in fault_ids:
            rows.append(
                {
                    "source_file": source_file,
                    "fault_id": int(fault_id),
                    "default_onset_step": int(onset),
                    "transition_len": int(cfg.transition_len),
                    "post_shift_start": int(onset + cfg.transition_len) if onset < 10**8 else int(10**9),
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(META_DIR / "onset_metadata.csv", index=False)
    return out


def add_phase_and_label_columns(df: pd.DataFrame, onset_meta: pd.DataFrame) -> pd.DataFrame:
    meta = onset_meta.copy()
    df = df.copy()

    df = df.merge(
        meta[["source_file", "fault_id", "default_onset_step", "transition_len"]],
        on=["source_file", "fault_id"],
        how="left",
    )

    df["onset_step"] = df["default_onset_step"].fillna(10**9).astype(int)
    df["transition_len"] = df["transition_len"].fillna(DEFAULT_TRANSITION_LEN).astype(int)

    df["phase"] = "normal"

    is_fault = df["fault_id"] != 0
    df.loc[is_fault, "phase"] = "pre"
    df.loc[
        is_fault
        & (df["sample_idx"] >= df["onset_step"])
        & (df["sample_idx"] < df["onset_step"] + df["transition_len"]),
        "phase",
    ] = "transition"
    df.loc[
        is_fault
        & (df["sample_idx"] >= df["onset_step"] + df["transition_len"]),
        "phase",
    ] = "post_shift"

    # normal/pre=0, transition/post_shift=1
    df["y"] = 0
    df.loc[df["phase"].isin(["transition", "post_shift"]), "y"] = 1

    return df.drop(columns=["default_onset_step"])


def _sample_run_ids(
    run_ids: np.ndarray,
    n_take: int,
    rng: np.random.Generator,
) -> np.ndarray:
    run_ids = np.asarray(run_ids)
    if len(run_ids) == 0:
        return run_ids
    n_take = min(n_take, len(run_ids))
    return rng.choice(run_ids, size=n_take, replace=False)


def sample_runs_for_train_val(
    df_train_domain: pd.DataFrame,
    cfg: SplitConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(cfg.random_state)

    run_df = (
        df_train_domain[["source_file", "fault_id", "run_id"]]
        .drop_duplicates()
        .sort_values(["source_file", "fault_id", "run_id"])
        .reset_index(drop=True)
    )

    train_keys = []
    val_keys = []

    for source_file, g_source in run_df.groupby("source_file", sort=False):
        g_norm = g_source[g_source["fault_id"] == 0]
        norm_run_ids = g_norm["run_id"].drop_duplicates().to_numpy()

        train_norm = _sample_run_ids(norm_run_ids, cfg.train_normal_n_runs, rng)
        remaining_norm = np.array([r for r in norm_run_ids if r not in set(train_norm)])
        val_norm = _sample_run_ids(remaining_norm, cfg.val_normal_n_runs, rng)

        for rid in train_norm:
            train_keys.append((source_file, 0, rid))
        for rid in val_norm:
            val_keys.append((source_file, 0, rid))

        fault_ids = sorted([int(x) for x in g_source["fault_id"].unique().tolist() if int(x) != 0])
        for fid in fault_ids:
            g_fault = g_source[g_source["fault_id"] == fid]
            fault_run_ids = g_fault["run_id"].drop_duplicates().to_numpy()

            train_fault = _sample_run_ids(fault_run_ids, cfg.train_fault_n_runs_per_fault, rng)
            remaining_fault = np.array([r for r in fault_run_ids if r not in set(train_fault)])
            val_fault = _sample_run_ids(remaining_fault, cfg.val_fault_n_runs_per_fault, rng)

            for rid in train_fault:
                train_keys.append((source_file, fid, rid))
            for rid in val_fault:
                val_keys.append((source_file, fid, rid))

    train_key_df = pd.DataFrame(train_keys, columns=["source_file", "fault_id", "run_id"]).drop_duplicates()
    val_key_df = pd.DataFrame(val_keys, columns=["source_file", "fault_id", "run_id"]).drop_duplicates()

    df_train = df_train_domain.merge(
        train_key_df.assign(split_group="train"),
        on=["source_file", "fault_id", "run_id"],
        how="inner",
    )
    df_val = df_train_domain.merge(
        val_key_df.assign(split_group="val"),
        on=["source_file", "fault_id", "run_id"],
        how="inner",
    )
    return df_train, df_val


def sample_runs_for_named_test(
    df_test_domain: pd.DataFrame,
    normal_n_runs: int,
    fault_n_runs_per_fault: int,
    random_state: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    run_df = (
        df_test_domain[["source_file", "fault_id", "run_id"]]
        .drop_duplicates()
        .sort_values(["source_file", "fault_id", "run_id"])
        .reset_index(drop=True)
    )

    test_keys = []

    for source_file, g_source in run_df.groupby("source_file", sort=False):
        g_norm = g_source[g_source["fault_id"] == 0]
        norm_run_ids = g_norm["run_id"].drop_duplicates().to_numpy()
        keep_norm = _sample_run_ids(norm_run_ids, normal_n_runs, rng)
        for rid in keep_norm:
            test_keys.append((source_file, 0, rid))

        fault_ids = sorted([int(x) for x in g_source["fault_id"].unique().tolist() if int(x) != 0])
        for fid in fault_ids:
            g_fault = g_source[g_source["fault_id"] == fid]
            fault_run_ids = g_fault["run_id"].drop_duplicates().to_numpy()
            keep_fault = _sample_run_ids(fault_run_ids, fault_n_runs_per_fault, rng)
            for rid in keep_fault:
                test_keys.append((source_file, fid, rid))

    test_key_df = pd.DataFrame(test_keys, columns=["source_file", "fault_id", "run_id"]).drop_duplicates()
    return df_test_domain.merge(test_key_df, on=["source_file", "fault_id", "run_id"], how="inner")


def build_shift_test(df_test_sampled: pd.DataFrame, split_name: str) -> pd.DataFrame:
    df_out = df_test_sampled[
        (df_test_sampled["phase"] == "normal")
        | (df_test_sampled["phase"].isin(["transition", "post_shift"]))
    ].copy()
    df_out["split_group"] = split_name
    return df_out


def _phase_targets(total: int, normal_ratio: float, transition_ratio: float, post_ratio: float) -> Dict[str, int]:
    n_normal = int(round(total * normal_ratio))
    n_transition = int(round(total * transition_ratio))
    n_post = total - n_normal - n_transition
    return {"normal": n_normal, "transition": n_transition, "post_shift": n_post}


def _sample_contiguous_from_runs(df: pd.DataFrame, target_n: int, rng: np.random.Generator) -> pd.DataFrame:
    if target_n <= 0 or len(df) == 0:
        return df.iloc[0:0].copy()

    groups = []
    for _, g in df.groupby(["source_file", "fault_id", "run_id"], sort=False):
        groups.append(g.sort_values("sample_idx").reset_index(drop=False))

    order = rng.permutation(len(groups))
    remaining = target_n
    parts = []

    for pos, group_idx in enumerate(order):
        if remaining <= 0:
            break

        g = groups[int(group_idx)]
        groups_left = max(len(order) - pos, 1)
        take = int(np.ceil(remaining / groups_left))
        take = max(1, min(take, len(g)))

        start_max = len(g) - take
        start = int(rng.integers(0, start_max + 1)) if start_max > 0 else 0
        picked = g.iloc[start:start + take].copy()
        parts.append(df.loc[picked["index"]])
        remaining -= len(picked)

    if not parts:
        return df.iloc[0:0].copy()
    return pd.concat(parts, axis=0).sort_values(["source_file", "fault_id", "run_id", "sample_idx"])


def compact_main_test(df_main_full: pd.DataFrame, cfg: SplitConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.random_state)
    targets = _phase_targets(
        total=cfg.test_main_target_rows,
        normal_ratio=cfg.test_main_normal_ratio,
        transition_ratio=cfg.test_main_transition_ratio,
        post_ratio=cfg.test_main_post_ratio,
    )

    parts = []

    normal_df = df_main_full[df_main_full["phase"] == "normal"]
    parts.append(_sample_contiguous_from_runs(normal_df, targets["normal"], rng))

    for phase in ["transition", "post_shift"]:
        sub = df_main_full[df_main_full["phase"] == phase].copy()
        target_n = targets[phase]
        if len(sub) == 0 or target_n <= 0:
            continue

        fault_ids = sorted([int(x) for x in sub["fault_id"].unique().tolist() if int(x) != 0])
        picked = []
        if fault_ids:
            base_quota = max(1, target_n // len(fault_ids))
            for fid in fault_ids:
                g = sub[sub["fault_id"] == fid].copy()
                picked.append(_sample_contiguous_from_runs(g, min(base_quota, len(g)), rng))
        picked_df = pd.concat([p for p in picked if len(p) > 0], ignore_index=False) if picked else sub.iloc[0:0].copy()

        remaining_target = target_n - len(picked_df)
        if remaining_target > 0:
            remaining_pool = sub.drop(index=picked_df.index, errors="ignore")
            extra = _sample_contiguous_from_runs(remaining_pool, remaining_target, rng)
            picked_df = pd.concat([picked_df, extra], ignore_index=False)

        parts.append(picked_df)

    if not parts:
        raise ValueError("Unable to build compact main test.")

    out = pd.concat(parts, axis=0).sort_values(["source_file", "fault_id", "run_id", "sample_idx"]).reset_index(drop=True)
    out["split_group"] = "test_main"
    return out


def sample_cost_test_from_main(df_main: pd.DataFrame, cfg: SplitConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.random_state + 7)
    targets = _phase_targets(
        total=cfg.test_cost_total,
        normal_ratio=cfg.test_cost_normal_ratio,
        transition_ratio=cfg.test_cost_transition_ratio,
        post_ratio=cfg.test_cost_post_ratio,
    )

    parts = []
    for phase in ["normal", "transition", "post_shift"]:
        sub = df_main[df_main["phase"] == phase]
        if len(sub) == 0:
            continue
        parts.append(_sample_contiguous_from_runs(sub, min(targets[phase], len(sub)), rng))

    if not parts:
        raise ValueError("No samples available for cost test creation.")

    out = pd.concat(parts, axis=0).sort_values(["source_file", "fault_id", "run_id", "sample_idx"]).reset_index(drop=True)
    out["split_group"] = "test_cost"
    return out


def balance_train_if_needed(df_train: pd.DataFrame) -> pd.DataFrame:
    pos = df_train[df_train["y"] == 1]
    neg = df_train[df_train["y"] == 0]

    if len(pos) == 0 or len(neg) == 0:
        return df_train.copy()

    n = min(len(pos), len(neg))
    pos_s = pos.sample(n=n, random_state=RANDOM_STATE)
    neg_s = neg.sample(n=n, random_state=RANDOM_STATE)
    out = pd.concat([pos_s, neg_s], axis=0).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    return out


def build_run_manifest(*dfs: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for df in dfs:
        cols = ["source_file", "domain_tag", "split_group", "run_id", "fault_id", "y", "onset_step", "phase"]
        sub = (
            df[cols]
            .drop_duplicates()
            .groupby(
                ["source_file", "domain_tag", "split_group", "fault_id", "run_id", "y", "onset_step", "phase"],
                as_index=False,
            )
            .size()
            .rename(columns={"size": "n_rows"})
        )
        parts.append(sub)

    manifest = pd.concat(parts, ignore_index=True)
    return manifest.sort_values(["split_group", "source_file", "fault_id", "run_id", "phase"]).reset_index(drop=True)


def build_fault_distribution(*named_dfs: Tuple[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for split_name, df in named_dfs:
        vc = df["fault_id"].value_counts(dropna=False).sort_index()
        for fault_id, count in vc.items():
            rows.append(
                {
                    "split_group": split_name,
                    "fault_id": int(fault_id),
                    "n_rows": int(count),
                }
            )
    return pd.DataFrame(rows)


def save_split_manifest(meta: Dict) -> None:
    with open(META_DIR / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def print_split_stats(name: str, df: pd.DataFrame) -> None:
    print(f"\n[{name}] rows={len(df):,}")
    if len(df) == 0:
        return
    print(f"  y ratio      : {df['y'].mean():.4f}")
    print(f"  y counts     : {df['y'].value_counts(dropna=False).to_dict()}")
    if "phase" in df.columns:
        print(f"  phase counts : {df['phase'].value_counts(dropna=False).to_dict()}")
    n_units = df[["source_file", "fault_id", "run_id"]].drop_duplicates().shape[0]
    print(f"  run units    : {n_units:,}")


def main() -> None:
    ensure_dirs()
    cfg = load_yaml_config()

    df_all, feature_cols = build_raw_merged()
    onset_meta = build_onset_metadata(df_all, cfg)
    df_all = add_phase_and_label_columns(df_all, onset_meta)

    df_train_domain = df_all[df_all["domain_tag"] == "train_domain"].copy()
    df_test_domain = df_all[df_all["domain_tag"] == "test_domain"].copy()

    # train / val
    df_train_raw, df_val = sample_runs_for_train_val(df_train_domain, cfg)

    # test sampling (run-preserving)
    df_test_sampled = sample_runs_for_named_test(
        df_test_domain=df_test_domain,
        normal_n_runs=cfg.test_main_normal_n_runs,
        fault_n_runs_per_fault=cfg.test_main_fault_n_runs_per_fault,
        random_state=cfg.random_state,
    )

    # TCN용 full-run contiguous test source
    # pre 포함 유지
    df_test_full_tcn = df_test_sampled.copy()
    df_test_full_tcn["split_group"] = "test_full_tcn"

    # 공통 평가용 row-level main/cost subsets
    # 여기서만 pre 제거
    df_test_eval_source = build_shift_test(df_test_sampled, split_name="test_eval_source")
    df_test_main = compact_main_test(df_test_eval_source.copy(), cfg)
    df_test_cost = sample_cost_test_from_main(df_test_main, cfg)

    # 공통 평가용 row-level main/cost subsets
    # 중요: 공통 평가셋만 pre를 제외한다
    df_test_eval_source = build_shift_test(df_test_sampled, split_name="test_eval_source")
    df_test_main = compact_main_test(df_test_eval_source.copy(), cfg)
    df_test_cost = sample_cost_test_from_main(df_test_main, cfg)

    df_train_raw = df_train_raw.copy()
    df_train_raw["split_group"] = "train"

    df_train_balanced = balance_train_if_needed(df_train_raw)
    df_train_balanced["split_group"] = "train"

    # save
    df_train_balanced.to_csv(PROCESSED_DIR / "te_train_rows.csv", index=False)
    df_train_raw.to_csv(PROCESSED_DIR / "te_train_rows_tcn.csv", index=False)

    df_val.to_csv(PROCESSED_DIR / "te_val_rows.csv", index=False)

    # 공통 평가용
    df_test_main.to_csv(PROCESSED_DIR / "te_test_main_rows.csv", index=False)
    df_test_cost.to_csv(PROCESSED_DIR / "te_test_cost_rows.csv", index=False)

    # TCN 문맥용
    df_test_full_tcn.to_csv(PROCESSED_DIR / "te_test_full_rows_tcn.csv", index=False)

    with open(META_DIR / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2, ensure_ascii=False)

    run_manifest = build_run_manifest(
        df_train_raw,
        df_train_balanced,
        df_val,
        df_test_full_tcn,
        df_test_main,
        df_test_cost,
    )
    run_manifest.to_csv(META_DIR / "run_manifest.csv", index=False)

    fault_dist = build_fault_distribution(
        ("train_tcn_raw", df_train_raw),
        ("train_ml_balanced", df_train_balanced),
        ("val", df_val),
        ("test_full_tcn", df_test_full_tcn),
        ("test_main", df_test_main),
        ("test_cost", df_test_cost),
    )
    fault_dist.to_csv(META_DIR / "fault_distribution.csv", index=False)

    split_manifest = {
        "random_state": cfg.random_state,
        "transition_len": cfg.transition_len,
        "onset_default_faulty_train": cfg.onset_default_faulty_train,
        "onset_default_faulty_test": cfg.onset_default_faulty_test,
        "train_normal_n_runs": cfg.train_normal_n_runs,
        "train_fault_n_runs_per_fault": cfg.train_fault_n_runs_per_fault,
        "val_normal_n_runs": cfg.val_normal_n_runs,
        "val_fault_n_runs_per_fault": cfg.val_fault_n_runs_per_fault,
        "test_main_normal_n_runs": cfg.test_main_normal_n_runs,
        "test_main_fault_n_runs_per_fault": cfg.test_main_fault_n_runs_per_fault,
        "test_main_target_rows": cfg.test_main_target_rows,
        "test_cost_total": cfg.test_cost_total,
        "n_train_rows_ml": int(len(df_train_balanced)),
        "n_train_rows_tcn": int(len(df_train_raw)),
        "n_val_rows": int(len(df_val)),
        "n_test_full_rows_tcn": int(len(df_test_full_tcn)),
        "n_test_main_rows": int(len(df_test_main)),
        "n_test_cost_rows": int(len(df_test_cost)),
        "train_ml_positive_ratio": float(df_train_balanced["y"].mean()) if len(df_train_balanced) else None,
        "train_tcn_positive_ratio": float(df_train_raw["y"].mean()) if len(df_train_raw) else None,
        "val_positive_ratio": float(df_val["y"].mean()) if len(df_val) else None,
        "test_full_tcn_positive_ratio": float(df_test_full_tcn["y"].mean()) if len(df_test_full_tcn) else None,
        "test_main_positive_ratio": float(df_test_main["y"].mean()) if len(df_test_main) else None,
        "test_cost_positive_ratio": float(df_test_cost["y"].mean()) if len(df_test_cost) else None,
        "label_definition": {
            "normal": 0,
            "pre": 0,
            "transition": 1,
            "post_shift": 1,
        },
    }
    save_split_manifest(split_manifest)

    print_split_stats("train_tcn_raw", df_train_raw)
    print_split_stats("train_ml_balanced", df_train_balanced)
    print_split_stats("val", df_val)
    print_split_stats("test_full_tcn", df_test_full_tcn)
    print_split_stats("test_main", df_test_main)
    print_split_stats("test_cost", df_test_cost)

    print("\nBuild split completed.")
    print(json.dumps(split_manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main() 
