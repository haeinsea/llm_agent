from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from src.utils.io import read_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
EXPERIMENT_CONFIG_PATH = CONFIG_DIR / "experiment.yaml"
DEFAULT_SEEDS = list(range(10))
DEFAULT_REPRESENTATIVE_SEED = 0
DEFAULT_LLM_SEED_POLICY = {
    "main": "all",
    "cost": "representative",
}


@lru_cache(maxsize=1)
def experiment_cfg() -> dict:
    return read_yaml(EXPERIMENT_CONFIG_PATH, default={}) or {}


def get_seed_list() -> list[int]:
    cfg = experiment_cfg()
    raw_seeds = cfg.get("seeds", DEFAULT_SEEDS)
    if isinstance(raw_seeds, int):
        seeds = list(range(int(raw_seeds)))
    else:
        seeds = [int(seed) for seed in raw_seeds]
    if not seeds:
        raise ValueError("Experiment seed list cannot be empty.")
    return seeds


def get_seed_count() -> int:
    return len(get_seed_list())


def get_representative_seed() -> int:
    cfg = experiment_cfg()
    seed = int(cfg.get("representative_seed", DEFAULT_REPRESENTATIVE_SEED))
    if seed not in get_seed_list():
        raise ValueError(f"Representative seed {seed} must be included in the experiment seed list.")
    return seed


def get_llm_seed_policy(dataset_name: str) -> str:
    cfg = experiment_cfg()
    default_policy = DEFAULT_LLM_SEED_POLICY.get(dataset_name, "representative")
    policy = str(cfg.get(f"{dataset_name}_llm_seed_policy", default_policy)).strip().lower()
    if policy not in {"representative", "all"}:
        raise ValueError(f"Unsupported LLM seed policy '{policy}' for dataset '{dataset_name}'.")
    return policy


def ensemble_component_label(component_name: str) -> str:
    return f"{component_name} Ensemble ({get_seed_count()} seeds)"
