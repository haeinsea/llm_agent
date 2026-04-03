from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.io import read_json


def load_base_runtime_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = read_json(path)
    return payload.get("records", {}) if isinstance(payload, dict) else {}


def get_base_runtime_stat(
    runtime_summary: dict[str, Any],
    split: str,
    component: str,
    field: str,
    default: float = 0.0,
) -> float:
    split_block = runtime_summary.get(split, {})
    component_block = split_block.get(component, {})
    try:
        return float(component_block.get(field, default))
    except Exception:
        return float(default)
