from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _parse_yaml_scalar(raw: str) -> Any:
    value = raw.strip()
    if value == "":
        return ""

    lowered = value.lower()
    if lowered in {"null", "none", "~"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def _parse_simple_yaml(lines: list[str], start_idx: int = 0, base_indent: int = 0) -> tuple[dict, int]:
    data: dict[str, Any] = {}
    idx = start_idx

    while idx < len(lines):
        raw = lines[idx]
        if not raw.strip() or raw.lstrip().startswith("#"):
            idx += 1
            continue

        indent = len(raw) - len(raw.lstrip(" "))
        if indent < base_indent:
            break
        if indent > base_indent:
            raise ValueError(f"Unexpected indentation while parsing YAML near line: {raw}")

        stripped = raw.strip()
        key, sep, value = stripped.partition(":")
        if sep == "":
            raise ValueError(f"Invalid YAML line: {raw}")

        key = key.strip()
        value = value.strip()

        if value == "":
            child, idx = _parse_simple_yaml(lines, start_idx=idx + 1, base_indent=base_indent + 2)
            data[key] = child
            continue

        data[key] = _parse_yaml_scalar(value)
        idx += 1

    return data, idx


def load_yaml_like(path: Path, default: dict | None = None) -> dict:
    if not path.exists():
        return {} if default is None else default

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    if yaml is not None:
        raw = yaml.safe_load(text)
        return raw or ({} if default is None else default)

    parsed, _ = _parse_simple_yaml(text.splitlines())
    return parsed or ({} if default is None else default)


def read_yaml(path: Path, default: dict | None = None) -> dict:
    return load_yaml_like(path, default=default)


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)
