from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(dotenv_path: Path | None = None) -> dict[str, str]:
    path = dotenv_path or Path.cwd() / ".env"
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        values[key] = value
        os.environ.setdefault(key, value)
    return values
