from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent

# Load the dashboard-local .env first, then fall back to the repo-root .env.
# Shell environment variables still take precedence.
load_dotenv(BASE_DIR / ".env", override=False)
load_dotenv(PROJECT_ROOT / ".env", override=False)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
METRIC_DIR = OUTPUT_DIR / "metrics"
PRED_DIR = OUTPUT_DIR / "predictions"
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_META_DIR = PROJECT_ROOT / "data" / "meta"

FEATURE_COLUMNS_PATH = DATA_META_DIR / "feature_columns.json"
SELECTED_Q_PATH = METRIC_DIR / "selected_q.json"
THRESHOLDS_PATH = METRIC_DIR / "thresholds.json"
GRAYZONE_GRID_PATH = METRIC_DIR / "grayzone_grid.csv"
GRAPHAD_ARTIFACT_PATH = MODELS_DIR / "graphad_artifact.json"
ROUTING_TRACE_PATH = BASE_DIR / "routing_trace.jsonl"

TE_VAR_PROCESS_MAP = BASE_DIR / "static" / "data" / "TE_variable_process_map.csv"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL_EXPLAIN = os.getenv("OPENAI_MODEL_EXPLAIN", "gpt-4o-mini")
OPENAI_MODEL_QA = os.getenv("OPENAI_MODEL_QA", "gpt-4o-mini")
