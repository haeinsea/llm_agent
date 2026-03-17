from __future__ import annotations

from pathlib import Path
from typing import Callable
import time

import json
import os
import traceback
import urllib.request
import urllib.error
import socket

import numpy as np
import pandas as pd

from src.utils.env import load_dotenv
from src.utils.io import read_csv, read_json, read_yaml, write_csv
from src.utils.metrics import binary_metrics, instability_score, prr, worst_case_recall
from src.utils.routing import compute_base_routing_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PRED_DIR = OUTPUT_DIR / "predictions"
METRIC_DIR = OUTPUT_DIR / "metrics"
SEEDS = [0, 1, 2, 3, 4]
DEFAULT_Q = 0.80
KEY_COLS = ["source_file", "domain_tag", "split_group", "run_id", "fault_id", "sample_idx", "y_true", "phase", "onset_step", "transition_len"]


def zero_usage() -> dict[str, float]:
    return {
        "prompt_tokens": 0.0,
        "completion_tokens": 0.0,
        "total_tokens": 0.0,
        "total_latency_ms": 0.0,
    }


def llm_stub_probability(row: pd.Series) -> float:
    base = float(row["p_utar_base"])
    disagreement = float(np.std([row["p_rf"], row["p_xgb"], row["p_tcn"]]))
    vote = float(np.mean([(row["p_rf"] >= 0.5), (row["p_xgb"] >= 0.5), (row["p_tcn"] >= 0.5)]))
    if vote >= 2 / 3:
        base = min(1.0, base + 0.10 + 0.15 * disagreement)
    else:
        base = max(0.0, base - 0.10 - 0.15 * disagreement)
    return float(base)


def llm_stub_probability_batch(df: pd.DataFrame) -> np.ndarray:
    base = df["p_utar_base"].to_numpy(dtype=float)
    scores = df[["p_rf", "p_xgb", "p_tcn"]].to_numpy(dtype=float)
    disagreement = scores.std(axis=1)
    vote = (scores >= 0.5).mean(axis=1)
    out = base.copy()
    pos = vote >= (2 / 3)
    out[pos] = np.minimum(1.0, out[pos] + 0.10 + 0.15 * disagreement[pos])
    out[~pos] = np.maximum(0.0, out[~pos] - 0.10 - 0.15 * disagreement[~pos])
    return out


class LLMProbabilityRunner:
    def __init__(self, cfg: dict, force_stub: bool = False):
        self.llm_cfg = cfg.get("llm", {})
        load_dotenv(PROJECT_ROOT / ".env")
        self.api_key = os.getenv(str(self.llm_cfg.get("api_env_key", "OPENAI_API_KEY")), "")
        self.model = os.getenv(str(self.llm_cfg.get("model_env_key", "OPENAI_MODEL")), str(self.llm_cfg.get("model", "gpt-4o-mini")))
        self.use_openai = (not force_stub) and bool(self.llm_cfg.get("enabled", False)) and str(self.llm_cfg.get("mode", "stub")).lower() == "openai" and self.api_key
        self.temperature = float(self.llm_cfg.get("temperature", 0.0))
        self.timeout_sec = int(self.llm_cfg.get("timeout_sec", 30))
        self.max_retries = int(self.llm_cfg.get("max_retries", 4))
        self.retry_backoff_sec = float(self.llm_cfg.get("retry_backoff_sec", 2.0))
        self.request_pause_sec = float(self.llm_cfg.get("request_pause_sec", 0.0))
        self.progress_every = int(self.llm_cfg.get("progress_every", 50))
        self.allow_fallback = bool(self.llm_cfg.get("allow_stub_fallback", False))
        self.error_log_path = METRIC_DIR / "selective_llm_errors.log"
        self.disabled = False

    @property
    def is_stub(self) -> bool:
        return not self.use_openai or self.disabled

    def _call_openai(self, row: pd.Series) -> tuple[float, dict[str, float]]:
        prompt = (
            "Refine a TE anomaly probability. Return JSON with key probability in [0,1]. "
            f"rf={float(row['p_rf']):.6f}, xgb={float(row['p_xgb']):.6f}, "
            f"tcn={float(row['p_tcn']):.6f}, base={float(row['p_utar_base']):.6f}."
        )
        payload = {
            "model": self.model,
            "input": prompt,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "probability_output",
                    "schema": {
                        "type": "object",
                        "properties": {"probability": {"type": "number"}},
                        "required": ["probability"],
                        "additionalProperties": False,
                    },
                }
            },
            "temperature": self.temperature,
        }
        req = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            if self.request_pause_sec > 0:
                time.sleep(self.request_pause_sec)
            started = time.perf_counter()
            print(
                f"[openai] request start attempt={attempt + 1}/{self.max_retries + 1} "
                f"sample_idx={int(row['sample_idx'])} run_id={int(row['run_id'])} phase={row['phase']}",
                flush=True,
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                text = self._extract_output_text(body)
                parsed = json.loads(text) if isinstance(text, str) else text
                prob = float(parsed.get("probability"))
                usage = body.get("usage") or {}
                print(
                    f"[openai] request success attempt={attempt + 1}/{self.max_retries + 1} "
                    f"sample_idx={int(row['sample_idx'])} latency_ms={elapsed_ms:.1f} "
                    f"input_tokens={usage.get('input_tokens', 0)} output_tokens={usage.get('output_tokens', 0)}",
                    flush=True,
                )
                return float(np.clip(prob, 0.0, 1.0)), {
                    "prompt_tokens": float(usage.get("input_tokens", 0.0) or 0.0),
                    "completion_tokens": float(usage.get("output_tokens", 0.0) or 0.0),
                    "total_tokens": float(usage.get("total_tokens", 0.0) or 0.0),
                    "total_latency_ms": float(elapsed_ms),
                }
            except urllib.error.HTTPError as exc:
                status = getattr(exc, "code", None)
                last_exc = RuntimeError(f"HTTPError status={status}: {self._safe_http_error_body(exc)}")
                print(
                    f"[openai] request http_error attempt={attempt + 1}/{self.max_retries + 1} "
                    f"sample_idx={int(row['sample_idx'])} status={status}",
                    flush=True,
                )
                if attempt >= self.max_retries or status not in {408, 409, 429, 500, 502, 503, 504}:
                    raise last_exc
            except (urllib.error.URLError, TimeoutError, socket.timeout, json.JSONDecodeError, ValueError) as exc:
                last_exc = exc
                print(
                    f"[openai] request retryable_error attempt={attempt + 1}/{self.max_retries + 1} "
                    f"sample_idx={int(row['sample_idx'])} error={type(exc).__name__}: {exc}",
                    flush=True,
                )
                if attempt >= self.max_retries:
                    raise
            time.sleep(self.retry_backoff_sec * (2 ** attempt))
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("OpenAI call failed without a captured exception.")

    def _safe_http_error_body(self, exc: urllib.error.HTTPError) -> str:
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = "<unavailable>"
        return body[:1000]

    def _extract_output_text(self, body: dict) -> str:
        if "output_text" in body and body["output_text"]:
            return str(body["output_text"])
        for output_item in body.get("output", []):
            for content_item in output_item.get("content", []):
                if "text" in content_item and content_item["text"]:
                    return str(content_item["text"])
        raise ValueError(f"Could not locate JSON text in Responses API payload: keys={list(body.keys())}")

    def _log_error(self, row: pd.Series, exc: Exception) -> None:
        payload = {
            "model": self.model,
            "sample_idx": int(row["sample_idx"]),
            "fault_id": int(row["fault_id"]),
            "run_id": int(row["run_id"]),
            "phase": str(row["phase"]),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "timeout_sec": self.timeout_sec,
            "max_retries": self.max_retries,
            "traceback": traceback.format_exc(),
        }
        with open(self.error_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def probability_with_usage(self, row: pd.Series) -> tuple[float, dict[str, float]]:
        if self.is_stub:
            return llm_stub_probability(row), zero_usage()
        try:
            return self._call_openai(row)
        except Exception as exc:
            self._log_error(row, exc)
            if self.allow_fallback:
                self.disabled = True
                return llm_stub_probability(row), zero_usage()
            raise RuntimeError(
                "OpenAI call failed during selective_llm_eval. "
                f"See {self.error_log_path} for details. "
                "Set llm.allow_stub_fallback: true only if you intentionally want silent fallback."
            ) from exc

    def apply(self, df: pd.DataFrame, progress_label: str = "") -> tuple[np.ndarray, pd.DataFrame]:
        if len(df) == 0:
            usage_df = pd.DataFrame(columns=["prompt_tokens", "completion_tokens", "total_tokens", "total_latency_ms"], index=df.index)
            return np.array([], dtype=float), usage_df
        if self.is_stub:
            probs = llm_stub_probability_batch(df)
            usage_df = pd.DataFrame([zero_usage()] * len(df), index=df.index)
            return probs, usage_df

        label = progress_label.strip() or "llm"
        print(f"[{label}] starting {len(df):,} OpenAI calls", flush=True)
        probs = []
        usages = []
        for idx, (_, row) in enumerate(df.iterrows(), start=1):
            prob, usage = self.probability_with_usage(row)
            probs.append(prob)
            usages.append(usage)
            if self.progress_every > 0 and (idx == 1 or idx % self.progress_every == 0 or idx == len(df)):
                print(
                    f"[{label}] completed {idx:,}/{len(df):,} calls "
                    f"(sample_idx={int(row['sample_idx'])}, run_id={int(row['run_id'])})",
                    flush=True,
                )
        usage_df = pd.DataFrame(usages, index=df.index)
        return np.asarray(probs, dtype=float), usage_df


def build_llm_runner(cfg: dict, force_stub: bool = False) -> LLMProbabilityRunner:
    return LLMProbabilityRunner(cfg, force_stub=force_stub)


def build_llm_probability_fn(cfg: dict) -> Callable[[pd.Series], float]:
    llm_cfg = cfg.get("llm", {})
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv(str(llm_cfg.get("api_env_key", "OPENAI_API_KEY")), "")
    model = os.getenv(str(llm_cfg.get("model_env_key", "OPENAI_MODEL")), str(llm_cfg.get("model", "gpt-4o-mini")))
    use_openai = bool(llm_cfg.get("enabled", False)) and str(llm_cfg.get("mode", "stub")).lower() == "openai" and api_key
    if not use_openai:
        return llm_stub_probability

    temperature = float(llm_cfg.get("temperature", 0.0))
    timeout_sec = int(llm_cfg.get("timeout_sec", 30))
    state = {"disabled": False}

    def openai_probability(row: pd.Series) -> float:
        if state["disabled"]:
            return llm_stub_probability(row)
        prompt = (
            "Refine a TE anomaly probability. Return JSON with key probability in [0,1]. "
            f"rf={float(row['p_rf']):.6f}, xgb={float(row['p_xgb']):.6f}, "
            f"tcn={float(row['p_tcn']):.6f}, base={float(row['p_utar_base']):.6f}."
        )
        payload = {
            "model": model,
            "input": prompt,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "probability_output",
                    "schema": {
                        "type": "object",
                        "properties": {"probability": {"type": "number"}},
                        "required": ["probability"],
                        "additionalProperties": False,
                    },
                }
            },
            "temperature": temperature,
        }
        req = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            text = body.get("output", [{}])[0].get("content", [{}])[0].get("text", "{}")
            prob = float(json.loads(text).get("probability"))
            return float(np.clip(prob, 0.0, 1.0))
        except Exception:
            state["disabled"] = True
            return llm_stub_probability(row)

    return openai_probability


def build_base_view(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = df[KEY_COLS].copy()
    out["p_rf"] = df["p_rf"]
    out["p_xgb"] = df["p_xgb"]
    out["p_tcn"] = df["p_tcn"]
    if "p_ensemble" in df.columns:
        out["p_ensemble"] = df["p_ensemble"]
    else:
        out["p_ensemble"] = out[["p_rf", "p_xgb", "p_tcn"]].mean(axis=1)
    out["p_utar_base"] = compute_base_routing_score(out, cfg)
    return out


def get_seed_view(df: pd.DataFrame, seed: int, cfg: dict) -> pd.DataFrame:
    out = df[KEY_COLS].copy()
    out["p_rf"] = df[f"p_rf_seed{seed}"]
    out["p_xgb"] = df[f"p_xgb_seed{seed}"]
    out["p_tcn"] = df[f"p_tcn_seed{seed}"]
    out["p_ensemble"] = out[["p_rf", "p_xgb", "p_tcn"]].mean(axis=1)
    out["p_utar_base"] = compute_base_routing_score(out, cfg)
    return out


def apply_mode(
    df: pd.DataFrame,
    tau: float,
    margin: float,
    xgb_low: float,
    xgb_high: float,
    mode: str,
    llm_runner: LLMProbabilityRunner,
    progress_label: str = "",
) -> pd.DataFrame:
    out = df.copy()
    out["gray_zone"] = (np.abs(out["p_utar_base"] - tau) <= margin).astype(int)
    out["xgb_shortcut"] = ((out["p_xgb"] <= xgb_low) | (out["p_xgb"] >= xgb_high)).astype(int)
    for col in ["prompt_tokens", "completion_tokens", "total_tokens", "llm_latency_ms"]:
        out[col] = 0.0

    if mode == "selective":
        out["llm_called"] = ((out["gray_zone"] == 1) & (out["xgb_shortcut"] == 0)).astype(int)
        out["p_llm"] = np.nan
        if out["llm_called"].sum() > 0:
            llm_idx = out.index[out["llm_called"] == 1]
            probs, usage_df = llm_runner.apply(out.loc[llm_idx], progress_label=progress_label or f"{mode}")
            out.loc[llm_idx, "p_llm"] = probs
            out.loc[llm_idx, "prompt_tokens"] = usage_df["prompt_tokens"]
            out.loc[llm_idx, "completion_tokens"] = usage_df["completion_tokens"]
            out.loc[llm_idx, "total_tokens"] = usage_df["total_tokens"]
            out.loc[llm_idx, "llm_latency_ms"] = usage_df["total_latency_ms"]
        out["p_final"] = np.where(out["llm_called"] == 1, out["p_llm"], out["p_utar_base"])
    elif mode == "no_llm":
        out["llm_called"] = 0
        out["p_llm"] = np.nan
        out["p_final"] = out["p_utar_base"]
    elif mode == "full_llm":
        out["gray_zone"] = 1
        out["xgb_shortcut"] = 0
        out["llm_called"] = 1
        probs, usage_df = llm_runner.apply(out, progress_label=progress_label or f"{mode}")
        out["p_llm"] = probs
        out["prompt_tokens"] = usage_df["prompt_tokens"]
        out["completion_tokens"] = usage_df["completion_tokens"]
        out["total_tokens"] = usage_df["total_tokens"]
        out["llm_latency_ms"] = usage_df["total_latency_ms"]
        out["p_final"] = out["p_llm"]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    out["decision_source"] = np.where(
        out["llm_called"] == 1,
        "llm",
        np.where((out["gray_zone"] == 1) & (out["xgb_shortcut"] == 1), "xgb_shortcut", "direct"),
    )
    return out


def cost_summary(df: pd.DataFrame, llm_cfg: dict) -> dict:
    n_calls = int(df["llm_called"].sum())
    in_cost = float(llm_cfg.get("input_cost_per_1m", 0.15))
    out_cost = float(llm_cfg.get("output_cost_per_1m", 0.60))
    avg_prompt = int(llm_cfg.get("avg_prompt_tokens", 700))
    avg_completion = int(llm_cfg.get("avg_completion_tokens", 120))
    latency_ms = float(llm_cfg.get("latency_ms_per_call", 900))
    has_actual_usage = {"prompt_tokens", "completion_tokens", "llm_latency_ms"}.issubset(df.columns) and (
        df["prompt_tokens"].sum() > 0 or df["completion_tokens"].sum() > 0 or df["llm_latency_ms"].sum() > 0
    )
    if has_actual_usage:
        prompt_tokens = float(df["prompt_tokens"].sum())
        completion_tokens = float(df["completion_tokens"].sum())
        total_tokens = float(df["total_tokens"].sum()) if "total_tokens" in df.columns else prompt_tokens + completion_tokens
        total_latency_ms = float(df["llm_latency_ms"].sum())
    else:
        prompt_tokens = float(n_calls * avg_prompt)
        completion_tokens = float(n_calls * avg_completion)
        total_tokens = float(prompt_tokens + completion_tokens)
        total_latency_ms = float(n_calls * latency_ms)
    cost_usd = (prompt_tokens / 1_000_000) * in_cost + (completion_tokens / 1_000_000) * out_cost
    return {
        "llm_calls": n_calls,
        "prompt_tokens": int(round(prompt_tokens)),
        "completion_tokens": int(round(completion_tokens)),
        "total_tokens": int(round(total_tokens)),
        "cost_usd": float(cost_usd),
        "avg_cost_per_call_usd": float(cost_usd / n_calls) if n_calls else 0.0,
        "total_latency_ms": float(total_latency_ms),
        "avg_latency_ms_per_sample": float(total_latency_ms / len(df)) if len(df) else 0.0,
        "uses_actual_api_usage": bool(has_actual_usage),
    }


def evaluate_frame(df: pd.DataFrame, tau: float, ref_recall: float) -> dict:
    m = binary_metrics(df["y_true"], df["p_final"], tau=tau)
    m["gray_ratio"] = float(df["gray_zone"].mean())
    m["llm_call_rate"] = float(df["llm_called"].mean())
    m["xgb_shortcut_rate"] = float(df["xgb_shortcut"].mean())
    m["instability"] = instability_score(df["p_final"], df["run_id"], df["phase"])
    m["worst_case_recall"] = worst_case_recall(df["y_true"], df["p_final"], df["run_id"], tau=tau, window=50)
    m["prr"] = prr(ref_recall, m["recall"])
    return m


def run_mode(
    base_df: pd.DataFrame,
    tau: float,
    margin: float,
    cfg: dict,
    mode: str,
    llm_runner: LLMProbabilityRunner,
    progress_label: str = "",
) -> tuple[pd.DataFrame, dict]:
    xgb_low = float(cfg.get("xgb_shortcut_low", 0.20))
    xgb_high = float(cfg.get("xgb_shortcut_high", 0.80))
    llm_cfg = cfg.get("llm", {})
    ref_recall = binary_metrics(base_df["y_true"], base_df["p_utar_base"], tau=tau)["recall"]
    out = apply_mode(base_df, tau, margin, xgb_low, xgb_high, mode, llm_runner, progress_label=progress_label)
    metrics = evaluate_frame(out, tau=tau, ref_recall=ref_recall)
    metrics.update(cost_summary(out, llm_cfg))
    metrics["tau"] = tau
    metrics["gray_margin"] = margin
    metrics["mode"] = mode
    return out, metrics


def aggregate_seed_predictions(seed_outputs: list[pd.DataFrame]) -> pd.DataFrame:
    if len(seed_outputs) == 1:
        return seed_outputs[0].copy()
    merged = seed_outputs[0][KEY_COLS].copy()
    numeric_cols = ["p_rf", "p_xgb", "p_tcn", "p_ensemble", "p_utar_base", "p_final", "gray_zone", "xgb_shortcut", "llm_called"]
    for col in numeric_cols:
        merged[col] = np.mean([df[col].to_numpy(dtype=float) for df in seed_outputs], axis=0)
    merged["decision_source"] = seed_outputs[0]["decision_source"].values
    merged["gray_zone"] = (merged["gray_zone"] >= 0.5).astype(int)
    merged["xgb_shortcut"] = (merged["xgb_shortcut"] >= 0.5).astype(int)
    merged["llm_called"] = (merged["llm_called"] >= 0.5).astype(int)
    return merged


def summarize_rows(rows: list[dict]) -> pd.DataFrame:
    summary_df = pd.DataFrame(rows).sort_values(["dataset", "q", "mode"]).reset_index(drop=True)
    summary_seed = summary_df.copy()
    metric_cols = [
        "f1",
        "recall",
        "precision",
        "roc_auc",
        "prr",
        "gray_ratio",
        "llm_call_rate",
        "worst_case_recall",
        "instability",
        "cost_usd",
        "total_latency_ms",
        "avg_latency_ms_per_sample",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "llm_calls",
        "uses_actual_api_usage",
        "xgb_shortcut_rate",
    ]
    renamed: dict[str, pd.Series] = {}
    for col in metric_cols:
        renamed[f"{col}_mean"] = summary_seed[col]
        renamed[f"{col}_std"] = 0.0
    keep = summary_seed[["dataset", "q", "mode"]].copy()
    for col, values in renamed.items():
        keep[col] = values
    return summary_df, keep


def collect_dataset_modes(
    dataset_name: str,
    base_pred: pd.DataFrame,
    q: float,
    tau: float,
    margin: float,
    cfg: dict,
    llm_runner: LLMProbabilityRunner,
) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    if dataset_name == "cost":
        active_modes = ["selective", "no_llm", "full_llm"]
    else:
        active_modes = ["selective", "no_llm"]

    base_df = build_base_view(base_pred, cfg)
    rows = []
    mode_outputs: dict[str, pd.DataFrame] = {}
    ref_recall = binary_metrics(base_df["y_true"], base_df["p_utar_base"], tau=tau)["recall"]
    for mode in active_modes:
        out, metrics = run_mode(base_df, tau, margin, cfg, mode, llm_runner, progress_label=f"{dataset_name} mode={mode} q={q:.2f}")
        metrics["prr"] = prr(ref_recall, metrics["recall"])
        rows.append({"dataset": dataset_name, "q": float(q), "seed": -1, **metrics})
        mode_outputs[mode] = out
    return rows, mode_outputs


def _gray_margin_for_q(gray_grid: pd.DataFrame, q: float) -> float:
    row = gray_grid[np.isclose(gray_grid["q"], q)]
    if row.empty:
        raise KeyError(f"Gray-zone summary not found for q={q:.2f}")
    return float(row.iloc[0]["gray_margin_mean"])


def _read_existing_summary_rows() -> list[dict]:
    path = METRIC_DIR / "selective_llm_seed_metrics.csv"
    if not path.exists():
        return []
    return read_csv(path).to_dict("records")


def _write_merged_summary(new_rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    existing_rows = _read_existing_summary_rows()
    merged_rows = existing_rows + new_rows
    if not merged_rows:
        empty = pd.DataFrame()
        write_csv(METRIC_DIR / "selective_llm_seed_metrics.csv", empty)
        write_csv(METRIC_DIR / "selective_llm_summary.csv", empty)
        return empty, empty

    summary_df = pd.DataFrame(merged_rows)
    dedup_cols = ["dataset", "q", "mode"]
    if "seed" in summary_df.columns:
        dedup_cols.append("seed")
    summary_df = summary_df.drop_duplicates(subset=dedup_cols, keep="last").sort_values(["dataset", "q", "mode"]).reset_index(drop=True)
    summary_rows = summary_df.to_dict("records")
    summary_df, summary_seed = summarize_rows(summary_rows)
    write_csv(METRIC_DIR / "selective_llm_seed_metrics.csv", summary_df)
    write_csv(METRIC_DIR / "selective_llm_summary.csv", summary_seed)
    return summary_df, summary_seed


def run_main_eval() -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = read_yaml(CONFIG_DIR / "routing.yaml", default={})
    tau_info = read_json(METRIC_DIR / "thresholds.json")
    gray_grid = read_csv(METRIC_DIR / "grayzone_grid.csv").sort_values("q").reset_index(drop=True)
    llm_runner_val = build_llm_runner(cfg, force_stub=True)
    llm_runner_live = build_llm_runner(cfg)

    pred_val = read_csv(PRED_DIR / "base_val_predictions.csv")
    pred_main = read_csv(PRED_DIR / "base_test_main_predictions.csv")

    tau = float(tau_info["tau"])
    default_q = DEFAULT_Q
    default_margin = _gray_margin_for_q(gray_grid, default_q)
    summary_rows: list[dict] = []

    for dataset_name, base_df in [("val", pred_val), ("main", pred_main)]:
        runner = llm_runner_val if dataset_name == "val" else llm_runner_live
        rows, mode_outputs = collect_dataset_modes(dataset_name, base_df, default_q, tau, default_margin, cfg, runner)
        summary_rows.extend(rows)
        prefix = "utar_val" if dataset_name == "val" else "utar_test_main"
        for mode, out_df in mode_outputs.items():
            write_csv(PRED_DIR / f"{prefix}_{mode}.csv", out_df)

    return _write_merged_summary(summary_rows)


def run_cost_eval() -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = read_yaml(CONFIG_DIR / "routing.yaml", default={})
    tau_info = read_json(METRIC_DIR / "thresholds.json")
    gray_grid = read_csv(METRIC_DIR / "grayzone_grid.csv").sort_values("q").reset_index(drop=True)
    llm_runner_live = build_llm_runner(cfg)
    pred_cost = read_csv(PRED_DIR / "base_test_cost_predictions.csv")

    tau = float(tau_info["tau"])
    summary_rows: list[dict] = []
    q_sweep_selective_parts = []
    q_sweep_no_llm_parts = []
    for q in gray_grid["q"]:
        margin = _gray_margin_for_q(gray_grid, float(q))
        rows, mode_outputs = collect_dataset_modes("cost", pred_cost, float(q), tau, margin, cfg, llm_runner_live)
        summary_rows.extend(rows)
        if np.isclose(q, DEFAULT_Q):
            for mode, out_df in mode_outputs.items():
                write_csv(PRED_DIR / f"utar_test_cost_{mode}.csv", out_df)

        sel_df = mode_outputs["selective"].copy()
        sel_df["q"] = float(q)
        sel_df["seed"] = -1
        q_sweep_selective_parts.append(sel_df)

        no_llm_df = mode_outputs["no_llm"].copy()
        no_llm_df["q"] = float(q)
        no_llm_df["seed"] = -1
        q_sweep_no_llm_parts.append(no_llm_df)

    write_csv(PRED_DIR / "utar_q_sweep.csv", pd.concat(q_sweep_selective_parts, ignore_index=True))
    write_csv(PRED_DIR / "utar_q_sweep_no_llm.csv", pd.concat(q_sweep_no_llm_parts, ignore_index=True))
    return _write_merged_summary(summary_rows)


def main() -> None:
    _, main_summary = run_main_eval()
    print("[selective_llm_eval] main 4000 evaluation completed")
    print(main_summary[main_summary["dataset"].isin(["val", "main"])].to_string(index=False))
    _, cost_summary = run_cost_eval()
    print("[selective_llm_eval] cost 500 q-sweep completed")
    print(cost_summary[cost_summary["dataset"] == "cost"].to_string(index=False))


if __name__ == "__main__":
    main()
