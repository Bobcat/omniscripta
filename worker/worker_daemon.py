from __future__ import annotations

import fcntl
import hashlib
import json
import os
import queue
import re
import shutil
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

from queue_fs import INBOX, claim_next_job, finish_job
from worker_status_io import _append_log, _utc_iso, _write_status
from phase_snipping import _make_snippet
from phase_speaker_lines import make_speaker_lines_from_srt
from phase_chunk_speaker_lines import chunk_speaker_lines
from phase_topics_llm import run_topics_llm
from phase_topics_parse import parse_topics_raw_file
from phase_topics_validate import validate_all_chunks
from phase_topics_merge import merge_topics
from progress_predictor import build_prediction, phase_order_for_job
from asr_client_remote import (
  fetch_remote_pending_status,
  fetch_remote_request_status,
  stream_remote_completions_forever,
  submit_remote_pool_request,
)
from event_loop import WorkerEventBus, WorkerEventType
from inbox_watch import start_inbox_watcher

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))
from shared.app_config import get_str, get_float, get_int, get_setting

def _repo_root() -> Path:
  # worker/worker_daemon.py -> worker -> repo root
  return Path(__file__).resolve().parents[1]


def _resolve_cfg_path(path_value: str, *, fallback_rel: str) -> Path:
  raw = str(path_value or "").strip() or fallback_rel
  p = Path(raw)
  return p if p.is_absolute() else (_repo_root() / p)


PROGRESS_DB_DIR = _resolve_cfg_path(
  get_str("worker.progress_db_dir", "data/progress_db"),
  fallback_rel="data/progress_db",
)
_runs_path_cfg = get_str("worker.progress_runs_path", "").strip()
if _runs_path_cfg:
  RUNS_V1_PATH = _resolve_cfg_path(_runs_path_cfg, fallback_rel="data/progress_db/runs_v1.jsonl")
else:
  RUNS_V1_PATH = (PROGRESS_DB_DIR / "runs_v1.jsonl").resolve()


def _load_service_config() -> dict:
  # Unified config source (settings.json + local.json via shared.app_config).
  cfg = {
    "snip": {
      "minutes_default": 15
    },
    "topics": {
      "chunk_minutes": 15,
      "ctx_len": 16384,
      "ctx_safety": 0.85,
      "prompt_overhead_tokens_est": 1200,
      "token_estimator": "chars_div4",
      "enabled": True,
      "prompt_id": "topics_v1",
      "prompt_path": "prompts/simple_prompt5.txt",
      "model": "matatonic_Mistral-Small-24B-Instruct-2501-4.0bpw-exl2",
      "generation": {
        "max_tokens": 2048,
        "temperature": 0.01,
        "top_p": 1,
        "top_k": 1,
        "typical": 1,
        "min_p": 0,
        "tfs": 1,
        "top_a": 0,
        "smoothing_factor": 0,
        "repetition_penalty": 1,
        "penalty_range": 1024,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "dry_multiplier": 0,
        "mirostat_mode": 0,
        "xtc_threshold": 0.1,
        "xtc_probability": 0,
        "stream": False,
      },
    },
    "tabby": {
      "base_url": "http://127.0.0.1:5001",
      "api_key_env": "TABBY_API_KEY",
      "timeout_s": 600,
      "retries": 2,
      "retry_sleep_s": 2,
    },
  }

  for key in ("snip", "topics", "tabby"):
    raw = get_setting(key, {})
    if isinstance(raw, dict):
      if key == "topics":
        merged_topics = dict(cfg["topics"])
        merged_topics.update(raw)
        if isinstance(cfg["topics"].get("generation"), dict):
          base_gen = dict(cfg["topics"]["generation"])
          override_gen = raw.get("generation")
          if isinstance(override_gen, dict):
            base_gen.update(override_gen)
          merged_topics["generation"] = base_gen
        cfg["topics"] = merged_topics
      else:
        merged = dict(cfg[key])
        merged.update(raw)
        cfg[key] = merged
  return cfg


def _format_timings_text(rows: list[tuple[str, float]], *, total_s: float | None = None) -> str:
  cumulative = 0.0
  done_rows: list[tuple[str, float]] = []
  for name, sec in rows:
    safe = max(0.0, float(sec))
    cumulative += safe
    done_rows.append((name, safe))

  shown_total = max(0.0, float(total_s)) if total_s is not None else cumulative
  parts: list[str] = []
  for name, sec in done_rows:
    parts.append(f"{name}={sec:.2f}s")
  parts.append(f"total={shown_total:.2f}s")

  return " | ".join(parts)


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
  h = hashlib.sha256()
  with path.open("rb") as f:
    while True:
      b = f.read(chunk_size)
      if not b:
        break
      h.update(b)
  return h.hexdigest()


def _phase_seconds_from_rows(rows: list[tuple[str, float]]) -> dict[str, float]:
  out: dict[str, float] = {}
  for name, sec in rows:
    safe = max(0.0, float(sec))
    out[name] = out.get(name, 0.0) + safe
  return {k: round(v, 6) for k, v in out.items()}


def _host_id() -> str:
  raw = get_str("worker.host_id", "").strip()
  if raw:
    return raw
  return (socket.gethostname().split(".")[0] or "unknown-host").strip() or "unknown-host"


def _worker_instance() -> str:
  raw = get_str("worker.instance", "").strip()
  if raw:
    return raw
  return "1"


def _worker_mode() -> str:
  env_mode = str(os.getenv("TRANSCRIBE_WORKER_MODE") or "").strip().lower()
  cfg_mode = get_str("worker.mode", "").strip().lower()
  mode = env_mode or cfg_mode or "upload"
  if mode not in {"live", "upload"}:
    raise RuntimeError(f"Unsupported worker.mode: {mode!r} (expected 'live' or 'upload')")
  return mode


def _worker_consumer_id(mode: str) -> str:
  return f"worker-{mode}@{_worker_instance()}"


def _worker_live_max_outstanding() -> int:
  return get_int("worker.live.max_outstanding_requests", 2, min_value=1)


def _worker_upload_max_outstanding() -> int:
  return get_int("worker.upload.max_outstanding_requests", 1, min_value=1)


def _worker_coordinator_tick_interval_s() -> float:
  return get_float("worker_events.coordinator_tick_s", 0.2, min_value=0.05)


def _worker_pending_status_poll_interval_s() -> float:
  return get_float("polling_intervals.asr_remote_pending_status_poll_s", 1.0, min_value=0.2)


def _worker_inbox_debounce_ms() -> int:
  return get_int("worker_events.inbox_debounce_ms", 40, min_value=0)


def _worker_metrics_log_interval_s() -> float:
  return get_float("worker_events.metrics_log_interval_s", 30.0, min_value=1.0)


@dataclass
class _WorkerLoopCounters:
  inbox_events: int = 0
  sse_reconnects: int = 0
  feed_resets: int = 0
  submits_started: int = 0
  submits_succeeded: int = 0
  submits_failed: int = 0
  scheduler_refill_cycles: int = 0
  completions_seen: int = 0
  completions_matched: int = 0
  last_log_mono: float = field(default_factory=time.monotonic)


def _maybe_log_worker_counters(
  *,
  mode: str,
  consumer_id: str,
  counters: _WorkerLoopCounters,
  pending_count: int,
  submitting_count: int,
  interval_s: float,
  force: bool = False,
) -> None:
  now = time.monotonic()
  if not force and (now - float(counters.last_log_mono)) < max(0.0, float(interval_s)):
    return
  counters.last_log_mono = now
  print(
    "worker_daemon counters "
    f"mode={mode} consumer_id={consumer_id} "
    f"inbox_events={int(counters.inbox_events)} "
    f"sse_reconnects={int(counters.sse_reconnects)} "
    f"feed_resets={int(counters.feed_resets)} "
    f"submits_started={int(counters.submits_started)} "
    f"submits_succeeded={int(counters.submits_succeeded)} "
    f"submits_failed={int(counters.submits_failed)} "
    f"scheduler_refill_cycles={int(counters.scheduler_refill_cycles)} "
    f"completions_seen={int(counters.completions_seen)} "
    f"completions_matched={int(counters.completions_matched)} "
    f"pending={int(max(0, int(pending_count)))} "
    f"submitting={int(max(0, int(submitting_count)))}",
    flush=True,
  )


def _hardware_key(host_id: str) -> str:
  raw = get_str("worker.hardware_key", "").strip()
  if raw:
    return raw
  if host_id == "dc1":
    return "dc1-rtx5070ti-cuda"
  if host_id == "dc2":
    return "dc2-rtx5090-cuda"
  return f"{host_id}-unknown"


def _normalize_speaker_mode(value: object) -> str:
  raw = str(value or "auto").strip().lower()
  if raw in {"none", "off", "disabled", "no_speaker", "nospeaker", "no-speaker"}:
    return "none"
  if raw == "fixed":
    return "fixed"
  return "auto"


def _resolve_job_kind(job_cfg: dict) -> str:
  raw = str(job_cfg.get("job_kind") or "").strip().lower()
  if raw:
    return raw
  opts = job_cfg.get("options", {}) or {}
  if bool(opts.get("live_chunk_mode", False)):
    return "live_chunk"
  return "upload_audio"


def _config_key(
  *,
  language: str,
  speaker_mode: str,
  snippet_seconds: int,
  topics_enabled: bool,
  prompt_id: str,
  whisperx_cfg: dict,
) -> str:
  payload = {
    "language": language,
    "speaker_mode": speaker_mode,
    "snippet_seconds": int(snippet_seconds),
    "topics_enabled": bool(topics_enabled),
    "prompt_id": prompt_id,
    "whisperx": {
      "model": whisperx_cfg.get("model"),
      "device": whisperx_cfg.get("device"),
      "compute_type": whisperx_cfg.get("compute_type"),
      "batch_size": whisperx_cfg.get("batch_size"),
      "chunk_size": whisperx_cfg.get("chunk_size"),
      "beam_size": whisperx_cfg.get("beam_size"),
      "align_model": whisperx_cfg.get("align_model"),
      "diarize_model": whisperx_cfg.get("diarize_model"),
    },
  }
  blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
  return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _append_progress_run_if_new_done(record: dict[str, object]) -> tuple[bool, str]:
  """
  Append done-run record to runs_v1.jsonl unless the same
  (hash, snippet_seconds, topics_enabled, speaker_mode)
  already has a done record.
  Returns (written, reason).
  """
  if str(record.get("outcome", "")) != "done":
    return False, "non_done_skipped"

  content_hash = str(record.get("content_hash_sha256", "") or "")
  if not content_hash:
    return False, "missing_hash"
  try:
    snippet_seconds = int(record.get("snippet_seconds", -1))
  except Exception:
    snippet_seconds = -1
  topics_enabled = bool(record.get("topics_enabled", False))
  speaker_mode = _normalize_speaker_mode(record.get("speaker_mode", "auto"))

  RUNS_V1_PATH.parent.mkdir(parents=True, exist_ok=True)
  with RUNS_V1_PATH.open("a+", encoding="utf-8") as f:
    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
    try:
      f.seek(0)
      for line in f:
        s = line.strip()
        if not s:
          continue
        try:
          obj = json.loads(s)
        except Exception:
          continue
        if str(obj.get("outcome", "")) != "done":
          continue
        if str(obj.get("content_hash_sha256", "")) != content_hash:
          continue
        try:
          obj_snip = int(obj.get("snippet_seconds", -1))
        except Exception:
          obj_snip = -1
        obj_topics = bool(obj.get("topics_enabled", False))
        obj_speaker_mode = _normalize_speaker_mode(obj.get("speaker_mode", "auto"))
        if obj_snip == snippet_seconds and obj_topics == topics_enabled and obj_speaker_mode == speaker_mode:
          return False, "duplicate_hash_snippet_topics_speaker_done"

      f.seek(0, os.SEEK_END)
      f.write(json.dumps(record, ensure_ascii=False) + "\n")
      f.flush()
      os.fsync(f.fileno())
      return True, "written"
    finally:
      fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _build_progress_tracker(
  *,
  status_path: Path,
  phase_order: list[str],
  phase_expected_s: dict[str, float],
  eta_confidence: float,
  eta_hints: list[str],
):
  """
  Returns closures:
    start_phase(phase_key, base_message, status_phase)
    finish_phase(phase_key, actual_elapsed_s)
    heartbeat()
    set_base_message(base_message)
  """
  completed_actual: dict[str, float] = {}
  current_phase_key: str | None = None
  current_phase_started_t = 0.0
  current_status_phase = ""
  current_base_message = "Running…"
  current_chunk_idx = 0
  current_chunk_total = 0
  current_chunk_started_t = 0.0
  last_progress = 0.0
  last_write_t = 0.0
  phase_overrun_active = False
  phase_expected_runtime: dict[str, float] = {
    p: max(0.1, float(phase_expected_s.get(p, 0.0)))
    for p in phase_order
  }
  total_expected_all = max(0.1, sum(max(0.0, float(phase_expected_runtime.get(p, 0.0))) for p in phase_order))
  hints = eta_hints
  cleaned: list[str] = []
  for raw in hints:
    h = str(raw).strip()
    if h and h not in cleaned:
      cleaned.append(h)
  hints[:] = cleaned

  def _after_current(ph: str | None) -> list[str]:
    if ph is None:
      return [p for p in phase_order if p not in completed_actual]
    if ph not in phase_order:
      return []
    i = phase_order.index(ph)
    return [p for p in phase_order[i + 1:] if p not in completed_actual]

  def _sum_completed() -> float:
    return sum(max(0.0, float(v)) for v in completed_actual.values())

  def _sum_completed_expected() -> float:
    total = 0.0
    for p in phase_order:
      if p in completed_actual:
        total += max(0.0, float(phase_expected_runtime.get(p, 0.0)))
    return total

  def _maybe_expand_phase_budget(phase_key: str | None, *, elapsed_s: float) -> None:
    nonlocal total_expected_all
    if not phase_key:
      return
    cur = max(0.1, float(phase_expected_runtime.get(phase_key, 0.1)))
    safe_elapsed = max(0.0, float(elapsed_s))
    # If a phase materially overruns, expand its runtime budget so the
    # progress bar keeps moving smoothly instead of plateauing.
    if safe_elapsed <= (cur * 1.05):
      return
    target = max(cur, safe_elapsed * 1.10)
    # Keep expansion bounded to avoid runaway estimates on pathological hangs.
    cap = max(cur * 8.0, 300.0)
    nxt = min(cap, target)
    if nxt <= cur:
      return
    phase_expected_runtime[phase_key] = float(nxt)
    total_expected_all = max(
      0.1,
      sum(max(0.0, float(phase_expected_runtime.get(p, 0.0))) for p in phase_order),
    )

  def _set_phase_overrun_hint(active: bool) -> None:
    nonlocal phase_overrun_active
    if active:
      if "phase_overrun" not in hints:
        hints.append("phase_overrun")
      phase_overrun_active = True
      return
    if "phase_overrun" in hints:
      hints[:] = [h for h in hints if h != "phase_overrun"]
    phase_overrun_active = False

  def _write_eta(*, force: bool = False) -> None:
    nonlocal last_progress, last_write_t
    now = time.monotonic()
    if not force and (now - last_write_t) < 1.0:
      return

    done_actual = _sum_completed()
    elapsed_current = max(0.0, now - current_phase_started_t) if current_phase_key else 0.0
    expected_current_base = max(0.1, float(phase_expected_runtime.get(current_phase_key or "", 0.0)))
    if current_phase_key:
      _maybe_expand_phase_budget(current_phase_key, elapsed_s=elapsed_current)
      expected_current_base = max(0.1, float(phase_expected_runtime.get(current_phase_key or "", expected_current_base)))
    expected_current = expected_current_base
    remaining_keys = _after_current(current_phase_key)
    remaining_after = sum(max(0.0, float(phase_expected_runtime.get(p, 0.0))) for p in remaining_keys)

    # Upload path runs one remote ASR call that internally includes align/diarize/finalize.
    # Treat this as one combined budget while the worker-visible phase is whisperx_transcribe.
    if current_phase_key == "whisperx_transcribe":
      proxied_keys = {"whisperx_align", "whisperx_diarize", "whisperx_finalize"}
      proxy_extra = sum(max(0.0, float(phase_expected_runtime.get(p, 0.0))) for p in remaining_keys if p in proxied_keys)
      if proxy_extra > 0.0:
        expected_current = max(expected_current, expected_current + proxy_extra)
        remaining_after = sum(max(0.0, float(phase_expected_runtime.get(p, 0.0))) for p in remaining_keys if p not in proxied_keys)
    overrun_factor = 1.1
    if current_phase_key == "whisperx_transcribe":
      # Remote ASR can include hidden sub-stages; avoid noisy overrun hints for this phase.
      overrun_factor = 3.0
    overrun_now = bool(
      current_phase_key
      and expected_current > 0
      and elapsed_current > (expected_current * overrun_factor)
    )
    if current_phase_key == "whisperx_transcribe":
      overrun_now = False
    _set_phase_overrun_hint(overrun_now)

    current_projected_total = max(expected_current, elapsed_current)

    if current_phase_key:
      est_total = done_actual + current_projected_total + remaining_after
      est_elapsed = done_actual + elapsed_current
      est_remaining = max(0.0, (current_projected_total - elapsed_current) + remaining_after)
    else:
      # Between phases
      est_total = max(0.1, done_actual + remaining_after)
      est_elapsed = done_actual
      est_remaining = max(0.0, est_total - est_elapsed)

    if current_phase_key and current_phase_key != "whisperx_transcribe" and elapsed_current > (expected_current * 1.05):
      # Prevent frozen ETA on long overruns by carrying a small dynamic tail for
      # the active phase itself (in addition to remaining planned phases).
      overrun_tail = min(120.0, max(3.0, elapsed_current * 0.25))
      est_remaining = max(est_remaining, overrun_tail + remaining_after)
      est_total = max(est_total, est_elapsed + est_remaining)

    # For chunked llm_topics, keep ETA chunk-aware so it does not collapse to
    # zero too early while there are clearly chunks left.
    if (
      current_phase_key == "llm_topics"
      and current_chunk_total > 1
      and 1 <= current_chunk_idx <= current_chunk_total
    ):
      expected_chunk = max(0.1, expected_current / float(current_chunk_total))
      elapsed_chunk = max(0.0, now - current_chunk_started_t) if current_chunk_started_t > 0.0 else 0.0
      projected_current_chunk = max(expected_chunk, elapsed_chunk)
      remaining_chunks = max(0, current_chunk_total - current_chunk_idx)

      if current_chunk_idx > 1:
        elapsed_prev_chunks = max(0.0, elapsed_current - elapsed_chunk)
        avg_done_chunk = max(0.1, elapsed_prev_chunks / float(current_chunk_idx - 1))
      else:
        avg_done_chunk = expected_chunk
      projected_next_chunk = max(expected_chunk, avg_done_chunk)

      llm_remaining = max(0.0, projected_current_chunk - elapsed_chunk) + (remaining_chunks * projected_next_chunk)
      if llm_remaining > est_remaining:
        est_remaining = llm_remaining
        est_total = max(est_total, est_elapsed + est_remaining)

    # Avoid showing 0:00 while work is still active.
    min_active_remaining = 3.0 if current_phase_key == "llm_topics" else 1.0
    if current_phase_key is not None and est_remaining < min_active_remaining:
      est_remaining = min_active_remaining
      est_total = max(est_total, est_elapsed + est_remaining)

    # UX progress is phase-weighted (plan based), so late overruns in earlier
    # phases do not collapse visibility for remaining phases (notably llm_topics).
    completed_expected = _sum_completed_expected()
    if current_phase_key:
      # Let long-running current phases consume remaining expected budget so
      # progress does not appear frozen when remote ASR bundles multiple
      # sub-stages behind one worker-visible phase.
      progress_phase_expected = expected_current_base
      phase_frac = min(0.995, max(0.0, elapsed_current / progress_phase_expected))
      if current_phase_key == "llm_topics" and current_chunk_total > 1 and 1 <= current_chunk_idx <= current_chunk_total:
        chunk_base = max(0.0, float(current_chunk_idx - 1) / float(current_chunk_total))
        chunk_ceiling = min(0.995, float(current_chunk_idx) / float(current_chunk_total))
        chunk_span = max(0.0001, chunk_ceiling - chunk_base)
        expected_chunk = max(0.1, progress_phase_expected / float(current_chunk_total))
        elapsed_chunk = max(0.0, now - current_chunk_started_t) if current_chunk_started_t > 0.0 else 0.0
        chunk_frac = min(0.995, max(0.0, elapsed_chunk / expected_chunk))
        phase_frac = min(chunk_ceiling, chunk_base + (chunk_frac * chunk_span))
      raw_progress = (completed_expected + (phase_frac * progress_phase_expected)) / total_expected_all
    else:
      raw_progress = completed_expected / total_expected_all

    progress_cap = 0.99
    if current_phase_key == "whisperx_transcribe":
      # Keep visible headroom for downstream upload phases (align/diarize/postprocess).
      progress_cap = 0.90
    progress = min(progress_cap, max(last_progress, float(raw_progress)))
    last_progress = progress
    last_write_t = now

    _write_status(
      status_path,
      progress=progress,
      phase=current_status_phase or None,
      message=current_base_message,
      progress_mode="predictive_v1",
      eta_total_s=round(est_total, 3),
      eta_remaining_s=round(est_remaining, 3),
      elapsed_s=round(est_elapsed, 3),
      eta_confidence=round(float(eta_confidence), 3),
      eta_hints=list(hints),
    )

  def start_phase(phase_key: str, base_message: str, status_phase: str) -> None:
    nonlocal current_phase_key, current_phase_started_t, current_status_phase, current_base_message
    nonlocal current_chunk_idx, current_chunk_total, current_chunk_started_t
    current_phase_key = phase_key
    current_phase_started_t = time.monotonic()
    current_status_phase = status_phase
    current_base_message = base_message
    current_chunk_idx = 0
    current_chunk_total = 0
    current_chunk_started_t = 0.0
    _write_eta(force=True)

  def finish_phase(phase_key: str, actual_elapsed_s: float) -> None:
    nonlocal current_phase_key, current_phase_started_t
    safe = max(0.0, float(actual_elapsed_s))
    completed_actual[phase_key] = completed_actual.get(phase_key, 0.0) + safe
    if current_phase_key == phase_key:
      current_phase_key = None
      current_phase_started_t = 0.0
      _set_phase_overrun_hint(False)
    _write_eta(force=True)

  def heartbeat() -> None:
    _write_eta(force=False)

  def set_base_message(base_message: str, *, status_phase: str | None = None) -> None:
    nonlocal current_base_message, current_status_phase
    nonlocal current_chunk_idx, current_chunk_total, current_chunk_started_t
    current_base_message = base_message
    if current_phase_key == "llm_topics":
      m = re.search(r"\bLLM\s+chunk\s+(\d+)\s*/\s*(\d+)\b", str(base_message), re.IGNORECASE)
      if m:
        try:
          idx = int(m.group(1))
          total = int(m.group(2))
        except Exception:
          idx = 0
          total = 0
        if total > 0 and 1 <= idx <= total:
          current_chunk_idx = idx
          current_chunk_total = total
          current_chunk_started_t = time.monotonic()
    if status_phase:
      current_status_phase = status_phase
    _write_eta(force=True)

  return start_phase, finish_phase, heartbeat, set_base_message


def _start_progress_heartbeat_thread(callback, *, interval_s: float = 0.5):
  stop_event = threading.Event()

  def _run() -> None:
    while not stop_event.wait(max(0.05, float(interval_s))):
      try:
        callback()
      except Exception:
        pass

  t = threading.Thread(target=_run, name="progress-heartbeat", daemon=True)
  t.start()
  return stop_event, t


@dataclass
class _PendingLiveJob:
  request_id: str
  job: object
  job_cfg: dict[str, Any]
  job_t0_mono: float


@dataclass
class _LiveSubmitWork:
  job: object
  job_cfg: dict[str, Any]
  job_t0_mono: float


def _noop_start_phase(_phase_key: str, _base_message: str, _status_phase: str) -> None:
  return None


def _noop_finish_phase(_phase_key: str, _actual_elapsed_s: float) -> None:
  return None


def _noop_progress_heartbeat() -> None:
  return None


def _noop_set_message(_base_message: str, *, status_phase: str | None = None) -> None:
  _ = status_phase
  return None


@dataclass
class _PendingUploadJob:
  job: object
  job_t0_mono: float
  job_started_utc: str
  timing_rows: list[tuple[str, float]] = field(default_factory=list)
  content_hash_sha256: str = ""
  chunks_count: int = 0
  eta_confidence: float = 0.0
  eta_hints: list[str] = field(default_factory=list)
  snippet_seconds: int = 0
  language: str = "nl"
  speaker_mode: str = "auto"
  topics_enabled: bool = False
  prompt_id: str = "topics_v1"
  cfg: dict[str, object] = field(default_factory=dict)
  service_cfg: dict[str, Any] = field(default_factory=dict)
  snippet_path: Path | None = None
  orig_filename: str = ""
  request_id: str = ""
  wx_t0_mono: float = 0.0
  wx_live_timing_keys: set[str] = field(default_factory=set)
  asr_stage: str = ""
  asr_stage_started_at_utc: str = ""
  asr_wait_message: str = ""
  progress_start_phase: Any = _noop_start_phase
  progress_finish_phase: Any = _noop_finish_phase
  progress_heartbeat: Any = _noop_progress_heartbeat
  progress_set_message: Any = _noop_set_message


@dataclass
class _UploadSubmitWork:
  pending: _PendingUploadJob


def _new_pending_upload_job(*, job: object) -> _PendingUploadJob:
  return _PendingUploadJob(
    job=job,
    job_t0_mono=time.monotonic(),
    job_started_utc=_utc_iso(),
  )


def _parse_utc_unix(value: Any) -> float | None:
  text = str(value or "").strip()
  if not text:
    return None
  try:
    return float(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp())
  except Exception:
    return None


def _elapsed_utc_s(start_utc: Any, end_utc: Any) -> float | None:
  start_unix = _parse_utc_unix(start_utc)
  end_unix = _parse_utc_unix(end_utc)
  if start_unix is None or end_unix is None:
    return None
  return max(0.0, float(end_unix - start_unix))


def _asr_stage_to_phase(stage: str) -> str:
  key = str(stage or "").strip().lower()
  mapping = {
    "prepare": "whisperx_prepare",
    "transcribe": "whisperx_transcribe",
    "align": "whisperx_align",
    "diarize": "whisperx_diarize",
    "done": "whisperx_finalize",
  }
  return str(mapping.get(key) or "")


def _upload_wait_message(
  *,
  request_id: str,
  state: str,
  stage: str,
  queue_position: int | None = None,
) -> str:
  rid = str(request_id or "").strip()
  state_key = str(state or "").strip().lower()
  stage_key = str(stage or "").strip().lower()
  base = f"Waiting for ASR completion ({rid})..." if rid else "Waiting for ASR completion..."
  if state_key == "queued":
    if queue_position is not None and int(queue_position) > 0:
      return f"Queued for ASR (position {int(queue_position)})..."
    return "Queued for ASR..."
  if state_key in {"running", "cancel_requested"}:
    stage_messages = {
      "prepare": "Preparing WhisperX...",
      "transcribe": "Transcribing...",
      "align": "Aligning...",
      "diarize": "Diarizing...",
      "done": "Finalizing...",
    }
    msg = stage_messages.get(stage_key)
    if msg:
      return msg
    if state_key == "cancel_requested":
      return "ASR cancel requested..."
    return "ASR running..."
  return base


def _record_upload_asr_stage_duration(
  *,
  pending: _PendingUploadJob,
  stage: str,
  stage_started_at_utc: str,
  next_stage_started_at_utc: str,
) -> None:
  phase_name = _asr_stage_to_phase(stage)
  if not phase_name:
    return
  if phase_name in pending.wx_live_timing_keys:
    return
  elapsed = _elapsed_utc_s(stage_started_at_utc, next_stage_started_at_utc)
  if elapsed is None:
    return
  pending.wx_live_timing_keys.add(phase_name)
  _record_upload_phase_timing(pending=pending, name=phase_name, elapsed_s=float(elapsed))
  pending.progress_finish_phase(phase_name, float(elapsed))


def _apply_upload_pending_status(*, pending: _PendingUploadJob, row: dict[str, Any]) -> None:
  state = str(row.get("state") or "").strip().lower()
  stage = str(row.get("stage") or "").strip().lower()
  stage_started_at_utc = str(row.get("stage_started_at_utc") or row.get("started_at_utc") or "").strip()
  queue_position_raw = row.get("queue_position")
  queue_position: int | None
  try:
    queue_position = int(queue_position_raw) if queue_position_raw is not None else None
  except Exception:
    queue_position = None
  if state in {"running", "cancel_requested"} and stage:
    prev_stage = str(pending.asr_stage or "").strip().lower()
    prev_started = str(pending.asr_stage_started_at_utc or "").strip()
    if prev_stage and stage != prev_stage and prev_started and stage_started_at_utc:
      _record_upload_asr_stage_duration(
        pending=pending,
        stage=prev_stage,
        stage_started_at_utc=prev_started,
        next_stage_started_at_utc=stage_started_at_utc,
      )
    phase_name = _asr_stage_to_phase(stage)
    if phase_name and stage != prev_stage:
      pending.progress_start_phase(phase_name, _upload_wait_message(
        request_id=pending.request_id,
        state=state,
        stage=stage,
        queue_position=queue_position,
      ), "whisperx_wait")
    pending.asr_stage = stage
    if stage_started_at_utc:
      pending.asr_stage_started_at_utc = stage_started_at_utc
  msg = _upload_wait_message(
    request_id=pending.request_id,
    state=state,
    stage=stage,
    queue_position=queue_position,
  )
  if msg != pending.asr_wait_message:
    pending.asr_wait_message = msg
    pending.progress_set_message(msg, status_phase="whisperx_wait")


def _is_asr_terminal_state(state: str) -> bool:
  return str(state or "").strip().lower() in {"completed", "failed", "cancelled", "superseded"}


def _upload_terminal_event_from_pending_row(*, request_id: str, row: dict[str, Any]) -> dict[str, Any]:
  state = str(row.get("state") or "").strip().lower()
  lifecycle = fetch_remote_request_status(request_id=request_id)
  if bool(lifecycle.get("ok", False)):
    body = dict(lifecycle.get("body") or {})
    if not str(body.get("request_id") or "").strip():
      body["request_id"] = str(request_id)
    return body

  status_code = int(lifecycle.get("status_code") or 0)
  body = dict(lifecycle.get("body") or {})
  code = str(body.get("code") or "ASR_REQUEST_STATUS_FETCH_FAILED")
  message = str(
    body.get("message")
    or f"Failed to fetch ASR request lifecycle after terminal pending-state (http={status_code})"
  )
  retryable = bool(body.get("retryable", False))
  if status_code == 404:
    # v3 intentionally does not recover in-flight requests across ASR pool restarts.
    # If pending-status reports terminal but lifecycle row vanished, fail fast so upload
    # queue throughput recovers and new jobs keep flowing after restart.
    return {
      "request_id": str(request_id),
      "state": "failed",
      "error": {
        "code": "ASR_REQUEST_LOST_AFTER_TERMINAL_PENDING",
        "message": (
          "ASR request lifecycle missing after terminal pending-state "
          "(likely ASR pool restart; in-flight recovery is not implemented in v3)."
        ),
        "retryable": False,
      },
    }
  return {
    "request_id": str(request_id),
    "state": ("failed" if state == "completed" else (state or "failed")),
    "error": {
      "code": code,
      "message": message,
      "retryable": retryable,
    },
  }


def _record_upload_phase_timing(*, pending: _PendingUploadJob, name: str, elapsed_s: float) -> None:
  safe_elapsed = max(0.0, float(elapsed_s))
  pending.timing_rows.append((name, safe_elapsed))
  txt = _format_timings_text(pending.timing_rows)
  _write_status(pending.job.status_path, timings_text=txt)
  _append_log(
    pending.job.log_path,
    f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] WORKER phase_timing name={name} seconds={safe_elapsed:.3f} timings_text={txt}",
  )


def _normalize_optional_language(value: Any) -> str | None:
  if value is None:
    return None
  text = str(value).strip()
  return text or None


def _prepare_live_chunk_request(*, job: object, job_cfg: dict[str, Any]) -> dict[str, Any]:
  opts = job_cfg.get("options", {}) or {}
  orig_filename = str(job_cfg.get("orig_filename") or "").strip()
  if not orig_filename:
    raise RuntimeError("Missing orig_filename in live chunk job config")

  input_path = job.upload_dir / orig_filename
  if not input_path.exists():
    raise RuntimeError(f"Upload missing: {input_path}")

  language = _normalize_optional_language(opts.get("language"))
  try:
    t0_ms = int(opts.get("live_chunk_t0_ms", 0) or 0)
    t1_ms = int(opts.get("live_chunk_t1_ms", 0) or 0)
    if t1_ms > t0_ms:
      snippet_seconds = max(1, int((max(0, t1_ms - t0_ms) + 999) // 1000))
    else:
      snippet_seconds = 1
  except Exception:
    snippet_seconds = 1
  align_enabled = get_setting("live_chunk.align_enabled", False)
  speaker_mode_raw = str(opts.get("speaker_mode", "none") or "none").strip().lower()
  if speaker_mode_raw in {"off", "disabled", "no_speaker", "nospeaker", "no-speaker"}:
    speaker_mode_raw = "none"
  if speaker_mode_raw not in {"none", "auto", "fixed"}:
    speaker_mode_raw = "auto"
  diarize_enabled = bool(opts.get("diarize_enabled", False)) and speaker_mode_raw != "none"
  min_speakers = opts.get("min_speakers")
  max_speakers = opts.get("max_speakers")
  initial_prompt = str(opts.get("initial_prompt") or "").strip()
  beam_size = opts.get("beam_size")
  live_lane = "single"

  _write_status(
    job.status_path,
    state="running",
    phase="whisperx_prepare",
    progress=0.0,
    started_at=_utc_iso(),
    message="Submitting live chunk to ASR pool…",
  )

  try:
    _append_log(
      job.log_path,
      f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] WORKER live_chunk_submit job_cfg={json.dumps(job_cfg, ensure_ascii=False)}",
    )
  except Exception:
    pass

  raw_request = {
    "schema_version": "asr_v1",
    "request_id": str(getattr(job, "job_id", "") or "live_chunk"),
    "profile_id": get_str("live_chunk.asr_profile", "live_fast"),
    "priority": "interactive",
    "audio": {
      "local_path": str(input_path),
      "format": str(input_path.suffix.lstrip(".") or "wav"),
      "sample_rate_hz": 16000,
      "channels": 1,
      "duration_ms": int(max(1, snippet_seconds) * 1000),
    },
    "options": {
      "align_enabled": bool(align_enabled),
      "diarize_enabled": bool(diarize_enabled),
      "speaker_mode": str(speaker_mode_raw),
    },
    "context": {
      "source_kind": "live_chunk",
      "live_session_id": str(opts.get("live_session_id") or ""),
      "live_chunk_index": int(opts.get("live_chunk_index", 0) or 0),
      "t0_offset_ms": int(opts.get("live_chunk_t0_ms", 0) or 0),
      "live_lane": live_lane,
      "job_id": str(getattr(job, "job_id", "") or ""),
    },
    "outputs": {
      "text": False,
      "segments": False,
      "srt": True,
      "srt_inline": False,
      "word_timestamps": False,
    },
  }
  if language is not None:
    raw_request["options"]["language"] = language
  if initial_prompt:
    raw_request["options"]["initial_prompt"] = initial_prompt
  if beam_size is not None:
    try:
      raw_request["options"]["beam_size"] = max(1, int(beam_size))
    except Exception:
      pass
  if speaker_mode_raw == "fixed":
    try:
      if min_speakers is not None:
        raw_request["options"]["min_speakers"] = max(1, int(min_speakers))
    except Exception:
      pass
    try:
      if max_speakers is not None:
        raw_request["options"]["max_speakers"] = max(1, int(max_speakers))
    except Exception:
      pass
  preview_seq = opts.get("preview_seq")
  if preview_seq is not None:
    try:
      raw_request["context"]["preview_seq"] = int(max(0, int(preview_seq)))
    except Exception:
      pass
  preview_audio_end_ms = opts.get("preview_audio_end_ms")
  if preview_audio_end_ms is not None:
    try:
      raw_request["context"]["preview_audio_end_ms"] = int(max(0, int(preview_audio_end_ms)))
    except Exception:
      pass

  request = dict(raw_request)
  try:
    _append_log(
      job.log_path,
      f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] WORKER live_chunk_asr_submit request_id={request.get('request_id')}",
    )
  except Exception:
    pass
  return request


def _finalize_live_chunk_completed(
  *,
  pending: _PendingLiveJob,
  response: dict[str, Any],
) -> None:
  job = pending.job
  job_cfg = pending.job_cfg
  opts = job_cfg.get("options", {}) or {}
  speaker_mode_raw = str(opts.get("speaker_mode", "none") or "none").strip().lower()
  if speaker_mode_raw in {"off", "disabled", "no_speaker", "nospeaker", "no-speaker"}:
    speaker_mode_raw = "none"
  align_enabled = bool(get_setting("live_chunk.align_enabled", False))
  result_obj = dict(response.get("result") or {})
  artifacts = dict(result_obj.get("artifacts") or {})
  srt_path_str = str(artifacts.get("srt_path") or "").strip()
  if not srt_path_str:
    raise RuntimeError("ASR backend response missing result.artifacts.srt_path")
  srt_path = Path(srt_path_str)
  if not srt_path.exists():
    raise RuntimeError(f"ASR backend SRT path missing: {srt_path}")
  local_srt_path = (job.whisperx_dir / srt_path.name).resolve()
  if srt_path.resolve() != local_srt_path:
    local_srt_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(srt_path, local_srt_path)
  else:
    local_srt_path.parent.mkdir(parents=True, exist_ok=True)

  timings = dict(response.get("timings") or {})
  timing_rows: list[tuple[str, float]] = []
  wx_elapsed = max(0.0, float(timings.get("total_s", 0.0) or 0.0))
  timing_rows.append(("whisperx", wx_elapsed))
  timing_map = {
    "prepare_s": "whisperx_prepare",
    "transcribe_s": "whisperx_transcribe",
    "align_s": "whisperx_align",
    "diarize_s": "whisperx_diarize",
    "finalize_s": "whisperx_finalize",
  }
  for src_key, out_key in timing_map.items():
    if src_key not in timings:
      continue
    try:
      timing_rows.append((out_key, max(0.0, float(timings[src_key]))))
    except Exception:
      continue

  resolved_options = dict(response.get("resolved_options") or {})
  runtime_meta = dict(response.get("runtime") or {})
  resolved_initial_prompt = str(resolved_options.get("initial_prompt") or "")
  resolved_initial_prompt_words = len([tok for tok in resolved_initial_prompt.split() if tok])
  total_elapsed = max(0.0, float(time.monotonic() - float(pending.job_t0_mono)))
  timings_text = " | ".join([f"{name}={sec:.2f}s" for name, sec in timing_rows] + [f"total={total_elapsed:.2f}s"])

  def _timing_value(key: str) -> float | None:
    if key not in timings:
      return None
    try:
      return max(0.0, float(timings[key]))
    except Exception:
      return None

  def _runtime_int(key: str) -> int | None:
    if key not in runtime_meta or runtime_meta.get(key) is None:
      return None
    try:
      return int(max(0, int(runtime_meta.get(key))))
    except Exception:
      return None

  def _runtime_float(key: str) -> float | None:
    if key not in runtime_meta or runtime_meta.get(key) is None:
      return None
    try:
      return max(0.0, float(runtime_meta.get(key)))
    except Exception:
      return None

  _write_status(
    job.status_path,
    state="done",
    phase="done",
    progress=1.0,
    finished_at=_utc_iso(),
    message="Done",
    srt_filename=local_srt_path.name,
    timings_text=timings_text,
    speaker_lines_filename="",
    speaker_lines_manifest_filename="",
    topics_status="skipped_live_chunk",
    topics_warning="",
    align_enabled=bool(resolved_options.get("align_enabled", align_enabled)),
    asr_profile_id=str(response.get("profile_id") or ""),
    asr_runner_kind=str(runtime_meta.get("runner_kind") or ""),
    asr_runner_reused=bool(runtime_meta.get("runner_reused", False)),
    asr_backend=str(runtime_meta.get("backend") or ""),
    asr_device=str(runtime_meta.get("device") or ""),
    asr_model=str(runtime_meta.get("model") or ""),
    asr_initial_prompt_chars=len(resolved_initial_prompt),
    asr_initial_prompt_words=int(max(0, resolved_initial_prompt_words)),
    asr_timing_whisperx_total_s=_timing_value("total_s"),
    asr_timing_whisperx_prepare_s=_timing_value("prepare_s"),
    asr_timing_whisperx_transcribe_s=_timing_value("transcribe_s"),
    asr_timing_whisperx_align_s=_timing_value("align_s"),
    asr_timing_whisperx_diarize_s=_timing_value("diarize_s"),
    asr_timing_whisperx_finalize_s=_timing_value("finalize_s"),
    asr_remote_submit_attempts=_runtime_int("remote_submit_attempts"),
    asr_remote_status_attempts_total=_runtime_int("remote_status_attempts_total"),
    asr_remote_status_http_calls=_runtime_int("remote_status_http_calls"),
    asr_remote_cancel_attempts=_runtime_int("remote_cancel_attempts"),
    asr_blob_fetch_ms=_runtime_float("blob_fetch_ms"),
    speaker_mode=speaker_mode_raw,
  )


def _finalize_live_chunk_terminal(pending: _PendingLiveJob, event: dict[str, Any]) -> None:
  state = str(event.get("state") or "").strip().lower()
  job = pending.job
  if state == "completed":
    response = dict(event.get("response") or {})
    if not response:
      raise RuntimeError("Missing response payload in completion event")
    _finalize_live_chunk_completed(pending=pending, response=response)
    finish_job(job, ok=True)
    return
  if state == "superseded":
    err = dict(event.get("error") or {})
    msg = str(err.get("message") or "Superseded by newer live request").strip()
    _write_status(
      job.status_path,
      state="superseded",
      phase="done",
      progress=1.0,
      finished_at=_utc_iso(),
      message=msg,
      error="",
    )
    finish_job(job, ok=True)
    return
  err_obj = dict(event.get("error") or {})
  err_code = str(err_obj.get("code") or "ASR_REMOTE_TERMINAL_ERROR")
  err_msg = str(err_obj.get("message") or f"ASR terminal state: {state or 'unknown'}")
  _write_status(
    job.status_path,
    state=("cancelled" if state == "cancelled" else "error"),
    phase="error",
    progress=1.0,
    finished_at=_utc_iso(),
    message=f"Worker error: {err_code}: {err_msg}",
    error=f"{err_code}: {err_msg}",
  )
  finish_job(job, ok=False)


def _completion_feed_reset_error(*, old_feed_id: str, new_feed_id: str) -> str:
  old_short = (str(old_feed_id or "").strip() or "unknown")[:12]
  new_short = (str(new_feed_id or "").strip() or "unknown")[:12]
  return (
    "ASR pool completion feed reset detected "
    f"(old_feed_id={old_short}, new_feed_id={new_short}); "
    "in-flight jobs before the restart are not recovered in v3."
  )


def _fail_pending_live_due_to_feed_reset(
  *,
  pending: dict[str, _PendingLiveJob],
  consumer_id: str,
  old_feed_id: str,
  new_feed_id: str,
) -> None:
  if not pending:
    return
  err_msg = _completion_feed_reset_error(old_feed_id=old_feed_id, new_feed_id=new_feed_id)
  keep_request_ids: set[str] = set()
  try:
    status_batch = fetch_remote_pending_status(
      consumer_id=consumer_id,
      request_ids=list(pending.keys()),
      limit=200,
    )
    if bool(status_batch.get("ok", False)):
      status_body = dict(status_batch.get("body") or {})
      rows = status_body.get("rows") or []
      if isinstance(rows, list):
        for row in rows:
          if not isinstance(row, dict):
            continue
          rid = str(row.get("request_id") or "").strip()
          if rid:
            keep_request_ids.add(rid)
  except Exception:
    pass
  event = {
    "state": "failed",
    "error": {
      "code": "ASR_POOL_FEED_RESET",
      "message": err_msg,
      "retryable": False,
    },
  }
  failed_request_ids: list[str] = []
  for request_id, pending_job in list(pending.items()):
    if str(request_id) in keep_request_ids:
      # Request still exists on the new feed; do not fail it as stale.
      continue
    try:
      ev = dict(event)
      ev["request_id"] = str(pending_job.request_id)
      _finalize_live_chunk_terminal(pending_job, ev)
      print(f"Error {pending_job.job.job_id}: {err_msg}")
      failed_request_ids.append(str(request_id))
    except Exception as e:
      _write_status(
        pending_job.job.status_path,
        state="error",
        phase="error",
        progress=1.0,
        finished_at=_utc_iso(),
        message=f"Worker error: {err_msg}",
        error=f"{err_msg} ({e!r})",
      )
      finish_job(pending_job.job, ok=False)
      print(f"Error {pending_job.job.job_id}: {err_msg} | fallback={e!r}")
      failed_request_ids.append(str(request_id))
  for rid in failed_request_ids:
    pending.pop(str(rid), None)


def _live_submit_worker_loop(
  *,
  submit_queue: "queue.Queue[_LiveSubmitWork | None]",
  event_bus: WorkerEventBus,
  consumer_id: str,
) -> None:
  while True:
    work = submit_queue.get()
    if work is None:
      return
    payload: dict[str, Any] = {
      "mode": "live",
      "job": work.job,
      "job_cfg": work.job_cfg,
      "job_t0_mono": float(work.job_t0_mono),
    }
    try:
      request_payload = _prepare_live_chunk_request(job=work.job, job_cfg=work.job_cfg)
      submit = submit_remote_pool_request(
        request_payload=request_payload,
        consumer_id=consumer_id,
      )
      payload["submit"] = dict(submit or {})
    except Exception as e:
      payload["error"] = str(e)
    event_bus.put(WorkerEventType.SUBMIT_RESULT, payload)


def _handle_live_submit_result(*, payload: dict[str, Any], pending: dict[str, _PendingLiveJob]) -> bool:
  job = payload.get("job")
  if job is None:
    return False
  job_cfg = dict(payload.get("job_cfg") or {})
  try:
    job_t0 = float(payload.get("job_t0_mono") or time.monotonic())
  except Exception:
    job_t0 = time.monotonic()
  err_msg = str(payload.get("error") or "").strip()
  if err_msg:
    _write_status(
      job.status_path,
      state="error",
      phase="error",
      progress=1.0,
      finished_at=_utc_iso(),
      message=f"Worker error: {err_msg}",
      error=err_msg,
    )
    finish_job(job, ok=False)
    print(f"Error {job.job_id}: {err_msg}")
    return True

  submit = dict(payload.get("submit") or {})
  if not bool(submit.get("ok", False)):
    err_response = dict(submit.get("error_response") or {})
    err = dict(err_response.get("error") or {})
    msg = str(err.get("message") or err_response.get("message") or "ASR submit failed")
    code = str(err.get("code") or err_response.get("code") or "ASR_SUBMIT_FAILED")
    _write_status(
      job.status_path,
      state="error",
      phase="error",
      progress=1.0,
      finished_at=_utc_iso(),
      message=f"Worker error: {code}: {msg}",
      error=f"{code}: {msg}",
    )
    finish_job(job, ok=False)
    print(f"Error {job.job_id}: submit_failed {code}: {msg}")
    return True

  request_id = str(submit.get("request_id") or "").strip()
  if not request_id:
    _write_status(
      job.status_path,
      state="error",
      phase="error",
      progress=1.0,
      finished_at=_utc_iso(),
      message="Worker error: ASR submit response missing request_id",
      error="ASR submit response missing request_id",
    )
    finish_job(job, ok=False)
    print(f"Error {job.job_id}: missing_request_id")
    return True

  lifecycle = dict(submit.get("submit_lifecycle") or {})
  lifecycle_state = str(lifecycle.get("state") or "").strip().lower()
  _write_status(
    job.status_path,
    state="running",
    phase="whisperx_wait",
    progress=0.1,
    message=f"Waiting for ASR completion ({request_id})…",
    asr_request_id=request_id,
  )
  rec = _PendingLiveJob(
    request_id=request_id,
    job=job,
    job_cfg=job_cfg,
    job_t0_mono=job_t0,
  )
  if lifecycle_state in {"completed", "failed", "cancelled", "superseded"}:
    event = dict(lifecycle)
    event["request_id"] = request_id
    try:
      _finalize_live_chunk_terminal(rec, event)
      print(f"Done {job.job_id} state={lifecycle_state}")
    except Exception as e:
      _write_status(
        job.status_path,
        state="error",
        phase="error",
        progress=1.0,
        finished_at=_utc_iso(),
        message=f"Worker error: {e!r}",
        error=str(e),
      )
      finish_job(job, ok=False)
      print(f"Error {job.job_id}: {e!r}")
    return True

  pending[request_id] = rec
  return True


def _live_submit_result_succeeded(payload: dict[str, Any]) -> bool:
  err_msg = str(payload.get("error") or "").strip()
  if err_msg:
    return False
  submit = dict(payload.get("submit") or {})
  if not bool(submit.get("ok", False)):
    return False
  request_id = str(submit.get("request_id") or "").strip()
  if not request_id:
    return False
  return True


def _completion_stream_worker_loop(
  *,
  consumer_id: str,
  event_bus: WorkerEventBus,
  stop_event: threading.Event,
) -> None:
  def _on_event(kind: str, payload: dict[str, Any]) -> None:
    if kind == "completion":
      event_bus.put(WorkerEventType.COMPLETION_EVENT, {"event": dict(payload or {})})
      return
    if kind == "feed_reset":
      event_bus.put(WorkerEventType.FEED_RESET, dict(payload or {}))
      return
    if kind == "stream_error":
      event_bus.put(WorkerEventType.TICK, {"reason": "completion_stream_error"})

  stream_remote_completions_forever(
    consumer_id=consumer_id,
    start_since_seq=0,
    stop_event=stop_event,
    on_event=_on_event,
  )


def _run_live_worker_submit_reap() -> int:
  mode = "live"
  consumer_id = _worker_consumer_id(mode)
  max_outstanding = _worker_live_max_outstanding()
  tick_interval_s = max(0.05, float(_worker_coordinator_tick_interval_s()))
  metrics_log_interval_s = max(1.0, float(_worker_metrics_log_interval_s()))
  event_bus = WorkerEventBus()
  inbox_watcher = start_inbox_watcher(
    inbox_dir=INBOX,
    event_bus=event_bus,
    debounce_ms=_worker_inbox_debounce_ms(),
  )
  submit_queue: "queue.Queue[_LiveSubmitWork | None]" = queue.Queue(maxsize=max(1, int(max_outstanding)))
  submit_thread = threading.Thread(
    target=_live_submit_worker_loop,
    kwargs={
      "submit_queue": submit_queue,
      "event_bus": event_bus,
      "consumer_id": consumer_id,
    },
    name="worker-live-submit",
    daemon=True,
  )
  submit_thread.start()
  completion_stop = threading.Event()
  completion_thread = threading.Thread(
    target=_completion_stream_worker_loop,
    kwargs={
      "consumer_id": consumer_id,
      "event_bus": event_bus,
      "stop_event": completion_stop,
    },
    name="worker-live-completion-stream",
    daemon=True,
  )
  completion_thread.start()
  pending: dict[str, _PendingLiveJob] = {}
  submitting: dict[str, _LiveSubmitWork] = {}
  counters = _WorkerLoopCounters()
  inbox_dirty = True
  # TODO(v3-followup): add restart recovery / re-request reconciliation for pending live jobs.
  print(f"worker_daemon started mode={mode} consumer_id={consumer_id} max_outstanding={max_outstanding}")
  event_bus.put(WorkerEventType.TICK, {"reason": "startup"})
  try:
    while True:
      ev = event_bus.get(timeout_s=tick_interval_s)
      if ev is not None and ev.kind == WorkerEventType.SHUTDOWN:
        break

      did_work = False
      if ev is not None:
        if ev.kind == WorkerEventType.INBOX_DIRTY:
          counters.inbox_events += 1
          inbox_dirty = True
        elif ev.kind == WorkerEventType.SUBMIT_RESULT:
          payload = dict(ev.payload or {})
          if str(payload.get("mode") or "") == "live":
            job = payload.get("job")
            job_id = str(getattr(job, "job_id", "") or "")
            if job_id:
              submitting.pop(job_id, None)
            if _live_submit_result_succeeded(payload):
              counters.submits_succeeded += 1
            else:
              counters.submits_failed += 1
            did_work = _handle_live_submit_result(payload=payload, pending=pending) or did_work
            inbox_dirty = True
        elif ev.kind == WorkerEventType.COMPLETION_EVENT:
          event = dict((ev.payload or {}).get("event") or {})
          rid = str(event.get("request_id") or "").strip()
          if rid:
            counters.completions_seen += 1
            pending_job = pending.pop(rid, None)
            if pending_job is not None:
              counters.completions_matched += 1
              did_work = True
              inbox_dirty = True
              try:
                _finalize_live_chunk_terminal(pending_job, event)
                print(f"Done {pending_job.job.job_id} state={str(event.get('state') or '')}")
              except Exception as e:
                _write_status(
                  pending_job.job.status_path,
                  state="error",
                  phase="error",
                  progress=1.0,
                  finished_at=_utc_iso(),
                  message=f"Worker error: {e!r}",
                  error=str(e),
                )
                finish_job(pending_job.job, ok=False)
                print(f"Error {pending_job.job.job_id}: {e!r}")
        elif ev.kind == WorkerEventType.FEED_RESET:
          counters.feed_resets += 1
          old_feed_id = str((ev.payload or {}).get("old_feed_id") or "").strip()
          new_feed_id = str((ev.payload or {}).get("new_feed_id") or "").strip()
          _fail_pending_live_due_to_feed_reset(
            pending=pending,
            consumer_id=consumer_id,
            old_feed_id=old_feed_id,
            new_feed_id=new_feed_id,
          )
          did_work = True
          inbox_dirty = True
          print(
            "worker_daemon live completion_feed_reset "
            f"old_feed_id={old_feed_id[:12]} new_feed_id={new_feed_id[:12]} since_seq_reset=0"
          )
        elif ev.kind == WorkerEventType.TICK:
          reason = str((ev.payload or {}).get("reason") or "").strip().lower()
          if reason == "completion_stream_error":
            counters.sse_reconnects += 1
        elif ev.kind != WorkerEventType.TICK:
          continue

      if inbox_dirty:
        counters.scheduler_refill_cycles += 1
        while (len(pending) + len(submitting)) < max_outstanding:
          job = claim_next_job(job_kind_filter="live_chunk")
          if not job:
            inbox_dirty = False
            break
          did_work = True
          try:
            job_cfg = json.loads(job.job_path.read_text(encoding="utf-8"))
            work = _LiveSubmitWork(
              job=job,
              job_cfg=job_cfg,
              job_t0_mono=time.monotonic(),
            )
            submitting[str(job.job_id)] = work
            counters.submits_started += 1
            submit_queue.put(work)
          except Exception as e:
            _write_status(
              job.status_path,
              state="error",
              phase="error",
              progress=1.0,
              finished_at=_utc_iso(),
              message=f"Worker error: {e!r}",
              error=str(e),
            )
            finish_job(job, ok=False)
            print(f"Error {job.job_id}: {e!r}")

      if did_work:
        # Keep coordinator responsive after progress without sleeping out the full tick.
        event_bus.put(WorkerEventType.TICK, {"reason": "followup"})
      _maybe_log_worker_counters(
        mode=mode,
        consumer_id=consumer_id,
        counters=counters,
        pending_count=len(pending),
        submitting_count=len(submitting),
        interval_s=metrics_log_interval_s,
        force=False,
      )
  finally:
    _maybe_log_worker_counters(
      mode=mode,
      consumer_id=consumer_id,
      counters=counters,
      pending_count=len(pending),
      submitting_count=len(submitting),
      interval_s=metrics_log_interval_s,
      force=True,
    )
    completion_stop.set()
    completion_thread.join(timeout=1.0)
    inbox_watcher.close()
    submit_queue.put(None)
    submit_thread.join(timeout=1.0)

  return 0


def _prepare_upload_job_for_submit(
  *,
  pending: _PendingUploadJob,
  consumer_id: str,
) -> dict[str, Any] | None:
  job = pending.job

  def record_phase_timing(name: str, elapsed_s: float) -> None:
    _record_upload_phase_timing(pending=pending, name=name, elapsed_s=elapsed_s)

  job_cfg = json.loads(job.job_path.read_text(encoding="utf-8"))
  opts = job_cfg.get("options", {}) or {}
  job_kind = _resolve_job_kind(job_cfg)
  if job_kind != "upload_audio":
    raise RuntimeError(f"Unsupported job_kind: {job_kind}")

  _write_status(
    job.status_path,
    state="running",
    phase="snipping",
    progress=0.0,
    started_at=pending.job_started_utc,
    message="Starting job…",
  )

  pending.orig_filename = str(job_cfg.get("orig_filename") or "")
  pending.service_cfg = _load_service_config()
  snip_cfg = (pending.service_cfg.get("snip") or {}) if isinstance(pending.service_cfg, dict) else {}
  default_min = int(snip_cfg.get("minutes_default", 5))
  if opts.get("snippet_seconds") is not None:
    pending.snippet_seconds = int(opts.get("snippet_seconds"))
  else:
    pending.snippet_seconds = int(default_min * 60)
  pending.language = str(opts.get("language", "nl") or "nl")
  pending.speaker_mode = _normalize_speaker_mode(opts.get("speaker_mode", "auto"))
  min_speakers = opts.get("min_speakers")
  max_speakers = opts.get("max_speakers")
  topics_cfg = pending.service_cfg.get("topics", {}) if isinstance(pending.service_cfg, dict) else {}
  pending.topics_enabled = bool(topics_cfg.get("enabled", False))
  pending.prompt_id = str(topics_cfg.get("prompt_id", "topics_v1"))
  host_id_val = _host_id()
  hardware_key_val = _hardware_key(host_id_val)

  input_path = job.upload_dir / pending.orig_filename
  if not input_path.exists():
    raise RuntimeError(f"Upload missing: {input_path}")
  try:
    pending.content_hash_sha256 = _sha256_file(input_path)
    _append_log(job.log_path, f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] WORKER content_hash_sha256={pending.content_hash_sha256}")
  except Exception as e_hash:
    pending.content_hash_sha256 = ""
    _append_log(job.log_path, f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] WARN content_hash_failed error={e_hash!r}")

  prediction = build_prediction(
    runs_path=RUNS_V1_PATH,
    hardware_key=hardware_key_val,
    topics_enabled=pending.topics_enabled,
    speaker_mode=pending.speaker_mode,
    snippet_seconds=pending.snippet_seconds,
  )
  pending.eta_confidence = float(prediction.confidence)
  pending.eta_hints = list(prediction.hints)
  phase_order = phase_order_for_job(topics_enabled=pending.topics_enabled, speaker_mode=pending.speaker_mode)
  pending.progress_start_phase, pending.progress_finish_phase, pending.progress_heartbeat, pending.progress_set_message = _build_progress_tracker(
    status_path=job.status_path,
    phase_order=phase_order,
    phase_expected_s=prediction.phase_expected_s,
    eta_confidence=prediction.confidence,
    eta_hints=pending.eta_hints,
  )

  disp = f"{pending.snippet_seconds//60} min" if pending.snippet_seconds > 0 and (pending.snippet_seconds % 60) == 0 else f"{pending.snippet_seconds} s"
  pending.progress_start_phase("snipping", f"Creating snippet ({disp})…", "snipping")
  snip_t0 = time.monotonic()
  snip_hb_stop, snip_hb_thread = _start_progress_heartbeat_thread(pending.progress_heartbeat, interval_s=0.5)
  try:
    pending.snippet_path = _make_snippet(input_path, job.snippet_dir, seconds=pending.snippet_seconds)
  finally:
    snip_hb_stop.set()
    snip_hb_thread.join(timeout=1.0)
  snip_elapsed = time.monotonic() - snip_t0
  record_phase_timing("snipping", snip_elapsed)
  pending.progress_finish_phase("snipping", snip_elapsed)
  _write_status(
    job.status_path,
    phase="snipping",
    snippet_filename=pending.snippet_path.name,
    message=f"Snippet created: {pending.snippet_path.name}",
  )

  # Keep config_key schema stable even though ASR now runs remotely via pool.
  pending.cfg = {}
  _append_log(job.log_path, f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] WORKER service_cfg={json.dumps(pending.service_cfg, ensure_ascii=False)}")
  pending.progress_start_phase("whisperx_prepare", "Preparing WhisperX…", "whisperx_prepare")
  pending.wx_t0_mono = time.monotonic()
  pending.wx_live_timing_keys = set()

  raw_asr_request = {
    "schema_version": "asr_v1",
    "request_id": f"{job.job_id}:upload_whisperx",
    "profile_id": "upload_full",
    "audio": {
      "local_path": str(pending.snippet_path.resolve()),
      "duration_ms": int(max(0, pending.snippet_seconds) * 1000),
    },
    "options": {
      "language": pending.language,
      "speaker_mode": pending.speaker_mode,
      "min_speakers": min_speakers,
      "max_speakers": max_speakers,
      "diarize_enabled": bool(pending.speaker_mode != "none"),
      "align_enabled": True,
      "initial_prompt": opts.get("initial_prompt"),
      "timestamps_mode": "segment",
    },
    "context": {
      "source_kind": "upload_audio",
      "job_id": str(job.job_id),
      "orig_filename": str(pending.orig_filename or ""),
    },
    "outputs": {
      "text": False,
      "segments": False,
      "srt": True,
      "srt_inline": False,
      "word_timestamps": False,
    },
    "priority": "background",
  }
  asr_request = dict(raw_asr_request)
  try:
    _append_log(
      job.log_path,
      f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] WORKER asr_request profile={asr_request.get('profile_id')} resolved_options={json.dumps(asr_request.get('resolved_options') or {}, ensure_ascii=False, sort_keys=True)}",
    )
  except Exception:
    pass
  submit = submit_remote_pool_request(
    request_payload=asr_request,
    consumer_id=consumer_id,
  )
  if not bool(submit.get("ok", False)):
    err_resp = dict(submit.get("error_response") or {})
    err = dict(err_resp.get("error") or {})
    raise RuntimeError(
      f"{err.get('code') or err_resp.get('code') or 'ASR_SUBMIT_FAILED'}: "
      f"{err.get('message') or err_resp.get('message') or 'ASR submit failed'}"
    )
  pending.request_id = str(submit.get("request_id") or "").strip()
  if not pending.request_id:
    raise RuntimeError("ASR submit response missing request_id")
  submit_lifecycle = dict(submit.get("submit_lifecycle") or {})
  lifecycle_state = str(submit_lifecycle.get("state") or "").strip().lower()
  lifecycle_stage = str(submit_lifecycle.get("stage") or "").strip().lower()
  queue_position: int | None
  try:
    queue_position = int(submit_lifecycle.get("queue_position")) if submit_lifecycle.get("queue_position") is not None else None
  except Exception:
    queue_position = None
  wait_msg = _upload_wait_message(
    request_id=pending.request_id,
    state=(lifecycle_state or "running"),
    stage=lifecycle_stage,
    queue_position=queue_position,
  )
  _write_status(
    job.status_path,
    phase="whisperx_wait",
    progress=max(0.1, float(0.1)),
    message=wait_msg,
    asr_request_id=pending.request_id,
  )
  pending.asr_wait_message = wait_msg
  pending.progress_set_message(wait_msg, status_phase="whisperx_wait")
  if lifecycle_state in {"completed", "failed", "cancelled", "superseded"}:
    event = dict(submit_lifecycle)
    event["request_id"] = pending.request_id
    return event
  return None


def _finalize_upload_job_terminal(
  *,
  pending: _PendingUploadJob,
  event: dict[str, Any],
) -> None:
  terminal_state = str((event or {}).get("state") or "").strip().lower()
  if terminal_state != "completed":
    err_obj = dict((event or {}).get("error") or {})
    err_msg = str(err_obj.get("message") or f"ASR terminal state: {terminal_state or 'unknown'}")
    raise RuntimeError(
      f"{err_obj.get('code') or 'ASR_REMOTE_TERMINAL_ERROR'}: "
      f"{err_msg}"
    )

  asr_response = dict((event or {}).get("response") or {})
  if not asr_response:
    raise RuntimeError("ASR terminal completion missing response payload")

  job = pending.job
  if pending.snippet_path is None:
    raise RuntimeError("Missing snippet_path while finalizing upload job")

  try:
    _append_log(
      job.log_path,
      f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] WORKER asr_response runtime={json.dumps(asr_response.get('runtime') or {}, ensure_ascii=False, sort_keys=True)} timings={json.dumps(asr_response.get('timings') or {}, ensure_ascii=False, sort_keys=True)}",
    )
  except Exception:
    pass

  def on_wx_phase_timing(phase_name: str, elapsed_s: float) -> None:
    pending.wx_live_timing_keys.add(str(phase_name or ""))
    _record_upload_phase_timing(pending=pending, name=phase_name, elapsed_s=elapsed_s)
    pending.progress_finish_phase(phase_name, elapsed_s)

  asr_result = dict(asr_response.get("result") or {})
  asr_artifacts = dict(asr_result.get("artifacts") or {})
  srt_path_str = str(asr_artifacts.get("srt_path") or "").strip()
  if not srt_path_str:
    raise RuntimeError("ASR backend response missing result.artifacts.srt_path")
  srt_path = Path(srt_path_str)
  if not srt_path.exists():
    raise RuntimeError(f"ASR backend SRT path missing: {srt_path}")
  # Keep upload job contract stable: transcript endpoint expects SRT in job.whisperx_dir.
  local_srt_path = (job.whisperx_dir / f"{pending.snippet_path.stem}.srt").resolve()
  try:
    if srt_path.resolve() != local_srt_path:
      local_srt_path.parent.mkdir(parents=True, exist_ok=True)
      shutil.copy2(srt_path, local_srt_path)
    else:
      local_srt_path.parent.mkdir(parents=True, exist_ok=True)
  except Exception as e:
    raise RuntimeError(f"Failed to stage SRT into job workspace: {e!r}") from e
  srt_path = local_srt_path

  wx_timings = dict(asr_response.get("timings") or {})
  wx_elapsed = time.monotonic() - pending.wx_t0_mono
  emitted = False
  replay_min_visible_s = 1.15
  replay_visible_phases = {"whisperx_align", "whisperx_diarize"}
  if isinstance(wx_timings, dict):
    order = ("prepare_s", "transcribe_s", "align_s", "diarize_s", "finalize_s")
    map_name = {
      "prepare_s": "whisperx_prepare",
      "transcribe_s": "whisperx_transcribe",
      "align_s": "whisperx_align",
      "diarize_s": "whisperx_diarize",
      "finalize_s": "whisperx_finalize",
    }
    phase_messages = {
      "whisperx_prepare": "Preparing WhisperX…",
      "whisperx_transcribe": "Transcribing…",
      "whisperx_align": "Aligning…",
      "whisperx_diarize": "Diarizing…",
      "whisperx_finalize": "Finalizing…",
    }
    seen: set[str] = set()
    for key in order:
      if key not in wx_timings:
        continue
      out_key = map_name.get(key, "")
      if out_key and out_key in pending.wx_live_timing_keys:
        seen.add(key)
        emitted = True
        continue
      try:
        phase_show_t0 = time.monotonic()
        if out_key:
          pending.progress_start_phase(out_key, phase_messages.get(out_key, out_key), out_key)
        on_wx_phase_timing(out_key or key, float(wx_timings[key]))
        if out_key in replay_visible_phases:
          shown_for = max(0.0, float(time.monotonic() - phase_show_t0))
          pad = max(0.0, replay_min_visible_s - shown_for)
          if pad > 0:
            time.sleep(min(0.5, pad))
        emitted = True
        seen.add(key)
      except Exception:
        pass
    for key in sorted(k for k in wx_timings.keys() if k not in seen and k != "total_s"):
      out_key = map_name.get(key, key)
      if out_key in pending.wx_live_timing_keys:
        emitted = True
        continue
      try:
        phase_show_t0 = time.monotonic()
        if out_key in phase_messages:
          pending.progress_start_phase(out_key, phase_messages.get(out_key, out_key), out_key)
        on_wx_phase_timing(out_key, float(wx_timings[key]))
        if out_key in replay_visible_phases:
          shown_for = max(0.0, float(time.monotonic() - phase_show_t0))
          pad = max(0.0, replay_min_visible_s - shown_for)
          if pad > 0:
            time.sleep(min(0.5, pad))
        emitted = True
      except Exception:
        pass
  if not emitted:
    on_wx_phase_timing("whisperx", wx_elapsed)

  # Phase 31: SRT -> speaker_lines
  orig_stem = Path(pending.orig_filename).stem if pending.orig_filename else "transcript"
  pending.progress_start_phase("postprocess", "Generating speaker_lines…", "postprocess")
  post_t0 = time.monotonic()
  _write_status(job.status_path, phase="postprocess", subphase="speaker_lines", message="Generating speaker_lines…")
  speaker_lines_path, transcript_end_hms = make_speaker_lines_from_srt(job=job, srt_path=srt_path, orig_stem=orig_stem)

  # Phase 32: chunk speaker_lines + manifest
  pending.progress_set_message("Chunking speaker_lines…", status_phase="postprocess")
  _write_status(job.status_path, phase="postprocess", subphase="chunk_speaker_lines", message="Chunking speaker_lines…")
  manifest_path = chunk_speaker_lines(
    job=job,
    speaker_lines_path=speaker_lines_path,
    orig_stem=orig_stem,
    service_cfg=pending.service_cfg,
    transcript_end_hms=transcript_end_hms,
  )
  try:
    manifest_for_count = json.loads(manifest_path.read_text(encoding="utf-8"))
    pending.chunks_count = len(manifest_for_count.get("chunks") or [])
  except Exception:
    pending.chunks_count = 0
  post_elapsed = time.monotonic() - post_t0
  _record_upload_phase_timing(pending=pending, name="postprocess", elapsed_s=post_elapsed)
  pending.progress_finish_phase("postprocess", post_elapsed)

  topics_status = "disabled"
  topics_warning = ""
  if pending.topics_enabled:
    pending.progress_start_phase("llm_topics", "Calling LLM…", "topics")
    topics_t0 = time.monotonic()
    topics_status = "ok"

    def on_topics_progress(message: str) -> None:
      pending.progress_set_message(message, status_phase="topics")

    try:
      llm_hb_stop, llm_hb_thread = _start_progress_heartbeat_thread(pending.progress_heartbeat, interval_s=0.5)
      try:
        run_topics_llm(
          job=job,
          manifest_path=manifest_path,
          orig_stem=orig_stem,
          prompt_id=pending.prompt_id,
          service_cfg=pending.service_cfg,
          on_progress=on_topics_progress,
        )
      finally:
        llm_hb_stop.set()
        llm_hb_thread.join(timeout=1.0)

      manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
      for ch in (manifest.get("chunks") or []):
        idx = int(ch["index"])
        raw_path = job.result_dir / f"{orig_stem}_{pending.prompt_id}_chunk_{idx:04d}_raw.txt"
        parsed_path = job.result_dir / f"{orig_stem}_{pending.prompt_id}_chunk_{idx:04d}.json"
        if raw_path.exists():
          parse_topics_raw_file(raw_txt_path=raw_path, out_json_path=parsed_path)

      report_path = job.result_dir / f"{orig_stem}_{pending.prompt_id}_validation.json"
      validate_all_chunks(
        manifest_path=manifest_path,
        parsed_dir=job.result_dir,
        orig_stem=orig_stem,
        prompt_id=pending.prompt_id,
        out_report_path=report_path,
      )
      report = json.loads(report_path.read_text(encoding="utf-8"))
      if not report.get("is_valid", False):
        topics_status = "validation_failed"
        topics_warning = f"Topics validation failed: {report_path.name}"
        _append_log(
          job.log_path,
          f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] WARN topics_nonfatal validation_failed report={report_path.name}",
        )
        pending.progress_set_message("Topics validation failed; continuing without topics.", status_phase="topics")
      else:
        merged_path = job.result_dir / f"{orig_stem}_{pending.prompt_id}_merged.json"
        merge_topics(
          manifest_path=manifest_path,
          parsed_dir=job.result_dir,
          orig_stem=orig_stem,
          prompt_id=pending.prompt_id,
          out_merged_path=merged_path,
        )
    except Exception as e_topics:
      topics_status = "failed"
      topics_warning = str(e_topics)
      _append_log(
        job.log_path,
        f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] WARN topics_nonfatal error={e_topics!r}",
      )
      pending.progress_set_message("Topics failed; continuing without topics.", status_phase="topics")
    finally:
      topics_elapsed = time.monotonic() - topics_t0
      _record_upload_phase_timing(pending=pending, name="llm_topics", elapsed_s=topics_elapsed)
      pending.progress_finish_phase("llm_topics", topics_elapsed)
  else:
    _write_status(job.status_path, phase="postprocess", subphase="chunk_speaker_lines", message="Topics disabled; skipping.")
    _record_upload_phase_timing(pending=pending, name="llm_topics_skipped", elapsed_s=0.0)
    pending.progress_finish_phase("llm_topics_skipped", 0.0)

  finished_at_utc = _utc_iso()
  actual_total_s = max(0.0, float(time.monotonic() - pending.job_t0_mono))
  final_timings = _format_timings_text(pending.timing_rows, total_s=actual_total_s)
  _write_status(
    job.status_path,
    state="done",
    phase="done",
    progress=1.0,
    finished_at=finished_at_utc,
    message="Done",
    srt_filename=srt_path.name,
    speaker_lines_filename=speaker_lines_path.name,
    speaker_lines_manifest_filename=manifest_path.name,
    timings_text=final_timings,
    progress_mode="predictive_v1",
    eta_total_s=round(actual_total_s, 3),
    eta_remaining_s=0.0,
    elapsed_s=round(actual_total_s, 3),
    eta_confidence=round(float(pending.eta_confidence), 3),
    eta_hints=list(pending.eta_hints),
    topics_status=topics_status,
    topics_warning=topics_warning,
  )

  try:
    host_id_val = _host_id()
    record = {
      "schema_version": "1.0",
      "run_id": job.job_id,
      "job_id": job.job_id,
      "content_hash_sha256": pending.content_hash_sha256,
      "ts_start_utc": pending.job_started_utc,
      "ts_end_utc": finished_at_utc,
      "host_id": host_id_val,
      "worker_instance": _worker_instance(),
      "snippet_seconds": int(pending.snippet_seconds),
      "topics_enabled": bool(pending.topics_enabled),
      "speaker_mode": pending.speaker_mode,
      "chunks_count": int(pending.chunks_count),
      "config_key": _config_key(
        language=pending.language,
        speaker_mode=pending.speaker_mode,
        snippet_seconds=pending.snippet_seconds,
        topics_enabled=pending.topics_enabled,
        prompt_id=pending.prompt_id,
        whisperx_cfg=pending.cfg,
      ),
      "hardware_key": _hardware_key(host_id_val),
      "phase_seconds": _phase_seconds_from_rows(pending.timing_rows),
      "wait_seconds": {},
      "total_seconds": actual_total_s,
      "outcome": "done",
      "error_text": "",
    }
    _, reason = _append_progress_run_if_new_done(record)
    _append_log(job.log_path, f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] WORKER progress_db {reason} runs_path={RUNS_V1_PATH}")
  except Exception as e_db:
    _append_log(job.log_path, f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] WARN progress_db_write_failed error={e_db!r}")

  finish_job(job, ok=True)


def _finalize_upload_job_error(*, pending: _PendingUploadJob, exc: Exception) -> None:
  actual_total_s = max(0.0, float(time.monotonic() - pending.job_t0_mono))
  final_timings = _format_timings_text(pending.timing_rows, total_s=actual_total_s)
  _write_status(
    pending.job.status_path,
    state="error",
    phase="error",
    progress=1.0,
    message=f"Worker error: {exc!r}",
    finished_at=_utc_iso(),
    error=str(exc),
    timings_text=final_timings,
    progress_mode="predictive_v1",
    eta_total_s=round(actual_total_s, 3),
    eta_remaining_s=0.0,
    elapsed_s=round(actual_total_s, 3),
    eta_confidence=round(float(pending.eta_confidence), 3),
    eta_hints=list(pending.eta_hints),
  )
  finish_job(pending.job, ok=False)


def _fail_pending_upload_due_to_feed_reset(*, pending: dict[str, _PendingUploadJob], old_feed_id: str, new_feed_id: str) -> None:
  if not pending:
    return
  err_msg = _completion_feed_reset_error(old_feed_id=old_feed_id, new_feed_id=new_feed_id)
  for pending_job in list(pending.values()):
    try:
      _finalize_upload_job_error(pending=pending_job, exc=RuntimeError(err_msg))
      print(f"Error {pending_job.job.job_id}: {err_msg}")
    except Exception as e:
      _write_status(
        pending_job.job.status_path,
        state="error",
        phase="error",
        progress=1.0,
        finished_at=_utc_iso(),
        message=f"Worker error: {err_msg}",
        error=f"{err_msg} ({e!r})",
      )
      finish_job(pending_job.job, ok=False)
      print(f"Error {pending_job.job.job_id}: {err_msg} | fallback={e!r}")
  pending.clear()


def _upload_submit_worker_loop(
  *,
  submit_queue: "queue.Queue[_UploadSubmitWork | None]",
  event_bus: WorkerEventBus,
  consumer_id: str,
) -> None:
  while True:
    work = submit_queue.get()
    if work is None:
      return
    pending = work.pending
    payload: dict[str, Any] = {
      "mode": "upload",
      "pending": pending,
    }
    try:
      terminal_event = _prepare_upload_job_for_submit(
        pending=pending,
        consumer_id=consumer_id,
      )
      if terminal_event is not None:
        payload["terminal_event"] = dict(terminal_event)
    except Exception as e:
      payload["error"] = str(e)
    event_bus.put(WorkerEventType.SUBMIT_RESULT, payload)


def _handle_upload_submit_result(*, payload: dict[str, Any], pending: dict[str, _PendingUploadJob]) -> bool:
  pending_job = payload.get("pending")
  if pending_job is None:
    return False
  err_msg = str(payload.get("error") or "").strip()
  if err_msg:
    _finalize_upload_job_error(pending=pending_job, exc=RuntimeError(err_msg))
    print(f"Error {pending_job.job.job_id}: {err_msg}")
    return True

  terminal_event = payload.get("terminal_event")
  if isinstance(terminal_event, dict):
    try:
      _finalize_upload_job_terminal(pending=pending_job, event=terminal_event)
      print(f"Done {pending_job.job.job_id} state={str(terminal_event.get('state') or '')}")
    except Exception as e:
      _finalize_upload_job_error(pending=pending_job, exc=e)
      print(f"Error {pending_job.job.job_id}: {e!r}")
    return True

  if not pending_job.request_id:
    _finalize_upload_job_error(pending=pending_job, exc=RuntimeError("ASR submit completed without request_id"))
    print(f"Error {pending_job.job.job_id}: missing_request_id")
    return True

  pending[pending_job.request_id] = pending_job
  return True


def _upload_submit_result_succeeded(payload: dict[str, Any]) -> bool:
  err_msg = str(payload.get("error") or "").strip()
  if err_msg:
    return False
  pending_job = payload.get("pending")
  if pending_job is None:
    return False
  return True


def _run_upload_worker_submit_reap() -> int:
  mode = "upload"
  consumer_id = _worker_consumer_id(mode)
  max_outstanding = _worker_upload_max_outstanding()
  tick_interval_s = max(0.05, float(_worker_coordinator_tick_interval_s()))
  metrics_log_interval_s = max(1.0, float(_worker_metrics_log_interval_s()))
  event_bus = WorkerEventBus()
  inbox_watcher = start_inbox_watcher(
    inbox_dir=INBOX,
    event_bus=event_bus,
    debounce_ms=_worker_inbox_debounce_ms(),
  )
  submit_queue: "queue.Queue[_UploadSubmitWork | None]" = queue.Queue(maxsize=max(1, int(max_outstanding)))
  submit_thread = threading.Thread(
    target=_upload_submit_worker_loop,
    kwargs={
      "submit_queue": submit_queue,
      "event_bus": event_bus,
      "consumer_id": consumer_id,
    },
    name="worker-upload-submit",
    daemon=True,
  )
  submit_thread.start()
  completion_stop = threading.Event()
  completion_thread = threading.Thread(
    target=_completion_stream_worker_loop,
    kwargs={
      "consumer_id": consumer_id,
      "event_bus": event_bus,
      "stop_event": completion_stop,
    },
    name="worker-upload-completion-stream",
    daemon=True,
  )
  completion_thread.start()
  pending_status_poll_interval_s = max(0.2, float(_worker_pending_status_poll_interval_s()))
  pending: dict[str, _PendingUploadJob] = {}
  submitting: dict[str, _UploadSubmitWork] = {}
  counters = _WorkerLoopCounters()
  inbox_dirty = True
  last_pending_status_poll_mono = 0.0
  # TODO(v3-followup): add restart recovery / re-request reconciliation for pending upload jobs.
  print(f"worker_daemon started mode={mode} consumer_id={consumer_id} max_outstanding={max_outstanding}")
  event_bus.put(WorkerEventType.TICK, {"reason": "startup"})
  try:
    while True:
      ev = event_bus.get(timeout_s=tick_interval_s)
      if ev is not None and ev.kind == WorkerEventType.SHUTDOWN:
        break

      did_work = False
      if ev is not None:
        if ev.kind == WorkerEventType.INBOX_DIRTY:
          counters.inbox_events += 1
          inbox_dirty = True
        elif ev.kind == WorkerEventType.SUBMIT_RESULT:
          payload = dict(ev.payload or {})
          if str(payload.get("mode") or "") == "upload":
            pending_job = payload.get("pending")
            job_id = str(getattr(getattr(pending_job, "job", None), "job_id", "") or "")
            if job_id:
              submitting.pop(job_id, None)
            if _upload_submit_result_succeeded(payload):
              counters.submits_succeeded += 1
            else:
              counters.submits_failed += 1
            did_work = _handle_upload_submit_result(payload=payload, pending=pending) or did_work
            inbox_dirty = True
        elif ev.kind == WorkerEventType.COMPLETION_EVENT:
          event = dict((ev.payload or {}).get("event") or {})
          rid = str(event.get("request_id") or "").strip()
          if rid:
            counters.completions_seen += 1
            pending_job = pending.pop(rid, None)
            if pending_job is not None:
              counters.completions_matched += 1
              did_work = True
              inbox_dirty = True
              try:
                _finalize_upload_job_terminal(pending=pending_job, event=event)
                print(f"Done {pending_job.job.job_id} state={str(event.get('state') or '')}")
              except Exception as e:
                _finalize_upload_job_error(pending=pending_job, exc=e)
                print(f"Error {pending_job.job.job_id}: {e!r}")
        elif ev.kind == WorkerEventType.FEED_RESET:
          counters.feed_resets += 1
          old_feed_id = str((ev.payload or {}).get("old_feed_id") or "").strip()
          new_feed_id = str((ev.payload or {}).get("new_feed_id") or "").strip()
          _fail_pending_upload_due_to_feed_reset(
            pending=pending,
            old_feed_id=old_feed_id,
            new_feed_id=new_feed_id,
          )
          did_work = True
          inbox_dirty = True
          print(
            "worker_daemon upload completion_feed_reset "
            f"old_feed_id={old_feed_id[:12]} new_feed_id={new_feed_id[:12]} since_seq_reset=0"
          )
        elif ev.kind == WorkerEventType.TICK:
          reason = str((ev.payload or {}).get("reason") or "").strip().lower()
          if reason == "completion_stream_error":
            counters.sse_reconnects += 1
        elif ev.kind != WorkerEventType.TICK:
          continue

      if pending:
        now_mono = time.monotonic()
        if (now_mono - last_pending_status_poll_mono) >= pending_status_poll_interval_s:
          last_pending_status_poll_mono = now_mono
          status_batch = fetch_remote_pending_status(
            consumer_id=consumer_id,
            request_ids=list(pending.keys()),
            limit=200,
          )
          if status_batch.get("ok", False):
            status_body = dict(status_batch.get("body") or {})
            rows = status_body.get("rows") or []
            if isinstance(rows, list):
              for row in rows:
                if not isinstance(row, dict):
                  continue
                rid = str(row.get("request_id") or "").strip()
                if not rid:
                  continue
                pending_job = pending.get(rid)
                if pending_job is None:
                  continue
                try:
                  _apply_upload_pending_status(pending=pending_job, row=row)
                except Exception:
                  pass
                state = str(row.get("state") or "").strip().lower()
                if not _is_asr_terminal_state(state):
                  continue
                pending.pop(rid, None)
                did_work = True
                inbox_dirty = True
                try:
                  terminal_event = _upload_terminal_event_from_pending_row(request_id=rid, row=row)
                  _finalize_upload_job_terminal(pending=pending_job, event=terminal_event)
                  print(f"Done {pending_job.job.job_id} state={str(terminal_event.get('state') or '')}")
                except Exception as e:
                  _finalize_upload_job_error(pending=pending_job, exc=e)
                  print(f"Error {pending_job.job.job_id}: {e!r}")
        for pending_job in list(pending.values()):
          try:
            pending_job.progress_heartbeat()
          except Exception:
            pass

      if inbox_dirty:
        counters.scheduler_refill_cycles += 1
        while (len(pending) + len(submitting)) < max_outstanding:
          job = claim_next_job(job_kind_filter="upload_audio")
          if not job:
            inbox_dirty = False
            break
          did_work = True
          pending_job = _new_pending_upload_job(job=job)
          try:
            work = _UploadSubmitWork(pending=pending_job)
            submitting[str(job.job_id)] = work
            counters.submits_started += 1
            submit_queue.put(work)
          except Exception as e:
            submitting.pop(str(job.job_id), None)
            _finalize_upload_job_error(pending=pending_job, exc=e)
            print(f"Error {job.job_id}: {e!r}")

      if did_work:
        # Keep coordinator responsive after progress without sleeping out the full tick.
        event_bus.put(WorkerEventType.TICK, {"reason": "followup"})
      _maybe_log_worker_counters(
        mode=mode,
        consumer_id=consumer_id,
        counters=counters,
        pending_count=len(pending),
        submitting_count=len(submitting),
        interval_s=metrics_log_interval_s,
        force=False,
      )
  finally:
    _maybe_log_worker_counters(
      mode=mode,
      consumer_id=consumer_id,
      counters=counters,
      pending_count=len(pending),
      submitting_count=len(submitting),
      interval_s=metrics_log_interval_s,
      force=True,
    )
    completion_stop.set()
    completion_thread.join(timeout=1.0)
    inbox_watcher.close()
    submit_queue.put(None)
    submit_thread.join(timeout=1.0)

  return 0


def main() -> int:
  mode = _worker_mode()
  if mode == "live":
    return _run_live_worker_submit_reap()
  return _run_upload_worker_submit_reap()


if __name__ == "__main__":
  raise SystemExit(main())
