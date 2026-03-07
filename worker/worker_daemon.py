from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import socket
import sys
import threading
import time
from pathlib import Path
from datetime import datetime, timezone

from queue_fs import claim_next_job, finish_job
from worker_status_io import _append_log, _utc_iso, _write_status
from phase_snipping import _make_snippet
from phase_speaker_lines import make_speaker_lines_from_srt
from phase_chunk_speaker_lines import chunk_speaker_lines
from phase_topics_llm import run_topics_llm
from phase_topics_parse import parse_topics_raw_file
from phase_topics_validate import validate_all_chunks
from phase_topics_merge import merge_topics
from pipeline_live_chunk import run_live_chunk_job
from progress_predictor import build_prediction, phase_order_for_job
from phase_whisperx import run_whisperx_phase_remote

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))
from shared.app_config import get_str, get_float, get_setting

SLEEP_IDLE_SECONDS = get_float("worker.sleep_idle_seconds", 2.0)

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


def main() -> int:
  print("worker_daemon started")
  while True:
    job = claim_next_job()
    if not job:
      time.sleep(SLEEP_IDLE_SECONDS)
      continue

    try:
      job_t0 = time.monotonic()
      timing_rows: list[tuple[str, float]] = []
      job_started_utc = _utc_iso()
      content_hash_sha256 = ""
      chunks_count = 0
      eta_confidence = 0.0
      eta_hints: list[str] = []

      def _noop_start(_phase_key: str, _base_message: str, _status_phase: str) -> None:
        return None

      def _noop_finish(_phase_key: str, _actual_elapsed_s: float) -> None:
        return None

      def _noop_heartbeat() -> None:
        return None

      def _noop_set_message(_base_message: str, *, status_phase: str | None = None) -> None:
        return None

      progress_start_phase = _noop_start
      progress_finish_phase = _noop_finish
      progress_heartbeat = _noop_heartbeat
      progress_set_message = _noop_set_message

      def record_phase_timing(name: str, elapsed_s: float) -> None:
        timing_rows.append((name, max(0.0, float(elapsed_s))))
        txt = _format_timings_text(timing_rows)
        _write_status(job.status_path, timings_text=txt)
        _append_log(
          job.log_path,
          f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] WORKER phase_timing name={name} seconds={max(0.0, float(elapsed_s)):.3f} timings_text={txt}",
        )

      job_cfg = json.loads(job.job_path.read_text(encoding="utf-8"))
      opts = job_cfg.get("options", {}) or {}
      job_kind = _resolve_job_kind(job_cfg)
      if job_kind == "live_chunk":
        run_live_chunk_job(job=job, job_cfg=job_cfg)
        finish_job(job, ok=True)
        print(f"Done {job.job_id}")
        continue
      if job_kind != "upload_audio":
        raise RuntimeError(f"Unsupported job_kind: {job_kind}")

      _write_status(
        job.status_path,
        state="running",
        phase="snipping",
        progress=0.0,
        started_at=job_started_utc,
        message="Starting job…",
      )

      orig_filename = job_cfg.get("orig_filename")
      service_cfg = _load_service_config()
      snip_cfg = (service_cfg.get("snip") or {}) if isinstance(service_cfg, dict) else {}
      default_min = int(snip_cfg.get("minutes_default", 5))
      if opts.get("snippet_seconds") is not None:
        snippet_seconds = int(opts.get("snippet_seconds"))
      else:
        snippet_seconds = int(default_min * 60)
      language = str(opts.get("language", "nl") or "nl")
      speaker_mode = _normalize_speaker_mode(opts.get("speaker_mode", "auto"))
      min_speakers = opts.get("min_speakers")
      max_speakers = opts.get("max_speakers")
      topics_cfg = service_cfg.get("topics", {}) if isinstance(service_cfg, dict) else {}
      topics_enabled = bool(topics_cfg.get("enabled", False))
      prompt_id = str(topics_cfg.get("prompt_id", "topics_v1"))
      host_id_val = _host_id()
      hardware_key_val = _hardware_key(host_id_val)

      input_path = job.upload_dir / orig_filename
      if not input_path.exists():
        raise RuntimeError(f"Upload missing: {input_path}")
      try:
        content_hash_sha256 = _sha256_file(input_path)
        _append_log(job.log_path, f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] WORKER content_hash_sha256={content_hash_sha256}")
      except Exception as e_hash:
        content_hash_sha256 = ""
        _append_log(job.log_path, f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] WARN content_hash_failed error={e_hash!r}")

      prediction = build_prediction(
        runs_path=RUNS_V1_PATH,
        hardware_key=hardware_key_val,
        topics_enabled=topics_enabled,
        speaker_mode=speaker_mode,
        snippet_seconds=snippet_seconds,
      )
      eta_confidence = float(prediction.confidence)
      eta_hints = list(prediction.hints)
      phase_order = phase_order_for_job(topics_enabled=topics_enabled, speaker_mode=speaker_mode)
      progress_start_phase, progress_finish_phase, progress_heartbeat, progress_set_message = _build_progress_tracker(
        status_path=job.status_path,
        phase_order=phase_order,
        phase_expected_s=prediction.phase_expected_s,
        eta_confidence=prediction.confidence,
        eta_hints=eta_hints,
      )

      disp = f"{snippet_seconds//60} min" if snippet_seconds > 0 and (snippet_seconds % 60) == 0 else f"{snippet_seconds} s"
      progress_start_phase("snipping", f"Creating snippet ({disp})…", "snipping")
      snip_t0 = time.monotonic()
      snip_hb_stop, snip_hb_thread = _start_progress_heartbeat_thread(progress_heartbeat, interval_s=0.5)
      try:
        snippet_path = _make_snippet(input_path, job.snippet_dir, seconds=snippet_seconds)
      finally:
        snip_hb_stop.set()
        snip_hb_thread.join(timeout=1.0)
      snip_elapsed = time.monotonic() - snip_t0
      record_phase_timing("snipping", snip_elapsed)
      progress_finish_phase("snipping", snip_elapsed)
      _write_status(
        job.status_path,
        phase="snipping",
        snippet_filename=snippet_path.name,
        message=f"Snippet created: {snippet_path.name}",
      )

      # Keep config_key schema stable even though ASR now runs remotely via pool.
      cfg: dict[str, object] = {}
      _append_log(job.log_path, f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] WORKER service_cfg={json.dumps(service_cfg, ensure_ascii=False)}")

      progress_start_phase("whisperx_prepare", "Preparing WhisperX…", "whisperx_prepare")

      wx_t0 = time.monotonic()
      wx_live_timing_keys: set[str] = set()

      def on_wx_phase_timing(phase_name: str, elapsed_s: float) -> None:
        wx_live_timing_keys.add(str(phase_name or ""))
        record_phase_timing(phase_name, elapsed_s)
        progress_finish_phase(phase_name, elapsed_s)

      raw_asr_request = {
        "schema_version": "asr_v1",
        "request_id": f"{job.job_id}:upload_whisperx",
        "profile_id": "upload_full",
        "audio": {
          "local_path": str(snippet_path.resolve()),
          "duration_ms": int(max(0, snippet_seconds) * 1000),
        },
        "options": {
          "language": language,
          "speaker_mode": speaker_mode,
          "min_speakers": min_speakers,
          "max_speakers": max_speakers,
          "diarize_enabled": bool(speaker_mode != "none"),
          "align_enabled": True,
          "initial_prompt": opts.get("initial_prompt"),
          "timestamps_mode": "segment",
        },
        "context": {
          "source_kind": "upload_audio",
          "job_id": str(job.job_id),
          "orig_filename": str(orig_filename or ""),
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

      asr_hb_stop, asr_hb_thread = _start_progress_heartbeat_thread(progress_heartbeat, interval_s=0.5)
      wx_live_stage_keys: set[str] = {"whisperx_prepare"}
      wx_live_finished_keys: set[str] = set()
      wx_live_current_phase_key: str | None = "whisperx_prepare"
      wx_live_current_started_t = time.monotonic()
      wx_stage_map: dict[str, tuple[str, str]] = {
        "prepare": ("whisperx_prepare", "Preparing WhisperX…"),
        "transcribe": ("whisperx_transcribe", "Transcribing…"),
        "align": ("whisperx_align", "Aligning…"),
        "diarize": ("whisperx_diarize", "Diarizing…"),
        "finalize": ("whisperx_finalize", "Finalizing…"),
      }

      def on_asr_lifecycle_update(lifecycle: dict[str, Any]) -> None:
        nonlocal wx_live_current_phase_key, wx_live_current_started_t
        stage_raw = str((lifecycle or {}).get("stage") or "").strip().lower()
        mapped = wx_stage_map.get(stage_raw)
        if not mapped:
          return
        phase_key, phase_msg = mapped
        now_mono = time.monotonic()
        if wx_live_current_phase_key and wx_live_current_phase_key != phase_key and wx_live_current_phase_key not in wx_live_finished_keys:
          try:
            elapsed_prev = max(0.0, float(now_mono - wx_live_current_started_t))
            on_wx_phase_timing(wx_live_current_phase_key, elapsed_prev)
            wx_live_finished_keys.add(wx_live_current_phase_key)
          except Exception:
            pass
        if phase_key in wx_live_stage_keys and phase_key == wx_live_current_phase_key:
          return
        wx_live_stage_keys.add(phase_key)
        wx_live_current_phase_key = phase_key
        wx_live_current_started_t = now_mono
        try:
          progress_start_phase(phase_key, phase_msg, phase_key)
        except Exception:
          pass
      try:
        asr_response = run_whisperx_phase_remote(
          request_payload=asr_request,
          on_lifecycle_update=on_asr_lifecycle_update,
        )
      finally:
        asr_hb_stop.set()
        asr_hb_thread.join(timeout=1.0)
      if wx_live_current_phase_key and wx_live_current_phase_key not in wx_live_finished_keys:
        try:
          elapsed_last = max(0.0, float(time.monotonic() - wx_live_current_started_t))
          on_wx_phase_timing(wx_live_current_phase_key, elapsed_last)
          wx_live_finished_keys.add(wx_live_current_phase_key)
        except Exception:
          pass
      if not bool(asr_response.get("ok", False)):
        err = dict(asr_response.get("error") or {})
        raise RuntimeError(f"{err.get('code') or 'ASR_ERROR'}: {err.get('message') or 'ASR request failed'}")
      try:
        _append_log(
          job.log_path,
          f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] WORKER asr_response runtime={json.dumps(asr_response.get('runtime') or {}, ensure_ascii=False, sort_keys=True)} timings={json.dumps(asr_response.get('timings') or {}, ensure_ascii=False, sort_keys=True)}",
        )
      except Exception:
        pass

      asr_result = dict(asr_response.get("result") or {})
      asr_artifacts = dict(asr_result.get("artifacts") or {})
      srt_path_str = str(asr_artifacts.get("srt_path") or "").strip()
      if not srt_path_str:
        raise RuntimeError("ASR backend response missing result.artifacts.srt_path")
      srt_path = Path(srt_path_str)
      if not srt_path.exists():
        raise RuntimeError(f"ASR backend SRT path missing: {srt_path}")
      # Keep upload job contract stable: transcript endpoint expects SRT in job.whisperx_dir.
      local_srt_path = (job.whisperx_dir / f"{snippet_path.stem}.srt").resolve()
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
      wx_elapsed = time.monotonic() - wx_t0
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
          if key in wx_timings:
            out_key = map_name.get(key, "")
            if out_key and out_key in wx_live_finished_keys:
              seen.add(key)
              emitted = True
              continue
            if out_key and out_key in wx_live_stage_keys:
              try:
                on_wx_phase_timing(out_key or key, float(wx_timings[key]))
                emitted = True
                seen.add(key)
              except Exception:
                pass
              continue
            if out_key and out_key in wx_live_timing_keys:
              seen.add(key)
              emitted = True
              continue
            try:
              phase_show_t0 = time.monotonic()
              if out_key:
                progress_start_phase(out_key, phase_messages.get(out_key, out_key), out_key)
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
          if out_key in wx_live_finished_keys:
            emitted = True
            continue
          if out_key in wx_live_stage_keys:
            try:
              on_wx_phase_timing(out_key, float(wx_timings[key]))
              emitted = True
            except Exception:
              pass
            continue
          if out_key in wx_live_timing_keys:
            emitted = True
            continue
          try:
            phase_show_t0 = time.monotonic()
            if out_key in phase_messages:
              progress_start_phase(out_key, phase_messages.get(out_key, out_key), out_key)
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
      orig_stem = Path(orig_filename).stem if orig_filename else "transcript"
      progress_start_phase("postprocess", "Generating speaker_lines…", "postprocess")
      post_t0 = time.monotonic()
      _write_status(job.status_path, phase="postprocess", subphase="speaker_lines", message="Generating speaker_lines…")
      speaker_lines_path, transcript_end_hms = make_speaker_lines_from_srt(job=job, srt_path=srt_path, orig_stem=orig_stem)

      # Phase 32: chunk speaker_lines + manifest
      progress_set_message("Chunking speaker_lines…", status_phase="postprocess")
      _write_status(job.status_path, phase="postprocess", subphase="chunk_speaker_lines", message="Chunking speaker_lines…")
      manifest_path = chunk_speaker_lines(job=job, speaker_lines_path=speaker_lines_path, orig_stem=orig_stem, service_cfg=service_cfg, transcript_end_hms=transcript_end_hms)
      try:
        manifest_for_count = json.loads(manifest_path.read_text(encoding="utf-8"))
        chunks_count = len(manifest_for_count.get("chunks") or [])
      except Exception:
        chunks_count = 0
      post_elapsed = time.monotonic() - post_t0
      record_phase_timing("postprocess", post_elapsed)
      progress_finish_phase("postprocess", post_elapsed)


      topics_status = "disabled"
      topics_warning = ""

      # Phase 40: topics (optional; disabled by default)
      if topics_enabled:
        progress_start_phase("llm_topics", "Calling LLM…", "topics")
        topics_t0 = time.monotonic()
        topics_status = "ok"

        def on_topics_progress(message: str) -> None:
          progress_set_message(message, status_phase="topics")

        try:
          # Keep progress moving during potentially long blocking LLM calls.
          llm_hb_stop, llm_hb_thread = _start_progress_heartbeat_thread(progress_heartbeat, interval_s=0.5)
          try:
            # 41) Call LLM (stub for now: only writes payloads; no raw output)
            run_topics_llm(
              job=job,
              manifest_path=manifest_path,
              orig_stem=orig_stem,
              prompt_id=prompt_id,
              service_cfg=service_cfg,
              on_progress=on_topics_progress,
            )
          finally:
            llm_hb_stop.set()
            llm_hb_thread.join(timeout=1.0)

          # 42) Parse (expects *_raw.txt files to exist; since stub doesn't write them, this will be a no-op for now)
          # When PC1 call is implemented, it will write:
          #   <orig_stem>_<prompt_id>_chunk_0001_raw.txt
          # ... and then this loop will produce parsed JSON per chunk.
          manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
          for ch in (manifest.get("chunks") or []):
            idx = int(ch["index"])
            raw_path = job.result_dir / f"{orig_stem}_{prompt_id}_chunk_{idx:04d}_raw.txt"
            parsed_path = job.result_dir / f"{orig_stem}_{prompt_id}_chunk_{idx:04d}.json"
            if raw_path.exists():
              parse_topics_raw_file(raw_txt_path=raw_path, out_json_path=parsed_path)

          # 43) Validate
          report_path = job.result_dir / f"{orig_stem}_{prompt_id}_validation.json"
          validate_all_chunks(
            manifest_path=manifest_path,
            parsed_dir=job.result_dir,
            orig_stem=orig_stem,
            prompt_id=prompt_id,
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
            progress_set_message("Topics validation failed; continuing without topics.", status_phase="topics")
          else:
            # 44) Merge
            merged_path = job.result_dir / f"{orig_stem}_{prompt_id}_merged.json"
            merge_topics(
              manifest_path=manifest_path,
              parsed_dir=job.result_dir,
              orig_stem=orig_stem,
              prompt_id=prompt_id,
              out_merged_path=merged_path,
            )
        except Exception as e_topics:
          topics_status = "failed"
          topics_warning = str(e_topics)
          _append_log(
            job.log_path,
            f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] WARN topics_nonfatal error={e_topics!r}",
          )
          progress_set_message("Topics failed; continuing without topics.", status_phase="topics")
        finally:
          topics_elapsed = time.monotonic() - topics_t0
          record_phase_timing("llm_topics", topics_elapsed)
          progress_finish_phase("llm_topics", topics_elapsed)
      else:
        _write_status(job.status_path, phase="postprocess", subphase="chunk_speaker_lines", message="Topics disabled; skipping.")
        record_phase_timing("llm_topics_skipped", 0.0)
        progress_finish_phase("llm_topics_skipped", 0.0)

      # Finalize
      finished_at_utc = _utc_iso()
      actual_total_s = max(0.0, float(time.monotonic() - job_t0))
      final_timings = _format_timings_text(timing_rows, total_s=actual_total_s)
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
        eta_confidence=round(float(eta_confidence), 3),
        eta_hints=list(eta_hints),
        topics_status=topics_status,
        topics_warning=topics_warning,
      )

      # Append one run record per unique upload content hash (done-only), to avoid DB pollution on repeated test files.
      try:
        host_id_val = _host_id()
        record = {
          "schema_version": "1.0",
          "run_id": job.job_id,
          "job_id": job.job_id,
          "content_hash_sha256": content_hash_sha256,
          "ts_start_utc": job_started_utc,
          "ts_end_utc": finished_at_utc,
          "host_id": host_id_val,
          "worker_instance": _worker_instance(),
          "snippet_seconds": int(snippet_seconds),
          "topics_enabled": bool(topics_enabled),
          "speaker_mode": speaker_mode,
          "chunks_count": int(chunks_count),
          "config_key": _config_key(
            language=language,
            speaker_mode=speaker_mode,
            snippet_seconds=snippet_seconds,
            topics_enabled=topics_enabled,
            prompt_id=prompt_id,
            whisperx_cfg=cfg,
          ),
          "hardware_key": _hardware_key(host_id_val),
          "phase_seconds": _phase_seconds_from_rows(timing_rows),
          "wait_seconds": {},
          "total_seconds": actual_total_s,
          "outcome": "done",
          "error_text": "",
        }
        written, reason = _append_progress_run_if_new_done(record)
        _append_log(job.log_path, f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] WORKER progress_db {reason} runs_path={RUNS_V1_PATH}")
      except Exception as e_db:
        _append_log(job.log_path, f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] WARN progress_db_write_failed error={e_db!r}")

      finish_job(job, ok=True)
      print(f"Done {job.job_id}")

    except Exception as e:
      actual_total_s = max(0.0, float(time.monotonic() - job_t0))
      final_timings = _format_timings_text(timing_rows, total_s=actual_total_s)
      _write_status(
        job.status_path,
        state="error",
        phase="error",
        progress=1.0,
        message=f"Worker error: {e!r}",
        finished_at=_utc_iso(),
        error=str(e),
        timings_text=final_timings,
        progress_mode="predictive_v1",
        eta_total_s=round(actual_total_s, 3),
        eta_remaining_s=0.0,
        elapsed_s=round(actual_total_s, 3),
        eta_confidence=round(float(eta_confidence), 3),
        eta_hints=list(eta_hints),
      )
      finish_job(job, ok=False)
      print(f"Error {job.job_id}: {e!r}")

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
