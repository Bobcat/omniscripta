from __future__ import annotations

import json
import os
import shutil
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from phase_whisperx import run_whisperx_phase_remote
from worker_status_io import _append_log, _utc_iso, _write_status


def _env_bool(name: str, default: bool) -> bool:
  raw = str(os.getenv(name, "") or "").strip().lower()
  if not raw:
    return bool(default)
  if raw in {"1", "true", "yes", "on", "y"}:
    return True
  if raw in {"0", "false", "no", "off", "n"}:
    return False
  return bool(default)


def _chunk_snippet_seconds(input_path: Path, opts: dict[str, Any]) -> int:
  try:
    t0_ms = int(opts.get("live_chunk_t0_ms", 0) or 0)
    t1_ms = int(opts.get("live_chunk_t1_ms", 0) or 0)
    if t1_ms > t0_ms:
      return max(1, int((max(0, t1_ms - t0_ms) + 999) // 1000))
  except Exception:
    pass

  try:
    with wave.open(str(input_path), "rb") as wf:
      rate = int(wf.getframerate() or 0)
      frames = int(wf.getnframes() or 0)
      if rate > 0 and frames >= 0:
        return max(1, int((frames + rate - 1) // rate))
  except Exception:
    pass

  return 1


def run_live_chunk_job(*, job: Any, job_cfg: dict[str, Any]) -> None:
  job_t0 = time.monotonic()
  timing_rows: list[tuple[str, float]] = []
  job_started_utc = _utc_iso()

  opts = job_cfg.get("options", {}) or {}
  orig_filename = str(job_cfg.get("orig_filename") or "").strip()
  if not orig_filename:
    raise RuntimeError("Missing orig_filename in job config")

  input_path = job.upload_dir / orig_filename
  if not input_path.exists():
    raise RuntimeError(f"Upload missing: {input_path}")

  language = str(opts.get("language", "en") or "en")
  snippet_seconds = _chunk_snippet_seconds(input_path, opts)
  align_enabled = _env_bool("TRANSCRIBE_LIVE_CHUNK_ALIGN_ENABLED", False)
  initial_prompt = str(opts.get("initial_prompt") or "")
  if not initial_prompt.strip():
    initial_prompt = ""
  beam_size = opts.get("beam_size")
  live_lane = "single"

  _write_status(
    job.status_path,
    state="running",
    phase="whisperx_prepare",
    progress=0.0,
    started_at=job_started_utc,
    message="Processing live chunk…",
  )

  try:
    _append_log(
      job.log_path,
      f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] WORKER live_chunk job_cfg={json.dumps(job_cfg, ensure_ascii=False)}",
    )
  except Exception:
    pass

  raw_request = {
    "schema_version": "asr_v1",
    "request_id": str(getattr(job, "job_id", "") or "live_chunk"),
    "profile_id": str(os.getenv("TRANSCRIBE_LIVE_CHUNK_ASR_PROFILE", "live_fast") or "live_fast"),
    "priority": "interactive",
    "audio": {
      "local_path": str(input_path),
      "format": str(input_path.suffix.lstrip(".") or "wav"),
      "sample_rate_hz": 16000,
      "channels": 1,
      "duration_ms": int(snippet_seconds * 1000),
    },
    "options": {
      "language": language,
      "align_enabled": bool(align_enabled),
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
  if initial_prompt:
    raw_request["options"]["initial_prompt"] = initial_prompt
  if beam_size is not None:
    try:
      raw_request["options"]["beam_size"] = max(1, int(beam_size))
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
    resolved_for_log = dict(request.get("resolved_options") or {})
    if "initial_prompt" in resolved_for_log:
      ptxt = str(resolved_for_log.get("initial_prompt") or "")
      resolved_for_log["initial_prompt"] = {
        "chars": len(ptxt),
        "words": len([tok for tok in ptxt.split() if tok]),
      }
    _append_log(
      job.log_path,
      f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] WORKER asr_request profile={request.get('profile_id')} resolved_options={json.dumps(resolved_for_log, ensure_ascii=False, sort_keys=True)}",
    )
  except Exception:
    pass
  response = run_whisperx_phase_remote(request_payload=request)

  if not bool(response.get("ok", False)):
    err = dict(response.get("error") or {})
    raise RuntimeError(f"{err.get('code') or 'ASR_ERROR'}: {err.get('message') or 'ASR request failed'}")

  result_obj = dict(response.get("result") or {})
  artifacts = dict(result_obj.get("artifacts") or {})
  srt_path_str = str(artifacts.get("srt_path") or "").strip()
  if not srt_path_str:
    raise RuntimeError("ASR backend response missing result.artifacts.srt_path")
  srt_path = Path(srt_path_str)
  if not srt_path.exists():
    raise RuntimeError(f"ASR backend SRT path missing: {srt_path}")
  # Keep chunk-job contract stable: live_chunk_transcribe reads SRT from job.whisperx_dir.
  # Remote ASR pool returns an external path, so mirror it into this job workspace.
  local_srt_path = (job.whisperx_dir / srt_path.name).resolve()
  try:
    if srt_path.resolve() != local_srt_path:
      local_srt_path.parent.mkdir(parents=True, exist_ok=True)
      shutil.copy2(srt_path, local_srt_path)
    else:
      local_srt_path.parent.mkdir(parents=True, exist_ok=True)
  except Exception as e:
    raise RuntimeError(f"Failed to stage SRT into job workspace: {e!r}") from e

  timings = dict(response.get("timings") or {})
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
  try:
    _append_log(
      job.log_path,
      f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] WORKER asr_response runtime={json.dumps(runtime_meta, ensure_ascii=False, sort_keys=True)} timings={json.dumps(timings, ensure_ascii=False, sort_keys=True)}",
    )
  except Exception:
    pass

  total_elapsed = max(0.0, float(time.monotonic() - job_t0))
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
    # Chunk jobs do not produce speaker_lines/topics; keep status explicit.
    speaker_lines_filename="",
    speaker_lines_manifest_filename="",
    topics_status="skipped_live_chunk",
    topics_warning="",
    align_enabled=bool(resolved_options.get("align_enabled", align_enabled)),
    asr_profile_id=str(response.get("profile_id") or raw_request.get("profile_id") or ""),
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
  )
