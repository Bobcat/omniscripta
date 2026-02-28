from __future__ import annotations

import os
import time
import wave
from pathlib import Path
from typing import Any, Callable

from asr_contract import ASR_SCHEMA_VERSION, build_error_response
from phase_whisperx import _load_server_config, _run_whisperx_streaming


class AsrBackendError(RuntimeError):
  def __init__(self, code: str, message: str, *, retryable: bool = True, details: dict[str, Any] | None = None) -> None:
    super().__init__(message)
    self.code = str(code)
    self.retryable = bool(retryable)
    self.details = dict(details or {})


def _audio_processed_ms(local_path: Path, request_audio: dict[str, Any]) -> int | None:
  try:
    with wave.open(str(local_path), "rb") as wf:
      rate = int(wf.getframerate() or 0)
      frames = int(wf.getnframes() or 0)
      if rate > 0 and frames >= 0:
        return int(round((frames / float(rate)) * 1000.0))
  except Exception:
    pass
  try:
    val = request_audio.get("duration_ms")
    if val is not None:
      return int(val)
  except Exception:
    pass
  return None


def _wave_frame_count(local_path: Path) -> int | None:
  try:
    with wave.open(str(local_path), "rb") as wf:
      return int(wf.getnframes() or 0)
  except Exception:
    return None


def _is_wave_path(local_path: Path) -> bool:
  return str(local_path.suffix or "").strip().lower() in {".wav", ".wave"}


def _snippet_seconds_from_ms(ms: int | None) -> int:
  if ms is None:
    return 1
  return max(1, int((max(0, int(ms)) + 999) // 1000))


class WhisperxOneShotBackend:
  backend_name = "whisperx"
  runner_kind = "oneshot_subprocess"

  def transcribe(
    self,
    *,
    job: Any,
    request: dict[str, Any],
    on_phase_timing: Callable[[str, float], None] | None = None,
    on_stage_change: Callable[[str], None] | None = None,
    on_heartbeat: Callable[[], None] | None = None,
  ) -> dict[str, Any]:
    req = dict(request or {})
    audio = dict(req.get("audio") or {})
    outputs = dict(req.get("outputs") or {})
    resolved = dict(req.get("resolved_options") or {})
    profile_id = str(req.get("profile_id") or "")
    unsupported_outputs = [k for k in ("text", "segments") if bool(outputs.get(k, False))]
    if unsupported_outputs:
      raise AsrBackendError(
        "ASR_UNSUPPORTED_OUTPUT",
        "WhisperxOneShotBackend does not yet populate requested outputs",
        retryable=False,
        details={"requested_outputs": unsupported_outputs},
      )

    local_path = Path(str(audio.get("local_path") or "")).resolve()
    if not local_path.exists():
      raise AsrBackendError(
        "ASR_INPUT_NOT_FOUND",
        f"ASR input not found: {local_path}",
        retryable=False,
        details={"local_path": str(local_path)},
      )
    try:
      file_size = int(local_path.stat().st_size)
    except Exception:
      file_size = -1
    if file_size == 0:
      raise AsrBackendError(
        "ASR_EMPTY_INPUT",
        "ASR input audio file is empty",
        retryable=False,
        details={"local_path": str(local_path), "bytes": int(file_size)},
      )
    if _is_wave_path(local_path):
      frame_count = _wave_frame_count(local_path)
      if frame_count is None:
        raise AsrBackendError(
          "ASR_INVALID_AUDIO",
          "ASR input audio could not be parsed as WAV",
          retryable=False,
          details={"local_path": str(local_path)},
        )
      if frame_count <= 0:
        raise AsrBackendError(
          "ASR_EMPTY_INPUT",
          "ASR input audio contains no frames",
          retryable=False,
          details={"local_path": str(local_path), "frames": int(frame_count)},
        )

    audio_ms = _audio_processed_ms(local_path, audio)
    snippet_seconds = _snippet_seconds_from_ms(audio_ms)

    cfg = dict(_load_server_config() or {})
    if "align_enabled" in resolved:
      cfg["align_enabled"] = bool(resolved.get("align_enabled"))
    if "beam_size" in resolved and resolved.get("beam_size") is not None:
      try:
        cfg["beam_size"] = max(1, int(resolved.get("beam_size")))
      except Exception:
        pass

    language = str(resolved.get("language") or "en")
    initial_prompt = resolved.get("initial_prompt")
    if initial_prompt is not None:
      initial_prompt = str(initial_prompt)
      if not initial_prompt.strip():
        initial_prompt = None
    speaker_mode = str(resolved.get("speaker_mode") or "none")
    min_speakers = resolved.get("min_speakers")
    max_speakers = resolved.get("max_speakers")

    t0 = time.monotonic()
    srt_path, wx_timings, _wx_emitted_live = _run_whisperx_streaming(
      job=job,
      snippet_path=local_path,
      whisperx_out_dir=job.whisperx_dir,
      snippet_seconds=snippet_seconds,
      language=language,
      speaker_mode=speaker_mode,
      min_speakers=(int(min_speakers) if min_speakers is not None else None),
      max_speakers=(int(max_speakers) if max_speakers is not None else None),
      initial_prompt=(str(initial_prompt) if initial_prompt is not None else None),
      cfg=cfg,
      on_phase_timing=on_phase_timing,
      on_stage_change=on_stage_change,
      on_heartbeat=on_heartbeat,
    )
    total_s = max(0.0, float(time.monotonic() - t0))

    timings = dict(wx_timings or {})
    resp_timings = {
      "total_s": round(total_s, 6),
    }
    if "prepare" in timings:
      resp_timings["prepare_s"] = round(float(timings["prepare"]), 6)
    if "transcribe" in timings:
      resp_timings["transcribe_s"] = round(float(timings["transcribe"]), 6)
    if "align" in timings:
      resp_timings["align_s"] = round(float(timings["align"]), 6)
    if "diarize" in timings:
      resp_timings["diarize_s"] = round(float(timings["diarize"]), 6)
    if "finalize" in timings:
      resp_timings["finalize_s"] = round(float(timings["finalize"]), 6)

    result: dict[str, Any] = {
      "artifacts": {
        "srt_path": str(srt_path),
      },
    }
    if audio_ms is not None:
      result["audio_processed_ms"] = int(audio_ms)
    if bool(outputs.get("srt_inline", False)):
      try:
        result["srt_text"] = srt_path.read_text(encoding="utf-8")
      except Exception:
        pass

    return {
      "schema_version": ASR_SCHEMA_VERSION,
      "request_id": str(req.get("request_id") or ""),
      "ok": True,
      "profile_id": profile_id,
      "resolved_options": resolved,
      "result": result,
      "timings": resp_timings,
      "runtime": {
        "backend": self.backend_name,
        "runner_kind": self.runner_kind,
        "runner_reused": False,
        "device": str(cfg.get("device") or ""),
        "model": str(cfg.get("model") or ""),
      },
      "warnings": [],
    }


def transcribe_with_oneshot_backend(
  *,
  job: Any,
  request: dict[str, Any],
  on_phase_timing: Callable[[str, float], None] | None = None,
  on_stage_change: Callable[[str], None] | None = None,
  on_heartbeat: Callable[[], None] | None = None,
) -> dict[str, Any]:
  backend = WhisperxOneShotBackend()
  try:
    return backend.transcribe(
      job=job,
      request=request,
      on_phase_timing=on_phase_timing,
      on_stage_change=on_stage_change,
      on_heartbeat=on_heartbeat,
    )
  except AsrBackendError as e:
    return build_error_response(
      request=request,
      code=e.code,
      message=str(e),
      retryable=e.retryable,
      details=e.details,
    )
  except Exception as e:
    return build_error_response(
      request=request,
      code="ASR_RUNTIME_FAILURE",
      message=f"ASR backend error: {e!r}",
      retryable=True,
      details={"exc_type": type(e).__name__},
    )


def _env_bool(name: str, default: bool) -> bool:
  raw = str(os.getenv(name, "") or "").strip().lower()
  if not raw:
    return bool(default)
  if raw in {"1", "true", "yes", "on", "y"}:
    return True
  if raw in {"0", "false", "no", "off", "n"}:
    return False
  return bool(default)


def transcribe_live_chunk_backend(*, job: Any, request: dict[str, Any]) -> dict[str, Any]:
  use_warm = _env_bool("TRANSCRIBE_LIVE_CHUNK_WARM_ENABLED", True)
  if use_warm:
    try:
      from asr_client_local import transcribe_with_persistent_local_runner

      resp = transcribe_with_persistent_local_runner(job=job, request=request)
      if isinstance(resp, dict) and bool(resp.get("ok", False)):
        return resp
      if isinstance(resp, dict):
        err = dict(resp.get("error") or {})
        raise RuntimeError(f"{err.get('code') or 'ASR_PERSISTENT_ERROR'}: {err.get('message') or 'persistent runner failed'}")
    except Exception as e:
      fallback = transcribe_with_oneshot_backend(job=job, request=request)
      if bool(fallback.get("ok", False)):
        warnings = list(fallback.get("warnings") or [])
        warnings.append(f"persistent_runner_fallback:{type(e).__name__}")
        fallback["warnings"] = warnings
        runtime = dict(fallback.get("runtime") or {})
        runtime["warm_fallback_used"] = True
        fallback["runtime"] = runtime
      return fallback
  return transcribe_with_oneshot_backend(job=job, request=request)
