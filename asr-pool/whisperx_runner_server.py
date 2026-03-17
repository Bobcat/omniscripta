from __future__ import annotations

import argparse
import gc
import inspect
import io
import json
import os
import sys
import time
import wave
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any

from whisperx_runner_imports import _apply_torch_thread_tuning, _as_positive_int, _cleanup_torch


LOW_LATENCY_BACKEND_WHISPERX = "whisperx"
LOW_LATENCY_BACKEND_FASTER_WHISPER_DIRECT = "faster_whisper_direct"
LOW_LATENCY_BACKEND_ALLOWED = {
  LOW_LATENCY_BACKEND_WHISPERX,
  LOW_LATENCY_BACKEND_FASTER_WHISPER_DIRECT,
}


def _write_json_atomic(path: Path, obj: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_suffix(path.suffix + ".tmp")
  tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
  os.replace(tmp, path)


def _now_iso() -> str:
  from datetime import datetime, timezone
  return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
  return json.loads(path.read_text(encoding="utf-8"))


def _write_progress(progress_path: Path | None, *, stage: str) -> None:
  if progress_path is None:
    return
  try:
    _write_json_atomic(
      progress_path,
      {
        "stage": str(stage or "").strip().lower(),
        "ts_utc": _now_iso(),
      },
    )
  except Exception:
    pass


def _audio_processed_ms_from_wave(path: Path, request_audio: dict[str, Any]) -> int | None:
  try:
    with wave.open(str(path), "rb") as wf:
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


def _wave_frame_count(path: Path) -> int | None:
  try:
    with wave.open(str(path), "rb") as wf:
      return int(wf.getnframes() or 0)
  except Exception:
    return None


def _is_wave_path(path: Path) -> bool:
  return str(path.suffix or "").strip().lower() in {".wav", ".wave"}


def _normalize_optional_language(value: Any) -> str | None:
  if value is None:
    return None
  text = str(value).strip()
  return text or None


class PersistentWhisperxRunner:
  def __init__(self, cfg: dict[str, Any]) -> None:
    self.cfg = dict(cfg or {})
    self.whisperx = None
    self.torch = None
    self.get_writer = None
    self.asr_model = None
    self.asr_key = None
    self.aligners: dict[tuple[str | None, str | None], tuple[Any, dict[str, Any]]] = {}
    self.diarizers: dict[tuple[str | None, str], Any] = {}
    self._imported = False

  def _import_deps(self) -> None:
    if self._imported:
      return
    import whisperx  # type: ignore
    import torch  # type: ignore
    from whisperx.utils import get_writer  # type: ignore

    self.whisperx = whisperx
    self.torch = torch
    self.get_writer = get_writer
    self._imported = True

    torch_num_threads = _as_positive_int(self.cfg.get("torch_num_threads"))
    torch_num_interop_threads = _as_positive_int(self.cfg.get("torch_num_interop_threads"))
    try:
      _apply_torch_thread_tuning(
        torch,
        torch_num_threads=torch_num_threads,
        torch_num_interop_threads=torch_num_interop_threads,
      )
    except Exception:
      pass

  def _asr_cache_key(self, *, language: str | None) -> tuple[Any, ...]:
    # Keep one language-agnostic warm ASR model per runner slot.
    # Per-call language hints are handled at transcribe() time.
    return (
      str(self.cfg.get("model") or "large-v3"),
      str(self.cfg.get("device") or "cuda"),
      str(self.cfg.get("compute_type") or "float16"),
      "__language_agnostic__",
      int(self.cfg.get("beam_size", 5) or 5),
      int(self.cfg.get("chunk_size", 30) or 30),
    )

  def _resolve_low_latency_backend(self) -> tuple[str, str]:
    raw = str(self.cfg.get("low_latency_backend") or "").strip().lower()
    if raw in LOW_LATENCY_BACKEND_ALLOWED:
      return raw, "configured"
    raise RuntimeError(f"Invalid low_latency_backend configuration: {raw!r}")

  def _transcribe_direct_faster_whisper(
    self,
    *,
    audio_arr: Any,
    language: str | None,
    initial_prompt: str | None,
    beam_size_override: int | None,
  ) -> tuple[dict[str, Any], dict[str, Any]]:
    if self.asr_model is None:
      raise RuntimeError("ASR model not loaded")
    fw_model = getattr(self.asr_model, "model", None)
    if fw_model is None:
      raise RuntimeError("ASR model has no direct faster-whisper backend")

    requested_kwargs: dict[str, Any] = {
      "condition_on_previous_text": False,
      "vad_filter": True,
      "beam_size": int(beam_size_override if beam_size_override is not None else int(self.cfg.get("beam_size", 5) or 5)),
      "initial_prompt": initial_prompt,
    }
    if language is not None:
      requested_kwargs["language"] = str(language)
    try:
      sig = inspect.signature(fw_model.transcribe)
      accepted = set(sig.parameters.keys())
    except Exception:
      accepted = set()

    call_kwargs: dict[str, Any] = {
      k: v
      for k, v in requested_kwargs.items()
      if v is not None and (not accepted or k in accepted)
    }
    dropped_kwargs = [
      k
      for k, v in requested_kwargs.items()
      if v is not None and accepted and k not in accepted
    ]

    # Experimental direct path note:
    # This bypasses WhisperX's transcribe pipeline conveniences before decode:
    # - no WhisperX VAD preprocess + merge_chunks
    # - no WhisperX tokenizer/language lifecycle in transcribe()
    # - no WhisperX postprocess conventions (it returns richer segment objects that we flatten)
    fw_output = fw_model.transcribe(audio_arr, **call_kwargs)
    if isinstance(fw_output, tuple) and len(fw_output) >= 2:
      fw_segments_iter, fw_info = fw_output[0], fw_output[1]
    else:
      fw_segments_iter, fw_info = fw_output, None

    audio_duration_s = 0.0
    try:
      audio_duration_s = float(len(audio_arr)) / 16000.0
    except Exception:
      audio_duration_s = 0.0

    segments: list[dict[str, Any]] = []
    for seg in fw_segments_iter:
      try:
        raw_text = str(getattr(seg, "text", "") or "").strip()
        if not raw_text:
          continue
        t0 = float(getattr(seg, "start", 0.0) or 0.0)
        t1 = float(getattr(seg, "end", t0) or t0)
        if audio_duration_s > 0.0:
          t0 = max(0.0, min(t0, audio_duration_s))
          t1 = max(t0, min(t1, audio_duration_s))
        segments.append(
          {
            "text": raw_text,
            "start": round(float(t0), 3),
            "end": round(float(t1), 3),
          }
        )
      except Exception:
        continue

    out_language = _normalize_optional_language(language)
    try:
      if fw_info is not None:
        fw_language = _normalize_optional_language(getattr(fw_info, "language", None))
        if fw_language is not None:
          out_language = fw_language
    except Exception:
      pass

    meta = {
      "accepted_kwargs": sorted(call_kwargs.keys()),
      "dropped_kwargs": sorted(dropped_kwargs),
      "segments_returned_count": int(len(segments)),
      "initial_prompt_applied": bool(initial_prompt) and ("initial_prompt" in call_kwargs),
      "initial_prompt_unsupported": bool(initial_prompt) and ("initial_prompt" not in call_kwargs),
      "beam_size_override_applied": (beam_size_override is not None) and ("beam_size" in call_kwargs),
      "beam_size_override_unsupported": (beam_size_override is not None) and ("beam_size" not in call_kwargs),
    }
    return {"segments": segments, "language": out_language}, meta

  def _ensure_asr_model(self, *, language: str | None) -> tuple[bool, float]:
    self._import_deps()
    key = self._asr_cache_key(language=language)
    if self.asr_model is not None and self.asr_key == key:
      return True, 0.0
    t0 = time.monotonic()
    if self.asr_model is not None:
      try:
        del self.asr_model
      except Exception:
        pass
      self.asr_model = None
      self.asr_key = None
      try:
        _cleanup_torch(self.torch)
      except Exception:
        pass

    whisperx = self.whisperx
    assert whisperx is not None
    self.asr_model = whisperx.load_model(
      str(self.cfg.get("model", "large-v3") or "large-v3"),
      device=str(self.cfg.get("device", "cuda") or "cuda"),
      compute_type=str(self.cfg.get("compute_type", "float16") or "float16"),
      language=None,
      asr_options={"beam_size": int(self.cfg.get("beam_size", 5) or 5)},
      vad_options={"chunk_size": int(self.cfg.get("chunk_size", 30) or 30)},
    )
    self.asr_key = key
    return False, max(0.0, float(time.monotonic() - t0))

  def _ensure_aligner(self, *, language: str) -> tuple[Any, dict[str, Any], bool, float]:
    self._import_deps()
    align_model = str(self.cfg.get("align_model") or "").strip() or None
    key = (language, align_model)
    if key in self.aligners:
      aligner, meta = self.aligners[key]
      return aligner, dict(meta or {}), True, 0.0
    t0 = time.monotonic()
    whisperx = self.whisperx
    assert whisperx is not None
    aligner, meta = whisperx.load_align_model(
      language,
      str(self.cfg.get("device", "cuda") or "cuda"),
      model_name=align_model,
    )
    self.aligners[key] = (aligner, dict(meta or {}))
    return aligner, dict(meta or {}), False, max(0.0, float(time.monotonic() - t0))

  def _ensure_diarizer(self, *, diarize_model: str | None) -> tuple[Any, bool, float]:
    self._import_deps()
    device = str(self.cfg.get("device", "cuda") or "cuda")
    key = (diarize_model, device)
    if key in self.diarizers:
      return self.diarizers[key], True, 0.0
    t0 = time.monotonic()
    from whisperx.diarize import DiarizationPipeline  # type: ignore

    diarize_pipe = DiarizationPipeline(
      model_name=diarize_model,
      use_auth_token=os.getenv("HF_TOKEN"),
      device=device,
    )
    self.diarizers[key] = diarize_pipe
    return diarize_pipe, False, max(0.0, float(time.monotonic() - t0))

  def _release_aux_models(self) -> None:
    for _k, (aligner, _meta) in list(self.aligners.items()):
      try:
        to_cpu = getattr(aligner, "cpu", None)
        if callable(to_cpu):
          try:
            to_cpu()
          except Exception:
            pass
        to_dev = getattr(aligner, "to", None)
        if callable(to_dev):
          try:
            to_dev("cpu")
          except Exception:
            pass
        del aligner
      except Exception:
        pass
    self.aligners.clear()
    for _k, diarize_pipe in list(self.diarizers.items()):
      try:
        model_obj = getattr(diarize_pipe, "model", None)
        if model_obj is not None:
          to_cpu = getattr(model_obj, "cpu", None)
          if callable(to_cpu):
            try:
              to_cpu()
            except Exception:
              pass
          to_dev = getattr(model_obj, "to", None)
          if callable(to_dev):
            try:
              to_dev("cpu")
            except Exception:
              pass
        pipe_to = getattr(diarize_pipe, "to", None)
        if callable(pipe_to):
          try:
            pipe_to("cpu")
          except Exception:
            pass
        del diarize_pipe
      except Exception:
        pass
    self.diarizers.clear()
    try:
      if self.torch is not None:
        _cleanup_torch(self.torch)
    except Exception:
      pass
    try:
      gc.collect()
    except Exception:
      pass

  def prewarm(self, *, language: str | None, align_enabled: bool = False) -> dict[str, Any]:
    timings: dict[str, float] = {}
    t0 = time.monotonic()
    resolved_language = _normalize_optional_language(language)
    model_reused, prepare_s = self._ensure_asr_model(language=resolved_language)
    timings["prepare_s"] = round(float(prepare_s), 6)
    aligner_reused: bool | None = None
    if bool(align_enabled) and resolved_language is not None:
      _aligner, _meta, aligner_reused, aligner_load_s = self._ensure_aligner(language=resolved_language)
      timings["aligner_prepare_s"] = round(float(max(0.0, aligner_load_s)), 6)
    timings["total_s"] = round(float(max(0.0, time.monotonic() - t0)), 6)
    return {
      "ok": True,
      "language": resolved_language,
      "align_enabled": bool(align_enabled),
      "runner_reused": bool(model_reused),
      "aligner_reused": (None if aligner_reused is None else bool(aligner_reused)),
      "timings": timings,
      "runtime": {
        "backend": "whisperx",
        "runner_kind": "persistent_local",
        "device": str(self.cfg.get("device") or ""),
        "model": str(self.cfg.get("model") or ""),
      },
    }

  def transcribe(self, envelope: dict[str, Any], *, progress_path: Path | None = None) -> dict[str, Any]:
    request = dict(envelope.get("request") or {})
    work = dict(envelope.get("work") or {})
    req_schema_version = str(request.get("schema_version") or "").strip().lower()
    req_id = str(request.get("request_id") or "")
    effective_options = dict(request.get("effective_options") or {})
    outputs = dict(request.get("outputs") or {})
    audio = dict(request.get("audio") or {})
    local_path = Path(str(audio.get("local_path") or ""))
    out_dir_raw = str(work.get("whisperx_out_dir") or "").strip()
    out_dir = Path(out_dir_raw) if out_dir_raw else Path()

    if req_schema_version != "asr_v2":
      return {
        "schema_version": "asr_v2",
        "request_id": req_id,
        "ok": False,
        "effective_options": effective_options,
        "error": {
          "code": "ASR_SCHEMA_UNSUPPORTED",
          "message": f"Unsupported schema_version: {req_schema_version or '<missing>'}",
          "retryable": False,
          "details": {"supported": ["asr_v2"]},
        },
        "warnings": [],
      }

    if not local_path.exists():
      return {
        "schema_version": "asr_v2",
        "request_id": req_id,
        "ok": False,
        "effective_options": effective_options,
        "error": {
          "code": "ASR_INPUT_NOT_FOUND",
          "message": f"ASR input not found: {local_path}",
          "retryable": False,
          "details": {"local_path": str(local_path)},
        },
        "warnings": [],
      }
    unsupported_outputs = [k for k in ("text", "segments") if bool(outputs.get(k, False))]
    if unsupported_outputs:
      return {
        "schema_version": "asr_v2",
        "request_id": req_id,
        "ok": False,
        "effective_options": effective_options,
        "error": {
          "code": "ASR_UNSUPPORTED_OUTPUT",
          "message": "persistent ASR pool runner does not populate requested outputs",
          "retryable": False,
          "details": {"requested_outputs": unsupported_outputs},
        },
        "warnings": [],
      }
    if not out_dir_raw:
      return {
        "schema_version": "asr_v2",
        "request_id": req_id,
        "ok": False,
        "effective_options": effective_options,
        "error": {
          "code": "ASR_OUTPUT_DIR_REQUIRED",
          "message": "Missing work.whisperx_out_dir",
          "retryable": False,
          "details": {},
        },
        "warnings": [],
      }
    try:
      file_size = int(local_path.stat().st_size)
    except Exception:
      file_size = -1
    if file_size == 0:
      return {
        "schema_version": "asr_v2",
        "request_id": req_id,
        "ok": False,
        "effective_options": effective_options,
        "error": {
          "code": "ASR_EMPTY_INPUT",
          "message": "ASR input audio file is empty",
          "retryable": False,
          "details": {"local_path": str(local_path), "bytes": int(file_size)},
        },
        "warnings": [],
      }
    if _is_wave_path(local_path):
      frame_count = _wave_frame_count(local_path)
      if frame_count is None:
        return {
          "schema_version": "asr_v2",
          "request_id": req_id,
          "ok": False,
          "effective_options": effective_options,
          "error": {
            "code": "ASR_INVALID_AUDIO",
            "message": "ASR input audio could not be parsed as WAV",
            "retryable": False,
            "details": {"local_path": str(local_path)},
          },
          "warnings": [],
        }
      if frame_count <= 0:
        return {
          "schema_version": "asr_v2",
          "request_id": req_id,
          "ok": False,
          "effective_options": effective_options,
          "error": {
            "code": "ASR_EMPTY_INPUT",
            "message": "ASR input audio contains no frames",
            "retryable": False,
            "details": {"local_path": str(local_path), "frames": int(frame_count)},
          },
          "warnings": [],
        }

    language = _normalize_optional_language(effective_options.get("language"))
    align_enabled = bool(effective_options.get("align_enabled", True))
    diarize_enabled = bool(effective_options.get("diarize_enabled", False))
    speaker_mode = str(effective_options.get("speaker_mode") or "none").strip().lower() or "none"
    min_speakers = effective_options.get("min_speakers")
    max_speakers = effective_options.get("max_speakers")
    diarize_model = str(self.cfg.get("diarize_model") or "").strip() or None
    initial_prompt = str(effective_options.get("initial_prompt") or "").strip() or None
    beam_size_override: int | None = None
    try:
      if effective_options.get("beam_size") is not None:
        beam_size_override = max(1, int(effective_options.get("beam_size")))
    except Exception:
      beam_size_override = None
    if speaker_mode in {"none", "off", "disabled", "no_speaker", "nospeaker", "no-speaker"}:
      speaker_mode = "none"
    elif speaker_mode not in {"auto", "fixed"}:
      speaker_mode = "auto"
    latency_mode = str(effective_options.get("latency_mode") or "default").strip().lower()
    if latency_mode not in {"low", "default"}:
      latency_mode = "default"
    low_latency_mode = latency_mode == "low"
    aux_sensitive_mode = bool(align_enabled) or bool(diarize_enabled and speaker_mode != "none")
    configured_low_latency_backend, low_latency_backend_cfg_reason = self._resolve_low_latency_backend()
    if low_latency_mode and configured_low_latency_backend == LOW_LATENCY_BACKEND_FASTER_WHISPER_DIRECT:
      selected_low_latency_backend = LOW_LATENCY_BACKEND_FASTER_WHISPER_DIRECT
      selected_low_latency_backend_reason = low_latency_backend_cfg_reason
    elif low_latency_mode:
      selected_low_latency_backend = LOW_LATENCY_BACKEND_WHISPERX
      selected_low_latency_backend_reason = low_latency_backend_cfg_reason
    else:
      selected_low_latency_backend = LOW_LATENCY_BACKEND_WHISPERX
      selected_low_latency_backend_reason = "latency_mode_default"

    timings: dict[str, float] = {}
    t_total = time.monotonic()
    self._import_deps()
    whisperx = self.whisperx
    torch = self.torch
    get_writer = self.get_writer
    assert whisperx is not None and torch is not None and get_writer is not None

    if aux_sensitive_mode:
      # Requests that use align/diarize can retain extra aux-model VRAM.
      # Release before request to keep a stable baseline.
      self._release_aux_models()

    try:
      _write_progress(progress_path, stage="prepare")
      # Keep one warm ASR model for low-latency requests regardless of per-call language hints.
      # Per-call language is still passed to transcribe(); this only avoids model
      # cache churn when sessions mix explicit language and auto-detect.
      model_cache_language = None if low_latency_mode else language
      # Prepare/load model lazily and keep it warm across requests.
      model_reused, prepare_s = self._ensure_asr_model(language=model_cache_language)
      timings["prepare_s"] = round(float(prepare_s), 6)

      t0 = time.monotonic()
      initial_prompt_applied = False
      initial_prompt_unsupported = False
      beam_override_applied = False
      beam_override_unsupported = False
      direct_backend_meta: dict[str, Any] = {}
      align_skipped_missing_language = False
      transcribe_kwargs: dict[str, Any] = {
        "batch_size": int(self.cfg.get("batch_size", 3) or 3),
        "chunk_size": int(self.cfg.get("chunk_size", 30) or 30),
        "print_progress": False,
        "verbose": False,
      }
      if language is not None:
        # Keep model cache language-agnostic, but still pass per-call language hints.
        transcribe_kwargs["language"] = str(language)
      if not low_latency_mode:
        try:
          batch_size_default_cap = int(self.cfg.get("batch_size_default_cap", 4) or 4)
        except Exception:
          batch_size_default_cap = 4
        if batch_size_default_cap > 0:
          transcribe_kwargs["batch_size"] = max(
            1,
            min(int(transcribe_kwargs["batch_size"]), int(batch_size_default_cap)),
          )
      else:
        # Low-latency path: use smaller chunk_size for better responsiveness.
        transcribe_kwargs["chunk_size"] = int(self.cfg.get("chunk_size_low_latency", 10) or 10)

      _write_progress(progress_path, stage="transcribe")
      transcribe_call_started_utc: str | None = None
      transcribe_call_finished_utc: str | None = None
      transcribe_call_duration_s: float | None = None
      with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        audio_arr = whisperx.load_audio(str(local_path))
        if selected_low_latency_backend == LOW_LATENCY_BACKEND_FASTER_WHISPER_DIRECT:
          transcribe_call_started_utc = _now_iso()
          transcribe_call_t0 = time.monotonic()
          try:
            result, direct_backend_meta = self._transcribe_direct_faster_whisper(
              audio_arr=audio_arr,
              language=language,
              initial_prompt=initial_prompt,
              beam_size_override=beam_size_override,
            )
          finally:
            transcribe_call_finished_utc = _now_iso()
            transcribe_call_duration_s = round(max(0.0, float(time.monotonic() - transcribe_call_t0)), 6)
          initial_prompt_applied = bool(direct_backend_meta.get("initial_prompt_applied"))
          initial_prompt_unsupported = bool(direct_backend_meta.get("initial_prompt_unsupported"))
          beam_override_applied = bool(direct_backend_meta.get("beam_size_override_applied"))
          beam_override_unsupported = bool(direct_backend_meta.get("beam_size_override_unsupported"))
        else:
          try:
            current_opts = getattr(self.asr_model, "options", None)  # type: ignore[union-attr]
            if current_opts is not None:
              replace_kwargs: dict[str, Any] = {"initial_prompt": initial_prompt}
              if beam_size_override is not None:
                replace_kwargs["beam_size"] = int(beam_size_override)
              self.asr_model.options = dataclass_replace(current_opts, **replace_kwargs)  # type: ignore[union-attr]
              initial_prompt_applied = bool(initial_prompt is not None)
              beam_override_applied = bool(beam_size_override is not None)
            elif initial_prompt is not None:
              initial_prompt_unsupported = True
          except Exception:
            if initial_prompt is not None:
              initial_prompt_unsupported = True
            if beam_size_override is not None:
              beam_override_unsupported = True
          transcribe_call_started_utc = _now_iso()
          transcribe_call_t0 = time.monotonic()
          try:
            result = self.asr_model.transcribe(audio_arr, **transcribe_kwargs)  # type: ignore[union-attr]
          finally:
            transcribe_call_finished_utc = _now_iso()
            transcribe_call_duration_s = round(max(0.0, float(time.monotonic() - transcribe_call_t0)), 6)
        # Debug: log segment details for confidence analysis
        try:
          segments = result.get("segments") or []
          import json as _json
          for idx, seg in enumerate(segments[:10]):
            seg_text = str(seg.get("text") or "").strip()
            seg_start = float(seg.get("start") or 0)
            seg_end = float(seg.get("end") or 0)
            seg_dur = round(seg_end - seg_start, 3)
            print(f"INFO seg_{idx} dur={seg_dur}s text={_json.dumps(seg_text, ensure_ascii=False)}", flush=True)
        except Exception:
          pass
      if (
        transcribe_call_started_utc is not None
        and transcribe_call_finished_utc is not None
        and transcribe_call_duration_s is not None
      ):
        print(
          "ASR_TRANSCRIBE_CALL_TIMING "
          + json.dumps(
            {
              "request_id": req_id,
              "backend": (
                "faster_whisper_direct"
                if selected_low_latency_backend == LOW_LATENCY_BACKEND_FASTER_WHISPER_DIRECT
                else "whisperx"
              ),
              "start_utc": str(transcribe_call_started_utc),
              "end_utc": str(transcribe_call_finished_utc),
              "duration_s": float(transcribe_call_duration_s),
            },
            ensure_ascii=False,
          ),
          flush=True,
        )
      timings["transcribe_s"] = round(max(0.0, float(time.monotonic() - t0)), 6)
      if transcribe_call_duration_s is not None:
        timings["transcribe_call_s"] = round(max(0.0, float(transcribe_call_duration_s)), 6)

      aligned: dict[str, Any]
      t0 = time.monotonic()
      align_language = _normalize_optional_language(result.get("language"))
      if align_language is None:
        align_language = language
      aligner_reused = None
      if align_enabled and align_language is not None:
        _write_progress(progress_path, stage="align")
        aligner, align_meta, aligner_reused, aligner_load_s = self._ensure_aligner(language=align_language)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
          if aligner is not None and len(result.get("segments") or []) > 0:
            aligned = whisperx.align(
              result["segments"],
              aligner,
              align_meta,
              audio_arr,
              str(self.cfg.get("device", "cuda") or "cuda"),
              return_char_alignments=False,
              print_progress=False,
            )
          else:
            aligned = {"segments": result.get("segments") or []}
        aligned["language"] = str((align_meta or {}).get("language") or align_language)
        timings["align_s"] = round(max(0.0, float(time.monotonic() - t0)), 6)
        if aligner_load_s > 0:
          timings["aligner_load_s"] = round(float(aligner_load_s), 6)
      else:
        aligned = {"segments": result.get("segments") or []}
        aligned["language"] = align_language
        if align_enabled and align_language is None:
          align_skipped_missing_language = True
        timings["align_s"] = round(max(0.0, float(time.monotonic() - t0)), 6)

      # Diarization (warm runner path supports it).
      diarize_applied = False
      diarizer_reused: bool | None = None
      t0 = time.monotonic()
      if diarize_enabled and speaker_mode != "none":
        _write_progress(progress_path, stage="diarize")
        diarize_kwargs: dict[str, Any] = {}
        if speaker_mode == "fixed":
          if min_speakers is not None:
            try:
              diarize_kwargs["min_speakers"] = int(min_speakers)
            except Exception:
              pass
          if max_speakers is not None:
            try:
              diarize_kwargs["max_speakers"] = int(max_speakers)
            except Exception:
              pass
        try:
          diarize_pipe, diarizer_reused, diarizer_load_s = self._ensure_diarizer(diarize_model=diarize_model)
          with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            diarize_df = diarize_pipe(str(local_path), **diarize_kwargs)
            aligned = whisperx.assign_word_speakers(diarize_df, aligned)
          diarize_applied = True
          if diarizer_load_s > 0:
            timings["diarizer_load_s"] = round(float(diarizer_load_s), 6)
        except Exception:
          diarize_applied = False
      timings["diarize_s"] = round(max(0.0, float(time.monotonic() - t0)), 6)

      _write_progress(progress_path, stage="finalize")
      t0 = time.monotonic()
      out_dir.mkdir(parents=True, exist_ok=True)
      writer = get_writer("srt", str(out_dir))
      writer_args = {"highlight_words": False, "max_line_count": None, "max_line_width": None}
      with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        writer(aligned, str(local_path), writer_args)
      timings["finalize_s"] = round(max(0.0, float(time.monotonic() - t0)), 6)

      srt_path = out_dir / f"{local_path.stem}.srt"
      if not srt_path.exists():
        srts = sorted(out_dir.glob("*.srt"), key=lambda p: p.stat().st_mtime)
        if not srts:
          return {
            "schema_version": "asr_v2",
            "request_id": req_id,
            "ok": False,
            "effective_options": effective_options,
            "error": {
              "code": "ASR_OUTPUT_MISSING",
              "message": f"No .srt produced in {out_dir}",
              "retryable": True,
              "details": {"out_dir": str(out_dir)},
            },
            "warnings": [],
          }
        srt_path = srts[-1]

      timings["total_s"] = round(max(0.0, float(time.monotonic() - t_total)), 6)
      audio_ms = _audio_processed_ms_from_wave(local_path, audio)

      result_obj: dict[str, Any] = {
        "artifacts": {
          "srt_path": str(srt_path),
        },
      }
      if audio_ms is not None:
        result_obj["audio_processed_ms"] = int(audio_ms)
      if bool(outputs.get("srt_inline", False)):
        try:
          result_obj["srt_text"] = srt_path.read_text(encoding="utf-8")
        except Exception:
          pass

      segments_returned_count = int(len(result.get("segments") or []))
      runtime = {
        "backend": (
          "faster_whisper_direct"
          if selected_low_latency_backend == LOW_LATENCY_BACKEND_FASTER_WHISPER_DIRECT
          else "whisperx"
        ),
        "runner_kind": "persistent_local",
        "runner_reused": bool(model_reused),
        "device": str(self.cfg.get("device") or ""),
        "model": str(self.cfg.get("model") or ""),
        "latency_mode": str(latency_mode),
        "low_latency_backend_selected": str(selected_low_latency_backend),
        "low_latency_backend_reason": str(selected_low_latency_backend_reason),
        "segments_returned_count": int(segments_returned_count),
        "effective_batch_size": int(transcribe_kwargs.get("batch_size") or 0),
        "diarize_applied": bool(diarize_applied),
        "initial_prompt_applied": bool(initial_prompt_applied),
        "beam_size_override_applied": bool(beam_override_applied),
        "beam_size_override": (int(beam_size_override) if beam_size_override is not None else None),
      }
      if aligner_reused is not None:
        runtime["aligner_reused"] = bool(aligner_reused)
      if diarizer_reused is not None:
        runtime["diarizer_reused"] = bool(diarizer_reused)
      if direct_backend_meta:
        runtime["direct_backend_meta"] = dict(direct_backend_meta)

      warnings: list[str] = []
      if align_skipped_missing_language:
        warnings.append("align_skipped_missing_language")
      if initial_prompt_unsupported:
        warnings.append("initial_prompt_unsupported_by_asr_pipeline")
      if beam_override_unsupported:
        warnings.append("beam_size_override_unsupported_by_asr_pipeline")
      if selected_low_latency_backend == LOW_LATENCY_BACKEND_FASTER_WHISPER_DIRECT:
        warnings.append("low_latency_backend_faster_whisper_direct_experimental")

      return {
        "schema_version": "asr_v2",
        "request_id": req_id,
        "ok": True,
        "effective_options": effective_options,
        "result": result_obj,
        "timings": timings,
        "runtime": runtime,
        "warnings": warnings,
      }
    finally:
      _write_progress(progress_path, stage="done")
      if aux_sensitive_mode:
        # Keep inter-request VRAM baseline low when auxiliary models are used.
        self._release_aux_models()

  def shutdown(self) -> None:
    try:
      if self.asr_model is not None:
        del self.asr_model
    except Exception:
      pass
    self.asr_model = None
    self.asr_key = None
    self._release_aux_models()


def _handle_command(runner: PersistentWhisperxRunner, cmd_obj: dict[str, Any]) -> bool:
  cmd = str(cmd_obj.get("cmd") or "").strip().lower()
  if cmd == "shutdown":
    return False
  if cmd == "prewarm":
    response_path = Path(str(cmd_obj.get("response_path") or ""))
    if not response_path:
      return True
    language = _normalize_optional_language(cmd_obj.get("language"))
    align_enabled = bool(cmd_obj.get("align_enabled", False))
    try:
      out = runner.prewarm(language=language, align_enabled=align_enabled)
    except Exception as e:
      out = {
        "ok": False,
        "error": {
          "code": "ASR_PERSISTENT_PREWARM_FAILURE",
          "message": f"Persistent prewarm error: {e!r}",
          "retryable": True,
          "details": {"exc_type": type(e).__name__},
        },
      }
    try:
      _write_json_atomic(response_path, out)
    except Exception:
      pass
    return True
  if cmd != "transcribe":
    return True
  payload_path = Path(str(cmd_obj.get("payload_path") or ""))
  response_path = Path(str(cmd_obj.get("response_path") or ""))
  if not payload_path or not response_path:
    return True
  try:
    envelope = _read_json(payload_path)
    progress_path_raw = str(cmd_obj.get("progress_path") or "").strip()
    progress_path = Path(progress_path_raw) if progress_path_raw else None
    response = runner.transcribe(envelope, progress_path=progress_path)
  except Exception as e:
    request = {}
    try:
      envelope = _read_json(payload_path)
      request = dict(envelope.get("request") or {})
    except Exception:
      request = {}
    response = {
      "schema_version": "asr_v2",
      "request_id": str(request.get("request_id") or ""),
      "ok": False,
      "effective_options": dict(request.get("effective_options") or {}),
      "error": {
        "code": "ASR_PERSISTENT_SERVER_FAILURE",
        "message": f"Persistent server error: {e!r}",
        "retryable": True,
        "details": {"exc_type": type(e).__name__},
      },
      "warnings": [],
    }
  try:
    _write_json_atomic(response_path, response)
  except Exception:
    pass
  return True


def main() -> int:
  parser = argparse.ArgumentParser(description="Persistent WhisperX runner for local ASR requests")
  parser.add_argument("--init-json", required=True)
  ns = parser.parse_args()
  init_obj = _read_json(Path(ns.init_json))
  cfg = dict(init_obj.get("cfg") or {})
  runner = PersistentWhisperxRunner(cfg=cfg)

  for raw in sys.stdin:
    line = str(raw or "").strip()
    if not line:
      continue
    try:
      cmd_obj = json.loads(line)
    except Exception:
      continue
    if not _handle_command(runner, cmd_obj):
      break

  try:
    runner.shutdown()
  except Exception:
    pass
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
