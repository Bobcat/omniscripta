from __future__ import annotations

import argparse
import gc
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


class PersistentWhisperxRunner:
  def __init__(self, cfg: dict[str, Any]) -> None:
    self.cfg = dict(cfg or {})
    self.whisperx = None
    self.torch = None
    self.get_writer = None
    self.asr_model = None
    self.asr_key = None
    self.aligners: dict[tuple[str, str | None], tuple[Any, dict[str, Any]]] = {}
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

  def _asr_cache_key(self, *, language: str) -> tuple[Any, ...]:
    return (
      str(self.cfg.get("model") or "large-v3"),
      str(self.cfg.get("device") or "cuda"),
      str(self.cfg.get("compute_type") or "float16"),
      str(language or "en"),
      int(self.cfg.get("beam_size", 5) or 5),
      int(self.cfg.get("chunk_size", 30) or 30),
    )

  def _ensure_asr_model(self, *, language: str) -> tuple[bool, float]:
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
      language=str(language or "en"),
      asr_options={"beam_size": int(self.cfg.get("beam_size", 5) or 5)},
      vad_options={"chunk_size": int(self.cfg.get("chunk_size", 30) or 30)},
    )
    self.asr_key = key
    return False, max(0.0, float(time.monotonic() - t0))

  def _ensure_aligner(self, *, language: str) -> tuple[Any, dict[str, Any], bool, float]:
    self._import_deps()
    align_model = str(self.cfg.get("align_model") or "").strip() or None
    key = (str(language or "en"), align_model)
    if key in self.aligners:
      aligner, meta = self.aligners[key]
      return aligner, dict(meta or {}), True, 0.0
    t0 = time.monotonic()
    whisperx = self.whisperx
    assert whisperx is not None
    aligner, meta = whisperx.load_align_model(
      str(language or "en"),
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

  def prewarm(self, *, language: str, align_enabled: bool = False) -> dict[str, Any]:
    timings: dict[str, float] = {}
    t0 = time.monotonic()
    model_reused, prepare_s = self._ensure_asr_model(language=str(language or "en"))
    timings["prepare_s"] = round(float(prepare_s), 6)
    aligner_reused: bool | None = None
    if bool(align_enabled):
      _aligner, _meta, aligner_reused, aligner_load_s = self._ensure_aligner(language=str(language or "en"))
      timings["aligner_prepare_s"] = round(float(max(0.0, aligner_load_s)), 6)
    timings["total_s"] = round(float(max(0.0, time.monotonic() - t0)), 6)
    return {
      "ok": True,
      "language": str(language or "en"),
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
    req_id = str(request.get("request_id") or "")
    profile_id = str(request.get("profile_id") or "")
    resolved = dict(request.get("resolved_options") or {})
    outputs = dict(request.get("outputs") or {})
    audio = dict(request.get("audio") or {})
    local_path = Path(str(audio.get("local_path") or ""))
    out_dir_raw = str(work.get("whisperx_out_dir") or "").strip()
    out_dir = Path(out_dir_raw) if out_dir_raw else Path()

    if not local_path.exists():
      return {
        "schema_version": "asr_v1",
        "request_id": req_id,
        "ok": False,
        "profile_id": profile_id,
        "resolved_options": resolved,
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
        "schema_version": "asr_v1",
        "request_id": req_id,
        "ok": False,
        "profile_id": profile_id,
        "resolved_options": resolved,
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
        "schema_version": "asr_v1",
        "request_id": req_id,
        "ok": False,
        "profile_id": profile_id,
        "resolved_options": resolved,
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
        "schema_version": "asr_v1",
        "request_id": req_id,
        "ok": False,
        "profile_id": profile_id,
        "resolved_options": resolved,
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
          "schema_version": "asr_v1",
          "request_id": req_id,
          "ok": False,
          "profile_id": profile_id,
          "resolved_options": resolved,
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
          "schema_version": "asr_v1",
          "request_id": req_id,
          "ok": False,
          "profile_id": profile_id,
          "resolved_options": resolved,
          "error": {
            "code": "ASR_EMPTY_INPUT",
            "message": "ASR input audio contains no frames",
            "retryable": False,
            "details": {"local_path": str(local_path), "frames": int(frame_count)},
          },
          "warnings": [],
        }

    language = str(resolved.get("language") or "en")
    align_enabled = bool(resolved.get("align_enabled", True))
    diarize_enabled = bool(resolved.get("diarize_enabled", False))
    speaker_mode = str(resolved.get("speaker_mode") or "none").strip().lower() or "none"
    min_speakers = resolved.get("min_speakers")
    max_speakers = resolved.get("max_speakers")
    diarize_model = str(self.cfg.get("diarize_model") or "").strip() or None
    initial_prompt = str(resolved.get("initial_prompt") or "").strip() or None
    beam_size_override: int | None = None
    try:
      if resolved.get("beam_size") is not None:
        beam_size_override = max(1, int(resolved.get("beam_size")))
    except Exception:
      beam_size_override = None
    if speaker_mode in {"none", "off", "disabled", "no_speaker", "nospeaker", "no-speaker"}:
      speaker_mode = "none"
    elif speaker_mode not in {"auto", "fixed"}:
      speaker_mode = "auto"
    upload_mode = str(profile_id or "").strip().lower() == "upload_full"

    timings: dict[str, float] = {}
    t_total = time.monotonic()
    self._import_deps()
    whisperx = self.whisperx
    torch = self.torch
    get_writer = self.get_writer
    assert whisperx is not None and torch is not None and get_writer is not None

    if upload_mode:
      # Upload requests run with extra ASR stages (align/diarize). Drop previous
      # aux-model retention first to avoid cumulative VRAM pressure across uploads.
      self._release_aux_models()

    try:
      _write_progress(progress_path, stage="prepare")
      # Prepare/load model lazily and keep it warm across requests.
      model_reused, prepare_s = self._ensure_asr_model(language=language)
      timings["prepare_s"] = round(float(prepare_s), 6)

      t0 = time.monotonic()
      initial_prompt_applied = False
      initial_prompt_unsupported = False
      beam_override_applied = False
      beam_override_unsupported = False
      _write_progress(progress_path, stage="transcribe")
      with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        audio_arr = whisperx.load_audio(str(local_path))
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
        transcribe_kwargs: dict[str, Any] = {
          "batch_size": int(self.cfg.get("batch_size", 3) or 3),
          "chunk_size": int(self.cfg.get("chunk_size", 30) or 30),
          "print_progress": False,
          "verbose": False,
        }
        if upload_mode:
          try:
            upload_batch_size = int(self.cfg.get("upload_batch_size", 4) or 4)
          except Exception:
            upload_batch_size = 4
          if upload_batch_size > 0:
            transcribe_kwargs["batch_size"] = max(1, min(int(transcribe_kwargs["batch_size"]), int(upload_batch_size)))
        result = self.asr_model.transcribe(audio_arr, **transcribe_kwargs)  # type: ignore[union-attr]
      timings["transcribe_s"] = round(max(0.0, float(time.monotonic() - t0)), 6)

      aligned: dict[str, Any]
      t0 = time.monotonic()
      align_language = str(result.get("language") or language or "en")
      aligner_reused = None
      if align_enabled:
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
            "schema_version": "asr_v1",
            "request_id": req_id,
            "ok": False,
            "profile_id": profile_id,
            "resolved_options": resolved,
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

      runtime = {
        "backend": "whisperx",
        "runner_kind": "persistent_local",
        "runner_reused": bool(model_reused),
        "device": str(self.cfg.get("device") or ""),
        "model": str(self.cfg.get("model") or ""),
        "upload_mode": bool(upload_mode),
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

      return {
        "schema_version": "asr_v1",
        "request_id": req_id,
        "ok": True,
        "profile_id": profile_id,
        "resolved_options": resolved,
        "result": result_obj,
        "timings": timings,
        "runtime": runtime,
        "warnings": (
          (["initial_prompt_unsupported_by_asr_pipeline"] if initial_prompt_unsupported else [])
          + (["beam_size_override_unsupported_by_asr_pipeline"] if beam_override_unsupported else [])
        ),
      }
    finally:
      _write_progress(progress_path, stage="done")
      if upload_mode:
        # Keep inter-request VRAM baseline low for uploads and live traffic.
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
    language = str(cmd_obj.get("language") or "en")
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
      "schema_version": "asr_v1",
      "request_id": str(request.get("request_id") or ""),
      "ok": False,
      "profile_id": str(request.get("profile_id") or ""),
      "resolved_options": dict(request.get("resolved_options") or {}),
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
