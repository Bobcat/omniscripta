from __future__ import annotations

import json
import os
import re
import selectors
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from worker_status_io import _StatusEmitter, _append_log, _append_wx


# WhisperX progress mapping (overall job progress 0.0..1.0)
P_SNIP_DONE = 0.05
P_WHISPERX_DONE = 0.80  # overall job progress when WhisperX is complete
R_VAD = (0.05, 0.08)
R_TRANSCRIBE = (0.08, 0.68)
R_ALIGN = (0.68, 0.74)
R_DIARIZE = (0.74, 0.79)
R_FINALIZE = (0.79, 0.80)

# Paths
SERVER_CFG_PATH = Path("/srv/transcribe/config/whisperx.json")
WHISPERX_ENTER_SH = Path.home() / "whisperx" / "enter.sh"
WHISPERX_RUN2_SH = Path.home() / "whisperx" / "run_whisperx2.sh"


TIMESTAMP_INFO_RE = re.compile(
  r"^(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}) - (?P<mod>[^ ]+) - (?P<lvl>[A-Z]+) - (?P<msg>.*)$"
)
TRANSCRIPT_RE = re.compile(
  r"^Transcript:\s*\[(?P<a>\d+(?:\.\d+)?)\s*-->\s*(?P<b>\d+(?:\.\d+)?)\]"
)


def _load_server_config() -> dict[str, Any]:
  # Safe defaults for your environment; overridden by whisperx.json when present.
  cfg: dict[str, Any] = {
    "model": "large-v3",
    "device": "cuda",
    "compute_type": "float16",
    "batch_size": 3,
    "chunk_size": 30,
    "beam_size": 5,
    "align_model": "",
    "keep_raw_whisperx_log": False,
  }
  if SERVER_CFG_PATH.exists():
    try:
      on_disk = json.loads(SERVER_CFG_PATH.read_text(encoding="utf-8"))
      if isinstance(on_disk, dict):
        cfg.update(on_disk)
    except Exception:
      pass
  return cfg


# ---- Simple performance cache for better progress estimates ----
PERF_CACHE_PATH = Path("/srv/transcribe/config/perf_cache.json")
PERF_EWMA_ALPHA = 0.35

# Default real-time factor (wall/audio). 0.15 => ~6.7x faster than realtime.
DEFAULT_TRANSCRIBE_RTF = 0.15
# Default time-to-first transcript marker (seconds). Used for diagnostics only.
DEFAULT_TTF_MARKER_S = 20.0


def _perf_key(cfg: dict[str, Any]) -> str:
  # Keep this stable; it keys progress expectations to the main perf-affecting knobs.
  return (
    f"model={cfg.get('model')}"
    f"|device={cfg.get('device')}"
    f"|compute={cfg.get('compute_type')}"
    f"|bs={cfg.get('batch_size')}"
    f"|cs={cfg.get('chunk_size')}"
    f"|beam={cfg.get('beam_size')}"
    f"|align={cfg.get('align_model', '')}"
  )


def _ewma(prev: Optional[float], new: float, alpha: float) -> float:
  if prev is None:
    return float(new)
  return float(alpha) * float(new) + (1.0 - float(alpha)) * float(prev)


def _load_perf_cache() -> dict[str, Any]:
  try:
    return json.loads(PERF_CACHE_PATH.read_text(encoding="utf-8"))
  except FileNotFoundError:
    return {}
  except Exception:
    # Corrupt cache should never break jobs; just ignore it.
    return {}


def _save_perf_cache(cache: dict[str, Any]) -> None:
  PERF_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
  tmp = PERF_CACHE_PATH.with_suffix(PERF_CACHE_PATH.suffix + ".tmp")
  tmp.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
  tmp.replace(PERF_CACHE_PATH)


def _get_perf_entry(cfg: dict[str, Any]) -> dict[str, Any]:
  cache = _load_perf_cache()
  return cache.get(_perf_key(cfg), {})


def _update_perf_cache(
  *,
  cfg: dict[str, Any],
  audio_s: float,
  transcribe_wall_s: Optional[float],
  ttf_marker_s: Optional[float],
) -> None:
  if transcribe_wall_s is None:
    return
  if audio_s <= 0:
    return

  cache = _load_perf_cache()
  key = _perf_key(cfg)
  ent = cache.get(key, {})
  n = int(ent.get("n", 0)) + 1
  ent["n"] = n

  rtf = float(transcribe_wall_s) / float(audio_s)
  ent["ewma_rtf"] = _ewma(ent.get("ewma_rtf"), rtf, PERF_EWMA_ALPHA)

  if ttf_marker_s is not None:
    ent["ewma_ttf_marker_s"] = _ewma(ent.get("ewma_ttf_marker_s"), float(ttf_marker_s), PERF_EWMA_ALPHA)

  cache[key] = ent
  _save_perf_cache(cache)


def _build_whisperx_cmd(
  *,
  snippet_path: Path,
  out_dir: Path,
  language: str,
  speaker_mode: str,
  min_speakers: Optional[int],
  max_speakers: Optional[int],
  cfg: dict[str, Any],
) -> str:
  """
  Run via bash so we can `source enter.sh` and reuse your existing WhisperX venv.

  IMPORTANT:
  We call the existing `run_whisperx2.sh` using its **positional** signature:

   run_whisperx2.sh <input_audio> <language> <speaker_mode:auto|fixed> <min_speakers_or_-> <max_speakers_or_-> \
           <output_dir> <model> <device> <compute_type> <batch_size> <chunk_size> <beam_size> <align_model_or_->

  (So we do NOT pass `--model ...` style flags here.)
  """
  model = str(cfg.get("model", "large-v3"))
  device = str(cfg.get("device", "cuda"))
  compute_type = str(cfg.get("compute_type", "float16"))
  batch_size = str(int(cfg.get("batch_size", 3)))
  chunk_size = str(int(cfg.get("chunk_size", 30)))
  beam_size = str(int(cfg.get("beam_size", 5)))

  # Speaker args: only meaningful in fixed mode
  min_arg = "-"
  max_arg = "-"
  if speaker_mode == "fixed":
    if min_speakers is not None:
      min_arg = str(int(min_speakers))
    if max_speakers is not None:
      max_arg = str(int(max_speakers))

  align_model = str(cfg.get("align_model") or "").strip() or "-"

  # Build command:
  #  source enter.sh
  #  bash run_whisperx2.sh <snippet> <language> <speaker_mode> <min> <max> <outdir> <model> <device> <compute> <batch> <chunk> <beam> <align_model>
  cmd = (
    "set -euo pipefail; "
    f"source {shlex.quote(str(WHISPERX_ENTER_SH))}; "
    f"bash {shlex.quote(str(WHISPERX_RUN2_SH))} "
    f"{shlex.quote(str(snippet_path))} "
    f"{shlex.quote(str(language))} "
    f"{shlex.quote(str(speaker_mode))} "
    f"{shlex.quote(str(min_arg))} "
    f"{shlex.quote(str(max_arg))} "
    f"{shlex.quote(str(out_dir))} "
    f"{shlex.quote(str(model))} "
    f"{shlex.quote(str(device))} "
    f"{shlex.quote(str(compute_type))} "
    f"{shlex.quote(str(batch_size))} "
    f"{shlex.quote(str(chunk_size))} "
    f"{shlex.quote(str(beam_size))} "
    f"{shlex.quote(str(align_model))}"
  )
  return cmd


def _filter_and_parse_line(line: str) -> tuple[Optional[str], dict[str, Any]]:
  """
  Returns (log_line_or_none, info_dict).

  info_dict may contain:
   - stage: "vad"|"transcribe"|"align"|"diarize"
   - transcript_end_s: float
  """
  info: dict[str, Any] = {}

  # WhisperX / tqdm output sometimes uses carriage returns; normalize for matching.
  raw = line.rstrip("\n")
  raw = raw.lstrip("\r")
  if not raw.strip():
    return None, info

  raw_strip = raw.lstrip()

  m_ts = TIMESTAMP_INFO_RE.match(raw_strip)
  if m_ts:
    msg = m_ts.group("msg")
    msg_l = msg.lower()
    if "performing voice activity detection" in msg_l:
      info["stage"] = "vad"
    elif "performing transcription" in msg_l:
      info["stage"] = "transcribe"
    elif "performing alignment" in msg_l or "alignment" in msg_l:
      info["stage"] = "align"
    elif "performing diarization" in msg_l or "diarization" in msg_l:
      info["stage"] = "diarize"
    # Keep timestamped lines (compact)
    return raw_strip, info

  m_tr = TRANSCRIPT_RE.match(raw_strip)
  if m_tr:
    a = m_tr.group("a")
    b = m_tr.group("b")
    try:
      info["transcript_end_s"] = float(b)
    except Exception:
      pass
    # Log only the bracketed time range, not the transcript text itself.
    return f"Transcript: [{a} --> {b}]", info

  # Keep only important non-timestamp lines (errors) + our own prefix lines
  lower = raw_strip.lower()
  if raw_strip.startswith("[run_whisperx2]") or raw_strip.startswith("WORKER "):
    return raw_strip, info
  if ("traceback" in lower) or ("error" in lower) or ("exception" in lower):
    return raw_strip, info

  # Otherwise: drop (suppresses long warnings / transcript text)
  return None, info


def _run_whisperx_streaming(
  *,
  job,
  snippet_path: Path,
  whisperx_out_dir: Path,
  snippet_seconds: int,
  language: str,
  speaker_mode: str,
  min_speakers: Optional[int],
  max_speakers: Optional[int],
  cfg: dict[str, Any],
) -> Path:
  """
  Run WhisperX while:
   - writing a filtered worker.log
   - updating status.json with realistic progress + messages
  Returns the produced .srt path.
  """
  emitter = _StatusEmitter(job.status_path)

  # stage tracking for timing + progress
  stage: str = "prepare"
  stage_t0: float = time.monotonic()
  durations: dict[str, float] = {}
  warned_pyannote = False
  warned_torch = False

  def close_stage(prev: str) -> None:
    nonlocal stage_t0
    t1 = time.monotonic()
    durations[prev] = durations.get(prev, 0.0) + (t1 - stage_t0)
    stage_t0 = t1

  def set_stage(new_stage: str) -> None:
    nonlocal stage
    if stage != new_stage:
      close_stage(stage)
      stage = new_stage

  # Initial status
  emitter.maybe_emit(progress=P_SNIP_DONE, message="Preparing WhisperX…", phase="whisperx_prepare")

  cmd = _build_whisperx_cmd(
    snippet_path=snippet_path,
    out_dir=whisperx_out_dir,
    language=language,
    speaker_mode=speaker_mode,
    min_speakers=min_speakers,
    max_speakers=max_speakers,
    cfg=cfg,
  )
  _append_log(job.log_path, f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] WORKER whisperx_cmd={cmd}")

  env = os.environ.copy()
  # Force WhisperX (python) to flush logs/transcript lines promptly when piped.
  env.setdefault("PYTHONUNBUFFERED", "1")
  env.setdefault("PYTHONIOENCODING", "utf-8")
  env.setdefault("FORCE_COLOR", "0")

  proc = subprocess.Popen(
    ["bash", "-lc", cmd],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
    universal_newlines=True,
    env=env,
  )
  assert proc.stdout is not None

  sel = selectors.DefaultSelector()
  sel.register(proc.stdout, selectors.EVENT_READ)

  last_tick = time.monotonic()
  last_transcribe_end = 0.0
  progress = P_SNIP_DONE

  # Performance expectations for progress smoothing (learned over time)
  perf_ent = _get_perf_entry(cfg)
  expected_rtf = float(perf_ent.get("ewma_rtf", DEFAULT_TRANSCRIBE_RTF))
  expected_ttf = float(perf_ent.get("ewma_ttf_marker_s", DEFAULT_TTF_MARKER_S))
  expected_transcribe_wall_s = max(5.0, expected_rtf * float(snippet_seconds))

  # Transcribe progress state
  transcribe_stage_t0: Optional[float] = None
  transcribe_first_marker_dt: Optional[float] = None
  transcribe_marker_floor_end = 0.0

  # Simple “creep” during stages that may be quiet
  def creep(current: float, r: tuple[float, float], step: float) -> float:
    lo, hi = r
    return min(max(current, lo), hi - 0.002) + step

  try:
    while True:
      # Periodic progress creep (helps when a stage produces little output)
      now = time.monotonic()
      if (now - last_tick) >= 1.0:
        last_tick = now
        if stage == "vad":
          progress = min(progress, R_VAD[1] - 0.002)
          progress = max(progress, R_VAD[0])
          progress = min(progress + 0.002, R_VAD[1] - 0.002)
          emitter.maybe_emit(progress=progress, message="Detecting speech…", phase="whisperx_vad")
        elif stage == "transcribe":
          # Progress that does not depend on Transcript markers (which often arrive late/bursty).
          # We estimate based on expected wall time for this config, and clamp using markers when they appear.
          if transcribe_stage_t0 is None:
            transcribe_stage_t0 = now
            _append_log(job.log_path, f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] WORKER expected_transcribe_wall_s={expected_transcribe_wall_s:.3f} expected_rtf={expected_rtf:.4f} expected_ttf_marker_s={expected_ttf:.3f}")

          elapsed = max(0.0, now - transcribe_stage_t0)
          base_frac = min(0.995, elapsed / max(1e-3, expected_transcribe_wall_s))
          est_end = base_frac * float(snippet_seconds)

          # Never go backwards; if we have marker info, use it as a floor.
          est_end = max(est_end, transcribe_marker_floor_end)
          est_end = min(float(snippet_seconds), est_end)

          frac = min(1.0, max(0.0, est_end / max(1.0, float(snippet_seconds))))
          progress = R_TRANSCRIBE[0] + frac * (R_TRANSCRIBE[1] - R_TRANSCRIBE[0])
          pct = int(round(frac * 100))
          shown_s = int(round(est_end))
          emitter.maybe_emit(
            progress=progress,
            message=f"Transcribing… {shown_s}s/{snippet_seconds}s",
            phase="whisperx_transcribe",
          )
        elif stage == "align":
          progress = max(progress, R_ALIGN[0])
          progress = min(progress + 0.0015, R_ALIGN[1] - 0.002)
          emitter.maybe_emit(progress=progress, message="Aligning…", phase="whisperx_align")
        elif stage == "diarize":
          progress = max(progress, R_DIARIZE[0])
          progress = min(progress + 0.001, R_DIARIZE[1] - 0.002)
          emitter.maybe_emit(progress=progress, message="Diarizing…", phase="whisperx_diarize")

      events = sel.select(timeout=0.5)

      if events:
        for key, _ in events:
          line = key.fileobj.readline()
          if not line:
            continue

          # Track suppressed mismatch warnings so we can log a summary once
          if "Model was trained with pyannote.audio" in line:
            warned_pyannote = True
          if "Model was trained with torch" in line:
            warned_torch = True

          log_line, info = _filter_and_parse_line(line)
          if log_line:
            _append_wx(job.log_path, log_line)

          if "stage" in info:
            new_stage = info["stage"]
            set_stage(new_stage)
            if new_stage == "vad":
              progress = max(progress, R_VAD[0])
              emitter.maybe_emit(progress=progress, message="Detecting speech…", phase="whisperx_vad")
            elif new_stage == "transcribe":
              progress = max(progress, R_TRANSCRIBE[0])
              # Start transcribe timing immediately so progress ramps from the start of the stage.
              transcribe_stage_t0 = time.monotonic()
              transcribe_first_marker_dt = None
              transcribe_marker_floor_end = 0.0
              last_transcribe_end = 0.0
              emitter.maybe_emit(
                progress=progress,
                message=f"Transcribing… 0s/{snippet_seconds}s",
                phase="whisperx_transcribe",
              )
            elif new_stage == "align":
              progress = max(progress, R_ALIGN[0])
              emitter.maybe_emit(progress=progress, message="Aligning…", phase="whisperx_align")
            elif new_stage == "diarize":
              progress = max(progress, R_DIARIZE[0])
              emitter.maybe_emit(progress=progress, message="Diarizing…", phase="whisperx_diarize")

          if "transcript_end_s" in info:
            end_s = float(info["transcript_end_s"])
            last_transcribe_end = max(last_transcribe_end, end_s)
            transcribe_marker_floor_end = max(transcribe_marker_floor_end, end_s)
            if transcribe_first_marker_dt is None and transcribe_stage_t0 is not None:
              transcribe_first_marker_dt = max(0.0, time.monotonic() - transcribe_stage_t0)
              _append_log(
                job.log_path,
                f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] WORKER ttf_marker_s={transcribe_first_marker_dt:.3f} first_marker_end_s={end_s:.3f}",
              )

      # exit condition: process ended AND no more data to read
      if proc.poll() is not None:
        # drain remaining
        while True:
          line = proc.stdout.readline()
          if not line:
            break
          log_line, info = _filter_and_parse_line(line)
          if log_line:
            _append_wx(job.log_path, log_line)
        break

    rc = proc.returncode or 0
    if warned_pyannote or warned_torch:
      parts = []
      if warned_pyannote:
        parts.append("pyannote.audio")
      if warned_torch:
        parts.append("torch")
      _append_log(job.log_path, f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] WARN version-mismatch lines suppressed for: {', '.join(parts)}")

    if rc != 0:
      raise RuntimeError(f"WhisperX exited with rc={rc}")

  finally:
    try:
      sel.unregister(proc.stdout)
    except Exception:
      pass
    try:
      sel.close()
    except Exception:
      pass

  # Close last stage timing
  close_stage(stage)

  # Update perf cache (learn expected speed for future progress estimates)
  try:
    _update_perf_cache(
      cfg=cfg,
      audio_s=float(snippet_seconds),
      transcribe_wall_s=durations.get("transcribe"),
      ttf_marker_s=transcribe_first_marker_dt,
    )
    # Log the learned parameters (best effort)
    ent = _get_perf_entry(cfg)
    if ent:
      _append_log(
        job.log_path,
        f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] WORKER perf_cache n={ent.get('n')} ewma_rtf={ent.get('ewma_rtf')} ewma_ttf_marker_s={ent.get('ewma_ttf_marker_s')}",
      )
  except Exception:
    pass

  # Find the produced SRT (assume exactly one .srt in output dir; otherwise pick newest)
  srts = sorted(whisperx_out_dir.glob("*.srt"), key=lambda p: p.stat().st_mtime)
  if not srts:
    raise RuntimeError(f"No .srt produced in {whisperx_out_dir}")
  srt_path = srts[-1]

  # Finalize status
  emitter.maybe_emit(progress=R_FINALIZE[0], message="Finalizing…", phase="finalizing")
  emitter.maybe_emit(
    progress=P_WHISPERX_DONE,
    message="WhisperX complete",
    phase="whisperx_done",
    extra={"srt_filename": srt_path.name, "timings": {k: round(v, 3) for k, v in durations.items()}},
  )

  # Also log timings explicitly
  tsz = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
  _append_log(job.log_path, f"[{tsz}] WORKER timings " + " ".join(f"{k}={durations[k]:.3f}s" for k in sorted(durations.keys())))

  return srt_path
