from __future__ import annotations

import json
import os
import selectors
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from worker_status_io import _StatusEmitter, _append_log, _append_wx


# Progress is intentionally kept flat (0.0) during processing.
# We only move to 1.0 when the job is finalized in worker_daemon.py.
P_SNIP_DONE = 0.0
P_WHISPERX_DONE = 0.0
R_VAD = (0.0, 0.0)
R_TRANSCRIBE = (0.0, 0.0)
R_ALIGN = (0.0, 0.0)
R_DIARIZE = (0.0, 0.0)
R_FINALIZE = (0.0, 0.0)

# Config paths
DEFAULT_SERVER_CFG_PATH = Path("/srv/transcribe/config/whisperx.json")
SERVER_CFG_PATH = Path(os.getenv("TRANSCRIBE_WHISPERX_CONFIG", str(DEFAULT_SERVER_CFG_PATH)))
WHISPERX_ENV_FILE = Path.home() / ".config" / "whisperx" / "env"
DEFAULT_WHISPERX_VENV = Path.home() / "whisperx" / ".venv"


def _load_server_config() -> dict[str, Any]:
  # Safe defaults for this environment; overridden by whisperx.json when present.
  cfg: dict[str, Any] = {
    "model": "large-v3",
    "device": "cuda",
    "compute_type": "float16",
    "batch_size": 3,
    "chunk_size": 30,
    "beam_size": 5,
    "align_model": "",
    "diarize_model": "",
    "whisperx_venv": str(DEFAULT_WHISPERX_VENV),
  }
  if SERVER_CFG_PATH.exists():
    try:
      on_disk = json.loads(SERVER_CFG_PATH.read_text(encoding="utf-8"))
      if isinstance(on_disk, dict):
        cfg.update(on_disk)
    except Exception:
      pass
  return cfg


def _load_env_file(path: Path) -> None:
  if not path.exists():
    return
  for raw in path.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
      continue
    if line.startswith("export "):
      line = line[len("export "):].strip()
    if "=" not in line:
      continue
    key, val = line.split("=", 1)
    key = key.strip()
    val = val.strip().strip("'").strip('"')
    if key and key not in os.environ:
      os.environ[key] = val


def _discover_site_packages(venv_path: Path) -> list[Path]:
  if not venv_path.exists():
    return []
  found = sorted(p for p in venv_path.glob("lib/python*/site-packages") if p.is_dir())
  if not found:
    return []
  py_tag = f"python{sys.version_info.major}.{sys.version_info.minor}"
  preferred = [p for p in found if py_tag in p.as_posix()]
  return preferred or found


def _collect_nvidia_lib_dirs(site_packages: list[Path]) -> list[str]:
  lib_dirs: list[str] = []
  for sp in site_packages:
    for p in sorted((sp / "nvidia").glob("*/lib")):
      if p.is_dir():
        lib_dirs.append(str(p))
  # keep order, drop duplicates
  unique: list[str] = []
  for p in lib_dirs:
    if p not in unique:
      unique.append(p)
  return unique


def _merge_ld_library_path(prepend_dirs: list[str], current: str) -> str:
  cur = [x for x in (current or "").split(":") if x]
  merged: list[str] = []
  for p in [*prepend_dirs, *cur]:
    if p and p not in merged:
      merged.append(p)
  return ":".join(merged)


def _resolve_whisperx_python(cfg: dict[str, Any]) -> Path:
  venv_from_cfg = str(cfg.get("whisperx_venv") or "").strip()
  candidates: list[Path] = []
  if venv_from_cfg:
    candidates.append(Path(venv_from_cfg).expanduser() / "bin" / "python")
  candidates.append(DEFAULT_WHISPERX_VENV / "bin" / "python")
  for c in candidates:
    if c.exists():
      return c
  # Last-resort fallback; should normally never be used.
  return Path(sys.executable)


def _build_runner_env(cfg: dict[str, Any]) -> tuple[dict[str, str], list[Path], list[str]]:
  _load_env_file(WHISPERX_ENV_FILE)
  env = dict(os.environ)
  env.setdefault("PYTHONUNBUFFERED", "1")
  env.setdefault("PYTHONIOENCODING", "utf-8")
  env.setdefault("FORCE_COLOR", "0")
  # Match old enter.sh behavior so pyannote checkpoints keep loading.
  env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

  site_packages: list[Path] = []
  venv_from_cfg = str(cfg.get("whisperx_venv") or "").strip()
  if venv_from_cfg:
    site_packages = _discover_site_packages(Path(venv_from_cfg).expanduser())
  if not site_packages:
    site_packages = _discover_site_packages(DEFAULT_WHISPERX_VENV)

  nvidia_lib_dirs = _collect_nvidia_lib_dirs(site_packages)
  if nvidia_lib_dirs:
    env["LD_LIBRARY_PATH"] = _merge_ld_library_path(nvidia_lib_dirs, env.get("LD_LIBRARY_PATH", ""))

  return env, site_packages, nvidia_lib_dirs


def _emit_stage(emitter: _StatusEmitter, *, stage: str, snippet_seconds: int) -> None:
  if stage == "transcribe":
    emitter.maybe_emit(progress=R_TRANSCRIBE[0], message=f"Transcribing… 0s/{snippet_seconds}s", phase="whisperx_transcribe")
    return
  if stage == "align":
    emitter.maybe_emit(progress=R_ALIGN[0], message="Aligning…", phase="whisperx_align")
    return
  if stage == "diarize":
    emitter.maybe_emit(progress=R_DIARIZE[0], message="Diarizing…", phase="whisperx_diarize")
    return
  if stage == "finalize":
    emitter.maybe_emit(progress=R_FINALIZE[0], message="Finalizing…", phase="finalizing")
    return
  emitter.maybe_emit(progress=P_SNIP_DONE, message="Preparing WhisperX…", phase="whisperx_prepare")


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
  on_phase_timing: Optional[Callable[[str, float], None]] = None,
  on_stage_change: Optional[Callable[[str], None]] = None,
  on_heartbeat: Optional[Callable[[], None]] = None,
) -> tuple[Path, dict[str, float], set[str]]:
  """
  Run WhisperX through a clean WhisperX-venv subprocess that uses direct import calls.
  This avoids mixed-venv CUDA/library issues while still using Python API (not CLI script wrapper).
  """
  use_internal_status = (on_phase_timing is None and on_stage_change is None and on_heartbeat is None)
  emitter = _StatusEmitter(job.status_path) if use_internal_status else None
  durations: dict[str, float] = {}
  emitted_live: set[str] = set()
  current_stage = "prepare"
  if emitter is not None:
    _emit_stage(emitter, stage=current_stage, snippet_seconds=snippet_seconds)
  if on_stage_change:
    try:
      on_stage_change("prepare")
    except Exception:
      pass

  try:
    t0 = time.monotonic()
    env, site_packages, nvidia_lib_dirs = _build_runner_env(cfg)
    runner_python = _resolve_whisperx_python(cfg)
    runner_script = Path(__file__).with_name("whisperx_import_runner.py")
    if not runner_script.exists():
      raise RuntimeError(f"Missing runner script: {runner_script}")

    whisperx_out_dir.mkdir(parents=True, exist_ok=True)
    args_path = whisperx_out_dir / "_import_runner_args.json"
    out_path = whisperx_out_dir / "_import_runner_out.json"
    args_obj = {
      "snippet_path": str(snippet_path),
      "whisperx_out_dir": str(whisperx_out_dir),
      "language": str(language),
      "speaker_mode": str(speaker_mode),
      "min_speakers": (int(min_speakers) if min_speakers is not None else None),
      "max_speakers": (int(max_speakers) if max_speakers is not None else None),
      "model": str(cfg.get("model", "large-v3")),
      "device": str(cfg.get("device", "cuda")),
      "compute_type": str(cfg.get("compute_type", "float16")),
      "batch_size": int(cfg.get("batch_size", 3)),
      "chunk_size": int(cfg.get("chunk_size", 30)),
      "beam_size": int(cfg.get("beam_size", 5)),
      "align_model": str(cfg.get("align_model") or "").strip(),
      "diarize_model": str(cfg.get("diarize_model") or "").strip(),
    }
    args_path.write_text(json.dumps(args_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    durations["prepare"] = time.monotonic() - t0
    if on_phase_timing:
      try:
        on_phase_timing("whisperx_prepare", float(durations["prepare"]))
        emitted_live.add("prepare")
      except Exception:
        pass

    _append_wx(job.log_path, f"whisperx_runner_python={runner_python}")
    if site_packages:
      _append_wx(job.log_path, f"whisperx_site_packages={[str(p) for p in site_packages]}")
    _append_wx(job.log_path, f"nvidia_lib_dirs={nvidia_lib_dirs}")

    cmd = [str(runner_python), str(runner_script), "--args-json", str(args_path), "--out-json", str(out_path)]
    _append_log(job.log_path, f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] WORKER whisperx_runner_cmd={cmd}")

    proc = subprocess.Popen(
      cmd,
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
    last_line = ""
    last_heartbeat_t = time.monotonic()

    try:
      while True:
        events = sel.select(timeout=0.5)
        now_t = time.monotonic()
        if on_heartbeat and (now_t - last_heartbeat_t) >= 1.0:
          try:
            on_heartbeat()
          except Exception:
            pass
          last_heartbeat_t = now_t

        if events:
          for key, _ in events:
            raw = key.fileobj.readline()
            if not raw:
              continue
            line = raw.rstrip("\n").lstrip("\r")
            if not line:
              continue
            last_line = line
            _append_wx(job.log_path, line)
            if line.startswith("STAGE "):
              new_stage = line[len("STAGE "):].strip().lower()
              if new_stage in {"transcribe", "align", "diarize", "finalize"}:
                current_stage = new_stage
                if emitter is not None:
                  _emit_stage(emitter, stage=current_stage, snippet_seconds=snippet_seconds)
                if on_stage_change:
                  try:
                    on_stage_change(new_stage)
                  except Exception:
                    pass
            if line.startswith("TIMING "):
              parts = line.split()
              if len(parts) >= 3:
                phase_key = parts[1].strip().lower()
                try:
                  sec = float(parts[2])
                except Exception:
                  sec = None
                if sec is not None:
                  durations[phase_key] = sec
                  if on_phase_timing:
                    try:
                      on_phase_timing(f"whisperx_{phase_key}", sec)
                      emitted_live.add(phase_key)
                    except Exception:
                      pass

        if proc.poll() is not None:
          # Drain remaining output.
          while True:
            raw = proc.stdout.readline()
            if not raw:
              break
            line = raw.rstrip("\n").lstrip("\r")
            if not line:
              continue
            last_line = line
            _append_wx(job.log_path, line)
          break
    finally:
      try:
        sel.unregister(proc.stdout)
      except Exception:
        pass
      try:
        sel.close()
      except Exception:
        pass

    rc = proc.returncode if proc.returncode is not None else proc.wait()
    if rc != 0:
      raise RuntimeError(f"WhisperX runner exited rc={rc} stage={current_stage} last_line={last_line!r}")
    if not out_path.exists():
      raise RuntimeError(f"WhisperX runner did not write output metadata: {out_path}")

    out = json.loads(out_path.read_text(encoding="utf-8"))
    out_timings = out.get("timings")
    if isinstance(out_timings, dict):
      for k, v in out_timings.items():
        try:
          durations[str(k)] = float(v)
        except Exception:
          pass

    srt_path = Path(str(out.get("srt_path", "") or ""))
    if not srt_path.exists():
      srts = sorted(whisperx_out_dir.glob("*.srt"), key=lambda p: p.stat().st_mtime)
      if not srts:
        raise RuntimeError(f"No .srt produced in {whisperx_out_dir}")
      srt_path = srts[-1]

    if emitter is not None:
      _emit_stage(emitter, stage="finalize", snippet_seconds=snippet_seconds)
      emitter.maybe_emit(
        progress=P_WHISPERX_DONE,
        message="WhisperX complete",
        phase="whisperx_done",
        extra={"srt_filename": srt_path.name, "timings": {k: round(v, 3) for k, v in durations.items()}},
      )

    tsz = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    _append_log(job.log_path, f"[{tsz}] WORKER timings " + " ".join(f"{k}={durations[k]:.3f}s" for k in sorted(durations.keys())))
    return srt_path, dict(durations), set(emitted_live)

  except Exception as e:
    tsz = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    _append_log(job.log_path, f"[{tsz}] WORKER whisperx_failed stage={current_stage} error={e!r}")
    raise
