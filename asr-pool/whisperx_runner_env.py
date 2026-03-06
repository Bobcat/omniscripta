from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from shared.app_config import get_bool, get_int, get_setting, get_str

WHISPERX_ENV_FILE = Path.home() / ".config" / "whisperx" / "env"
DEFAULT_WHISPERX_VENV = Path.home() / "whisperx" / ".venv"


def _load_server_config() -> dict[str, Any]:
  cfg: dict[str, Any] = {
    "model": get_str("asr_pool.whisperx.model", "large-v3"),
    "device": get_str("asr_pool.whisperx.device", "cuda"),
    "compute_type": get_str("asr_pool.whisperx.compute_type", "int8"),
    "batch_size": get_int("asr_pool.whisperx.batch_size", 1, min_value=1),
    "chunk_size": get_int("asr_pool.whisperx.chunk_size", 20, min_value=1),
    "chunk_size_live": get_int("asr_pool.whisperx.chunk_size_live", 10, min_value=1),
    "live_chunk_backend": get_str("asr_pool.whisperx.live_chunk_backend", "whisperx"),
    "beam_size": get_int("asr_pool.whisperx.beam_size", 5, min_value=1),
    "align_model": get_str("asr_pool.whisperx.align_model", ""),
    "diarize_model": get_str("asr_pool.whisperx.diarize_model", ""),
    "whisperx_venv": get_str("asr_pool.whisperx.venv", str(DEFAULT_WHISPERX_VENV)),
  }
  raw_threads = get_setting("asr_pool.whisperx.threads", {})
  if isinstance(raw_threads, dict):
    try:
      cfg["omp_num_threads"] = int(raw_threads.get("omp")) if raw_threads.get("omp") is not None else None
    except Exception:
      cfg["omp_num_threads"] = None
    try:
      cfg["mkl_num_threads"] = int(raw_threads.get("mkl")) if raw_threads.get("mkl") is not None else None
    except Exception:
      cfg["mkl_num_threads"] = None
    try:
      cfg["torch_num_threads"] = int(raw_threads.get("torch")) if raw_threads.get("torch") is not None else None
    except Exception:
      cfg["torch_num_threads"] = None
    try:
      cfg["torch_num_interop_threads"] = int(raw_threads.get("torch_interop")) if raw_threads.get("torch_interop") is not None else None
    except Exception:
      cfg["torch_num_interop_threads"] = None
  else:
    cfg["omp_num_threads"] = None
    cfg["mkl_num_threads"] = None
    cfg["torch_num_threads"] = None
    cfg["torch_num_interop_threads"] = None

  cfg["live_chunk_backend"] = str(cfg["live_chunk_backend"] or "whisperx").strip().lower() or "whisperx"
  if cfg["live_chunk_backend"] not in {"whisperx", "faster_whisper_direct"}:
    cfg["live_chunk_backend"] = "whisperx"

  # Keep this flag available in cfg for diagnostics parity.
  cfg["vad_filter"] = bool(get_bool("asr_pool.whisperx.vad_filter", True))
  return cfg


def _cfg_positive_int(cfg: dict[str, Any], key: str) -> int | None:
  try:
    val = int(cfg.get(key))
  except Exception:
    return None
  if val <= 0:
    return None
  return val


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
  return Path(sys.executable)


def _build_runner_env(cfg: dict[str, Any]) -> tuple[dict[str, str], list[Path], list[str]]:
  _load_env_file(WHISPERX_ENV_FILE)
  env = dict(os.environ)
  env.setdefault("PYTHONUNBUFFERED", "1")
  env.setdefault("PYTHONIOENCODING", "utf-8")
  env.setdefault("FORCE_COLOR", "0")
  env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
  omp_num_threads = _cfg_positive_int(cfg, "omp_num_threads")
  mkl_num_threads = _cfg_positive_int(cfg, "mkl_num_threads")
  if omp_num_threads is not None:
    env["OMP_NUM_THREADS"] = str(omp_num_threads)
  if mkl_num_threads is not None:
    env["MKL_NUM_THREADS"] = str(mkl_num_threads)

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
