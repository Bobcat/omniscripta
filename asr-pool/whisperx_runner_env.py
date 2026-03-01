from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


DEFAULT_SERVER_CFG_PATH = Path("/srv/transcribe/config/whisperx.json")
SERVER_CFG_PATH = Path(os.getenv("TRANSCRIBE_WHISPERX_CONFIG", str(DEFAULT_SERVER_CFG_PATH)))
WHISPERX_ENV_FILE = Path.home() / ".config" / "whisperx" / "env"
DEFAULT_WHISPERX_VENV = Path.home() / "whisperx" / ".venv"


def _load_server_config() -> dict[str, Any]:
  cfg: dict[str, Any] = {
    "model": "large-v3",
    "device": "cuda",
    "compute_type": "float16",
    "batch_size": 3,
    "chunk_size": 30,
    "beam_size": 5,
    "align_model": "",
    "diarize_model": "",
    "omp_num_threads": None,
    "mkl_num_threads": None,
    "torch_num_threads": None,
    "torch_num_interop_threads": None,
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
