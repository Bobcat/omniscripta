from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from whisperx_runner_env import _build_runner_env, _load_server_config, _resolve_whisperx_python


def _env_bool(name: str, default: bool) -> bool:
  raw = str(os.getenv(name, "") or "").strip().lower()
  if not raw:
    return bool(default)
  if raw in {"1", "true", "yes", "on", "y"}:
    return True
  if raw in {"0", "false", "no", "off", "n"}:
    return False
  return bool(default)


def _env_float(name: str, default: float) -> float:
  try:
    return float(os.getenv(name, "").strip() or default)
  except Exception:
    return float(default)


def _fingerprint_cfg(cfg: dict[str, Any]) -> str:
  keys = [
    "model",
    "device",
    "compute_type",
    "batch_size",
    "chunk_size",
    "beam_size",
    "align_model",
    "torch_num_threads",
    "torch_num_interop_threads",
    "whisperx_venv",
  ]
  payload = {k: cfg.get(k) for k in keys}
  return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class PersistentRunnerClientError(RuntimeError):
  pass


def _log(msg: str) -> None:
  try:
    print(f"warm_runner {msg}", flush=True)
  except Exception:
    pass


class _AsrPoolWarmRunnerClient:
  def __init__(self) -> None:
    self._lock = threading.RLock()
    self._proc: subprocess.Popen[str] | None = None
    self._cfg_fingerprint: str | None = None
    self._last_used_t = 0.0
    self._server_init_path: Path | None = None

  def prewarm(self) -> None:
    # Force eager model load without waiting for a first live request.
    with self._lock:
      self._ensure_runner_locked()
      proc = self._proc
      if proc is None or proc.poll() is not None or proc.stdin is None:
        raise PersistentRunnerClientError("Persistent runner is not available")

      prewarm_timeout_s = max(5.0, _env_float("TRANSCRIBE_ASR_POOL_WARM_PREWARM_TIMEOUT_S", 180.0))
      poll_s = max(0.02, _env_float("TRANSCRIBE_ASR_POOL_WARM_RESPONSE_POLL_S", 0.05))
      prewarm_language = str(os.getenv("TRANSCRIBE_ASR_POOL_WARM_PREWARM_LANGUAGE", "en") or "en").strip() or "en"
      prewarm_align_enabled = _env_bool("TRANSCRIBE_ASR_POOL_ALIGN_ENABLED", False)

      ipc_dir = (Path("/tmp") / "transcribe_asr_pool_runner" / "_ipc").resolve()
      ipc_dir.mkdir(parents=True, exist_ok=True)
      token = uuid.uuid4().hex
      response_path = ipc_dir / f"{token}.prewarm.response.json"
      cmd_obj = {
        "cmd": "prewarm",
        "response_path": str(response_path),
        "language": prewarm_language,
        "align_enabled": bool(prewarm_align_enabled),
      }
      try:
        proc.stdin.write(json.dumps(cmd_obj) + "\n")
        proc.stdin.flush()
      except Exception as e:
        self._shutdown_locked(reason="stdin_write_failed")
        raise PersistentRunnerClientError(f"Failed to send prewarm command: {e!r}") from e

      data = self._wait_for_response_locked(
        proc=proc,
        response_path=response_path,
        timeout_s=prewarm_timeout_s,
        poll_s=poll_s,
        timeout_reason="prewarm_timeout",
      )
      if not bool(data.get("ok", False)):
        err = dict(data.get("error") or {})
        raise PersistentRunnerClientError(
          f"{err.get('code') or 'ASR_PERSISTENT_PREWARM_FAILURE'}: {err.get('message') or 'prewarm failed'}"
        )
      self._last_used_t = time.monotonic()

  def _server_script(self) -> Path:
    return Path(__file__).with_name("whisperx_runner_server.py")

  def _spawn_locked(self, cfg: dict[str, Any]) -> None:
    self._shutdown_locked(reason="respawn")

    env, _site_packages, _nvidia_lib_dirs = _build_runner_env(cfg)
    runner_python = _resolve_whisperx_python(cfg)
    server_script = self._server_script()
    if not server_script.exists():
      raise PersistentRunnerClientError(f"Missing persistent server script: {server_script}")

    init_dir = Path("/tmp") / "transcribe_asr_pool_runner"
    init_dir.mkdir(parents=True, exist_ok=True)
    init_path = init_dir / f"init_{os.getpid()}_{uuid.uuid4().hex}.json"
    init_path.write_text(json.dumps({"cfg": cfg}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    cmd = [str(runner_python), str(server_script), "--init-json", str(init_path)]
    proc = subprocess.Popen(
      cmd,
      stdin=subprocess.PIPE,
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL,
      text=True,
      bufsize=1,
      universal_newlines=True,
      env=env,
    )
    if proc.stdin is None:
      raise PersistentRunnerClientError("Failed to open stdin for persistent runner")

    self._proc = proc
    self._cfg_fingerprint = _fingerprint_cfg(cfg)
    self._last_used_t = time.monotonic()
    self._server_init_path = init_path
    _log(
      "spawned "
      f"pid={proc.pid} model={cfg.get('model')} device={cfg.get('device')} "
      f"compute_type={cfg.get('compute_type')} batch_size={cfg.get('batch_size')} chunk_size={cfg.get('chunk_size')}"
    )

  def _ensure_runner_locked(self) -> None:
    cfg = dict(_load_server_config() or {})
    fp = _fingerprint_cfg(cfg)
    if self._proc is not None and self._proc.poll() is None and self._cfg_fingerprint == fp:
      self._last_used_t = time.monotonic()
      return
    self._spawn_locked(cfg)

  def _shutdown_locked(self, *, reason: str) -> None:
    proc = self._proc
    self._proc = None
    self._cfg_fingerprint = None
    self._last_used_t = 0.0
    init_path = self._server_init_path
    self._server_init_path = None
    if proc is not None:
      pid = proc.pid
      try:
        if proc.poll() is None and proc.stdin is not None:
          try:
            proc.stdin.write(json.dumps({"cmd": "shutdown", "reason": reason}) + "\n")
            proc.stdin.flush()
          except Exception:
            pass
      except Exception:
        pass
      try:
        proc.wait(timeout=1.0)
      except Exception:
        try:
          proc.terminate()
        except Exception:
          pass
        try:
          proc.wait(timeout=2.0)
        except Exception:
          try:
            proc.kill()
          except Exception:
            pass
      _log(f"stopped pid={pid} reason={reason}")
    if init_path is not None:
      try:
        init_path.unlink(missing_ok=True)
      except Exception:
        pass

  def shutdown(self, *, reason: str = "manual") -> None:
    with self._lock:
      self._shutdown_locked(reason=reason)

  def maybe_shutdown_idle(self) -> None:
    idle_s = max(0.0, _env_float("TRANSCRIBE_ASR_POOL_WARM_IDLE_S", 120.0))
    if idle_s <= 0:
      return
    with self._lock:
      if self._proc is None or self._proc.poll() is not None:
        return
      if self._last_used_t <= 0:
        return
      if (time.monotonic() - self._last_used_t) >= idle_s:
        self._shutdown_locked(reason="idle_timeout")

  def transcribe(self, *, job: Any, request: dict[str, Any], progress_path: Path | None = None) -> dict[str, Any]:
    request_timeout_s = max(1.0, _env_float("TRANSCRIBE_ASR_POOL_WARM_REQUEST_TIMEOUT_S", 120.0))
    poll_s = max(0.02, _env_float("TRANSCRIBE_ASR_POOL_WARM_RESPONSE_POLL_S", 0.05))
    with self._lock:
      self._ensure_runner_locked()
      proc = self._proc
      if proc is None or proc.poll() is not None or proc.stdin is None:
        raise PersistentRunnerClientError("Persistent runner is not available")

      ipc_dir = (Path(job.whisperx_dir) / "_ipc").resolve()
      ipc_dir.mkdir(parents=True, exist_ok=True)
      token = uuid.uuid4().hex
      payload_path = ipc_dir / f"{token}.request.json"
      response_path = ipc_dir / f"{token}.response.json"

      envelope = {
        "request": dict(request or {}),
        "work": {
          "whisperx_out_dir": str(Path(job.whisperx_dir).resolve()),
        },
      }
      payload_path.write_text(json.dumps(envelope, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
      try:
        proc.stdin.write(
          json.dumps(
            {
              "cmd": "transcribe",
              "payload_path": str(payload_path),
              "response_path": str(response_path),
              "progress_path": (str(progress_path) if progress_path is not None else ""),
            }
          )
          + "\n"
        )
        proc.stdin.flush()
      except Exception as e:
        self._shutdown_locked(reason="stdin_write_failed")
        raise PersistentRunnerClientError(f"Failed to send request to persistent runner: {e!r}") from e

      try:
        data = self._wait_for_response_locked(
          proc=proc,
          response_path=response_path,
          timeout_s=request_timeout_s,
          poll_s=poll_s,
          timeout_reason="response_timeout",
        )
        self._last_used_t = time.monotonic()
        return data
      finally:
        try:
          payload_path.unlink(missing_ok=True)
        except Exception:
          pass

  def _wait_for_response_locked(
    self,
    *,
    proc: subprocess.Popen[str],
    response_path: Path,
    timeout_s: float,
    poll_s: float,
    timeout_reason: str,
  ) -> dict[str, Any]:
    deadline = time.monotonic() + max(0.1, float(timeout_s))
    while time.monotonic() < deadline:
      if response_path.exists():
        try:
          return json.loads(response_path.read_text(encoding="utf-8"))
        except Exception as e:
          raise PersistentRunnerClientError(f"Failed to parse persistent runner response: {e!r}") from e
        finally:
          try:
            response_path.unlink(missing_ok=True)
          except Exception:
            pass
      if proc.poll() is not None:
        code = proc.returncode
        self._shutdown_locked(reason="runner_exited")
        raise PersistentRunnerClientError(f"Persistent runner exited unexpectedly (code={code})")
      time.sleep(max(0.02, float(poll_s)))

    self._shutdown_locked(reason=timeout_reason)
    raise PersistentRunnerClientError(f"Persistent runner timed out after {float(timeout_s):.1f}s")
