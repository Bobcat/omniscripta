from __future__ import annotations

import fcntl
from pathlib import Path
from typing import Any, Mapping

from upload._util import _read_json, _write_json_atomic


def _status_lock_path(status_path: Path) -> Path:
  return status_path.with_suffix(status_path.suffix + ".lock")


def _read_status_unlocked(status_path: Path) -> dict[str, Any]:
  if not status_path.exists():
    return {}
  try:
    return _read_json(status_path)
  except Exception:
    return {}


def _write_status_snapshot(status_path: Path, status: dict[str, Any]) -> None:
  lock_path = _status_lock_path(status_path)
  lock_path.parent.mkdir(parents=True, exist_ok=True)
  with lock_path.open("a+b") as lock_f:
    fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
    try:
      _write_json_atomic(status_path, dict(status or {}))
    finally:
      fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)


def _write_status_safely(status_path: Path, **patch: Any) -> None:
  try:
    _write_status(status_path, **patch)
  except Exception:
    pass


def _write_topics_status(status_path: Path, *, subphase: str, message: str, **patch: Any) -> None:
  _write_status(status_path, state="running", phase="topics", subphase=subphase, status_owner="api-topics", message=message, **patch)


def _fmt_eta(seconds: float) -> str:
  s = max(0, int(round(float(seconds))))
  h = s // 3600
  m = (s % 3600) // 60
  sec = s % 60
  if h > 0:
    return f"{h:02d}:{m:02d}:{sec:02d}"
  return f"{m:02d}:{sec:02d}"


def _timings_with_running_total(timings_text: str, running_total_s: float | None) -> str:
  timings = str(timings_text or "").strip()
  if not timings:
    return timings
  if running_total_s is None:
    return timings
  try:
    total = max(0.0, float(running_total_s))
  except Exception:
    return timings

  parts = [p.strip() for p in timings.split("|")]
  out: list[str] = []
  replaced = False
  for p in parts:
    if not p:
      continue
    if p.startswith("total="):
      out.append(f"total={total:.2f}s")
      replaced = True
    else:
      out.append(p)
  if not replaced:
    out.append(f"total={total:.2f}s")
  return " | ".join(out)


def _status_running_total_s(status: Mapping[str, Any]) -> float | None:
  elapsed_s = status.get("elapsed_s")
  eta_total = status.get("eta_total_s")
  eta_remaining = status.get("eta_remaining_s")
  if elapsed_s is not None:
    try:
      return float(elapsed_s)
    except Exception:
      return None
  if eta_total is not None and eta_remaining is not None:
    try:
      return float(eta_total) - float(eta_remaining)
    except Exception:
      return None
  return None


def _status_message_hints(status: Mapping[str, Any]) -> list[str]:
  raw_hints = status.get("eta_hints")
  if not isinstance(raw_hints, list):
    return []
  hints: list[str] = []
  for raw_hint in raw_hints:
    hint = str(raw_hint).strip()
    if hint and hint not in hints:
      hints.append(hint)
  return hints


def format_status_message(status: Mapping[str, Any]) -> str | None:
  msg = str(status.get("message") or "")
  if not msg:
    return None
  if " || eta: " in msg:
    msg = msg.split(" || eta: ", 1)[0]
  if " || timings: " in msg:
    msg = msg.split(" || timings: ", 1)[0]

  timings = _timings_with_running_total(
    str(status.get("timings_text", "") or "").strip(),
    _status_running_total_s(status),
  )
  eta_total = status.get("eta_total_s")
  eta_remaining = status.get("eta_remaining_s")
  eta_hints = _status_message_hints(status)

  eta_suffix = ""
  if eta_total is not None and eta_remaining is not None:
    try:
      eta_suffix = f" || eta: ETA {_fmt_eta(float(eta_remaining))} | est_total {_fmt_eta(float(eta_total))}"
    except Exception:
      eta_suffix = ""
  if eta_suffix and eta_hints:
    eta_suffix += f" | hints: {','.join(eta_hints)}"

  if timings:
    return f"{msg}{eta_suffix} || timings: {timings}"
  return f"{msg}{eta_suffix}"


def project_status_message(status: Mapping[str, Any]) -> dict[str, Any]:
  out = dict(status or {})
  message = format_status_message(out)
  if message is not None:
    out["message"] = message
  return out


def _write_status(status_path: Path, **patch: Any) -> None:
  lock_path = _status_lock_path(status_path)
  lock_path.parent.mkdir(parents=True, exist_ok=True)
  with lock_path.open("a+b") as lock_f:
    fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
    try:
      cur = _read_status_unlocked(status_path)
      cur.update({k: v for k, v in patch.items() if v is not None})
      message = format_status_message(cur)
      if message is not None:
        cur["message"] = message

      _write_json_atomic(status_path, cur)
    finally:
      fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
