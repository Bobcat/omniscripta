from __future__ import annotations

import fcntl
from pathlib import Path
from typing import Any

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


def _write_status(status_path: Path, **patch: Any) -> None:
  lock_path = _status_lock_path(status_path)
  lock_path.parent.mkdir(parents=True, exist_ok=True)
  with lock_path.open("a+b") as lock_f:
    fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
    try:
      cur = _read_status_unlocked(status_path)
      cur.update({k: v for k, v in patch.items() if v is not None})

      timings = str(cur.get("timings_text", "") or "").strip()
      eta_total = cur.get("eta_total_s")
      eta_remaining = cur.get("eta_remaining_s")
      elapsed_s = cur.get("elapsed_s")
      eta_hints_raw = cur.get("eta_hints")
      msg = str(cur.get("message", "") or "")

      running_total_s: float | None = None
      if elapsed_s is not None:
        try:
          running_total_s = float(elapsed_s)
        except Exception:
          running_total_s = None
      elif eta_total is not None and eta_remaining is not None:
        try:
          running_total_s = float(eta_total) - float(eta_remaining)
        except Exception:
          running_total_s = None
      timings = _timings_with_running_total(timings, running_total_s)

      if msg:
        if " || eta: " in msg:
          msg = msg.split(" || eta: ", 1)[0]
        if " || timings: " in msg:
          msg = msg.split(" || timings: ", 1)[0]
        eta_suffix = ""
        if eta_total is not None and eta_remaining is not None:
          try:
            eta_suffix = f" || eta: ETA {_fmt_eta(float(eta_remaining))} | est_total {_fmt_eta(float(eta_total))}"
          except Exception:
            eta_suffix = ""
        if eta_suffix and isinstance(eta_hints_raw, list) and eta_hints_raw:
          hints = [str(x).strip() for x in eta_hints_raw if str(x).strip()]
          if hints:
            eta_suffix += f" | hints: {','.join(hints)}"
        if timings:
          cur["message"] = f"{msg}{eta_suffix} || timings: {timings}"
        else:
          cur["message"] = f"{msg}{eta_suffix}"

      _write_json_atomic(status_path, cur)
    finally:
      fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
