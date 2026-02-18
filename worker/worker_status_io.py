from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# Update status.json at most this often (seconds)
STATUS_THROTTLE_S = 1.0


def _utc_iso() -> str:
  return datetime.now(timezone.utc).isoformat()


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


def _append_log(log_path: Path, line: str) -> None:
  log_path.parent.mkdir(parents=True, exist_ok=True)
  with log_path.open("a", encoding="utf-8") as f:
    f.write(line.rstrip("\n") + "\n")


def _append_wx(log_path: Path, line: str) -> None:
  """Append a filtered WhisperX output line, prefixed with worker wallclock UTC time."""
  ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
  s = (line or "").rstrip("\n")
  _append_log(log_path, f"[{ts}] WX {s}")


def _read_json(p: Path) -> dict[str, Any]:
  return json.loads(p.read_text(encoding="utf-8"))


def _write_json(p: Path, obj: dict[str, Any]) -> None:
  p.parent.mkdir(parents=True, exist_ok=True)
  p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_status(status_path: Path, **patch: Any) -> None:
  """Patch-merge into status.json (simple shallow merge)."""
  cur: dict[str, Any] = {}
  if status_path.exists():
    try:
      cur = _read_json(status_path)
    except Exception:
      cur = {}
  cur.update({k: v for k, v in patch.items() if v is not None})

  # Keep UX compatible: append ETA + timings into message so existing frontend
  # surfaces predictive progress details without requiring UI changes.
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

  _write_json(status_path, cur)


@dataclass
class _StatusEmitter:
  status_path: Path
  last_write: float = 0.0
  last_progress: float = -1.0
  last_message: str = ""

  def maybe_emit(
    self,
    *,
    progress: float,
    message: str,
    phase: str,
    extra: Optional[dict[str, Any]] = None
  ) -> None:
    now = time.monotonic()
    progress = float(max(0.0, min(1.0, progress)))

    # Never go backwards within a job (prevents UI “reset”)
    if self.last_progress >= 0.0:
      progress = max(progress, self.last_progress)

    msg_changed = (message != self.last_message)
    prog_changed = (progress - self.last_progress) >= 0.002  # avoid noise
    if (now - self.last_write) < STATUS_THROTTLE_S and not msg_changed and not prog_changed:
      return

    patch: dict[str, Any] = {"progress": progress, "message": message, "phase": phase}
    if extra:
      patch.update(extra)

    _write_status(self.status_path, **patch)
    self.last_write = now
    self.last_progress = progress
    self.last_message = message
