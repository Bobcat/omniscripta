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
