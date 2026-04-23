from __future__ import annotations

import json
import os
import secrets
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from app.config.settings import get_str


def _repo_root() -> Path:
  # workers/llm/queue_fs.py -> llm -> workers -> repo root
  return Path(__file__).resolve().parents[2]


def _tasks_base() -> Path:
  raw = get_str("llm.queue_dir", "data/upload/llm/tasks").strip()
  p = Path(raw)
  return p if p.is_absolute() else (_repo_root() / p)


BASE = _tasks_base()
INBOX = BASE / "inbox"
RUNNING = BASE / "running"
DONE = BASE / "done"
ERROR = BASE / "error"


@dataclass(frozen=True)
class TaskPaths:
  task_id: str
  dir: Path
  output_dir: Path
  status_path: Path
  task_path: Path
  log_path: Path


def _utc_stamp() -> str:
  return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def new_task_id() -> str:
  return f"llm_{_utc_stamp()}_{secrets.token_hex(4)}"


def _write_json_atomic(path: Path, obj: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_suffix(path.suffix + ".tmp")
  tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
  os.replace(tmp, path)


def _task_kind_from_task_json(task_dir: Path) -> str:
  task_path = (task_dir / "task.json").resolve()
  try:
    raw = json.loads(task_path.read_text(encoding="utf-8"))
  except Exception:
    return ""
  if not isinstance(raw, dict):
    return ""
  return str(raw.get("task_kind") or "").strip().lower()


def init_task_in_inbox(
  *,
  task_kind: str = "prompt_run",
  spec: dict[str, Any] | None = None,
) -> TaskPaths:
  INBOX.mkdir(parents=True, exist_ok=True)

  task_id = new_task_id()
  final_dir = INBOX / task_id
  tmp_dir = INBOX / f".tmp_{task_id}"

  if tmp_dir.exists():
    raise RuntimeError(f"Temp dir already exists: {tmp_dir}")
  if final_dir.exists():
    raise RuntimeError(f"Task dir already exists: {final_dir}")

  output_dir = tmp_dir / "output"
  output_dir.mkdir(parents=True, exist_ok=True)

  status_path = tmp_dir / "status.json"
  task_path = tmp_dir / "task.json"
  log_path = tmp_dir / "worker.log"

  status = {
    "task_id": task_id,
    "task_kind": str(task_kind or "prompt_run"),
    "state": "queued",
    "phase": "queued",
    "progress": 0.0,
    "message": "Queued",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "started_at": None,
    "finished_at": None,
    "error": None,
  }
  _write_json_atomic(status_path, status)

  task_obj = {
    "task_id": task_id,
    "task_kind": str(task_kind or "prompt_run"),
    "spec": dict(spec or {}),
  }
  _write_json_atomic(task_path, task_obj)

  log_path.write_text("", encoding="utf-8")
  os.replace(tmp_dir, final_dir)

  return TaskPaths(
    task_id=task_id,
    dir=final_dir,
    output_dir=final_dir / "output",
    status_path=final_dir / "status.json",
    task_path=final_dir / "task.json",
    log_path=final_dir / "worker.log",
  )


def claim_next_task(*, task_kind_filter: str | None = None) -> TaskPaths | None:
  RUNNING.mkdir(parents=True, exist_ok=True)
  wanted = str(task_kind_filter or "").strip().lower()

  candidates = sorted(
    p for p in INBOX.iterdir()
    if p.is_dir() and not p.name.startswith(".tmp_")
  )
  if not candidates:
    return None

  for task_dir in candidates:
    if wanted:
      kind = _task_kind_from_task_json(task_dir)
      if kind != wanted:
        continue
    target = RUNNING / task_dir.name
    try:
      os.replace(task_dir, target)
    except FileNotFoundError:
      continue
    except OSError:
      continue
    return TaskPaths(
      task_id=target.name,
      dir=target,
      output_dir=target / "output",
      status_path=target / "status.json",
      task_path=target / "task.json",
      log_path=target / "worker.log",
    )

  return None


def finish_task(task: TaskPaths, *, ok: bool) -> Path:
  DONE.mkdir(parents=True, exist_ok=True)
  ERROR.mkdir(parents=True, exist_ok=True)
  src = task.dir
  dst_base = DONE if ok else ERROR
  dst = dst_base / src.name
  os.replace(src, dst)
  return dst
