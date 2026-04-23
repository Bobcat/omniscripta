from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.config.settings import get_str
from upload.jobs.queue_fs import QueueRoot


def _resolve_base(path_value: str, *, default_rel: str) -> Path:
    raw = str(path_value or "").strip() or default_rel
    p = Path(raw)
    return p if p.is_absolute() else (_REPO_ROOT / p).resolve()


def _queue_root(name: str, *, setting_path: str, default_rel: str) -> QueueRoot:
    base = _resolve_base(get_str(setting_path, default_rel), default_rel=default_rel)
    return QueueRoot(
        name=str(name),
        base=base,
        inbox=base / "inbox",
        running=base / "running",
        done=base / "done",
        error=base / "error",
    )


UPLOAD_PREP_QUEUE = _queue_root(
    "upload_prep",
    setting_path="upload.queue.prep_base",
    default_rel="data/upload/jobs/prep",
)
UPLOAD_WORKER_QUEUE = _queue_root(
    "upload_worker",
    setting_path="upload.queue.worker_base",
    default_rel="data/upload/jobs/worker",
)
