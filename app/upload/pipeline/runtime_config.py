from __future__ import annotations

import socket
from pathlib import Path

from app.config.settings import get_str

_REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_cfg_path(path_value: str, *, fallback_rel: str) -> Path:
    raw = str(path_value or "").strip() or fallback_rel
    path = Path(raw)
    return path if path.is_absolute() else (_REPO_ROOT / path)


def progress_runs_path() -> Path:
    raw = get_str("upload.worker.progress_runs_path", "").strip()
    if raw:
        return resolve_cfg_path(raw, fallback_rel="data/upload/progress_db/runs_v1.jsonl")
    base = resolve_cfg_path(
        get_str("upload.worker.progress_db_dir", "data/upload/progress_db"),
        fallback_rel="data/upload/progress_db",
    )
    return (base / "runs_v1.jsonl").resolve()


def host_id() -> str:
    raw = get_str("upload.worker.host_id", "").strip()
    if raw:
        return raw
    return socket.gethostname().split(".")[0]


def hardware_key(current_host_id: str) -> str:
    raw = get_str("upload.worker.hardware_key", "").strip()
    if raw:
        return raw
    if current_host_id == "dc1":
        return "dc1-rtx5070ti-cuda"
    if current_host_id == "dc2":
        return "dc2-rtx5090-cuda"
    return f"{current_host_id}-unknown"
