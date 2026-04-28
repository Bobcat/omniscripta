from __future__ import annotations

from pathlib import Path
from typing import Any

from upload._util import _read_json

UPLOAD_REQUEST_FILENAME = "upload_request.json"


def upload_request_path(job_dir: Path) -> Path:
    return (Path(job_dir).resolve() / UPLOAD_REQUEST_FILENAME).resolve()


def read_upload_request(job_dir: Path) -> dict[str, Any]:
    try:
        return _read_json(upload_request_path(job_dir))
    except Exception:
        return {}
