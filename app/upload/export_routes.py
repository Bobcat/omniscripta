from __future__ import annotations

import io
import secrets
import time
import zipfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from app.config.settings import get_int, get_str
from upload._util import _read_json, _resolve_child_path, _write_bytes_atomic, _write_json_atomic

router = APIRouter()

_REPO_ROOT = Path(__file__).resolve().parents[2]

EXPORT_ALLOWED_FORMATS = {
    "txt": "text/plain; charset=utf-8",
    "srt": "application/x-subrip; charset=utf-8",
}


def _rooted_api_path(path: str) -> str:
    root = str(get_str("service.root_path", "/api") or "/api").strip()
    root = "/" + root.strip("/")
    return f"{root}/" + str(path or "").strip("/")


def _export_root_dir() -> Path:
    raw = str(get_str("upload.export.root", "data/exports") or "").strip() or "data/exports"
    root = Path(raw)
    if not root.is_absolute():
        root = (_REPO_ROOT / root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _export_ttl_s() -> int:
    return int(get_int("upload.export.ttl_s", 1800, min_value=60))


def _export_max_payload_bytes() -> int:
    return int(get_int("upload.export.max_payload_bytes", 8 * 1024 * 1024, min_value=4096))


def _safe_export_name(name: str) -> str:
    raw = Path(str(name or "").strip()).name or "transcript"
    clean = "".join(ch if (ch.isalnum() or ch in {"-", "_", "."}) else "_" for ch in raw).strip("._")
    return (clean or "transcript")[:160]


def _safe_export_base_name(name: str) -> str:
    base = _safe_export_name(name)
    for ext in (".txt", ".srt", ".zip"):
        if base.lower().endswith(ext):
            return base[: -len(ext)].rstrip("._") or "transcript"
    return base or "transcript"


def _unlink_if_present(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass
def _cleanup_expired_export_files(root: Path, *, now_unix: float) -> None:
    for meta_path in root.glob("*.meta.json"):
        try:
            meta = _read_json(meta_path)
        except Exception:
            meta = {}

        data_file = Path(str(meta.get("data_file") or "")).name
        data_path = _resolve_child_path(root, data_file) if data_file else None

        expires_unix = float(meta.get("expires_unix") or 0.0)
        expired = expires_unix <= 0.0 or expires_unix <= now_unix
        missing_data = data_path is not None and not data_path.exists()
        if not expired and not missing_data:
            continue

        _unlink_if_present(meta_path)
        _unlink_if_present(data_path)


@router.post("/demo/exports")
async def create_demo_export(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Export payload must be a JSON object")

    artifacts_raw = payload.get("artifacts")
    if not isinstance(artifacts_raw, list) or not artifacts_raw:
        raise HTTPException(status_code=400, detail="artifacts must be a non-empty list")
    if len(artifacts_raw) > len(EXPORT_ALLOWED_FORMATS):
        raise HTTPException(status_code=400, detail="Too many export artifacts")

    total_bytes = 0
    max_payload_bytes = _export_max_payload_bytes()
    artifacts: list[dict[str, Any]] = []
    seen_formats: set[str] = set()
    for idx, item in enumerate(artifacts_raw, start=1):
        if not isinstance(item, dict):
            raise HTTPException(status_code=400, detail=f"Artifact #{idx} must be an object")
        fmt = str(item.get("format") or "").strip().lower()
        if fmt not in EXPORT_ALLOWED_FORMATS:
            raise HTTPException(status_code=400, detail=f"Unsupported export format: {fmt or '<empty>'}")
        if fmt in seen_formats:
            raise HTTPException(status_code=400, detail=f"Duplicate export format: {fmt}")
        seen_formats.add(fmt)
        text = item.get("text")
        if text is None:
            text = ""
        if not isinstance(text, str):
            text = str(text)
        encoded = text.encode("utf-8")
        total_bytes += len(encoded)
        if total_bytes > max_payload_bytes:
            raise HTTPException(status_code=413, detail="Export payload too large")
        artifacts.append({
            "format": fmt,
            "text": text,
            "bytes": encoded,
        })

    base_name = _safe_export_base_name(str(payload.get("base_name") or "transcript"))
    if len(artifacts) == 1:
        only = artifacts[0]
        out_filename = _safe_export_name(f"{base_name}.{only['format']}")
        out_media_type = EXPORT_ALLOWED_FORMATS[str(only["format"])]
        out_bytes = bytes(only["bytes"])
    else:
        out_filename = _safe_export_name(f"{base_name}.zip")
        out_media_type = "application/zip"
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for item in artifacts:
                member_name = _safe_export_name(f"{base_name}.{item['format']}")
                zf.writestr(member_name, item["text"])
        out_bytes = zip_buf.getvalue()

    now_unix = float(time.time())
    ttl_s = _export_ttl_s()
    expires_unix = now_unix + float(ttl_s)
    export_root = _export_root_dir()
    _cleanup_expired_export_files(export_root, now_unix=now_unix)

    export_id = secrets.token_urlsafe(18).replace("-", "").replace("_", "")
    data_path = (export_root / f"{export_id}.bin").resolve()
    meta_path = (export_root / f"{export_id}.meta.json").resolve()
    _write_bytes_atomic(data_path, out_bytes)
    _write_json_atomic(
        meta_path,
        {
            "export_id": export_id,
            "filename": out_filename,
            "media_type": out_media_type,
            "data_file": data_path.name,
            "created_unix": now_unix,
            "expires_unix": expires_unix,
        },
    )
    return {
        "export_id": export_id,
        "filename": out_filename,
        "download_url": _rooted_api_path(f"/demo/exports/{export_id}/{out_filename}"),
        "expires_in_s": ttl_s,
        "archive": len(artifacts) > 1,
    }


@router.get("/demo/exports/{export_id}/{filename}")
def get_demo_export(export_id: str, filename: str) -> FileResponse:
    safe_export_id = "".join(ch for ch in str(export_id or "") if ch.isalnum())
    if not safe_export_id:
        raise HTTPException(status_code=400, detail="Invalid export id")
    export_root = _export_root_dir()
    _cleanup_expired_export_files(export_root, now_unix=float(time.time()))

    meta_path = (export_root / f"{safe_export_id}.meta.json").resolve()
    try:
        meta = _read_json(meta_path) if meta_path.exists() else {}
    except Exception:
        meta = {}
    if not meta:
        raise HTTPException(status_code=404, detail="Export not found")

    expires_unix = float(meta.get("expires_unix") or 0.0)
    if expires_unix <= float(time.time()):
        _cleanup_expired_export_files(export_root, now_unix=float(time.time()))
        raise HTTPException(status_code=404, detail="Export expired")

    stored_filename = _safe_export_name(str(meta.get("filename") or ""))
    if not stored_filename or filename != stored_filename:
        raise HTTPException(status_code=404, detail="Export not found")

    data_file = Path(str(meta.get("data_file") or f"{safe_export_id}.bin")).name
    data_path = _resolve_child_path(export_root, data_file)
    if data_path is None:
        raise HTTPException(status_code=404, detail="Export not found")
    if not data_path.exists():
        raise HTTPException(status_code=404, detail="Export file missing")

    media_type = str(meta.get("media_type") or "application/octet-stream")
    return FileResponse(
        path=str(data_path),
        media_type=media_type,
        filename=stored_filename,
    )
