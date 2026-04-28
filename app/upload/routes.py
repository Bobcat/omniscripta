from __future__ import annotations

import io
import json
import mimetypes
import secrets
import shutil
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from urllib.parse import parse_qs, urlparse

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response

from app.config.settings import get_int, get_str
from upload._util import (
    _normalize_speaker_mode,
    _normalize_upload_language,
    _read_json,
    _resolve_child_path,
    _resolve_status_owner,
    _topics_enabled_for_job,
    _topics_merged_filename,
    _topics_prompt_id,
    _write_bytes_atomic,
    _write_json_atomic,
)
from upload.jobs.queue_fs import JobPaths, find_job_dir, init_job_in_inbox
from upload.jobs.request import read_upload_request
from upload.jobs.roots import UPLOAD_PREP_QUEUE, UPLOAD_WORKER_QUEUE
from upload.jobs.status import project_status_message
from upload.pipeline.progress_plan import build_prediction, phase_order_for_job
from upload.pipeline.runtime_config import hardware_key, host_id, progress_runs_path

router = APIRouter()

_REPO_ROOT = Path(__file__).resolve().parents[2]

NO_SPEAKER_VALUES = {"none", "off", "disabled", "no_speaker", "nospeaker", "no-speaker"}
TOPICS_ENABLED_VALUES = {"1", "true", "yes", "on", "enabled"}
TOPICS_DISABLED_VALUES = {"0", "false", "no", "off", "disabled"}
EXPORT_ALLOWED_FORMATS = {
    "txt": "text/plain; charset=utf-8",
    "srt": "application/x-subrip; charset=utf-8",
}


def _parse_speaker_options(speakers: str) -> dict[str, Any]:
    speaker_value = (speakers or "none").strip().lower()
    if speaker_value in NO_SPEAKER_VALUES:
        return {
            "speaker_mode": "none",
            "expected_speakers": None,
            "min_speakers": None,
            "max_speakers": None,
        }
    if speaker_value == "auto":
        return {
            "speaker_mode": "auto",
            "expected_speakers": None,
            "min_speakers": None,
            "max_speakers": None,
        }

    try:
        expected_speakers = int(speaker_value)
    except ValueError:
        raise HTTPException(status_code=400, detail="speakers must be 'none', 'auto' or an integer")
    if expected_speakers < 1 or expected_speakers > 32:
        raise HTTPException(status_code=400, detail="speakers out of range (1..32)")

    return {
        "speaker_mode": "fixed",
        "expected_speakers": expected_speakers,
        "min_speakers": max(1, expected_speakers - 1),
        "max_speakers": min(32, expected_speakers + 2),
    }


def _parse_topics_enabled(topics_enabled: str | None) -> bool:
    raw = str(topics_enabled or "").strip().lower()
    if not raw:
        return True
    if raw in TOPICS_ENABLED_VALUES:
        return True
    if raw in TOPICS_DISABLED_VALUES:
        return False
    raise HTTPException(status_code=400, detail="topics_enabled must be enabled/disabled or true/false")


def _snip_seconds_override(request: Request) -> int | None:
    raw = request.query_params.get("snip")
    if raw is None:
        ref = request.headers.get("referer") or request.headers.get("referrer")
        if ref:
            try:
                vals = parse_qs(urlparse(ref).query or "").get("snip") or []
                raw = vals[0] if vals else None
            except Exception:
                raw = None
    raw = str(raw or "").strip()
    if not raw:
        return None
    try:
        minutes = int(raw)
    except ValueError:
        raise HTTPException(status_code=400, detail="snip must be an integer number of minutes")
    if minutes < 1 or minutes > 720:
        raise HTTPException(status_code=400, detail="snip out of range (1..720 minutes)")
    return int(minutes * 60)


def _job_dir(job_id: str) -> Path:
    job_dir = find_job_dir(job_id, queue_roots=(UPLOAD_PREP_QUEUE, UPLOAD_WORKER_QUEUE))
    if job_dir is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job_dir


def _read_job_status(job_dir: Path) -> dict[str, Any]:
    status_path = job_dir / "status.json"
    if not status_path.exists():
        raise HTTPException(status_code=404, detail="Job status not found")
    try:
        return json.loads(status_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read status.json: {exc!r}")


def _queue_position(job_dir: Path, *, inbox_dir: Path) -> int | None:
    try:
        candidates = sorted(p for p in inbox_dir.iterdir() if p.is_dir() and not p.name.startswith(".tmp_"))
    except Exception:
        return None
    for idx, candidate in enumerate(candidates, start=1):
        try:
            if candidate.resolve() == job_dir.resolve():
                return idx
        except Exception:
            continue
    return None


def _snippet_seconds_for_projection(*, status: dict[str, Any], job_cfg: dict[str, Any] | None) -> int:
    try:
        status_value = int(status.get("snippet_seconds"))
    except Exception:
        status_value = None
    if status_value is not None and status_value > 0:
        return int(status_value)
    cfg = dict((job_cfg or {}).get("options") or {})
    try:
        cfg_value = int(cfg.get("snippet_seconds"))
    except Exception:
        cfg_value = None
    if cfg_value is not None and cfg_value > 0:
        return int(cfg_value)
    return 15 * 60


def _unique_hints(*values: Any) -> list[str]:
    out: list[str] = []
    for raw_values in values:
        if not isinstance(raw_values, list):
            continue
        for raw in raw_values:
            item = str(raw).strip()
            if item and item not in out:
                out.append(item)
    return out


def _project_asr_progress(status: dict[str, Any], *, job_cfg: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(status or {})
    if out.get("topics_status") is not None:
        return out
    state = str(out.get("state") or "").strip().lower()
    if state not in {"running", "done"}:
        return out
    phase = str(out.get("phase") or "").strip().lower()
    if phase in {"topics", "error"}:
        return out

    raw_asr_progress = out.get("asr_progress")
    if raw_asr_progress is None:
        return out

    try:
        asr_progress = min(1.0, max(0.0, float(raw_asr_progress)))
    except Exception:
        return out

    topics_enabled = _topics_enabled_for_job(status=out, job_cfg=job_cfg)
    if topics_enabled is None:
        topics_enabled = True
    speaker_mode = _normalize_speaker_mode(out.get("speaker_mode", "auto"))
    snippet_seconds = _snippet_seconds_for_projection(status=out, job_cfg=job_cfg)
    prediction = build_prediction(
        runs_path=progress_runs_path(),
        hardware_key=hardware_key(host_id()),
        topics_enabled=bool(topics_enabled),
        speaker_mode=speaker_mode,
        snippet_seconds=snippet_seconds,
    )
    phase_order = phase_order_for_job(topics_enabled=bool(topics_enabled), speaker_mode=speaker_mode)
    total_expected = max(1.0, float(prediction.total_expected_s))
    completed_before_asr = max(0.0, float(prediction.phase_expected_s.get("snipping", 0.0)))
    asr_expected = sum(
        max(0.0, float(prediction.phase_expected_s.get(phase_name, 0.0)))
        for phase_name in phase_order
        if phase_name.startswith("whisperx_")
    )
    topics_phase = phase_order[-1]
    topics_expected = max(0.0, float(prediction.phase_expected_s.get(topics_phase, 0.0)))
    try:
        base_elapsed = max(0.0, float(out.get("elapsed_s") or 0.0))
    except Exception:
        base_elapsed = 0.0
    try:
        asr_elapsed = max(0.0, float(out.get("asr_elapsed_s") or 0.0))
    except Exception:
        asr_elapsed = 0.0
    if asr_elapsed <= 0.0:
        try:
            asr_eta_total = float(out.get("asr_eta_total_s") or 0.0)
            asr_eta_remaining = float(out.get("asr_eta_remaining_s") or 0.0)
            asr_elapsed = max(0.0, asr_eta_total - asr_eta_remaining)
        except Exception:
            asr_elapsed = 0.0
    try:
        asr_remaining = max(0.0, float(out.get("asr_eta_remaining_s") or 0.0))
    except Exception:
        asr_remaining = 0.0
    projected = (completed_before_asr + (asr_progress * asr_expected)) / total_expected
    out["progress"] = min(0.99, max(0.0, float(projected)))
    upload_elapsed = max(0.0, base_elapsed + asr_elapsed)
    upload_remaining = max(0.0, asr_remaining + topics_expected)
    out["elapsed_s"] = round(upload_elapsed, 3)
    out["eta_remaining_s"] = round(upload_remaining, 3)
    out["eta_total_s"] = round(upload_elapsed + upload_remaining, 3)
    try:
        out["eta_confidence"] = round(float(out.get("asr_eta_confidence")), 3)
    except Exception:
        out["eta_confidence"] = round(float(prediction.confidence), 3)
    out["eta_hints"] = _unique_hints(out.get("asr_eta_hints"), prediction.hints)
    return out


def _project_upload_ui_status(status: dict[str, Any], *, job_dir: Path | None = None) -> dict[str, Any]:
    """Keep the upload UI on an in-progress topics phase until topics_status exists."""
    out = dict(status or {})
    job_cfg = read_upload_request(job_dir) if job_dir is not None else None
    out = _project_asr_progress(out, job_cfg=job_cfg)
    if out.get("topics_enabled") is None and job_cfg is not None:
        topics_enabled = _topics_enabled_for_job(status=out, job_cfg=job_cfg)
        if topics_enabled is not None:
            out["topics_enabled"] = topics_enabled
    state = str(out.get("state") or "").strip().lower()
    phase = str(out.get("phase") or "").strip().lower()
    topics_status = out.get("topics_status")
    if state == "done" and phase == "done" and topics_status is None:
        out["state"] = "running"
        out["phase"] = "topics"
        out["status_owner"] = _resolve_status_owner(key="api_topics", default="api-topics")
        out["subphase"] = str(out.get("subphase") or "wait")
        out["progress"] = min(0.99, max(0.0, float(out.get("progress") or 0.0)))
        if not str(out.get("message") or "").strip().lower().startswith("topics:"):
            out["message"] = "Topics: processing"
    out = project_status_message(out)
    if job_dir is not None and state == "queued":
        try:
            in_upload_worker_inbox = job_dir.parent.resolve() == UPLOAD_WORKER_QUEUE.inbox.resolve()
        except Exception:
            in_upload_worker_inbox = False
        if in_upload_worker_inbox:
            pos = _queue_position(job_dir, inbox_dir=UPLOAD_WORKER_QUEUE.inbox)
            if pos is not None:
                out["queue_position"] = pos
                out["message"] = (
                    f"Waiting for ASR (position {pos} in queue)"
                    if pos > 1
                    else "Waiting for ASR (next in queue)"
                )
    return out


def _resolve_job_artifact(
    job_dir: Path,
    *,
    subdir: str,
    filename: str,
    invalid_detail: str,
    missing_detail: str,
) -> Path:
    artifact_dir = (job_dir / subdir).resolve()
    artifact_path = _resolve_child_path(artifact_dir, filename)
    if artifact_path is None:
        raise HTTPException(status_code=400, detail=invalid_detail)
    if not artifact_path.exists():
        raise HTTPException(status_code=404, detail=missing_detail)
    return artifact_path


def _load_job_artifact(
    job_id: str,
    *,
    status_field: str,
    not_ready_detail: str,
    subdir: str,
    invalid_detail: str,
    missing_detail: str,
) -> tuple[Path, dict[str, Any], Path]:
    job_dir = _job_dir(job_id)
    status = _read_job_status(job_dir)
    artifact_name = str(status.get(status_field) or "").strip()
    if not artifact_name:
        raise HTTPException(status_code=409, detail=not_ready_detail)
    artifact_path = _resolve_job_artifact(
        job_dir,
        subdir=subdir,
        filename=artifact_name,
        invalid_detail=invalid_detail,
        missing_detail=missing_detail,
    )
    return job_dir, status, artifact_path


def _rooted_api_path(path: str) -> str:
    root = str(get_str("service.root_path", "/api") or "/api").strip()
    root = "/" + root.strip("/")
    return f"{root}/" + str(path or "").strip("/")


def _export_root_dir() -> Path:
    raw = str(get_str("upload.export.root", "data/upload/exports") or "").strip() or "data/upload/exports"
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


@router.post("/demo/jobs")
def create_demo_job(
    request: Request,
    file: UploadFile = File(...),
    language: str = Form(""),
    speakers: str = Form("none"),
    align: str = Form("disabled"),
    topics_enabled: str = Form("enabled"),
) -> dict[str, Any]:
    orig_name = Path(file.filename or "").name or "upload.bin"

    base_options: dict[str, Any] = {
        "language": _normalize_upload_language(language),
        **_parse_speaker_options(speakers),
        "align_enabled": str(align or "").strip().lower() == "enabled",
        "topics_enabled": _parse_topics_enabled(topics_enabled),
    }
    snippet_seconds_override = _snip_seconds_override(request)
    if snippet_seconds_override is not None:
        base_options["snippet_seconds"] = snippet_seconds_override

    staging_dir = (UPLOAD_PREP_QUEUE.base / "_staging_uploads").resolve()
    staging_dir.mkdir(parents=True, exist_ok=True)
    staged_upload_path: Path | None = None
    try:
        with NamedTemporaryFile(prefix="upload_", suffix=".bin", dir=str(staging_dir), delete=False) as tmp_f:
            shutil.copyfileobj(file.file, tmp_f)
            staged_upload_path = Path(tmp_f.name).resolve()
    finally:
        file.file.close()

    if staged_upload_path is None or not staged_upload_path.exists():
        raise HTTPException(status_code=500, detail="Failed to stage upload file")

    try:
        upload_request = {
            "orig_filename": orig_name,
            "options": dict(base_options),
        }
        job_paths: JobPaths = init_job_in_inbox(
            queue_root=UPLOAD_PREP_QUEUE,
            job_json={"kind": "upload_worker_payload_pending"},
            status_json={
                "state": "queued",
                "phase": "upload",
                "status_owner": _resolve_status_owner(key="api", default="api"),
                "progress": 0.0,
                "message": "Queued",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "started_at": None,
                "finished_at": None,
                "error": None,
                "orig_filename": orig_name,
                **{k: base_options.get(k) for k in (
                    "language",
                    "speaker_mode",
                    "expected_speakers",
                    "min_speakers",
                    "max_speakers",
                    "snippet_seconds",
                    "align_enabled",
                    "topics_enabled",
                )},
                "asr_input_relpath": None,
                "snippet_filename": None,
                "srt_filename": None,
            },
            upload_request_json=upload_request,
            input_src_path=staged_upload_path,
            input_dst_relpath=str(Path("upload") / orig_name),
            move_upload_src=False,
        )
        return {
            "job_id": job_paths.job_id,
            "state": "queued",
            "snippet_seconds": base_options.get("snippet_seconds"),
            "topics_enabled": base_options.get("topics_enabled"),
        }
    finally:
        try:
            staged_upload_path.unlink(missing_ok=True)
        except Exception:
            pass


@router.get("/demo/jobs/{job_id}")
def get_demo_job(job_id: str) -> dict[str, Any]:
    job_dir = _job_dir(job_id)
    return _project_upload_ui_status(_read_job_status(job_dir), job_dir=job_dir)


@router.get("/demo/jobs/{job_id}/snippet")
def get_demo_job_snippet(job_id: str):
    _, _, snippet_path = _load_job_artifact(
        job_id,
        status_field="snippet_filename",
        not_ready_detail="Snippet not ready",
        subdir="snippet",
        invalid_detail="Invalid snippet path",
        missing_detail="Snippet file missing",
    )

    media_type, _ = mimetypes.guess_type(snippet_path.name)
    headers = {"Content-Disposition": f'inline; filename="{snippet_path.name}"'}
    return FileResponse(
        path=str(snippet_path),
        media_type=media_type or "application/octet-stream",
        headers=headers,
    )


@router.get("/demo/jobs/{job_id}/transcript.srt")
def get_demo_job_srt(job_id: str):
    job_dir, status, srt_path = _load_job_artifact(
        job_id,
        status_field="srt_filename",
        not_ready_detail="Transcript not ready",
        subdir="whisperx",
        invalid_detail="Invalid transcript path",
        missing_detail="Transcript file missing",
    )

    srt_content = srt_path.read_text(encoding="utf-8")
    try:
        orig_filename = status.get("orig_filename")
        if orig_filename:
            base = Path(orig_filename).stem
            topics_prompt_id = _topics_prompt_id(get_str("upload.topics.prompt_id", "topics_v1"))
            topics_name = _topics_merged_filename(orig_stem=base, prompt_id=topics_prompt_id)
            topics_path = (job_dir / "result" / topics_name).resolve()
            if topics_path.exists():
                topics_data = json.loads(topics_path.read_text(encoding="utf-8"))
                if topics_data and "rows" in topics_data:
                    meta = {"topics": topics_data["rows"]}
                    srt_content += f"\n\n<!-- OMNISCRIPTA_META: {json.dumps(meta)} -->"
    except Exception as exc:
        print(f"Failed to inject topics: {exc}")

    headers = {"Content-Disposition": f'inline; filename="{srt_path.name}"'}
    return Response(
        content=srt_content,
        media_type="application/x-subrip",
        headers=headers,
    )


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
        with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_file:
            for item in artifacts:
                member_name = _safe_export_name(f"{base_name}.{item['format']}")
                zip_file.writestr(member_name, item["text"])
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
