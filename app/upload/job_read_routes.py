from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

from app.config.settings import get_str
from upload._util import (
    _resolve_child_path,
    _resolve_status_owner,
    _topics_merged_filename,
    _topics_prompt_id,
    _topics_enabled_for_job,
)
from upload.jobs.queue_fs import find_job_dir
from upload.queue_roots import UPLOAD_PREP_QUEUE, UPLOAD_WORKER_QUEUE
from upload.upload_request_io import read_upload_request

router = APIRouter()


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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read status.json: {e!r}")


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


def _project_upload_ui_status(status: dict[str, Any], *, job_dir: Path | None = None) -> dict[str, Any]:
    """Keep the upload UI on an in-progress topics phase until topics_status exists."""
    out = dict(status or {})
    if out.get("topics_enabled") is None and job_dir is not None:
        topics_enabled = _topics_enabled_for_job(status=out, job_cfg=read_upload_request(job_dir))
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
    except Exception as e:
        print(f"Failed to inject topics: {e}")

    headers = {"Content-Disposition": f'inline; filename="{srt_path.name}"'}
    return Response(
        content=srt_content,
        media_type="application/x-subrip",
        headers=headers,
    )
