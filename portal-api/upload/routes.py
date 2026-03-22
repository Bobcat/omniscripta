from __future__ import annotations

import json
import mimetypes
import shutil
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from urllib.parse import parse_qs, urlparse

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response

from jobs.queue_fs import JobPaths, find_job_dir, init_job_in_inbox
from queue_roots import UPLOAD_PREP_QUEUE, UPLOAD_WORKER_QUEUE

router = APIRouter()

NO_SPEAKER_VALUES = {"none", "off", "disabled", "no_speaker", "nospeaker", "no-speaker"}


def _job_dir(job_id: str) -> Path:
    job_dir = find_job_dir(
        job_id,
        queue_roots=(UPLOAD_PREP_QUEUE, UPLOAD_WORKER_QUEUE),
    )
    if job_dir is not None:
        return job_dir
    raise HTTPException(status_code=404, detail="Job not found")


def _read_job_status(job_dir: Path) -> dict[str, Any]:
    status_path = job_dir / "status.json"
    if not status_path.exists():
        raise HTTPException(status_code=404, detail="Job status not found")
    try:
        return json.loads(status_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read status.json: {e!r}")


def _project_upload_ui_status(status: dict[str, Any]) -> dict[str, Any]:
    """
    Worker now writes done as soon as ASR is done.
    For upload UI, keep showing an in-progress topics phase until topics_status is present.
    """
    out = dict(status or {})
    state = str(out.get("state") or "").strip().lower()
    phase = str(out.get("phase") or "").strip().lower()
    topics_status = out.get("topics_status")
    if state == "done" and phase == "done" and topics_status is None:
        out["state"] = "running"
        out["phase"] = "topics"
        out["subphase"] = str(out.get("subphase") or "wait")
        out["progress"] = min(0.99, max(0.0, float(out.get("progress") or 0.0)))
        if not str(out.get("message") or "").strip().lower().startswith("topics:"):
            out["message"] = "Topics: processing"
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
    artifact_path = (artifact_dir / filename).resolve()
    try:
        artifact_path.relative_to(artifact_dir)
    except ValueError:
        raise HTTPException(status_code=400, detail=invalid_detail)
    if not artifact_path.exists():
        raise HTTPException(status_code=404, detail=missing_detail)
    return artifact_path


def _request_param(request: Request, key: str) -> str:
    qv = request.query_params.get(key)
    if qv is not None:
        return str(qv).strip()

    ref = request.headers.get("referer") or request.headers.get("referrer")
    if ref:
        try:
            parsed = urlparse(ref)
            ref_q = parse_qs(parsed.query or "")
            vals = ref_q.get(key) or []
            if vals:
                return str(vals[0]).strip()
        except Exception:
            pass
    return ""


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


def _snip_seconds_override(request: Request) -> int | None:
    raw = _request_param(request, "snip")
    if not raw:
        return None
    try:
        minutes = int(raw)
    except ValueError:
        raise HTTPException(status_code=400, detail="snip must be an integer number of minutes")
    if minutes < 1 or minutes > 720:
        raise HTTPException(status_code=400, detail="snip out of range (1..720 minutes)")
    return int(minutes * 60)


@router.post("/demo/jobs")
def create_demo_job(
    request: Request,
    file: UploadFile = File(...),
    language: str = Form("nl"),
    speakers: str = Form("none"),
) -> dict[str, Any]:
    orig_name = Path(file.filename or "").name or "upload.bin"

    base_options: dict[str, Any] = {
        "language": language,
        **_parse_speaker_options(speakers),
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
        jp: JobPaths = init_job_in_inbox(
            queue_root=UPLOAD_PREP_QUEUE,
            job_json={
                "orig_filename": orig_name,
                "options": dict(base_options),
            },
            status_json={
                "state": "queued",
                "phase": "upload",
                "progress": 0.0,
                "message": "Queued",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "started_at": None,
                "finished_at": None,
                "error": None,
                "orig_filename": orig_name,
                "language": base_options.get("language"),
                "speaker_mode": base_options.get("speaker_mode"),
                "expected_speakers": base_options.get("expected_speakers"),
                "min_speakers": base_options.get("min_speakers"),
                "max_speakers": base_options.get("max_speakers"),
                "snippet_seconds": base_options.get("snippet_seconds"),
                "asr_input_relpath": None,
                "snippet_filename": None,
                "srt_filename": None,
            },
            input_src_path=staged_upload_path,
            input_dst_relpath=str(Path("upload") / orig_name),
            move_upload_src=False,
        )
        return {
            "job_id": jp.job_id,
            "state": "queued",
            "snippet_seconds": base_options.get("snippet_seconds"),
        }
    finally:
        try:
            staged_upload_path.unlink(missing_ok=True)
        except Exception:
            pass


@router.get("/demo/jobs/{job_id}")
def get_demo_job(job_id: str) -> dict[str, Any]:
    return _project_upload_ui_status(_read_job_status(_job_dir(job_id)))


@router.get("/demo/jobs/{job_id}/snippet")
def get_demo_job_snippet(job_id: str):
    job_dir = _job_dir(job_id)
    status = _read_job_status(job_dir)
    snippet_name = str(status.get("snippet_filename") or "").strip()
    if not snippet_name:
        raise HTTPException(status_code=409, detail="Snippet not ready")

    snippet_path = _resolve_job_artifact(
        job_dir,
        subdir="snippet",
        filename=snippet_name,
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
    job_dir = _job_dir(job_id)
    status = _read_job_status(job_dir)
    srt_name = str(status.get("srt_filename") or "").strip()
    if not srt_name:
        raise HTTPException(status_code=409, detail="Transcript not ready")

    srt_path = _resolve_job_artifact(
        job_dir,
        subdir="whisperx",
        filename=srt_name,
        invalid_detail="Invalid transcript path",
        missing_detail="Transcript file missing",
    )

    srt_content = srt_path.read_text(encoding="utf-8")
    try:
        orig_filename = status.get("orig_filename")
        if orig_filename:
            base = Path(orig_filename).stem
            topics_name = f"{base}_topics_v1_merged.json"
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
