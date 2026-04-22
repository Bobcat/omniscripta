from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from urllib.parse import parse_qs, urlparse

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from upload._util import _normalize_upload_language, _resolve_status_owner
from upload.jobs.queue_fs import JobPaths, init_job_in_inbox
from upload.queue_roots import UPLOAD_PREP_QUEUE
router = APIRouter()

NO_SPEAKER_VALUES = {"none", "off", "disabled", "no_speaker", "nospeaker", "no-speaker"}
TOPICS_ENABLED_VALUES = {"1", "true", "yes", "on", "enabled"}
TOPICS_DISABLED_VALUES = {"0", "false", "no", "off", "disabled"}


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
        jp: JobPaths = init_job_in_inbox(
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
            "job_id": jp.job_id,
            "state": "queued",
            "snippet_seconds": base_options.get("snippet_seconds"),
            "topics_enabled": base_options.get("topics_enabled"),
        }
    finally:
        try:
            staged_upload_path.unlink(missing_ok=True)
        except Exception:
            pass
