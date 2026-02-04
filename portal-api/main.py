from __future__ import annotations

import json
import shutil
import mimetypes
from pathlib import Path
from typing import Any, Dict
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, Response
from queue_fs import init_job_in_inbox, JobPaths

app = FastAPI(root_path="/api")


@app.get("/health")
def health() -> Dict[str, bool]:
    return {"ok": True}


def _safe_filename(name: str) -> str:
    # Voorkomt path traversal zoals ../../etc/passwd
    return Path(name).name or "upload.bin"


BASE_JOBS = Path("/srv/transcribe/data/demo_jobs")

def _find_job_dir(job_id: str) -> Path | None:
    for state in ("inbox", "running", "done", "error"):
        d = BASE_JOBS / state / job_id
        if d.exists():
            return d
    return None


@app.post("/demo/jobs")
def create_demo_job(
    file: UploadFile = File(...),
    language: str = Form("nl"),
    speakers: str = Form("auto"),  # "auto" of bv. "4"
) -> Dict[str, Any]:
    orig_name = _safe_filename(file.filename or "")
    if not orig_name:
        raise HTTPException(status_code=400, detail="Missing filename")

    sp = (speakers or "auto").strip().lower()

    speaker_mode = "auto"
    expected_speakers = None
    min_speakers = None
    max_speakers = None

    if sp != "auto":
        try:
            s = int(sp)
        except ValueError:
            raise HTTPException(status_code=400, detail="speakers must be 'auto' or an integer")

        if s < 1 or s > 32:
            raise HTTPException(status_code=400, detail="speakers out of range (1..32)")

        speaker_mode = "fixed"
        expected_speakers = s
        min_speakers = max(1, s - 1)
        max_speakers = min(32, s + 2)

    # Maak jobdir in inbox (atomic publish)
    jp: JobPaths = init_job_in_inbox(
        orig_filename=orig_name,
        options={
            "language": language,
            "speaker_mode": speaker_mode,
            "expected_speakers": expected_speakers,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
        },
    )

    # Schrijf upload naar job upload/
    dst = jp.upload_dir / orig_name
    try:
        with dst.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        file.file.close()

    return {"job_id": jp.job_id, "state": "queued"}


@app.get("/demo/jobs/{job_id}")
def get_demo_job(job_id: str) -> Dict[str, Any]:
    """Return status.json for a job, wherever it currently lives."""
    base = Path("/srv/transcribe/data/demo_jobs")
    for state in ("inbox", "running", "done", "error"):
        status_path = base / state / job_id / "status.json"
        if status_path.exists():
            try:
                return json.loads(status_path.read_text(encoding="utf-8"))
            except Exception as e:
                # Debug-friendly for v0; later kun je dit verbergen
                raise HTTPException(status_code=500, detail=f"Failed to read status.json: {e!r}")

    raise HTTPException(status_code=404, detail="Job not found")


@app.get("/demo/jobs/{job_id}/snippet")
def get_demo_job_snippet(job_id: str):
    job_dir = _find_job_dir(job_id)
    if not job_dir:
        raise HTTPException(status_code=404, detail="Job not found")

    status_path = job_dir / "status.json"
    if not status_path.exists():
        raise HTTPException(status_code=404, detail="Job status not found")

    status = json.loads(status_path.read_text(encoding="utf-8"))
    snippet_name = status.get("snippet_filename")
    if not snippet_name:
        # job nog bezig of snip gefaald
        raise HTTPException(status_code=409, detail="Snippet not ready")

    snippet_dir = (job_dir / "snippet").resolve()
    snippet_path = (snippet_dir / snippet_name).resolve()

    # beveiliging tegen path traversal
    try:
        snippet_path.relative_to(snippet_dir)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid snippet path")

    if not snippet_path.exists():
        raise HTTPException(status_code=404, detail="Snippet file missing")

    media_type, _ = mimetypes.guess_type(snippet_path.name)
    headers = {"Content-Disposition": f'inline; filename="{snippet_path.name}"'}
    return FileResponse(
        path=str(snippet_path),
        media_type=media_type or "application/octet-stream",
        headers=headers,
    )


@app.get("/demo/jobs/{job_id}/transcript.srt")
def get_demo_job_srt(job_id: str):
    job_dir = _find_job_dir(job_id)
    if not job_dir:
        raise HTTPException(status_code=404, detail="Job not found")

    status_path = job_dir / "status.json"
    if not status_path.exists():
        raise HTTPException(status_code=404, detail="Job status not found")

    status = json.loads(status_path.read_text(encoding="utf-8"))
    srt_name = status.get("srt_filename")
    if not srt_name:
        raise HTTPException(status_code=409, detail="Transcript not ready")

    srt_dir = (job_dir / "whisperx").resolve()
    srt_path = (srt_dir / srt_name).resolve()

    # beveiliging tegen path traversal
    try:
        srt_path.relative_to(srt_dir)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid transcript path")

    if not srt_path.exists():
        raise HTTPException(status_code=404, detail="Transcript file missing")
    
    # Read SRT content
    srt_content = srt_path.read_text(encoding="utf-8")

    # Try to find and inject topics
    try:
        orig_filename = status.get("orig_filename")
        if orig_filename:
            # Construct topics filename: base_topics_v1_merged.json
            # e.g. "file.mp3" -> "file"
            base = Path(orig_filename).stem if "." in orig_filename else orig_filename
            topics_name = f"{base}_topics_v1_merged.json"
            topics_path = (job_dir / "result" / topics_name).resolve()
            
            if topics_path.exists():
                topics_data = json.loads(topics_path.read_text(encoding="utf-8"))
                # Only inject if we have rows
                if topics_data and "rows" in topics_data:
                    # Construct metadata block
                    meta = {"topics": topics_data["rows"]}
                    block = f"\n\n<!-- OMNISCRIPTA_META: {json.dumps(meta)} -->"
                    srt_content += block
    except Exception as e:
        # Don't fail the SRT download if topics fail, just log/ignore
        print(f"Failed to inject topics: {e}")

    headers = {"Content-Disposition": f'inline; filename="{srt_path.name}"'}
    # Return content directly instead of FileResponse since we modified it
    return Response(
        content=srt_content,
        media_type="application/x-subrip",
        headers=headers,
    )
