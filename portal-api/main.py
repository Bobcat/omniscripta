from __future__ import annotations

import json
import os
import shutil
import mimetypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse, parse_qs

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.responses import FileResponse, Response
from queue_fs import init_job_in_inbox, JobPaths, BASE as BASE_JOBS

ROOT_PATH = os.getenv("TRANSCRIBE_ROOT_PATH", "/api")
app = FastAPI(root_path=ROOT_PATH)


DEFAULT_CALIBRATION_SNIPPET_SECONDS = [60, 180, 300, 480, 600, 1200, 1800, 2700, 3600]


@app.get("/health")
def health() -> Dict[str, bool]:
    return {"ok": True}


def _repo_root() -> Path:
    # portal-api/main.py -> portal-api -> repo root
    return Path(__file__).resolve().parents[1]


def _resolve_json_config_path(env_key: str, default_rel: str) -> Path:
    raw = (os.getenv(env_key) or "").strip()
    if raw:
        p = Path(raw)
        if p.is_absolute():
            return p
        return (_repo_root() / p).resolve()
    return (_repo_root() / default_rel).resolve()


def _iso_utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_sensitive_key(name: str) -> bool:
    k = str(name or "").strip().lower()
    if not k or k.endswith("_env"):
        return False
    if k in {
        "token",
        "hf_token",
        "api_key",
        "apikey",
        "password",
        "secret",
        "access_token",
        "refresh_token",
        "authorization",
        "bearer_token",
    }:
        return True
    return (
        k.endswith("_token")
        or k.endswith("_api_key")
        or k.endswith("_apikey")
        or k.endswith("_password")
        or k.endswith("_secret")
    )


def _redact_sensitive(value: Any) -> Any:
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, child in value.items():
            if _is_sensitive_key(str(key)):
                out[str(key)] = "***REDACTED***"
            else:
                out[str(key)] = _redact_sensitive(child)
        return out
    if isinstance(value, list):
        return [_redact_sensitive(v) for v in value]
    return value


def _read_config_source(*, source_id: str, title: str, path: Path) -> Dict[str, Any]:
    item: Dict[str, Any] = {
        "id": source_id,
        "title": title,
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": None,
        "mtime_utc": None,
        "parse_ok": False,
        "data": None,
        "error": None,
    }
    if not path.exists():
        item["error"] = "file_not_found"
        return item
    try:
        st = path.stat()
        item["size_bytes"] = int(st.st_size)
        item["mtime_utc"] = _iso_utc(st.st_mtime)
    except Exception:
        pass
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        item["data"] = _redact_sensitive(obj)
        item["parse_ok"] = True
    except Exception as e:
        item["error"] = f"{type(e).__name__}: {e}"
    return item


@app.get("/demo/settings")
def get_demo_settings() -> Dict[str, Any]:
    service_path = _resolve_json_config_path("TRANSCRIBE_SERVICE_CONFIG", "config/service.json")
    whisperx_path = _resolve_json_config_path("TRANSCRIBE_WHISPERX_CONFIG", "config/whisperx.json")
    return {
        "generated_at_utc": _iso_utc(datetime.now(timezone.utc).timestamp()),
        "sources": [
            _read_config_source(source_id="service", title="service.json", path=service_path),
            _read_config_source(source_id="whisperx", title="whisperx.json", path=whisperx_path),
        ],
    }


def _safe_filename(name: str) -> str:
    # Voorkomt path traversal zoals ../../etc/passwd
    return Path(name).name or "upload.bin"


def _find_job_dir(job_id: str) -> Path | None:
    for state in ("inbox", "running", "done", "error"):
        d = BASE_JOBS / state / job_id
        if d.exists():
            return d
    return None


def _as_bool(raw: Any) -> bool:
    s = str(raw or "").strip().lower()
    return s in {"1", "true", "yes", "on"}


def _param_from_request_or_referer(request: Request, key: str) -> str:
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


def _parse_calibration_seconds_env() -> list[int]:
    raw = str(os.getenv("TRANSCRIBE_CALIBRATION_SECONDS", "") or "").strip()
    if not raw:
        return list(DEFAULT_CALIBRATION_SNIPPET_SECONDS)

    vals: list[int] = []
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        try:
            v = int(p)
        except ValueError:
            continue
        if v > 0 and v not in vals:
            vals.append(v)
    return vals or list(DEFAULT_CALIBRATION_SNIPPET_SECONDS)


def _calibration_requested(request: Request) -> bool:
    q = _param_from_request_or_referer(request, "calibration")
    return _as_bool(q) if q else False


def _snip_seconds_override(request: Request) -> int | None:
    raw = _param_from_request_or_referer(request, "snip")
    if not raw:
        return None
    try:
        minutes = int(raw)
    except ValueError:
        raise HTTPException(status_code=400, detail="snip must be an integer number of minutes")
    if minutes < 1 or minutes > 720:
        raise HTTPException(status_code=400, detail="snip out of range (1..720 minutes)")
    return int(minutes * 60)


@app.post("/demo/jobs")
def create_demo_job(
    request: Request,
    file: UploadFile = File(...),
    language: str = Form("nl"),
    speakers: str = Form("none"),  # "none", "auto" of bv. "4"
) -> Dict[str, Any]:
    orig_name = _safe_filename(file.filename or "")
    if not orig_name:
        raise HTTPException(status_code=400, detail="Missing filename")

    sp = (speakers or "none").strip().lower()

    speaker_mode = "none"
    expected_speakers = None
    min_speakers = None
    max_speakers = None

    if sp in {"none", "off", "disabled", "no_speaker", "nospeaker", "no-speaker"}:
        speaker_mode = "none"
    elif sp == "auto":
        speaker_mode = "auto"
    else:
        try:
            s = int(sp)
        except ValueError:
            raise HTTPException(status_code=400, detail="speakers must be 'none', 'auto' or an integer")

        if s < 1 or s > 32:
            raise HTTPException(status_code=400, detail="speakers out of range (1..32)")

        speaker_mode = "fixed"
        expected_speakers = s
        min_speakers = max(1, s - 1)
        max_speakers = min(32, s + 2)

    base_options: Dict[str, Any] = {
        "language": language,
        "speaker_mode": speaker_mode,
        "expected_speakers": expected_speakers,
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
    }
    snippet_seconds_override = _snip_seconds_override(request)
    if snippet_seconds_override is not None:
        base_options["snippet_seconds"] = int(snippet_seconds_override)

    calibration_enabled = _calibration_requested(request)
    calibration_seconds = _parse_calibration_seconds_env() if calibration_enabled else []

    # Primary job (the one returned to frontend)
    jp: JobPaths = init_job_in_inbox(
        orig_filename=orig_name,
        options=dict(base_options),
    )

    # Schrijf upload naar job upload/
    dst_primary = jp.upload_dir / orig_name
    try:
        with dst_primary.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        file.file.close()

    extra_job_ids: list[str] = []
    extra_failed: list[str] = []
    if calibration_enabled:
        sec_list = list(calibration_seconds)
        if snippet_seconds_override is not None:
            sec_list = [s for s in sec_list if int(s) != int(snippet_seconds_override)]
        for sec in sec_list:
            try:
                opts = dict(base_options)
                opts["snippet_seconds"] = int(sec)
                extra = init_job_in_inbox(orig_filename=orig_name, options=opts)
                dst_extra = extra.upload_dir / orig_name
                shutil.copy2(dst_primary, dst_extra)
                extra_job_ids.append(extra.job_id)
            except Exception as e:
                extra_failed.append(f"{sec}s:{type(e).__name__}")

    return {
        "job_id": jp.job_id,
        "state": "queued",
        "calibration_enqueued": len(extra_job_ids),
        "calibration_seconds": calibration_seconds if calibration_enabled else [],
        "calibration_failed": extra_failed,
        "snippet_seconds": base_options.get("snippet_seconds"),
    }


@app.get("/demo/jobs/{job_id}")
def get_demo_job(job_id: str) -> Dict[str, Any]:
    """Return status.json for a job, wherever it currently lives."""
    base = BASE_JOBS
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
