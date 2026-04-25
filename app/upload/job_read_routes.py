from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

from app.config.settings import get_str
from upload._util import (
    _normalize_speaker_mode,
    _resolve_child_path,
    _resolve_status_owner,
    _topics_merged_filename,
    _topics_prompt_id,
    _topics_enabled_for_job,
)
from upload.pipeline.coordinator import _hardware_key, _host_id, _progress_runs_path
from upload.pipeline.progress_plan import build_prediction, phase_order_for_job
from upload.status_io import _fmt_eta, _timings_with_running_total
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


def _project_status_message(status: dict[str, Any]) -> dict[str, Any]:
    out = dict(status or {})
    msg = str(out.get("message") or "")
    if not msg:
        return out
    if " || eta: " in msg:
        msg = msg.split(" || eta: ", 1)[0]
    if " || timings: " in msg:
        msg = msg.split(" || timings: ", 1)[0]

    timings = str(out.get("timings_text", "") or "").strip()
    eta_total = out.get("eta_total_s")
    eta_remaining = out.get("eta_remaining_s")
    elapsed_s = out.get("elapsed_s")
    eta_hints = _unique_hints(out.get("eta_hints"))

    running_total_s: float | None = None
    if elapsed_s is not None:
        try:
            running_total_s = float(elapsed_s)
        except Exception:
            running_total_s = None
    elif eta_total is not None and eta_remaining is not None:
        try:
            running_total_s = float(eta_total) - float(eta_remaining)
        except Exception:
            running_total_s = None
    timings = _timings_with_running_total(timings, running_total_s)

    eta_suffix = ""
    if eta_total is not None and eta_remaining is not None:
        try:
            eta_suffix = f" || eta: ETA {_fmt_eta(float(eta_remaining))} | est_total {_fmt_eta(float(eta_total))}"
        except Exception:
            eta_suffix = ""
    if eta_suffix and eta_hints:
        eta_suffix += f" | hints: {','.join(eta_hints)}"

    if timings:
        out["message"] = f"{msg}{eta_suffix} || timings: {timings}"
    else:
        out["message"] = f"{msg}{eta_suffix}"
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
        runs_path=_progress_runs_path(),
        hardware_key=_hardware_key(_host_id()),
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
    out = _project_status_message(out)
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
