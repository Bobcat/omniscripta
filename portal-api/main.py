from __future__ import annotations

import asyncio
import json
import os
import shutil
import mimetypes
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict
from urllib.parse import urlparse, parse_qs

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request, WebSocket
from fastapi.responses import FileResponse, Response
from live_protocol import (
    PROTOCOL_VERSION,
    control_ack_event,
    ended_event,
    error_event,
    parse_client_message,
    pong_event,
    ready_event,
    stats_event,
)
from live_quality import score_live_text_against_fixture, load_fixture_reference
from live_sessions import LiveSessionManager
from live_engine_rolling_context import run_live_session_ws_rolling_context
from queue_fs import init_job_in_inbox, JobPaths, BASE as BASE_JOBS

# Add repo root to path for shared imports
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from shared.app_config import get_str, get_int, get_float, get_bool, get_setting

# Configuration via new config system (env vars still work as override)
ROOT_PATH = get_str("service.root_path", "/api")
app = FastAPI(root_path=ROOT_PATH)
LIVE_RECORDINGS_ROOT = (Path(__file__).resolve().parents[1] / "data" / "live_recordings").resolve()
LIVE_BENCHMARK_EXPORT_ROOT = (Path(__file__).resolve().parents[1] / "data" / "live_benchmark_exports").resolve()


DEFAULT_CALIBRATION_SNIPPET_SECONDS = [60, 180, 300, 480, 600, 1200, 1800, 2700, 3600]
LIVE_SESSION_TTL_S = get_int("live.session_ttl_s", 900, min_value=60)
LIVE_SESSION_PRECONNECT_TTL_S = get_int("live.session_preconnect_ttl_s", 30, min_value=5)
LIVE_MAX_SESSIONS = get_int("live.max_sessions", 1, min_value=1)
LIVE_ARCHIVE_TTL_S = get_int("live.archive_ttl_s", 3600, min_value=60)
LIVE_MAX_ARCHIVES = get_int("live.max_archives", 256, min_value=1)
LIVE_AUDIO_SAMPLE_RATE_HZ = get_int("live.audio_sample_rate_hz", 16000, min_value=8000)
LIVE_AUDIO_CHANNELS = get_int("live.audio_channels", 1, min_value=1)
LIVE_AUDIO_SAMPLE_WIDTH_BYTES = 2
LIVE_AUDIO_BYTES_PER_SECOND = int(max(1, LIVE_AUDIO_SAMPLE_RATE_HZ * LIVE_AUDIO_CHANNELS * LIVE_AUDIO_SAMPLE_WIDTH_BYTES))

LIVE_ENGINE = "rolling_context"
LIVE_DRAIN_WAIT_S = get_float("live.drain_wait_s", 20.0, min_value=0.0)
LIVE_POST_CLOSE_WAIT_S = get_float("live.post_close_wait_s", 60.0, min_value=0.0)
LIVE_ASR_LANGUAGE = get_str("live.asr_language", "en")

# Rolling context settings
LIVE_ROLLING_POLL_INTERVAL_MS = get_int("live.rolling.poll_interval_ms", 250, min_value=100)
LIVE_ROLLING_MIN_INFER_AUDIO_MS = get_int("live.rolling.min_infer_audio_ms", 1000, min_value=200)
LIVE_ROLLING_SINGLE_COMMIT_MIN_MS = max(
    LIVE_ROLLING_MIN_INFER_AUDIO_MS,
    get_int("live.rolling.single_segment_commit_min_ms", 12000, min_value=1000)
)
LIVE_ROLLING_FORCE_COMMIT_REPEATS = get_int("live.rolling.force_commit_repeats", 8, min_value=1)
LIVE_ROLLING_MAX_UNCOMMITTED_MS = max(
    LIVE_ROLLING_MIN_INFER_AUDIO_MS,
    get_int("live.rolling.max_uncommitted_ms", 15000, min_value=1000)
)
LIVE_ROLLING_HARD_CLIP_KEEP_TAIL_MS = max(
    LIVE_ROLLING_MIN_INFER_AUDIO_MS,
    get_int("live.rolling.hard_clip_keep_tail_ms", 5000, min_value=1000)
)
LIVE_ROLLING_MAX_DECODE_WINDOW_MS = max(
    LIVE_ROLLING_MIN_INFER_AUDIO_MS,
    get_int("live.rolling.max_decode_window_ms", 12000, min_value=1000)
)
LIVE_ROLLING_BUFFER_TRIM_THRESHOLD_MS = max(
    LIVE_ROLLING_MAX_DECODE_WINDOW_MS,
    get_int("live.rolling.buffer_trim_threshold_ms", 30000, min_value=5000)
)
LIVE_ROLLING_BUFFER_TRIM_DROP_MS = max(
    LIVE_ROLLING_MIN_INFER_AUDIO_MS,
    get_int("live.rolling.buffer_trim_drop_ms", 20000, min_value=1000)
)
LIVE_ROLLING_REQUIRE_SINGLE_INFLIGHT = get_bool("live.rolling.require_single_inflight", True)
LIVE_SESSIONS = LiveSessionManager(
    default_ttl_seconds=LIVE_SESSION_TTL_S,
    preconnect_ttl_seconds=LIVE_SESSION_PRECONNECT_TTL_S,
    max_sessions=LIVE_MAX_SESSIONS,
    archive_ttl_seconds=LIVE_ARCHIVE_TTL_S,
    max_archives=LIVE_MAX_ARCHIVES,
)
def _live_engine_rolling_context_config() -> dict[str, Any]:
    return {
        "LIVE_ENGINE": LIVE_ENGINE,
        "LIVE_AUDIO_SAMPLE_RATE_HZ": LIVE_AUDIO_SAMPLE_RATE_HZ,
        "LIVE_AUDIO_CHANNELS": LIVE_AUDIO_CHANNELS,
        "LIVE_AUDIO_SAMPLE_WIDTH_BYTES": LIVE_AUDIO_SAMPLE_WIDTH_BYTES,
        "LIVE_AUDIO_BYTES_PER_SECOND": LIVE_AUDIO_BYTES_PER_SECOND,
        "LIVE_DRAIN_WAIT_S": LIVE_DRAIN_WAIT_S,
        "LIVE_POST_CLOSE_WAIT_S": LIVE_POST_CLOSE_WAIT_S,
        "LIVE_ASR_LANGUAGE": LIVE_ASR_LANGUAGE,
        "LIVE_ROLLING_POLL_INTERVAL_MS": LIVE_ROLLING_POLL_INTERVAL_MS,
        "LIVE_ROLLING_MIN_INFER_AUDIO_MS": LIVE_ROLLING_MIN_INFER_AUDIO_MS,
        "LIVE_ROLLING_SINGLE_COMMIT_MIN_MS": LIVE_ROLLING_SINGLE_COMMIT_MIN_MS,
        "LIVE_ROLLING_FORCE_COMMIT_REPEATS": LIVE_ROLLING_FORCE_COMMIT_REPEATS,
        "LIVE_ROLLING_MAX_UNCOMMITTED_MS": LIVE_ROLLING_MAX_UNCOMMITTED_MS,
        "LIVE_ROLLING_HARD_CLIP_KEEP_TAIL_MS": LIVE_ROLLING_HARD_CLIP_KEEP_TAIL_MS,
        "LIVE_ROLLING_MAX_DECODE_WINDOW_MS": LIVE_ROLLING_MAX_DECODE_WINDOW_MS,
        "LIVE_ROLLING_BUFFER_TRIM_THRESHOLD_MS": LIVE_ROLLING_BUFFER_TRIM_THRESHOLD_MS,
        "LIVE_ROLLING_BUFFER_TRIM_DROP_MS": LIVE_ROLLING_BUFFER_TRIM_DROP_MS,
        "LIVE_ROLLING_REQUIRE_SINGLE_INFLIGHT": LIVE_ROLLING_REQUIRE_SINGLE_INFLIGHT,
    }


@app.get("/health")
def health() -> Dict[str, bool]:
    return {"ok": True}


@app.post("/demo/live/sessions")
def create_live_session(request: Request) -> Dict[str, Any]:
    ttl_override: int | None = None
    ttl_raw = request.query_params.get("ttl_s")
    if ttl_raw is not None and str(ttl_raw).strip() != "":
        try:
            ttl_override = int(str(ttl_raw).strip())
        except ValueError:
            raise HTTPException(status_code=400, detail="ttl_s must be an integer number of seconds")
        if ttl_override < 10 or ttl_override > 21600:
            raise HTTPException(status_code=400, detail="ttl_s out of range (10..21600 seconds)")

    try:
        session = LIVE_SESSIONS.create_session(
            ttl_seconds=ttl_override,
            live_engine=LIVE_ENGINE,
        )
    except RuntimeError as e:
        code = str(e or "live_session_create_failed")
        raise HTTPException(
            status_code=429,
            detail={
                "code": code,
                "message": "Live session capacity reached. Stop the active session and retry.",
            },
        )

    session_id = str(session["session_id"])
    ws_path = _rooted_path(f"/demo/live/sessions/{session_id}/ws")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "live_engine": LIVE_ENGINE,
        "session": session,
        "ws_path": ws_path,
        "ws_url": _ws_url_for_request(request, ws_path),
        "audio_input": {
            "format": "pcm16le",
            "sample_rate_hz": LIVE_AUDIO_SAMPLE_RATE_HZ,
            "channels": LIVE_AUDIO_CHANNELS,
        },
    }


@app.get("/demo/live/sessions/{session_id}")
def get_live_session(session_id: str) -> Dict[str, Any]:
    try:
        session = LIVE_SESSIONS.snapshot(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Live session not found")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "live_engine": LIVE_ENGINE,
        "session": session,
    }


@app.get("/demo/live/sessions/{session_id}/final")
def get_live_session_final(session_id: str) -> Dict[str, Any]:
    try:
        archive = LIVE_SESSIONS.archived_transcript(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Live session final transcript not found")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "session_id": str(session_id),
        "archive": archive,
    }


@app.get("/demo/live/sessions/{session_id}/result")
def get_live_session_result(session_id: str) -> Dict[str, Any]:
    try:
        result = LIVE_SESSIONS.live_result_snapshot(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Live session result not found")
    result = dict(result)
    effective_engine = str(result.get("live_engine") or LIVE_ENGINE)
    result["live_engine"] = effective_engine
    final_text = str(result.get("final_text") or "")
    final_segments = result.get("final_segments")
    has_segments = isinstance(final_segments, list) and any(isinstance(s, dict) for s in final_segments)
    has_recording_wav = _live_recording_wav_path_from_result(result) is not None
    can_export = bool(final_text.strip()) or has_segments
    finalization_state = str(result.get("finalization_state") or "").strip().lower()
    ready_states = {"ready", "finalized", "recording_finalized"}
    if effective_engine == "rolling_context":
        # Rolling should only report ready once final commit/drain is complete.
        ready_states = {"ready", "finalized"}
    return {
        "protocol_version": PROTOCOL_VERSION,
        "session_id": str(session_id),
        "live_engine": effective_engine,
        "result": result,
        "ready": finalization_state in ready_states,
        "can_export_txt": bool(can_export),
        "can_export_srt": bool(has_segments),
        "can_export_wav": bool(has_recording_wav),
        "transcript_txt_url": _rooted_path(f"/demo/live/sessions/{session_id}/transcript.txt") if can_export else None,
        "transcript_srt_url": _rooted_path(f"/demo/live/sessions/{session_id}/transcript.srt") if has_segments else None,
        "recording_wav_url": _rooted_path(f"/demo/live/sessions/{session_id}/recording.wav") if has_recording_wav else None,
    }


@app.post("/demo/live/sessions/{session_id}/fixture")
async def set_live_session_fixture(session_id: str, request: Request) -> Dict[str, Any]:
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="fixture payload must be a JSON object")

    fixture_id = str(payload.get("fixture_id") or "").strip()
    fixture_version = str(payload.get("fixture_version") or "").strip()
    fixture_test_mode = str(payload.get("fixture_test_mode") or "").strip()
    if not fixture_id:
        raise HTTPException(status_code=400, detail="fixture_id is required")

    try:
        session = LIVE_SESSIONS.set_fixture_metadata(
            session_id,
            fixture_id=fixture_id,
            fixture_version=fixture_version,
            fixture_test_mode=(fixture_test_mode or "playback"),
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Live session not found")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "session": session,
    }


@app.get("/demo/live/sessions/{session_id}/quality")
def get_live_session_quality(session_id: str, fixture_id: str | None = None) -> Dict[str, Any]:
    try:
        result = LIVE_SESSIONS.live_result_snapshot(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Live session result not found")

    finalization_state = str(result.get("finalization_state") or "").strip().lower()
    # `recording_finalized` means mic stopped; chunk processing may still be in flight.
    # Only score once the transcript is fully ready/finalized.
    if finalization_state not in {"ready", "finalized"}:
        raise HTTPException(status_code=409, detail="Transcript result not ready")

    resolved_fixture_id = str(fixture_id or result.get("fixture_id") or "").strip()
    if not resolved_fixture_id:
        raise HTTPException(status_code=409, detail="No fixture metadata for this session")

    final_text = str(result.get("final_text") or "")
    if not final_text.strip():
        raise HTTPException(status_code=409, detail="Transcript text not ready")

    try:
        quality = score_live_text_against_fixture(
            fixture_id=resolved_fixture_id,
            live_text=final_text,
            live_result=result,
            stats_log_path=LIVE_SESSIONS.stats_log_path(session_id),
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"quality_score_failed:{type(e).__name__}")

    envelope = {
        "protocol_version": PROTOCOL_VERSION,
        "session_id": str(session_id),
        "fixture_id": resolved_fixture_id,
        "ready": True,
        "quality": quality,
    }
    _try_autosave_live_benchmark_snapshot(
        session_id=str(session_id),
        artifact_name="final-quality",
        envelope=envelope,
    )
    return envelope


@app.get("/demo/live/sessions/{session_id}/transcript.txt")
def get_live_session_transcript_txt(session_id: str) -> Response:
    try:
        result = LIVE_SESSIONS.live_result_snapshot(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Live session result not found")
    text = str(result.get("final_text") or "")
    if not text.strip():
        raise HTTPException(status_code=409, detail="Transcript text not ready")
    headers = {"Content-Disposition": f'attachment; filename="{_safe_filename(session_id)}.txt"'}
    return Response(content=text, media_type="text/plain; charset=utf-8", headers=headers)


@app.get("/demo/live/sessions/{session_id}/transcript.srt")
def get_live_session_transcript_srt(session_id: str) -> Response:
    try:
        result = LIVE_SESSIONS.live_result_snapshot(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Live session result not found")
    srt_text = _live_result_to_srt_text(result)
    if not srt_text.strip():
        raise HTTPException(status_code=409, detail="Transcript segments not ready")
    headers = {"Content-Disposition": f'attachment; filename="{_safe_filename(session_id)}.srt"'}
    return Response(content=srt_text, media_type="application/x-subrip", headers=headers)


@app.get("/demo/live/sessions/{session_id}/recording.wav")
def get_live_session_recording_wav(session_id: str) -> FileResponse:
    try:
        result = LIVE_SESSIONS.live_result_snapshot(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Live session result not found")
    wav_path = _live_recording_wav_path_from_result(result)
    if wav_path is None:
        raise HTTPException(status_code=404, detail="Live recording WAV not found")
    return FileResponse(
        path=str(wav_path),
        media_type="audio/wav",
        filename=f"{_safe_filename(session_id)}.wav",
    )


@app.get("/demo/live/metrics")
def get_live_metrics() -> Dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "live_engine": LIVE_ENGINE,
        "metrics": LIVE_SESSIONS.metrics_snapshot(),
    }


@app.websocket("/demo/live/sessions/{session_id}/ws")
async def live_session_ws(session_id: str, websocket: WebSocket) -> None:
    await run_live_session_ws_rolling_context(
        session_id=session_id,
        websocket=websocket,
        live_sessions=LIVE_SESSIONS,
        rooted_path_cb=_rooted_path,
        config=_live_engine_rolling_context_config(),
    )


def _repo_root() -> Path:
    # portal-api/main.py -> portal-api -> repo root
    return Path(__file__).resolve().parents[1]


def _format_srt_timestamp(ms: int) -> str:
    total_ms = int(max(0, ms))
    hours = total_ms // 3_600_000
    rem = total_ms % 3_600_000
    minutes = rem // 60_000
    rem = rem % 60_000
    seconds = rem // 1000
    millis = rem % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _live_result_to_srt_text(result: dict[str, Any]) -> str:
    segments_any = result.get("final_segments")
    if not isinstance(segments_any, list):
        return ""
    rows: list[str] = []
    idx = 0
    for seg in segments_any:
        if not isinstance(seg, dict):
            continue
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        t0_ms = int(max(0, int(seg.get("t0_ms") or 0)))
        t1_ms = int(max(t0_ms + 1, int(seg.get("t1_ms") or (t0_ms + 1))))
        idx += 1
        rows.append(str(idx))
        rows.append(f"{_format_srt_timestamp(t0_ms)} --> {_format_srt_timestamp(t1_ms)}")
        rows.append(text)
        rows.append("")
    return "\n".join(rows).strip() + ("\n" if rows else "")


def _live_recording_wav_path_from_result(result: dict[str, Any]) -> Path | None:
    raw = str((result or {}).get("recording_path") or "").strip()
    if not raw:
        return None
    try:
        candidate = Path(raw).expanduser().resolve()
    except Exception:
        return None
    try:
        candidate.relative_to(LIVE_RECORDINGS_ROOT)
    except Exception:
        return None
    if candidate.suffix.lower() != ".wav":
        return None
    if not candidate.is_file():
        return None
    return candidate


def _rooted_path(path: str) -> str:
    p = str(path or "").strip()
    if not p.startswith("/"):
        p = "/" + p
    rp = str(ROOT_PATH or "").rstrip("/")
    if rp in {"", "/"}:
        return p
    return rp + p


def _ws_url_for_request(request: Request, ws_path: str) -> str:
    forwarded_proto = (request.headers.get("x-forwarded-proto") or "").split(",")[0].strip().lower()
    if forwarded_proto in {"https", "wss"}:
        scheme = "wss"
    elif forwarded_proto in {"http", "ws"}:
        scheme = "ws"
    else:
        origin = (request.headers.get("origin") or "").strip()
        try:
            origin_scheme = urlparse(origin).scheme.lower()
        except Exception:
            origin_scheme = ""
        if origin_scheme == "https":
            scheme = "wss"
        elif origin_scheme == "http":
            scheme = "ws"
        else:
            scheme = "wss" if request.url.scheme == "https" else "ws"

    forwarded_host = (request.headers.get("x-forwarded-host") or "").split(",")[0].strip()
    host = forwarded_host or request.headers.get("host") or request.url.netloc
    return f"{scheme}://{host}{ws_path}"


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


def _virtual_config_source(*, source_id: str, title: str, data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": source_id,
        "title": title,
        "path": "<config:settings/local>",
        "exists": True,
        "size_bytes": None,
        "mtime_utc": None,
        "parse_ok": True,
        "data": _redact_sensitive(data),
        "error": None,
    }


@app.get("/demo/settings")
def get_demo_settings() -> Dict[str, Any]:
    snip = get_setting("snip", {})
    topics = get_setting("topics", {})
    tabby = get_setting("tabby", {})
    whisperx = get_setting("asr_pool.whisperx", {})
    service_data = {
        "snip": dict(snip) if isinstance(snip, dict) else {},
        "topics": dict(topics) if isinstance(topics, dict) else {},
        "tabby": dict(tabby) if isinstance(tabby, dict) else {},
    }
    whisperx_data = dict(whisperx) if isinstance(whisperx, dict) else {}
    return {
        "generated_at_utc": _iso_utc(datetime.now(timezone.utc).timestamp()),
        "sources": [
            _virtual_config_source(source_id="service", title="service.config", data=service_data),
            _virtual_config_source(source_id="whisperx", title="asr_pool.whisperx", data=whisperx_data),
        ],
    }


def _safe_filename(name: str) -> str:
    # Voorkomt path traversal zoals ../../etc/passwd
    return Path(name).name or "upload.bin"


def _write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)


def _autosave_live_benchmark_snapshot(
    *,
    session_id: str,
    artifact_name: str,
    envelope: Dict[str, Any],
    request_meta: Dict[str, Any] | None = None,
) -> None:
    safe_session_id = _safe_filename(str(session_id or "session"))
    artifact = _safe_filename(str(artifact_name or "benchmark"))
    now_ts = time.time()
    now_iso = _iso_utc(now_ts)
    root = LIVE_BENCHMARK_EXPORT_ROOT

    record: Dict[str, Any] = {
        "saved_at_utc": now_iso,
        "saved_at_unix": round(float(now_ts), 6),
        "session_id": str(session_id or ""),
        "artifact_name": artifact_name,
        "request_meta": dict(request_meta or {}),
        "payload": envelope,
    }

    latest_path = (root / f"{safe_session_id}.{artifact}.latest.json").resolve()
    history_path = (root / f"{safe_session_id}.{artifact}.history.jsonl").resolve()

    _write_json_atomic(latest_path, record)
    _append_jsonl(history_path, record)


def _try_autosave_live_benchmark_snapshot(
    *,
    session_id: str,
    artifact_name: str,
    envelope: Dict[str, Any],
    request_meta: Dict[str, Any] | None = None,
) -> None:
    try:
        _autosave_live_benchmark_snapshot(
            session_id=session_id,
            artifact_name=artifact_name,
            envelope=envelope,
            request_meta=request_meta,
        )
    except Exception as e:
        print(f"[live-benchmark-autosave] failed {artifact_name} session={session_id}: {type(e).__name__}: {e}")


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


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


def _parse_calibration_seconds() -> list[int]:
    from shared.app_config import get_list

    raw_list = get_list("live.calibration_seconds", [])
    if raw_list:
        vals: list[int] = []
        for item in raw_list:
            try:
                v = int(item)
                if v > 0 and v not in vals:
                    vals.append(v)
            except (ValueError, TypeError):
                continue
        return vals or list(DEFAULT_CALIBRATION_SNIPPET_SECONDS)

    return list(DEFAULT_CALIBRATION_SNIPPET_SECONDS)


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
    calibration_seconds = _parse_calibration_seconds() if calibration_enabled else []

    staging_dir = (BASE_JOBS / "_staging_uploads").resolve()
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
        # Primary job (the one returned to frontend)
        jp: JobPaths = init_job_in_inbox(
            orig_filename=orig_name,
            options=dict(base_options),
            job_kind="upload_audio",
            upload_src_path=staged_upload_path,
            move_upload_src=False,
        )
        dst_primary = jp.upload_dir / orig_name

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
                    extra = init_job_in_inbox(
                        orig_filename=orig_name,
                        options=opts,
                        job_kind="upload_audio",
                        upload_src_path=dst_primary,
                        move_upload_src=False,
                    )
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
    finally:
        try:
            staged_upload_path.unlink(missing_ok=True)
        except Exception:
            pass


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
