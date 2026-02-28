from __future__ import annotations

import asyncio
import json
import os
import shutil
import mimetypes
import time
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict
from urllib.parse import urlparse, parse_qs

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request, WebSocket, WebSocketDisconnect, status
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
from live_chunker import LiveAudioChunker, LiveChunkerConfig
from live_chunk_transcribe import LiveChunkBatchBridge
from live_recordings import LiveWavRecorder
from live_quality import score_semilive_text_against_fixture, load_fixture_reference
from live_sessions import LiveSessionManager
from speculative_quality import score_speculative_history_against_final, score_speculative_history_against_reference
import speculative_tuning_runner
import asr_loadtest_runner
from queue_fs import init_job_in_inbox, JobPaths, BASE as BASE_JOBS

ROOT_PATH = os.getenv("TRANSCRIBE_ROOT_PATH", "/api")
app = FastAPI(root_path=ROOT_PATH)
LIVE_RECORDINGS_ROOT = (Path(__file__).resolve().parents[1] / "data" / "live_recordings").resolve()
LIVE_BENCHMARK_EXPORT_ROOT = (Path(__file__).resolve().parents[1] / "data" / "live_benchmark_exports").resolve()


DEFAULT_CALIBRATION_SNIPPET_SECONDS = [60, 180, 300, 480, 600, 1200, 1800, 2700, 3600]
LIVE_SESSION_TTL_S = int(os.getenv("TRANSCRIBE_LIVE_SESSION_TTL_S", "900"))
LIVE_SESSION_PRECONNECT_TTL_S = int(os.getenv("TRANSCRIBE_LIVE_PRECONNECT_TTL_S", "30"))
LIVE_MAX_SESSIONS = int(os.getenv("TRANSCRIBE_LIVE_MAX_SESSIONS", "1"))
LIVE_ARCHIVE_TTL_S = int(os.getenv("TRANSCRIBE_LIVE_ARCHIVE_TTL_S", "3600"))
LIVE_MAX_ARCHIVES = int(os.getenv("TRANSCRIBE_LIVE_MAX_ARCHIVES", "256"))
LIVE_AUDIO_SAMPLE_RATE_HZ = int(os.getenv("TRANSCRIBE_LIVE_SAMPLE_RATE_HZ", "16000"))
LIVE_AUDIO_CHANNELS = int(os.getenv("TRANSCRIBE_LIVE_CHANNELS", "1"))
LIVE_AUDIO_SAMPLE_WIDTH_BYTES = 2
LIVE_AUDIO_BYTES_PER_SECOND = int(max(1, LIVE_AUDIO_SAMPLE_RATE_HZ * LIVE_AUDIO_CHANNELS * LIVE_AUDIO_SAMPLE_WIDTH_BYTES))
LIVE_SEMILIVE_CHUNK_BATCH_SHADOW = str(os.getenv("TRANSCRIBE_LIVE_SEMILIVE_CHUNK_BATCH_SHADOW", "1")).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
try:
    LIVE_SEMILIVE_CHUNK_POLL_INTERVAL_S = max(
        0.1,
        float(str(os.getenv("TRANSCRIBE_LIVE_SEMILIVE_CHUNK_POLL_INTERVAL_S", "0.75")).strip() or "0.75"),
    )
except Exception:
    LIVE_SEMILIVE_CHUNK_POLL_INTERVAL_S = 0.75
try:
    LIVE_SEMILIVE_CHUNK_STOP_WAIT_S = max(
        0.0,
        float(str(os.getenv("TRANSCRIBE_LIVE_SEMILIVE_CHUNK_STOP_WAIT_S", "20.0")).strip() or "20.0"),
    )
except Exception:
    LIVE_SEMILIVE_CHUNK_STOP_WAIT_S = 20.0
try:
    LIVE_SEMILIVE_CHUNK_POST_CLOSE_WAIT_S = max(
        0.0,
        float(str(os.getenv("TRANSCRIBE_LIVE_SEMILIVE_CHUNK_POST_CLOSE_WAIT_S", "60.0")).strip() or "60.0"),
    )
except Exception:
    LIVE_SEMILIVE_CHUNK_POST_CLOSE_WAIT_S = 60.0
LIVE_SEMILIVE_CHUNK_LANGUAGE = (str(os.getenv("TRANSCRIBE_LIVE_SEMILIVE_CHUNK_LANGUAGE", "en")) or "en").strip() or "en"
LIVE_SEMILIVE_CHUNK_ENERGY_THRESHOLD = max(
    0,
    int(str(os.getenv("TRANSCRIBE_LIVE_SEMILIVE_CHUNK_ENERGY_THRESHOLD", "12")).strip() or "12"),
)
LIVE_SEMILIVE_CHUNK_SILENCE_MS = max(
    0,
    int(str(os.getenv("TRANSCRIBE_LIVE_SEMILIVE_CHUNK_SILENCE_MS", "1200")).strip() or "1200"),
)
LIVE_SEMILIVE_CHUNK_MAX_MS = max(
    200,
    int(str(os.getenv("TRANSCRIBE_LIVE_SEMILIVE_CHUNK_MAX_MS", "20000")).strip() or "20000"),
)
LIVE_SEMILIVE_CHUNK_MIN_MS = max(
    0,
    int(str(os.getenv("TRANSCRIBE_LIVE_SEMILIVE_CHUNK_MIN_MS", "800")).strip() or "800"),
)
LIVE_SEMILIVE_CHUNK_PRE_ROLL_MS = max(
    0,
    int(str(os.getenv("TRANSCRIBE_LIVE_SEMILIVE_CHUNK_PRE_ROLL_MS", "800")).strip() or "800"),
)
LIVE_SEMILIVE_DEDUP_ENABLED = str(os.getenv("TRANSCRIBE_LIVE_SEMILIVE_DEDUP_ENABLED", "1")).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
LIVE_SEMILIVE_DEDUP_MIN_WORDS = max(
    1,
    int(str(os.getenv("TRANSCRIBE_LIVE_SEMILIVE_DEDUP_MIN_WORDS", "3")).strip() or "3"),
)
LIVE_SEMILIVE_DEDUP_MAX_TRIM_WORDS = max(
    LIVE_SEMILIVE_DEDUP_MIN_WORDS,
    int(str(os.getenv("TRANSCRIBE_LIVE_SEMILIVE_DEDUP_MAX_TRIM_WORDS", "24")).strip() or "24"),
)
LIVE_SEMILIVE_INITIAL_PROMPT_ENABLED = str(
    os.getenv("TRANSCRIBE_LIVE_SEMILIVE_INITIAL_PROMPT_ENABLED", "1")
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
LIVE_SEMILIVE_SPECULATIVE_INITIAL_PROMPT_ENABLED = str(
    os.getenv("TRANSCRIBE_LIVE_SEMILIVE_SPECULATIVE_INITIAL_PROMPT_ENABLED", "1")
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
LIVE_SEMILIVE_INITIAL_PROMPT_TAIL_WORDS = max(
    1,
    int(str(os.getenv("TRANSCRIBE_LIVE_SEMILIVE_INITIAL_PROMPT_TAIL_WORDS", "30")).strip() or "30"),
)
LIVE_SEMILIVE_INITIAL_PROMPT_MIN_WORDS = max(
    1,
    int(str(os.getenv("TRANSCRIBE_LIVE_SEMILIVE_INITIAL_PROMPT_MIN_WORDS", "6")).strip() or "6"),
)
LIVE_SEMILIVE_INITIAL_PROMPT_MAX_CHARS = max(
    0,
    int(str(os.getenv("TRANSCRIBE_LIVE_SEMILIVE_INITIAL_PROMPT_MAX_CHARS", "400")).strip() or "400"),
)
LIVE_SEMILIVE_SPECULATIVE_ENABLED = str(
    os.getenv("TRANSCRIBE_LIVE_SEMILIVE_SPECULATIVE_ENABLED", "0")
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
LIVE_SEMILIVE_SPECULATIVE_INTERVAL_MS = max(
    200,
    int(str(os.getenv("TRANSCRIBE_LIVE_SEMILIVE_SPECULATIVE_INTERVAL_MS", "1800")).strip() or "1800"),
)
LIVE_SEMILIVE_SPECULATIVE_WINDOW_MS = max(
    LIVE_SEMILIVE_SPECULATIVE_INTERVAL_MS,
    int(str(os.getenv("TRANSCRIBE_LIVE_SEMILIVE_SPECULATIVE_WINDOW_MS", "3000")).strip() or "3000"),
)
LIVE_SEMILIVE_SPECULATIVE_OVERLAP_MS = max(
    0,
    int(str(os.getenv("TRANSCRIBE_LIVE_SEMILIVE_SPECULATIVE_OVERLAP_MS", "800")).strip() or "800"),
)
LIVE_SEMILIVE_SPECULATIVE_MAX_STALENESS_MS = max(
    0,
    int(str(os.getenv("TRANSCRIBE_LIVE_SEMILIVE_SPECULATIVE_MAX_STALENESS_MS", "1200")).strip() or "1200"),
)
LIVE_SEMILIVE_SPECULATIVE_REQUIRE_NO_FINAL_PENDING = str(
    os.getenv("TRANSCRIBE_LIVE_SEMILIVE_SPECULATIVE_REQUIRE_NO_FINAL_PENDING", "1")
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
LIVE_SEMILIVE_FINAL_BEAM_SIZE_OVERRIDE = max(
    0,
    int(str(os.getenv("TRANSCRIBE_LIVE_SEMILIVE_FINAL_BEAM_SIZE_OVERRIDE", "0")).strip() or "0"),
)
LIVE_SWEEP_WS_BASE_URL = (str(os.getenv("TRANSCRIBE_LIVE_SWEEP_WS_BASE_URL", "ws://127.0.0.1:8001")) or "").strip()
ASR_POOL_STATUS_URL = (str(os.getenv("TRANSCRIBE_ASR_POOL_STATUS_URL", "http://127.0.0.1:8090/asr/v1/pool")) or "").strip()
ASR_POOL_TOKEN = (str(os.getenv("TRANSCRIBE_ASR_POOL_TOKEN", "")) or "").strip()
LIVE_SESSIONS = LiveSessionManager(
    default_ttl_seconds=LIVE_SESSION_TTL_S,
    preconnect_ttl_seconds=LIVE_SESSION_PRECONNECT_TTL_S,
    max_sessions=LIVE_MAX_SESSIONS,
    archive_ttl_seconds=LIVE_ARCHIVE_TTL_S,
    max_archives=LIVE_MAX_ARCHIVES,
    semilive_text_dedup_enabled=LIVE_SEMILIVE_DEDUP_ENABLED,
    semilive_text_dedup_min_words=LIVE_SEMILIVE_DEDUP_MIN_WORDS,
    semilive_text_dedup_max_trim_words=LIVE_SEMILIVE_DEDUP_MAX_TRIM_WORDS,
)


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
        session = LIVE_SESSIONS.create_session(ttl_seconds=ttl_override)
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
        result = LIVE_SESSIONS.semilive_result_snapshot(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Live session result not found")
    final_text = str(result.get("final_text") or "")
    final_segments = result.get("final_segments")
    has_segments = isinstance(final_segments, list) and any(isinstance(s, dict) for s in final_segments)
    has_recording_wav = _live_recording_wav_path_from_result(result) is not None
    can_export = bool(final_text.strip()) or has_segments
    return {
        "protocol_version": PROTOCOL_VERSION,
        "session_id": str(session_id),
        "result": result,
        "ready": str(result.get("finalization_state") or "").strip().lower() in {"ready", "finalized", "recording_finalized"},
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
        result = LIVE_SESSIONS.semilive_result_snapshot(session_id)
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
        quality = score_semilive_text_against_fixture(
            fixture_id=resolved_fixture_id,
            semilive_text=final_text,
            semilive_result=result,
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
    _try_autosave_speculative_lane_trace(str(session_id))
    return envelope


@app.get("/demo/live/sessions/{session_id}/speculative-quality")
def get_live_session_speculative_quality(
    session_id: str,
    verbose: bool = False,
    fixture_id: str | None = None,
) -> Dict[str, Any]:
    try:
        history = LIVE_SESSIONS.semilive_speculative_history_snapshot(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Live session speculative history not found")

    quality = score_speculative_history_against_final(
        speculative_history=history,
        verbose=bool(verbose),
    )
    envelope = {
        "protocol_version": PROTOCOL_VERSION,
        "session_id": str(session_id),
        "ready": True,
        "speculative_quality": quality,
    }
    resolved_fixture_id = str(fixture_id or "").strip()
    if not resolved_fixture_id:
        try:
            semilive_result = LIVE_SESSIONS.semilive_result_snapshot(session_id)
            resolved_fixture_id = str((semilive_result or {}).get("fixture_id") or "").strip()
        except Exception:
            resolved_fixture_id = ""
    if resolved_fixture_id:
        try:
            fixture = load_fixture_reference(resolved_fixture_id)
            ref_text = str(fixture.get("reference_text") or "")
            vs_ref = score_speculative_history_against_reference(
                speculative_history=history,
                reference_text=ref_text,
                verbose=bool(verbose),
            )
            envelope["speculative_quality_vs_reference"] = vs_ref
            envelope["fixture_id"] = str(resolved_fixture_id)
        except Exception:
            pass
    _try_autosave_live_benchmark_snapshot(
        session_id=str(session_id),
        artifact_name="speculative-quality",
        envelope=envelope,
        request_meta={"verbose": bool(verbose), "fixture_id": str(resolved_fixture_id or "")},
    )
    return envelope


@app.post("/demo/live/speculative-tuning/run")
async def start_live_speculative_tuning_run(request: Request) -> Dict[str, Any]:
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    _configure_speculative_tuning_runner()
    return await speculative_tuning_runner.start_run(payload if isinstance(payload, dict) else {})


@app.get("/demo/live/speculative-tuning/report/{run_id}")
def get_live_speculative_tuning_report(run_id: str) -> Dict[str, Any]:
    _configure_speculative_tuning_runner()
    return speculative_tuning_runner.get_report(run_id)


@app.post("/demo/live/asr-loadtest/run")
async def start_live_asr_loadtest_run(request: Request) -> Dict[str, Any]:
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    _configure_asr_loadtest_runner()
    return await asr_loadtest_runner.start_run(payload if isinstance(payload, dict) else {})


@app.get("/demo/live/asr-loadtest/report/{run_id}")
def get_live_asr_loadtest_report(run_id: str) -> Dict[str, Any]:
    _configure_asr_loadtest_runner()
    return asr_loadtest_runner.get_report(run_id)


@app.get("/demo/live/sessions/{session_id}/transcript.txt")
def get_live_session_transcript_txt(session_id: str) -> Response:
    try:
        result = LIVE_SESSIONS.semilive_result_snapshot(session_id)
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
        result = LIVE_SESSIONS.semilive_result_snapshot(session_id)
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
        result = LIVE_SESSIONS.semilive_result_snapshot(session_id)
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
        "metrics": LIVE_SESSIONS.metrics_snapshot(),
    }


@app.websocket("/demo/live/sessions/{session_id}/ws")
async def live_session_ws(session_id: str, websocket: WebSocket) -> None:
    try:
        LIVE_SESSIONS.open_websocket(session_id)
    except KeyError:
        await websocket.close(
            code=status.WS_1008_POLICY_VIOLATION,
            reason="session_not_found",
        )
        return
    except RuntimeError as e:
        await websocket.close(
            code=status.WS_1008_POLICY_VIOLATION,
            reason=str(e),
        )
        return

    await websocket.accept()
    stop_reason = "client_disconnected"
    websocket_closed = False
    recorder: LiveWavRecorder | None = None
    chunker: LiveAudioChunker | None = None
    chunk_bridge: LiveChunkBatchBridge | None = None
    semilive_chunk_jobs_enabled = bool(LIVE_SEMILIVE_CHUNK_BATCH_SHADOW)
    semilive_chunk_jobs_pending: dict[int, dict[str, Any]] = {}
    semilive_chunk_jobs_to_enqueue: list[Any] = []
    semilive_chunk_jobs_last_poll_mono = 0.0
    semilive_recording_state = "idle"
    semilive_recording_path = ""
    semilive_recording_bytes = 0
    semilive_recording_duration_ms = 0
    semilive_chunk_index_next = 0
    semilive_chunks_total = 0
    semilive_chunks_done = 0
    semilive_chunks_failed = 0
    semilive_finalization_state = "idle"
    semilive_shadow_disabled_reason = ""
    semilive_recording_finalized = False
    semilive_chunker_snapshot: dict[str, Any] = {}
    semilive_speculative_enabled = bool(LIVE_SEMILIVE_SPECULATIVE_ENABLED and semilive_chunk_jobs_enabled)
    semilive_speculative_jobs_pending: dict[int, dict[str, Any]] = {}
    semilive_speculative_last_emit_mono = 0.0
    semilive_speculative_last_poll_mono = 0.0
    semilive_speculative_seq_next = 0
    semilive_speculative_recent_pcm = bytearray()
    semilive_speculative_effective_window_ms = int(
        max(
            LIVE_SEMILIVE_SPECULATIVE_WINDOW_MS,
            LIVE_SEMILIVE_SPECULATIVE_INTERVAL_MS + LIVE_SEMILIVE_SPECULATIVE_OVERLAP_MS,
        )
    )
    semilive_speculative_recent_pcm_max_bytes = int(
        max(
            LIVE_AUDIO_SAMPLE_WIDTH_BYTES,
            round(((semilive_speculative_effective_window_ms + 1000) / 1000.0) * LIVE_AUDIO_BYTES_PER_SECOND),
        )
    )
    if (semilive_speculative_recent_pcm_max_bytes % LIVE_AUDIO_SAMPLE_WIDTH_BYTES) != 0:
        semilive_speculative_recent_pcm_max_bytes += (
            LIVE_AUDIO_SAMPLE_WIDTH_BYTES - (semilive_speculative_recent_pcm_max_bytes % LIVE_AUDIO_SAMPLE_WIDTH_BYTES)
        )
    semilive_speculative_metrics: dict[str, Any] = {
        "enqueued": 0,
        "shown": 0,
        "dropped_busy": 0,
        "dropped_stale": 0,
        "time_to_first_speculative_ms": None,
    }

    async def send_event(payload: Dict[str, Any]) -> None:
        payload = dict(payload)
        try:
            payload["seq"] = LIVE_SESSIONS.next_seq(session_id)
        except KeyError:
            pass
        await websocket.send_json(payload)

    def _semilive_archive_kwargs() -> dict[str, Any]:
        return {
            "recording_path": str(semilive_recording_path or ""),
            "recording_bytes": int(max(0, semilive_recording_bytes)),
            "recording_duration_ms": int(max(0, semilive_recording_duration_ms)),
            "chunks_total": int(max(0, semilive_chunks_total)),
            "chunks_done": int(max(0, semilive_chunks_done)),
            "chunks_failed": int(max(0, semilive_chunks_failed)),
            "finalization_state": str(semilive_finalization_state or ""),
            "batch_job_id": "",
        }

    def _archive_current_semilive_result(*, close_reason: str) -> dict[str, Any]:
        try:
            semilive_result = LIVE_SESSIONS.semilive_result_snapshot(session_id)
        except Exception:
            return {}
        if not semilive_result:
            return {}
        has_content = (
            bool(str(semilive_result.get("final_text") or "").strip())
            or bool(semilive_result.get("final_segments"))
            or int(semilive_result.get("chunks_total") or 0) > 0
            or int(max(0, semilive_recording_duration_ms)) > 0
        )
        if not has_content:
            return semilive_result
        LIVE_SESSIONS.archive_transcript(
            session_id,
            close_reason=str(close_reason or stop_reason or "closed"),
            final_text=str(semilive_result.get("final_text") or ""),
            final_segments=[
                dict(seg)
                for seg in (semilive_result.get("final_segments") or [])
                if isinstance(seg, dict)
            ],
            transcript_revision=int(max(0, int(semilive_result.get("transcript_revision") or 0))),
            **_semilive_archive_kwargs(),
        )
        return semilive_result

    def _append_semilive_log(kind: str, **fields: Any) -> None:
        try:
            row = {"kind": str(kind)}
            row.update(fields)
            LIVE_SESSIONS.append_stats_log(session_id, row)
        except Exception:
            pass

    def _update_semilive_session_state() -> None:
        try:
            LIVE_SESSIONS.update_semilive(
                session_id,
                recording_state=semilive_recording_state,
                recording_path=semilive_recording_path,
                recording_bytes=semilive_recording_bytes,
                recording_duration_ms=semilive_recording_duration_ms,
                chunk_index_next=semilive_chunk_index_next,
                chunks_total=semilive_chunks_total,
                chunks_done=semilive_chunks_done,
                chunks_failed=semilive_chunks_failed,
                finalization_state=semilive_finalization_state,
                batch_job_id="",
            )
        except Exception:
            pass
        if semilive_speculative_enabled:
            try:
                LIVE_SESSIONS.update_semilive_speculative_metrics(
                    session_id,
                    enqueued=int(max(0, int(semilive_speculative_metrics.get("enqueued") or 0))),
                    shown=int(max(0, int(semilive_speculative_metrics.get("shown") or 0))),
                    dropped_busy=int(max(0, int(semilive_speculative_metrics.get("dropped_busy") or 0))),
                    dropped_stale=int(max(0, int(semilive_speculative_metrics.get("dropped_stale") or 0))),
                    time_to_first_speculative_ms=(
                        int(semilive_speculative_metrics.get("time_to_first_speculative_ms"))
                        if semilive_speculative_metrics.get("time_to_first_speculative_ms") is not None
                        else None
                    ),
                )
            except Exception:
                pass
            if str(semilive_finalization_state or "").strip().lower() in {"ready", "error", "finalized"}:
                try:
                    LIVE_SESSIONS.clear_semilive_speculative_preview(session_id)
                except Exception:
                    pass

    def _build_semilive_initial_prompt() -> tuple[str, dict[str, Any]]:
        if not LIVE_SEMILIVE_INITIAL_PROMPT_ENABLED:
            return "", {"enabled": False, "used": False, "reason": "disabled"}
        try:
            result = LIVE_SESSIONS.semilive_result_snapshot(session_id)
        except Exception as e:
            return "", {"enabled": True, "used": False, "reason": f"snapshot_error:{type(e).__name__}"}

        final_text = " ".join(str(result.get("final_text") or "").split()).strip()
        if not final_text:
            return "", {"enabled": True, "used": False, "reason": "no_context_text"}

        tokens = [tok for tok in final_text.split(" ") if tok]
        source_words = len(tokens)
        if source_words < LIVE_SEMILIVE_INITIAL_PROMPT_MIN_WORDS:
            return "", {
                "enabled": True,
                "used": False,
                "reason": "too_few_words",
                "source_words": int(source_words),
            }

        tail_tokens = tokens[-LIVE_SEMILIVE_INITIAL_PROMPT_TAIL_WORDS :]
        prompt = " ".join(tail_tokens).strip()
        if LIVE_SEMILIVE_INITIAL_PROMPT_MAX_CHARS > 0 and len(prompt) > LIVE_SEMILIVE_INITIAL_PROMPT_MAX_CHARS:
            tail = prompt[-LIVE_SEMILIVE_INITIAL_PROMPT_MAX_CHARS :].strip()
            if " " in tail:
                tail = tail.split(" ", 1)[1].strip()
            prompt = tail or prompt[-LIVE_SEMILIVE_INITIAL_PROMPT_MAX_CHARS :].strip()
        prompt_words = len([tok for tok in prompt.split() if tok])
        if prompt_words < LIVE_SEMILIVE_INITIAL_PROMPT_MIN_WORDS:
            return "", {
                "enabled": True,
                "used": False,
                "reason": "trimmed_too_short",
                "source_words": int(source_words),
                "tail_words": int(prompt_words),
            }
        return prompt, {
            "enabled": True,
            "used": True,
            "chars": len(prompt),
            "words": int(prompt_words),
            "source_words": int(source_words),
            "transcript_revision": int(max(0, int(result.get("transcript_revision") or 0))),
        }

    def _append_speculative_pcm(raw_bytes: bytes) -> None:
        nonlocal semilive_speculative_recent_pcm
        if not semilive_speculative_enabled:
            return
        raw = bytes(raw_bytes or b"")
        if not raw:
            return
        if (len(raw) % LIVE_AUDIO_SAMPLE_WIDTH_BYTES) != 0:
            raw = raw[: len(raw) - (len(raw) % LIVE_AUDIO_SAMPLE_WIDTH_BYTES)]
        if not raw:
            return
        semilive_speculative_recent_pcm.extend(raw)
        overflow = len(semilive_speculative_recent_pcm) - int(max(0, semilive_speculative_recent_pcm_max_bytes))
        if overflow > 0:
            if (overflow % LIVE_AUDIO_SAMPLE_WIDTH_BYTES) != 0:
                overflow += LIVE_AUDIO_SAMPLE_WIDTH_BYTES - (overflow % LIVE_AUDIO_SAMPLE_WIDTH_BYTES)
            del semilive_speculative_recent_pcm[:overflow]

    async def _maybe_enqueue_speculative_job() -> None:
        nonlocal semilive_speculative_last_emit_mono, semilive_speculative_seq_next
        if not semilive_speculative_enabled or chunk_bridge is None:
            return
        if semilive_recording_finalized or str(semilive_recording_state or "") != "recording":
            return
        now_mono = time.monotonic()
        interval_s = max(0.2, float(LIVE_SEMILIVE_SPECULATIVE_INTERVAL_MS) / 1000.0)
        if (now_mono - semilive_speculative_last_emit_mono) < interval_s:
            return
        if semilive_speculative_jobs_pending:
            semilive_speculative_last_emit_mono = now_mono
            semilive_speculative_metrics["dropped_busy"] = int(
                max(0, int(semilive_speculative_metrics.get("dropped_busy") or 0)) + 1
            )
            return
        if LIVE_SEMILIVE_SPECULATIVE_REQUIRE_NO_FINAL_PENDING and (
            semilive_chunk_jobs_pending or semilive_chunk_jobs_to_enqueue
        ):
            semilive_speculative_last_emit_mono = now_mono
            semilive_speculative_metrics["dropped_busy"] = int(
                max(0, int(semilive_speculative_metrics.get("dropped_busy") or 0)) + 1
            )
            return
        end_ms = int(max(0, semilive_recording_duration_ms))
        if end_ms <= 0 or not semilive_speculative_recent_pcm:
            return
        want_bytes = int(
            max(
                LIVE_AUDIO_SAMPLE_WIDTH_BYTES,
                round((float(semilive_speculative_effective_window_ms) / 1000.0) * LIVE_AUDIO_BYTES_PER_SECOND),
            )
        )
        if (want_bytes % LIVE_AUDIO_SAMPLE_WIDTH_BYTES) != 0:
            want_bytes += LIVE_AUDIO_SAMPLE_WIDTH_BYTES - (want_bytes % LIVE_AUDIO_SAMPLE_WIDTH_BYTES)
        use_bytes = min(len(semilive_speculative_recent_pcm), want_bytes)
        if (use_bytes % LIVE_AUDIO_SAMPLE_WIDTH_BYTES) != 0:
            use_bytes -= use_bytes % LIVE_AUDIO_SAMPLE_WIDTH_BYTES
        min_bytes = int(max(LIVE_AUDIO_SAMPLE_WIDTH_BYTES, LIVE_AUDIO_BYTES_PER_SECOND * 0.5))
        if use_bytes < min_bytes:
            return
        pcm = bytes(semilive_speculative_recent_pcm[-use_bytes:])
        actual_ms = int(round((float(use_bytes) / float(LIVE_AUDIO_BYTES_PER_SECOND)) * 1000.0))
        t1_ms = int(end_ms)
        t0_ms = int(max(0, t1_ms - max(1, actual_ms)))
        spec_seq = int(max(0, semilive_speculative_seq_next))
        if LIVE_SEMILIVE_SPECULATIVE_INITIAL_PROMPT_ENABLED:
            initial_prompt, initial_prompt_meta = _build_semilive_initial_prompt()
        else:
            initial_prompt = ""
            initial_prompt_meta = {
                "enabled": False,
                "used": False,
                "reason": "disabled_for_speculative_lane",
            }
        try:
            enq = await asyncio.to_thread(
                chunk_bridge.enqueue_chunk_pcm16,
                session_id=session_id,
                chunk_index=spec_seq,
                t0_ms=t0_ms,
                t1_ms=t1_ms,
                pcm16le=pcm,
                language=LIVE_SEMILIVE_CHUNK_LANGUAGE,
                initial_prompt=initial_prompt,
                live_lane="speculative",
                speculative_seq=spec_seq,
                speculative_audio_end_ms=t1_ms,
            )
        except Exception as e:
            _append_semilive_log(
                "semilive_speculative_enqueue_error",
                speculative_seq=spec_seq,
                error=f"{type(e).__name__}: {e}",
            )
            return
        semilive_speculative_seq_next = spec_seq + 1
        semilive_speculative_last_emit_mono = now_mono
        semilive_speculative_jobs_pending[spec_seq] = {
            "speculative_seq": spec_seq,
            "job_id": str(enq.job_id),
            "t0_ms": int(t0_ms),
            "t1_ms": int(t1_ms),
            "audio_end_ms": int(t1_ms),
            "reported_state": "",
            "enqueued_mono": now_mono,
        }
        semilive_speculative_metrics["enqueued"] = int(max(0, int(semilive_speculative_metrics.get("enqueued") or 0)) + 1)
        _append_semilive_log(
            "semilive_speculative_enqueued",
            speculative_seq=spec_seq,
            job_id=str(enq.job_id),
            t0_ms=int(t0_ms),
            t1_ms=int(t1_ms),
            audio_bytes=int(len(pcm)),
            window_ms=int(max(0, t1_ms - t0_ms)),
            initial_prompt_meta=dict(initial_prompt_meta) if isinstance(initial_prompt_meta, dict) else {},
        )

    async def _poll_speculative_jobs(*, force: bool = False) -> None:
        nonlocal semilive_speculative_last_poll_mono
        if not semilive_speculative_enabled or chunk_bridge is None:
            return
        now_mono = time.monotonic()
        if not force and (now_mono - semilive_speculative_last_poll_mono) < LIVE_SEMILIVE_CHUNK_POLL_INTERVAL_S:
            return
        semilive_speculative_last_poll_mono = now_mono
        for spec_seq in sorted(list(semilive_speculative_jobs_pending.keys())):
            item = semilive_speculative_jobs_pending.get(spec_seq)
            if not item:
                continue
            job_id = str(item.get("job_id") or "")
            t0_ms = int(item.get("t0_ms") or 0)
            enqueued_mono = float(item.get("enqueued_mono") or 0.0)
            if not job_id:
                continue
            try:
                poll = await asyncio.to_thread(
                    chunk_bridge.poll_job,
                    job_id,
                    t0_offset_ms=t0_ms,
                )
            except FileNotFoundError:
                continue
            except Exception as e:
                _append_semilive_log(
                    "semilive_speculative_poll_error",
                    speculative_seq=int(spec_seq),
                    job_id=job_id,
                    error=f"{type(e).__name__}: {e}",
                )
                semilive_speculative_jobs_pending.pop(spec_seq, None)
                continue

            poll_state = "ready" if poll.ok else ("error" if poll.done else str(poll.state or "queued"))
            if str(item.get("reported_state") or "") != poll_state:
                item["reported_state"] = poll_state
                if poll_state == "ready":
                    status_obj = dict(poll.status) if isinstance(poll.status, dict) else {}
                    def _status_int(name: str) -> int | None:
                        if name not in status_obj or status_obj.get(name) is None:
                            return None
                        try:
                            return int(max(0, int(status_obj.get(name))))
                        except Exception:
                            return None
                    def _status_float(name: str) -> float | None:
                        if name not in status_obj or status_obj.get(name) is None:
                            return None
                        try:
                            return max(0.0, float(status_obj.get(name)))
                        except Exception:
                            return None
                    wait_ms = None
                    if enqueued_mono > 0.0:
                        try:
                            wait_ms = int(max(0, round((time.monotonic() - enqueued_mono) * 1000.0)))
                        except Exception:
                            wait_ms = None
                    audio_end_ms = int(max(0, int(item.get("audio_end_ms") or 0)))
                    final_covered_ms = 0
                    try:
                        final_snapshot = LIVE_SESSIONS.semilive_result_snapshot(session_id)
                        final_covered_ms = int(max(0, int((final_snapshot or {}).get("final_covered_ms") or 0)))
                    except Exception:
                        final_covered_ms = 0
                    staleness_ms = int(
                        max(0, int(semilive_recording_duration_ms) - audio_end_ms)
                    )
                    stale_by_final = bool(audio_end_ms > 0 and final_covered_ms >= audio_end_ms)
                    stale_by_cursor = bool(
                        (LIVE_SEMILIVE_SPECULATIVE_MAX_STALENESS_MS > 0)
                        and (staleness_ms > int(LIVE_SEMILIVE_SPECULATIVE_MAX_STALENESS_MS))
                        and not semilive_recording_finalized
                        and final_covered_ms <= 0
                    )
                    stale = bool(stale_by_final or stale_by_cursor)
                    if stale:
                        semilive_speculative_metrics["dropped_stale"] = int(
                            max(0, int(semilive_speculative_metrics.get("dropped_stale") or 0)) + 1
                        )
                        _append_semilive_log(
                            "semilive_speculative_dropped_stale",
                            speculative_seq=int(spec_seq),
                            job_id=job_id,
                            staleness_ms=int(max(0, staleness_ms)),
                            final_covered_ms=int(max(0, final_covered_ms)),
                            stale_reason=("final_covered" if stale_by_final else "live_cursor"),
                            text_chars=len(str(poll.text or "")),
                        )
                    else:
                        semilive_speculative_metrics["shown"] = int(
                            max(0, int(semilive_speculative_metrics.get("shown") or 0)) + 1
                        )
                        if semilive_speculative_metrics.get("time_to_first_speculative_ms") is None:
                            semilive_speculative_metrics["time_to_first_speculative_ms"] = int(
                                max(0, int(item.get("audio_end_ms") or 0))
                            )
                        try:
                            LIVE_SESSIONS.update_semilive_speculative_preview(
                                session_id,
                                text=str(poll.text or ""),
                                speculative_seq=int(spec_seq),
                                audio_end_ms=int(max(0, int(item.get("audio_end_ms") or 0))),
                            )
                        except Exception:
                            pass
                        try:
                            LIVE_SESSIONS.update_semilive_speculative_metrics(
                                session_id,
                                enqueued=int(max(0, int(semilive_speculative_metrics.get("enqueued") or 0))),
                                shown=int(max(0, int(semilive_speculative_metrics.get("shown") or 0))),
                                dropped_busy=int(max(0, int(semilive_speculative_metrics.get("dropped_busy") or 0))),
                                dropped_stale=int(max(0, int(semilive_speculative_metrics.get("dropped_stale") or 0))),
                                time_to_first_speculative_ms=(
                                    int(semilive_speculative_metrics.get("time_to_first_speculative_ms"))
                                    if semilive_speculative_metrics.get("time_to_first_speculative_ms") is not None
                                    else None
                                ),
                            )
                        except Exception:
                            pass
                        _append_semilive_log(
                            "semilive_speculative_ready",
                            speculative_seq=int(spec_seq),
                            job_id=job_id,
                            staleness_ms=int(max(0, staleness_ms)),
                            final_covered_ms=int(max(0, final_covered_ms)),
                            text_chars=len(str(poll.text or "")),
                            segments_count=len(poll.segments or []),
                            wait_ms=(int(wait_ms) if wait_ms is not None else None),
                            remote_submit_attempts=_status_int("asr_remote_submit_attempts"),
                            remote_status_attempts_total=_status_int("asr_remote_status_attempts_total"),
                            remote_status_http_calls=_status_int("asr_remote_status_http_calls"),
                            remote_cancel_attempts=_status_int("asr_remote_cancel_attempts"),
                            blob_fetch_ms=_status_float("asr_blob_fetch_ms"),
                        )
                elif poll_state == "error":
                    _append_semilive_log(
                        "semilive_speculative_error",
                        speculative_seq=int(spec_seq),
                        job_id=job_id,
                        error=str(poll.error or ""),
                        status={"state": poll.state, "ok": poll.ok, "done": poll.done},
                    )
            if poll.done:
                semilive_speculative_jobs_pending.pop(spec_seq, None)

    async def _process_semilive_speculative_jobs(*, force_poll: bool = False) -> None:
        if not semilive_speculative_enabled:
            return
        await _maybe_enqueue_speculative_job()
        await _poll_speculative_jobs(force=force_poll)

    def _ingest_closed_chunks(chunks: list[Any]) -> None:
        nonlocal semilive_chunks_total, semilive_chunk_index_next, semilive_chunker_snapshot
        nonlocal semilive_chunk_jobs_to_enqueue
        if not chunks:
            return
        for chunk in chunks:
            semilive_chunks_total += 1
            semilive_chunk_index_next = max(
                int(semilive_chunk_index_next),
                int(getattr(chunk, "chunk_index", semilive_chunk_index_next)) + 1,
            )
            _append_semilive_log(
                "semilive_chunk_closed",
                chunk=getattr(chunk, "to_dict", lambda: {})(),
            )
            if semilive_chunk_jobs_enabled and chunk_bridge is not None:
                semilive_chunk_jobs_to_enqueue.append(chunk)
        if chunker is not None:
            try:
                semilive_chunker_snapshot = dict(chunker.snapshot())
            except Exception:
                semilive_chunker_snapshot = {}
        if (semilive_chunk_jobs_pending or semilive_chunk_jobs_to_enqueue) and semilive_finalization_state not in {"error", "ready"}:
            # Shadow processing is in progress, even if chunk jobs are feature-flagged.
            pass
        _update_semilive_session_state()

    def _sync_semilive_counts_from_result(result: dict[str, Any]) -> None:
        nonlocal semilive_chunks_total, semilive_chunks_done, semilive_chunks_failed
        nonlocal semilive_finalization_state
        semilive_chunks_total = int(max(0, int(result.get("chunks_total") or semilive_chunks_total)))
        semilive_chunks_done = int(max(0, int(result.get("chunks_done") or semilive_chunks_done)))
        semilive_chunks_failed = int(max(0, int(result.get("chunks_failed") or semilive_chunks_failed)))
        if semilive_recording_finalized:
            if semilive_chunk_jobs_pending or semilive_chunk_jobs_to_enqueue:
                semilive_finalization_state = "processing_chunks"
            elif semilive_chunks_failed > 0:
                semilive_finalization_state = "error"
            elif semilive_chunk_jobs_enabled and (semilive_chunks_total > 0 or semilive_chunks_done > 0):
                semilive_finalization_state = "ready"

    async def _record_semilive_chunk_job_state(
        *,
        chunk_index: int,
        t0_ms: int,
        t1_ms: int,
        state: str,
        text: str = "",
        segments: list[dict[str, Any]] | None = None,
        error: str = "",
        chunk_meta: dict[str, Any] | None = None,
        job_status: dict[str, Any] | None = None,
    ) -> None:
        meta = chunk_meta if isinstance(chunk_meta, dict) else {}
        status_obj = job_status if isinstance(job_status, dict) else {}

        def _parse_status_float(name: str) -> float | None:
            if name not in status_obj or status_obj.get(name) is None:
                return None
            try:
                return max(0.0, float(status_obj.get(name)))
            except Exception:
                return None

        try:
            result = LIVE_SESSIONS.record_semilive_chunk_result(
                session_id,
                chunk_index=int(chunk_index),
                t0_ms=int(t0_ms),
                t1_ms=int(t1_ms),
                text=str(text or ""),
                segments=segments,
                state=str(state or "ready"),
                error=str(error or ""),
                reason=str(meta.get("reason") or ""),
                speech_frames=(int(meta.get("speech_frames")) if meta.get("speech_frames") is not None else None),
                silence_frames_tail=(
                    int(meta.get("silence_frames_tail")) if meta.get("silence_frames_tail") is not None else None
                ),
                chunk_duration_ms=(int(meta.get("duration_ms")) if meta.get("duration_ms") is not None else None),
                asr_pipeline_time_s=_parse_status_float("asr_timing_whisperx_total_s"),
                asr_transcribe_time_s=_parse_status_float("asr_timing_whisperx_transcribe_s"),
            )
            _sync_semilive_counts_from_result(result)
            _update_semilive_session_state()
        except Exception as e:
            _append_semilive_log(
                "semilive_chunk_result_store_error",
                chunk_index=int(chunk_index),
                state=str(state or ""),
                error=f"{type(e).__name__}: {e}",
            )

    async def _drain_semilive_chunk_enqueues() -> None:
        nonlocal semilive_finalization_state
        nonlocal semilive_shadow_disabled_reason
        if not semilive_chunk_jobs_enabled or chunk_bridge is None:
            return
        while semilive_chunk_jobs_to_enqueue:
            chunk = semilive_chunk_jobs_to_enqueue.pop(0)
            chunk_meta = getattr(chunk, "to_dict", lambda: {})()
            chunk_index = int(getattr(chunk, "chunk_index", 0))
            t0_ms = int(getattr(chunk, "t0_ms", 0))
            t1_ms = int(getattr(chunk, "t1_ms", t0_ms))
            initial_prompt, initial_prompt_meta = _build_semilive_initial_prompt()
            if semilive_finalization_state not in {"error"}:
                semilive_finalization_state = "processing_chunks"
            final_beam_override = (
                int(max(1, int(LIVE_SEMILIVE_FINAL_BEAM_SIZE_OVERRIDE)))
                if int(max(0, int(LIVE_SEMILIVE_FINAL_BEAM_SIZE_OVERRIDE or 0))) > 0
                else None
            )
            try:
                enq = await asyncio.to_thread(
                    chunk_bridge.enqueue_chunk_pcm16,
                    session_id=session_id,
                    chunk_index=chunk_index,
                    t0_ms=t0_ms,
                    t1_ms=t1_ms,
                    pcm16le=bytes(getattr(chunk, "pcm16le", b"") or b""),
                    language=LIVE_SEMILIVE_CHUNK_LANGUAGE,
                    asr_beam_size=final_beam_override,
                    initial_prompt=initial_prompt,
                )
                semilive_chunk_jobs_pending[chunk_index] = {
                    "chunk_index": chunk_index,
                    "t0_ms": t0_ms,
                    "t1_ms": t1_ms,
                    "job_id": str(enq.job_id),
                    "job_dir": str(enq.job_dir),
                    "state": "queued",
                    "reported_state": "",
                    "enqueued_mono": time.monotonic(),
                    "last_poll_mono": 0.0,
                    "chunk_meta": dict(chunk_meta) if isinstance(chunk_meta, dict) else {},
                    "initial_prompt_meta": dict(initial_prompt_meta) if isinstance(initial_prompt_meta, dict) else {},
                }
                await _record_semilive_chunk_job_state(
                    chunk_index=chunk_index,
                    t0_ms=t0_ms,
                    t1_ms=t1_ms,
                    state="queued",
                    chunk_meta=dict(chunk_meta) if isinstance(chunk_meta, dict) else None,
                )
                _append_semilive_log(
                    "semilive_chunk_enqueued",
                    chunk=dict(chunk_meta) if isinstance(chunk_meta, dict) else {"chunk_index": chunk_index},
                    job=enq.to_dict(),
                    final_beam_size_override=(
                        int(final_beam_override) if final_beam_override is not None else None
                    ),
                    initial_prompt_meta=dict(initial_prompt_meta) if isinstance(initial_prompt_meta, dict) else {},
                )
            except Exception as e:
                semilive_shadow_disabled_reason = f"chunk_enqueue_failed:{type(e).__name__}"
                semilive_finalization_state = "error"
                await _record_semilive_chunk_job_state(
                    chunk_index=chunk_index,
                    t0_ms=t0_ms,
                    t1_ms=t1_ms,
                    state="error",
                    error=f"{type(e).__name__}: {e}",
                    chunk_meta=dict(chunk_meta) if isinstance(chunk_meta, dict) else None,
                )
                _append_semilive_log(
                    "semilive_chunk_enqueue_error",
                    chunk=dict(chunk_meta) if isinstance(chunk_meta, dict) else {"chunk_index": chunk_index},
                    error=f"{type(e).__name__}: {e}",
                )
                _update_semilive_session_state()

    async def _poll_semilive_chunk_jobs(*, force: bool = False) -> None:
        nonlocal semilive_chunk_jobs_last_poll_mono
        nonlocal semilive_finalization_state
        nonlocal semilive_shadow_disabled_reason
        if not semilive_chunk_jobs_enabled or chunk_bridge is None:
            return
        now_mono = time.monotonic()
        if not force and (now_mono - semilive_chunk_jobs_last_poll_mono) < LIVE_SEMILIVE_CHUNK_POLL_INTERVAL_S:
            return
        semilive_chunk_jobs_last_poll_mono = now_mono
        if semilive_chunk_jobs_pending and semilive_finalization_state not in {"error"}:
            semilive_finalization_state = "processing_chunks"
        for chunk_index in sorted(list(semilive_chunk_jobs_pending.keys())):
            item = semilive_chunk_jobs_pending.get(chunk_index)
            if not item:
                continue
            item["last_poll_mono"] = now_mono
            t0_ms = int(item.get("t0_ms") or 0)
            t1_ms = int(item.get("t1_ms") or t0_ms)
            job_id = str(item.get("job_id") or "")
            enqueued_mono = float(item.get("enqueued_mono") or 0.0)
            chunk_meta = item.get("chunk_meta") if isinstance(item.get("chunk_meta"), dict) else None
            if not job_id:
                continue
            try:
                poll = await asyncio.to_thread(
                    chunk_bridge.poll_job,
                    job_id,
                    t0_offset_ms=t0_ms,
                )
            except FileNotFoundError:
                # Worker may not have published the dir yet; keep queued.
                continue
            except Exception as e:
                await _record_semilive_chunk_job_state(
                    chunk_index=chunk_index,
                    t0_ms=t0_ms,
                    t1_ms=t1_ms,
                    state="error",
                    error=f"{type(e).__name__}: {e}",
                    chunk_meta=chunk_meta,
                )
                semilive_chunk_jobs_pending.pop(chunk_index, None)
                semilive_shadow_disabled_reason = f"chunk_poll_failed:{type(e).__name__}"
                semilive_finalization_state = "error"
                _append_semilive_log(
                    "semilive_chunk_poll_error",
                    chunk_index=chunk_index,
                    job_id=job_id,
                    error=f"{type(e).__name__}: {e}",
                )
                continue

            poll_state = "ready" if poll.ok else ("error" if poll.done else str(poll.state or "queued"))
            if str(item.get("reported_state") or "") != poll_state:
                item["reported_state"] = poll_state
                if poll_state == "ready":
                    status_obj = dict(poll.status) if isinstance(poll.status, dict) else {}
                    def _status_int(name: str) -> int | None:
                        if name not in status_obj or status_obj.get(name) is None:
                            return None
                        try:
                            return int(max(0, int(status_obj.get(name))))
                        except Exception:
                            return None
                    def _status_float(name: str) -> float | None:
                        if name not in status_obj or status_obj.get(name) is None:
                            return None
                        try:
                            return max(0.0, float(status_obj.get(name)))
                        except Exception:
                            return None
                    wait_ms = None
                    if enqueued_mono > 0.0:
                        try:
                            wait_ms = int(max(0, round((time.monotonic() - enqueued_mono) * 1000.0)))
                        except Exception:
                            wait_ms = None
                    await _record_semilive_chunk_job_state(
                        chunk_index=chunk_index,
                        t0_ms=t0_ms,
                        t1_ms=t1_ms,
                        state="ready",
                        text=str(poll.text or ""),
                        segments=[dict(seg) for seg in (poll.segments or []) if isinstance(seg, dict)],
                        chunk_meta=chunk_meta,
                        job_status=dict(poll.status) if isinstance(poll.status, dict) else None,
                    )
                    _append_semilive_log(
                        "semilive_chunk_ready",
                        chunk_index=chunk_index,
                        job_id=job_id,
                        status={"state": poll.state, "ok": poll.ok, "done": poll.done},
                        text_chars=len(str(poll.text or "")),
                        segments_count=len(poll.segments or []),
                        wait_ms=(int(wait_ms) if wait_ms is not None else None),
                        remote_submit_attempts=_status_int("asr_remote_submit_attempts"),
                        remote_status_attempts_total=_status_int("asr_remote_status_attempts_total"),
                        remote_status_http_calls=_status_int("asr_remote_status_http_calls"),
                        remote_cancel_attempts=_status_int("asr_remote_cancel_attempts"),
                        blob_fetch_ms=_status_float("asr_blob_fetch_ms"),
                    )
                elif poll_state == "error":
                    await _record_semilive_chunk_job_state(
                        chunk_index=chunk_index,
                        t0_ms=t0_ms,
                        t1_ms=t1_ms,
                        state="error",
                        error=str(poll.error or f"job_state:{poll.state}"),
                        chunk_meta=chunk_meta,
                        job_status=dict(poll.status) if isinstance(poll.status, dict) else None,
                    )
                    _append_semilive_log(
                        "semilive_chunk_error",
                        chunk_index=chunk_index,
                        job_id=job_id,
                        status={"state": poll.state, "ok": poll.ok, "done": poll.done},
                        error=str(poll.error or ""),
                    )
                else:
                    await _record_semilive_chunk_job_state(
                        chunk_index=chunk_index,
                        t0_ms=t0_ms,
                        t1_ms=t1_ms,
                        state=poll_state,
                        chunk_meta=chunk_meta,
                        job_status=dict(poll.status) if isinstance(poll.status, dict) else None,
                    )
            item["state"] = poll_state
            if poll.done:
                semilive_chunk_jobs_pending.pop(chunk_index, None)

        if semilive_recording_finalized and not semilive_chunk_jobs_pending and not semilive_chunk_jobs_to_enqueue:
            if semilive_chunks_failed > 0:
                semilive_finalization_state = "error"
            elif semilive_chunk_jobs_enabled and (semilive_chunks_total > 0 or semilive_chunks_done > 0):
                semilive_finalization_state = "ready"
        _update_semilive_session_state()

    async def _process_semilive_chunk_jobs(*, force_poll: bool = False) -> None:
        if not semilive_chunk_jobs_enabled:
            return
        await _drain_semilive_chunk_enqueues()
        await _poll_semilive_chunk_jobs(force=force_poll)

    async def _await_semilive_chunk_jobs_until_done(*, timeout_s: float) -> None:
        if not semilive_chunk_jobs_enabled:
            return
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while True:
            await _process_semilive_chunk_jobs(force_poll=True)
            if not semilive_chunk_jobs_to_enqueue and not semilive_chunk_jobs_pending:
                return
            if time.monotonic() >= deadline:
                return
            await asyncio.sleep(0.25)

    def _finalize_semilive_shadow(*, reason: str) -> None:
        nonlocal semilive_recording_finalized
        nonlocal semilive_recording_state
        nonlocal semilive_recording_path
        nonlocal semilive_recording_bytes
        nonlocal semilive_recording_duration_ms
        nonlocal semilive_chunker_snapshot
        nonlocal semilive_finalization_state
        nonlocal semilive_shadow_disabled_reason
        if semilive_recording_finalized:
            return
        semilive_finalization_state = "finalizing"
        if chunker is not None:
            try:
                _ingest_closed_chunks(chunker.flush_tail())
                semilive_chunker_snapshot = dict(chunker.snapshot())
            except Exception as e:
                semilive_shadow_disabled_reason = f"chunker_finalize_failed:{type(e).__name__}"
                semilive_recording_state = "error"
                semilive_finalization_state = "error"
                _append_semilive_log(
                    "semilive_chunker_finalize_error",
                    reason=reason,
                    error=f"{type(e).__name__}: {e}",
                )
        if recorder is not None:
            try:
                rs = recorder.finalize()
                semilive_recording_path = str(rs.wav_path)
                semilive_recording_bytes = int(rs.bytes_written)
                semilive_recording_duration_ms = int(rs.duration_ms)
                semilive_recording_state = "finalized"
                if semilive_finalization_state != "error":
                    semilive_finalization_state = "recording_finalized"
                _append_semilive_log(
                    "semilive_recording_finalized",
                    reason=reason,
                    recording=rs.to_dict(),
                    chunker_snapshot=dict(semilive_chunker_snapshot),
                )
            except Exception as e:
                semilive_shadow_disabled_reason = f"recording_finalize_failed:{type(e).__name__}"
                semilive_recording_state = "error"
                semilive_finalization_state = "error"
                _append_semilive_log(
                    "semilive_recording_finalize_error",
                    reason=reason,
                    error=f"{type(e).__name__}: {e}",
                )
        else:
            if semilive_finalization_state != "error":
                semilive_finalization_state = "idle"
        semilive_recording_finalized = True
        _update_semilive_session_state()

    try:
        await send_event(
            ready_event(
                session_id,
                message="Live websocket connected. Send binary PCM16 frames and JSON controls.",
                engine="semilive_chunked",
            )
        )

        # Semilive pipeline (WAV recorder + silence chunker + chunk batch jobs).
        try:
            recorder = LiveWavRecorder(
                session_id=session_id,
                sample_rate_hz=LIVE_AUDIO_SAMPLE_RATE_HZ,
                channels=LIVE_AUDIO_CHANNELS,
            )
            rec_snap = recorder.start()
            chunker = LiveAudioChunker(
                config=LiveChunkerConfig(
                    sample_rate_hz=LIVE_AUDIO_SAMPLE_RATE_HZ,
                    channels=LIVE_AUDIO_CHANNELS,
                    energy_threshold=LIVE_SEMILIVE_CHUNK_ENERGY_THRESHOLD,
                    silence_threshold_ms=LIVE_SEMILIVE_CHUNK_SILENCE_MS,
                    max_chunk_ms=LIVE_SEMILIVE_CHUNK_MAX_MS,
                    min_chunk_ms=LIVE_SEMILIVE_CHUNK_MIN_MS,
                    pre_roll_ms=LIVE_SEMILIVE_CHUNK_PRE_ROLL_MS,
                )
            )
            if semilive_chunk_jobs_enabled:
                chunk_bridge = LiveChunkBatchBridge(
                    sample_rate_hz=LIVE_AUDIO_SAMPLE_RATE_HZ,
                    channels=LIVE_AUDIO_CHANNELS,
                    language=LIVE_SEMILIVE_CHUNK_LANGUAGE,
                )
            semilive_recording_state = "recording"
            semilive_recording_path = str(rec_snap.wav_path)
            semilive_recording_bytes = int(rec_snap.bytes_written)
            semilive_recording_duration_ms = int(rec_snap.duration_ms)
            semilive_chunk_index_next = 0
            semilive_chunks_total = 0
            semilive_chunks_done = 0
            semilive_chunks_failed = 0
            semilive_finalization_state = "recording"
            semilive_chunker_snapshot = dict(chunker.snapshot())
            _update_semilive_session_state()
            _append_semilive_log(
                "semilive_shadow_started",
                recording=rec_snap.to_dict(),
                chunk_jobs_enabled=bool(semilive_chunk_jobs_enabled),
                speculative_enabled=bool(semilive_speculative_enabled),
                speculative_config={
                    "interval_ms": int(LIVE_SEMILIVE_SPECULATIVE_INTERVAL_MS),
                    "window_ms": int(LIVE_SEMILIVE_SPECULATIVE_WINDOW_MS),
                    "overlap_ms": int(LIVE_SEMILIVE_SPECULATIVE_OVERLAP_MS),
                    "effective_window_ms": int(semilive_speculative_effective_window_ms),
                    "max_staleness_ms": int(LIVE_SEMILIVE_SPECULATIVE_MAX_STALENESS_MS),
                    "require_no_final_pending": bool(LIVE_SEMILIVE_SPECULATIVE_REQUIRE_NO_FINAL_PENDING),
                    "final_beam_size_override": (
                        int(max(1, int(LIVE_SEMILIVE_FINAL_BEAM_SIZE_OVERRIDE)))
                        if int(max(0, int(LIVE_SEMILIVE_FINAL_BEAM_SIZE_OVERRIDE or 0))) > 0
                        else None
                    ),
                },
                chunker_config=chunker.config.__dict__,
                chunker_snapshot=dict(semilive_chunker_snapshot),
            )
        except Exception as e:
            recorder = None
            chunker = None
            semilive_recording_state = "error"
            semilive_finalization_state = "error"
            semilive_shadow_disabled_reason = f"semilive_init_failed:{type(e).__name__}"
            _update_semilive_session_state()
            _append_semilive_log(
                "semilive_shadow_init_error",
                error=f"{type(e).__name__}: {e}",
            )

        while True:
            incoming = await websocket.receive()

            if incoming.get("type") == "websocket.disconnect":
                stop_reason = "client_disconnected"
                break

            raw_bytes = incoming.get("bytes")
            if raw_bytes is not None:
                snapshot = LIVE_SESSIONS.record_audio(session_id, byte_count=len(raw_bytes))
                if recorder is not None:
                    try:
                        rec_snap = recorder.append_pcm16(raw_bytes)
                        semilive_recording_bytes = int(rec_snap.bytes_written)
                        semilive_recording_duration_ms = int(rec_snap.duration_ms)
                        semilive_recording_path = str(rec_snap.wav_path)
                    except Exception as e:
                        semilive_shadow_disabled_reason = f"recording_append_failed:{type(e).__name__}"
                        semilive_recording_state = "error"
                        semilive_finalization_state = "error"
                        _append_semilive_log(
                            "semilive_recording_append_error",
                            error=f"{type(e).__name__}: {e}",
                            at_frame=int(snapshot.get("frames_received") or 0),
                        )
                        try:
                            recorder.abort()
                        except Exception:
                            pass
                        recorder = None
                        _update_semilive_session_state()
                if chunker is not None:
                    try:
                        closed_chunks = chunker.feed_pcm16(raw_bytes)
                        semilive_chunker_snapshot = dict(chunker.snapshot())
                        _ingest_closed_chunks(closed_chunks)
                    except Exception as e:
                        semilive_shadow_disabled_reason = f"chunker_feed_failed:{type(e).__name__}"
                        semilive_recording_state = "error"
                        semilive_finalization_state = "error"
                        _append_semilive_log(
                            "semilive_chunker_feed_error",
                            error=f"{type(e).__name__}: {e}",
                            at_frame=int(snapshot.get("frames_received") or 0),
                        )
                        chunker = None
                        _update_semilive_session_state()
                _append_speculative_pcm(raw_bytes)
                await _process_semilive_chunk_jobs(force_poll=False)
                await _process_semilive_speculative_jobs(force_poll=False)
                # Emit transport + semilive stats at startup and periodically.
                should_emit_stats = (
                    snapshot["frames_received"] == 1
                    or (snapshot["frames_received"] % 50) == 0
                )
                if should_emit_stats:
                    stats_payload = stats_event(
                        session_id,
                        bytes_received=snapshot["bytes_received"],
                        frames_received=snapshot["frames_received"],
                        controls_received=snapshot["controls_received"],
                        uptime_s=snapshot["age_s"],
                        live_mode="semilive_chunked",
                        semilive_recording_state=str(semilive_recording_state or ""),
                        semilive_recording_bytes=int(max(0, semilive_recording_bytes)),
                        semilive_recording_duration_ms=int(max(0, semilive_recording_duration_ms)),
                        semilive_chunk_index_next=int(max(0, semilive_chunk_index_next)),
                        semilive_chunks_total=int(max(0, semilive_chunks_total)),
                        semilive_chunks_done=int(max(0, semilive_chunks_done)),
                        semilive_chunks_failed=int(max(0, semilive_chunks_failed)),
                        semilive_finalization_state=str(semilive_finalization_state or ""),
                        semilive_chunk_jobs_enabled=bool(semilive_chunk_jobs_enabled),
                        semilive_chunk_jobs_pending=int(max(0, len(semilive_chunk_jobs_pending))),
                        semilive_chunk_jobs_to_enqueue=int(max(0, len(semilive_chunk_jobs_to_enqueue))),
                        semilive_speculative_enabled=bool(semilive_speculative_enabled),
                        semilive_speculative_pending=int(max(0, len(semilive_speculative_jobs_pending))),
                        semilive_speculative_enqueued=int(
                            max(0, int(semilive_speculative_metrics.get("enqueued") or 0))
                        ),
                        semilive_speculative_shown=int(max(0, int(semilive_speculative_metrics.get("shown") or 0))),
                        semilive_speculative_dropped_busy=int(
                            max(0, int(semilive_speculative_metrics.get("dropped_busy") or 0))
                        ),
                        semilive_speculative_dropped_stale=int(
                            max(0, int(semilive_speculative_metrics.get("dropped_stale") or 0))
                        ),
                        semilive_time_to_first_speculative_ms=(
                            int(semilive_speculative_metrics.get("time_to_first_speculative_ms"))
                            if semilive_speculative_metrics.get("time_to_first_speculative_ms") is not None
                            else None
                        ),
                        semilive_shadow_disabled_reason=str(semilive_shadow_disabled_reason or ""),
                        semilive_chunker_chunk_open=bool(semilive_chunker_snapshot.get("chunk_open")),
                        semilive_chunker_active_chunk_duration_ms=int(
                            max(0, int(semilive_chunker_snapshot.get("active_chunk_duration_ms") or 0))
                        ),
                        semilive_chunker_pre_roll_frames=int(
                            max(0, int(semilive_chunker_snapshot.get("pre_roll_frames_buffered") or 0))
                        ),
                    )
                    try:
                        LIVE_SESSIONS.append_stats_log(session_id, stats_payload)
                    except Exception:
                        pass
                    await send_event(stats_payload)
                continue

            raw_text = incoming.get("text")
            if raw_text is None:
                await send_event(
                    error_event(
                        session_id,
                        code="invalid_frame",
                        message="Expected binary audio frame or JSON control message.",
                    )
                )
                continue

            control_type, _obj, parse_err = parse_client_message(raw_text)
            if parse_err:
                await send_event(
                    error_event(
                        session_id,
                        code=parse_err,
                        message="Invalid control message.",
                    )
                )
                continue

            LIVE_SESSIONS.record_control(session_id)

            if control_type == "ping":
                await send_event(pong_event(session_id))
                continue

            if control_type == "start":
                snapshot = LIVE_SESSIONS.mark_state(session_id, state="listening")
                semilive_recording_state = "recording"
                _update_semilive_session_state()
                await send_event(
                    control_ack_event(
                        session_id,
                        control_type="start",
                        state=snapshot["state"],
                    )
                )
                continue

            if control_type == "pause":
                snapshot = LIVE_SESSIONS.mark_state(session_id, state="paused")
                semilive_recording_state = "paused"
                _update_semilive_session_state()
                await send_event(
                    control_ack_event(
                        session_id,
                        control_type="pause",
                        state=snapshot["state"],
                    )
                )
                continue

            if control_type == "resume":
                snapshot = LIVE_SESSIONS.mark_state(session_id, state="listening")
                semilive_recording_state = "recording"
                _update_semilive_session_state()
                await send_event(
                    control_ack_event(
                        session_id,
                        control_type="resume",
                        state=snapshot["state"],
                    )
                )
                continue

            if control_type == "stop":
                stop_reason = "client_stop"
                semilive_result: dict[str, Any] = {}
                _finalize_semilive_shadow(reason="client_stop")
                await _process_semilive_chunk_jobs(force_poll=True)
                if LIVE_SEMILIVE_CHUNK_STOP_WAIT_S > 0:
                    await _await_semilive_chunk_jobs_until_done(timeout_s=LIVE_SEMILIVE_CHUNK_STOP_WAIT_S)
                try:
                    semilive_result = _archive_current_semilive_result(close_reason=stop_reason)
                except Exception:
                    semilive_result = {}
                await send_event(
                    ended_event(
                        session_id,
                        reason=stop_reason,
                        transcript_revision=int(max(0, int(semilive_result.get("transcript_revision") or 0))),
                        final_segments_count=len(semilive_result.get("final_segments") or []),
                        final_text=str(semilive_result.get("final_text") or ""),
                        final_transcript_url=_rooted_path(f"/demo/live/sessions/{session_id}/final"),
                    )
                )
                await websocket.close(code=status.WS_1000_NORMAL_CLOSURE)
                websocket_closed = True
                break

    except WebSocketDisconnect:
        stop_reason = "client_disconnected"
    except Exception as e:
        stop_reason = "server_error"
        try:
            await send_event(
                error_event(
                    session_id,
                    code="internal_error",
                    message=f"{type(e).__name__}: {e}",
                    fatal=True,
                )
            )
        except Exception:
            pass
        if not websocket_closed:
            try:
                await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
            except Exception:
                pass
    finally:
        _finalize_semilive_shadow(reason=stop_reason)
        await _process_semilive_chunk_jobs(force_poll=True)
        if stop_reason == "client_stop":
            if LIVE_SEMILIVE_CHUNK_POST_CLOSE_WAIT_S > 0:
                await _await_semilive_chunk_jobs_until_done(timeout_s=LIVE_SEMILIVE_CHUNK_POST_CLOSE_WAIT_S)
        else:
            if LIVE_SEMILIVE_CHUNK_STOP_WAIT_S > 0:
                await _await_semilive_chunk_jobs_until_done(timeout_s=LIVE_SEMILIVE_CHUNK_STOP_WAIT_S)
        try:
            _archive_current_semilive_result(close_reason=stop_reason)
        except Exception:
            pass
        # Release capacity slot first; downstream close operations may block.
        LIVE_SESSIONS.close_session(session_id, reason=stop_reason)
        if recorder is not None and not semilive_recording_finalized:
            try:
                recorder.abort()
            except Exception:
                pass


def _get_speculative_timing_runtime() -> dict[str, int]:
    return {
        "interval_ms": int(max(200, int(LIVE_SEMILIVE_SPECULATIVE_INTERVAL_MS))),
        "window_ms": int(max(200, int(LIVE_SEMILIVE_SPECULATIVE_WINDOW_MS))),
        "overlap_ms": int(max(0, int(LIVE_SEMILIVE_SPECULATIVE_OVERLAP_MS))),
    }


def _set_speculative_timing_runtime(*, interval_ms: int, window_ms: int, overlap_ms: int) -> dict[str, int]:
    global LIVE_SEMILIVE_SPECULATIVE_INTERVAL_MS
    global LIVE_SEMILIVE_SPECULATIVE_WINDOW_MS
    global LIVE_SEMILIVE_SPECULATIVE_OVERLAP_MS
    prev = _get_speculative_timing_runtime()
    LIVE_SEMILIVE_SPECULATIVE_INTERVAL_MS = int(max(200, int(interval_ms)))
    LIVE_SEMILIVE_SPECULATIVE_WINDOW_MS = int(max(LIVE_SEMILIVE_SPECULATIVE_INTERVAL_MS, int(window_ms)))
    LIVE_SEMILIVE_SPECULATIVE_OVERLAP_MS = int(max(0, int(overlap_ms)))
    return prev


def _set_final_beam_override_runtime(*, final_beam_size: int | None) -> int:
    global LIVE_SEMILIVE_FINAL_BEAM_SIZE_OVERRIDE
    prev = int(max(0, int(LIVE_SEMILIVE_FINAL_BEAM_SIZE_OVERRIDE or 0)))
    if final_beam_size is None:
        LIVE_SEMILIVE_FINAL_BEAM_SIZE_OVERRIDE = 0
    else:
        LIVE_SEMILIVE_FINAL_BEAM_SIZE_OVERRIDE = int(max(1, int(final_beam_size)))
    return prev


def _configure_speculative_tuning_runner() -> None:
    speculative_tuning_runner.configure(
        protocol_version=PROTOCOL_VERSION,
        repo_root=_repo_root(),
        live_benchmark_export_root=LIVE_BENCHMARK_EXPORT_ROOT,
        live_sessions=LIVE_SESSIONS,
        rooted_path_cb=_rooted_path,
        autosave_snapshot_cb=_try_autosave_live_benchmark_snapshot,
        autosave_spec_trace_cb=_try_autosave_speculative_lane_trace,
        get_spec_timing_cb=_get_speculative_timing_runtime,
        set_spec_timing_cb=_set_speculative_timing_runtime,
        set_final_beam_override_cb=_set_final_beam_override_runtime,
        live_audio_sample_width_bytes=LIVE_AUDIO_SAMPLE_WIDTH_BYTES,
        live_audio_bytes_per_second=LIVE_AUDIO_BYTES_PER_SECOND,
        semilive_chunk_stop_wait_s=LIVE_SEMILIVE_CHUNK_STOP_WAIT_S,
        semilive_chunk_post_close_wait_s=LIVE_SEMILIVE_CHUNK_POST_CLOSE_WAIT_S,
        ws_base_url=LIVE_SWEEP_WS_BASE_URL,
    )


def _configure_asr_loadtest_runner() -> None:
    asr_loadtest_runner.configure(
        protocol_version=PROTOCOL_VERSION,
        repo_root=_repo_root(),
        live_benchmark_export_root=LIVE_BENCHMARK_EXPORT_ROOT,
        live_sessions=LIVE_SESSIONS,
        rooted_path_cb=_rooted_path,
        autosave_snapshot_cb=_try_autosave_live_benchmark_snapshot,
        ws_base_url=LIVE_SWEEP_WS_BASE_URL,
        live_audio_sample_width_bytes=LIVE_AUDIO_SAMPLE_WIDTH_BYTES,
        live_audio_bytes_per_second=LIVE_AUDIO_BYTES_PER_SECOND,
        semilive_chunk_stop_wait_s=LIVE_SEMILIVE_CHUNK_STOP_WAIT_S,
        semilive_chunk_post_close_wait_s=LIVE_SEMILIVE_CHUNK_POST_CLOSE_WAIT_S,
        asr_pool_status_url=ASR_POOL_STATUS_URL,
        asr_pool_token=ASR_POOL_TOKEN,
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


def _build_speculative_lane_trace(
    *,
    session_id: str,
    speculative_history: Dict[str, Any],
) -> Dict[str, Any]:
    history = speculative_history if isinstance(speculative_history, dict) else {}
    windows_src = history.get("speculative_windows")
    windows = [dict(w) for w in windows_src] if isinstance(windows_src, list) else []
    open_window_src = history.get("speculative_open_window")
    open_window = [dict(x) for x in open_window_src] if isinstance(open_window_src, list) else []

    trace_windows: list[Dict[str, Any]] = []
    timeline_rows: list[Dict[str, Any]] = []

    for w in windows:
        window_index = int(max(0, _safe_int(w.get("window_index"), 0)))
        final_chunk_raw = str(w.get("target_final_chunk_text_raw") or "")
        final_chunk_effective = str(w.get("target_final_chunk_text_effective") or "")
        final_chunk_rendered = final_chunk_effective or final_chunk_raw
        final_row = {
            "row_type": "final_chunk",
            "window_index": int(window_index),
            "text": final_chunk_rendered,
            "raw_text": final_chunk_raw,
            "effective_text": final_chunk_effective,
            "ended_by_final_chunk_index": (
                int(max(0, _safe_int(w.get("ended_by_final_chunk_index"), 0)))
                if w.get("ended_by_final_chunk_index") is not None
                else None
            ),
            "ended_by_final_t1_ms": (
                int(max(0, _safe_int(w.get("ended_by_final_t1_ms"), 0)))
                if w.get("ended_by_final_t1_ms") is not None
                else None
            ),
        }

        timeline_rows.append(dict(final_row))
        item_rows: list[Dict[str, Any]] = []
        items_src = w.get("items")
        items = [dict(item) for item in items_src] if isinstance(items_src, list) else []
        for item_index, item in enumerate(items, start=1):
            seq = int(max(-1, _safe_int(item.get("speculative_seq"), -1)))
            audio_end_ms = int(max(0, _safe_int(item.get("audio_end_ms"), 0)))
            raw_text = str(item.get("raw_text") or "")
            suffix_text = str(item.get("suffix_text_after_final_dedup") or "")
            merged_text = str(item.get("merged_text_after_seam_dedup") or "")
            raw_row = {
                "row_type": "raw_speculative",
                "window_index": int(window_index),
                "item_index": int(item_index),
                "speculative_seq": int(seq),
                "audio_end_ms": int(audio_end_ms),
                "text": raw_text,
            }
            suffix_row = {
                "row_type": "suffix_speculative",
                "window_index": int(window_index),
                "item_index": int(item_index),
                "speculative_seq": int(seq),
                "audio_end_ms": int(audio_end_ms),
                "text": suffix_text,
            }
            timeline_rows.append(dict(raw_row))
            timeline_rows.append(dict(suffix_row))
            item_rows.append(
                {
                    "item_index": int(item_index),
                    "speculative_seq": int(seq),
                    "audio_end_ms": int(audio_end_ms),
                    "raw_text": raw_text,
                    "suffix_text": suffix_text,
                    "merged_text": merged_text,
                    "seam_dedup_applied": bool(item.get("seam_dedup_applied")),
                    "seam_dedup_words_trimmed": int(max(0, _safe_int(item.get("seam_dedup_words_trimmed"), 0))),
                    "final_dedup_applied": bool(item.get("final_dedup_applied")),
                    "final_dedup_words_trimmed": int(max(0, _safe_int(item.get("final_dedup_words_trimmed"), 0))),
                    "received_at_utc": str(item.get("received_at_utc") or ""),
                }
            )

        trace_windows.append(
            {
                "window_index": int(window_index),
                "started_at_revision": int(max(0, _safe_int(w.get("started_at_revision"), 0))),
                "ended_by_final_revision": (
                    int(max(0, _safe_int(w.get("ended_by_final_revision"), 0)))
                    if w.get("ended_by_final_revision") is not None
                    else None
                ),
                "close_reason": str(w.get("close_reason") or ""),
                "items_count": int(len(item_rows)),
                "final_chunk": final_row,
                "speculative_items": item_rows,
            }
        )

    open_window_rows: list[Dict[str, Any]] = []
    for item_index, item in enumerate(open_window, start=1):
        seq = int(max(-1, _safe_int(item.get("speculative_seq"), -1)))
        audio_end_ms = int(max(0, _safe_int(item.get("audio_end_ms"), 0)))
        raw_text = str(item.get("raw_text") or "")
        suffix_text = str(item.get("suffix_text_after_final_dedup") or "")
        open_window_rows.append(
            {
                "item_index": int(item_index),
                "speculative_seq": int(seq),
                "audio_end_ms": int(audio_end_ms),
                "raw_text": raw_text,
                "suffix_text": suffix_text,
                "merged_text": str(item.get("merged_text_after_seam_dedup") or ""),
            }
        )

    return {
        "metric_version": "speculative_lane_trace_v1",
        "session_id": str(session_id or ""),
        "history_source": str(history.get("source") or ""),
        "finalization_state": str(history.get("finalization_state") or ""),
        "transcript_revision": int(max(0, _safe_int(history.get("transcript_revision"), 0))),
        "windows_count": int(len(trace_windows)),
        "timeline_rows_count": int(len(timeline_rows)),
        "windows": trace_windows,
        "timeline_rows": timeline_rows,
        "open_window": {
            "window_index": int(max(0, _safe_int(history.get("speculative_open_window_index"), 0))),
            "started_at_revision": int(max(0, _safe_int(history.get("speculative_open_window_started_revision"), 0))),
            "items_count": int(len(open_window_rows)),
            "items": open_window_rows,
        },
    }


def _try_autosave_speculative_lane_trace(session_id: str) -> None:
    sid = str(session_id or "").strip()
    if not sid:
        return
    try:
        history = LIVE_SESSIONS.semilive_speculative_history_snapshot(sid)
    except Exception:
        return
    try:
        trace = _build_speculative_lane_trace(
            session_id=sid,
            speculative_history=history,
        )
        trace_envelope = {
            "protocol_version": PROTOCOL_VERSION,
            "session_id": sid,
            "ready": True,
            "speculative_lane_trace": trace,
        }
        _try_autosave_live_benchmark_snapshot(
            session_id=sid,
            artifact_name="speculative-lane-trace",
            envelope=trace_envelope,
            request_meta={"source": "final-quality"},
        )
    except Exception as e:
        print(f"[speculative-lane-trace-autosave] failed session={sid}: {type(e).__name__}: {e}")


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
