from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Request, WebSocket
from fastapi.responses import FileResponse, Response

from live.results.exports import (
    build_live_result_metrics_snapshot,
    build_live_result_envelope,
    list_live_benchmark_exports,
    live_pc_events_to_text,
    live_recording_wav_path_from_result,
    live_result_to_plain_text,
    live_result_to_srt_text,
    safe_filename,
    try_autosave_live_benchmark_snapshot,
)
from live.config import (
    LIVE_ASR_LANGUAGE,
    LIVE_AUDIO_CHANNELS,
    LIVE_AUDIO_SAMPLE_RATE_HZ,
    LIVE_ENGINE,
    LIVE_SESSIONS,
    live_benchmark_tuning_snapshot,
    live_engine_rolling_context_config,
    rooted_path,
)
from live.quality.fixture_scoring import score_live_text_against_fixture
from live.runtime.protocol import PROTOCOL_VERSION
from live.runtime.util import LIVE_ASR_LANGUAGE_ERROR, parse_live_asr_language
from live.runtime.ws_session import run_live_session_ws

router = APIRouter()


def parse_live_session_asr_language(request: Request) -> str | None:
    try:
        return parse_live_asr_language(request.query_params.get("language"))
    except ValueError:
        raise HTTPException(status_code=400, detail=LIVE_ASR_LANGUAGE_ERROR)


def parse_live_session_ttl_override(request: Request) -> int | None:
    ttl_raw = request.query_params.get("ttl_s")
    if ttl_raw is None or str(ttl_raw).strip() == "":
        return None
    try:
        ttl_override = int(str(ttl_raw).strip())
    except ValueError:
        raise HTTPException(status_code=400, detail="ttl_s must be an integer number of seconds")
    if ttl_override < 10 or ttl_override > 21600:
        raise HTTPException(status_code=400, detail="ttl_s out of range (10..21600 seconds)")
    return ttl_override


def ws_url_for_request(request: Request, ws_path: str) -> str:
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


@router.post("/demo/live/sessions")
def create_live_session(request: Request) -> Dict[str, Any]:
    ttl_override = parse_live_session_ttl_override(request)
    session_asr_language = parse_live_session_asr_language(request)

    try:
        session = LIVE_SESSIONS.create_session(
            ttl_seconds=ttl_override,
            asr_language=session_asr_language,
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
    ws_path = rooted_path(f"/demo/live/sessions/{session_id}/ws")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "live_engine": LIVE_ENGINE,
        "session": session,
        "asr_language_default": LIVE_ASR_LANGUAGE,
        "asr_language_effective": str(session.get("asr_language") or LIVE_ASR_LANGUAGE or ""),
        "asr_language_source": (
            "session_override"
            if str(session.get("asr_language") or "").strip()
            else ("service_default" if LIVE_ASR_LANGUAGE else "auto_detect")
        ),
        "ws_path": ws_path,
        "ws_url": ws_url_for_request(request, ws_path),
        "audio_input": {
            "format": "pcm16le",
            "sample_rate_hz": LIVE_AUDIO_SAMPLE_RATE_HZ,
            "channels": LIVE_AUDIO_CHANNELS,
        },
    }


@router.get("/demo/live/sessions/{session_id}")
def get_live_session(session_id: str) -> Dict[str, Any]:
    try:
        session = LIVE_SESSIONS.session_payload(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Live session not found")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "live_engine": LIVE_ENGINE,
        "session": session,
    }


@router.get("/demo/live/sessions/{session_id}/final")
def get_live_session_final(session_id: str) -> Dict[str, Any]:
    try:
        archive = LIVE_SESSIONS.archive_payload(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Live session final transcript not found")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "session_id": str(session_id),
        "archive": archive,
    }


@router.get("/demo/live/sessions/{session_id}/result")
def get_live_session_result(session_id: str) -> Dict[str, Any]:
    try:
        result = LIVE_SESSIONS.live_result_payload(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Live session result not found")
    envelope = build_live_result_envelope(
        session_id=str(session_id),
        result_payload=result,
        rooted_path_cb=rooted_path,
    )
    envelope["protocol_version"] = PROTOCOL_VERSION
    return envelope


@router.post("/demo/live/sessions/{session_id}/fixture")
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


@router.get("/demo/live/sessions/{session_id}/quality")
def get_live_session_quality(session_id: str, fixture_id: str | None = None) -> Dict[str, Any]:
    try:
        result = LIVE_SESSIONS.live_result_payload(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Live session result not found")

    finalization_state = str(result.get("finalization_state") or "").strip().lower()
    if finalization_state not in {"ready", "finalized"}:
        raise HTTPException(status_code=409, detail="Transcript result not ready")

    resolved_fixture_id = str(fixture_id or result.get("fixture_id") or "").strip()
    if not resolved_fixture_id:
        raise HTTPException(status_code=409, detail="No fixture metadata for this session")

    final_text = live_result_to_plain_text(result)
    if not final_text.strip():
        raise HTTPException(status_code=409, detail="Transcript text not ready")

    try:
        quality = score_live_text_against_fixture(
            fixture_id=resolved_fixture_id,
            live_text=final_text,
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
    benchmark_envelope = dict(envelope)
    benchmark_envelope["result_metrics"] = build_live_result_metrics_snapshot(result)
    try_autosave_live_benchmark_snapshot(
        session_id=str(session_id),
        artifact_name="final-quality",
        envelope=benchmark_envelope,
        request_meta={
            "fixture_test_mode": str(result.get("fixture_test_mode") or ""),
            "fixture_version": str(result.get("fixture_version") or ""),
            "live_tuning_snapshot": live_benchmark_tuning_snapshot(),
        },
    )
    return envelope


@router.get("/demo/live/sessions/{session_id}/transcript.srt")
def get_live_session_transcript_srt(session_id: str) -> Response:
    try:
        result = LIVE_SESSIONS.live_result_payload(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Live session result not found")
    srt_text = live_result_to_srt_text(result)
    if not srt_text.strip():
        raise HTTPException(status_code=409, detail="Transcript segments not ready")
    headers = {"Content-Disposition": f'attachment; filename="{safe_filename(session_id)}.srt"'}
    return Response(content=srt_text, media_type="application/x-subrip", headers=headers)


@router.get("/demo/live/sessions/{session_id}/recording.wav")
def get_live_session_recording_wav(session_id: str) -> FileResponse:
    try:
        result = LIVE_SESSIONS.live_result_payload(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Live session result not found")
    wav_path = live_recording_wav_path_from_result(result)
    if wav_path is None:
        raise HTTPException(status_code=404, detail="Live recording WAV not found")
    return FileResponse(
        path=str(wav_path),
        media_type="audio/wav",
        filename=f"{safe_filename(session_id)}.wav",
    )


@router.get("/demo/live/sessions/{session_id}/transcript.pc")
def get_live_session_transcript_pc(session_id: str) -> Response:
    try:
        pc_events = LIVE_SESSIONS.live_pc_events(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Live session replay transcript not found")
    pc_text = live_pc_events_to_text(pc_events)
    if not pc_text:
        raise HTTPException(status_code=409, detail="Transcript replay events not ready")
    headers = {"Content-Disposition": f'attachment; filename="{safe_filename(session_id)}.pc"'}
    return Response(content=pc_text, media_type="text/plain", headers=headers)


@router.get("/demo/live/metrics")
def get_live_metrics() -> Dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "live_engine": LIVE_ENGINE,
        "metrics": LIVE_SESSIONS.metrics_payload(),
    }


@router.get("/demo/live/benchmarks")
def get_live_benchmarks(limit: int = 30, mode: str | None = None) -> Dict[str, Any]:
    normalized_mode = str(mode or "").strip().lower()
    if normalized_mode and normalized_mode not in {"inject", "playback"}:
        raise HTTPException(status_code=400, detail="mode must be inject or playback")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_tuning_snapshot": live_benchmark_tuning_snapshot(),
        "rows": list_live_benchmark_exports(
            limit=limit,
            fixture_test_mode=(normalized_mode or None),
        ),
    }


@router.websocket("/demo/live/sessions/{session_id}/ws")
async def live_session_ws(session_id: str, websocket: WebSocket) -> None:
    await run_live_session_ws(
        session_id=session_id,
        websocket=websocket,
        live_sessions=LIVE_SESSIONS,
        rooted_path_cb=rooted_path,
        config=live_engine_rolling_context_config(),
    )
