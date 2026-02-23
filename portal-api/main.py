from __future__ import annotations

import asyncio
import json
import os
import shutil
import mimetypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse, parse_qs

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request, WebSocket, WebSocketDisconnect, status
from fastapi.responses import FileResponse, Response
from live_protocol import (
    PROTOCOL_VERSION,
    control_ack_event,
    final_event,
    ended_event,
    error_event,
    partial_event,
    parse_client_message,
    pong_event,
    ready_event,
    stats_event,
)
from live_sessions import LiveSessionManager
from live_transcriber import LiveTranscriber
from queue_fs import init_job_in_inbox, JobPaths, BASE as BASE_JOBS

ROOT_PATH = os.getenv("TRANSCRIBE_ROOT_PATH", "/api")
app = FastAPI(root_path=ROOT_PATH)


DEFAULT_CALIBRATION_SNIPPET_SECONDS = [60, 180, 300, 480, 600, 1200, 1800, 2700, 3600]
LIVE_SESSION_TTL_S = int(os.getenv("TRANSCRIBE_LIVE_SESSION_TTL_S", "900"))
LIVE_SESSION_PRECONNECT_TTL_S = int(os.getenv("TRANSCRIBE_LIVE_PRECONNECT_TTL_S", "30"))
LIVE_MAX_SESSIONS = int(os.getenv("TRANSCRIBE_LIVE_MAX_SESSIONS", "1"))
LIVE_ARCHIVE_TTL_S = int(os.getenv("TRANSCRIBE_LIVE_ARCHIVE_TTL_S", "3600"))
LIVE_MAX_ARCHIVES = int(os.getenv("TRANSCRIBE_LIVE_MAX_ARCHIVES", "256"))
LIVE_AUDIO_SAMPLE_RATE_HZ = int(os.getenv("TRANSCRIBE_LIVE_SAMPLE_RATE_HZ", "16000"))
LIVE_AUDIO_CHANNELS = int(os.getenv("TRANSCRIBE_LIVE_CHANNELS", "1"))
LIVE_SESSIONS = LiveSessionManager(
    default_ttl_seconds=LIVE_SESSION_TTL_S,
    preconnect_ttl_seconds=LIVE_SESSION_PRECONNECT_TTL_S,
    max_sessions=LIVE_MAX_SESSIONS,
    archive_ttl_seconds=LIVE_ARCHIVE_TTL_S,
    max_archives=LIVE_MAX_ARCHIVES,
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
    mock_toggle_raw = _param_from_request_or_referer(request, "live_recording_mock")
    mock_toggle = _optional_bool_param(mock_toggle_raw)
    if mock_toggle is not None:
        val = "1" if mock_toggle else "0"
        ws_path = f"{ws_path}?live_recording_mock={val}"
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
    transcriber: LiveTranscriber | None = None
    mock_toggle = _optional_bool_param(websocket.query_params.get("live_recording_mock"))
    driver_override: str | None = None
    if mock_toggle is True:
        driver_override = "mock_stream"
    elif mock_toggle is False:
        driver_override = "whisperlive_sidecar"

    async def send_event(payload: Dict[str, Any]) -> None:
        payload = dict(payload)
        try:
            payload["seq"] = LIVE_SESSIONS.next_seq(session_id)
        except KeyError:
            pass
        await websocket.send_json(payload)

    async def emit_engine_events(events: list[Any]) -> None:
        for ev in events:
            kind = str(getattr(ev, "kind", "")).strip().lower()
            txt = str(getattr(ev, "text", "") or "").strip()
            if not txt:
                continue
            t0_ms = int(max(0, int(getattr(ev, "t0_ms", 0))))
            t1_ms = int(max(t0_ms, int(getattr(ev, "t1_ms", t0_ms))))
            segment_id = str(getattr(ev, "segment_id", "") or "")
            revision = int(max(0, int(getattr(ev, "revision", 0))))
            committed_chars = int(max(0, int(getattr(ev, "committed_chars", 0))))
            committed_segments = int(max(0, int(getattr(ev, "committed_segments", 0))))
            committed_until_ms = int(max(0, int(getattr(ev, "committed_until_ms", 0))))
            if kind == "partial":
                await send_event(
                    partial_event(
                        session_id,
                        text=txt,
                        t0_ms=t0_ms,
                        t1_ms=t1_ms,
                        segment_id=segment_id or "tail",
                        revision=revision,
                        committed_chars=committed_chars,
                        committed_segments=committed_segments,
                        committed_until_ms=committed_until_ms,
                    )
                )
            elif kind == "final":
                await send_event(
                    final_event(
                        session_id,
                        text=txt,
                        t0_ms=t0_ms,
                        t1_ms=t1_ms,
                        segment_id=segment_id,
                        revision=revision,
                        committed_chars=committed_chars,
                        committed_segments=committed_segments,
                        committed_until_ms=committed_until_ms,
                    )
                )

    try:
        try:
            transcriber = await asyncio.to_thread(
                LiveTranscriber,
                session_id=session_id,
                sample_rate_hz=LIVE_AUDIO_SAMPLE_RATE_HZ,
                channels=LIVE_AUDIO_CHANNELS,
                driver_override=driver_override,
            )
        except Exception as e:
            stop_reason = "engine_init_failed"
            await send_event(
                error_event(
                    session_id,
                    code="engine_init_failed",
                    message=f"{type(e).__name__}: {e}",
                    fatal=True,
                )
            )
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
            websocket_closed = True
            return

        await send_event(
            ready_event(
                session_id,
                message="Live websocket connected. Send binary PCM16 frames and JSON controls.",
                engine=transcriber.driver_name,
            )
        )

        while True:
            incoming = await websocket.receive()

            if incoming.get("type") == "websocket.disconnect":
                stop_reason = "client_disconnected"
                break

            raw_bytes = incoming.get("bytes")
            if raw_bytes is not None:
                snapshot = LIVE_SESSIONS.record_audio(session_id, byte_count=len(raw_bytes))
                engine_events: list[Any] = []
                try:
                    if transcriber is not None:
                        engine_events = await asyncio.to_thread(transcriber.feed_audio, raw_bytes)
                except Exception as e:
                    stop_reason = "engine_runtime_error"
                    await send_event(
                        error_event(
                            session_id,
                            code="engine_runtime_error",
                            message=f"{type(e).__name__}: {e}",
                            fatal=True,
                        )
                    )
                    if not websocket_closed:
                        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
                        websocket_closed = True
                    break

                await emit_engine_events(engine_events)

                # Emit transport + decode stats at startup, periodically and after transcript updates.
                should_emit_stats = (
                    snapshot["frames_received"] == 1
                    or (snapshot["frames_received"] % 50) == 0
                    or bool(engine_events)
                )
                if should_emit_stats:
                    tstats = transcriber.stats_snapshot() if transcriber is not None else {}
                    await send_event(
                        stats_event(
                            session_id,
                            bytes_received=snapshot["bytes_received"],
                            frames_received=snapshot["frames_received"],
                            controls_received=snapshot["controls_received"],
                            uptime_s=snapshot["age_s"],
                            engine=tstats.get("engine_driver"),
                            decode_calls=tstats.get("decode_calls"),
                            decode_ms_total=tstats.get("decode_ms_total"),
                            decode_ms_last=tstats.get("decode_ms_last"),
                            rtf=tstats.get("rtf"),
                            partials_emitted=tstats.get("partials_emitted"),
                            finals_emitted=tstats.get("finals_emitted"),
                            engine_ready=tstats.get("engine_ready"),
                            engine_rx_messages=tstats.get("engine_rx_messages"),
                            engine_rx_parse_errors=tstats.get("engine_rx_parse_errors"),
                            engine_rx_unknown_messages=tstats.get("engine_rx_unknown_messages"),
                            engine_tx_sidecar_audio_bytes=tstats.get("engine_tx_sidecar_audio_bytes"),
                            revision=tstats.get("revision"),
                            committed_until_ms=tstats.get("committed_until_ms"),
                            committed_segments=tstats.get("committed_segments"),
                            committed_chars=tstats.get("committed_chars"),
                            has_partial_tail=tstats.get("has_partial_tail"),
                        )
                    )
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
                transcript_snapshot: dict[str, Any] = {}
                if transcriber is not None:
                    try:
                        await emit_engine_events(await asyncio.to_thread(transcriber.flush))
                    except Exception as e:
                        await send_event(
                            error_event(
                                session_id,
                                code="engine_flush_error",
                                message=f"{type(e).__name__}: {e}",
                                fatal=False,
                            )
                        )
                    transcript_snapshot = await asyncio.to_thread(transcriber.transcript_snapshot)
                if transcript_snapshot:
                    LIVE_SESSIONS.archive_transcript(
                        session_id,
                        close_reason=stop_reason,
                        final_text=str(transcript_snapshot.get("final_text") or ""),
                        final_segments=[
                            dict(seg)
                            for seg in (transcript_snapshot.get("final_segments") or [])
                            if isinstance(seg, dict)
                        ],
                        transcript_revision=int(max(0, int(transcript_snapshot.get("revision") or 0))),
                    )
                await send_event(
                    ended_event(
                        session_id,
                        reason=stop_reason,
                        transcript_revision=int(max(0, int(transcript_snapshot.get("revision") or 0))),
                        final_segments_count=len(transcript_snapshot.get("final_segments") or []),
                        final_text=str(transcript_snapshot.get("final_text") or ""),
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
        # Release capacity slot first; downstream close operations may block.
        LIVE_SESSIONS.close_session(session_id, reason=stop_reason)
        if transcriber is not None:
            try:
                if stop_reason != "client_stop":
                    snapshot = await asyncio.to_thread(transcriber.transcript_snapshot)
                    if snapshot and (
                        str(snapshot.get("final_text") or "").strip()
                        or bool(snapshot.get("final_segments"))
                    ):
                        LIVE_SESSIONS.archive_transcript(
                            session_id,
                            close_reason=stop_reason,
                            final_text=str(snapshot.get("final_text") or ""),
                            final_segments=[
                                dict(seg)
                                for seg in (snapshot.get("final_segments") or [])
                                if isinstance(seg, dict)
                            ],
                            transcript_revision=int(max(0, int(snapshot.get("revision") or 0))),
                        )
            except Exception:
                pass
        if transcriber is not None:
            try:
                await asyncio.to_thread(transcriber.close)
            except Exception:
                pass


def _repo_root() -> Path:
    # portal-api/main.py -> portal-api -> repo root
    return Path(__file__).resolve().parents[1]


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


def _find_job_dir(job_id: str) -> Path | None:
    for state in ("inbox", "running", "done", "error"):
        d = BASE_JOBS / state / job_id
        if d.exists():
            return d
    return None


def _as_bool(raw: Any) -> bool:
    s = str(raw or "").strip().lower()
    return s in {"1", "true", "yes", "on"}


def _optional_bool_param(raw: Any) -> bool | None:
    s = str(raw or "").strip().lower()
    if not s:
        return None
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off"}:
        return False
    return None


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
