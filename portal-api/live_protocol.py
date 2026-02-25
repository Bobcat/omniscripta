from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

PROTOCOL_VERSION = "live_v1"
ALLOWED_CLIENT_TYPES = {"start", "pause", "resume", "stop", "ping"}


def utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_event(
    event_type: str,
    session_id: str,
    *,
    seq: int | None = None,
    **fields: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": str(event_type),
        "protocol_version": PROTOCOL_VERSION,
        "session_id": str(session_id),
        "ts_utc": utc_iso_now(),
    }
    if seq is not None:
        payload["seq"] = int(seq)
    payload.update(fields)
    return payload


def ready_event(
    session_id: str,
    *,
    seq: int | None = None,
    message: str = "Live session is ready.",
    engine: str | None = None,
) -> dict[str, Any]:
    payload = make_event(
        "ready",
        session_id,
        seq=seq,
        message=message,
        audio_format="pcm16le",
        sample_rate_hz=16000,
        channels=1,
    )
    if engine:
        payload["engine"] = str(engine)
    return payload


def control_ack_event(
    session_id: str,
    *,
    control_type: str,
    state: str,
    seq: int | None = None,
) -> dict[str, Any]:
    return make_event(
        "control_ack",
        session_id,
        seq=seq,
        control_type=str(control_type),
        state=str(state),
    )


def stats_event(
    session_id: str,
    *,
    seq: int | None = None,
    bytes_received: int,
    frames_received: int,
    controls_received: int,
    uptime_s: float,
    **extra: Any,
) -> dict[str, Any]:
    payload = make_event(
        "stats",
        session_id,
        seq=seq,
        bytes_received=int(max(0, bytes_received)),
        frames_received=int(max(0, frames_received)),
        controls_received=int(max(0, controls_received)),
        uptime_s=round(max(0.0, float(uptime_s)), 3),
    )
    for key, val in extra.items():
        payload[str(key)] = val
    return payload


def pong_event(session_id: str, *, seq: int | None = None) -> dict[str, Any]:
    return make_event("pong", session_id, seq=seq)


def error_event(
    session_id: str,
    *,
    code: str,
    message: str,
    fatal: bool = False,
    seq: int | None = None,
) -> dict[str, Any]:
    return make_event(
        "error",
        session_id,
        seq=seq,
        code=str(code),
        message=str(message),
        fatal=bool(fatal),
    )


def ended_event(
    session_id: str,
    *,
    reason: str,
    seq: int | None = None,
    **extra: Any,
) -> dict[str, Any]:
    payload = make_event("ended", session_id, seq=seq, reason=str(reason))
    for key, val in extra.items():
        payload[str(key)] = val
    return payload


def parse_client_message(raw: str) -> tuple[str | None, dict[str, Any], str | None]:
    text = str(raw or "")
    try:
        obj = json.loads(text)
    except Exception:
        return None, {}, "invalid_json"

    if not isinstance(obj, dict):
        return None, {}, "invalid_payload_type"

    msg_type = str(obj.get("type") or "").strip().lower()
    if not msg_type:
        return None, obj, "missing_type"
    if msg_type not in ALLOWED_CLIENT_TYPES:
        return None, obj, "unsupported_type"
    return msg_type, obj, None
