from __future__ import annotations

import json
import secrets
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()


def _repo_root() -> Path:
    # portal-api/live_sessions.py -> portal-api -> repo root
    return Path(__file__).resolve().parents[1]


@dataclass
class LiveSession:
    session_id: str
    created_monotonic: float
    created_unix: float
    expires_unix: float
    ttl_seconds: int
    state: str = "created"
    ws_connected: bool = False
    closed: bool = False
    close_reason: str = ""
    last_seen_unix: float = 0.0
    seq: int = 0
    bytes_received: int = 0
    frames_received: int = 0
    controls_received: int = 0


@dataclass
class ClosedSessionArchive:
    session_id: str
    closed_unix: float
    expires_unix: float
    close_reason: str
    final_text: str
    final_segments: list[dict[str, Any]]
    transcript_revision: int


class LiveSessionManager:
    def __init__(
        self,
        *,
        default_ttl_seconds: int = 900,
        preconnect_ttl_seconds: int = 30,
        max_sessions: int = 64,
        archive_ttl_seconds: int = 3600,
        max_archives: int = 256,
    ):
        self._default_ttl_seconds = int(max(10, default_ttl_seconds))
        self._preconnect_ttl_seconds = int(max(5, preconnect_ttl_seconds))
        self._preconnect_ttl_seconds = int(min(self._preconnect_ttl_seconds, self._default_ttl_seconds))
        self._max_sessions = int(max(1, max_sessions))
        self._archive_ttl_seconds = int(max(60, archive_ttl_seconds))
        self._max_archives = int(max(1, max_archives))
        self._sessions: dict[str, LiveSession] = {}
        self._archives: dict[str, ClosedSessionArchive] = {}
        self._lock = threading.Lock()
        self._stats_log_dir = (_repo_root() / "data" / "live_stats").resolve()

    def _new_session_id(self) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"live_{ts}_{secrets.token_hex(4)}"

    def _cleanup_expired_locked(self, now_unix: float) -> None:
        dead = [
            sid
            for sid, sess in self._sessions.items()
            if sess.closed or now_unix >= sess.expires_unix
        ]
        for sid in dead:
            self._sessions.pop(sid, None)
        dead_archives = [
            sid
            for sid, arc in self._archives.items()
            if now_unix >= arc.expires_unix
        ]
        for sid in dead_archives:
            self._archives.pop(sid, None)

        if len(self._archives) > self._max_archives:
            ordered = sorted(self._archives.values(), key=lambda a: a.closed_unix)
            overflow = max(0, len(ordered) - self._max_archives)
            for arc in ordered[:overflow]:
                self._archives.pop(arc.session_id, None)

    def create_session(self, *, ttl_seconds: int | None = None) -> dict[str, Any]:
        now_unix = time.time()
        now_mono = time.monotonic()
        ttl = self._default_ttl_seconds if ttl_seconds is None else int(ttl_seconds)
        ttl = int(max(10, ttl))
        preconnect_ttl = int(max(5, min(ttl, self._preconnect_ttl_seconds)))

        with self._lock:
            self._cleanup_expired_locked(now_unix)
            if len(self._sessions) >= self._max_sessions:
                raise RuntimeError("live_session_capacity_reached")

            session_id = self._new_session_id()
            sess = LiveSession(
                session_id=session_id,
                created_monotonic=now_mono,
                created_unix=now_unix,
                expires_unix=(now_unix + preconnect_ttl),
                ttl_seconds=ttl,
                last_seen_unix=now_unix,
            )
            self._sessions[session_id] = sess
            snapshot = self._snapshot_locked(sess)
        try:
            self.append_stats_log(session_id, {"kind": "session_created", "session": snapshot})
        except Exception:
            pass
        return snapshot

    def open_websocket(self, session_id: str) -> dict[str, Any]:
        now_unix = time.time()
        with self._lock:
            self._cleanup_expired_locked(now_unix)
            sess = self._sessions.get(session_id)
            if not sess:
                raise KeyError("session_not_found")
            if sess.ws_connected:
                raise RuntimeError("session_already_connected")
            if sess.closed:
                raise RuntimeError("session_closed")

            sess.ws_connected = True
            sess.state = "connected"
            sess.last_seen_unix = now_unix
            # Expand short preconnect TTL to the requested full session TTL.
            sess.expires_unix = now_unix + int(max(10, sess.ttl_seconds))
            return self._snapshot_locked(sess)

    def mark_state(self, session_id: str, *, state: str) -> dict[str, Any]:
        now_unix = time.time()
        with self._lock:
            sess = self._sessions.get(session_id)
            if not sess:
                raise KeyError("session_not_found")
            sess.state = str(state or "connected")
            sess.last_seen_unix = now_unix
            return self._snapshot_locked(sess)

    def record_audio(self, session_id: str, *, byte_count: int) -> dict[str, Any]:
        now_unix = time.time()
        safe_bytes = max(0, int(byte_count))
        with self._lock:
            sess = self._sessions.get(session_id)
            if not sess:
                raise KeyError("session_not_found")
            sess.last_seen_unix = now_unix
            sess.bytes_received += safe_bytes
            sess.frames_received += 1
            return self._snapshot_locked(sess)

    def record_control(self, session_id: str) -> dict[str, Any]:
        now_unix = time.time()
        with self._lock:
            sess = self._sessions.get(session_id)
            if not sess:
                raise KeyError("session_not_found")
            sess.last_seen_unix = now_unix
            sess.controls_received += 1
            return self._snapshot_locked(sess)

    def next_seq(self, session_id: str) -> int:
        with self._lock:
            sess = self._sessions.get(session_id)
            if not sess:
                raise KeyError("session_not_found")
            sess.seq += 1
            return int(sess.seq)

    def snapshot(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            sess = self._sessions.get(session_id)
            if not sess:
                raise KeyError("session_not_found")
            return self._snapshot_locked(sess)

    def close_session(self, session_id: str, *, reason: str) -> dict[str, Any] | None:
        with self._lock:
            self._cleanup_expired_locked(time.time())
            sess = self._sessions.pop(session_id, None)
            if not sess:
                return None
            sess.closed = True
            sess.ws_connected = False
            sess.close_reason = str(reason or "closed")
            sess.state = "ended"
            sess.last_seen_unix = time.time()
            snapshot = self._snapshot_locked(sess)
        try:
            self.append_stats_log(
                session_id,
                {
                    "kind": "session_closed",
                    "close_reason": str(reason or "closed"),
                    "session": snapshot,
                },
            )
        except Exception:
            pass
        return snapshot

    def archive_transcript(
        self,
        session_id: str,
        *,
        close_reason: str,
        final_text: str,
        final_segments: list[dict[str, Any]],
        transcript_revision: int,
    ) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            self._cleanup_expired_locked(now)
            arc = ClosedSessionArchive(
                session_id=str(session_id),
                closed_unix=now,
                expires_unix=(now + self._archive_ttl_seconds),
                close_reason=str(close_reason or ""),
                final_text=str(final_text or ""),
                final_segments=[dict(seg) for seg in (final_segments or [])],
                transcript_revision=int(max(0, transcript_revision)),
            )
            self._archives[arc.session_id] = arc
            return self._archive_snapshot_locked(arc)

    def archived_transcript(self, session_id: str) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            self._cleanup_expired_locked(now)
            arc = self._archives.get(session_id)
            if not arc:
                raise KeyError("archive_not_found")
            return self._archive_snapshot_locked(arc)

    def _snapshot_locked(self, sess: LiveSession) -> dict[str, Any]:
        now_mono = time.monotonic()
        age_s = max(0.0, float(now_mono - sess.created_monotonic))
        ttl_remaining = max(0.0, float(sess.expires_unix - time.time()))
        return {
            "session_id": sess.session_id,
            "state": sess.state,
            "ws_connected": bool(sess.ws_connected),
            "closed": bool(sess.closed),
            "close_reason": sess.close_reason,
            "created_at_utc": _utc_iso(sess.created_unix),
            "last_seen_utc": _utc_iso(sess.last_seen_unix) if sess.last_seen_unix > 0 else None,
            "expires_at_utc": _utc_iso(sess.expires_unix),
            "ttl_seconds": int(sess.ttl_seconds),
            "age_s": round(age_s, 3),
            "ttl_remaining_s": round(ttl_remaining, 3),
            "seq": int(sess.seq),
            "bytes_received": int(sess.bytes_received),
            "frames_received": int(sess.frames_received),
            "controls_received": int(sess.controls_received),
            "stats_log_path": str(self._stats_log_path(sess.session_id)),
        }

    def _archive_snapshot_locked(self, arc: ClosedSessionArchive) -> dict[str, Any]:
        return {
            "session_id": arc.session_id,
            "close_reason": arc.close_reason,
            "closed_at_utc": _utc_iso(arc.closed_unix),
            "expires_at_utc": _utc_iso(arc.expires_unix),
            "transcript_revision": int(arc.transcript_revision),
            "final_text": str(arc.final_text),
            "final_segments": [dict(seg) for seg in arc.final_segments],
            "final_segments_count": len(arc.final_segments),
        }

    def metrics_snapshot(self) -> dict[str, Any]:
        now_unix = time.time()
        now_mono = time.monotonic()
        with self._lock:
            self._cleanup_expired_locked(now_unix)
            active = list(self._sessions.values())
            archives = list(self._archives.values())

            states: dict[str, int] = {}
            connected = 0
            bytes_received = 0
            frames_received = 0
            controls_received = 0
            max_age_s = 0.0
            for sess in active:
                state = str(sess.state or "unknown")
                states[state] = int(states.get(state, 0) + 1)
                if sess.ws_connected:
                    connected += 1
                bytes_received += int(max(0, sess.bytes_received))
                frames_received += int(max(0, sess.frames_received))
                controls_received += int(max(0, sess.controls_received))
                age_s = max(0.0, float(now_mono - sess.created_monotonic))
                if age_s > max_age_s:
                    max_age_s = age_s

            return {
                "active_sessions": len(active),
                "active_ws_connected": int(connected),
                "active_states": states,
                "active_max_age_s": round(max_age_s, 3),
                "active_bytes_received": int(bytes_received),
                "active_frames_received": int(frames_received),
                "active_controls_received": int(controls_received),
                "archived_sessions": len(archives),
                "limits": {
                    "max_sessions": int(self._max_sessions),
                    "default_ttl_seconds": int(self._default_ttl_seconds),
                    "preconnect_ttl_seconds": int(self._preconnect_ttl_seconds),
                    "archive_ttl_seconds": int(self._archive_ttl_seconds),
                    "max_archives": int(self._max_archives),
                },
            }

    def _stats_log_path(self, session_id: str) -> Path:
        safe_id = str(session_id or "unknown").strip() or "unknown"
        return (self._stats_log_dir / f"{safe_id}.stats.jsonl").resolve()

    def stats_log_path(self, session_id: str) -> str:
        return str(self._stats_log_path(session_id))

    def append_stats_log(self, session_id: str, payload: dict[str, Any]) -> None:
        path = self._stats_log_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        row: dict[str, Any] = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "session_id": str(session_id),
        }
        if isinstance(payload, dict):
            row.update(payload)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")
