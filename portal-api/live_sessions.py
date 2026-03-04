from __future__ import annotations

import json
import secrets
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _append_preview_text(existing_text: str, incoming_text: str) -> str:
    existing = str(existing_text or "").strip()
    incoming = str(incoming_text or "").strip()
    if not incoming:
        return existing
    if not existing:
        return incoming
    if incoming == existing:
        return existing
    return f"{existing} {incoming}"


def _utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()


def _live_commit_rows_debug_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_rows = 0
    invalid_index_rows = 0
    by_index: dict[int, dict[str, Any]] = {}
    reason_counts: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        total_rows += 1
        try:
            idx = int(row.get("chunk_index"))
        except Exception:
            invalid_index_rows += 1
            continue
        # Snapshot contract is one row per chunk_index. If duplicates exist, keep the last one.
        by_index[idx] = row
    for row in by_index.values():
        reason = str(row.get("reason") or "").strip()
        if not reason:
            continue
        reason_counts[reason] = int(reason_counts.get(reason, 0) + 1)
    unique_rows = len(by_index)
    duplicate_rows = max(0, total_rows - invalid_index_rows - unique_rows)
    return {
        "chunk_reason_counts": dict(sorted(reason_counts.items(), key=lambda kv: kv[0])),
        "chunk_results_rows_count": int(max(0, total_rows)),
        "chunk_results_unique_count": int(max(0, unique_rows)),
        "chunk_results_duplicate_index_rows": int(max(0, duplicate_rows)),
        "chunk_results_invalid_index_rows": int(max(0, invalid_index_rows)),
    }


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
    live_engine: str = "rolling_context"
    state: str = "created"
    ws_connected: bool = False
    closed: bool = False
    close_reason: str = ""
    last_seen_unix: float = 0.0
    seq: int = 0
    bytes_received: int = 0
    frames_received: int = 0
    controls_received: int = 0
    recording_state: str = "idle"
    recording_path: str = ""
    recording_bytes: int = 0
    recording_duration_ms: int = 0
    chunk_index_next: int = 0
    chunks_total: int = 0
    chunks_done: int = 0
    chunks_failed: int = 0
    finalization_state: str = "idle"
    batch_job_id: str = ""
    live_transcript_revision: int = 0
    live_final_text: str = ""
    live_final_segments: list[dict[str, Any]] = field(default_factory=list)
    live_commit_results: list[dict[str, Any]] = field(default_factory=list)
    live_preview_text: str = ""
    live_preview_seq: int = -1
    live_preview_audio_end_ms: int = 0
    live_preview_updated_unix: float = 0.0
    live_engine_runtime: dict[str, Any] = field(default_factory=dict)
    fixture_id: str = ""
    fixture_version: str = ""
    fixture_test_mode: str = ""


@dataclass
class ClosedSessionArchive:
    session_id: str
    closed_unix: float
    expires_unix: float
    close_reason: str
    final_text: str
    final_segments: list[dict[str, Any]]
    transcript_revision: int
    live_engine: str = "rolling_context"
    recording_path: str = ""
    recording_bytes: int = 0
    recording_duration_ms: int = 0
    chunks_total: int = 0
    chunks_done: int = 0
    chunks_failed: int = 0
    finalization_state: str = ""
    batch_job_id: str = ""
    live_transcript_revision: int = 0
    live_final_text: str = ""
    live_final_segments: list[dict[str, Any]] = field(default_factory=list)
    live_commit_results: list[dict[str, Any]] = field(default_factory=list)
    live_engine_runtime: dict[str, Any] = field(default_factory=dict)
    fixture_id: str = ""
    fixture_version: str = ""
    fixture_test_mode: str = ""


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

    def get_max_sessions(self) -> int:
        with self._lock:
            return int(max(1, int(self._max_sessions)))

    def set_max_sessions(self, max_sessions: int) -> int:
        safe = int(max(1, int(max_sessions)))
        with self._lock:
            prev = int(max(1, int(self._max_sessions)))
            self._max_sessions = safe
            return prev

    def create_session(
        self,
        *,
        ttl_seconds: int | None = None,
        live_engine: str | None = None,
    ) -> dict[str, Any]:
        now_unix = time.time()
        now_mono = time.monotonic()
        ttl = self._default_ttl_seconds if ttl_seconds is None else int(ttl_seconds)
        ttl = int(max(10, ttl))
        preconnect_ttl = int(max(5, min(ttl, self._preconnect_ttl_seconds)))
        engine_name = str(live_engine or "").strip().lower() or "rolling_context"

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
                live_engine=engine_name,
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

    def update_live_state(
        self,
        session_id: str,
        *,
        recording_state: str | None = None,
        recording_path: str | Path | None = None,
        recording_bytes: int | None = None,
        recording_duration_ms: int | None = None,
        chunk_index_next: int | None = None,
        chunks_total: int | None = None,
        chunks_done: int | None = None,
        chunks_failed: int | None = None,
        finalization_state: str | None = None,
        batch_job_id: str | None = None,
    ) -> dict[str, Any]:
        now_unix = time.time()
        with self._lock:
            sess = self._sessions.get(session_id)
            if not sess:
                raise KeyError("session_not_found")
            sess.last_seen_unix = now_unix
            if recording_state is not None:
                sess.recording_state = str(recording_state or "idle")
            if recording_path is not None:
                sess.recording_path = str(recording_path)
            if recording_bytes is not None:
                sess.recording_bytes = int(max(0, recording_bytes))
            if recording_duration_ms is not None:
                sess.recording_duration_ms = int(max(0, recording_duration_ms))
            if chunk_index_next is not None:
                sess.chunk_index_next = int(max(0, chunk_index_next))
            if chunks_total is not None:
                sess.chunks_total = int(max(0, chunks_total))
            if chunks_done is not None:
                sess.chunks_done = int(max(0, chunks_done))
            if chunks_failed is not None:
                sess.chunks_failed = int(max(0, chunks_failed))
            if finalization_state is not None:
                sess.finalization_state = str(finalization_state or "idle")
            if batch_job_id is not None:
                sess.batch_job_id = str(batch_job_id)
        return self._snapshot_locked(sess)

    def set_fixture_metadata(
        self,
        session_id: str,
        *,
        fixture_id: str | None = None,
        fixture_version: str | None = None,
        fixture_test_mode: str | None = None,
    ) -> dict[str, Any]:
        now_unix = time.time()
        with self._lock:
            sess = self._sessions.get(session_id)
            if not sess:
                raise KeyError("session_not_found")
            sess.last_seen_unix = now_unix
            if fixture_id is not None:
                sess.fixture_id = str(fixture_id or "").strip()
            if fixture_version is not None:
                sess.fixture_version = str(fixture_version or "").strip()
            if fixture_test_mode is not None:
                sess.fixture_test_mode = str(fixture_test_mode or "").strip()
            return self._snapshot_locked(sess)

    def update_live_preview(
        self,
        session_id: str,
        *,
        text: str,
        preview_seq: int,
        audio_end_ms: int,
        append_to_existing: bool = True,
    ) -> dict[str, Any]:
        now_unix = time.time()
        incoming_raw_text = str(text or "")
        safe_seq = int(max(0, int(preview_seq)))
        safe_audio_end_ms = int(max(0, int(audio_end_ms)))
        with self._lock:
            sess = self._sessions.get(session_id)
            if not sess:
                raise KeyError("session_not_found")
            sess.last_seen_unix = now_unix
            current_seq = int(getattr(sess, "live_preview_seq", -1) or -1)
            if safe_seq <= current_seq:
                return self._snapshot_locked(sess)

            if append_to_existing:
                preview_text = _append_preview_text(
                    existing_text=str(sess.live_preview_text or ""),
                    incoming_text=incoming_raw_text,
                )
            else:
                preview_text = str(incoming_raw_text or "").strip()

            sess.live_preview_text = str(preview_text or "")
            sess.live_preview_seq = int(safe_seq)
            sess.live_preview_audio_end_ms = int(safe_audio_end_ms)
            sess.live_preview_updated_unix = now_unix
            return self._snapshot_locked(sess)

    def clear_live_preview(
        self,
        session_id: str,
        *,
        max_seq: int | None = None,
    ) -> dict[str, Any]:
        now_unix = time.time()
        with self._lock:
            sess = self._sessions.get(session_id)
            if not sess:
                raise KeyError("session_not_found")
            sess.last_seen_unix = now_unix
            current_seq = int(getattr(sess, "live_preview_seq", -1) or -1)
            if max_seq is None or current_seq <= int(max_seq):
                sess.live_preview_text = ""
                sess.live_preview_seq = -1
                sess.live_preview_audio_end_ms = 0
                sess.live_preview_updated_unix = 0.0
            return self._snapshot_locked(sess)

    def set_live_engine_runtime(
        self,
        session_id: str,
        *,
        runtime: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now_unix = time.time()
        payload = dict(runtime or {})
        with self._lock:
            sess = self._sessions.get(session_id)
            if not sess:
                raise KeyError("session_not_found")
            sess.last_seen_unix = now_unix
            sess.live_engine_runtime = payload
            return self._snapshot_locked(sess)

    def record_live_commit(
        self,
        session_id: str,
        *,
        chunk_index: int,
        t0_ms: int,
        t1_ms: int,
        text: str,
        segments: list[dict[str, Any]] | None = None,
        state: str = "ready",
        error: str = "",
        reason: str = "",
        speech_frames: int | None = None,
        silence_frames_tail: int | None = None,
        chunk_duration_ms: int | None = None,
        asr_pipeline_time_s: float | None = None,
        asr_transcribe_time_s: float | None = None,
    ) -> dict[str, Any]:
        now_unix = time.time()
        idx = int(max(0, chunk_index))
        safe_t0 = int(max(0, t0_ms))
        safe_t1 = int(max(safe_t0, t1_ms))
        safe_text = str(text or "")
        safe_state = str(state or "ready")
        safe_error = str(error or "")
        safe_reason = str(reason or "").strip()
        segs = [dict(seg) for seg in (segments or []) if isinstance(seg, dict)]
        safe_speech_frames = None if speech_frames is None else int(max(0, int(speech_frames)))
        safe_silence_frames_tail = None if silence_frames_tail is None else int(max(0, int(silence_frames_tail)))
        safe_chunk_duration_ms = None if chunk_duration_ms is None else int(max(0, int(chunk_duration_ms)))
        safe_asr_pipeline_time_s = None if asr_pipeline_time_s is None else max(0.0, float(asr_pipeline_time_s))
        safe_asr_transcribe_time_s = (
            None if asr_transcribe_time_s is None else max(0.0, float(asr_transcribe_time_s))
        )
        with self._lock:
            sess = self._sessions.get(session_id)
            if not sess:
                raise KeyError("session_not_found")
            sess.last_seen_unix = now_unix

            row = {
                "chunk_index": idx,
                "t0_ms": safe_t0,
                "t1_ms": safe_t1,
                "text": safe_text,
                "state": safe_state,
                "error": safe_error,
                "segments": segs,
                "reason": safe_reason,
                "speech_frames": safe_speech_frames,
                "silence_frames_tail": safe_silence_frames_tail,
                "chunk_duration_ms": safe_chunk_duration_ms,
                "asr_pipeline_time_s": safe_asr_pipeline_time_s,
                "asr_transcribe_time_s": safe_asr_transcribe_time_s,
            }
            replaced = False
            for i, existing in enumerate(sess.live_commit_results):
                try:
                    existing_idx = int(existing.get("chunk_index"))
                except Exception:
                    existing_idx = -1
                if existing_idx == idx:
                    if not row["reason"]:
                        row["reason"] = str(existing.get("reason") or "")
                    if row["speech_frames"] is None and existing.get("speech_frames") is not None:
                        row["speech_frames"] = int(max(0, int(existing.get("speech_frames") or 0)))
                    if row["silence_frames_tail"] is None and existing.get("silence_frames_tail") is not None:
                        row["silence_frames_tail"] = int(max(0, int(existing.get("silence_frames_tail") or 0)))
                    if row["chunk_duration_ms"] is None and existing.get("chunk_duration_ms") is not None:
                        row["chunk_duration_ms"] = int(max(0, int(existing.get("chunk_duration_ms") or 0)))
                    if row["asr_pipeline_time_s"] is None and existing.get("asr_pipeline_time_s") is not None:
                        try:
                            row["asr_pipeline_time_s"] = max(0.0, float(existing.get("asr_pipeline_time_s")))
                        except Exception:
                            row["asr_pipeline_time_s"] = None
                    if row["asr_transcribe_time_s"] is None and existing.get("asr_transcribe_time_s") is not None:
                        try:
                            row["asr_transcribe_time_s"] = max(0.0, float(existing.get("asr_transcribe_time_s")))
                        except Exception:
                            row["asr_transcribe_time_s"] = None
                    sess.live_commit_results[i] = row
                    replaced = True
                    break
            if not replaced:
                sess.live_commit_results.append(row)
                sess.live_commit_results.sort(key=lambda r: int(r.get("chunk_index") or 0))

            sess.chunks_total = max(int(sess.chunks_total), idx + 1)
            sess.chunk_index_next = max(int(sess.chunk_index_next), idx + 1)
            if safe_state == "ready":
                sess.chunks_done = max(
                    0,
                    sum(1 for r in sess.live_commit_results if str(r.get("state") or "") == "ready"),
                )
                sess.chunks_failed = max(
                    0,
                    sum(1 for r in sess.live_commit_results if str(r.get("state") or "") == "error"),
                )
                # A new final chunk supersedes any preview suffix that was shown before it.
                sess.live_preview_text = ""
                sess.live_preview_seq = -1
                sess.live_preview_audio_end_ms = 0
                sess.live_preview_updated_unix = 0.0
            elif safe_state == "error":
                sess.chunks_failed = max(
                    0,
                    sum(1 for r in sess.live_commit_results if str(r.get("state") or "") == "error"),
                )
                sess.chunks_done = max(
                    0,
                    sum(1 for r in sess.live_commit_results if str(r.get("state") or "") == "ready"),
                )

            appended_final_text = ""
            for r in sess.live_commit_results:
                if str(r.get("state") or "") != "ready":
                    continue
                try:
                    row_idx = int(r.get("chunk_index"))
                except Exception:
                    row_idx = -1
                raw_row_text = str(r.get("text") or "")
                row_text = raw_row_text.strip()
                if not row_text:
                    continue

                append_text = str(row_text or "").strip()
                if not append_text:
                    continue
                appended_final_text = append_text if not appended_final_text else f"{appended_final_text}\n{append_text}"

            sess.live_final_text = appended_final_text

            appended_segments: list[dict[str, Any]] = []
            if any(r.get("segments") for r in sess.live_commit_results):
                seg_counter = 0
                for r in sess.live_commit_results:
                    if str(r.get("state") or "") != "ready":
                        continue
                    row_t0 = int(max(0, int(r.get("t0_ms") or 0)))
                    row_t1 = int(max(row_t0, int(r.get("t1_ms") or row_t0)))
                    row_segments = r.get("segments")
                    if isinstance(row_segments, list) and row_segments:
                        for seg in row_segments:
                            if not isinstance(seg, dict):
                                continue
                            seg_text = str(seg.get("text") or "").strip()
                            if not seg_text:
                                continue
                            seg_t0 = int(max(0, int(seg.get("t0_ms") or row_t0)))
                            seg_t1 = int(max(seg_t0, int(seg.get("t1_ms") or row_t1)))
                            seg_counter += 1
                            appended_segments.append(
                                {
                                    "segment_id": str(seg.get("segment_id") or f"c{idx:04d}s{seg_counter:04d}"),
                                    "text": seg_text,
                                    "t0_ms": seg_t0,
                                    "t1_ms": seg_t1,
                                }
                            )
                    else:
                        row_text = str(r.get("text") or "").strip()
                        if row_text:
                            seg_counter += 1
                            appended_segments.append(
                                {
                                    "segment_id": f"c{idx:04d}",
                                    "text": row_text,
                                    "t0_ms": row_t0,
                                    "t1_ms": row_t1,
                                }
                            )
            else:
                for r in sess.live_commit_results:
                    if str(r.get("state") or "") != "ready":
                        continue
                    row_text = str(r.get("text") or "").strip()
                    if not row_text:
                        continue
                    idx2 = int(max(0, int(r.get("chunk_index") or 0)))
                    row_t0 = int(max(0, int(r.get("t0_ms") or 0)))
                    row_t1 = int(max(row_t0, int(r.get("t1_ms") or row_t0)))
                    appended_segments.append(
                        {
                            "segment_id": f"c{idx2:04d}",
                            "text": row_text,
                            "t0_ms": row_t0,
                            "t1_ms": row_t1,
                        }
                    )

            sess.live_final_segments = appended_segments
            sess.live_transcript_revision += 1
            return self._live_result_snapshot_locked(sess)

    def live_result_snapshot(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            sess = self._sessions.get(session_id)
            if sess is not None:
                return self._live_result_snapshot_locked(sess)
            arc = self._archives.get(session_id)
            if arc is not None:
                return self._live_archive_result_snapshot_locked(arc)
        raise KeyError("session_or_archive_not_found")

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
        recording_path: str = "",
        recording_bytes: int = 0,
        recording_duration_ms: int = 0,
        chunks_total: int = 0,
        chunks_done: int = 0,
        chunks_failed: int = 0,
        finalization_state: str = "",
        batch_job_id: str = "",
        live_engine: str | None = None,
    ) -> dict[str, Any]:
        now = time.time()
        requested_engine = str(live_engine or "").strip().lower()
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
                live_engine=(requested_engine or "rolling_context"),
                recording_path=str(recording_path or ""),
                recording_bytes=int(max(0, recording_bytes)),
                recording_duration_ms=int(max(0, recording_duration_ms)),
                chunks_total=int(max(0, chunks_total)),
                chunks_done=int(max(0, chunks_done)),
                chunks_failed=int(max(0, chunks_failed)),
                finalization_state=str(finalization_state or ""),
                batch_job_id=str(batch_job_id or ""),
                live_transcript_revision=0,
                live_final_text="",
                live_final_segments=[],
                live_commit_results=[],
                fixture_id="",
                fixture_version="",
                fixture_test_mode="",
            )
            src_sess = self._sessions.get(str(session_id))
            if src_sess is not None:
                arc.live_transcript_revision = int(max(0, src_sess.live_transcript_revision))
                arc.live_final_text = str(src_sess.live_final_text or "")
                arc.live_final_segments = [dict(seg) for seg in src_sess.live_final_segments]
                arc.live_commit_results = [dict(r) for r in src_sess.live_commit_results]
                arc.live_engine_runtime = dict(src_sess.live_engine_runtime or {})
                arc.fixture_id = str(src_sess.fixture_id or "")
                arc.fixture_version = str(src_sess.fixture_version or "")
                arc.fixture_test_mode = str(src_sess.fixture_test_mode or "")
                if not requested_engine:
                    arc.live_engine = str(src_sess.live_engine or "rolling_context")
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

    @staticmethod
    def _preview_source_for_engine(live_engine: str) -> str:
        _ = live_engine
        return "uncommitted_preview"

    def _snapshot_locked(self, sess: LiveSession) -> dict[str, Any]:
        now_mono = time.monotonic()
        age_s = max(0.0, float(now_mono - sess.created_monotonic))
        ttl_remaining = max(0.0, float(sess.expires_unix - time.time()))
        return {
            "session_id": sess.session_id,
            "live_engine": str(sess.live_engine or "rolling_context"),
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
            "recording_state": str(sess.recording_state or "idle"),
            "recording_path": str(sess.recording_path or ""),
            "recording_bytes": int(max(0, sess.recording_bytes)),
            "recording_duration_ms": int(max(0, sess.recording_duration_ms)),
            "chunk_index_next": int(max(0, sess.chunk_index_next)),
            "chunks_total": int(max(0, sess.chunks_total)),
            "chunks_done": int(max(0, sess.chunks_done)),
            "chunks_failed": int(max(0, sess.chunks_failed)),
            "chunks_pending": int(max(0, sess.chunks_total - sess.chunks_done - sess.chunks_failed)),
            "finalization_state": str(sess.finalization_state or "idle"),
            "batch_job_id": str(sess.batch_job_id or ""),
            "live_transcript_revision": int(max(0, sess.live_transcript_revision)),
            "live_final_text_chars": len(str(sess.live_final_text or "")),
            "live_final_segments_count": len(sess.live_final_segments),
            "live_commit_results_count": len(sess.live_commit_results),
            "fixture_id": str(sess.fixture_id or ""),
            "fixture_version": str(sess.fixture_version or ""),
            "fixture_test_mode": str(sess.fixture_test_mode or ""),
            "stats_log_path": str(self._stats_log_path(sess.session_id)),
        }

    def _archive_snapshot_locked(self, arc: ClosedSessionArchive) -> dict[str, Any]:
        return {
            "session_id": arc.session_id,
            "live_engine": str(arc.live_engine or "rolling_context"),
            "close_reason": arc.close_reason,
            "closed_at_utc": _utc_iso(arc.closed_unix),
            "expires_at_utc": _utc_iso(arc.expires_unix),
            "transcript_revision": int(arc.transcript_revision),
            "final_text": str(arc.final_text),
            "final_segments": [dict(seg) for seg in arc.final_segments],
            "final_segments_count": len(arc.final_segments),
            "recording_path": str(arc.recording_path or ""),
            "recording_bytes": int(max(0, arc.recording_bytes)),
            "recording_duration_ms": int(max(0, arc.recording_duration_ms)),
            "chunks_total": int(max(0, arc.chunks_total)),
            "chunks_done": int(max(0, arc.chunks_done)),
            "chunks_failed": int(max(0, arc.chunks_failed)),
            "finalization_state": str(arc.finalization_state or ""),
            "batch_job_id": str(arc.batch_job_id or ""),
            "live_transcript_revision": int(max(0, arc.live_transcript_revision)),
            "live_final_text_chars": len(str(arc.live_final_text or "")),
            "live_final_segments_count": len(arc.live_final_segments),
            "live_commit_results_count": len(arc.live_commit_results),
            "fixture_id": str(arc.fixture_id or ""),
            "fixture_version": str(arc.fixture_version or ""),
            "fixture_test_mode": str(arc.fixture_test_mode or ""),
        }

    def _live_result_snapshot_locked(self, sess: LiveSession) -> dict[str, Any]:
        chunks = [dict(r) for r in sess.live_commit_results]
        chunk_debug = _live_commit_rows_debug_metrics(chunks)
        live_engine = str(sess.live_engine or "rolling_context")
        preview_source = self._preview_source_for_engine(live_engine)
        preview_text = str(sess.live_preview_text or "").strip()
        final_covered_ms = 0
        for seg in sess.live_final_segments:
            if not isinstance(seg, dict):
                continue
            try:
                t1 = int(seg.get("t1_ms") or 0)
            except Exception:
                t1 = 0
            if t1 > final_covered_ms:
                final_covered_ms = t1
        if final_covered_ms <= 0:
            for r in chunks:
                if str(r.get("state") or "") != "ready":
                    continue
                try:
                    t1 = int(r.get("t1_ms") or 0)
                except Exception:
                    t1 = 0
                if t1 > final_covered_ms:
                    final_covered_ms = t1
        engine_runtime = {
            "mode": "single_lane",
            "preview_source": str(preview_source),
            "uncommitted_audio_ms": int(max(0, int(sess.recording_duration_ms) - int(final_covered_ms))),
        }
        extra_engine_runtime = dict(sess.live_engine_runtime or {})
        if extra_engine_runtime:
            engine_runtime["engine_state"] = extra_engine_runtime
        return {
            "session_id": str(sess.session_id),
            "source": "active",
            "live_engine": live_engine,
            "state": str(sess.state or ""),
            "recording_state": str(sess.recording_state or ""),
            "finalization_state": str(sess.finalization_state or ""),
            "batch_job_id": str(sess.batch_job_id or ""),
            "recording_path": str(sess.recording_path or ""),
            "recording_bytes": int(max(0, sess.recording_bytes)),
            "recording_duration_ms": int(max(0, sess.recording_duration_ms)),
            "chunks_total": int(max(0, sess.chunks_total)),
            "chunks_done": int(max(0, sess.chunks_done)),
            "chunks_failed": int(max(0, sess.chunks_failed)),
            "chunks_pending": int(max(0, sess.chunks_total - sess.chunks_done - sess.chunks_failed)),
            "transcript_revision": int(max(0, sess.live_transcript_revision)),
            "final_text": str(sess.live_final_text or ""),
            "final_segments": [dict(seg) for seg in sess.live_final_segments],
            "final_segments_count": len(sess.live_final_segments),
            "final_covered_ms": int(max(0, final_covered_ms)),
            "chunk_results": chunks,
            "chunk_results_count": len(chunks),
            "chunk_reason_counts": dict(chunk_debug.get("chunk_reason_counts") or {}),
            "chunk_results_rows_count": int(max(0, int(chunk_debug.get("chunk_results_rows_count") or 0))),
            "chunk_results_unique_count": int(max(0, int(chunk_debug.get("chunk_results_unique_count") or 0))),
            "chunk_results_duplicate_index_rows": int(
                max(0, int(chunk_debug.get("chunk_results_duplicate_index_rows") or 0))
            ),
            "chunk_results_invalid_index_rows": int(max(0, int(chunk_debug.get("chunk_results_invalid_index_rows") or 0))),
            "preview": {
                "text": str(preview_text or ""),
                "source": str(preview_source),
                "preview_seq": int(max(-1, int(sess.live_preview_seq))),
                "audio_end_ms": int(max(0, int(sess.live_preview_audio_end_ms or 0))),
                "updated_at_utc": (
                    _utc_iso(sess.live_preview_updated_unix)
                    if float(sess.live_preview_updated_unix or 0.0) > 0
                    else ""
                ),
            },
            "engine_runtime": engine_runtime,
            "fixture_id": str(sess.fixture_id or ""),
            "fixture_version": str(sess.fixture_version or ""),
            "fixture_test_mode": str(sess.fixture_test_mode or ""),
        }

    def _live_archive_result_snapshot_locked(self, arc: ClosedSessionArchive) -> dict[str, Any]:
        chunks = [dict(r) for r in arc.live_commit_results]
        chunk_debug = _live_commit_rows_debug_metrics(chunks)
        live_engine = str(arc.live_engine or "rolling_context")
        preview_source = self._preview_source_for_engine(live_engine)
        final_segments_src = arc.live_final_segments or arc.final_segments
        final_covered_ms = 0
        for seg in final_segments_src:
            if not isinstance(seg, dict):
                continue
            try:
                t1 = int(seg.get("t1_ms") or 0)
            except Exception:
                t1 = 0
            if t1 > final_covered_ms:
                final_covered_ms = t1
        if final_covered_ms <= 0:
            for r in chunks:
                if str(r.get("state") or "") != "ready":
                    continue
                try:
                    t1 = int(r.get("t1_ms") or 0)
                except Exception:
                    t1 = 0
                if t1 > final_covered_ms:
                    final_covered_ms = t1
        engine_runtime = {
            "mode": "single_lane",
            "preview_source": str(preview_source),
            "uncommitted_audio_ms": int(max(0, int(arc.recording_duration_ms) - int(final_covered_ms))),
        }
        extra_engine_runtime = dict(arc.live_engine_runtime or {})
        if extra_engine_runtime:
            engine_runtime["engine_state"] = extra_engine_runtime
        return {
            "session_id": str(arc.session_id),
            "source": "archive",
            "live_engine": live_engine,
            "close_reason": str(arc.close_reason or ""),
            "finalization_state": str(arc.finalization_state or ""),
            "batch_job_id": str(arc.batch_job_id or ""),
            "recording_path": str(arc.recording_path or ""),
            "recording_bytes": int(max(0, arc.recording_bytes)),
            "recording_duration_ms": int(max(0, arc.recording_duration_ms)),
            "chunks_total": int(max(0, arc.chunks_total)),
            "chunks_done": int(max(0, arc.chunks_done)),
            "chunks_failed": int(max(0, arc.chunks_failed)),
            "chunks_pending": int(max(0, arc.chunks_total - arc.chunks_done - arc.chunks_failed)),
            "transcript_revision": int(max(0, arc.live_transcript_revision or arc.transcript_revision)),
            "final_text": str(arc.live_final_text or arc.final_text or ""),
            "final_segments": [dict(seg) for seg in final_segments_src],
            "final_segments_count": len(final_segments_src),
            "final_covered_ms": int(max(0, final_covered_ms)),
            "chunk_results": chunks,
            "chunk_results_count": len(chunks),
            "chunk_reason_counts": dict(chunk_debug.get("chunk_reason_counts") or {}),
            "chunk_results_rows_count": int(max(0, int(chunk_debug.get("chunk_results_rows_count") or 0))),
            "chunk_results_unique_count": int(max(0, int(chunk_debug.get("chunk_results_unique_count") or 0))),
            "chunk_results_duplicate_index_rows": int(
                max(0, int(chunk_debug.get("chunk_results_duplicate_index_rows") or 0))
            ),
            "chunk_results_invalid_index_rows": int(max(0, int(chunk_debug.get("chunk_results_invalid_index_rows") or 0))),
            "preview": {
                "text": "",
                "source": str(preview_source),
                "preview_seq": -1,
                "audio_end_ms": 0,
                "updated_at_utc": "",
            },
            "engine_runtime": engine_runtime,
            "fixture_id": str(arc.fixture_id or ""),
            "fixture_version": str(arc.fixture_version or ""),
            "fixture_test_mode": str(arc.fixture_test_mode or ""),
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
