from __future__ import annotations

from contextlib import suppress
import json
import secrets
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from live.session.metrics import build_live_session_manager_metrics_payload
from live.session.payloads import (
    build_archive_payload,
    build_live_archive_result_payload,
    build_live_result_payload,
    build_live_session_payload,
)
from live.session.state import (
    ClosedSessionArchive,
    LiveSession,
    append_preview_text,
    copy_pc_events,
    count_commit_results,
    materialize_live_final_segments,
    merge_live_commit_row,
)


def _repo_root() -> Path:
    # app/live/session/manager.py -> session -> live -> app -> repo root
    return Path(__file__).resolve().parents[3]


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
        self._stats_log_dir = (_repo_root() / "data" / "live" / "stats").resolve()

    @staticmethod
    def _append_pc_event(sess: LiveSession, *, kind: str, text: str) -> None:
        safe_kind = str(kind or "").strip().lower()
        if safe_kind not in {"p", "c"}:
            return
        safe_text = str(text or "")
        if safe_kind == "c" and not safe_text:
            return
        sess.live_pc_events.append({"kind": safe_kind, "text": safe_text})

    def _upsert_live_commit_row(self, sess: LiveSession, *, idx: int, row: dict[str, Any]) -> None:
        for i, existing in enumerate(sess.live_commit_results):
            try:
                existing_idx = int(existing.get("chunk_index"))
            except Exception:
                existing_idx = -1
            if existing_idx == idx:
                sess.live_commit_results[i] = merge_live_commit_row(existing, row)
                return
        sess.live_commit_results.append(dict(row))
        sess.live_commit_results.sort(key=lambda r: int(r.get("chunk_index") or 0))

    def _sync_live_commit_counts(self, sess: LiveSession) -> None:
        sess.chunks_done = count_commit_results(sess.live_commit_results, state="ready")
        sess.chunks_failed = count_commit_results(sess.live_commit_results, state="error")

    def _new_session_id(self) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"live_{ts}_{secrets.token_hex(4)}"

    def _session_for_update_locked(self, session_id: str, *, now_unix: float) -> LiveSession:
        sess = self._sessions.get(session_id)
        if not sess:
            raise KeyError("session_not_found")
        sess.last_seen_unix = now_unix
        return sess

    def _append_stats_log_best_effort(self, session_id: str, payload: dict[str, Any]) -> None:
        with suppress(Exception):
            self.append_stats_log(session_id, payload)

    def _cleanup_expired_locked(self, now_unix: float) -> None:
        dead = [
            sid
            for sid, sess in self._sessions.items()
            if sess.closed or (not sess.ws_connected and now_unix >= sess.expires_unix)
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
        asr_language: str | None = None,
    ) -> dict[str, Any]:
        now_unix = time.time()
        now_mono = time.monotonic()
        ttl = self._default_ttl_seconds if ttl_seconds is None else int(ttl_seconds)
        ttl = int(max(10, ttl))
        preconnect_ttl = int(max(5, min(ttl, self._preconnect_ttl_seconds)))
        session_asr_language = str(asr_language or "").strip().lower()

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
                asr_language=session_asr_language,
            )
            self._sessions[session_id] = sess
            snapshot = self._session_payload_locked(sess)
        self._append_stats_log_best_effort(session_id, {"kind": "session_created", "session": snapshot})
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
            return self._session_payload_locked(sess)

    def mark_state(self, session_id: str, *, state: str) -> dict[str, Any]:
        now_unix = time.time()
        with self._lock:
            sess = self._session_for_update_locked(session_id, now_unix=now_unix)
            sess.state = str(state or "connected")
            return self._session_payload_locked(sess)

    def record_audio(self, session_id: str, *, byte_count: int) -> dict[str, Any]:
        now_unix = time.time()
        safe_bytes = max(0, int(byte_count))
        with self._lock:
            sess = self._session_for_update_locked(session_id, now_unix=now_unix)
            sess.bytes_received += safe_bytes
            sess.frames_received += 1
            return self._session_payload_locked(sess)

    def record_control(self, session_id: str) -> dict[str, Any]:
        now_unix = time.time()
        with self._lock:
            sess = self._session_for_update_locked(session_id, now_unix=now_unix)
            sess.controls_received += 1
            return self._session_payload_locked(sess)

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
        asr_transcribe_s: float | None = None,
        asr_load_audio_s: float | None = None,
        asr_runner_wall_s: float | None = None,
        asr_pool_wall_s: float | None = None,
        asr_pool_ingest_s: float | None = None,
        asr_pool_ingest_body_read_s: float | None = None,
        asr_pool_ingest_multipart_parse_s: float | None = None,
        asr_pool_ingest_audio_write_s: float | None = None,
        asr_pool_ingest_submit_enqueue_s: float | None = None,
        asr_pool_queue_wait_s: float | None = None,
        asr_pool_outside_runner_s: float | None = None,
        asr_backend_wall_s: float | None = None,
        asr_backend_wav_write_s: float | None = None,
        asr_backend_submit_s: float | None = None,
        asr_backend_result_collect_s: float | None = None,
        asr_backend_artifact_get_s: float | None = None,
        asr_backend_srt_parse_s: float | None = None,
        asr_backend_outside_pool_s: float | None = None,
    ) -> dict[str, Any]:
        now_unix = time.time()
        with self._lock:
            sess = self._session_for_update_locked(session_id, now_unix=now_unix)
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
            if asr_transcribe_s is not None:
                sess.asr_transcribe_s = max(0.0, float(asr_transcribe_s))
            if asr_load_audio_s is not None:
                sess.asr_load_audio_s = max(0.0, float(asr_load_audio_s))
            if asr_runner_wall_s is not None:
                sess.asr_runner_wall_s = max(0.0, float(asr_runner_wall_s))
            if asr_pool_wall_s is not None:
                sess.asr_pool_wall_s = max(0.0, float(asr_pool_wall_s))
            if asr_pool_ingest_s is not None:
                sess.asr_pool_ingest_s = max(0.0, float(asr_pool_ingest_s))
            if asr_pool_ingest_body_read_s is not None:
                sess.asr_pool_ingest_body_read_s = max(0.0, float(asr_pool_ingest_body_read_s))
            if asr_pool_ingest_multipart_parse_s is not None:
                sess.asr_pool_ingest_multipart_parse_s = max(0.0, float(asr_pool_ingest_multipart_parse_s))
            if asr_pool_ingest_audio_write_s is not None:
                sess.asr_pool_ingest_audio_write_s = max(0.0, float(asr_pool_ingest_audio_write_s))
            if asr_pool_ingest_submit_enqueue_s is not None:
                sess.asr_pool_ingest_submit_enqueue_s = max(0.0, float(asr_pool_ingest_submit_enqueue_s))
            if asr_pool_queue_wait_s is not None:
                sess.asr_pool_queue_wait_s = max(0.0, float(asr_pool_queue_wait_s))
            if asr_pool_outside_runner_s is not None:
                sess.asr_pool_outside_runner_s = max(0.0, float(asr_pool_outside_runner_s))
            if asr_backend_wall_s is not None:
                sess.asr_backend_wall_s = max(0.0, float(asr_backend_wall_s))
            if asr_backend_wav_write_s is not None:
                sess.asr_backend_wav_write_s = max(0.0, float(asr_backend_wav_write_s))
            if asr_backend_submit_s is not None:
                sess.asr_backend_submit_s = max(0.0, float(asr_backend_submit_s))
            if asr_backend_result_collect_s is not None:
                sess.asr_backend_result_collect_s = max(0.0, float(asr_backend_result_collect_s))
            if asr_backend_artifact_get_s is not None:
                sess.asr_backend_artifact_get_s = max(0.0, float(asr_backend_artifact_get_s))
            if asr_backend_srt_parse_s is not None:
                sess.asr_backend_srt_parse_s = max(0.0, float(asr_backend_srt_parse_s))
            if asr_backend_outside_pool_s is not None:
                sess.asr_backend_outside_pool_s = max(0.0, float(asr_backend_outside_pool_s))
            return self._session_payload_locked(sess)

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
            sess = self._session_for_update_locked(session_id, now_unix=now_unix)
            if fixture_id is not None:
                sess.fixture_id = str(fixture_id or "").strip()
            if fixture_version is not None:
                sess.fixture_version = str(fixture_version or "").strip()
            if fixture_test_mode is not None:
                sess.fixture_test_mode = str(fixture_test_mode or "").strip()
            return self._session_payload_locked(sess)

    def set_asr_language(
        self,
        session_id: str,
        *,
        asr_language: str | None = None,
    ) -> dict[str, Any]:
        now_unix = time.time()
        safe_language = str(asr_language or "").strip().lower()
        with self._lock:
            sess = self._session_for_update_locked(session_id, now_unix=now_unix)
            sess.asr_language = safe_language
            return self._session_payload_locked(sess)

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
            sess = self._session_for_update_locked(session_id, now_unix=now_unix)
            current_seq = int(getattr(sess, "live_preview_seq", -1) or -1)
            if safe_seq <= current_seq:
                return self._session_payload_locked(sess)

            if append_to_existing:
                preview_text = append_preview_text(
                    existing_text=str(sess.live_preview_text or ""),
                    incoming_text=incoming_raw_text,
                )
            else:
                preview_text = str(incoming_raw_text or "").strip()

            sess.live_preview_text = str(preview_text or "")
            sess.live_preview_seq = int(safe_seq)
            sess.live_preview_audio_end_ms = int(safe_audio_end_ms)
            sess.live_preview_updated_unix = now_unix
            self._append_pc_event(sess, kind="p", text=sess.live_preview_text)
            return self._session_payload_locked(sess)

    def clear_live_preview(
        self,
        session_id: str,
        *,
        max_seq: int | None = None,
    ) -> dict[str, Any]:
        now_unix = time.time()
        with self._lock:
            sess = self._session_for_update_locked(session_id, now_unix=now_unix)
            current_seq = int(getattr(sess, "live_preview_seq", -1) or -1)
            if max_seq is None or current_seq <= int(max_seq):
                previous_preview_text = str(sess.live_preview_text or "")
                sess.live_preview_text = ""
                sess.live_preview_seq = -1
                sess.live_preview_audio_end_ms = 0
                sess.live_preview_updated_unix = 0.0
                if previous_preview_text:
                    self._append_pc_event(sess, kind="p", text="")
            return self._session_payload_locked(sess)

    def set_live_engine_runtime(
        self,
        session_id: str,
        *,
        runtime: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now_unix = time.time()
        payload = dict(runtime or {})
        with self._lock:
            sess = self._session_for_update_locked(session_id, now_unix=now_unix)
            sess.live_engine_runtime = payload
            return self._session_payload_locked(sess)

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
        with self._lock:
            sess = self._session_for_update_locked(session_id, now_unix=now_unix)

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
            }
            self._upsert_live_commit_row(sess, idx=idx, row=row)

            sess.chunks_total = max(int(sess.chunks_total), idx + 1)
            sess.chunk_index_next = max(int(sess.chunk_index_next), idx + 1)
            if safe_state == "ready":
                had_preview_text = bool(str(sess.live_preview_text or ""))
                self._append_pc_event(sess, kind="c", text=safe_text)
                # Keep preview-clear coupled to the ready-commit mutation.
                # This makes commit + preview-clear an atomic payload-visible update.
                sess.live_preview_text = ""
                sess.live_preview_seq = -1
                sess.live_preview_audio_end_ms = 0
                sess.live_preview_updated_unix = 0.0
                if had_preview_text:
                    self._append_pc_event(sess, kind="p", text="")
            self._sync_live_commit_counts(sess)
            sess.live_final_segments = materialize_live_final_segments(
                sess.live_commit_results,
                fallback_chunk_index=idx,
            )
            sess.live_transcript_revision += 1
            return self._live_result_payload_locked(sess)

    def live_result_payload(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            sess = self._sessions.get(session_id)
            if sess is not None:
                return self._live_result_payload_locked(sess)
            arc = self._archives.get(session_id)
            if arc is not None:
                return self._live_archive_result_payload_locked(arc)
        raise KeyError("session_or_archive_not_found")

    def live_pc_events(self, session_id: str) -> list[dict[str, str]]:
        with self._lock:
            sess = self._sessions.get(session_id)
            if sess is not None:
                return copy_pc_events(sess.live_pc_events)
            arc = self._archives.get(session_id)
            if arc is not None:
                return copy_pc_events(arc.live_pc_events)
        raise KeyError("session_or_archive_not_found")

    def next_seq(self, session_id: str) -> int:
        with self._lock:
            sess = self._sessions.get(session_id)
            if not sess:
                raise KeyError("session_not_found")
            sess.seq += 1
            return int(sess.seq)

    def session_payload(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            sess = self._sessions.get(session_id)
            if not sess:
                raise KeyError("session_not_found")
            return self._session_payload_locked(sess)

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
            snapshot = self._session_payload_locked(sess)
        self._append_stats_log_best_effort(
            session_id,
            {
                "kind": "session_closed",
                "close_reason": str(reason or "closed"),
                "session": snapshot,
            },
        )
        return snapshot

    def archive_transcript(
        self,
        session_id: str,
        *,
        close_reason: str,
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
    ) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            self._cleanup_expired_locked(now)
            arc = ClosedSessionArchive(
                session_id=str(session_id),
                closed_unix=now,
                expires_unix=now + self._archive_ttl_seconds,
                close_reason=str(close_reason or ""),
                final_segments=[dict(seg) for seg in (final_segments or [])],
                transcript_revision=int(max(0, transcript_revision)),
                recording_path=str(recording_path or ""),
                recording_bytes=int(max(0, recording_bytes)),
                recording_duration_ms=int(max(0, recording_duration_ms)),
                chunks_total=int(max(0, chunks_total)),
                chunks_done=int(max(0, chunks_done)),
                chunks_failed=int(max(0, chunks_failed)),
                finalization_state=str(finalization_state or ""),
                batch_job_id=str(batch_job_id or ""),
            )
            src_sess = self._sessions.get(arc.session_id)
            if src_sess is not None:
                arc.live_transcript_revision = int(max(0, src_sess.live_transcript_revision))
                arc.live_final_segments = [dict(seg) for seg in src_sess.live_final_segments]
                arc.live_commit_results = [dict(r) for r in src_sess.live_commit_results]
                arc.live_pc_events = copy_pc_events(src_sess.live_pc_events)
                arc.live_engine_runtime = dict(src_sess.live_engine_runtime or {})
                arc.fixture_id = str(src_sess.fixture_id or "")
                arc.fixture_version = str(src_sess.fixture_version or "")
                arc.fixture_test_mode = str(src_sess.fixture_test_mode or "")
                arc.asr_language = str(src_sess.asr_language or "")
                arc.asr_transcribe_s = float(max(0.0, src_sess.asr_transcribe_s))
                arc.asr_load_audio_s = float(max(0.0, src_sess.asr_load_audio_s))
                arc.asr_runner_wall_s = float(max(0.0, src_sess.asr_runner_wall_s))
                arc.asr_pool_wall_s = float(max(0.0, src_sess.asr_pool_wall_s))
                arc.asr_pool_ingest_s = float(max(0.0, src_sess.asr_pool_ingest_s))
                arc.asr_pool_ingest_body_read_s = float(max(0.0, src_sess.asr_pool_ingest_body_read_s))
                arc.asr_pool_ingest_multipart_parse_s = float(
                    max(0.0, src_sess.asr_pool_ingest_multipart_parse_s)
                )
                arc.asr_pool_ingest_audio_write_s = float(max(0.0, src_sess.asr_pool_ingest_audio_write_s))
                arc.asr_pool_ingest_submit_enqueue_s = float(
                    max(0.0, src_sess.asr_pool_ingest_submit_enqueue_s)
                )
                arc.asr_pool_queue_wait_s = float(max(0.0, src_sess.asr_pool_queue_wait_s))
                arc.asr_pool_outside_runner_s = float(max(0.0, src_sess.asr_pool_outside_runner_s))
                arc.asr_backend_wall_s = float(max(0.0, src_sess.asr_backend_wall_s))
                arc.asr_backend_wav_write_s = float(max(0.0, src_sess.asr_backend_wav_write_s))
                arc.asr_backend_submit_s = float(max(0.0, src_sess.asr_backend_submit_s))
                arc.asr_backend_result_collect_s = float(max(0.0, src_sess.asr_backend_result_collect_s))
                arc.asr_backend_artifact_get_s = float(max(0.0, src_sess.asr_backend_artifact_get_s))
                arc.asr_backend_srt_parse_s = float(max(0.0, src_sess.asr_backend_srt_parse_s))
                arc.asr_backend_outside_pool_s = float(max(0.0, src_sess.asr_backend_outside_pool_s))
            self._archives[arc.session_id] = arc
            return self._archive_payload_locked(arc)

    def archive_payload(self, session_id: str) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            self._cleanup_expired_locked(now)
            arc = self._archives.get(session_id)
            if not arc:
                raise KeyError("archive_not_found")
            return self._archive_payload_locked(arc)

    def _session_payload_locked(self, sess: LiveSession) -> dict[str, Any]:
        now_unix = time.time()
        return build_live_session_payload(
            sess,
            stats_log_path=str(self._stats_log_path(sess.session_id)),
            now_unix=now_unix,
            now_mono=time.monotonic(),
        )

    def _archive_payload_locked(self, arc: ClosedSessionArchive) -> dict[str, Any]:
        return build_archive_payload(arc)

    def _live_result_payload_locked(self, sess: LiveSession) -> dict[str, Any]:
        return build_live_result_payload(sess)

    def _live_archive_result_payload_locked(self, arc: ClosedSessionArchive) -> dict[str, Any]:
        return build_live_archive_result_payload(arc)

    def metrics_payload(self) -> dict[str, Any]:
        now_unix = time.time()
        now_mono = time.monotonic()
        with self._lock:
            self._cleanup_expired_locked(now_unix)
            active = list(self._sessions.values())
            return build_live_session_manager_metrics_payload(
                active_sessions=active,
                archived_sessions_count=len(self._archives),
                now_mono=now_mono,
                limits={
                    "max_sessions": int(self._max_sessions),
                    "default_ttl_seconds": int(self._default_ttl_seconds),
                    "preconnect_ttl_seconds": int(self._preconnect_ttl_seconds),
                    "archive_ttl_seconds": int(self._archive_ttl_seconds),
                    "max_archives": int(self._max_archives),
                },
            )

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
