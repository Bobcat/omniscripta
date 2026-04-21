from __future__ import annotations

import json
import secrets
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from live.session.metrics import (
    build_live_session_manager_metrics_snapshot,
    live_commit_rows_debug_metrics,
)


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


def _repo_root() -> Path:
    # app/live/session/sessions.py -> session -> live -> app -> repo root
    return Path(__file__).resolve().parents[3]


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
    live_final_segments: list[dict[str, Any]] = field(default_factory=list)
    live_commit_results: list[dict[str, Any]] = field(default_factory=list)
    live_pc_events: list[dict[str, str]] = field(default_factory=list)
    live_preview_text: str = ""
    live_preview_seq: int = -1
    live_preview_audio_end_ms: int = 0
    live_preview_updated_unix: float = 0.0
    fixture_id: str = ""
    fixture_version: str = ""
    fixture_test_mode: str = ""
    asr_language: str = ""
    live_engine_runtime: dict[str, Any] = field(default_factory=dict)
    gpu_proxy_transcribe_s: float = 0.0
    gpu_proxy_pipeline_s: float = 0.0


@dataclass
class ClosedSessionArchive:
    session_id: str
    closed_unix: float
    expires_unix: float
    close_reason: str
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
    live_final_segments: list[dict[str, Any]] = field(default_factory=list)
    live_commit_results: list[dict[str, Any]] = field(default_factory=list)
    live_pc_events: list[dict[str, str]] = field(default_factory=list)
    fixture_id: str = ""
    fixture_version: str = ""
    fixture_test_mode: str = ""
    asr_language: str = ""
    live_engine_runtime: dict[str, Any] = field(default_factory=dict)
    gpu_proxy_transcribe_s: float = 0.0
    gpu_proxy_pipeline_s: float = 0.0


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

    @staticmethod
    def _copy_commit_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [dict(row) for row in rows]

    @staticmethod
    def _copy_pc_events(events: list[dict[str, str]]) -> list[dict[str, str]]:
        return [dict(event) for event in events]

    @staticmethod
    def _append_pc_event(sess: LiveSession, *, kind: str, text: str) -> None:
        safe_kind = str(kind or "").strip().lower()
        if safe_kind not in {"p", "c"}:
            return
        safe_text = str(text or "")
        if safe_kind == "c" and not safe_text:
            return
        sess.live_pc_events.append({"kind": safe_kind, "text": safe_text})

    @staticmethod
    def _count_commit_results(rows: list[dict[str, Any]], *, state: str) -> int:
        return max(0, sum(1 for row in rows if str(row.get("state") or "") == state))

    @staticmethod
    def _final_covered_ms(
        final_segments: list[dict[str, Any]],
        chunk_rows: list[dict[str, Any]],
    ) -> int:
        final_covered_ms = 0
        for seg in final_segments:
            if not isinstance(seg, dict):
                continue
            try:
                t1 = int(seg.get("t1_ms") or 0)
            except Exception:
                t1 = 0
            if t1 > final_covered_ms:
                final_covered_ms = t1
        if final_covered_ms > 0:
            return int(final_covered_ms)
        for row in chunk_rows:
            if str(row.get("state") or "") != "ready":
                continue
            try:
                t1 = int(row.get("t1_ms") or 0)
            except Exception:
                t1 = 0
            if t1 > final_covered_ms:
                final_covered_ms = t1
        return int(max(0, final_covered_ms))

    @staticmethod
    def _build_engine_runtime(
        *,
        recording_duration_ms: int,
        final_covered_ms: int,
        live_engine_runtime: dict[str, Any],
    ) -> dict[str, Any]:
        engine_runtime = {
            "mode": "single_lane",
            "preview_source": "uncommitted_preview",
            "uncommitted_audio_ms": int(max(0, int(recording_duration_ms) - int(final_covered_ms))),
        }
        extra_engine_runtime = dict(live_engine_runtime or {})
        if extra_engine_runtime:
            engine_runtime["engine_state"] = extra_engine_runtime
        return engine_runtime

    @staticmethod
    def _merge_live_commit_row(existing: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
        merged = dict(row)
        if not merged["reason"]:
            merged["reason"] = str(existing.get("reason") or "")
        if merged["speech_frames"] is None and existing.get("speech_frames") is not None:
            merged["speech_frames"] = int(max(0, int(existing.get("speech_frames") or 0)))
        if merged["silence_frames_tail"] is None and existing.get("silence_frames_tail") is not None:
            merged["silence_frames_tail"] = int(max(0, int(existing.get("silence_frames_tail") or 0)))
        if merged["chunk_duration_ms"] is None and existing.get("chunk_duration_ms") is not None:
            merged["chunk_duration_ms"] = int(max(0, int(existing.get("chunk_duration_ms") or 0)))
        return merged

    def _upsert_live_commit_row(self, sess: LiveSession, *, idx: int, row: dict[str, Any]) -> None:
        for i, existing in enumerate(sess.live_commit_results):
            try:
                existing_idx = int(existing.get("chunk_index"))
            except Exception:
                existing_idx = -1
            if existing_idx == idx:
                sess.live_commit_results[i] = self._merge_live_commit_row(existing, row)
                return
        sess.live_commit_results.append(dict(row))
        sess.live_commit_results.sort(key=lambda r: int(r.get("chunk_index") or 0))

    def _sync_live_commit_counts(self, sess: LiveSession) -> None:
        sess.chunks_done = self._count_commit_results(sess.live_commit_results, state="ready")
        sess.chunks_failed = self._count_commit_results(sess.live_commit_results, state="error")

    def _materialize_live_final_segments(
        self,
        rows: list[dict[str, Any]],
        *,
        fallback_chunk_index: int,
    ) -> list[dict[str, Any]]:
        appended_segments: list[dict[str, Any]] = []
        if any(row.get("segments") for row in rows):
            seg_counter = 0
            for row in rows:
                if str(row.get("state") or "") != "ready":
                    continue
                row_t0 = int(max(0, int(row.get("t0_ms") or 0)))
                row_t1 = int(max(row_t0, int(row.get("t1_ms") or row_t0)))
                row_segments = row.get("segments")
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
                                "segment_id": str(seg.get("segment_id") or f"c{fallback_chunk_index:04d}s{seg_counter:04d}"),
                                "text": seg_text,
                                "t0_ms": seg_t0,
                                "t1_ms": seg_t1,
                                "speaker": str(seg.get("speaker") or "").strip(),
                            }
                        )
                else:
                    row_text = str(row.get("text") or "").strip()
                    if row_text:
                        seg_counter += 1
                        appended_segments.append(
                            {
                                "segment_id": f"c{fallback_chunk_index:04d}",
                                "text": row_text,
                                "t0_ms": row_t0,
                                "t1_ms": row_t1,
                                "speaker": "",
                            }
                        )
            return appended_segments

        for row in rows:
            if str(row.get("state") or "") != "ready":
                continue
            row_text = str(row.get("text") or "").strip()
            if not row_text:
                continue
            idx2 = int(max(0, int(row.get("chunk_index") or 0)))
            row_t0 = int(max(0, int(row.get("t0_ms") or 0)))
            row_t1 = int(max(row_t0, int(row.get("t1_ms") or row_t0)))
            appended_segments.append(
                {
                    "segment_id": f"c{idx2:04d}",
                    "text": row_text,
                    "t0_ms": row_t0,
                    "t1_ms": row_t1,
                    "speaker": "",
                }
            )
        return appended_segments

    def _build_live_result_snapshot(
        self,
        *,
        session_id: str,
        source: str,
        live_engine: str,
        recording_state: str = "",
        close_reason: str = "",
        state: str = "",
        finalization_state: str,
        batch_job_id: str,
        recording_path: str,
        recording_bytes: int,
        recording_duration_ms: int,
        chunks_total: int,
        chunks_done: int,
        chunks_failed: int,
        transcript_revision: int,
        final_segments: list[dict[str, Any]],
        commit_results: list[dict[str, Any]],
        pc_events_count: int,
        preview_text: str,
        preview_seq: int,
        preview_audio_end_ms: int,
        preview_updated_unix: float,
        live_engine_runtime: dict[str, Any],
        fixture_id: str,
        fixture_version: str,
        fixture_test_mode: str,
        asr_language: str,
        gpu_proxy_transcribe_s: float,
        gpu_proxy_pipeline_s: float,
    ) -> dict[str, Any]:
        chunks = self._copy_commit_rows(commit_results)
        chunk_debug = live_commit_rows_debug_metrics(chunks)
        final_covered_ms = self._final_covered_ms(final_segments, chunks)
        engine_runtime = self._build_engine_runtime(
            recording_duration_ms=recording_duration_ms,
            final_covered_ms=final_covered_ms,
            live_engine_runtime=live_engine_runtime,
        )
        payload = {
            "session_id": str(session_id),
            "source": str(source),
            "live_engine": str(live_engine),
            "finalization_state": str(finalization_state or ""),
            "batch_job_id": str(batch_job_id or ""),
            "recording_path": str(recording_path or ""),
            "recording_bytes": int(max(0, recording_bytes)),
            "recording_duration_ms": int(max(0, recording_duration_ms)),
            "chunks_total": int(max(0, chunks_total)),
            "chunks_done": int(max(0, chunks_done)),
            "chunks_failed": int(max(0, chunks_failed)),
            "chunks_pending": int(max(0, chunks_total - chunks_done - chunks_failed)),
            "transcript_revision": int(max(0, transcript_revision)),
            "final_segments": self._copy_commit_rows(final_segments),
            "final_segments_count": len(final_segments),
            "final_covered_ms": int(max(0, final_covered_ms)),
            "chunk_results": chunks,
            "chunk_results_count": len(chunks),
            "pc_events_count": int(max(0, pc_events_count)),
            "chunk_reason_counts": dict(chunk_debug.get("chunk_reason_counts") or {}),
            "chunk_results_rows_count": int(max(0, int(chunk_debug.get("chunk_results_rows_count") or 0))),
            "chunk_results_unique_count": int(max(0, int(chunk_debug.get("chunk_results_unique_count") or 0))),
            "chunk_results_duplicate_index_rows": int(
                max(0, int(chunk_debug.get("chunk_results_duplicate_index_rows") or 0))
            ),
            "chunk_results_invalid_index_rows": int(max(0, int(chunk_debug.get("chunk_results_invalid_index_rows") or 0))),
            "preview": {
                "text": str(preview_text or ""),
                "preview_seq": int(preview_seq),
                "audio_end_ms": int(max(0, int(preview_audio_end_ms or 0))),
                "updated_at_utc": (
                    _utc_iso(preview_updated_unix)
                    if float(preview_updated_unix or 0.0) > 0
                    else ""
                ),
            },
            "engine_runtime": engine_runtime,
            "fixture_id": str(fixture_id or ""),
            "fixture_version": str(fixture_version or ""),
            "fixture_test_mode": str(fixture_test_mode or ""),
            "asr_language": str(asr_language or ""),
            "gpu_proxy_transcribe_s": round(float(max(0.0, gpu_proxy_transcribe_s)), 3),
            "gpu_proxy_pipeline_s": round(float(max(0.0, gpu_proxy_pipeline_s)), 3),
        }
        if source == "active":
            payload["state"] = str(state or "")
            payload["recording_state"] = str(recording_state or "")
        else:
            payload["close_reason"] = str(close_reason or "")
        return payload

    def _new_session_id(self) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"live_{ts}_{secrets.token_hex(4)}"

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
        live_engine: str | None = None,
        asr_language: str | None = None,
    ) -> dict[str, Any]:
        now_unix = time.time()
        now_mono = time.monotonic()
        ttl = self._default_ttl_seconds if ttl_seconds is None else int(ttl_seconds)
        ttl = int(max(10, ttl))
        preconnect_ttl = int(max(5, min(ttl, self._preconnect_ttl_seconds)))
        engine_name = str(live_engine or "").strip().lower() or "rolling_context"
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
                live_engine=engine_name,
                last_seen_unix=now_unix,
                asr_language=session_asr_language,
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
        gpu_proxy_transcribe_s: float | None = None,
        gpu_proxy_pipeline_s: float | None = None,
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
            if gpu_proxy_transcribe_s is not None:
                sess.gpu_proxy_transcribe_s = max(0.0, float(gpu_proxy_transcribe_s))
            if gpu_proxy_pipeline_s is not None:
                sess.gpu_proxy_pipeline_s = max(0.0, float(gpu_proxy_pipeline_s))
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

    def set_asr_language(
        self,
        session_id: str,
        *,
        asr_language: str | None = None,
    ) -> dict[str, Any]:
        now_unix = time.time()
        safe_language = str(asr_language or "").strip().lower()
        with self._lock:
            sess = self._sessions.get(session_id)
            if not sess:
                raise KeyError("session_not_found")
            sess.last_seen_unix = now_unix
            sess.asr_language = safe_language
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
            self._append_pc_event(sess, kind="p", text=sess.live_preview_text)
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
                previous_preview_text = str(sess.live_preview_text or "")
                sess.live_preview_text = ""
                sess.live_preview_seq = -1
                sess.live_preview_audio_end_ms = 0
                sess.live_preview_updated_unix = 0.0
                if previous_preview_text:
                    self._append_pc_event(sess, kind="p", text="")
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
            }
            self._upsert_live_commit_row(sess, idx=idx, row=row)

            sess.chunks_total = max(int(sess.chunks_total), idx + 1)
            sess.chunk_index_next = max(int(sess.chunk_index_next), idx + 1)
            if safe_state == "ready":
                had_preview_text = bool(str(sess.live_preview_text or ""))
                self._append_pc_event(sess, kind="c", text=safe_text)
                self._sync_live_commit_counts(sess)
                # Keep preview-clear coupled to the ready-commit mutation.
                # This makes commit + preview-clear an atomic snapshot update.
                sess.live_preview_text = ""
                sess.live_preview_seq = -1
                sess.live_preview_audio_end_ms = 0
                sess.live_preview_updated_unix = 0.0
                if had_preview_text:
                    self._append_pc_event(sess, kind="p", text="")
            elif safe_state == "error":
                self._sync_live_commit_counts(sess)

            sess.live_final_segments = self._materialize_live_final_segments(
                sess.live_commit_results,
                fallback_chunk_index=idx,
            )
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

    def live_pc_events(self, session_id: str) -> list[dict[str, str]]:
        with self._lock:
            sess = self._sessions.get(session_id)
            if sess is not None:
                return self._copy_pc_events(sess.live_pc_events)
            arc = self._archives.get(session_id)
            if arc is not None:
                return self._copy_pc_events(arc.live_pc_events)
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
                live_final_segments=[],
                live_commit_results=[],
                fixture_id="",
                fixture_version="",
                fixture_test_mode="",
                asr_language="",
            )
            src_sess = self._sessions.get(str(session_id))
            if src_sess is not None:
                arc.live_transcript_revision = int(max(0, src_sess.live_transcript_revision))
                arc.live_final_segments = [dict(seg) for seg in src_sess.live_final_segments]
                arc.live_commit_results = [dict(r) for r in src_sess.live_commit_results]
                arc.live_pc_events = [dict(event) for event in src_sess.live_pc_events]
                arc.live_engine_runtime = dict(src_sess.live_engine_runtime or {})
                arc.fixture_id = str(src_sess.fixture_id or "")
                arc.fixture_version = str(src_sess.fixture_version or "")
                arc.fixture_test_mode = str(src_sess.fixture_test_mode or "")
                arc.asr_language = str(src_sess.asr_language or "")
                arc.gpu_proxy_transcribe_s = float(max(0.0, src_sess.gpu_proxy_transcribe_s))
                arc.gpu_proxy_pipeline_s = float(max(0.0, src_sess.gpu_proxy_pipeline_s))
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
            "live_final_segments_count": len(sess.live_final_segments),
            "live_commit_results_count": len(sess.live_commit_results),
            "fixture_id": str(sess.fixture_id or ""),
            "fixture_version": str(sess.fixture_version or ""),
            "fixture_test_mode": str(sess.fixture_test_mode or ""),
            "asr_language": str(sess.asr_language or ""),
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
            "live_final_segments_count": len(arc.live_final_segments),
            "live_commit_results_count": len(arc.live_commit_results),
            "fixture_id": str(arc.fixture_id or ""),
            "fixture_version": str(arc.fixture_version or ""),
            "fixture_test_mode": str(arc.fixture_test_mode or ""),
            "asr_language": str(arc.asr_language or ""),
        }

    def _live_result_snapshot_locked(self, sess: LiveSession) -> dict[str, Any]:
        return self._build_live_result_snapshot(
            session_id=str(sess.session_id),
            source="active",
            live_engine=str(sess.live_engine or "rolling_context"),
            state=str(sess.state or ""),
            recording_state=str(sess.recording_state or ""),
            finalization_state=str(sess.finalization_state or ""),
            batch_job_id=str(sess.batch_job_id or ""),
            recording_path=str(sess.recording_path or ""),
            recording_bytes=int(max(0, sess.recording_bytes)),
            recording_duration_ms=int(max(0, sess.recording_duration_ms)),
            chunks_total=int(max(0, sess.chunks_total)),
            chunks_done=int(max(0, sess.chunks_done)),
            chunks_failed=int(max(0, sess.chunks_failed)),
            transcript_revision=int(max(0, sess.live_transcript_revision)),
            final_segments=sess.live_final_segments,
            commit_results=sess.live_commit_results,
            pc_events_count=len(sess.live_pc_events),
            preview_text=str(sess.live_preview_text or "").strip(),
            preview_seq=int(max(-1, int(sess.live_preview_seq))),
            preview_audio_end_ms=int(max(0, int(sess.live_preview_audio_end_ms or 0))),
            preview_updated_unix=float(sess.live_preview_updated_unix or 0.0),
            live_engine_runtime=dict(sess.live_engine_runtime or {}),
            fixture_id=str(sess.fixture_id or ""),
            fixture_version=str(sess.fixture_version or ""),
            fixture_test_mode=str(sess.fixture_test_mode or ""),
            asr_language=str(sess.asr_language or ""),
            gpu_proxy_transcribe_s=float(max(0.0, sess.gpu_proxy_transcribe_s)),
            gpu_proxy_pipeline_s=float(max(0.0, sess.gpu_proxy_pipeline_s)),
        )

    def _live_archive_result_snapshot_locked(self, arc: ClosedSessionArchive) -> dict[str, Any]:
        final_segments_src = arc.live_final_segments or arc.final_segments
        return self._build_live_result_snapshot(
            session_id=str(arc.session_id),
            source="archive",
            close_reason=str(arc.close_reason or ""),
            live_engine=str(arc.live_engine or "rolling_context"),
            finalization_state=str(arc.finalization_state or ""),
            batch_job_id=str(arc.batch_job_id or ""),
            recording_path=str(arc.recording_path or ""),
            recording_bytes=int(max(0, arc.recording_bytes)),
            recording_duration_ms=int(max(0, arc.recording_duration_ms)),
            chunks_total=int(max(0, arc.chunks_total)),
            chunks_done=int(max(0, arc.chunks_done)),
            chunks_failed=int(max(0, arc.chunks_failed)),
            transcript_revision=int(max(0, arc.live_transcript_revision or arc.transcript_revision)),
            final_segments=final_segments_src,
            commit_results=arc.live_commit_results,
            pc_events_count=len(arc.live_pc_events),
            preview_text="",
            preview_seq=-1,
            preview_audio_end_ms=0,
            preview_updated_unix=0.0,
            live_engine_runtime=dict(arc.live_engine_runtime or {}),
            fixture_id=str(arc.fixture_id or ""),
            fixture_version=str(arc.fixture_version or ""),
            fixture_test_mode=str(arc.fixture_test_mode or ""),
            asr_language=str(arc.asr_language or ""),
            gpu_proxy_transcribe_s=float(max(0.0, arc.gpu_proxy_transcribe_s)),
            gpu_proxy_pipeline_s=float(max(0.0, arc.gpu_proxy_pipeline_s)),
        )

    def metrics_snapshot(self) -> dict[str, Any]:
        now_unix = time.time()
        now_mono = time.monotonic()
        with self._lock:
            self._cleanup_expired_locked(now_unix)
            active = list(self._sessions.values())
            return build_live_session_manager_metrics_snapshot(
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
