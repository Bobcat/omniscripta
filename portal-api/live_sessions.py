from __future__ import annotations

import json
import re
import secrets
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_DEDUP_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _word_tokens_with_spans(text: str) -> list[tuple[str, int, int]]:
    raw = str(text or "")
    if not raw:
        return []
    lowered = raw.lower()
    out: list[tuple[str, int, int]] = []
    for m in _DEDUP_WORD_RE.finditer(lowered):
        word = m.group(0)
        if not word:
            continue
        out.append((word, int(m.start()), int(m.end())))
    return out


def _dedup_trim_incoming_prefix_words(
    existing_text: str,
    incoming_text: str,
    *,
    min_overlap_words: int,
    max_trim_words: int,
) -> tuple[str, dict[str, int | bool]]:
    incoming_raw = str(incoming_text or "")
    existing_raw = str(existing_text or "")
    meta: dict[str, int | bool] = {
        "dedup_applied": False,
        "merge_overlap_words": 0,
        "dedup_words_trimmed": 0,
    }
    if not existing_raw.strip() or not incoming_raw.strip():
        return incoming_raw, meta
    max_words = int(max(0, max_trim_words))
    min_words = int(max(1, min_overlap_words))
    if max_words <= 0:
        return incoming_raw, meta

    existing_tokens = _word_tokens_with_spans(existing_raw)
    incoming_tokens = _word_tokens_with_spans(incoming_raw)
    if not existing_tokens or not incoming_tokens:
        return incoming_raw, meta

    suffix_words = [w for (w, _, _) in existing_tokens[-max_words:]]
    prefix_tokens = incoming_tokens[:max_words]
    prefix_words = [w for (w, _, _) in prefix_tokens]
    limit = min(len(suffix_words), len(prefix_words), max_words)
    if limit < min_words:
        return incoming_raw, meta

    overlap = 0
    for k in range(limit, min_words - 1, -1):
        if suffix_words[-k:] == prefix_words[:k]:
            overlap = int(k)
            break
    if overlap < min_words:
        return incoming_raw, meta

    cutoff = int(prefix_tokens[overlap - 1][2])
    # Skip separator run after the duplicated prefix.
    while cutoff < len(incoming_raw) and not incoming_raw[cutoff].isalnum():
        cutoff += 1
    trimmed = incoming_raw[cutoff:].lstrip()

    meta["dedup_applied"] = True
    meta["merge_overlap_words"] = int(overlap)
    meta["dedup_words_trimmed"] = int(overlap)
    return trimmed, meta


def _utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()


def _semilive_chunk_rows_debug_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
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
    semilive_transcript_revision: int = 0
    semilive_final_text: str = ""
    semilive_final_segments: list[dict[str, Any]] = field(default_factory=list)
    semilive_chunk_results: list[dict[str, Any]] = field(default_factory=list)
    semilive_speculative_text: str = ""
    semilive_speculative_seq: int = -1
    semilive_speculative_audio_end_ms: int = 0
    semilive_speculative_updated_unix: float = 0.0
    semilive_speculative_enqueued: int = 0
    semilive_speculative_shown: int = 0
    semilive_speculative_dropped_busy: int = 0
    semilive_speculative_dropped_stale: int = 0
    semilive_time_to_first_speculative_ms: int | None = None
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
    recording_path: str = ""
    recording_bytes: int = 0
    recording_duration_ms: int = 0
    chunks_total: int = 0
    chunks_done: int = 0
    chunks_failed: int = 0
    finalization_state: str = ""
    batch_job_id: str = ""
    semilive_transcript_revision: int = 0
    semilive_final_text: str = ""
    semilive_final_segments: list[dict[str, Any]] = field(default_factory=list)
    semilive_chunk_results: list[dict[str, Any]] = field(default_factory=list)
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
        semilive_text_dedup_enabled: bool = False,
        semilive_text_dedup_min_words: int = 3,
        semilive_text_dedup_max_trim_words: int = 24,
    ):
        self._default_ttl_seconds = int(max(10, default_ttl_seconds))
        self._preconnect_ttl_seconds = int(max(5, preconnect_ttl_seconds))
        self._preconnect_ttl_seconds = int(min(self._preconnect_ttl_seconds, self._default_ttl_seconds))
        self._max_sessions = int(max(1, max_sessions))
        self._archive_ttl_seconds = int(max(60, archive_ttl_seconds))
        self._max_archives = int(max(1, max_archives))
        self._semilive_text_dedup_enabled = bool(semilive_text_dedup_enabled)
        self._semilive_text_dedup_min_words = int(max(1, semilive_text_dedup_min_words))
        self._semilive_text_dedup_max_trim_words = int(
            max(self._semilive_text_dedup_min_words, semilive_text_dedup_max_trim_words)
        )
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

    def update_semilive(
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

    def update_semilive_speculative_preview(
        self,
        session_id: str,
        *,
        text: str,
        speculative_seq: int,
        audio_end_ms: int,
    ) -> dict[str, Any]:
        now_unix = time.time()
        with self._lock:
            sess = self._sessions.get(session_id)
            if not sess:
                raise KeyError("session_not_found")
            sess.last_seen_unix = now_unix
            sess.semilive_speculative_text = str(text or "")
            sess.semilive_speculative_seq = int(speculative_seq)
            sess.semilive_speculative_audio_end_ms = int(max(0, int(audio_end_ms)))
            sess.semilive_speculative_updated_unix = now_unix
            return self._snapshot_locked(sess)

    def clear_semilive_speculative_preview(
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
            current_seq = int(getattr(sess, "semilive_speculative_seq", -1) or -1)
            if max_seq is None or current_seq <= int(max_seq):
                sess.semilive_speculative_text = ""
                sess.semilive_speculative_seq = -1
                sess.semilive_speculative_audio_end_ms = 0
                sess.semilive_speculative_updated_unix = 0.0
            return self._snapshot_locked(sess)

    def update_semilive_speculative_metrics(
        self,
        session_id: str,
        *,
        enqueued: int | None = None,
        shown: int | None = None,
        dropped_busy: int | None = None,
        dropped_stale: int | None = None,
        time_to_first_speculative_ms: int | None = None,
    ) -> dict[str, Any]:
        now_unix = time.time()
        with self._lock:
            sess = self._sessions.get(session_id)
            if not sess:
                raise KeyError("session_not_found")
            sess.last_seen_unix = now_unix
            if enqueued is not None:
                sess.semilive_speculative_enqueued = int(max(0, int(enqueued)))
            if shown is not None:
                sess.semilive_speculative_shown = int(max(0, int(shown)))
            if dropped_busy is not None:
                sess.semilive_speculative_dropped_busy = int(max(0, int(dropped_busy)))
            if dropped_stale is not None:
                sess.semilive_speculative_dropped_stale = int(max(0, int(dropped_stale)))
            if time_to_first_speculative_ms is not None:
                sess.semilive_time_to_first_speculative_ms = int(max(0, int(time_to_first_speculative_ms)))
            return self._snapshot_locked(sess)

    def record_semilive_chunk_result(
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
            for i, existing in enumerate(sess.semilive_chunk_results):
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
                    sess.semilive_chunk_results[i] = row
                    replaced = True
                    break
            if not replaced:
                sess.semilive_chunk_results.append(row)
                sess.semilive_chunk_results.sort(key=lambda r: int(r.get("chunk_index") or 0))

            sess.chunks_total = max(int(sess.chunks_total), idx + 1)
            sess.chunk_index_next = max(int(sess.chunk_index_next), idx + 1)
            if safe_state == "ready":
                sess.chunks_done = max(
                    0,
                    sum(1 for r in sess.semilive_chunk_results if str(r.get("state") or "") == "ready"),
                )
                sess.chunks_failed = max(
                    0,
                    sum(1 for r in sess.semilive_chunk_results if str(r.get("state") or "") == "error"),
                )
                # A new final chunk supersedes any speculative suffix that was shown before it.
                sess.semilive_speculative_text = ""
                sess.semilive_speculative_seq = -1
                sess.semilive_speculative_audio_end_ms = 0
                sess.semilive_speculative_updated_unix = 0.0
            elif safe_state == "error":
                sess.chunks_failed = max(
                    0,
                    sum(1 for r in sess.semilive_chunk_results if str(r.get("state") or "") == "error"),
                )
                sess.chunks_done = max(
                    0,
                    sum(1 for r in sess.semilive_chunk_results if str(r.get("state") or "") == "ready"),
                )

            merged_final_text = ""
            for r in sess.semilive_chunk_results:
                if not isinstance(r, dict):
                    continue
                r["dedup_applied"] = False
                r["merge_overlap_words"] = 0
                r["dedup_words_trimmed"] = 0

            for r in sess.semilive_chunk_results:
                if str(r.get("state") or "") != "ready":
                    continue
                raw_row_text = str(r.get("text") or "")
                row_text = raw_row_text.strip()
                if not row_text:
                    continue

                append_text = row_text
                if self._semilive_text_dedup_enabled and merged_final_text:
                    append_text, dedup_meta = _dedup_trim_incoming_prefix_words(
                        merged_final_text,
                        row_text,
                        min_overlap_words=self._semilive_text_dedup_min_words,
                        max_trim_words=self._semilive_text_dedup_max_trim_words,
                    )
                    r["dedup_applied"] = bool(dedup_meta.get("dedup_applied", False))
                    r["merge_overlap_words"] = int(max(0, int(dedup_meta.get("merge_overlap_words") or 0)))
                    r["dedup_words_trimmed"] = int(max(0, int(dedup_meta.get("dedup_words_trimmed") or 0)))
                append_text = str(append_text or "").strip()
                if not append_text:
                    continue
                merged_final_text = append_text if not merged_final_text else f"{merged_final_text}\n{append_text}"

            sess.semilive_final_text = merged_final_text

            merged_segments: list[dict[str, Any]] = []
            if any(r.get("segments") for r in sess.semilive_chunk_results):
                seg_counter = 0
                for r in sess.semilive_chunk_results:
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
                            merged_segments.append(
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
                            merged_segments.append(
                                {
                                    "segment_id": f"c{idx:04d}",
                                    "text": row_text,
                                    "t0_ms": row_t0,
                                    "t1_ms": row_t1,
                                }
                            )
            else:
                for r in sess.semilive_chunk_results:
                    if str(r.get("state") or "") != "ready":
                        continue
                    row_text = str(r.get("text") or "").strip()
                    if not row_text:
                        continue
                    idx2 = int(max(0, int(r.get("chunk_index") or 0)))
                    row_t0 = int(max(0, int(r.get("t0_ms") or 0)))
                    row_t1 = int(max(row_t0, int(r.get("t1_ms") or row_t0)))
                    merged_segments.append(
                        {
                            "segment_id": f"c{idx2:04d}",
                            "text": row_text,
                            "t0_ms": row_t0,
                            "t1_ms": row_t1,
                        }
                    )

            sess.semilive_final_segments = merged_segments
            sess.semilive_transcript_revision += 1
            return self._semilive_result_snapshot_locked(sess)

    def semilive_result_snapshot(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            sess = self._sessions.get(session_id)
            if sess is not None:
                return self._semilive_result_snapshot_locked(sess)
            arc = self._archives.get(session_id)
            if arc is not None:
                return self._semilive_archive_result_snapshot_locked(arc)
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
                recording_path=str(recording_path or ""),
                recording_bytes=int(max(0, recording_bytes)),
                recording_duration_ms=int(max(0, recording_duration_ms)),
                chunks_total=int(max(0, chunks_total)),
                chunks_done=int(max(0, chunks_done)),
                chunks_failed=int(max(0, chunks_failed)),
                finalization_state=str(finalization_state or ""),
                batch_job_id=str(batch_job_id or ""),
                semilive_transcript_revision=0,
                semilive_final_text="",
                semilive_final_segments=[],
                semilive_chunk_results=[],
                fixture_id="",
                fixture_version="",
                fixture_test_mode="",
            )
            src_sess = self._sessions.get(str(session_id))
            if src_sess is not None:
                arc.semilive_transcript_revision = int(max(0, src_sess.semilive_transcript_revision))
                arc.semilive_final_text = str(src_sess.semilive_final_text or "")
                arc.semilive_final_segments = [dict(seg) for seg in src_sess.semilive_final_segments]
                arc.semilive_chunk_results = [dict(r) for r in src_sess.semilive_chunk_results]
                arc.fixture_id = str(src_sess.fixture_id or "")
                arc.fixture_version = str(src_sess.fixture_version or "")
                arc.fixture_test_mode = str(src_sess.fixture_test_mode or "")
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
            "semilive_transcript_revision": int(max(0, sess.semilive_transcript_revision)),
            "semilive_final_text_chars": len(str(sess.semilive_final_text or "")),
            "semilive_final_segments_count": len(sess.semilive_final_segments),
            "semilive_chunk_results_count": len(sess.semilive_chunk_results),
            "fixture_id": str(sess.fixture_id or ""),
            "fixture_version": str(sess.fixture_version or ""),
            "fixture_test_mode": str(sess.fixture_test_mode or ""),
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
            "recording_path": str(arc.recording_path or ""),
            "recording_bytes": int(max(0, arc.recording_bytes)),
            "recording_duration_ms": int(max(0, arc.recording_duration_ms)),
            "chunks_total": int(max(0, arc.chunks_total)),
            "chunks_done": int(max(0, arc.chunks_done)),
            "chunks_failed": int(max(0, arc.chunks_failed)),
            "finalization_state": str(arc.finalization_state or ""),
            "batch_job_id": str(arc.batch_job_id or ""),
            "semilive_transcript_revision": int(max(0, arc.semilive_transcript_revision)),
            "semilive_final_text_chars": len(str(arc.semilive_final_text or "")),
            "semilive_final_segments_count": len(arc.semilive_final_segments),
            "semilive_chunk_results_count": len(arc.semilive_chunk_results),
            "fixture_id": str(arc.fixture_id or ""),
            "fixture_version": str(arc.fixture_version or ""),
            "fixture_test_mode": str(arc.fixture_test_mode or ""),
        }

    def _semilive_result_snapshot_locked(self, sess: LiveSession) -> dict[str, Any]:
        chunks = [dict(r) for r in sess.semilive_chunk_results]
        dedup_chunks_applied = sum(1 for r in chunks if bool(r.get("dedup_applied")))
        dedup_words_trimmed_total = sum(int(max(0, int(r.get("dedup_words_trimmed") or 0))) for r in chunks)
        chunk_debug = _semilive_chunk_rows_debug_metrics(chunks)
        final_covered_ms = 0
        for seg in sess.semilive_final_segments:
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
        return {
            "session_id": str(sess.session_id),
            "source": "active",
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
            "transcript_revision": int(max(0, sess.semilive_transcript_revision)),
            "final_text": str(sess.semilive_final_text or ""),
            "final_segments": [dict(seg) for seg in sess.semilive_final_segments],
            "final_segments_count": len(sess.semilive_final_segments),
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
            "dedup_chunks_applied": int(max(0, dedup_chunks_applied)),
            "dedup_words_trimmed_total": int(max(0, dedup_words_trimmed_total)),
            "speculative_preview": {
                "text": str(sess.semilive_speculative_text or ""),
                "speculative_seq": int(max(-1, int(sess.semilive_speculative_seq))),
                "audio_end_ms": int(max(0, int(sess.semilive_speculative_audio_end_ms or 0))),
                "updated_at_utc": (
                    _utc_iso(sess.semilive_speculative_updated_unix)
                    if float(sess.semilive_speculative_updated_unix or 0.0) > 0
                    else ""
                ),
            },
            "speculative_metrics": {
                "enqueued": int(max(0, int(sess.semilive_speculative_enqueued or 0))),
                "shown": int(max(0, int(sess.semilive_speculative_shown or 0))),
                "dropped_busy": int(max(0, int(sess.semilive_speculative_dropped_busy or 0))),
                "dropped_stale": int(max(0, int(sess.semilive_speculative_dropped_stale or 0))),
                "time_to_first_speculative_ms": (
                    int(sess.semilive_time_to_first_speculative_ms)
                    if sess.semilive_time_to_first_speculative_ms is not None
                    else None
                ),
            },
            "fixture_id": str(sess.fixture_id or ""),
            "fixture_version": str(sess.fixture_version or ""),
            "fixture_test_mode": str(sess.fixture_test_mode or ""),
        }

    def _semilive_archive_result_snapshot_locked(self, arc: ClosedSessionArchive) -> dict[str, Any]:
        chunks = [dict(r) for r in arc.semilive_chunk_results]
        dedup_chunks_applied = sum(1 for r in chunks if bool(r.get("dedup_applied")))
        dedup_words_trimmed_total = sum(int(max(0, int(r.get("dedup_words_trimmed") or 0))) for r in chunks)
        chunk_debug = _semilive_chunk_rows_debug_metrics(chunks)
        final_segments_src = arc.semilive_final_segments or arc.final_segments
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
        return {
            "session_id": str(arc.session_id),
            "source": "archive",
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
            "transcript_revision": int(max(0, arc.semilive_transcript_revision or arc.transcript_revision)),
            "final_text": str(arc.semilive_final_text or arc.final_text or ""),
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
            "dedup_chunks_applied": int(max(0, dedup_chunks_applied)),
            "dedup_words_trimmed_total": int(max(0, dedup_words_trimmed_total)),
            "speculative_preview": {
                "text": "",
                "speculative_seq": -1,
                "audio_end_ms": 0,
                "updated_at_utc": "",
            },
            "speculative_metrics": {
                "enqueued": 0,
                "shown": 0,
                "dropped_busy": 0,
                "dropped_stale": 0,
                "time_to_first_speculative_ms": None,
            },
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
