from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def append_preview_text(existing_text: str, incoming_text: str) -> str:
    existing = str(existing_text or "").strip()
    incoming = str(incoming_text or "").strip()
    if not incoming:
        return existing
    if not existing:
        return incoming
    if incoming == existing:
        return existing
    return f"{existing} {incoming}"


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
    asr_transcribe_s: float = 0.0
    asr_load_audio_s: float = 0.0
    asr_runner_wall_s: float = 0.0
    asr_pool_wall_s: float = 0.0
    asr_pool_ingest_s: float = 0.0
    asr_pool_ingest_body_read_s: float = 0.0
    asr_pool_ingest_multipart_parse_s: float = 0.0
    asr_pool_ingest_audio_write_s: float = 0.0
    asr_pool_ingest_submit_enqueue_s: float = 0.0
    asr_pool_queue_wait_s: float = 0.0
    asr_pool_outside_runner_s: float = 0.0
    asr_backend_wall_s: float = 0.0
    asr_backend_wav_write_s: float = 0.0
    asr_backend_submit_s: float = 0.0
    asr_backend_result_collect_s: float = 0.0
    asr_backend_artifact_get_s: float = 0.0
    asr_backend_srt_parse_s: float = 0.0
    asr_backend_outside_pool_s: float = 0.0


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
    asr_transcribe_s: float = 0.0
    asr_load_audio_s: float = 0.0
    asr_runner_wall_s: float = 0.0
    asr_pool_wall_s: float = 0.0
    asr_pool_ingest_s: float = 0.0
    asr_pool_ingest_body_read_s: float = 0.0
    asr_pool_ingest_multipart_parse_s: float = 0.0
    asr_pool_ingest_audio_write_s: float = 0.0
    asr_pool_ingest_submit_enqueue_s: float = 0.0
    asr_pool_queue_wait_s: float = 0.0
    asr_pool_outside_runner_s: float = 0.0
    asr_backend_wall_s: float = 0.0
    asr_backend_wav_write_s: float = 0.0
    asr_backend_submit_s: float = 0.0
    asr_backend_result_collect_s: float = 0.0
    asr_backend_artifact_get_s: float = 0.0
    asr_backend_srt_parse_s: float = 0.0
    asr_backend_outside_pool_s: float = 0.0


def copy_pc_events(events: list[dict[str, str]]) -> list[dict[str, str]]:
    return [dict(event) for event in events]


def count_commit_results(rows: list[dict[str, Any]], *, state: str) -> int:
    return max(0, sum(1 for row in rows if str(row.get("state") or "") == state))


def merge_live_commit_row(existing: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
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


def materialize_live_final_segments(
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
        idx = int(max(0, int(row.get("chunk_index") or 0)))
        row_t0 = int(max(0, int(row.get("t0_ms") or 0)))
        row_t1 = int(max(row_t0, int(row.get("t1_ms") or row_t0)))
        appended_segments.append(
            {
                "segment_id": f"c{idx:04d}",
                "text": row_text,
                "t0_ms": row_t0,
                "t1_ms": row_t1,
                "speaker": "",
            }
        )
    return appended_segments
