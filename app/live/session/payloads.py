from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from live.session.metrics import live_commit_rows_debug_metrics
from live.session.state import ClosedSessionArchive, LiveSession


def copy_commit_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def _utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()


def final_covered_ms(
    final_segments: list[dict[str, Any]],
    chunk_rows: list[dict[str, Any]],
) -> int:
    covered_ms = 0
    for seg in final_segments:
        if not isinstance(seg, dict):
            continue
        try:
            t1 = int(seg.get("t1_ms") or 0)
        except Exception:
            t1 = 0
        if t1 > covered_ms:
            covered_ms = t1
    if covered_ms > 0:
        return int(covered_ms)
    for row in chunk_rows:
        if str(row.get("state") or "") != "ready":
            continue
        try:
            t1 = int(row.get("t1_ms") or 0)
        except Exception:
            t1 = 0
        if t1 > covered_ms:
            covered_ms = t1
    return int(max(0, covered_ms))


def build_engine_runtime_payload(
    *,
    recording_duration_ms: int,
    final_covered_ms_value: int,
    live_engine_runtime: dict[str, Any],
) -> dict[str, Any]:
    engine_runtime = {
        "mode": "single_lane",
        "preview_source": "uncommitted_preview",
        "uncommitted_audio_ms": int(max(0, int(recording_duration_ms) - int(final_covered_ms_value))),
    }
    extra_engine_runtime = dict(live_engine_runtime or {})
    if extra_engine_runtime:
        engine_runtime["engine_state"] = extra_engine_runtime
    return engine_runtime


def build_live_result_payload_from_parts(
    *,
    session_id: str,
    source: str,
    live_engine: str,
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
    asr_transcribe_s: float,
    asr_load_audio_s: float,
    asr_runner_wall_s: float,
    asr_pool_wall_s: float,
    asr_pool_ingest_s: float,
    asr_pool_queue_wait_s: float,
    asr_pool_outside_runner_s: float,
    asr_backend_wall_s: float,
    asr_backend_wav_write_s: float,
    asr_backend_submit_s: float,
    asr_backend_result_collect_s: float,
    asr_backend_artifact_get_s: float,
    asr_backend_srt_parse_s: float,
    asr_backend_outside_pool_s: float,
    recording_state: str = "",
    close_reason: str = "",
    state: str = "",
) -> dict[str, Any]:
    chunks = copy_commit_rows(commit_results)
    chunk_debug = live_commit_rows_debug_metrics(chunks)
    covered_ms = final_covered_ms(final_segments, chunks)
    engine_runtime = build_engine_runtime_payload(
        recording_duration_ms=recording_duration_ms,
        final_covered_ms_value=covered_ms,
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
        "final_segments": copy_commit_rows(final_segments),
        "final_segments_count": len(final_segments),
        "final_covered_ms": int(max(0, covered_ms)),
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
            "updated_at_utc": _utc_iso(preview_updated_unix) if float(preview_updated_unix or 0.0) > 0 else "",
        },
        "engine_runtime": engine_runtime,
        "fixture_id": str(fixture_id or ""),
        "fixture_version": str(fixture_version or ""),
        "fixture_test_mode": str(fixture_test_mode or ""),
        "asr_language": str(asr_language or ""),
        "asr_transcribe_s": round(float(max(0.0, asr_transcribe_s)), 3),
        "asr_load_audio_s": round(float(max(0.0, asr_load_audio_s)), 3),
        "asr_runner_wall_s": round(float(max(0.0, asr_runner_wall_s)), 3),
        "asr_pool_wall_s": round(float(max(0.0, asr_pool_wall_s)), 3),
        "asr_pool_ingest_s": round(float(max(0.0, asr_pool_ingest_s)), 3),
        "asr_pool_queue_wait_s": round(float(max(0.0, asr_pool_queue_wait_s)), 3),
        "asr_pool_outside_runner_s": round(float(max(0.0, asr_pool_outside_runner_s)), 3),
        "asr_backend_wall_s": round(float(max(0.0, asr_backend_wall_s)), 3),
        "asr_backend_wav_write_s": round(float(max(0.0, asr_backend_wav_write_s)), 3),
        "asr_backend_submit_s": round(float(max(0.0, asr_backend_submit_s)), 3),
        "asr_backend_result_collect_s": round(float(max(0.0, asr_backend_result_collect_s)), 3),
        "asr_backend_artifact_get_s": round(float(max(0.0, asr_backend_artifact_get_s)), 3),
        "asr_backend_srt_parse_s": round(float(max(0.0, asr_backend_srt_parse_s)), 3),
        "asr_backend_outside_pool_s": round(float(max(0.0, asr_backend_outside_pool_s)), 3),
    }
    if source == "active":
        payload["state"] = str(state or "")
        payload["recording_state"] = str(recording_state or "")
    else:
        payload["close_reason"] = str(close_reason or "")
    return payload


def build_live_session_payload(
    sess: LiveSession,
    *,
    stats_log_path: str,
    now_unix: float,
    now_mono: float,
) -> dict[str, Any]:
    age_s = max(0.0, float(now_mono - sess.created_monotonic))
    ttl_remaining = max(0.0, float(sess.expires_unix - now_unix))
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
        "stats_log_path": str(stats_log_path),
    }


def build_archive_payload(arc: ClosedSessionArchive) -> dict[str, Any]:
    return {
        "session_id": arc.session_id,
        "live_engine": str(arc.live_engine or "rolling_context"),
        "close_reason": arc.close_reason,
        "closed_at_utc": _utc_iso(arc.closed_unix),
        "expires_at_utc": _utc_iso(arc.expires_unix),
        "transcript_revision": int(arc.transcript_revision),
        "final_segments": copy_commit_rows(arc.final_segments),
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


def build_live_result_payload(sess: LiveSession) -> dict[str, Any]:
    return build_live_result_payload_from_parts(
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
        asr_transcribe_s=float(max(0.0, sess.asr_transcribe_s)),
        asr_load_audio_s=float(max(0.0, sess.asr_load_audio_s)),
        asr_runner_wall_s=float(max(0.0, sess.asr_runner_wall_s)),
        asr_pool_wall_s=float(max(0.0, sess.asr_pool_wall_s)),
        asr_pool_ingest_s=float(max(0.0, sess.asr_pool_ingest_s)),
        asr_pool_queue_wait_s=float(max(0.0, sess.asr_pool_queue_wait_s)),
        asr_pool_outside_runner_s=float(max(0.0, sess.asr_pool_outside_runner_s)),
        asr_backend_wall_s=float(max(0.0, sess.asr_backend_wall_s)),
        asr_backend_wav_write_s=float(max(0.0, sess.asr_backend_wav_write_s)),
        asr_backend_submit_s=float(max(0.0, sess.asr_backend_submit_s)),
        asr_backend_result_collect_s=float(max(0.0, sess.asr_backend_result_collect_s)),
        asr_backend_artifact_get_s=float(max(0.0, sess.asr_backend_artifact_get_s)),
        asr_backend_srt_parse_s=float(max(0.0, sess.asr_backend_srt_parse_s)),
        asr_backend_outside_pool_s=float(max(0.0, sess.asr_backend_outside_pool_s)),
    )


def build_live_archive_result_payload(arc: ClosedSessionArchive) -> dict[str, Any]:
    final_segments_src = arc.live_final_segments or arc.final_segments
    return build_live_result_payload_from_parts(
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
        asr_transcribe_s=float(max(0.0, arc.asr_transcribe_s)),
        asr_load_audio_s=float(max(0.0, arc.asr_load_audio_s)),
        asr_runner_wall_s=float(max(0.0, arc.asr_runner_wall_s)),
        asr_pool_wall_s=float(max(0.0, arc.asr_pool_wall_s)),
        asr_pool_ingest_s=float(max(0.0, arc.asr_pool_ingest_s)),
        asr_pool_queue_wait_s=float(max(0.0, arc.asr_pool_queue_wait_s)),
        asr_pool_outside_runner_s=float(max(0.0, arc.asr_pool_outside_runner_s)),
        asr_backend_wall_s=float(max(0.0, arc.asr_backend_wall_s)),
        asr_backend_wav_write_s=float(max(0.0, arc.asr_backend_wav_write_s)),
        asr_backend_submit_s=float(max(0.0, arc.asr_backend_submit_s)),
        asr_backend_result_collect_s=float(max(0.0, arc.asr_backend_result_collect_s)),
        asr_backend_artifact_get_s=float(max(0.0, arc.asr_backend_artifact_get_s)),
        asr_backend_srt_parse_s=float(max(0.0, arc.asr_backend_srt_parse_s)),
        asr_backend_outside_pool_s=float(max(0.0, arc.asr_backend_outside_pool_s)),
    )
