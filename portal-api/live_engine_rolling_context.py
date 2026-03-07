from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Mapping

from fastapi import WebSocket, WebSocketDisconnect, status

from live_chunk_transcribe import LiveChunkBatchBridge
from live_protocol import (
    control_ack_event,
    ended_event,
    error_event,
    parse_client_message,
    pong_event,
    ready_event,
    stats_event,
)
from live_recordings import LiveWavRecorder


def _cfg(config: Mapping[str, Any], key: str) -> Any:
    if key not in config:
        raise RuntimeError(f"missing_live_engine_config:{key}")
    return config[key]


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _normalize_optional_language(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _ms_to_byte_offset(ms: int, *, bytes_per_second: int, sample_width_bytes: int) -> int:
    raw = int(round((max(0.0, float(ms)) / 1000.0) * float(max(1, bytes_per_second))))
    align = int(max(1, sample_width_bytes))
    if (raw % align) != 0:
        raw -= raw % align
    return int(max(0, raw))


def _bytes_to_ms(byte_count: int, *, bytes_per_second: int, sample_width_bytes: int) -> int:
    aligned = int(max(0, int(byte_count)))
    align = int(max(1, sample_width_bytes))
    if (aligned % align) != 0:
        aligned -= aligned % align
    return int((aligned * 1000) // max(1, int(bytes_per_second)))


async def run_live_session_ws_rolling_context(
    session_id: str,
    websocket: WebSocket,
    *,
    live_sessions: Any,
    rooted_path_cb: Callable[[str], str],
    config: Mapping[str, Any],
) -> None:
    LIVE_SESSIONS = live_sessions
    _rooted_path = rooted_path_cb

    LIVE_ENGINE = str(_cfg(config, "LIVE_ENGINE"))
    LIVE_AUDIO_SAMPLE_RATE_HZ = int(_cfg(config, "LIVE_AUDIO_SAMPLE_RATE_HZ"))
    LIVE_AUDIO_CHANNELS = int(_cfg(config, "LIVE_AUDIO_CHANNELS"))
    LIVE_AUDIO_SAMPLE_WIDTH_BYTES = int(_cfg(config, "LIVE_AUDIO_SAMPLE_WIDTH_BYTES"))
    LIVE_AUDIO_BYTES_PER_SECOND = int(_cfg(config, "LIVE_AUDIO_BYTES_PER_SECOND"))
    LIVE_DRAIN_WAIT_S = float(_cfg(config, "LIVE_DRAIN_WAIT_S"))
    LIVE_POST_CLOSE_WAIT_S = float(_cfg(config, "LIVE_POST_CLOSE_WAIT_S"))
    LIVE_ASR_LANGUAGE = _normalize_optional_language(_cfg(config, "LIVE_ASR_LANGUAGE"))
    LIVE_ROLLING_POLL_INTERVAL_MS = int(_cfg(config, "LIVE_ROLLING_POLL_INTERVAL_MS"))
    LIVE_ROLLING_MIN_INFER_AUDIO_MS = int(_cfg(config, "LIVE_ROLLING_MIN_INFER_AUDIO_MS"))
    LIVE_ROLLING_SINGLE_COMMIT_MIN_MS = int(_cfg(config, "LIVE_ROLLING_SINGLE_COMMIT_MIN_MS"))
    LIVE_ROLLING_FORCE_COMMIT_REPEATS = int(_cfg(config, "LIVE_ROLLING_FORCE_COMMIT_REPEATS"))
    LIVE_ROLLING_MAX_UNCOMMITTED_MS = int(_cfg(config, "LIVE_ROLLING_MAX_UNCOMMITTED_MS"))
    LIVE_ROLLING_HARD_CLIP_KEEP_TAIL_MS = int(_cfg(config, "LIVE_ROLLING_HARD_CLIP_KEEP_TAIL_MS"))
    LIVE_ROLLING_MAX_DECODE_WINDOW_MS = int(_cfg(config, "LIVE_ROLLING_MAX_DECODE_WINDOW_MS"))
    LIVE_ROLLING_BUFFER_TRIM_THRESHOLD_MS = int(_cfg(config, "LIVE_ROLLING_BUFFER_TRIM_THRESHOLD_MS"))
    LIVE_ROLLING_BUFFER_TRIM_DROP_MS = int(_cfg(config, "LIVE_ROLLING_BUFFER_TRIM_DROP_MS"))
    LIVE_ROLLING_REQUIRE_SINGLE_INFLIGHT = bool(_cfg(config, "LIVE_ROLLING_REQUIRE_SINGLE_INFLIGHT"))

    poll_interval_s = max(0.1, float(LIVE_ROLLING_POLL_INTERVAL_MS) / 1000.0)
    single_segment_commit_min_ms = int(max(LIVE_ROLLING_MIN_INFER_AUDIO_MS, LIVE_ROLLING_SINGLE_COMMIT_MIN_MS))
    force_commit_repeats = int(max(1, LIVE_ROLLING_FORCE_COMMIT_REPEATS))
    max_decode_window_ms = int(max(LIVE_ROLLING_MIN_INFER_AUDIO_MS, LIVE_ROLLING_MAX_DECODE_WINDOW_MS))
    max_uncommitted_ms = int(max(LIVE_ROLLING_MIN_INFER_AUDIO_MS, LIVE_ROLLING_MAX_UNCOMMITTED_MS))
    if max_uncommitted_ms <= max_decode_window_ms:
        max_uncommitted_ms = int(max_decode_window_ms + LIVE_ROLLING_MIN_INFER_AUDIO_MS)
    hard_clip_keep_tail_ms = int(
        max(LIVE_ROLLING_MIN_INFER_AUDIO_MS, LIVE_ROLLING_HARD_CLIP_KEEP_TAIL_MS, single_segment_commit_min_ms)
    )
    buffer_trim_threshold_ms = int(max(max_decode_window_ms, LIVE_ROLLING_BUFFER_TRIM_THRESHOLD_MS))
    buffer_trim_drop_ms = int(max(LIVE_ROLLING_MIN_INFER_AUDIO_MS, LIVE_ROLLING_BUFFER_TRIM_DROP_MS))

    try:
        LIVE_SESSIONS.open_websocket(session_id)
    except KeyError:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="session_not_found")
        return
    except RuntimeError as e:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason=str(e))
        return

    await websocket.accept()

    stop_reason = "client_disconnected"
    websocket_closed = False
    archived_result = False

    recorder: LiveWavRecorder | None = None
    chunk_bridge: LiveChunkBatchBridge | None = None

    recording_state = "idle"
    recording_path = ""
    recording_bytes = 0
    recording_duration_ms = 0
    finalization_state = "idle"
    shadow_disabled_reason = ""
    recording_finalized = False

    rolling_pcm = bytearray()
    rolling_pcm_base_ms = 0
    rolling_processed_offset_ms = 0
    rolling_commit_index_next = 0
    rolling_chunks_total = 0
    rolling_chunks_done = 0
    rolling_chunks_failed = 0

    rolling_infer_seq_next = 0
    rolling_inflight: dict[str, Any] | None = None
    rolling_last_emit_mono = 0.0
    rolling_last_poll_mono = 0.0

    rolling_call_audit_recent: list[dict[str, Any]] = []
    rolling_call_audit_summary: dict[str, Any] = {
        "calls_done": 0,
        "segments_returned_min": None,
        "segments_returned_max": None,
        "segments_returned_sum": 0,
        "segments_per_s_min": None,
        "segments_per_s_max": None,
        "segments_per_s_sum": 0.0,
        "outcome_counts": {
            "commit": 0,
            "preview_only": 0,
            "empty": 0,
            "error": 0,
        },
    }
    rolling_last_preview_signature = ""
    rolling_same_preview_repeats = 0
    rolling_last_preview_audio_end_ms = -1
    rolling_same_preview_audio_repeats = 0
    rolling_last_preview_text = ""
    rolling_last_preview_audio_end_fallback_ms = 0
    rolling_guardrail_metrics: dict[str, int] = {
        "force_commit_repeats_count": 0,
        "decode_window_cap_count": 0,
        "hard_clip_count": 0,
        "hard_clip_dropped_audio_ms": 0,
        "buffer_trim_count": 0,
        "buffer_trim_dropped_audio_ms": 0,
        "single_inflight_skips": 0,
    }

    async def send_event(payload: dict[str, Any]) -> None:
        out = dict(payload)
        try:
            out["seq"] = LIVE_SESSIONS.next_seq(session_id)
        except KeyError:
            pass
        await websocket.send_json(out)

    def _append_log(kind: str, **fields: Any) -> None:
        try:
            row = {"kind": str(kind)}
            row.update(fields)
            LIVE_SESSIONS.append_stats_log(session_id, row)
        except Exception:
            pass

    def _sync_counts_from_result(result: dict[str, Any]) -> None:
        nonlocal rolling_chunks_total, rolling_chunks_done, rolling_chunks_failed
        rolling_chunks_total = int(max(0, int(result.get("chunks_total") or rolling_chunks_total)))
        rolling_chunks_done = int(max(0, int(result.get("chunks_done") or rolling_chunks_done)))
        rolling_chunks_failed = int(max(0, int(result.get("chunks_failed") or rolling_chunks_failed)))

    def _preview_signature(value: str) -> str:
        return " ".join(str(value or "").strip().lower().split())

    def _engine_runtime_payload() -> dict[str, Any]:
        calls_done = int(max(0, int(rolling_call_audit_summary.get("calls_done") or 0)))
        segs_sum = int(max(0, int(rolling_call_audit_summary.get("segments_returned_sum") or 0)))
        ratio_sum = float(max(0.0, float(rolling_call_audit_summary.get("segments_per_s_sum") or 0.0)))
        avg_segs = (float(segs_sum) / float(calls_done)) if calls_done > 0 else None
        avg_ratio = (ratio_sum / float(calls_done)) if calls_done > 0 else None
        return {
            "inflight": bool(rolling_inflight is not None),
            "recording_duration_ms": int(max(0, recording_duration_ms)),
            "pcm_base_ms": int(max(0, rolling_pcm_base_ms)),
            "processed_offset_ms": int(max(0, rolling_processed_offset_ms)),
            "buffer_audio_ms": int(max(0, int(recording_duration_ms) - int(rolling_pcm_base_ms))),
            "unprocessed_audio_ms": int(max(0, int(recording_duration_ms) - int(rolling_processed_offset_ms))),
            "guardrails": dict(rolling_guardrail_metrics),
            "config": {
                "poll_interval_ms": int(LIVE_ROLLING_POLL_INTERVAL_MS),
                "min_infer_audio_ms": int(LIVE_ROLLING_MIN_INFER_AUDIO_MS),
                "single_segment_commit_min_ms": int(single_segment_commit_min_ms),
                "force_commit_repeats": int(force_commit_repeats),
                "max_uncommitted_ms": int(max_uncommitted_ms),
                "hard_clip_keep_tail_ms": int(hard_clip_keep_tail_ms),
                "max_decode_window_ms": int(max_decode_window_ms),
                "buffer_trim_threshold_ms": int(buffer_trim_threshold_ms),
                "buffer_trim_drop_ms": int(buffer_trim_drop_ms),
                "require_single_inflight": bool(LIVE_ROLLING_REQUIRE_SINGLE_INFLIGHT),
            },
            "call_audit_summary": {
                "calls_done": calls_done,
                "segments_returned_min": rolling_call_audit_summary.get("segments_returned_min"),
                "segments_returned_max": rolling_call_audit_summary.get("segments_returned_max"),
                "segments_returned_avg": avg_segs,
                "segments_per_s_min": rolling_call_audit_summary.get("segments_per_s_min"),
                "segments_per_s_max": rolling_call_audit_summary.get("segments_per_s_max"),
                "segments_per_s_avg": avg_ratio,
                "outcome_counts": dict(rolling_call_audit_summary.get("outcome_counts") or {}),
            },
            "call_audit_recent": list(rolling_call_audit_recent[-50:]),
        }

    def _record_call_audit(
        *,
        seq: int,
        job_id: str,
        call_t0_ms: int,
        call_t1_ms: int,
        segments_returned_count: int,
        outcome: str,
        error: str = "",
    ) -> None:
        safe_t0 = int(max(0, int(call_t0_ms)))
        safe_t1 = int(max(safe_t0, int(call_t1_ms)))
        duration_ms = int(max(0, safe_t1 - safe_t0))
        duration_s = float(duration_ms) / 1000.0
        seg_count = int(max(0, int(segments_returned_count)))
        seg_per_s = (float(seg_count) / duration_s) if duration_s > 0.0 else 0.0
        safe_outcome = str(outcome or "").strip().lower()
        if safe_outcome not in {"commit", "preview_only", "empty", "error"}:
            safe_outcome = "error"
        row = {
            "seq": int(max(0, int(seq))),
            "job_id": str(job_id or ""),
            "call_t0_ms": safe_t0,
            "call_t1_ms": safe_t1,
            "call_duration_ms": duration_ms,
            "segments_returned_count": seg_count,
            "segments_per_s": seg_per_s,
            "outcome": safe_outcome,
        }
        if error:
            row["error"] = str(error)
        rolling_call_audit_recent.append(row)
        if len(rolling_call_audit_recent) > 200:
            del rolling_call_audit_recent[:-200]

        calls_done = int(max(0, int(rolling_call_audit_summary.get("calls_done") or 0))) + 1
        rolling_call_audit_summary["calls_done"] = calls_done
        rolling_call_audit_summary["segments_returned_sum"] = int(
            max(0, int(rolling_call_audit_summary.get("segments_returned_sum") or 0)) + seg_count
        )
        rolling_call_audit_summary["segments_per_s_sum"] = float(
            max(0.0, float(rolling_call_audit_summary.get("segments_per_s_sum") or 0.0)) + seg_per_s
        )
        seg_min = rolling_call_audit_summary.get("segments_returned_min")
        seg_max = rolling_call_audit_summary.get("segments_returned_max")
        ratio_min = rolling_call_audit_summary.get("segments_per_s_min")
        ratio_max = rolling_call_audit_summary.get("segments_per_s_max")
        rolling_call_audit_summary["segments_returned_min"] = (
            seg_count if seg_min is None else int(min(int(seg_min), seg_count))
        )
        rolling_call_audit_summary["segments_returned_max"] = (
            seg_count if seg_max is None else int(max(int(seg_max), seg_count))
        )
        rolling_call_audit_summary["segments_per_s_min"] = (
            seg_per_s if ratio_min is None else float(min(float(ratio_min), seg_per_s))
        )
        rolling_call_audit_summary["segments_per_s_max"] = (
            seg_per_s if ratio_max is None else float(max(float(ratio_max), seg_per_s))
        )
        out_counts = dict(rolling_call_audit_summary.get("outcome_counts") or {})
        out_counts[safe_outcome] = int(max(0, int(out_counts.get(safe_outcome) or 0)) + 1)
        rolling_call_audit_summary["outcome_counts"] = out_counts
        _append_log("rolling_inference_call_done", **row)

    def _drop_pcm_prefix_to_ms(*, target_base_ms: int, reason: str) -> int:
        nonlocal rolling_pcm_base_ms, rolling_processed_offset_ms
        safe_target = int(max(rolling_pcm_base_ms, target_base_ms))
        if safe_target <= rolling_pcm_base_ms:
            return 0
        drop_window_ms = int(max(0, safe_target - rolling_pcm_base_ms))
        drop_bytes = _ms_to_byte_offset(
            drop_window_ms,
            bytes_per_second=LIVE_AUDIO_BYTES_PER_SECOND,
            sample_width_bytes=LIVE_AUDIO_SAMPLE_WIDTH_BYTES,
        )
        drop_bytes = int(max(0, min(drop_bytes, len(rolling_pcm))))
        if drop_bytes <= 0:
            return 0
        del rolling_pcm[:drop_bytes]
        dropped_ms = _bytes_to_ms(
            drop_bytes,
            bytes_per_second=LIVE_AUDIO_BYTES_PER_SECOND,
            sample_width_bytes=LIVE_AUDIO_SAMPLE_WIDTH_BYTES,
        )
        if dropped_ms <= 0:
            dropped_ms = int(max(1, drop_window_ms))
        rolling_pcm_base_ms = int(rolling_pcm_base_ms + dropped_ms)
        if rolling_processed_offset_ms < rolling_pcm_base_ms:
            rolling_processed_offset_ms = int(rolling_pcm_base_ms)
        _append_log(
            "rolling_buffer_drop",
            reason=str(reason or ""),
            dropped_audio_ms=int(max(0, dropped_ms)),
            pcm_base_ms=int(max(0, rolling_pcm_base_ms)),
            pcm_buffer_bytes=int(max(0, len(rolling_pcm))),
        )
        return int(max(0, dropped_ms))

    def _maybe_trim_pcm_buffer() -> None:
        committed_in_buffer_ms = int(max(0, int(rolling_processed_offset_ms) - int(rolling_pcm_base_ms)))
        if committed_in_buffer_ms < buffer_trim_threshold_ms:
            return
        target_base = int(min(rolling_processed_offset_ms, rolling_pcm_base_ms + buffer_trim_drop_ms))
        dropped_ms = _drop_pcm_prefix_to_ms(target_base_ms=target_base, reason="buffer_trim")
        if dropped_ms > 0:
            rolling_guardrail_metrics["buffer_trim_count"] = int(
                max(0, int(rolling_guardrail_metrics.get("buffer_trim_count") or 0)) + 1
            )
            rolling_guardrail_metrics["buffer_trim_dropped_audio_ms"] = int(
                max(0, int(rolling_guardrail_metrics.get("buffer_trim_dropped_audio_ms") or 0)) + int(dropped_ms)
            )

    def _maybe_apply_hard_clip(*, end_ms: int) -> None:
        nonlocal rolling_processed_offset_ms
        unprocessed_ms = int(max(0, int(end_ms) - int(rolling_processed_offset_ms)))
        if unprocessed_ms <= max_uncommitted_ms:
            return
        clip_target_ms = int(max(rolling_processed_offset_ms, int(end_ms) - int(hard_clip_keep_tail_ms)))
        if clip_target_ms <= rolling_processed_offset_ms:
            return
        dropped_uncommitted_ms = int(max(0, clip_target_ms - rolling_processed_offset_ms))
        rolling_processed_offset_ms = int(clip_target_ms)
        dropped_buffer_ms = _drop_pcm_prefix_to_ms(target_base_ms=clip_target_ms, reason="hard_clip")
        rolling_guardrail_metrics["hard_clip_count"] = int(
            max(0, int(rolling_guardrail_metrics.get("hard_clip_count") or 0)) + 1
        )
        rolling_guardrail_metrics["hard_clip_dropped_audio_ms"] = int(
            max(0, int(rolling_guardrail_metrics.get("hard_clip_dropped_audio_ms") or 0))
            + int(max(dropped_uncommitted_ms, dropped_buffer_ms))
        )
        _append_log(
            "rolling_hard_clip_applied",
            unprocessed_audio_ms=int(unprocessed_ms),
            dropped_uncommitted_audio_ms=int(max(0, dropped_uncommitted_ms)),
            keep_tail_ms=int(max(0, hard_clip_keep_tail_ms)),
            processed_offset_ms=int(max(0, rolling_processed_offset_ms)),
        )
        try:
            LIVE_SESSIONS.clear_live_preview(session_id)
        except Exception:
            pass

    def _update_state() -> None:
        try:
            LIVE_SESSIONS.update_live_state(
                session_id,
                recording_state=recording_state,
                recording_path=recording_path,
                recording_bytes=recording_bytes,
                recording_duration_ms=recording_duration_ms,
                chunk_index_next=rolling_commit_index_next,
                chunks_total=rolling_chunks_total,
                chunks_done=rolling_chunks_done,
                chunks_failed=rolling_chunks_failed,
                finalization_state=finalization_state,
                batch_job_id="",
            )
        except Exception:
            pass
        try:
            LIVE_SESSIONS.set_live_engine_runtime(
                session_id,
                runtime=_engine_runtime_payload(),
            )
        except Exception:
            pass
        if str(finalization_state or "").strip().lower() in {"ready", "error", "finalized"}:
            try:
                LIVE_SESSIONS.clear_live_preview(session_id)
            except Exception:
                pass

    def _archive_current_result(*, close_reason: str) -> dict[str, Any]:
        try:
            live_result = LIVE_SESSIONS.live_result_snapshot(session_id)
        except Exception:
            return {}
        if not live_result:
            return {}
        has_content = (
            bool(str(live_result.get("final_text") or "").strip())
            or bool(live_result.get("final_segments"))
            or int(live_result.get("chunks_total") or 0) > 0
            or int(max(0, recording_duration_ms)) > 0
        )
        if not has_content:
            return live_result
        LIVE_SESSIONS.archive_transcript(
            session_id,
            close_reason=str(close_reason or stop_reason or "closed"),
            final_text=str(live_result.get("final_text") or ""),
            final_segments=[
                dict(seg)
                for seg in (live_result.get("final_segments") or [])
                if isinstance(seg, dict)
            ],
            transcript_revision=int(max(0, int(live_result.get("transcript_revision") or 0))),
            recording_path=str(recording_path or ""),
            recording_bytes=int(max(0, recording_bytes)),
            recording_duration_ms=int(max(0, recording_duration_ms)),
            chunks_total=int(max(0, rolling_chunks_total)),
            chunks_done=int(max(0, rolling_chunks_done)),
            chunks_failed=int(max(0, rolling_chunks_failed)),
            finalization_state=str(finalization_state or ""),
            batch_job_id="",
            live_engine=LIVE_ENGINE,
        )
        return live_result

    def _finalize_recording(*, reason: str) -> None:
        nonlocal recording_finalized
        nonlocal recording_state
        nonlocal recording_path
        nonlocal recording_bytes
        nonlocal recording_duration_ms
        nonlocal finalization_state
        nonlocal shadow_disabled_reason

        if recording_finalized:
            return
        finalization_state = "finalizing"
        if recorder is not None:
            try:
                rs = recorder.finalize()
                recording_path = str(rs.wav_path)
                recording_bytes = int(rs.bytes_written)
                recording_duration_ms = int(rs.duration_ms)
                recording_state = "finalized"
                if finalization_state != "error":
                    finalization_state = "recording_finalized"
                _append_log("rolling_recording_finalized", reason=reason, recording=rs.to_dict())
            except Exception as e:
                shadow_disabled_reason = f"recording_finalize_failed:{type(e).__name__}"
                recording_state = "error"
                finalization_state = "error"
                _append_log(
                    "rolling_recording_finalize_error",
                    reason=reason,
                    error=f"{type(e).__name__}: {e}",
                )
        else:
            if finalization_state != "error":
                finalization_state = "idle"
        recording_finalized = True
        _update_state()

    async def _enqueue_inference(*, force: bool = False) -> None:
        nonlocal rolling_inflight
        nonlocal rolling_infer_seq_next
        nonlocal rolling_last_emit_mono
        nonlocal finalization_state

        if chunk_bridge is None:
            return
        now_mono = time.monotonic()
        if rolling_inflight is not None:
            rolling_guardrail_metrics["single_inflight_skips"] = int(
                max(0, int(rolling_guardrail_metrics.get("single_inflight_skips") or 0)) + 1
            )
            return
        if str(recording_state or "") not in {"recording", "finalizing"}:
            return
        if (not force) and ((now_mono - rolling_last_emit_mono) < poll_interval_s):
            return

        end_ms = int(max(0, recording_duration_ms))
        _maybe_apply_hard_clip(end_ms=end_ms)
        if end_ms <= rolling_processed_offset_ms:
            return
        unprocessed_ms = int(max(0, end_ms - rolling_processed_offset_ms))
        if (not force) and (unprocessed_ms < LIVE_ROLLING_MIN_INFER_AUDIO_MS):
            return

        infer_t0_ms = int(max(rolling_processed_offset_ms, rolling_pcm_base_ms))
        infer_t1_ms = int(max(infer_t0_ms, end_ms))
        infer_window_ms = int(max(0, infer_t1_ms - infer_t0_ms))
        if infer_window_ms > max_decode_window_ms:
            infer_t1_ms = int(max(infer_t0_ms, infer_t0_ms + max_decode_window_ms))
            if infer_t1_ms < end_ms:
                rolling_guardrail_metrics["decode_window_cap_count"] = int(
                    max(0, int(rolling_guardrail_metrics.get("decode_window_cap_count") or 0)) + 1
                )
                _append_log(
                    "rolling_decode_window_capped",
                    end_ms=int(end_ms),
                    processed_offset_ms=int(rolling_processed_offset_ms),
                    infer_t0_ms=int(infer_t0_ms),
                    infer_t1_ms=int(infer_t1_ms),
                    max_decode_window_ms=int(max_decode_window_ms),
                )

        start_b = _ms_to_byte_offset(
            int(max(0, infer_t0_ms - rolling_pcm_base_ms)),
            bytes_per_second=LIVE_AUDIO_BYTES_PER_SECOND,
            sample_width_bytes=LIVE_AUDIO_SAMPLE_WIDTH_BYTES,
        )
        end_b = _ms_to_byte_offset(
            int(max(0, infer_t1_ms - rolling_pcm_base_ms)),
            bytes_per_second=LIVE_AUDIO_BYTES_PER_SECOND,
            sample_width_bytes=LIVE_AUDIO_SAMPLE_WIDTH_BYTES,
        )
        end_b = min(end_b, len(rolling_pcm))
        if end_b <= start_b:
            return
        pcm = bytes(rolling_pcm[start_b:end_b])
        if not pcm:
            return

        infer_seq = int(max(0, rolling_infer_seq_next))
        try:
            job = chunk_bridge.enqueue_chunk_pcm16(
                session_id=session_id,
                chunk_index=infer_seq,
                t0_ms=int(infer_t0_ms),
                t1_ms=int(infer_t1_ms),
                pcm16le=pcm,
                language=LIVE_ASR_LANGUAGE,
                live_lane="final",
                preview_seq=infer_seq,
                preview_audio_end_ms=int(infer_t1_ms),
            )
        except Exception as e:
            _append_log(
                "rolling_inference_enqueue_error",
                seq=int(infer_seq),
                t0_ms=int(infer_t0_ms),
                t1_ms=int(infer_t1_ms),
                error=f"{type(e).__name__}: {e}",
            )
            finalization_state = "error"
            _update_state()
            return

        rolling_infer_seq_next = infer_seq + 1
        rolling_last_emit_mono = now_mono
        rolling_inflight = {
            "seq": int(infer_seq),
            "job_id": str(job.job_id),
            "t0_ms": int(infer_t0_ms),
            "t1_ms": int(infer_t1_ms),
            "enqueued_mono": float(now_mono),
        }
        if finalization_state not in {"error", "ready"}:
            finalization_state = "processing_chunks"
        _update_state()
        _append_log(
            "rolling_inference_enqueued",
            seq=int(infer_seq),
            job_id=str(job.job_id),
            t0_ms=int(infer_t0_ms),
            t1_ms=int(infer_t1_ms),
            audio_bytes=len(pcm),
        )

    async def _poll_inference(*, force: bool = False) -> None:
        nonlocal rolling_inflight
        nonlocal rolling_last_poll_mono
        nonlocal rolling_commit_index_next
        nonlocal rolling_processed_offset_ms
        nonlocal rolling_last_preview_signature
        nonlocal rolling_same_preview_repeats
        nonlocal rolling_last_preview_audio_end_ms
        nonlocal rolling_same_preview_audio_repeats
        nonlocal rolling_last_preview_text
        nonlocal rolling_last_preview_audio_end_fallback_ms
        nonlocal finalization_state

        if chunk_bridge is None or rolling_inflight is None:
            return
        now_mono = time.monotonic()
        if (not force) and ((now_mono - rolling_last_poll_mono) < poll_interval_s):
            return
        rolling_last_poll_mono = now_mono

        inflight = dict(rolling_inflight)
        seq = int(inflight.get("seq") or 0)
        job_id = str(inflight.get("job_id") or "")
        t0_ms = int(max(0, int(inflight.get("t0_ms") or 0)))
        t1_ms = int(max(t0_ms, int(inflight.get("t1_ms") or t0_ms)))

        try:
            poll = chunk_bridge.poll_job(job_id, t0_offset_ms=t0_ms)
        except Exception as e:
            _append_log("rolling_inference_poll_error", seq=seq, job_id=job_id, error=f"{type(e).__name__}: {e}")
            return
        if not bool(poll.done):
            return

        if bool(poll.ok):
            raw_segments = poll.segments if isinstance(poll.segments, list) else []
            segments_returned_count = int(len(raw_segments))
            segments = [dict(seg) for seg in (poll.segments or []) if isinstance(seg, dict)]
            segments.sort(key=lambda seg: int(seg.get("t0_ms") or 0))

            commit_segments: list[dict[str, Any]] = []
            preview_text = ""
            preview_audio_end_ms = int(t1_ms)
            single_segment_forced_commit = False
            force_commit_repeats_applied = False
            commit_reason = "rolling_context_commit"
            committed_this_poll = False
            call_outcome = "empty"
            call_error = ""

            if len(segments) >= 2:
                commit_segments = segments[:-1]
                last_seg = dict(segments[-1])
                preview_text = str(last_seg.get("text") or "").strip()
                preview_audio_end_ms = int(max(t0_ms, int(last_seg.get("t1_ms") or t1_ms)))
            elif len(segments) == 1:
                last_seg = dict(segments[0])
                single_text = str(last_seg.get("text") or "").strip()
                single_t0 = int(max(t0_ms, int(last_seg.get("t0_ms") or t0_ms)))
                single_t1 = int(max(single_t0, int(last_seg.get("t1_ms") or t1_ms)))
                single_duration_ms = int(max(0, single_t1 - single_t0))
                infer_window_duration_ms = int(max(0, int(t1_ms) - int(t0_ms)))
                if single_text and max(single_duration_ms, infer_window_duration_ms) >= single_segment_commit_min_ms:
                    commit_segments = [last_seg]
                    single_segment_forced_commit = True
                    preview_text = ""
                    preview_audio_end_ms = int(single_t1)
                else:
                    preview_text = single_text
                    preview_audio_end_ms = int(single_t1)
            else:
                preview_text = str(poll.text or "").strip()
                preview_audio_end_ms = int(t1_ms)

            preview_sig = _preview_signature(preview_text)
            if preview_sig:
                if preview_sig == rolling_last_preview_signature:
                    rolling_same_preview_repeats += 1
                else:
                    rolling_last_preview_signature = preview_sig
                    rolling_same_preview_repeats = 1
                if int(preview_audio_end_ms) == int(rolling_last_preview_audio_end_ms):
                    rolling_same_preview_audio_repeats += 1
                else:
                    rolling_last_preview_audio_end_ms = int(preview_audio_end_ms)
                    rolling_same_preview_audio_repeats = 1
            else:
                rolling_last_preview_signature = ""
                rolling_same_preview_repeats = 0
                rolling_last_preview_audio_end_ms = -1
                rolling_same_preview_audio_repeats = 0

            if (
                preview_sig
                and rolling_same_preview_audio_repeats >= force_commit_repeats
                and segments
            ):
                forced_segments = [dict(seg) for seg in segments if isinstance(seg, dict)]
                if forced_segments:
                    last_seg = dict(forced_segments[-1])
                    last_text = str(last_seg.get("text") or "").strip()
                    if last_text:
                        commit_segments = forced_segments
                        preview_text = ""
                        preview_audio_end_ms = int(max(t0_ms, int(last_seg.get("t1_ms") or t1_ms)))
                        force_commit_repeats_applied = True
                        commit_reason = "rolling_context_force_commit_repeats"
                        rolling_guardrail_metrics["force_commit_repeats_count"] = int(
                            max(0, int(rolling_guardrail_metrics.get("force_commit_repeats_count") or 0)) + 1
                        )
                        _append_log(
                            "rolling_force_commit_repeats_triggered",
                            seq=seq,
                            job_id=job_id,
                            repeats=int(max(0, rolling_same_preview_audio_repeats)),
                            preview_chars=len(last_text),
                            force_commit_repeats=int(force_commit_repeats),
                        )

            if commit_segments:
                commit_text = "\n".join(
                    str(seg.get("text") or "").strip() for seg in commit_segments if str(seg.get("text") or "").strip()
                ).strip()
                if commit_text:
                    commit_t0_ms = int(max(0, rolling_processed_offset_ms))
                    commit_t1_ms = int(max(commit_t0_ms, int(commit_segments[-1].get("t1_ms") or commit_t0_ms)))
                    if single_segment_forced_commit or force_commit_repeats_applied:
                        # For single-segment/forced commits, advance to polled window end to prevent
                        # endless re-enqueue of the exact same rolling window.
                        commit_t1_ms = int(max(commit_t1_ms, t1_ms))
                    normalized_commit_segments: list[dict[str, Any]] = []
                    for raw_seg in commit_segments:
                        if not isinstance(raw_seg, dict):
                            continue
                        seg_text = str(raw_seg.get("text") or "").strip()
                        if not seg_text:
                            continue
                        seg_t0 = int(max(commit_t0_ms, int(raw_seg.get("t0_ms") or commit_t0_ms)))
                        seg_t1 = int(max(seg_t0, int(raw_seg.get("t1_ms") or seg_t0)))
                        normalized_commit_segments.append(
                            {
                                "segment_id": str(raw_seg.get("segment_id") or ""),
                                "text": seg_text,
                                "t0_ms": int(seg_t0),
                                "t1_ms": int(seg_t1),
                            }
                        )
                    if not normalized_commit_segments:
                        normalized_commit_segments = [
                            {
                                "segment_id": f"s{int(max(0, rolling_commit_index_next)) + 1:04d}",
                                "text": commit_text,
                                "t0_ms": int(commit_t0_ms),
                                "t1_ms": int(commit_t1_ms),
                            }
                        ]
                    else:
                        normalized_commit_segments[0]["t0_ms"] = int(commit_t0_ms)
                        normalized_commit_segments[-1]["t1_ms"] = int(max(commit_t1_ms, normalized_commit_segments[-1]["t1_ms"]))
                    status_obj = dict(poll.status or {})
                    asr_pipeline_time_s = _safe_float(status_obj.get("asr_timing_whisperx_total_s"))
                    asr_transcribe_time_s = _safe_float(status_obj.get("asr_timing_whisperx_transcribe_s"))
                    try:
                        result = LIVE_SESSIONS.record_live_commit(
                            session_id,
                            chunk_index=int(max(0, rolling_commit_index_next)),
                            t0_ms=int(commit_t0_ms),
                            t1_ms=int(commit_t1_ms),
                            text=commit_text,
                            segments=normalized_commit_segments,
                            state="ready",
                            error="",
                            reason=str(commit_reason),
                            chunk_duration_ms=int(max(0, commit_t1_ms - commit_t0_ms)),
                            asr_pipeline_time_s=asr_pipeline_time_s,
                            asr_transcribe_time_s=asr_transcribe_time_s,
                        )
                        _sync_counts_from_result(result)
                        rolling_commit_index_next += 1
                        rolling_processed_offset_ms = int(max(rolling_processed_offset_ms, commit_t1_ms))
                        committed_this_poll = True
                        call_outcome = "commit"
                        _maybe_trim_pcm_buffer()
                        if single_segment_forced_commit:
                            _append_log(
                                "rolling_single_segment_forced_commit",
                                seq=seq,
                                job_id=job_id,
                                commit_t0_ms=int(commit_t0_ms),
                                commit_t1_ms=int(commit_t1_ms),
                                commit_duration_ms=int(max(0, commit_t1_ms - commit_t0_ms)),
                            )
                        if force_commit_repeats_applied:
                            rolling_same_preview_repeats = 0
                            rolling_last_preview_signature = ""
                            rolling_last_preview_audio_end_ms = -1
                            rolling_same_preview_audio_repeats = 0
                            rolling_last_preview_text = ""
                            rolling_last_preview_audio_end_fallback_ms = 0
                    except Exception as e:
                        call_outcome = "error"
                        call_error = f"{type(e).__name__}: {e}"
                        finalization_state = "error"
                        _append_log(
                            "rolling_commit_store_error",
                            seq=seq,
                            job_id=job_id,
                            error=f"{type(e).__name__}: {e}",
                        )

            if preview_text:
                if call_outcome != "commit":
                    call_outcome = "preview_only"
                rolling_last_preview_text = str(preview_text or "")
                rolling_last_preview_audio_end_fallback_ms = int(max(0, preview_audio_end_ms))
                try:
                    LIVE_SESSIONS.update_live_preview(
                        session_id,
                        text=preview_text,
                        preview_seq=int(max(0, seq)),
                        audio_end_ms=int(max(0, preview_audio_end_ms)),
                        append_to_existing=False,
                    )
                except Exception:
                    pass
                _append_log(
                    "rolling_preview_ready",
                    seq=seq,
                    job_id=job_id,
                    preview_chars=len(preview_text),
                    preview_audio_end_ms=int(max(0, preview_audio_end_ms)),
                )
            else:
                if call_outcome not in {"commit", "error"}:
                    call_outcome = "empty"
                if committed_this_poll:
                    # Only clear preview when this poll actually committed content.
                    # Otherwise keep previous preview to avoid UX flicker/vanishing text.
                    rolling_last_preview_signature = ""
                    rolling_same_preview_repeats = 0
                    rolling_last_preview_audio_end_ms = -1
                    rolling_same_preview_audio_repeats = 0
                    try:
                        LIVE_SESSIONS.clear_live_preview(
                            session_id,
                            max_seq=int(max(0, seq)),
                        )
                    except Exception:
                        pass
            _record_call_audit(
                seq=seq,
                job_id=job_id,
                call_t0_ms=t0_ms,
                call_t1_ms=t1_ms,
                segments_returned_count=segments_returned_count,
                outcome=call_outcome,
                error=call_error,
            )
        else:
            err = str(poll.error or "asr_error")
            try:
                result = LIVE_SESSIONS.record_live_commit(
                    session_id,
                    chunk_index=int(max(0, rolling_commit_index_next)),
                    t0_ms=int(t0_ms),
                    t1_ms=int(t1_ms),
                    text="",
                    segments=[],
                    state="error",
                    error=err,
                    reason="rolling_context_error",
                )
                _sync_counts_from_result(result)
                rolling_commit_index_next += 1
            except Exception:
                pass
            rolling_processed_offset_ms = int(max(rolling_processed_offset_ms, t1_ms))
            _maybe_trim_pcm_buffer()
            finalization_state = "error"
            _append_log("rolling_inference_error", seq=seq, job_id=job_id, error=err)
            _record_call_audit(
                seq=seq,
                job_id=job_id,
                call_t0_ms=t0_ms,
                call_t1_ms=t1_ms,
                segments_returned_count=0,
                outcome="error",
                error=err,
            )

        rolling_inflight = None
        if finalization_state not in {"error", "ready"}:
            finalization_state = "recording" if str(recording_state or "") == "recording" else "processing_chunks"
        _update_state()

    async def _process_rolling(*, force_poll: bool = False, force_emit: bool = False) -> None:
        await _poll_inference(force=force_poll)
        await _enqueue_inference(force=force_emit)

    async def _drain_inflight_only(*, force_poll: bool = True) -> None:
        await _poll_inference(force=force_poll)

    def _commit_preview_tail_if_needed() -> None:
        nonlocal rolling_commit_index_next
        nonlocal rolling_processed_offset_ms
        nonlocal rolling_last_preview_signature
        nonlocal rolling_same_preview_repeats
        nonlocal rolling_last_preview_audio_end_ms
        nonlocal rolling_same_preview_audio_repeats
        nonlocal rolling_last_preview_text
        nonlocal rolling_last_preview_audio_end_fallback_ms
        if recording_duration_ms <= rolling_processed_offset_ms:
            return
        try:
            result = LIVE_SESSIONS.live_result_snapshot(session_id)
        except Exception:
            return
        preview = result.get("preview") or {}
        preview_text = str(preview.get("text") or "").strip()
        if not preview_text:
            preview_text = str(rolling_last_preview_text or "").strip()
        if not preview_text:
            return

        preview_audio_end_ms = int(max(0, int(preview.get("audio_end_ms") or 0)))
        if preview_audio_end_ms <= 0:
            preview_audio_end_ms = int(max(0, rolling_last_preview_audio_end_fallback_ms))

        commit_t0 = int(max(0, rolling_processed_offset_ms))
        commit_t1 = int(
            max(
                commit_t0,
                int(preview_audio_end_ms),
                int(max(0, recording_duration_ms)),
            )
        )
        if commit_t1 <= commit_t0:
            commit_t1 = int(max(commit_t0 + 1, recording_duration_ms))
        seg = {
            "segment_id": f"s{int(max(0, rolling_commit_index_next)) + 1:04d}",
            "text": preview_text,
            "t0_ms": int(commit_t0),
            "t1_ms": int(commit_t1),
        }
        try:
            stored = LIVE_SESSIONS.record_live_commit(
                session_id,
                chunk_index=int(max(0, rolling_commit_index_next)),
                t0_ms=int(commit_t0),
                t1_ms=int(commit_t1),
                text=preview_text,
                segments=[seg],
                state="ready",
                error="",
                reason="rolling_context_tail_preview_commit",
                chunk_duration_ms=int(max(0, commit_t1 - commit_t0)),
            )
            _sync_counts_from_result(stored)
            rolling_commit_index_next += 1
            rolling_processed_offset_ms = int(max(rolling_processed_offset_ms, commit_t1))
            _maybe_trim_pcm_buffer()
            rolling_last_preview_signature = ""
            rolling_same_preview_repeats = 0
            rolling_last_preview_audio_end_ms = -1
            rolling_same_preview_audio_repeats = 0
            rolling_last_preview_text = ""
            rolling_last_preview_audio_end_fallback_ms = 0
            try:
                LIVE_SESSIONS.clear_live_preview(session_id)
            except Exception:
                pass
            _append_log("rolling_tail_preview_committed", t0_ms=commit_t0, t1_ms=commit_t1, chars=len(preview_text))
        except Exception:
            pass

    try:
        ready_payload = ready_event(
            session_id,
            message="Live websocket connected. Send binary PCM16 frames and JSON controls.",
            engine="rolling_context",
        )
        ready_payload["live_engine"] = LIVE_ENGINE
        await send_event(ready_payload)

        try:
            recorder = LiveWavRecorder(
                session_id=session_id,
                sample_rate_hz=LIVE_AUDIO_SAMPLE_RATE_HZ,
                channels=LIVE_AUDIO_CHANNELS,
            )
            rec_snap = recorder.start()
            chunk_bridge = LiveChunkBatchBridge(
                sample_rate_hz=LIVE_AUDIO_SAMPLE_RATE_HZ,
                channels=LIVE_AUDIO_CHANNELS,
                language=LIVE_ASR_LANGUAGE,
            )
            recording_state = "recording"
            recording_path = str(rec_snap.wav_path)
            recording_bytes = int(rec_snap.bytes_written)
            recording_duration_ms = int(rec_snap.duration_ms)
            finalization_state = "recording"
            _update_state()
            _append_log(
                "rolling_context_started",
                recording=rec_snap.to_dict(),
                config={
                    "poll_interval_ms": int(LIVE_ROLLING_POLL_INTERVAL_MS),
                    "min_infer_audio_ms": int(LIVE_ROLLING_MIN_INFER_AUDIO_MS),
                    "single_segment_commit_min_ms": int(single_segment_commit_min_ms),
                    "force_commit_repeats": int(force_commit_repeats),
                    "max_uncommitted_ms": int(max_uncommitted_ms),
                    "hard_clip_keep_tail_ms": int(hard_clip_keep_tail_ms),
                    "max_decode_window_ms": int(max_decode_window_ms),
                    "buffer_trim_threshold_ms": int(buffer_trim_threshold_ms),
                    "buffer_trim_drop_ms": int(buffer_trim_drop_ms),
                    "require_single_inflight": bool(LIVE_ROLLING_REQUIRE_SINGLE_INFLIGHT),
                    "language": LIVE_ASR_LANGUAGE,
                },
            )
        except Exception as e:
            recorder = None
            chunk_bridge = None
            recording_state = "error"
            finalization_state = "error"
            shadow_disabled_reason = f"rolling_init_failed:{type(e).__name__}"
            _update_state()
            _append_log("rolling_context_init_error", error=f"{type(e).__name__}: {e}")

        while True:
            incoming = await websocket.receive()

            if incoming.get("type") == "websocket.disconnect":
                stop_reason = "client_disconnected"
                break

            raw_bytes = incoming.get("bytes")
            if raw_bytes is not None:
                snapshot = LIVE_SESSIONS.record_audio(session_id, byte_count=len(raw_bytes))
                raw = bytes(raw_bytes or b"")
                if (len(raw) % LIVE_AUDIO_SAMPLE_WIDTH_BYTES) != 0:
                    raw = raw[: len(raw) - (len(raw) % LIVE_AUDIO_SAMPLE_WIDTH_BYTES)]
                if recorder is not None:
                    try:
                        rec_snap = recorder.append_pcm16(raw)
                        recording_bytes = int(rec_snap.bytes_written)
                        recording_duration_ms = int(rec_snap.duration_ms)
                        recording_path = str(rec_snap.wav_path)
                    except Exception as e:
                        shadow_disabled_reason = f"recording_append_failed:{type(e).__name__}"
                        recording_state = "error"
                        finalization_state = "error"
                        _append_log(
                            "rolling_recording_append_error",
                            error=f"{type(e).__name__}: {e}",
                            at_frame=int(snapshot.get("frames_received") or 0),
                        )
                        try:
                            recorder.abort()
                        except Exception:
                            pass
                        recorder = None
                        _update_state()
                if raw:
                    rolling_pcm.extend(raw)

                await _process_rolling(force_poll=False, force_emit=False)

                should_emit_stats = snapshot["frames_received"] == 1 or (snapshot["frames_received"] % 50) == 0
                if should_emit_stats:
                    stats_payload = stats_event(
                        session_id,
                        bytes_received=snapshot["bytes_received"],
                        frames_received=snapshot["frames_received"],
                        controls_received=snapshot["controls_received"],
                        uptime_s=snapshot["age_s"],
                        live_engine=LIVE_ENGINE,
                        live_mode="rolling_context",
                        live_recording_state=str(recording_state or ""),
                        live_recording_bytes=int(max(0, recording_bytes)),
                        live_recording_duration_ms=int(max(0, recording_duration_ms)),
                        live_commit_index_next=int(max(0, rolling_commit_index_next)),
                        live_commits_total=int(max(0, rolling_chunks_total)),
                        live_commits_done=int(max(0, rolling_chunks_done)),
                        live_commits_failed=int(max(0, rolling_chunks_failed)),
                        live_finalization_state=str(finalization_state or ""),
                        live_jobs_enabled=True,
                        live_jobs_pending=(1 if rolling_inflight is not None else 0),
                        live_shadow_disabled_reason=str(shadow_disabled_reason or ""),
                        live_inflight=bool(rolling_inflight is not None),
                        rolling_guardrails=dict(rolling_guardrail_metrics),
                        rolling_unprocessed_audio_ms=int(max(0, int(recording_duration_ms) - int(rolling_processed_offset_ms))),
                        rolling_pcm_base_ms=int(max(0, rolling_pcm_base_ms)),
                    )
                    try:
                        LIVE_SESSIONS.append_stats_log(session_id, stats_payload)
                    except Exception:
                        pass
                    await send_event(stats_payload)
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
                await send_event(error_event(session_id, code=parse_err, message="Invalid control message."))
                continue

            LIVE_SESSIONS.record_control(session_id)

            if control_type == "ping":
                await send_event(pong_event(session_id))
                continue

            if control_type == "start":
                snapshot = LIVE_SESSIONS.mark_state(session_id, state="listening")
                recording_state = "recording"
                if finalization_state not in {"error", "ready"}:
                    finalization_state = "recording"
                _update_state()
                await send_event(control_ack_event(session_id, control_type="start", state=snapshot["state"]))
                continue

            if control_type == "pause":
                snapshot = LIVE_SESSIONS.mark_state(session_id, state="paused")
                recording_state = "paused"
                _update_state()
                await send_event(control_ack_event(session_id, control_type="pause", state=snapshot["state"]))
                continue

            if control_type == "resume":
                snapshot = LIVE_SESSIONS.mark_state(session_id, state="listening")
                recording_state = "recording"
                if finalization_state not in {"error", "ready"}:
                    finalization_state = "recording"
                _update_state()
                await send_event(control_ack_event(session_id, control_type="resume", state=snapshot["state"]))
                continue

            if control_type == "stop":
                stop_reason = "client_stop"
                live_result: dict[str, Any] = {}
                if recording_state in {"recording", "paused"}:
                    recording_state = "finalizing"
                if finalization_state not in {"error", "ready"}:
                    finalization_state = "finalizing"
                _update_state()
                # Enqueue maximaal één laatste rolling inferentie en drain daarna alleen polls.
                await _process_rolling(force_poll=True, force_emit=True)
                wait_deadline = time.monotonic() + max(0.0, LIVE_DRAIN_WAIT_S)
                while time.monotonic() < wait_deadline:
                    await _drain_inflight_only(force_poll=True)
                    if rolling_inflight is None:
                        break
                    await asyncio.sleep(min(0.1, poll_interval_s))

                _finalize_recording(reason=stop_reason)
                await _drain_inflight_only(force_poll=True)
                _commit_preview_tail_if_needed()
                if finalization_state != "error":
                    finalization_state = "ready"
                _update_state()
                try:
                    live_result = _archive_current_result(close_reason=stop_reason)
                    archived_result = True
                except Exception:
                    live_result = {}

                await send_event(
                    ended_event(
                        session_id,
                        reason=stop_reason,
                        transcript_revision=int(max(0, int(live_result.get("transcript_revision") or 0))),
                        final_segments_count=len(live_result.get("final_segments") or []),
                        final_text=str(live_result.get("final_text") or ""),
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
        _finalize_recording(reason=stop_reason)
        await _drain_inflight_only(force_poll=True)
        if stop_reason == "client_stop":
            # The explicit stop path already drains in-flight work before emitting ended.
            wait_timeout = 0.0
        else:
            wait_timeout = LIVE_DRAIN_WAIT_S
        wait_deadline = time.monotonic() + max(0.0, float(wait_timeout))
        while time.monotonic() < wait_deadline:
            await _drain_inflight_only(force_poll=True)
            remaining_ms = int(max(0, recording_duration_ms - rolling_processed_offset_ms))
            if rolling_inflight is None and remaining_ms < LIVE_ROLLING_MIN_INFER_AUDIO_MS:
                break
            await asyncio.sleep(min(0.1, poll_interval_s))

        _commit_preview_tail_if_needed()
        if finalization_state not in {"error", "ready"}:
            finalization_state = "ready"
        _update_state()

        if not archived_result:
            try:
                _archive_current_result(close_reason=stop_reason)
            except Exception:
                pass

        LIVE_SESSIONS.close_session(session_id, reason=stop_reason)
        if recorder is not None and not recording_finalized:
            try:
                recorder.abort()
            except Exception:
                pass
