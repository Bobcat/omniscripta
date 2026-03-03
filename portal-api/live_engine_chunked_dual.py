from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Mapping

from fastapi import WebSocket, WebSocketDisconnect, status

from live_chunk_transcribe import LiveChunkBatchBridge
from live_chunker import LiveAudioChunker, LiveChunkerConfig
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


async def run_live_session_ws_chunked_dual(
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
    LIVE_SEMILIVE_CHUNK_BATCH_SHADOW = bool(_cfg(config, "LIVE_SEMILIVE_CHUNK_BATCH_SHADOW"))
    LIVE_SEMILIVE_CHUNK_POLL_INTERVAL_S = float(_cfg(config, "LIVE_SEMILIVE_CHUNK_POLL_INTERVAL_S"))
    LIVE_SEMILIVE_CHUNK_STOP_WAIT_S = float(_cfg(config, "LIVE_SEMILIVE_CHUNK_STOP_WAIT_S"))
    LIVE_SEMILIVE_CHUNK_POST_CLOSE_WAIT_S = float(_cfg(config, "LIVE_SEMILIVE_CHUNK_POST_CLOSE_WAIT_S"))
    LIVE_SEMILIVE_CHUNK_LANGUAGE = str(_cfg(config, "LIVE_SEMILIVE_CHUNK_LANGUAGE"))
    LIVE_SEMILIVE_CHUNK_ENERGY_THRESHOLD = int(_cfg(config, "LIVE_SEMILIVE_CHUNK_ENERGY_THRESHOLD"))
    LIVE_SEMILIVE_CHUNK_SILENCE_MS = int(_cfg(config, "LIVE_SEMILIVE_CHUNK_SILENCE_MS"))
    LIVE_SEMILIVE_CHUNK_MAX_MS = int(_cfg(config, "LIVE_SEMILIVE_CHUNK_MAX_MS"))
    LIVE_SEMILIVE_CHUNK_MIN_MS = int(_cfg(config, "LIVE_SEMILIVE_CHUNK_MIN_MS"))
    LIVE_SEMILIVE_CHUNK_PRE_ROLL_MS = int(_cfg(config, "LIVE_SEMILIVE_CHUNK_PRE_ROLL_MS"))
    LIVE_SEMILIVE_INITIAL_PROMPT_ENABLED = bool(_cfg(config, "LIVE_SEMILIVE_INITIAL_PROMPT_ENABLED"))
    LIVE_SEMILIVE_SPECULATIVE_INITIAL_PROMPT_ENABLED = bool(
        _cfg(config, "LIVE_SEMILIVE_SPECULATIVE_INITIAL_PROMPT_ENABLED")
    )
    LIVE_SEMILIVE_INITIAL_PROMPT_TAIL_WORDS = int(_cfg(config, "LIVE_SEMILIVE_INITIAL_PROMPT_TAIL_WORDS"))
    LIVE_SEMILIVE_INITIAL_PROMPT_MIN_WORDS = int(_cfg(config, "LIVE_SEMILIVE_INITIAL_PROMPT_MIN_WORDS"))
    LIVE_SEMILIVE_INITIAL_PROMPT_MAX_CHARS = int(_cfg(config, "LIVE_SEMILIVE_INITIAL_PROMPT_MAX_CHARS"))
    LIVE_SEMILIVE_SPECULATIVE_ENABLED = bool(_cfg(config, "LIVE_SEMILIVE_SPECULATIVE_ENABLED"))
    LIVE_SEMILIVE_SPECULATIVE_INTERVAL_MS = int(_cfg(config, "LIVE_SEMILIVE_SPECULATIVE_INTERVAL_MS"))
    LIVE_SEMILIVE_SPECULATIVE_WINDOW_MS = int(_cfg(config, "LIVE_SEMILIVE_SPECULATIVE_WINDOW_MS"))
    LIVE_SEMILIVE_SPECULATIVE_OVERLAP_MS = int(_cfg(config, "LIVE_SEMILIVE_SPECULATIVE_OVERLAP_MS"))
    LIVE_SEMILIVE_SPECULATIVE_MAX_STALENESS_MS = int(_cfg(config, "LIVE_SEMILIVE_SPECULATIVE_MAX_STALENESS_MS"))
    LIVE_SEMILIVE_SPECULATIVE_REQUIRE_NO_FINAL_PENDING = bool(
        _cfg(config, "LIVE_SEMILIVE_SPECULATIVE_REQUIRE_NO_FINAL_PENDING")
    )
    LIVE_SEMILIVE_FINAL_BEAM_SIZE_OVERRIDE = int(_cfg(config, "LIVE_SEMILIVE_FINAL_BEAM_SIZE_OVERRIDE"))
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
    recorder: LiveWavRecorder | None = None
    chunker: LiveAudioChunker | None = None
    chunk_bridge: LiveChunkBatchBridge | None = None
    semilive_chunk_jobs_enabled = bool(LIVE_SEMILIVE_CHUNK_BATCH_SHADOW)
    semilive_chunk_jobs_pending: dict[int, dict[str, Any]] = {}
    semilive_chunk_jobs_to_enqueue: list[Any] = []
    semilive_chunk_jobs_last_poll_mono = 0.0
    semilive_recording_state = "idle"
    semilive_recording_path = ""
    semilive_recording_bytes = 0
    semilive_recording_duration_ms = 0
    semilive_chunk_index_next = 0
    semilive_chunks_total = 0
    semilive_chunks_done = 0
    semilive_chunks_failed = 0
    semilive_finalization_state = "idle"
    semilive_shadow_disabled_reason = ""
    semilive_recording_finalized = False
    semilive_chunker_snapshot: dict[str, Any] = {}
    semilive_speculative_enabled = bool(LIVE_SEMILIVE_SPECULATIVE_ENABLED and semilive_chunk_jobs_enabled)
    semilive_speculative_jobs_pending: dict[int, dict[str, Any]] = {}
    semilive_speculative_last_emit_mono = 0.0
    semilive_speculative_last_poll_mono = 0.0
    semilive_speculative_seq_next = 0
    semilive_speculative_recent_pcm = bytearray()
    semilive_speculative_effective_window_ms = int(
        max(
            LIVE_SEMILIVE_SPECULATIVE_WINDOW_MS,
            LIVE_SEMILIVE_SPECULATIVE_INTERVAL_MS + LIVE_SEMILIVE_SPECULATIVE_OVERLAP_MS,
        )
    )
    semilive_speculative_recent_pcm_max_bytes = int(
        max(
            LIVE_AUDIO_SAMPLE_WIDTH_BYTES,
            round(((semilive_speculative_effective_window_ms + 1000) / 1000.0) * LIVE_AUDIO_BYTES_PER_SECOND),
        )
    )
    if (semilive_speculative_recent_pcm_max_bytes % LIVE_AUDIO_SAMPLE_WIDTH_BYTES) != 0:
        semilive_speculative_recent_pcm_max_bytes += (
            LIVE_AUDIO_SAMPLE_WIDTH_BYTES - (semilive_speculative_recent_pcm_max_bytes % LIVE_AUDIO_SAMPLE_WIDTH_BYTES)
        )
    semilive_speculative_metrics: dict[str, Any] = {
        "enqueued": 0,
        "shown": 0,
        "dropped_busy": 0,
        "dropped_stale": 0,
        "time_to_first_speculative_ms": None,
    }

    async def send_event(payload: dict[str, Any]) -> None:
        payload = dict(payload)
        try:
            payload["seq"] = LIVE_SESSIONS.next_seq(session_id)
        except KeyError:
            pass
        await websocket.send_json(payload)

    def _semilive_archive_kwargs() -> dict[str, Any]:
        return {
            "recording_path": str(semilive_recording_path or ""),
            "recording_bytes": int(max(0, semilive_recording_bytes)),
            "recording_duration_ms": int(max(0, semilive_recording_duration_ms)),
            "chunks_total": int(max(0, semilive_chunks_total)),
            "chunks_done": int(max(0, semilive_chunks_done)),
            "chunks_failed": int(max(0, semilive_chunks_failed)),
            "finalization_state": str(semilive_finalization_state or ""),
            "batch_job_id": "",
        }

    def _archive_current_semilive_result(*, close_reason: str) -> dict[str, Any]:
        try:
            semilive_result = LIVE_SESSIONS.semilive_result_snapshot(session_id)
        except Exception:
            return {}
        if not semilive_result:
            return {}
        has_content = (
            bool(str(semilive_result.get("final_text") or "").strip())
            or bool(semilive_result.get("final_segments"))
            or int(semilive_result.get("chunks_total") or 0) > 0
            or int(max(0, semilive_recording_duration_ms)) > 0
        )
        if not has_content:
            return semilive_result
        LIVE_SESSIONS.archive_transcript(
            session_id,
            close_reason=str(close_reason or stop_reason or "closed"),
            final_text=str(semilive_result.get("final_text") or ""),
            final_segments=[
                dict(seg)
                for seg in (semilive_result.get("final_segments") or [])
                if isinstance(seg, dict)
            ],
            transcript_revision=int(max(0, int(semilive_result.get("transcript_revision") or 0))),
            live_engine=LIVE_ENGINE,
            **_semilive_archive_kwargs(),
        )
        return semilive_result

    def _append_semilive_log(kind: str, **fields: Any) -> None:
        try:
            row = {"kind": str(kind)}
            row.update(fields)
            LIVE_SESSIONS.append_stats_log(session_id, row)
        except Exception:
            pass

    def _update_semilive_session_state() -> None:
        try:
            LIVE_SESSIONS.update_semilive(
                session_id,
                recording_state=semilive_recording_state,
                recording_path=semilive_recording_path,
                recording_bytes=semilive_recording_bytes,
                recording_duration_ms=semilive_recording_duration_ms,
                chunk_index_next=semilive_chunk_index_next,
                chunks_total=semilive_chunks_total,
                chunks_done=semilive_chunks_done,
                chunks_failed=semilive_chunks_failed,
                finalization_state=semilive_finalization_state,
                batch_job_id="",
            )
        except Exception:
            pass
        if semilive_speculative_enabled:
            try:
                LIVE_SESSIONS.update_semilive_speculative_metrics(
                    session_id,
                    enqueued=int(max(0, int(semilive_speculative_metrics.get("enqueued") or 0))),
                    shown=int(max(0, int(semilive_speculative_metrics.get("shown") or 0))),
                    dropped_busy=int(max(0, int(semilive_speculative_metrics.get("dropped_busy") or 0))),
                    dropped_stale=int(max(0, int(semilive_speculative_metrics.get("dropped_stale") or 0))),
                    time_to_first_speculative_ms=(
                        int(semilive_speculative_metrics.get("time_to_first_speculative_ms"))
                        if semilive_speculative_metrics.get("time_to_first_speculative_ms") is not None
                        else None
                    ),
                )
            except Exception:
                pass
            if str(semilive_finalization_state or "").strip().lower() in {"ready", "error", "finalized"}:
                try:
                    LIVE_SESSIONS.clear_semilive_speculative_preview(session_id)
                except Exception:
                    pass

    def _build_semilive_initial_prompt() -> tuple[str, dict[str, Any]]:
        if not LIVE_SEMILIVE_INITIAL_PROMPT_ENABLED:
            return "", {"enabled": False, "used": False, "reason": "disabled"}
        try:
            result = LIVE_SESSIONS.semilive_result_snapshot(session_id)
        except Exception as e:
            return "", {"enabled": True, "used": False, "reason": f"snapshot_error:{type(e).__name__}"}

        final_text = " ".join(str(result.get("final_text") or "").split()).strip()
        if not final_text:
            return "", {"enabled": True, "used": False, "reason": "no_context_text"}

        tokens = [tok for tok in final_text.split(" ") if tok]
        source_words = len(tokens)
        if source_words < LIVE_SEMILIVE_INITIAL_PROMPT_MIN_WORDS:
            return "", {
                "enabled": True,
                "used": False,
                "reason": "too_few_words",
                "source_words": int(source_words),
            }

        tail_tokens = tokens[-LIVE_SEMILIVE_INITIAL_PROMPT_TAIL_WORDS :]
        prompt = " ".join(tail_tokens).strip()
        if LIVE_SEMILIVE_INITIAL_PROMPT_MAX_CHARS > 0 and len(prompt) > LIVE_SEMILIVE_INITIAL_PROMPT_MAX_CHARS:
            tail = prompt[-LIVE_SEMILIVE_INITIAL_PROMPT_MAX_CHARS :].strip()
            if " " in tail:
                tail = tail.split(" ", 1)[1].strip()
            prompt = tail or prompt[-LIVE_SEMILIVE_INITIAL_PROMPT_MAX_CHARS :].strip()
        prompt_words = len([tok for tok in prompt.split() if tok])
        if prompt_words < LIVE_SEMILIVE_INITIAL_PROMPT_MIN_WORDS:
            return "", {
                "enabled": True,
                "used": False,
                "reason": "trimmed_too_short",
                "source_words": int(source_words),
                "tail_words": int(prompt_words),
            }
        return prompt, {
            "enabled": True,
            "used": True,
            "chars": len(prompt),
            "words": int(prompt_words),
            "source_words": int(source_words),
            "transcript_revision": int(max(0, int(result.get("transcript_revision") or 0))),
        }

    def _append_speculative_pcm(raw_bytes: bytes) -> None:
        nonlocal semilive_speculative_recent_pcm
        if not semilive_speculative_enabled:
            return
        raw = bytes(raw_bytes or b"")
        if not raw:
            return
        if (len(raw) % LIVE_AUDIO_SAMPLE_WIDTH_BYTES) != 0:
            raw = raw[: len(raw) - (len(raw) % LIVE_AUDIO_SAMPLE_WIDTH_BYTES)]
        if not raw:
            return
        semilive_speculative_recent_pcm.extend(raw)
        overflow = len(semilive_speculative_recent_pcm) - int(max(0, semilive_speculative_recent_pcm_max_bytes))
        if overflow > 0:
            if (overflow % LIVE_AUDIO_SAMPLE_WIDTH_BYTES) != 0:
                overflow += LIVE_AUDIO_SAMPLE_WIDTH_BYTES - (overflow % LIVE_AUDIO_SAMPLE_WIDTH_BYTES)
            del semilive_speculative_recent_pcm[:overflow]

    async def _maybe_enqueue_speculative_job() -> None:
        nonlocal semilive_speculative_last_emit_mono, semilive_speculative_seq_next
        if not semilive_speculative_enabled or chunk_bridge is None:
            return
        if semilive_recording_finalized or str(semilive_recording_state or "") != "recording":
            return
        now_mono = time.monotonic()
        interval_s = max(0.2, float(LIVE_SEMILIVE_SPECULATIVE_INTERVAL_MS) / 1000.0)
        if (now_mono - semilive_speculative_last_emit_mono) < interval_s:
            return
        if semilive_speculative_jobs_pending:
            semilive_speculative_last_emit_mono = now_mono
            semilive_speculative_metrics["dropped_busy"] = int(
                max(0, int(semilive_speculative_metrics.get("dropped_busy") or 0)) + 1
            )
            return
        if LIVE_SEMILIVE_SPECULATIVE_REQUIRE_NO_FINAL_PENDING and (
            semilive_chunk_jobs_pending or semilive_chunk_jobs_to_enqueue
        ):
            semilive_speculative_last_emit_mono = now_mono
            semilive_speculative_metrics["dropped_busy"] = int(
                max(0, int(semilive_speculative_metrics.get("dropped_busy") or 0)) + 1
            )
            return
        end_ms = int(max(0, semilive_recording_duration_ms))
        if end_ms <= 0 or not semilive_speculative_recent_pcm:
            return
        want_bytes = int(
            max(
                LIVE_AUDIO_SAMPLE_WIDTH_BYTES,
                round((float(semilive_speculative_effective_window_ms) / 1000.0) * LIVE_AUDIO_BYTES_PER_SECOND),
            )
        )
        if (want_bytes % LIVE_AUDIO_SAMPLE_WIDTH_BYTES) != 0:
            want_bytes += LIVE_AUDIO_SAMPLE_WIDTH_BYTES - (want_bytes % LIVE_AUDIO_SAMPLE_WIDTH_BYTES)
        use_bytes = min(len(semilive_speculative_recent_pcm), want_bytes)
        if (use_bytes % LIVE_AUDIO_SAMPLE_WIDTH_BYTES) != 0:
            use_bytes -= use_bytes % LIVE_AUDIO_SAMPLE_WIDTH_BYTES
        min_bytes = int(max(LIVE_AUDIO_SAMPLE_WIDTH_BYTES, LIVE_AUDIO_BYTES_PER_SECOND * 0.5))
        if use_bytes < min_bytes:
            return
        pcm = bytes(semilive_speculative_recent_pcm[-use_bytes:])
        actual_ms = int(round((float(use_bytes) / float(LIVE_AUDIO_BYTES_PER_SECOND)) * 1000.0))
        t1_ms = int(end_ms)
        t0_ms = int(max(0, t1_ms - max(1, actual_ms)))
        spec_seq = int(max(0, semilive_speculative_seq_next))
        if LIVE_SEMILIVE_SPECULATIVE_INITIAL_PROMPT_ENABLED:
            initial_prompt, initial_prompt_meta = _build_semilive_initial_prompt()
        else:
            initial_prompt = ""
            initial_prompt_meta = {
                "enabled": False,
                "used": False,
                "reason": "disabled_for_speculative_lane",
            }
        try:
            enq = await asyncio.to_thread(
                chunk_bridge.enqueue_chunk_pcm16,
                session_id=session_id,
                chunk_index=spec_seq,
                t0_ms=t0_ms,
                t1_ms=t1_ms,
                pcm16le=pcm,
                language=LIVE_SEMILIVE_CHUNK_LANGUAGE,
                initial_prompt=initial_prompt,
                live_lane="speculative",
                speculative_seq=spec_seq,
                speculative_audio_end_ms=t1_ms,
            )
        except Exception as e:
            _append_semilive_log(
                "semilive_speculative_enqueue_error",
                speculative_seq=spec_seq,
                error=f"{type(e).__name__}: {e}",
            )
            return
        semilive_speculative_seq_next = spec_seq + 1
        semilive_speculative_last_emit_mono = now_mono
        semilive_speculative_jobs_pending[spec_seq] = {
            "speculative_seq": spec_seq,
            "job_id": str(enq.job_id),
            "t0_ms": int(t0_ms),
            "t1_ms": int(t1_ms),
            "audio_end_ms": int(t1_ms),
            "reported_state": "",
            "enqueued_mono": now_mono,
        }
        semilive_speculative_metrics["enqueued"] = int(max(0, int(semilive_speculative_metrics.get("enqueued") or 0)) + 1)
        _append_semilive_log(
            "semilive_speculative_enqueued",
            speculative_seq=spec_seq,
            job_id=str(enq.job_id),
            t0_ms=int(t0_ms),
            t1_ms=int(t1_ms),
            audio_bytes=int(len(pcm)),
            window_ms=int(max(0, t1_ms - t0_ms)),
            initial_prompt_meta=dict(initial_prompt_meta) if isinstance(initial_prompt_meta, dict) else {},
        )

    async def _poll_speculative_jobs(*, force: bool = False) -> None:
        nonlocal semilive_speculative_last_poll_mono
        if not semilive_speculative_enabled or chunk_bridge is None:
            return
        now_mono = time.monotonic()
        if not force and (now_mono - semilive_speculative_last_poll_mono) < LIVE_SEMILIVE_CHUNK_POLL_INTERVAL_S:
            return
        semilive_speculative_last_poll_mono = now_mono
        for spec_seq in sorted(list(semilive_speculative_jobs_pending.keys())):
            item = semilive_speculative_jobs_pending.get(spec_seq)
            if not item:
                continue
            job_id = str(item.get("job_id") or "")
            t0_ms = int(item.get("t0_ms") or 0)
            enqueued_mono = float(item.get("enqueued_mono") or 0.0)
            if not job_id:
                continue
            try:
                poll = await asyncio.to_thread(
                    chunk_bridge.poll_job,
                    job_id,
                    t0_offset_ms=t0_ms,
                )
            except FileNotFoundError:
                continue
            except Exception as e:
                _append_semilive_log(
                    "semilive_speculative_poll_error",
                    speculative_seq=int(spec_seq),
                    job_id=job_id,
                    error=f"{type(e).__name__}: {e}",
                )
                semilive_speculative_jobs_pending.pop(spec_seq, None)
                continue

            poll_state = "ready" if poll.ok else ("error" if poll.done else str(poll.state or "queued"))
            if str(item.get("reported_state") or "") != poll_state:
                item["reported_state"] = poll_state
                if poll_state == "ready":
                    status_obj = dict(poll.status) if isinstance(poll.status, dict) else {}
                    def _status_int(name: str) -> int | None:
                        if name not in status_obj or status_obj.get(name) is None:
                            return None
                        try:
                            return int(max(0, int(status_obj.get(name))))
                        except Exception:
                            return None
                    def _status_float(name: str) -> float | None:
                        if name not in status_obj or status_obj.get(name) is None:
                            return None
                        try:
                            return max(0.0, float(status_obj.get(name)))
                        except Exception:
                            return None
                    wait_ms = None
                    if enqueued_mono > 0.0:
                        try:
                            wait_ms = int(max(0, round((time.monotonic() - enqueued_mono) * 1000.0)))
                        except Exception:
                            wait_ms = None
                    audio_end_ms = int(max(0, int(item.get("audio_end_ms") or 0)))
                    final_covered_ms = 0
                    try:
                        final_snapshot = LIVE_SESSIONS.semilive_result_snapshot(session_id)
                        final_covered_ms = int(max(0, int((final_snapshot or {}).get("final_covered_ms") or 0)))
                    except Exception:
                        final_covered_ms = 0
                    staleness_ms = int(
                        max(0, int(semilive_recording_duration_ms) - audio_end_ms)
                    )
                    stale_by_final = bool(audio_end_ms > 0 and final_covered_ms >= audio_end_ms)
                    stale_by_cursor = bool(
                        (LIVE_SEMILIVE_SPECULATIVE_MAX_STALENESS_MS > 0)
                        and (staleness_ms > int(LIVE_SEMILIVE_SPECULATIVE_MAX_STALENESS_MS))
                        and not semilive_recording_finalized
                        and final_covered_ms <= 0
                    )
                    stale = bool(stale_by_final or stale_by_cursor)
                    if stale:
                        semilive_speculative_metrics["dropped_stale"] = int(
                            max(0, int(semilive_speculative_metrics.get("dropped_stale") or 0)) + 1
                        )
                        _append_semilive_log(
                            "semilive_speculative_dropped_stale",
                            speculative_seq=int(spec_seq),
                            job_id=job_id,
                            staleness_ms=int(max(0, staleness_ms)),
                            final_covered_ms=int(max(0, final_covered_ms)),
                            stale_reason=("final_covered" if stale_by_final else "live_cursor"),
                            text_chars=len(str(poll.text or "")),
                        )
                    else:
                        semilive_speculative_metrics["shown"] = int(
                            max(0, int(semilive_speculative_metrics.get("shown") or 0)) + 1
                        )
                        if semilive_speculative_metrics.get("time_to_first_speculative_ms") is None:
                            semilive_speculative_metrics["time_to_first_speculative_ms"] = int(
                                max(0, int(item.get("audio_end_ms") or 0))
                            )
                        try:
                            LIVE_SESSIONS.update_semilive_speculative_preview(
                                session_id,
                                text=str(poll.text or ""),
                                speculative_seq=int(spec_seq),
                                audio_end_ms=int(max(0, int(item.get("audio_end_ms") or 0))),
                            )
                        except Exception:
                            pass
                        try:
                            LIVE_SESSIONS.update_semilive_speculative_metrics(
                                session_id,
                                enqueued=int(max(0, int(semilive_speculative_metrics.get("enqueued") or 0))),
                                shown=int(max(0, int(semilive_speculative_metrics.get("shown") or 0))),
                                dropped_busy=int(max(0, int(semilive_speculative_metrics.get("dropped_busy") or 0))),
                                dropped_stale=int(max(0, int(semilive_speculative_metrics.get("dropped_stale") or 0))),
                                time_to_first_speculative_ms=(
                                    int(semilive_speculative_metrics.get("time_to_first_speculative_ms"))
                                    if semilive_speculative_metrics.get("time_to_first_speculative_ms") is not None
                                    else None
                                ),
                            )
                        except Exception:
                            pass
                        _append_semilive_log(
                            "semilive_speculative_ready",
                            speculative_seq=int(spec_seq),
                            job_id=job_id,
                            staleness_ms=int(max(0, staleness_ms)),
                            final_covered_ms=int(max(0, final_covered_ms)),
                            text_chars=len(str(poll.text or "")),
                            segments_count=len(poll.segments or []),
                            wait_ms=(int(wait_ms) if wait_ms is not None else None),
                            remote_submit_attempts=_status_int("asr_remote_submit_attempts"),
                            remote_status_attempts_total=_status_int("asr_remote_status_attempts_total"),
                            remote_status_http_calls=_status_int("asr_remote_status_http_calls"),
                            remote_cancel_attempts=_status_int("asr_remote_cancel_attempts"),
                            blob_fetch_ms=_status_float("asr_blob_fetch_ms"),
                        )
                elif poll_state == "error":
                    _append_semilive_log(
                        "semilive_speculative_error",
                        speculative_seq=int(spec_seq),
                        job_id=job_id,
                        error=str(poll.error or ""),
                        status={"state": poll.state, "ok": poll.ok, "done": poll.done},
                    )
            if poll.done:
                semilive_speculative_jobs_pending.pop(spec_seq, None)

    async def _process_semilive_speculative_jobs(*, force_poll: bool = False) -> None:
        if not semilive_speculative_enabled:
            return
        await _maybe_enqueue_speculative_job()
        await _poll_speculative_jobs(force=force_poll)

    def _ingest_closed_chunks(chunks: list[Any]) -> None:
        nonlocal semilive_chunks_total, semilive_chunk_index_next, semilive_chunker_snapshot
        nonlocal semilive_chunk_jobs_to_enqueue
        if not chunks:
            return
        for chunk in chunks:
            semilive_chunks_total += 1
            semilive_chunk_index_next = max(
                int(semilive_chunk_index_next),
                int(getattr(chunk, "chunk_index", semilive_chunk_index_next)) + 1,
            )
            _append_semilive_log(
                "semilive_chunk_closed",
                chunk=getattr(chunk, "to_dict", lambda: {})(),
            )
            if semilive_chunk_jobs_enabled and chunk_bridge is not None:
                semilive_chunk_jobs_to_enqueue.append(chunk)
        if chunker is not None:
            try:
                semilive_chunker_snapshot = dict(chunker.snapshot())
            except Exception:
                semilive_chunker_snapshot = {}
        if (semilive_chunk_jobs_pending or semilive_chunk_jobs_to_enqueue) and semilive_finalization_state not in {"error", "ready"}:
            # Shadow processing is in progress, even if chunk jobs are feature-flagged.
            pass
        _update_semilive_session_state()

    def _sync_semilive_counts_from_result(result: dict[str, Any]) -> None:
        nonlocal semilive_chunks_total, semilive_chunks_done, semilive_chunks_failed
        nonlocal semilive_finalization_state
        semilive_chunks_total = int(max(0, int(result.get("chunks_total") or semilive_chunks_total)))
        semilive_chunks_done = int(max(0, int(result.get("chunks_done") or semilive_chunks_done)))
        semilive_chunks_failed = int(max(0, int(result.get("chunks_failed") or semilive_chunks_failed)))
        if semilive_recording_finalized:
            if semilive_chunk_jobs_pending or semilive_chunk_jobs_to_enqueue:
                semilive_finalization_state = "processing_chunks"
            elif semilive_chunks_failed > 0:
                semilive_finalization_state = "error"
            elif semilive_chunk_jobs_enabled and (semilive_chunks_total > 0 or semilive_chunks_done > 0):
                semilive_finalization_state = "ready"

    async def _record_semilive_chunk_job_state(
        *,
        chunk_index: int,
        t0_ms: int,
        t1_ms: int,
        state: str,
        text: str = "",
        segments: list[dict[str, Any]] | None = None,
        error: str = "",
        chunk_meta: dict[str, Any] | None = None,
        job_status: dict[str, Any] | None = None,
    ) -> None:
        meta = chunk_meta if isinstance(chunk_meta, dict) else {}
        status_obj = job_status if isinstance(job_status, dict) else {}

        def _parse_status_float(name: str) -> float | None:
            if name not in status_obj or status_obj.get(name) is None:
                return None
            try:
                return max(0.0, float(status_obj.get(name)))
            except Exception:
                return None

        try:
            result = LIVE_SESSIONS.record_semilive_chunk_result(
                session_id,
                chunk_index=int(chunk_index),
                t0_ms=int(t0_ms),
                t1_ms=int(t1_ms),
                text=str(text or ""),
                segments=segments,
                state=str(state or "ready"),
                error=str(error or ""),
                reason=str(meta.get("reason") or ""),
                speech_frames=(int(meta.get("speech_frames")) if meta.get("speech_frames") is not None else None),
                silence_frames_tail=(
                    int(meta.get("silence_frames_tail")) if meta.get("silence_frames_tail") is not None else None
                ),
                chunk_duration_ms=(int(meta.get("duration_ms")) if meta.get("duration_ms") is not None else None),
                asr_pipeline_time_s=_parse_status_float("asr_timing_whisperx_total_s"),
                asr_transcribe_time_s=_parse_status_float("asr_timing_whisperx_transcribe_s"),
            )
            _sync_semilive_counts_from_result(result)
            _update_semilive_session_state()
        except Exception as e:
            _append_semilive_log(
                "semilive_chunk_result_store_error",
                chunk_index=int(chunk_index),
                state=str(state or ""),
                error=f"{type(e).__name__}: {e}",
            )

    async def _drain_semilive_chunk_enqueues() -> None:
        nonlocal semilive_finalization_state
        nonlocal semilive_shadow_disabled_reason
        if not semilive_chunk_jobs_enabled or chunk_bridge is None:
            return
        while semilive_chunk_jobs_to_enqueue:
            chunk = semilive_chunk_jobs_to_enqueue.pop(0)
            chunk_meta = getattr(chunk, "to_dict", lambda: {})()
            chunk_index = int(getattr(chunk, "chunk_index", 0))
            t0_ms = int(getattr(chunk, "t0_ms", 0))
            t1_ms = int(getattr(chunk, "t1_ms", t0_ms))
            initial_prompt, initial_prompt_meta = _build_semilive_initial_prompt()
            if semilive_finalization_state not in {"error"}:
                semilive_finalization_state = "processing_chunks"
            final_beam_override = (
                int(max(1, int(LIVE_SEMILIVE_FINAL_BEAM_SIZE_OVERRIDE)))
                if int(max(0, int(LIVE_SEMILIVE_FINAL_BEAM_SIZE_OVERRIDE or 0))) > 0
                else None
            )
            try:
                enq = await asyncio.to_thread(
                    chunk_bridge.enqueue_chunk_pcm16,
                    session_id=session_id,
                    chunk_index=chunk_index,
                    t0_ms=t0_ms,
                    t1_ms=t1_ms,
                    pcm16le=bytes(getattr(chunk, "pcm16le", b"") or b""),
                    language=LIVE_SEMILIVE_CHUNK_LANGUAGE,
                    asr_beam_size=final_beam_override,
                    initial_prompt=initial_prompt,
                )
                semilive_chunk_jobs_pending[chunk_index] = {
                    "chunk_index": chunk_index,
                    "t0_ms": t0_ms,
                    "t1_ms": t1_ms,
                    "job_id": str(enq.job_id),
                    "job_dir": str(enq.job_dir),
                    "state": "queued",
                    "reported_state": "",
                    "enqueued_mono": time.monotonic(),
                    "last_poll_mono": 0.0,
                    "chunk_meta": dict(chunk_meta) if isinstance(chunk_meta, dict) else {},
                    "initial_prompt_meta": dict(initial_prompt_meta) if isinstance(initial_prompt_meta, dict) else {},
                }
                await _record_semilive_chunk_job_state(
                    chunk_index=chunk_index,
                    t0_ms=t0_ms,
                    t1_ms=t1_ms,
                    state="queued",
                    chunk_meta=dict(chunk_meta) if isinstance(chunk_meta, dict) else None,
                )
                _append_semilive_log(
                    "semilive_chunk_enqueued",
                    chunk=dict(chunk_meta) if isinstance(chunk_meta, dict) else {"chunk_index": chunk_index},
                    job=enq.to_dict(),
                    final_beam_size_override=(
                        int(final_beam_override) if final_beam_override is not None else None
                    ),
                    initial_prompt_meta=dict(initial_prompt_meta) if isinstance(initial_prompt_meta, dict) else {},
                )
            except Exception as e:
                semilive_shadow_disabled_reason = f"chunk_enqueue_failed:{type(e).__name__}"
                semilive_finalization_state = "error"
                await _record_semilive_chunk_job_state(
                    chunk_index=chunk_index,
                    t0_ms=t0_ms,
                    t1_ms=t1_ms,
                    state="error",
                    error=f"{type(e).__name__}: {e}",
                    chunk_meta=dict(chunk_meta) if isinstance(chunk_meta, dict) else None,
                )
                _append_semilive_log(
                    "semilive_chunk_enqueue_error",
                    chunk=dict(chunk_meta) if isinstance(chunk_meta, dict) else {"chunk_index": chunk_index},
                    error=f"{type(e).__name__}: {e}",
                )
                _update_semilive_session_state()

    async def _poll_semilive_chunk_jobs(*, force: bool = False) -> None:
        nonlocal semilive_chunk_jobs_last_poll_mono
        nonlocal semilive_finalization_state
        nonlocal semilive_shadow_disabled_reason
        if not semilive_chunk_jobs_enabled or chunk_bridge is None:
            return
        now_mono = time.monotonic()
        if not force and (now_mono - semilive_chunk_jobs_last_poll_mono) < LIVE_SEMILIVE_CHUNK_POLL_INTERVAL_S:
            return
        semilive_chunk_jobs_last_poll_mono = now_mono
        if semilive_chunk_jobs_pending and semilive_finalization_state not in {"error"}:
            semilive_finalization_state = "processing_chunks"
        for chunk_index in sorted(list(semilive_chunk_jobs_pending.keys())):
            item = semilive_chunk_jobs_pending.get(chunk_index)
            if not item:
                continue
            item["last_poll_mono"] = now_mono
            t0_ms = int(item.get("t0_ms") or 0)
            t1_ms = int(item.get("t1_ms") or t0_ms)
            job_id = str(item.get("job_id") or "")
            enqueued_mono = float(item.get("enqueued_mono") or 0.0)
            chunk_meta = item.get("chunk_meta") if isinstance(item.get("chunk_meta"), dict) else None
            if not job_id:
                continue
            try:
                poll = await asyncio.to_thread(
                    chunk_bridge.poll_job,
                    job_id,
                    t0_offset_ms=t0_ms,
                )
            except FileNotFoundError:
                # Worker may not have published the dir yet; keep queued.
                continue
            except Exception as e:
                await _record_semilive_chunk_job_state(
                    chunk_index=chunk_index,
                    t0_ms=t0_ms,
                    t1_ms=t1_ms,
                    state="error",
                    error=f"{type(e).__name__}: {e}",
                    chunk_meta=chunk_meta,
                )
                semilive_chunk_jobs_pending.pop(chunk_index, None)
                semilive_shadow_disabled_reason = f"chunk_poll_failed:{type(e).__name__}"
                semilive_finalization_state = "error"
                _append_semilive_log(
                    "semilive_chunk_poll_error",
                    chunk_index=chunk_index,
                    job_id=job_id,
                    error=f"{type(e).__name__}: {e}",
                )
                continue

            poll_state = "ready" if poll.ok else ("error" if poll.done else str(poll.state or "queued"))
            if str(item.get("reported_state") or "") != poll_state:
                item["reported_state"] = poll_state
                if poll_state == "ready":
                    status_obj = dict(poll.status) if isinstance(poll.status, dict) else {}
                    def _status_int(name: str) -> int | None:
                        if name not in status_obj or status_obj.get(name) is None:
                            return None
                        try:
                            return int(max(0, int(status_obj.get(name))))
                        except Exception:
                            return None
                    def _status_float(name: str) -> float | None:
                        if name not in status_obj or status_obj.get(name) is None:
                            return None
                        try:
                            return max(0.0, float(status_obj.get(name)))
                        except Exception:
                            return None
                    wait_ms = None
                    if enqueued_mono > 0.0:
                        try:
                            wait_ms = int(max(0, round((time.monotonic() - enqueued_mono) * 1000.0)))
                        except Exception:
                            wait_ms = None
                    await _record_semilive_chunk_job_state(
                        chunk_index=chunk_index,
                        t0_ms=t0_ms,
                        t1_ms=t1_ms,
                        state="ready",
                        text=str(poll.text or ""),
                        segments=[dict(seg) for seg in (poll.segments or []) if isinstance(seg, dict)],
                        chunk_meta=chunk_meta,
                        job_status=dict(poll.status) if isinstance(poll.status, dict) else None,
                    )
                    _append_semilive_log(
                        "semilive_chunk_ready",
                        chunk_index=chunk_index,
                        job_id=job_id,
                        status={"state": poll.state, "ok": poll.ok, "done": poll.done},
                        text_chars=len(str(poll.text or "")),
                        segments_count=len(poll.segments or []),
                        wait_ms=(int(wait_ms) if wait_ms is not None else None),
                        remote_submit_attempts=_status_int("asr_remote_submit_attempts"),
                        remote_status_attempts_total=_status_int("asr_remote_status_attempts_total"),
                        remote_status_http_calls=_status_int("asr_remote_status_http_calls"),
                        remote_cancel_attempts=_status_int("asr_remote_cancel_attempts"),
                        blob_fetch_ms=_status_float("asr_blob_fetch_ms"),
                    )
                elif poll_state == "error":
                    await _record_semilive_chunk_job_state(
                        chunk_index=chunk_index,
                        t0_ms=t0_ms,
                        t1_ms=t1_ms,
                        state="error",
                        error=str(poll.error or f"job_state:{poll.state}"),
                        chunk_meta=chunk_meta,
                        job_status=dict(poll.status) if isinstance(poll.status, dict) else None,
                    )
                    _append_semilive_log(
                        "semilive_chunk_error",
                        chunk_index=chunk_index,
                        job_id=job_id,
                        status={"state": poll.state, "ok": poll.ok, "done": poll.done},
                        error=str(poll.error or ""),
                    )
                else:
                    await _record_semilive_chunk_job_state(
                        chunk_index=chunk_index,
                        t0_ms=t0_ms,
                        t1_ms=t1_ms,
                        state=poll_state,
                        chunk_meta=chunk_meta,
                        job_status=dict(poll.status) if isinstance(poll.status, dict) else None,
                    )
            item["state"] = poll_state
            if poll.done:
                semilive_chunk_jobs_pending.pop(chunk_index, None)

        if semilive_recording_finalized and not semilive_chunk_jobs_pending and not semilive_chunk_jobs_to_enqueue:
            if semilive_chunks_failed > 0:
                semilive_finalization_state = "error"
            elif semilive_chunk_jobs_enabled and (semilive_chunks_total > 0 or semilive_chunks_done > 0):
                semilive_finalization_state = "ready"
        _update_semilive_session_state()

    async def _process_semilive_chunk_jobs(*, force_poll: bool = False) -> None:
        if not semilive_chunk_jobs_enabled:
            return
        await _drain_semilive_chunk_enqueues()
        await _poll_semilive_chunk_jobs(force=force_poll)

    async def _await_semilive_chunk_jobs_until_done(*, timeout_s: float) -> None:
        if not semilive_chunk_jobs_enabled:
            return
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while True:
            await _process_semilive_chunk_jobs(force_poll=True)
            if not semilive_chunk_jobs_to_enqueue and not semilive_chunk_jobs_pending:
                return
            if time.monotonic() >= deadline:
                return
            await asyncio.sleep(0.25)

    def _finalize_semilive_shadow(*, reason: str) -> None:
        nonlocal semilive_recording_finalized
        nonlocal semilive_recording_state
        nonlocal semilive_recording_path
        nonlocal semilive_recording_bytes
        nonlocal semilive_recording_duration_ms
        nonlocal semilive_chunker_snapshot
        nonlocal semilive_finalization_state
        nonlocal semilive_shadow_disabled_reason
        if semilive_recording_finalized:
            return
        semilive_finalization_state = "finalizing"
        if chunker is not None:
            try:
                _ingest_closed_chunks(chunker.flush_tail())
                semilive_chunker_snapshot = dict(chunker.snapshot())
            except Exception as e:
                semilive_shadow_disabled_reason = f"chunker_finalize_failed:{type(e).__name__}"
                semilive_recording_state = "error"
                semilive_finalization_state = "error"
                _append_semilive_log(
                    "semilive_chunker_finalize_error",
                    reason=reason,
                    error=f"{type(e).__name__}: {e}",
                )
        if recorder is not None:
            try:
                rs = recorder.finalize()
                semilive_recording_path = str(rs.wav_path)
                semilive_recording_bytes = int(rs.bytes_written)
                semilive_recording_duration_ms = int(rs.duration_ms)
                semilive_recording_state = "finalized"
                if semilive_finalization_state != "error":
                    semilive_finalization_state = "recording_finalized"
                _append_semilive_log(
                    "semilive_recording_finalized",
                    reason=reason,
                    recording=rs.to_dict(),
                    chunker_snapshot=dict(semilive_chunker_snapshot),
                )
            except Exception as e:
                semilive_shadow_disabled_reason = f"recording_finalize_failed:{type(e).__name__}"
                semilive_recording_state = "error"
                semilive_finalization_state = "error"
                _append_semilive_log(
                    "semilive_recording_finalize_error",
                    reason=reason,
                    error=f"{type(e).__name__}: {e}",
                )
        else:
            if semilive_finalization_state != "error":
                semilive_finalization_state = "idle"
        semilive_recording_finalized = True
        _update_semilive_session_state()

    try:
        ready_payload = ready_event(
            session_id,
            message="Live websocket connected. Send binary PCM16 frames and JSON controls.",
            engine="semilive_chunked",
        )
        ready_payload["live_engine"] = LIVE_ENGINE
        await send_event(ready_payload)

        # Semilive pipeline (WAV recorder + silence chunker + chunk batch jobs).
        try:
            recorder = LiveWavRecorder(
                session_id=session_id,
                sample_rate_hz=LIVE_AUDIO_SAMPLE_RATE_HZ,
                channels=LIVE_AUDIO_CHANNELS,
            )
            rec_snap = recorder.start()
            chunker = LiveAudioChunker(
                config=LiveChunkerConfig(
                    sample_rate_hz=LIVE_AUDIO_SAMPLE_RATE_HZ,
                    channels=LIVE_AUDIO_CHANNELS,
                    energy_threshold=LIVE_SEMILIVE_CHUNK_ENERGY_THRESHOLD,
                    silence_threshold_ms=LIVE_SEMILIVE_CHUNK_SILENCE_MS,
                    max_chunk_ms=LIVE_SEMILIVE_CHUNK_MAX_MS,
                    min_chunk_ms=LIVE_SEMILIVE_CHUNK_MIN_MS,
                    pre_roll_ms=LIVE_SEMILIVE_CHUNK_PRE_ROLL_MS,
                )
            )
            if semilive_chunk_jobs_enabled:
                chunk_bridge = LiveChunkBatchBridge(
                    sample_rate_hz=LIVE_AUDIO_SAMPLE_RATE_HZ,
                    channels=LIVE_AUDIO_CHANNELS,
                    language=LIVE_SEMILIVE_CHUNK_LANGUAGE,
                )
            semilive_recording_state = "recording"
            semilive_recording_path = str(rec_snap.wav_path)
            semilive_recording_bytes = int(rec_snap.bytes_written)
            semilive_recording_duration_ms = int(rec_snap.duration_ms)
            semilive_chunk_index_next = 0
            semilive_chunks_total = 0
            semilive_chunks_done = 0
            semilive_chunks_failed = 0
            semilive_finalization_state = "recording"
            semilive_chunker_snapshot = dict(chunker.snapshot())
            _update_semilive_session_state()
            _append_semilive_log(
                "semilive_shadow_started",
                recording=rec_snap.to_dict(),
                chunk_jobs_enabled=bool(semilive_chunk_jobs_enabled),
                speculative_enabled=bool(semilive_speculative_enabled),
                speculative_config={
                    "interval_ms": int(LIVE_SEMILIVE_SPECULATIVE_INTERVAL_MS),
                    "window_ms": int(LIVE_SEMILIVE_SPECULATIVE_WINDOW_MS),
                    "overlap_ms": int(LIVE_SEMILIVE_SPECULATIVE_OVERLAP_MS),
                    "effective_window_ms": int(semilive_speculative_effective_window_ms),
                    "max_staleness_ms": int(LIVE_SEMILIVE_SPECULATIVE_MAX_STALENESS_MS),
                    "require_no_final_pending": bool(LIVE_SEMILIVE_SPECULATIVE_REQUIRE_NO_FINAL_PENDING),
                    "final_beam_size_override": (
                        int(max(1, int(LIVE_SEMILIVE_FINAL_BEAM_SIZE_OVERRIDE)))
                        if int(max(0, int(LIVE_SEMILIVE_FINAL_BEAM_SIZE_OVERRIDE or 0))) > 0
                        else None
                    ),
                },
                chunker_config=chunker.config.__dict__,
                chunker_snapshot=dict(semilive_chunker_snapshot),
            )
        except Exception as e:
            recorder = None
            chunker = None
            semilive_recording_state = "error"
            semilive_finalization_state = "error"
            semilive_shadow_disabled_reason = f"semilive_init_failed:{type(e).__name__}"
            _update_semilive_session_state()
            _append_semilive_log(
                "semilive_shadow_init_error",
                error=f"{type(e).__name__}: {e}",
            )

        while True:
            incoming = await websocket.receive()

            if incoming.get("type") == "websocket.disconnect":
                stop_reason = "client_disconnected"
                break

            raw_bytes = incoming.get("bytes")
            if raw_bytes is not None:
                snapshot = LIVE_SESSIONS.record_audio(session_id, byte_count=len(raw_bytes))
                if recorder is not None:
                    try:
                        rec_snap = recorder.append_pcm16(raw_bytes)
                        semilive_recording_bytes = int(rec_snap.bytes_written)
                        semilive_recording_duration_ms = int(rec_snap.duration_ms)
                        semilive_recording_path = str(rec_snap.wav_path)
                    except Exception as e:
                        semilive_shadow_disabled_reason = f"recording_append_failed:{type(e).__name__}"
                        semilive_recording_state = "error"
                        semilive_finalization_state = "error"
                        _append_semilive_log(
                            "semilive_recording_append_error",
                            error=f"{type(e).__name__}: {e}",
                            at_frame=int(snapshot.get("frames_received") or 0),
                        )
                        try:
                            recorder.abort()
                        except Exception:
                            pass
                        recorder = None
                        _update_semilive_session_state()
                if chunker is not None:
                    try:
                        closed_chunks = chunker.feed_pcm16(raw_bytes)
                        semilive_chunker_snapshot = dict(chunker.snapshot())
                        _ingest_closed_chunks(closed_chunks)
                    except Exception as e:
                        semilive_shadow_disabled_reason = f"chunker_feed_failed:{type(e).__name__}"
                        semilive_recording_state = "error"
                        semilive_finalization_state = "error"
                        _append_semilive_log(
                            "semilive_chunker_feed_error",
                            error=f"{type(e).__name__}: {e}",
                            at_frame=int(snapshot.get("frames_received") or 0),
                        )
                        chunker = None
                        _update_semilive_session_state()
                _append_speculative_pcm(raw_bytes)
                await _process_semilive_chunk_jobs(force_poll=False)
                await _process_semilive_speculative_jobs(force_poll=False)
                # Emit transport + semilive stats at startup and periodically.
                should_emit_stats = (
                    snapshot["frames_received"] == 1
                    or (snapshot["frames_received"] % 50) == 0
                )
                if should_emit_stats:
                    stats_payload = stats_event(
                        session_id,
                        bytes_received=snapshot["bytes_received"],
                        frames_received=snapshot["frames_received"],
                        controls_received=snapshot["controls_received"],
                        uptime_s=snapshot["age_s"],
                        live_engine=LIVE_ENGINE,
                        live_mode="semilive_chunked",
                        semilive_recording_state=str(semilive_recording_state or ""),
                        semilive_recording_bytes=int(max(0, semilive_recording_bytes)),
                        semilive_recording_duration_ms=int(max(0, semilive_recording_duration_ms)),
                        semilive_chunk_index_next=int(max(0, semilive_chunk_index_next)),
                        semilive_chunks_total=int(max(0, semilive_chunks_total)),
                        semilive_chunks_done=int(max(0, semilive_chunks_done)),
                        semilive_chunks_failed=int(max(0, semilive_chunks_failed)),
                        semilive_finalization_state=str(semilive_finalization_state or ""),
                        semilive_chunk_jobs_enabled=bool(semilive_chunk_jobs_enabled),
                        semilive_chunk_jobs_pending=int(max(0, len(semilive_chunk_jobs_pending))),
                        semilive_chunk_jobs_to_enqueue=int(max(0, len(semilive_chunk_jobs_to_enqueue))),
                        semilive_speculative_enabled=bool(semilive_speculative_enabled),
                        semilive_speculative_pending=int(max(0, len(semilive_speculative_jobs_pending))),
                        semilive_speculative_enqueued=int(
                            max(0, int(semilive_speculative_metrics.get("enqueued") or 0))
                        ),
                        semilive_speculative_shown=int(max(0, int(semilive_speculative_metrics.get("shown") or 0))),
                        semilive_speculative_dropped_busy=int(
                            max(0, int(semilive_speculative_metrics.get("dropped_busy") or 0))
                        ),
                        semilive_speculative_dropped_stale=int(
                            max(0, int(semilive_speculative_metrics.get("dropped_stale") or 0))
                        ),
                        semilive_time_to_first_speculative_ms=(
                            int(semilive_speculative_metrics.get("time_to_first_speculative_ms"))
                            if semilive_speculative_metrics.get("time_to_first_speculative_ms") is not None
                            else None
                        ),
                        semilive_shadow_disabled_reason=str(semilive_shadow_disabled_reason or ""),
                        semilive_chunker_chunk_open=bool(semilive_chunker_snapshot.get("chunk_open")),
                        semilive_chunker_active_chunk_duration_ms=int(
                            max(0, int(semilive_chunker_snapshot.get("active_chunk_duration_ms") or 0))
                        ),
                        semilive_chunker_pre_roll_frames=int(
                            max(0, int(semilive_chunker_snapshot.get("pre_roll_frames_buffered") or 0))
                        ),
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
                semilive_recording_state = "recording"
                _update_semilive_session_state()
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
                semilive_recording_state = "paused"
                _update_semilive_session_state()
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
                semilive_recording_state = "recording"
                _update_semilive_session_state()
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
                semilive_result: dict[str, Any] = {}
                _finalize_semilive_shadow(reason="client_stop")
                await _process_semilive_chunk_jobs(force_poll=True)
                if LIVE_SEMILIVE_CHUNK_STOP_WAIT_S > 0:
                    await _await_semilive_chunk_jobs_until_done(timeout_s=LIVE_SEMILIVE_CHUNK_STOP_WAIT_S)
                try:
                    semilive_result = _archive_current_semilive_result(close_reason=stop_reason)
                except Exception:
                    semilive_result = {}
                await send_event(
                    ended_event(
                        session_id,
                        reason=stop_reason,
                        transcript_revision=int(max(0, int(semilive_result.get("transcript_revision") or 0))),
                        final_segments_count=len(semilive_result.get("final_segments") or []),
                        final_text=str(semilive_result.get("final_text") or ""),
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
        _finalize_semilive_shadow(reason=stop_reason)
        await _process_semilive_chunk_jobs(force_poll=True)
        if stop_reason == "client_stop":
            if LIVE_SEMILIVE_CHUNK_POST_CLOSE_WAIT_S > 0:
                await _await_semilive_chunk_jobs_until_done(timeout_s=LIVE_SEMILIVE_CHUNK_POST_CLOSE_WAIT_S)
        else:
            if LIVE_SEMILIVE_CHUNK_STOP_WAIT_S > 0:
                await _await_semilive_chunk_jobs_until_done(timeout_s=LIVE_SEMILIVE_CHUNK_STOP_WAIT_S)
        try:
            _archive_current_semilive_result(close_reason=stop_reason)
        except Exception:
            pass
        # Release capacity slot first; downstream close operations may block.
        LIVE_SESSIONS.close_session(session_id, reason=stop_reason)
        if recorder is not None and not semilive_recording_finalized:
            try:
                recorder.abort()
            except Exception:
                pass
