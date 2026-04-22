from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
import json
import time
from typing import Any, Callable, Mapping

from fastapi import WebSocket, WebSocketDisconnect, status
from realtime_asr_engine import ASRResult as EngineASRResult
from realtime_asr_engine import AudioFormat
from realtime_asr_engine import LiveASRRunner
from realtime_asr_engine import LiveASRRunnerSettings
from realtime_asr_engine import TranscriptSegment

from live.runtime.asr_bridge import LiveChunkBatchBridge
from live.runtime.protocol import (
    PROTOCOL_VERSION,
    control_ack_event,
    ended_event,
    error_event,
    parse_client_message,
    pong_event,
    ready_event,
    result_event,
    stats_event,
)
from live.runtime.util import _normalize_optional_language, _safe_float, parse_live_asr_language
from live.results.exports import build_live_result_envelope
from live.runtime.recorder import LiveWavRecorder


def _cfg(config: Mapping[str, Any], key: str) -> Any:
    if key not in config:
        raise RuntimeError(f"missing_live_engine_config:{key}")
    return config[key]


def _normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


_STATE_CONTROLS = {
    "start": ("listening", "recording"),
    "pause": ("paused", "paused"),
    "resume": ("listening", "recording"),
}


@dataclass
class _RollingRuntime:
    recorder: LiveWavRecorder | None = None
    chunk_bridge: LiveChunkBatchBridge | None = None
    runner: LiveASRRunner | None = None
    session_live_asr_language: str | None = None

    stop_reason: str = "client_disconnected"
    websocket_closed: bool = False
    archived_result: bool = False

    recording_state: str = "idle"
    recording_path: str = ""
    recording_bytes: int = 0
    recording_duration_ms: int = 0
    finalization_state: str = "idle"
    recording_finalized: bool = False

    rolling_commit_index_next: int = 0
    rolling_chunks_total: int = 0
    rolling_chunks_done: int = 0
    rolling_chunks_failed: int = 0

    rolling_inflight: dict[str, Any] | None = None
    rolling_last_applied_seq: int = -1
    rolling_gpu_proxy_transcribe_s: float = 0.0
    rolling_gpu_proxy_pipeline_s: float = 0.0
    last_result_event_signature: str = ""


class LiveWebSocketSession:
    def __init__(
        self,
        session_id: str,
        websocket: WebSocket,
        *,
        live_sessions: Any,
        rooted_path_cb: Callable[[str], str],
        config: Mapping[str, Any],
    ) -> None:
        self.session_id = session_id
        self.websocket = websocket
        self.live_sessions = live_sessions
        self.rooted_path_cb = rooted_path_cb
        self.config = config
        self.rt = _RollingRuntime()
        self._ctx: dict[str, Any] = {}
        self._completion_ready: asyncio.Event | None = None
        self._completion_loop: asyncio.AbstractEventLoop | None = None

    async def _send_event(self, payload: dict[str, Any]) -> None:
        out = dict(payload)
        try:
            out["seq"] = self.live_sessions.next_seq(self.session_id)
        except KeyError:
            pass
        await self.websocket.send_json(out)

    @staticmethod
    async def _cancel_pending_task(task: asyncio.Task[Any] | None) -> None:
        if task is None or task.done():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    def _notify_terminal_ready(self) -> None:
        loop = self._completion_loop
        ready = self._completion_ready
        if loop is None or ready is None:
            return
        try:
            loop.call_soon_threadsafe(ready.set)
        except RuntimeError:
            pass

    async def _wait_for_websocket_or_completion(self) -> tuple[str, dict[str, Any] | None]:
        ready = self._completion_ready
        if ready is not None and ready.is_set():
            ready.clear()
            return "completion", None

        receive_task: asyncio.Task[dict[str, Any]] = asyncio.create_task(self.websocket.receive())
        completion_task: asyncio.Task[bool] | None = None
        tasks: set[asyncio.Task[Any]] = {receive_task}
        if ready is not None:
            completion_task = asyncio.create_task(ready.wait())
            tasks.add(completion_task)

        done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        if receive_task in done:
            await self._cancel_pending_task(completion_task)
            return "websocket", receive_task.result()

        if ready is not None:
            ready.clear()
        await self._cancel_pending_task(receive_task)
        return "completion", None

    async def _wait_for_completion_or_timeout(self, *, timeout_s: float) -> bool:
        safe_timeout_s = float(max(0.0, timeout_s))
        if safe_timeout_s <= 0.0:
            return False
        ready = self._completion_ready
        if ready is None:
            await asyncio.sleep(safe_timeout_s)
            return False
        if ready.is_set():
            ready.clear()
            return True
        try:
            await asyncio.wait_for(ready.wait(), timeout=safe_timeout_s)
        except asyncio.TimeoutError:
            return False
        ready.clear()
        return True

    def _configure_context(self) -> None:
        config = self.config

        live_audio_sample_rate_hz = int(_cfg(config, "LIVE_AUDIO_SAMPLE_RATE_HZ"))
        live_audio_channels = int(_cfg(config, "LIVE_AUDIO_CHANNELS"))
        live_audio_sample_width_bytes = int(_cfg(config, "LIVE_AUDIO_SAMPLE_WIDTH_BYTES"))
        audio_format = AudioFormat(
            sample_rate_hz=live_audio_sample_rate_hz,
            channels=live_audio_channels,
            sample_width_bytes=live_audio_sample_width_bytes,
        )
        live_asr_language = _normalize_optional_language(_cfg(config, "LIVE_ASR_LANGUAGE"))
        runner_settings = LiveASRRunnerSettings.from_live_config(
            {
                "timing": {
                    "emit_min_ms": int(_cfg(config, "LIVE_ROLLING_MIN_EMIT_INTERVAL_MS")),
                },
                "rolling": {
                    "min_infer_audio_ms": int(_cfg(config, "LIVE_ROLLING_MIN_INFER_AUDIO_MS")),
                    "single_segment_commit_min_ms": int(_cfg(config, "LIVE_ROLLING_SINGLE_COMMIT_MIN_MS")),
                    "force_commit_repeats": int(_cfg(config, "LIVE_ROLLING_FORCE_COMMIT_REPEATS")),
                    "max_uncommitted_ms": int(_cfg(config, "LIVE_ROLLING_MAX_UNCOMMITTED_MS")),
                    "hard_clip_keep_tail_ms": int(_cfg(config, "LIVE_ROLLING_HARD_CLIP_KEEP_TAIL_MS")),
                    "max_decode_window_ms": int(_cfg(config, "LIVE_ROLLING_MAX_DECODE_WINDOW_MS")),
                    "buffer_trim_threshold_ms": int(_cfg(config, "LIVE_ROLLING_BUFFER_TRIM_THRESHOLD_MS")),
                    "buffer_trim_drop_ms": int(_cfg(config, "LIVE_ROLLING_BUFFER_TRIM_DROP_MS")),
                    "min_new_audio_ms": int(_cfg(config, "LIVE_ROLLING_MIN_NEW_AUDIO_MS")),
                    "pacing": {
                        "base_emit_ms": int(_cfg(config, "LIVE_ROLLING_PACING_BASE_EMIT_MS")),
                        "startup": {
                            "duration_ms": int(_cfg(config, "LIVE_ROLLING_PACING_STARTUP_DURATION_MS")),
                            "emit_ms": int(_cfg(config, "LIVE_ROLLING_PACING_STARTUP_EMIT_MS")),
                            "min_infer_audio_ms": int(_cfg(config, "LIVE_ROLLING_PACING_STARTUP_MIN_INFER_AUDIO_MS")),
                            "min_new_audio_ms": int(_cfg(config, "LIVE_ROLLING_PACING_STARTUP_MIN_NEW_AUDIO_MS")),
                        },
                    },
                    "vad": {
                        "enabled": bool(_cfg(config, "LIVE_ROLLING_VAD_ENABLED")),
                        "venv": _normalize_optional_text(_cfg(config, "LIVE_ROLLING_VAD_VENV")),
                        "threshold": float(_cfg(config, "LIVE_ROLLING_VAD_THRESHOLD")),
                        "max_speech_duration_s": float(_cfg(config, "LIVE_ROLLING_VAD_MAX_SPEECH_DURATION_S")),
                        "min_speech_ms": int(_cfg(config, "LIVE_ROLLING_VAD_MIN_SPEECH_MS")),
                        "hangover_ms": int(_cfg(config, "LIVE_ROLLING_VAD_HANGOVER_MS")),
                    },
                    "speech_gate": {
                        "silence_enter_ms": int(_cfg(config, "LIVE_ROLLING_SPEECH_GATE_SILENCE_ENTER_MS")),
                        "rearm_hits": int(_cfg(config, "LIVE_ROLLING_SPEECH_GATE_REARM_HITS")),
                        "rearm_window_ms": int(_cfg(config, "LIVE_ROLLING_SPEECH_GATE_REARM_WINDOW_MS")),
                        "force_commit_silence_ms": int(
                            _cfg(config, "LIVE_ROLLING_SPEECH_GATE_FORCE_COMMIT_SILENCE_MS")
                        ),
                    },
                },
            }
        )

        self.rt = _RollingRuntime(session_live_asr_language=live_asr_language)
        self.rt.runner = LiveASRRunner(
            audio_format=audio_format,
            settings=runner_settings,
            language=live_asr_language,
        )
        self._ctx = {
            "LIVE_AUDIO_SAMPLE_RATE_HZ": live_audio_sample_rate_hz,
            "LIVE_AUDIO_CHANNELS": live_audio_channels,
            "LIVE_AUDIO_SAMPLE_WIDTH_BYTES": live_audio_sample_width_bytes,
            "LIVE_DRAIN_WAIT_S": float(_cfg(config, "LIVE_DRAIN_WAIT_S")),
            "LIVE_ASR_LANGUAGE": live_asr_language,
        }

    async def _emit_result_event(self, *, force: bool = False) -> bool:
        rt = self.rt
        try:
            result_payload = self.live_sessions.live_result_payload(self.session_id)
        except Exception:
            return False
        envelope = build_live_result_envelope(
            session_id=str(self.session_id),
            result_payload=result_payload,
            rooted_path_cb=self.rooted_path_cb,
        )
        envelope["protocol_version"] = PROTOCOL_VERSION
        try:
            signature = json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        except Exception:
            signature = ""
        if not force and signature and signature == rt.last_result_event_signature:
            return False
        try:
            await self._send_event(result_event(self.session_id, envelope=envelope))
        except Exception:
            return False
        if signature:
            rt.last_result_event_signature = signature
        return True

    def _sync_counts_from_result(self, result: dict[str, Any]) -> None:
        rt = self.rt
        rt.rolling_chunks_total = int(max(0, int(result.get("chunks_total") or rt.rolling_chunks_total)))
        rt.rolling_chunks_done = int(max(0, int(result.get("chunks_done") or rt.rolling_chunks_done)))
        rt.rolling_chunks_failed = int(max(0, int(result.get("chunks_failed") or rt.rolling_chunks_failed)))

    def _guardrail_metrics(self) -> dict[str, int]:
        runner = self.rt.runner
        if runner is None:
            return {}
        return {str(k): int(max(0, int(v))) for k, v in runner.guardrail_metrics.items()}

    def _engine_runtime_payload(self) -> dict[str, Any]:
        rt = self.rt
        runner = rt.runner
        if runner is None:
            return {}
        return runner.engine_runtime_payload(now_mono=time.monotonic())

    def _update_state(self) -> None:
        rt = self.rt
        with contextlib.suppress(Exception):
            self.live_sessions.update_live_state(
                self.session_id,
                recording_state=rt.recording_state,
                recording_path=rt.recording_path,
                recording_bytes=rt.recording_bytes,
                recording_duration_ms=rt.recording_duration_ms,
                chunk_index_next=rt.rolling_commit_index_next,
                chunks_total=rt.rolling_chunks_total,
                chunks_done=rt.rolling_chunks_done,
                chunks_failed=rt.rolling_chunks_failed,
                finalization_state=rt.finalization_state,
                batch_job_id="",
                gpu_proxy_transcribe_s=rt.rolling_gpu_proxy_transcribe_s,
                gpu_proxy_pipeline_s=rt.rolling_gpu_proxy_pipeline_s,
            )
        with contextlib.suppress(Exception):
            self.live_sessions.set_live_engine_runtime(
                self.session_id,
                runtime=self._engine_runtime_payload(),
            )
        if str(rt.finalization_state or "").strip().lower() in {"ready", "error", "finalized"}:
            with contextlib.suppress(Exception):
                self.live_sessions.clear_live_preview(self.session_id)

    async def _update_state_and_emit_result(self, *, force_result: bool = False) -> None:
        self._update_state()
        await self._emit_result_event(force=force_result)

    def _apply_recording_snapshot(self, snapshot: Any, *, state: str | None = None) -> None:
        rt = self.rt
        rt.recording_path = str(snapshot.wav_path)
        rt.recording_bytes = int(snapshot.bytes_written)
        rt.recording_duration_ms = int(snapshot.duration_ms)
        if state is not None:
            rt.recording_state = state

    def _archive_current_result(self, *, close_reason: str) -> dict[str, Any]:
        rt = self.rt
        try:
            live_result = self.live_sessions.live_result_payload(self.session_id)
        except Exception:
            return {}
        if not live_result:
            return {}
        has_content = (
            bool(live_result.get("final_segments"))
            or int(live_result.get("chunks_total") or 0) > 0
            or int(max(0, rt.recording_duration_ms)) > 0
        )
        if not has_content:
            return live_result
        self.live_sessions.archive_transcript(
            self.session_id,
            close_reason=str(close_reason or rt.stop_reason or "closed"),
            final_segments=[
                dict(seg)
                for seg in (live_result.get("final_segments") or [])
                if isinstance(seg, dict)
            ],
            transcript_revision=int(max(0, int(live_result.get("transcript_revision") or 0))),
            recording_path=str(rt.recording_path or ""),
            recording_bytes=int(max(0, rt.recording_bytes)),
            recording_duration_ms=int(max(0, rt.recording_duration_ms)),
            chunks_total=int(max(0, rt.rolling_chunks_total)),
            chunks_done=int(max(0, rt.rolling_chunks_done)),
            chunks_failed=int(max(0, rt.rolling_chunks_failed)),
            finalization_state=str(rt.finalization_state or ""),
            batch_job_id="",
        )
        return live_result

    def _finalize_recording(self) -> None:
        rt = self.rt
        if rt.recording_finalized:
            return
        rt.finalization_state = "finalizing"
        if rt.recorder is not None:
            try:
                rs = rt.recorder.finalize()
                self._apply_recording_snapshot(rs, state="finalized")
                if rt.finalization_state != "error":
                    rt.finalization_state = "recording_finalized"
            except Exception:
                rt.recording_state = "error"
                rt.finalization_state = "error"
        else:
            if rt.finalization_state != "error":
                rt.finalization_state = "idle"
        rt.recording_finalized = True
        self._update_state()

    def _commit_preview_tail_if_needed(
        self,
        *,
        include_recording_end: bool = True,
        max_t1_ms: int | None = None,
        speech_gate_forced: bool = False,
    ) -> bool:
        rt = self.rt
        runner = rt.runner
        if runner is None:
            return False
        if runner.recording_duration_ms <= runner.processed_offset_ms:
            return False
        preview = runner.transcript_state.preview
        preview_text = str(preview.text or "").strip()
        if not preview_text:
            preview_text = str(runner.preview_history.last_preview_text or "").strip()
        if not preview_text:
            return False

        preview_audio_end_ms = int(max(0, int(preview.audio_end_ms or 0)))
        if preview_audio_end_ms <= 0:
            preview_audio_end_ms = int(max(0, runner.preview_history.last_preview_audio_end_fallback_ms))

        commit_t0 = int(max(0, runner.processed_offset_ms, runner.preview_history.last_preview_source_t0_ms))
        commit_t1_candidates = [int(commit_t0), int(preview_audio_end_ms)]
        if include_recording_end:
            commit_t1_candidates.append(int(max(0, runner.recording_duration_ms)))
        commit_t1 = int(max(commit_t1_candidates))
        if max_t1_ms is not None:
            commit_t1 = int(min(commit_t1, int(max(0, max_t1_ms))))
        if commit_t1 <= commit_t0:
            if include_recording_end:
                commit_t1 = int(max(commit_t0 + 1, runner.recording_duration_ms))
            else:
                return False
        seg = {
            "segment_id": f"s{int(max(0, rt.rolling_commit_index_next)) + 1:04d}",
            "text": preview_text,
            "t0_ms": int(commit_t0),
            "t1_ms": int(commit_t1),
        }
        if not self._record_commit_row(
            t0_ms=int(commit_t0),
            t1_ms=int(commit_t1),
            text=preview_text,
            segments=[seg],
            state="ready",
            error="",
            reason="rolling_context_tail_preview_commit",
            chunk_duration_ms=int(max(0, commit_t1 - commit_t0)),
        ):
            return False
        runner.commit_preview_tail(
            include_recording_end=include_recording_end,
            max_t1_ms=max_t1_ms,
            speech_gate_forced=speech_gate_forced,
        )
        return True

    def _record_commit_row(
        self,
        *,
        t0_ms: int,
        t1_ms: int,
        text: str,
        segments: list[dict[str, Any]],
        state: str,
        error: str,
        reason: str,
        chunk_duration_ms: int | None = None,
    ) -> bool:
        rt = self.rt
        try:
            result = self.live_sessions.record_live_commit(
                self.session_id,
                chunk_index=int(max(0, rt.rolling_commit_index_next)),
                t0_ms=int(max(0, t0_ms)),
                t1_ms=int(max(max(0, t0_ms), t1_ms)),
                text=str(text or ""),
                segments=[dict(seg) for seg in segments if isinstance(seg, dict)],
                state=str(state or ""),
                error=str(error or ""),
                reason=str(reason or ""),
                chunk_duration_ms=(
                    None if chunk_duration_ms is None else int(max(0, int(chunk_duration_ms)))
                ),
            )
        except Exception:
            return False
        self._sync_counts_from_result(result)
        rt.rolling_commit_index_next += 1
        return True

    async def _enqueue_inference(self, *, force: bool = False) -> None:
        config = self.config
        rt = self.rt
        if rt.chunk_bridge is None or rt.runner is None:
            return
        now_mono = time.monotonic()
        if str(rt.recording_state or "") not in {"recording", "finalizing"} or rt.rolling_inflight is not None:
            return

        forced_preview_committed = False
        prev_hard_clip_count = int(max(0, int(rt.runner.guardrail_metrics.get("hard_clip_count") or 0)))

        dispatch_decision = rt.runner.maybe_dispatch_work(
            now_mono=now_mono,
            force=bool(force),
        )
        gate_decision = dispatch_decision.speech_gate_decision
        runner_guardrails = dict(rt.runner.guardrail_metrics)
        if str(dispatch_decision.error or "").strip():
            rt.finalization_state = "error"
            await self._update_state_and_emit_result()
            return
        if gate_decision is not None and gate_decision.force_commit_requested:
            committed = self._commit_preview_tail_if_needed(
                include_recording_end=False,
                max_t1_ms=(int(rt.runner.last_submitted_t1_ms) if int(rt.runner.last_submitted_t1_ms) > 0 else None),
                speech_gate_forced=True,
            )
            if committed:
                forced_preview_committed = True
        decision = dispatch_decision.work_decision

        if decision.reason in {"quiet_waiting_rearm", "no_recent_speech"}:
            if forced_preview_committed:
                await self._update_state_and_emit_result()
            return

        hard_clip_count = int(max(0, int(runner_guardrails.get("hard_clip_count") or 0)))
        if hard_clip_count > prev_hard_clip_count:
            with contextlib.suppress(Exception):
                self.live_sessions.clear_live_preview(self.session_id)

        if decision.reason == "already_inflight":
            return
        if decision.reason in {
            "emit_interval_wait",
            "pacing_slot_wait",
            "insufficient_new_audio",
            "window_no_progress",
            "no_unprocessed_audio",
            "input_drained",
            "insufficient_unprocessed_audio",
            "empty_audio_window",
        } or decision.work_item is None:
            if forced_preview_committed:
                await self._update_state_and_emit_result()
            return

        work_item = decision.work_item
        infer_seq = int(max(0, work_item.sequence_id))
        infer_t0_ms = int(max(0, work_item.t0_ms))
        infer_t1_ms = int(max(infer_t0_ms, work_item.t1_ms))
        pcm = bytes(work_item.pcm16le or b"")
        if not pcm:
            if forced_preview_committed:
                await self._update_state_and_emit_result()
            return

        try:
            job = await asyncio.to_thread(
                rt.chunk_bridge.enqueue_chunk_pcm16,
                session_id=self.session_id,
                chunk_index=infer_seq,
                t0_ms=infer_t0_ms,
                t1_ms=infer_t1_ms,
                pcm16le=pcm,
                language=rt.session_live_asr_language,
                asr_beam_size=_cfg(config, "LIVE_ASR_BEAM_SIZE"),
                asr_chunk_size=_cfg(config, "LIVE_ASR_CHUNK_SIZE"),
                asr_backend=_normalize_optional_text(_cfg(config, "LIVE_ASR_BACKEND")),
            )
        except Exception:
            rt.runner.rollback_inflight_work(sequence_id=infer_seq)
            rt.finalization_state = "error"
            await self._update_state_and_emit_result()
            return

        rt.rolling_inflight = {
            "seq": int(infer_seq),
            "job_id": str(job.job_id),
            "t0_ms": int(infer_t0_ms),
            "t1_ms": int(infer_t1_ms),
        }
        if rt.finalization_state not in {"error", "ready"}:
            rt.finalization_state = "processing_chunks"
        await self._update_state_and_emit_result()

    async def _poll_inference(self) -> None:
        rt = self.rt
        if rt.chunk_bridge is None or rt.rolling_inflight is None or rt.runner is None:
            return

        inflight = dict(rt.rolling_inflight or {})
        seq = int(inflight.get("seq") or 0)
        job_id = str(inflight.get("job_id") or "")
        t0_ms = int(max(0, int(inflight.get("t0_ms") or 0)))
        t1_ms = int(max(t0_ms, int(inflight.get("t1_ms") or t0_ms)))

        try:
            has_terminal = rt.chunk_bridge.has_terminal_result(job_id)
        except Exception:
            return
        if not bool(has_terminal):
            return
        try:
            poll = await asyncio.to_thread(rt.chunk_bridge.take_terminal_result, job_id, t0_offset_ms=t0_ms)
        except Exception:
            return
        if not bool(poll.done):
            return
        if seq < rt.rolling_last_applied_seq:
            rt.runner.clear_inflight_work(sequence_id=seq)
            rt.rolling_inflight = None
            await self._update_state_and_emit_result()
            return

        poll_status = dict(poll.status or {})
        for status_key, state_key in (
            ("asr_timing_whisperx_transcribe_call_s", "rolling_gpu_proxy_transcribe_s"),
            ("asr_timing_whisperx_total_s", "rolling_gpu_proxy_pipeline_s"),
        ):
            value = _safe_float(poll_status.get(status_key))
            if value is not None:
                setattr(rt, state_key, getattr(rt, state_key) + value)

        if bool(poll.ok):
            raw_segments = poll.segments if isinstance(poll.segments, list) else []
            apply = rt.runner.apply_result(
                EngineASRResult(
                    sequence_id=int(seq),
                    t0_ms=int(t0_ms),
                    t1_ms=int(t1_ms),
                    ok=True,
                    text=str(poll.text or ""),
                    segments=tuple(
                        TranscriptSegment.from_dict(seg)
                        for seg in raw_segments
                        if isinstance(seg, dict)
                    ),
                )
            )

            if apply.reason == "commit_applied" and apply.committed_segments:
                normalized_commit_segments = [seg.to_dict() for seg in apply.committed_segments]
                commit_t0_ms = int(normalized_commit_segments[0]["t0_ms"])
                commit_t1_ms = int(normalized_commit_segments[-1]["t1_ms"])
                commit_text = "\n".join(
                    str(seg.get("text") or "").strip()
                    for seg in normalized_commit_segments
                    if str(seg.get("text") or "").strip()
                ).strip()
                if commit_text and not self._record_commit_row(
                    t0_ms=commit_t0_ms,
                    t1_ms=commit_t1_ms,
                    text=commit_text,
                    segments=normalized_commit_segments,
                    state="ready",
                    error="",
                    reason=str(apply.commit_reason or "rolling_context_commit"),
                    chunk_duration_ms=max(0, commit_t1_ms - commit_t0_ms),
                ):
                    rt.finalization_state = "error"

            preview_text = str(apply.preview.text or "").strip()
            if apply.reason in {"preview_applied", "commit_applied"} and preview_text:
                try:
                    self.live_sessions.update_live_preview(
                        self.session_id,
                        text=preview_text,
                        preview_seq=int(max(0, seq)),
                        audio_end_ms=int(max(0, apply.preview.audio_end_ms)),
                        append_to_existing=False,
                    )
                except Exception:
                    pass
        else:
            poll_state = str(poll.state or "").strip().lower()
            err = str(poll.error or f"asr_state:{poll_state}" or "asr_error")
            rt.runner.apply_result(
                EngineASRResult(
                    sequence_id=int(seq),
                    t0_ms=int(t0_ms),
                    t1_ms=int(t1_ms),
                    ok=False,
                    error=err,
                )
            )
            self._record_commit_row(
                t0_ms=t0_ms,
                t1_ms=t1_ms,
                text="",
                segments=[],
                state="error",
                error=err,
                reason="rolling_context_error",
            )
            rt.finalization_state = "error"
        rt.rolling_last_applied_seq = int(max(rt.rolling_last_applied_seq, seq))
        rt.rolling_inflight = None
        if rt.finalization_state not in {"error", "ready"}:
            rt.finalization_state = "recording" if str(rt.recording_state or "") == "recording" else "processing_chunks"
        await self._update_state_and_emit_result()

    async def _process_rolling(self, *, force_emit: bool = False) -> None:
        await self._poll_inference()
        await self._enqueue_inference(force=force_emit)

    async def _drain_inflight_only(self) -> None:
        await self._poll_inference()

    async def _open_websocket_session(self) -> bool:
        session_id = self.session_id
        websocket = self.websocket
        live_sessions = self.live_sessions
        rt = self.rt

        try:
            live_sessions.open_websocket(session_id)
        except KeyError:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="session_not_found")
            return False
        except RuntimeError as e:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason=str(e))
            return False

        await websocket.accept()
        self._completion_loop = asyncio.get_running_loop()
        self._completion_ready = asyncio.Event()
        try:
            session_payload = live_sessions.session_payload(session_id)
            snap_lang = _normalize_optional_language((session_payload or {}).get("asr_language"))
            if snap_lang is not None:
                rt.session_live_asr_language = snap_lang
        except Exception:
            pass
        if rt.session_live_asr_language is None:
            rt.session_live_asr_language = self._ctx["LIVE_ASR_LANGUAGE"]

        if rt.runner is not None:
            try:
                rt.runner.ensure_vad_ready()
            except Exception as e:
                try:
                    await websocket.close(code=status.WS_1011_INTERNAL_ERROR, reason="live_vad_init_failed")
                except Exception:
                    pass
                live_sessions.close_session(session_id, reason=f"live_vad_init_failed:{type(e).__name__}")
                return False
        return True

    async def _start_live_runtime(self) -> None:
        session_id = self.session_id
        ctx = self._ctx
        config = self.config
        rt = self.rt

        ready_payload = ready_event(
            session_id,
            message="Live websocket connected. Send binary PCM16 frames and JSON controls.",
            engine="rolling_context",
        )
        ready_payload["live_engine"] = "rolling_context"
        await self._send_event(ready_payload)

        try:
            rt.recorder = LiveWavRecorder(
                session_id=session_id,
                sample_rate_hz=ctx["LIVE_AUDIO_SAMPLE_RATE_HZ"],
                channels=ctx["LIVE_AUDIO_CHANNELS"],
            )
            rec_snap = rt.recorder.start()
            rt.chunk_bridge = LiveChunkBatchBridge(
                sample_rate_hz=ctx["LIVE_AUDIO_SAMPLE_RATE_HZ"],
                channels=ctx["LIVE_AUDIO_CHANNELS"],
                language=rt.session_live_asr_language,
                diarize_enabled=bool(_cfg(config, "LIVE_DIARIZE_ENABLED")),
                diarize_speaker_mode=str(_cfg(config, "LIVE_DIARIZE_SPEAKER_MODE") or "fixed").strip().lower(),
                diarize_min_speakers=int(max(1, int(_cfg(config, "LIVE_DIARIZE_MIN_SPEAKERS")))),
                diarize_max_speakers=int(max(1, int(_cfg(config, "LIVE_DIARIZE_MAX_SPEAKERS")))),
            )
            rt.chunk_bridge.start_completion_stream(
                session_id=session_id,
                on_terminal_event=self._notify_terminal_ready,
            )
            self._apply_recording_snapshot(rec_snap, state="recording")
            rt.finalization_state = "recording"
            await self._update_state_and_emit_result(force_result=True)
        except Exception:
            rt.recorder = None
            rt.chunk_bridge = None
            rt.recording_state = "error"
            rt.finalization_state = "error"
            await self._update_state_and_emit_result(force_result=True)

    async def _maybe_send_audio_stats(self, snapshot: dict[str, Any]) -> None:
        should_emit_stats = snapshot["frames_received"] == 1 or (snapshot["frames_received"] % 50) == 0
        if not should_emit_stats:
            return

        session_id = self.session_id
        rt = self.rt
        stats_payload = stats_event(
            session_id,
            bytes_received=snapshot["bytes_received"],
            frames_received=snapshot["frames_received"],
            controls_received=snapshot["controls_received"],
            uptime_s=snapshot["age_s"],
            live_engine="rolling_context",
            live_mode="rolling_context",
            live_recording_state=str(rt.recording_state or ""),
            live_recording_bytes=int(max(0, rt.recording_bytes)),
            live_recording_duration_ms=int(max(0, rt.recording_duration_ms)),
            live_commit_index_next=int(max(0, rt.rolling_commit_index_next)),
            live_commits_total=int(max(0, rt.rolling_chunks_total)),
            live_commits_done=int(max(0, rt.rolling_chunks_done)),
            live_commits_failed=int(max(0, rt.rolling_chunks_failed)),
            live_finalization_state=str(rt.finalization_state or ""),
            live_jobs_enabled=True,
            live_jobs_pending=(1 if rt.rolling_inflight is not None else 0),
            live_inflight=bool(rt.rolling_inflight is not None),
            rolling_guardrails=self._guardrail_metrics(),
        )
        await self._send_event(stats_payload)

    async def _handle_audio_frame(self, raw_bytes: bytes) -> None:
        ctx = self._ctx
        rt = self.rt

        snapshot = self.live_sessions.record_audio(self.session_id, byte_count=len(raw_bytes))
        raw = bytes(raw_bytes or b"")
        sample_width_bytes = int(ctx["LIVE_AUDIO_SAMPLE_WIDTH_BYTES"])
        if (len(raw) % sample_width_bytes) != 0:
            raw = raw[: len(raw) - (len(raw) % sample_width_bytes)]
        if rt.recorder is not None:
            try:
                rec_snap = rt.recorder.append_pcm16(raw)
                self._apply_recording_snapshot(rec_snap)
            except Exception:
                rt.recording_state = "error"
                rt.finalization_state = "error"
                try:
                    rt.recorder.abort()
                except Exception:
                    pass
                rt.recorder = None
                await self._update_state_and_emit_result()
        if raw and rt.runner is not None:
            rt.runner.ingest_audio(raw)

        await self._process_rolling(force_emit=False)
        await self._maybe_send_audio_stats(snapshot)

    async def _handle_state_control(self, *, control_type: str) -> None:
        state, recording_state = _STATE_CONTROLS[control_type]
        snapshot = self.live_sessions.mark_state(self.session_id, state=state)
        rt = self.rt
        rt.recording_state = recording_state
        if recording_state == "recording" and rt.finalization_state not in {"error", "ready"}:
            rt.finalization_state = "recording"
        await self._update_state_and_emit_result()
        await self._send_event(control_ack_event(self.session_id, control_type=control_type, state=snapshot["state"]))

    async def _drain_stop_inflight(self) -> None:
        rt = self.rt
        wait_deadline = time.monotonic() + max(0.0, self._ctx["LIVE_DRAIN_WAIT_S"])
        while time.monotonic() < wait_deadline:
            await self._drain_inflight_only()
            if rt.rolling_inflight is None:
                break
            remaining_s = max(0.0, wait_deadline - time.monotonic())
            await self._wait_for_completion_or_timeout(timeout_s=min(0.1, remaining_s))

    async def _handle_stop_control(self) -> None:
        rt = self.rt

        rt.stop_reason = "client_stop"
        live_result: dict[str, Any] = {}
        if rt.recording_state in {"recording", "paused"}:
            rt.recording_state = "finalizing"
        if rt.finalization_state not in {"error", "ready"}:
            rt.finalization_state = "finalizing"
        if rt.runner is not None:
            rt.runner.finalize_input()
        await self._update_state_and_emit_result()
        await self._process_rolling(force_emit=True)
        await self._drain_stop_inflight()

        self._finalize_recording()
        await self._drain_inflight_only()
        self._commit_preview_tail_if_needed()
        if rt.finalization_state != "error":
            rt.finalization_state = "ready"
        await self._update_state_and_emit_result(force_result=True)
        with contextlib.suppress(Exception):
            live_result = self._archive_current_result(close_reason=rt.stop_reason)
            rt.archived_result = True
        await self._emit_result_event(force=True)

        await self._send_event(
            ended_event(
                self.session_id,
                reason=rt.stop_reason,
                transcript_revision=int(max(0, int(live_result.get("transcript_revision") or 0))),
                final_segments_count=len(live_result.get("final_segments") or []),
                final_transcript_url=self.rooted_path_cb(f"/demo/live/sessions/{self.session_id}/final"),
            )
        )
        await self.websocket.close(code=status.WS_1000_NORMAL_CLOSURE)
        rt.websocket_closed = True

    async def _handle_control_message(self, raw_text: str) -> bool:
        ctx = self._ctx
        rt = self.rt

        control_type, obj, parse_err = parse_client_message(raw_text)
        if parse_err:
            await self._send_event(error_event(self.session_id, code=parse_err, message="Invalid control message."))
            return True

        self.live_sessions.record_control(self.session_id)

        if control_type == "ping":
            await self._send_event(pong_event(self.session_id))
            return True

        if control_type == "set_language":
            try:
                next_language = parse_live_asr_language((obj or {}).get("language")) or ""
            except ValueError as e:
                await self._send_event(error_event(self.session_id, code="invalid_language", message=str(e)))
                return True
            snapshot = self.live_sessions.set_asr_language(
                self.session_id,
                asr_language=next_language,
            )
            rt.session_live_asr_language = _normalize_optional_language(snapshot.get("asr_language"))
            if rt.session_live_asr_language is None:
                rt.session_live_asr_language = ctx["LIVE_ASR_LANGUAGE"]
            if rt.runner is not None:
                rt.runner.language = rt.session_live_asr_language
            await self._update_state_and_emit_result(force_result=True)
            await self._send_event(control_ack_event(self.session_id, control_type="set_language", state=snapshot["state"]))
            return True

        if control_type in _STATE_CONTROLS:
            await self._handle_state_control(control_type=control_type)
            return True

        if control_type == "stop":
            await self._handle_stop_control()
            return False

        return True

    async def _run_loop_step(self) -> bool:
        wait_kind, incoming = await self._wait_for_websocket_or_completion()
        if wait_kind == "completion":
            await self._process_rolling(force_emit=False)
            return True

        if incoming is None:
            return True
        if incoming.get("type") == "websocket.disconnect":
            self.rt.stop_reason = "client_disconnected"
            return False

        raw_bytes = incoming.get("bytes")
        if raw_bytes is not None:
            await self._handle_audio_frame(raw_bytes)
            return True

        raw_text = incoming.get("text")
        if raw_text is None:
            await self._send_event(
                error_event(
                    self.session_id,
                    code="invalid_frame",
                    message="Expected binary audio frame or JSON control message.",
                )
            )
            return True

        return await self._handle_control_message(raw_text)

    async def _handle_server_error(self, error: Exception) -> None:
        self.rt.stop_reason = "server_error"
        with contextlib.suppress(Exception):
            await self._send_event(
                error_event(
                    self.session_id,
                    code="internal_error",
                    message=f"{type(error).__name__}: {error}",
                    fatal=True,
                )
            )
        if not self.rt.websocket_closed:
            with contextlib.suppress(Exception):
                await self.websocket.close(code=status.WS_1011_INTERNAL_ERROR)

    async def _drain_cleanup_runtime(self) -> None:
        ctx = self._ctx
        rt = self.rt

        if rt.runner is not None:
            rt.runner.finalize_input()
        await self._drain_inflight_only()
        wait_timeout = 0.0 if rt.stop_reason == "client_stop" else ctx["LIVE_DRAIN_WAIT_S"]
        min_infer_audio_ms = int(rt.runner.settings.rolling.min_infer_audio_ms) if rt.runner is not None else 0
        wait_deadline = time.monotonic() + max(0.0, float(wait_timeout))
        while time.monotonic() < wait_deadline:
            await self._drain_inflight_only()
            remaining_ms = (
                int(max(0, rt.runner.recording_duration_ms - rt.runner.processed_offset_ms))
                if rt.runner is not None
                else 0
            )
            if (rt.rolling_inflight is None) and remaining_ms < min_infer_audio_ms:
                break
            remaining_s = max(0.0, wait_deadline - time.monotonic())
            await self._wait_for_completion_or_timeout(timeout_s=min(0.1, remaining_s))

    async def _cleanup_after_run(self) -> None:
        rt = self.rt

        self._finalize_recording()
        await self._drain_cleanup_runtime()
        self._commit_preview_tail_if_needed()
        if rt.finalization_state not in {"error", "ready"}:
            rt.finalization_state = "ready"
        await self._update_state_and_emit_result(force_result=True)

        if not rt.archived_result:
            with contextlib.suppress(Exception):
                self._archive_current_result(close_reason=rt.stop_reason)
                rt.archived_result = True
        await self._emit_result_event(force=True)

        self.live_sessions.close_session(self.session_id, reason=rt.stop_reason)
        if rt.chunk_bridge is not None:
            with contextlib.suppress(Exception):
                rt.chunk_bridge.stop_completion_stream()
        if rt.recorder is not None and not rt.recording_finalized:
            with contextlib.suppress(Exception):
                rt.recorder.abort()
        self._completion_ready = None
        self._completion_loop = None

    async def run(self) -> None:
        self._configure_context()
        rt = self.rt

        if not await self._open_websocket_session():
            return

        try:
            await self._start_live_runtime()
            while await self._run_loop_step():
                pass
        except WebSocketDisconnect:
            rt.stop_reason = "client_disconnected"
        except Exception as e:
            await self._handle_server_error(e)
        finally:
            await self._cleanup_after_run()

async def run_live_session_ws(
    session_id: str,
    websocket: WebSocket,
    *,
    live_sessions: Any,
    rooted_path_cb: Callable[[str], str],
    config: Mapping[str, Any],
) -> None:
    session = LiveWebSocketSession(
        session_id,
        websocket,
        live_sessions=live_sessions,
        rooted_path_cb=rooted_path_cb,
        config=config,
    )
    await session.run()
