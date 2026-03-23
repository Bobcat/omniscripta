from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from fastapi import WebSocket, WebSocketDisconnect, status

from live.engine.chunk_transcribe import LiveChunkBatchBridge
from live._util import _normalize_optional_language, _safe_float
from live.engine.vad_silero import LiveSileroVadGate, LiveSileroVadSettings
from live.protocol import (
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
from live.engine.recordings import LiveWavRecorder


def _cfg(config: Mapping[str, Any], key: str) -> Any:
    if key not in config:
        raise RuntimeError(f"missing_live_engine_config:{key}")
    return config[key]


def _normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


_LIVE_SESSION_LANGUAGE_RE = re.compile(r"^[a-z]{2,3}(?:[-_][a-z0-9]{2,8})?$")


def _parse_control_language(value: Any) -> str:
    text = _normalize_optional_language(value)
    if text is None:
        return ""
    normalized = str(text).lower()
    if normalized in {"auto", "default", "server-default", "server_default"}:
        return ""
    if not _LIVE_SESSION_LANGUAGE_RE.match(normalized):
        raise ValueError("language must be empty/auto or a short code like 'en', 'nl', 'pt-br'")
    return normalized


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


@dataclass
class _RollingRuntime:
    recorder: LiveWavRecorder | None = None
    chunk_bridge: LiveChunkBatchBridge | None = None
    vad_gate: LiveSileroVadGate | None = None
    session_live_asr_language: str | None = None

    stop_reason: str = "client_disconnected"
    websocket_closed: bool = False
    archived_result: bool = False

    recording_state: str = "idle"
    recording_path: str = ""
    recording_bytes: int = 0
    recording_duration_ms: int = 0
    finalization_state: str = "idle"
    shadow_disabled_reason: str = ""
    recording_finalized: bool = False

    rolling_pcm: bytearray = field(default_factory=bytearray)
    rolling_pcm_base_ms: int = 0
    rolling_processed_offset_ms: int = 0
    rolling_commit_index_next: int = 0
    rolling_chunks_total: int = 0
    rolling_chunks_done: int = 0
    rolling_chunks_failed: int = 0

    rolling_infer_seq_next: int = 0
    rolling_inflight: dict[str, Any] | None = None
    rolling_last_submitted_t1_ms: int = 0
    rolling_last_applied_seq: int = -1
    rolling_last_emit_mono: float = 0.0
    rolling_last_poll_mono: float = 0.0
    rolling_gpu_proxy_transcribe_s: float = 0.0
    rolling_gpu_proxy_pipeline_s: float = 0.0

    rolling_call_audit_recent: list[dict[str, Any]] = field(default_factory=list)
    rolling_call_audit_summary: dict[str, Any] = field(
        default_factory=lambda: {
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
    )
    rolling_last_preview_signature: str = ""
    rolling_same_preview_repeats: int = 0
    rolling_last_preview_audio_end_ms: int = -1
    rolling_same_preview_audio_repeats: int = 0
    rolling_last_preview_text: str = ""
    rolling_last_preview_audio_end_fallback_ms: int = 0
    rolling_pacing_epoch_mono_ms: int = 0
    rolling_pacing_last_slot_index: int = -1
    rolling_speech_gate_state: str = "active"
    rolling_speech_gate_recent_hits_mono: list[float] = field(default_factory=list)
    rolling_last_recent_speech_mono: float = 0.0
    rolling_speech_gate_rearm_from_ms: int = 0
    rolling_guardrail_metrics: dict[str, int] = field(
        default_factory=lambda: {
            "force_commit_repeats_count": 0,
            "decode_window_cap_count": 0,
            "hard_clip_count": 0,
            "hard_clip_dropped_audio_ms": 0,
            "buffer_trim_count": 0,
            "buffer_trim_dropped_audio_ms": 0,
            "outstanding_limit_skips": 0,
            "emit_interval_skips": 0,
            "pacing_slot_skips": 0,
            "min_new_audio_skips": 0,
            "window_no_progress_skips": 0,
            "stale_completion_ignored_count": 0,
            "vad_checks": 0,
            "vad_speech_allows": 0,
            "vad_hangover_allows": 0,
            "vad_silence_skips": 0,
            "vad_errors": 0,
            "speech_gate_quiet_skips": 0,
            "speech_gate_rearm_count": 0,
            "speech_gate_state_transitions": 0,
            "speech_gate_silence_flush_count": 0,
            "speech_gate_forced_commit_count": 0,
        }
    )
    last_result_event_signature: str = ""


class RollingContextSession:
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

    async def _send_event(self, payload: dict[str, Any]) -> None:
        out = dict(payload)
        try:
            out["seq"] = self.live_sessions.next_seq(self.session_id)
        except KeyError:
            pass
        await self.websocket.send_json(out)

    def _result_envelope_from_snapshot(self, result_snapshot: dict[str, Any], *, live_engine: str) -> dict[str, Any]:
        result = dict(result_snapshot or {})
        effective_engine = str(result.get("live_engine") or live_engine)
        result["live_engine"] = effective_engine
        final_segments = result.get("final_segments")
        has_segments = isinstance(final_segments, list) and any(isinstance(s, dict) for s in final_segments)

        has_recording_wav = False
        raw_recording_path = str(result.get("recording_path") or "").strip()
        if raw_recording_path:
            try:
                wav_candidate = Path(raw_recording_path).expanduser().resolve()
                has_recording_wav = wav_candidate.suffix.lower() == ".wav" and wav_candidate.is_file()
            except Exception:
                has_recording_wav = False

        finalization_state = str(result.get("finalization_state") or "").strip().lower()
        ready_states = {"ready", "finalized", "recording_finalized"}
        if effective_engine == "rolling_context":
            ready_states = {"ready", "finalized"}

        return {
            "protocol_version": PROTOCOL_VERSION,
            "session_id": str(self.session_id),
            "live_engine": effective_engine,
            "result": result,
            "ready": finalization_state in ready_states,
            "can_export_srt": bool(has_segments),
            "can_export_wav": bool(has_recording_wav),
            "transcript_srt_url": self.rooted_path_cb(f"/demo/live/sessions/{self.session_id}/transcript.srt") if has_segments else None,
            "recording_wav_url": self.rooted_path_cb(f"/demo/live/sessions/{self.session_id}/recording.wav") if has_recording_wav else None,
        }

    def _append_log(self, kind: str, **fields: Any) -> None:
        try:
            row = {"kind": str(kind)}
            row.update(fields)
            self.live_sessions.append_stats_log(self.session_id, row)
        except Exception:
            pass

    @staticmethod
    def _preview_signature(value: str) -> str:
        return " ".join(str(value or "").strip().lower().split())

    def _configure_context(self) -> dict[str, Any]:
        config = self.config
        session_id = self.session_id

        live_engine = str(_cfg(config, "LIVE_ENGINE"))
        live_audio_sample_rate_hz = int(_cfg(config, "LIVE_AUDIO_SAMPLE_RATE_HZ"))
        live_audio_channels = int(_cfg(config, "LIVE_AUDIO_CHANNELS"))
        live_audio_sample_width_bytes = int(_cfg(config, "LIVE_AUDIO_SAMPLE_WIDTH_BYTES"))
        live_audio_bytes_per_second = int(_cfg(config, "LIVE_AUDIO_BYTES_PER_SECOND"))
        live_drain_wait_s = float(_cfg(config, "LIVE_DRAIN_WAIT_S"))
        live_post_close_wait_s = float(_cfg(config, "LIVE_POST_CLOSE_WAIT_S"))
        live_asr_language = _normalize_optional_language(_cfg(config, "LIVE_ASR_LANGUAGE"))
        live_diarize_enabled = bool(_cfg(config, "LIVE_DIARIZE_ENABLED"))
        live_diarize_speaker_mode = str(_cfg(config, "LIVE_DIARIZE_SPEAKER_MODE") or "fixed").strip().lower()
        live_diarize_min_speakers = int(max(1, int(_cfg(config, "LIVE_DIARIZE_MIN_SPEAKERS"))))
        live_diarize_max_speakers = int(max(1, int(_cfg(config, "LIVE_DIARIZE_MAX_SPEAKERS"))))
        live_rolling_poll_interval_ms = int(_cfg(config, "LIVE_ROLLING_POLL_INTERVAL_MS"))
        live_rolling_min_infer_audio_ms = int(_cfg(config, "LIVE_ROLLING_MIN_INFER_AUDIO_MS"))
        live_rolling_single_commit_min_ms = int(_cfg(config, "LIVE_ROLLING_SINGLE_COMMIT_MIN_MS"))
        live_rolling_force_commit_repeats = int(_cfg(config, "LIVE_ROLLING_FORCE_COMMIT_REPEATS"))
        live_rolling_max_uncommitted_ms = int(_cfg(config, "LIVE_ROLLING_MAX_UNCOMMITTED_MS"))
        live_rolling_hard_clip_keep_tail_ms = int(_cfg(config, "LIVE_ROLLING_HARD_CLIP_KEEP_TAIL_MS"))
        live_rolling_max_decode_window_ms = int(_cfg(config, "LIVE_ROLLING_MAX_DECODE_WINDOW_MS"))
        live_rolling_buffer_trim_threshold_ms = int(_cfg(config, "LIVE_ROLLING_BUFFER_TRIM_THRESHOLD_MS"))
        live_rolling_buffer_trim_drop_ms = int(_cfg(config, "LIVE_ROLLING_BUFFER_TRIM_DROP_MS"))
        live_rolling_min_new_audio_ms = int(_cfg(config, "LIVE_ROLLING_MIN_NEW_AUDIO_MS"))
        live_rolling_min_emit_interval_ms = int(_cfg(config, "LIVE_ROLLING_MIN_EMIT_INTERVAL_MS"))
        live_rolling_pacing_base_emit_ms = int(_cfg(config, "LIVE_ROLLING_PACING_BASE_EMIT_MS"))
        live_rolling_pacing_startup_duration_ms = int(_cfg(config, "LIVE_ROLLING_PACING_STARTUP_DURATION_MS"))
        live_rolling_pacing_startup_emit_ms = int(_cfg(config, "LIVE_ROLLING_PACING_STARTUP_EMIT_MS"))
        live_rolling_pacing_startup_min_infer_audio_ms = int(_cfg(config, "LIVE_ROLLING_PACING_STARTUP_MIN_INFER_AUDIO_MS"))
        live_rolling_pacing_startup_min_new_audio_ms = int(_cfg(config, "LIVE_ROLLING_PACING_STARTUP_MIN_NEW_AUDIO_MS"))
        live_rolling_vad_enabled = bool(_cfg(config, "LIVE_ROLLING_VAD_ENABLED"))
        live_rolling_vad_whisperx_venv = _normalize_optional_text(_cfg(config, "LIVE_ROLLING_VAD_WHISPERX_VENV"))
        live_rolling_vad_threshold = float(_cfg(config, "LIVE_ROLLING_VAD_THRESHOLD"))
        live_rolling_vad_max_speech_duration_s = float(_cfg(config, "LIVE_ROLLING_VAD_MAX_SPEECH_DURATION_S"))
        live_rolling_vad_min_speech_ms = int(_cfg(config, "LIVE_ROLLING_VAD_MIN_SPEECH_MS"))
        live_rolling_vad_hangover_ms = int(_cfg(config, "LIVE_ROLLING_VAD_HANGOVER_MS"))
        live_rolling_speech_gate_silence_enter_ms = int(_cfg(config, "LIVE_ROLLING_SPEECH_GATE_SILENCE_ENTER_MS"))
        live_rolling_speech_gate_rearm_hits = int(_cfg(config, "LIVE_ROLLING_SPEECH_GATE_REARM_HITS"))
        live_rolling_speech_gate_rearm_window_ms = int(_cfg(config, "LIVE_ROLLING_SPEECH_GATE_REARM_WINDOW_MS"))
        live_rolling_speech_gate_force_commit_silence_ms = int(
            _cfg(config, "LIVE_ROLLING_SPEECH_GATE_FORCE_COMMIT_SILENCE_MS")
        )

        poll_interval_s = max(0.02, float(live_rolling_poll_interval_ms) / 1000.0)
        min_new_audio_ms = int(max(0, live_rolling_min_new_audio_ms))
        single_segment_commit_min_ms = int(max(live_rolling_min_infer_audio_ms, live_rolling_single_commit_min_ms))
        force_commit_repeats = int(max(1, live_rolling_force_commit_repeats))
        max_decode_window_ms = int(max(live_rolling_min_infer_audio_ms, live_rolling_max_decode_window_ms))
        max_uncommitted_ms = int(max(live_rolling_min_infer_audio_ms, live_rolling_max_uncommitted_ms))
        if max_uncommitted_ms <= max_decode_window_ms:
            max_uncommitted_ms = int(max_decode_window_ms + live_rolling_min_infer_audio_ms)
        hard_clip_keep_tail_ms = int(
            max(live_rolling_min_infer_audio_ms, live_rolling_hard_clip_keep_tail_ms, single_segment_commit_min_ms)
        )
        buffer_trim_threshold_ms = int(max(max_decode_window_ms, live_rolling_buffer_trim_threshold_ms))
        buffer_trim_drop_ms = int(max(live_rolling_min_infer_audio_ms, live_rolling_buffer_trim_drop_ms))
        pacing_base_emit_ms = int(max(1, live_rolling_pacing_base_emit_ms))
        startup_duration_ms = int(max(0, live_rolling_pacing_startup_duration_ms))
        startup_emit_ms = int(max(1, live_rolling_pacing_startup_emit_ms))
        startup_min_infer_audio_ms = int(max(0, live_rolling_pacing_startup_min_infer_audio_ms))
        startup_min_new_audio_ms = int(max(0, live_rolling_pacing_startup_min_new_audio_ms))
        vad_enabled = bool(live_rolling_vad_enabled)
        vad_threshold = float(max(0.0, min(1.0, live_rolling_vad_threshold)))
        vad_max_speech_duration_s = float(max(0.1, live_rolling_vad_max_speech_duration_s))
        vad_min_speech_ms = int(max(0, live_rolling_vad_min_speech_ms))
        vad_hangover_ms = int(max(0, live_rolling_vad_hangover_ms))
        speech_gate_silence_enter_ms = int(max(100, live_rolling_speech_gate_silence_enter_ms))
        speech_gate_rearm_hits = int(max(1, live_rolling_speech_gate_rearm_hits))
        speech_gate_rearm_window_ms = int(max(100, live_rolling_speech_gate_rearm_window_ms))
        speech_gate_force_commit_silence_ms = int(max(100, live_rolling_speech_gate_force_commit_silence_ms))
        speech_gate_force_threshold_ms = int(max(speech_gate_silence_enter_ms, speech_gate_force_commit_silence_ms))
        speech_gate_rearm_window_s = float(speech_gate_rearm_window_ms) / 1000.0
        vad_settings = LiveSileroVadSettings(
            enabled=bool(vad_enabled),
            whisperx_venv=_normalize_optional_text(live_rolling_vad_whisperx_venv),
            threshold=vad_threshold,
            max_speech_duration_s=vad_max_speech_duration_s,
            min_speech_ms=vad_min_speech_ms,
            hangover_ms=vad_hangover_ms,
        )
        pacing_effective_emit_ms = int(max(live_rolling_min_emit_interval_ms, pacing_base_emit_ms))
        pacing_phase_seed = int.from_bytes(
            hashlib.sha1(str(session_id).encode("utf-8")).digest()[:8],
            byteorder="big",
            signed=False,
        )
        pacing_phase_ms = int(pacing_phase_seed % int(max(1, pacing_effective_emit_ms)))

        self.rt = _RollingRuntime(
            session_live_asr_language=live_asr_language,
            rolling_speech_gate_state=("quiet" if vad_enabled else "active"),
        )
        self._ctx = {
            "LIVE_ENGINE": live_engine,
            "LIVE_AUDIO_SAMPLE_RATE_HZ": live_audio_sample_rate_hz,
            "LIVE_AUDIO_CHANNELS": live_audio_channels,
            "LIVE_AUDIO_SAMPLE_WIDTH_BYTES": live_audio_sample_width_bytes,
            "LIVE_AUDIO_BYTES_PER_SECOND": live_audio_bytes_per_second,
            "LIVE_DRAIN_WAIT_S": live_drain_wait_s,
            "LIVE_POST_CLOSE_WAIT_S": live_post_close_wait_s,
            "LIVE_ASR_LANGUAGE": live_asr_language,
            "LIVE_DIARIZE_ENABLED": live_diarize_enabled,
            "LIVE_DIARIZE_SPEAKER_MODE": live_diarize_speaker_mode,
            "LIVE_DIARIZE_MIN_SPEAKERS": live_diarize_min_speakers,
            "LIVE_DIARIZE_MAX_SPEAKERS": live_diarize_max_speakers,
            "LIVE_ROLLING_POLL_INTERVAL_MS": live_rolling_poll_interval_ms,
            "LIVE_ROLLING_MIN_INFER_AUDIO_MS": live_rolling_min_infer_audio_ms,
            "LIVE_ROLLING_MIN_EMIT_INTERVAL_MS": live_rolling_min_emit_interval_ms,
            "poll_interval_s": poll_interval_s,
            "min_new_audio_ms": min_new_audio_ms,
            "single_segment_commit_min_ms": single_segment_commit_min_ms,
            "force_commit_repeats": force_commit_repeats,
            "max_decode_window_ms": max_decode_window_ms,
            "max_uncommitted_ms": max_uncommitted_ms,
            "hard_clip_keep_tail_ms": hard_clip_keep_tail_ms,
            "buffer_trim_threshold_ms": buffer_trim_threshold_ms,
            "buffer_trim_drop_ms": buffer_trim_drop_ms,
            "pacing_base_emit_ms": pacing_base_emit_ms,
            "startup_duration_ms": startup_duration_ms,
            "startup_emit_ms": startup_emit_ms,
            "startup_min_infer_audio_ms": startup_min_infer_audio_ms,
            "startup_min_new_audio_ms": startup_min_new_audio_ms,
            "vad_enabled": vad_enabled,
            "vad_threshold": vad_threshold,
            "vad_max_speech_duration_s": vad_max_speech_duration_s,
            "vad_min_speech_ms": vad_min_speech_ms,
            "vad_hangover_ms": vad_hangover_ms,
            "speech_gate_silence_enter_ms": speech_gate_silence_enter_ms,
            "speech_gate_rearm_hits": speech_gate_rearm_hits,
            "speech_gate_rearm_window_ms": speech_gate_rearm_window_ms,
            "speech_gate_force_commit_silence_ms": speech_gate_force_commit_silence_ms,
            "speech_gate_force_threshold_ms": speech_gate_force_threshold_ms,
            "speech_gate_rearm_window_s": speech_gate_rearm_window_s,
            "vad_settings": vad_settings,
            "pacing_effective_emit_ms": pacing_effective_emit_ms,
            "pacing_phase_ms": pacing_phase_ms,
        }
        return self._ctx

    async def _emit_result_event(self, *, force: bool = False) -> None:
        rt = self.rt
        try:
            result_snapshot = self.live_sessions.live_result_snapshot(self.session_id)
        except Exception:
            return
        envelope = self._result_envelope_from_snapshot(result_snapshot, live_engine=self._ctx["LIVE_ENGINE"])
        try:
            signature = json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        except Exception:
            signature = ""
        if not force and signature and signature == rt.last_result_event_signature:
            return
        try:
            await self._send_event(result_event(self.session_id, envelope=envelope))
        except Exception:
            return
        if signature:
            rt.last_result_event_signature = signature

    def _sync_counts_from_result(self, result: dict[str, Any]) -> None:
        rt = self.rt
        rt.rolling_chunks_total = int(max(0, int(result.get("chunks_total") or rt.rolling_chunks_total)))
        rt.rolling_chunks_done = int(max(0, int(result.get("chunks_done") or rt.rolling_chunks_done)))
        rt.rolling_chunks_failed = int(max(0, int(result.get("chunks_failed") or rt.rolling_chunks_failed)))

    def _set_speech_gate_state(
        self,
        *,
        next_state: str,
        reason: str,
        now_mono: float,
        rearm_from_ms: int | None = None,
    ) -> None:
        rt = self.rt
        safe_next = str(next_state or "").strip().lower()
        if safe_next not in {"quiet", "active", "flush"}:
            return
        if safe_next == rt.rolling_speech_gate_state:
            return
        prev = str(rt.rolling_speech_gate_state or "")
        rt.rolling_speech_gate_state = safe_next
        if safe_next == "quiet":
            rt.rolling_speech_gate_recent_hits_mono = []
            rt.rolling_last_recent_speech_mono = 0.0
            base_rearm_from_ms = int(max(0, rt.rolling_processed_offset_ms))
            if rearm_from_ms is not None:
                base_rearm_from_ms = int(max(base_rearm_from_ms, int(rearm_from_ms)))
            rt.rolling_speech_gate_rearm_from_ms = int(base_rearm_from_ms)
        elif safe_next == "active":
            rt.rolling_speech_gate_recent_hits_mono = []
            rt.rolling_last_recent_speech_mono = float(max(0.0, now_mono))
        rt.rolling_guardrail_metrics["speech_gate_state_transitions"] = int(
            max(0, int(rt.rolling_guardrail_metrics.get("speech_gate_state_transitions") or 0)) + 1
        )
        self._append_log(
            "rolling_speech_gate_transition",
            prev_state=prev,
            next_state=safe_next,
            reason=str(reason or ""),
        )

    def _recent_pcm_window(self, *, end_ms: int, window_ms: int, min_t0_ms: int | None = None) -> bytes:
        ctx = self._ctx
        rt = self.rt
        safe_end_ms = int(max(0, end_ms))
        safe_window_ms = int(max(1, window_ms))
        t0_ms = int(max(rt.rolling_pcm_base_ms, safe_end_ms - safe_window_ms))
        if min_t0_ms is not None:
            t0_ms = int(max(t0_ms, int(max(0, min_t0_ms))))
        t1_ms = int(max(t0_ms, safe_end_ms))
        start_b = _ms_to_byte_offset(
            int(max(0, t0_ms - rt.rolling_pcm_base_ms)),
            bytes_per_second=ctx["LIVE_AUDIO_BYTES_PER_SECOND"],
            sample_width_bytes=ctx["LIVE_AUDIO_SAMPLE_WIDTH_BYTES"],
        )
        end_b = _ms_to_byte_offset(
            int(max(0, t1_ms - rt.rolling_pcm_base_ms)),
            bytes_per_second=ctx["LIVE_AUDIO_BYTES_PER_SECOND"],
            sample_width_bytes=ctx["LIVE_AUDIO_SAMPLE_WIDTH_BYTES"],
        )
        end_b = int(max(start_b, min(end_b, len(rt.rolling_pcm))))
        if end_b <= start_b:
            return b""
        return bytes(rt.rolling_pcm[start_b:end_b])

    def _recent_speech_hit(
        self,
        *,
        now_mono: float,
        end_ms: int,
        min_t0_ms: int | None = None,
        window_ms: int | None = None,
    ) -> dict[str, Any]:
        ctx = self._ctx
        rt = self.rt
        rt.rolling_guardrail_metrics["vad_checks"] = int(
            max(0, int(rt.rolling_guardrail_metrics.get("vad_checks") or 0)) + 1
        )
        effective_window_ms = int(max(1, int(window_ms or ctx["speech_gate_rearm_window_ms"])))
        pcm_recent = self._recent_pcm_window(
            end_ms=end_ms,
            window_ms=effective_window_ms,
            min_t0_ms=min_t0_ms,
        )
        if not pcm_recent:
            rt.rolling_guardrail_metrics["vad_silence_skips"] = int(
                max(0, int(rt.rolling_guardrail_metrics.get("vad_silence_skips") or 0)) + 1
            )
            return {"speech_hit": False, "reason": "empty_recent_window", "speech_ms": 0, "segments_count": 0}
        try:
            decision = (
                rt.vad_gate.should_enqueue_pcm16(
                    pcm_recent,
                    now_mono=now_mono,
                    allow_hangover=False,
                )
                if rt.vad_gate is not None
                else {"allow": False, "reason": "disabled"}
            )
        except Exception as e:
            rt.rolling_guardrail_metrics["vad_errors"] = int(
                max(0, int(rt.rolling_guardrail_metrics.get("vad_errors") or 0)) + 1
            )
            return {
                "speech_hit": False,
                "error": f"{type(e).__name__}: {e}",
                "reason": "vad_error",
                "speech_ms": 0,
                "segments_count": 0,
            }

        allow = bool(decision.get("allow"))
        reason = str(decision.get("reason") or "").strip().lower()
        speech_hit = bool(allow and reason == "speech")
        if speech_hit:
            rt.rolling_guardrail_metrics["vad_speech_allows"] = int(
                max(0, int(rt.rolling_guardrail_metrics.get("vad_speech_allows") or 0)) + 1
            )
        else:
            rt.rolling_guardrail_metrics["vad_silence_skips"] = int(
                max(0, int(rt.rolling_guardrail_metrics.get("vad_silence_skips") or 0)) + 1
            )
        return {
            "speech_hit": speech_hit,
            "reason": reason or ("speech" if speech_hit else "silence"),
            "speech_ms": int(max(0, int(decision.get("speech_ms") or 0))),
            "segments_count": int(max(0, int(decision.get("segments_count") or 0))),
        }

    def _consume_pacing_slot(self, *, now_mono: float) -> bool:
        ctx = self._ctx
        rt = self.rt
        safe_now_ms = int(round(float(max(0.0, now_mono)) * 1000.0))
        if rt.rolling_pacing_epoch_mono_ms <= 0:
            rt.rolling_pacing_epoch_mono_ms = int(max(0, safe_now_ms))

        interval_ms = int(max(1, ctx["pacing_effective_emit_ms"]))
        safe_phase_ms = int(max(0, ctx["pacing_phase_ms"] % interval_ms))
        elapsed_ms = int(max(0, safe_now_ms - int(rt.rolling_pacing_epoch_mono_ms)))
        if elapsed_ms < safe_phase_ms:
            rt.rolling_guardrail_metrics["pacing_slot_skips"] = int(
                max(0, int(rt.rolling_guardrail_metrics.get("pacing_slot_skips") or 0)) + 1
            )
            return False

        slot_index = int((elapsed_ms - safe_phase_ms) // interval_ms)
        if slot_index <= int(rt.rolling_pacing_last_slot_index):
            rt.rolling_guardrail_metrics["pacing_slot_skips"] = int(
                max(0, int(rt.rolling_guardrail_metrics.get("pacing_slot_skips") or 0)) + 1
            )
            return False

        rt.rolling_pacing_last_slot_index = int(slot_index)
        return True

    def _engine_runtime_payload(self) -> dict[str, Any]:
        ctx = self._ctx
        rt = self.rt
        calls_done = int(max(0, int(rt.rolling_call_audit_summary.get("calls_done") or 0)))
        segs_sum = int(max(0, int(rt.rolling_call_audit_summary.get("segments_returned_sum") or 0)))
        ratio_sum = float(max(0.0, float(rt.rolling_call_audit_summary.get("segments_per_s_sum") or 0.0)))
        avg_segs = (float(segs_sum) / float(calls_done)) if calls_done > 0 else None
        avg_ratio = (ratio_sum / float(calls_done)) if calls_done > 0 else None
        inflight_count = 1 if rt.rolling_inflight is not None else 0
        silence_elapsed_ms = None
        if rt.rolling_last_recent_speech_mono > 0.0:
            silence_elapsed_ms = int(max(0.0, float(time.monotonic() - rt.rolling_last_recent_speech_mono) * 1000.0))
        return {
            "inflight": bool(inflight_count > 0),
            "inflight_count": int(inflight_count),
            "recording_duration_ms": int(max(0, rt.recording_duration_ms)),
            "pcm_base_ms": int(max(0, rt.rolling_pcm_base_ms)),
            "processed_offset_ms": int(max(0, rt.rolling_processed_offset_ms)),
            "buffer_audio_ms": int(max(0, int(rt.recording_duration_ms) - int(rt.rolling_pcm_base_ms))),
            "unprocessed_audio_ms": int(max(0, int(rt.recording_duration_ms) - int(rt.rolling_processed_offset_ms))),
            "pacing": {
                "base_emit_ms": int(max(1, ctx["pacing_base_emit_ms"])),
                "effective_emit_ms": int(max(1, ctx["pacing_effective_emit_ms"])),
                "phase_ms": int(max(0, ctx["pacing_phase_ms"])),
                "last_slot_index": int(rt.rolling_pacing_last_slot_index),
                "startup_duration_ms": int(max(0, ctx["startup_duration_ms"])),
                "startup_emit_ms": int(max(1, ctx["startup_emit_ms"])),
                "startup_min_infer_audio_ms": int(max(0, ctx["startup_min_infer_audio_ms"])),
                "startup_min_new_audio_ms": int(max(0, ctx["startup_min_new_audio_ms"])),
            },
            "vad": (
                {
                    "enabled": True,
                    "config": dict(rt.vad_gate.config_payload()),
                    "state": dict(rt.vad_gate.state_payload()),
                }
                if rt.vad_gate is not None
                else {
                    "enabled": False,
                    "config": {
                        "provider": "silero",
                        "threshold": float(ctx["vad_threshold"]),
                        "max_speech_duration_s": float(ctx["vad_max_speech_duration_s"]),
                        "min_speech_ms": int(ctx["vad_min_speech_ms"]),
                        "hangover_ms": int(ctx["vad_hangover_ms"]),
                        "whisperx_venv": str(ctx["vad_settings"].whisperx_venv or ""),
                        "sample_rate_hz": int(ctx["LIVE_AUDIO_SAMPLE_RATE_HZ"]),
                    },
                    "state": {},
                }
            ),
            "speech_gate": {
                "state": str(rt.rolling_speech_gate_state or ""),
                "recent_hits_count": int(max(0, len(rt.rolling_speech_gate_recent_hits_mono))),
                "silence_elapsed_ms": silence_elapsed_ms,
                "rearm_from_ms": int(max(0, rt.rolling_speech_gate_rearm_from_ms)),
                "silence_enter_ms": int(max(100, ctx["speech_gate_silence_enter_ms"])),
                "rearm_hits": int(max(1, ctx["speech_gate_rearm_hits"])),
                "rearm_window_ms": int(max(100, ctx["speech_gate_rearm_window_ms"])),
                "force_commit_silence_ms": int(max(100, ctx["speech_gate_force_commit_silence_ms"])),
            },
            "guardrails": dict(rt.rolling_guardrail_metrics),
            "config": {
                "poll_interval_ms": int(ctx["LIVE_ROLLING_POLL_INTERVAL_MS"]),
                "min_infer_audio_ms": int(ctx["LIVE_ROLLING_MIN_INFER_AUDIO_MS"]),
                "single_segment_commit_min_ms": int(ctx["single_segment_commit_min_ms"]),
                "force_commit_repeats": int(ctx["force_commit_repeats"]),
                "max_uncommitted_ms": int(ctx["max_uncommitted_ms"]),
                "hard_clip_keep_tail_ms": int(ctx["hard_clip_keep_tail_ms"]),
                "max_decode_window_ms": int(ctx["max_decode_window_ms"]),
                "buffer_trim_threshold_ms": int(ctx["buffer_trim_threshold_ms"]),
                "buffer_trim_drop_ms": int(ctx["buffer_trim_drop_ms"]),
                "min_new_audio_ms": int(ctx["min_new_audio_ms"]),
                "min_emit_interval_ms": int(ctx["LIVE_ROLLING_MIN_EMIT_INTERVAL_MS"]),
                "pacing_base_emit_ms": int(max(1, ctx["pacing_base_emit_ms"])),
                "pacing_effective_emit_ms": int(max(1, ctx["pacing_effective_emit_ms"])),
                "pacing_startup_duration_ms": int(max(0, ctx["startup_duration_ms"])),
                "pacing_startup_emit_ms": int(max(1, ctx["startup_emit_ms"])),
                "pacing_startup_min_infer_audio_ms": int(max(0, ctx["startup_min_infer_audio_ms"])),
                "pacing_startup_min_new_audio_ms": int(max(0, ctx["startup_min_new_audio_ms"])),
                "vad_enabled": bool(ctx["vad_enabled"]),
                "vad_threshold": float(ctx["vad_threshold"]),
                "vad_max_speech_duration_s": float(ctx["vad_max_speech_duration_s"]),
                "vad_min_speech_ms": int(ctx["vad_min_speech_ms"]),
                "vad_hangover_ms": int(ctx["vad_hangover_ms"]),
                "vad_whisperx_venv": str(ctx["vad_settings"].whisperx_venv or ""),
                "speech_gate_silence_enter_ms": int(max(100, ctx["speech_gate_silence_enter_ms"])),
                "speech_gate_rearm_hits": int(max(1, ctx["speech_gate_rearm_hits"])),
                "speech_gate_rearm_window_ms": int(max(100, ctx["speech_gate_rearm_window_ms"])),
                "speech_gate_force_commit_silence_ms": int(max(100, ctx["speech_gate_force_commit_silence_ms"])),
                "diarize_enabled": bool(ctx["LIVE_DIARIZE_ENABLED"]),
                "diarize_speaker_mode": str(ctx["LIVE_DIARIZE_SPEAKER_MODE"]),
                "diarize_min_speakers": int(ctx["LIVE_DIARIZE_MIN_SPEAKERS"]),
                "diarize_max_speakers": int(ctx["LIVE_DIARIZE_MAX_SPEAKERS"]),
            },
            "call_audit_summary": {
                "calls_done": calls_done,
                "segments_returned_min": rt.rolling_call_audit_summary.get("segments_returned_min"),
                "segments_returned_max": rt.rolling_call_audit_summary.get("segments_returned_max"),
                "segments_returned_avg": avg_segs,
                "segments_per_s_min": rt.rolling_call_audit_summary.get("segments_per_s_min"),
                "segments_per_s_max": rt.rolling_call_audit_summary.get("segments_per_s_max"),
                "segments_per_s_avg": avg_ratio,
                "outcome_counts": dict(rt.rolling_call_audit_summary.get("outcome_counts") or {}),
            },
            "call_audit_recent": list(rt.rolling_call_audit_recent[-50:]),
        }

    def _record_call_audit(
        self,
        *,
        seq: int,
        job_id: str,
        call_t0_ms: int,
        call_t1_ms: int,
        segments_returned_count: int,
        outcome: str,
        error: str = "",
    ) -> None:
        rt = self.rt
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
        rt.rolling_call_audit_recent.append(row)
        if len(rt.rolling_call_audit_recent) > 200:
            del rt.rolling_call_audit_recent[:-200]

        calls_done = int(max(0, int(rt.rolling_call_audit_summary.get("calls_done") or 0))) + 1
        rt.rolling_call_audit_summary["calls_done"] = calls_done
        rt.rolling_call_audit_summary["segments_returned_sum"] = int(
            max(0, int(rt.rolling_call_audit_summary.get("segments_returned_sum") or 0)) + seg_count
        )
        rt.rolling_call_audit_summary["segments_per_s_sum"] = float(
            max(0.0, float(rt.rolling_call_audit_summary.get("segments_per_s_sum") or 0.0)) + seg_per_s
        )
        seg_min = rt.rolling_call_audit_summary.get("segments_returned_min")
        seg_max = rt.rolling_call_audit_summary.get("segments_returned_max")
        ratio_min = rt.rolling_call_audit_summary.get("segments_per_s_min")
        ratio_max = rt.rolling_call_audit_summary.get("segments_per_s_max")
        rt.rolling_call_audit_summary["segments_returned_min"] = (
            seg_count if seg_min is None else int(min(int(seg_min), seg_count))
        )
        rt.rolling_call_audit_summary["segments_returned_max"] = (
            seg_count if seg_max is None else int(max(int(seg_max), seg_count))
        )
        rt.rolling_call_audit_summary["segments_per_s_min"] = (
            seg_per_s if ratio_min is None else float(min(float(ratio_min), seg_per_s))
        )
        rt.rolling_call_audit_summary["segments_per_s_max"] = (
            seg_per_s if ratio_max is None else float(max(float(ratio_max), seg_per_s))
        )
        out_counts = dict(rt.rolling_call_audit_summary.get("outcome_counts") or {})
        out_counts[safe_outcome] = int(max(0, int(out_counts.get(safe_outcome) or 0)) + 1)
        rt.rolling_call_audit_summary["outcome_counts"] = out_counts
        self._append_log("rolling_inference_call_done", **row)

    def _drop_pcm_prefix_to_ms(self, *, target_base_ms: int, reason: str) -> int:
        ctx = self._ctx
        rt = self.rt
        safe_target = int(max(rt.rolling_pcm_base_ms, target_base_ms))
        if safe_target <= rt.rolling_pcm_base_ms:
            return 0
        drop_window_ms = int(max(0, safe_target - rt.rolling_pcm_base_ms))
        drop_bytes = _ms_to_byte_offset(
            drop_window_ms,
            bytes_per_second=ctx["LIVE_AUDIO_BYTES_PER_SECOND"],
            sample_width_bytes=ctx["LIVE_AUDIO_SAMPLE_WIDTH_BYTES"],
        )
        drop_bytes = int(max(0, min(drop_bytes, len(rt.rolling_pcm))))
        if drop_bytes <= 0:
            return 0
        del rt.rolling_pcm[:drop_bytes]
        dropped_ms = _bytes_to_ms(
            drop_bytes,
            bytes_per_second=ctx["LIVE_AUDIO_BYTES_PER_SECOND"],
            sample_width_bytes=ctx["LIVE_AUDIO_SAMPLE_WIDTH_BYTES"],
        )
        if dropped_ms <= 0:
            dropped_ms = int(max(1, drop_window_ms))
        rt.rolling_pcm_base_ms = int(rt.rolling_pcm_base_ms + dropped_ms)
        if rt.rolling_processed_offset_ms < rt.rolling_pcm_base_ms:
            rt.rolling_processed_offset_ms = int(rt.rolling_pcm_base_ms)
        self._append_log(
            "rolling_buffer_drop",
            reason=str(reason or ""),
            dropped_audio_ms=int(max(0, dropped_ms)),
            pcm_base_ms=int(max(0, rt.rolling_pcm_base_ms)),
            pcm_buffer_bytes=int(max(0, len(rt.rolling_pcm))),
        )
        return int(max(0, dropped_ms))

    def _maybe_trim_pcm_buffer(self) -> None:
        ctx = self._ctx
        rt = self.rt
        committed_in_buffer_ms = int(max(0, int(rt.rolling_processed_offset_ms) - int(rt.rolling_pcm_base_ms)))
        if committed_in_buffer_ms < ctx["buffer_trim_threshold_ms"]:
            return
        target_base = int(min(rt.rolling_processed_offset_ms, rt.rolling_pcm_base_ms + ctx["buffer_trim_drop_ms"]))
        dropped_ms = self._drop_pcm_prefix_to_ms(target_base_ms=target_base, reason="buffer_trim")
        if dropped_ms > 0:
            rt.rolling_guardrail_metrics["buffer_trim_count"] = int(
                max(0, int(rt.rolling_guardrail_metrics.get("buffer_trim_count") or 0)) + 1
            )
            rt.rolling_guardrail_metrics["buffer_trim_dropped_audio_ms"] = int(
                max(0, int(rt.rolling_guardrail_metrics.get("buffer_trim_dropped_audio_ms") or 0)) + int(dropped_ms)
            )

    def _maybe_apply_hard_clip(self, *, end_ms: int) -> None:
        ctx = self._ctx
        rt = self.rt
        unprocessed_ms = int(max(0, int(end_ms) - int(rt.rolling_processed_offset_ms)))
        if unprocessed_ms <= ctx["max_uncommitted_ms"]:
            return
        clip_target_ms = int(max(rt.rolling_processed_offset_ms, int(end_ms) - int(ctx["hard_clip_keep_tail_ms"])))
        if clip_target_ms <= rt.rolling_processed_offset_ms:
            return
        dropped_uncommitted_ms = int(max(0, clip_target_ms - rt.rolling_processed_offset_ms))
        rt.rolling_processed_offset_ms = int(clip_target_ms)
        dropped_buffer_ms = self._drop_pcm_prefix_to_ms(target_base_ms=clip_target_ms, reason="hard_clip")
        rt.rolling_guardrail_metrics["hard_clip_count"] = int(
            max(0, int(rt.rolling_guardrail_metrics.get("hard_clip_count") or 0)) + 1
        )
        rt.rolling_guardrail_metrics["hard_clip_dropped_audio_ms"] = int(
            max(0, int(rt.rolling_guardrail_metrics.get("hard_clip_dropped_audio_ms") or 0))
            + int(max(dropped_uncommitted_ms, dropped_buffer_ms))
        )
        self._append_log(
            "rolling_hard_clip_applied",
            unprocessed_audio_ms=int(unprocessed_ms),
            dropped_uncommitted_audio_ms=int(max(0, dropped_uncommitted_ms)),
            keep_tail_ms=int(max(0, ctx["hard_clip_keep_tail_ms"])),
            processed_offset_ms=int(max(0, rt.rolling_processed_offset_ms)),
        )
        try:
            self.live_sessions.clear_live_preview(self.session_id)
        except Exception:
            pass

    def _update_state(self) -> None:
        rt = self.rt
        try:
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
        except Exception:
            pass
        try:
            self.live_sessions.set_live_engine_runtime(
                self.session_id,
                runtime=self._engine_runtime_payload(),
            )
        except Exception:
            pass
        if str(rt.finalization_state or "").strip().lower() in {"ready", "error", "finalized"}:
            try:
                self.live_sessions.clear_live_preview(self.session_id)
            except Exception:
                pass

    async def _update_state_and_emit_result(self, *, force_result: bool = False) -> None:
        self._update_state()
        await self._emit_result_event(force=force_result)

    def _archive_current_result(self, *, close_reason: str) -> dict[str, Any]:
        rt = self.rt
        try:
            live_result = self.live_sessions.live_result_snapshot(self.session_id)
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
            live_engine=self._ctx["LIVE_ENGINE"],
        )
        return live_result

    def _finalize_recording(self, *, reason: str) -> None:
        rt = self.rt
        if rt.recording_finalized:
            return
        rt.finalization_state = "finalizing"
        if rt.recorder is not None:
            try:
                rs = rt.recorder.finalize()
                rt.recording_path = str(rs.wav_path)
                rt.recording_bytes = int(rs.bytes_written)
                rt.recording_duration_ms = int(rs.duration_ms)
                rt.recording_state = "finalized"
                if rt.finalization_state != "error":
                    rt.finalization_state = "recording_finalized"
                self._append_log("rolling_recording_finalized", reason=reason, recording=rs.to_dict())
            except Exception as e:
                rt.shadow_disabled_reason = f"recording_finalize_failed:{type(e).__name__}"
                rt.recording_state = "error"
                rt.finalization_state = "error"
                self._append_log(
                    "rolling_recording_finalize_error",
                    reason=reason,
                    error=f"{type(e).__name__}: {e}",
                )
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
    ) -> bool:
        rt = self.rt
        if rt.recording_duration_ms <= rt.rolling_processed_offset_ms:
            return False
        try:
            result = self.live_sessions.live_result_snapshot(self.session_id)
        except Exception:
            return False
        preview = result.get("preview") or {}
        preview_text = str(preview.get("text") or "").strip()
        if not preview_text:
            preview_text = str(rt.rolling_last_preview_text or "").strip()
        if not preview_text:
            return False

        preview_audio_end_ms = int(max(0, int(preview.get("audio_end_ms") or 0)))
        if preview_audio_end_ms <= 0:
            preview_audio_end_ms = int(max(0, rt.rolling_last_preview_audio_end_fallback_ms))

        commit_t0 = int(max(0, rt.rolling_processed_offset_ms))
        commit_t1_candidates = [int(commit_t0), int(preview_audio_end_ms)]
        if include_recording_end:
            commit_t1_candidates.append(int(max(0, rt.recording_duration_ms)))
        commit_t1 = int(max(commit_t1_candidates))
        if max_t1_ms is not None:
            commit_t1 = int(min(commit_t1, int(max(0, max_t1_ms))))
        if commit_t1 <= commit_t0:
            if include_recording_end:
                commit_t1 = int(max(commit_t0 + 1, rt.recording_duration_ms))
            else:
                return False
        seg = {
            "segment_id": f"s{int(max(0, rt.rolling_commit_index_next)) + 1:04d}",
            "text": preview_text,
            "t0_ms": int(commit_t0),
            "t1_ms": int(commit_t1),
        }
        try:
            stored = self.live_sessions.record_live_commit(
                self.session_id,
                chunk_index=int(max(0, rt.rolling_commit_index_next)),
                t0_ms=int(commit_t0),
                t1_ms=int(commit_t1),
                text=preview_text,
                segments=[seg],
                state="ready",
                error="",
                reason="rolling_context_tail_preview_commit",
                chunk_duration_ms=int(max(0, commit_t1 - commit_t0)),
            )
            self._sync_counts_from_result(stored)
            rt.rolling_commit_index_next += 1
            rt.rolling_processed_offset_ms = int(max(rt.rolling_processed_offset_ms, commit_t1))
            self._maybe_trim_pcm_buffer()
            rt.rolling_last_preview_signature = ""
            rt.rolling_same_preview_repeats = 0
            rt.rolling_last_preview_audio_end_ms = -1
            rt.rolling_same_preview_audio_repeats = 0
            rt.rolling_last_preview_text = ""
            rt.rolling_last_preview_audio_end_fallback_ms = 0
            self._append_log("rolling_tail_preview_committed", t0_ms=commit_t0, t1_ms=commit_t1, chars=len(preview_text))
            return True
        except Exception:
            return False

    async def _enqueue_inference(self, *, force: bool = False) -> None:
        ctx = self._ctx
        rt = self.rt
        if rt.chunk_bridge is None:
            return
        now_mono = time.monotonic()
        if str(rt.recording_state or "") not in {"recording", "finalizing"}:
            return
        if rt.rolling_inflight is not None:
            rt.rolling_guardrail_metrics["outstanding_limit_skips"] = int(
                max(0, int(rt.rolling_guardrail_metrics.get("outstanding_limit_skips") or 0)) + 1
            )
            return

        end_ms = int(max(0, rt.recording_duration_ms))
        self._maybe_apply_hard_clip(end_ms=end_ms)
        if end_ms <= rt.rolling_processed_offset_ms:
            return
        use_force = bool(force)
        startup_active = (not use_force) and (ctx["startup_duration_ms"] > 0) and (int(end_ms) < int(ctx["startup_duration_ms"]))
        forced_preview_committed = False

        if not use_force:
            if startup_active:
                elapsed_since_emit_ms = int(max(0.0, float(now_mono - float(rt.rolling_last_emit_mono))) * 1000.0)
                if (rt.rolling_last_emit_mono > 0.0) and (elapsed_since_emit_ms < int(ctx["startup_emit_ms"])):
                    rt.rolling_guardrail_metrics["emit_interval_skips"] = int(
                        max(0, int(rt.rolling_guardrail_metrics.get("emit_interval_skips") or 0)) + 1
                    )
                    return
            elif not self._consume_pacing_slot(now_mono=now_mono):
                return

        if (not use_force) and (rt.vad_gate is not None):
            pending_t0_ms = int(max(rt.rolling_processed_offset_ms, rt.rolling_last_submitted_t1_ms))
            pending_ms = int(max(0, end_ms - pending_t0_ms))
            gate_max_lookback_ms = int(max(ctx["speech_gate_rearm_window_ms"], 4000))
            gate_window_ms = int(max(ctx["speech_gate_rearm_window_ms"], min(gate_max_lookback_ms, pending_ms)))
            gate_min_t0_ms = int(max(0, pending_t0_ms)) if pending_ms > 0 else int(max(0, end_ms))

            vad_recent = self._recent_speech_hit(
                now_mono=now_mono,
                end_ms=end_ms,
                min_t0_ms=gate_min_t0_ms,
                window_ms=gate_window_ms,
            )
            if "error" in vad_recent:
                rt.finalization_state = "error"
                self._append_log("rolling_vad_error", error=str(vad_recent.get("error") or "vad_error"))
                await self._update_state_and_emit_result()
                return

            speech_hit = bool(vad_recent.get("speech_hit"))
            cutoff_mono = float(max(0.0, now_mono - ctx["speech_gate_rearm_window_s"]))
            rt.rolling_speech_gate_recent_hits_mono = [
                float(ts) for ts in rt.rolling_speech_gate_recent_hits_mono if float(ts) >= cutoff_mono
            ]
            if speech_hit:
                rt.rolling_last_recent_speech_mono = float(max(0.0, now_mono))
                rt.rolling_speech_gate_recent_hits_mono.append(float(now_mono))

            if rt.rolling_speech_gate_state == "quiet":
                if len(rt.rolling_speech_gate_recent_hits_mono) >= int(max(1, ctx["speech_gate_rearm_hits"])):
                    self._set_speech_gate_state(
                        next_state="active",
                        reason="speech_rearm_hits",
                        now_mono=now_mono,
                    )
                    rt.rolling_guardrail_metrics["speech_gate_rearm_count"] = int(
                        max(0, int(rt.rolling_guardrail_metrics.get("speech_gate_rearm_count") or 0)) + 1
                    )
                else:
                    rt.rolling_guardrail_metrics["speech_gate_quiet_skips"] = int(
                        max(0, int(rt.rolling_guardrail_metrics.get("speech_gate_quiet_skips") or 0)) + 1
                    )
                    return
            elif rt.rolling_speech_gate_state == "active":
                silence_elapsed_ms = int(max(0.0, float(now_mono - rt.rolling_last_recent_speech_mono) * 1000.0))
                if rt.rolling_last_recent_speech_mono <= 0.0 or silence_elapsed_ms >= int(max(100, ctx["speech_gate_force_threshold_ms"])):
                    committed = self._commit_preview_tail_if_needed(
                        include_recording_end=False,
                        max_t1_ms=(int(rt.rolling_last_submitted_t1_ms) if int(rt.rolling_last_submitted_t1_ms) > 0 else None),
                    )
                    if committed:
                        forced_preview_committed = True
                        rt.rolling_guardrail_metrics["speech_gate_forced_commit_count"] = int(
                            max(0, int(rt.rolling_guardrail_metrics.get("speech_gate_forced_commit_count") or 0)) + 1
                        )
                        self._append_log(
                            "rolling_speech_gate_forced_commit",
                            silence_elapsed_ms=int(max(0, silence_elapsed_ms)),
                            threshold_ms=int(max(100, ctx["speech_gate_force_threshold_ms"])),
                        )
                    self._set_speech_gate_state(
                        next_state="quiet",
                        reason="silence_force_commit",
                        now_mono=now_mono,
                        rearm_from_ms=pending_t0_ms,
                    )

            if not speech_hit:
                rt.rolling_guardrail_metrics["speech_gate_quiet_skips"] = int(
                    max(0, int(rt.rolling_guardrail_metrics.get("speech_gate_quiet_skips") or 0)) + 1
                )
                if forced_preview_committed:
                    await self._update_state_and_emit_result()
                return

        effective_min_infer_audio_ms = int(ctx["LIVE_ROLLING_MIN_INFER_AUDIO_MS"])
        if startup_active and ctx["startup_min_infer_audio_ms"] > 0:
            effective_min_infer_audio_ms = int(max(1, ctx["startup_min_infer_audio_ms"]))
        unprocessed_ms = int(max(0, end_ms - rt.rolling_processed_offset_ms))
        if (not use_force) and (unprocessed_ms < effective_min_infer_audio_ms):
            return
        effective_min_new_audio_ms = int(ctx["min_new_audio_ms"])
        if startup_active and ctx["startup_min_new_audio_ms"] > 0:
            effective_min_new_audio_ms = int(max(0, ctx["startup_min_new_audio_ms"]))
        if (not use_force) and rt.rolling_last_submitted_t1_ms > 0:
            delta_new_audio_ms = int(max(0, end_ms - rt.rolling_last_submitted_t1_ms))
            if delta_new_audio_ms < effective_min_new_audio_ms:
                rt.rolling_guardrail_metrics["min_new_audio_skips"] = int(
                    max(0, int(rt.rolling_guardrail_metrics.get("min_new_audio_skips") or 0)) + 1
                )
                return

        infer_t0_ms = int(max(rt.rolling_processed_offset_ms, rt.rolling_pcm_base_ms))
        infer_t1_ms = int(max(infer_t0_ms, end_ms))
        infer_window_ms = int(max(0, infer_t1_ms - infer_t0_ms))
        if infer_window_ms > ctx["max_decode_window_ms"]:
            infer_t1_ms = int(max(infer_t0_ms, infer_t0_ms + ctx["max_decode_window_ms"]))
            if infer_t1_ms < end_ms:
                rt.rolling_guardrail_metrics["decode_window_cap_count"] = int(
                    max(0, int(rt.rolling_guardrail_metrics.get("decode_window_cap_count") or 0)) + 1
                )
                self._append_log(
                    "rolling_decode_window_capped",
                    end_ms=int(end_ms),
                    processed_offset_ms=int(rt.rolling_processed_offset_ms),
                    infer_t0_ms=int(infer_t0_ms),
                    infer_t1_ms=int(infer_t1_ms),
                    max_decode_window_ms=int(ctx["max_decode_window_ms"]),
                )
        if (not use_force) and infer_t1_ms <= rt.rolling_last_submitted_t1_ms:
            rt.rolling_guardrail_metrics["window_no_progress_skips"] = int(
                max(0, int(rt.rolling_guardrail_metrics.get("window_no_progress_skips") or 0)) + 1
            )
            return

        start_b = _ms_to_byte_offset(
            int(max(0, infer_t0_ms - rt.rolling_pcm_base_ms)),
            bytes_per_second=ctx["LIVE_AUDIO_BYTES_PER_SECOND"],
            sample_width_bytes=ctx["LIVE_AUDIO_SAMPLE_WIDTH_BYTES"],
        )
        end_b = _ms_to_byte_offset(
            int(max(0, infer_t1_ms - rt.rolling_pcm_base_ms)),
            bytes_per_second=ctx["LIVE_AUDIO_BYTES_PER_SECOND"],
            sample_width_bytes=ctx["LIVE_AUDIO_SAMPLE_WIDTH_BYTES"],
        )
        end_b = min(end_b, len(rt.rolling_pcm))
        if end_b <= start_b:
            return
        pcm = bytes(rt.rolling_pcm[start_b:end_b])
        if not pcm:
            return

        infer_seq = int(max(0, rt.rolling_infer_seq_next))

        try:
            job = rt.chunk_bridge.enqueue_chunk_pcm16(
                session_id=self.session_id,
                chunk_index=infer_seq,
                t0_ms=int(infer_t0_ms),
                t1_ms=int(infer_t1_ms),
                pcm16le=pcm,
                language=rt.session_live_asr_language,
                live_lane="single",
                preview_seq=infer_seq,
                preview_audio_end_ms=int(infer_t1_ms),
            )
        except Exception as e:
            self._append_log(
                "rolling_inference_enqueue_error",
                seq=int(infer_seq),
                t0_ms=int(infer_t0_ms),
                t1_ms=int(infer_t1_ms),
                error=f"{type(e).__name__}: {e}",
            )
            rt.finalization_state = "error"
            await self._update_state_and_emit_result()
            return

        rt.rolling_infer_seq_next = infer_seq + 1
        rt.rolling_last_emit_mono = now_mono
        rt.rolling_last_submitted_t1_ms = int(max(rt.rolling_last_submitted_t1_ms, infer_t1_ms))
        rt.rolling_inflight = {
            "seq": int(infer_seq),
            "job_id": str(job.job_id),
            "t0_ms": int(infer_t0_ms),
            "t1_ms": int(infer_t1_ms),
            "enqueued_mono": float(now_mono),
        }
        if rt.finalization_state not in {"error", "ready"}:
            rt.finalization_state = "processing_chunks"
        await self._update_state_and_emit_result()
        self._append_log(
            "rolling_inference_enqueued",
            seq=int(infer_seq),
            job_id=str(job.job_id),
            t0_ms=int(infer_t0_ms),
            t1_ms=int(infer_t1_ms),
            audio_bytes=len(pcm),
            forced=bool(use_force),
            speech_gate_state=str(rt.rolling_speech_gate_state or ""),
        )

    async def _poll_inference(self, *, force: bool = False) -> None:
        ctx = self._ctx
        rt = self.rt
        if rt.chunk_bridge is None or rt.rolling_inflight is None:
            return
        now_mono = time.monotonic()
        if (not force) and ((now_mono - rt.rolling_last_poll_mono) < ctx["poll_interval_s"]):
            return
        rt.rolling_last_poll_mono = now_mono

        inflight = dict(rt.rolling_inflight or {})
        seq = int(inflight.get("seq") or 0)
        job_id = str(inflight.get("job_id") or "")
        t0_ms = int(max(0, int(inflight.get("t0_ms") or 0)))
        t1_ms = int(max(t0_ms, int(inflight.get("t1_ms") or t0_ms)))

        try:
            poll = rt.chunk_bridge.poll_job(job_id, t0_offset_ms=t0_ms)
        except Exception as e:
            self._append_log("rolling_inference_poll_error", seq=seq, job_id=job_id, error=f"{type(e).__name__}: {e}")
            return
        if not bool(poll.done):
            return
        stale_completion = bool(seq < int(max(-1, rt.rolling_last_applied_seq)))
        if stale_completion:
            rt.rolling_guardrail_metrics["stale_completion_ignored_count"] = int(
                max(0, int(rt.rolling_guardrail_metrics.get("stale_completion_ignored_count") or 0)) + 1
            )
            self._append_log(
                "rolling_inference_stale_ignored",
                seq=seq,
                job_id=job_id,
                state=str(poll.state or ""),
                latest_applied_seq=int(rt.rolling_last_applied_seq),
            )
            self._record_call_audit(
                seq=seq,
                job_id=job_id,
                call_t0_ms=t0_ms,
                call_t1_ms=t1_ms,
                segments_returned_count=0,
                outcome="empty",
                error="",
            )
            rt.rolling_inflight = None
            await self._update_state_and_emit_result()
            return

        if bool(poll.ok):
            raw_segments = poll.segments if isinstance(poll.segments, list) else []
            segments_returned_count = int(len(raw_segments))
            segments = [dict(seg) for seg in (poll.segments or []) if isinstance(seg, dict)]
            segments.sort(key=lambda seg: int(seg.get("t0_ms") or 0))

            _poll_status = dict(poll.status or {})
            _gpu_t = _safe_float(_poll_status.get("asr_timing_whisperx_transcribe_call_s"))
            if _gpu_t is not None:
                rt.rolling_gpu_proxy_transcribe_s += _gpu_t
            _gpu_p = _safe_float(_poll_status.get("asr_timing_whisperx_total_s"))
            if _gpu_p is not None:
                rt.rolling_gpu_proxy_pipeline_s += _gpu_p

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
                if single_text and max(single_duration_ms, infer_window_duration_ms) >= ctx["single_segment_commit_min_ms"]:
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

            preview_sig = self._preview_signature(preview_text)
            if preview_sig:
                if preview_sig == rt.rolling_last_preview_signature:
                    rt.rolling_same_preview_repeats += 1
                else:
                    rt.rolling_last_preview_signature = preview_sig
                    rt.rolling_same_preview_repeats = 1
                if int(preview_audio_end_ms) == int(rt.rolling_last_preview_audio_end_ms):
                    rt.rolling_same_preview_audio_repeats += 1
                else:
                    rt.rolling_last_preview_audio_end_ms = int(preview_audio_end_ms)
                    rt.rolling_same_preview_audio_repeats = 1
            else:
                rt.rolling_last_preview_signature = ""
                rt.rolling_same_preview_repeats = 0
                rt.rolling_last_preview_audio_end_ms = -1
                rt.rolling_same_preview_audio_repeats = 0

            if preview_sig and rt.rolling_same_preview_audio_repeats >= ctx["force_commit_repeats"] and segments:
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
                        rt.rolling_guardrail_metrics["force_commit_repeats_count"] = int(
                            max(0, int(rt.rolling_guardrail_metrics.get("force_commit_repeats_count") or 0)) + 1
                        )
                        self._append_log(
                            "rolling_force_commit_repeats_triggered",
                            seq=seq,
                            job_id=job_id,
                            repeats=int(max(0, rt.rolling_same_preview_audio_repeats)),
                            preview_chars=len(last_text),
                            force_commit_repeats=int(ctx["force_commit_repeats"]),
                        )

            if commit_segments:
                commit_t0_ms = int(max(0, rt.rolling_processed_offset_ms))
                normalized_commit_segments: list[dict[str, Any]] = []
                for raw_seg in commit_segments:
                    if not isinstance(raw_seg, dict):
                        continue
                    seg_text = str(raw_seg.get("text") or "").strip()
                    if not seg_text:
                        continue
                    seg_t0_raw = int(raw_seg.get("t0_ms") or commit_t0_ms)
                    seg_t1_raw = int(raw_seg.get("t1_ms") or seg_t0_raw)
                    seg_t0 = int(max(commit_t0_ms, seg_t0_raw))
                    seg_t1 = int(max(seg_t0, seg_t1_raw))
                    if seg_t1 <= commit_t0_ms:
                        continue
                    normalized_commit_segments.append(
                        {
                            "segment_id": str(raw_seg.get("segment_id") or ""),
                            "text": seg_text,
                            "t0_ms": int(seg_t0),
                            "t1_ms": int(seg_t1),
                            "speaker": str(raw_seg.get("speaker") or "").strip(),
                        }
                    )
                if normalized_commit_segments:
                    commit_t1_ms = int(max(commit_t0_ms, int(normalized_commit_segments[-1]["t1_ms"])))
                    if single_segment_forced_commit or force_commit_repeats_applied:
                        commit_t1_ms = int(max(commit_t1_ms, t1_ms))
                    normalized_commit_segments[0]["t0_ms"] = int(commit_t0_ms)
                    normalized_commit_segments[-1]["t1_ms"] = int(
                        max(commit_t1_ms, normalized_commit_segments[-1]["t1_ms"])
                    )
                    commit_text = "\n".join(
                        str(seg.get("text") or "").strip()
                        for seg in normalized_commit_segments
                        if str(seg.get("text") or "").strip()
                    ).strip()
                    if commit_text:
                        try:
                            result = self.live_sessions.record_live_commit(
                                self.session_id,
                                chunk_index=int(max(0, rt.rolling_commit_index_next)),
                                t0_ms=int(commit_t0_ms),
                                t1_ms=int(commit_t1_ms),
                                text=commit_text,
                                segments=normalized_commit_segments,
                                state="ready",
                                error="",
                                reason=str(commit_reason),
                                chunk_duration_ms=int(max(0, commit_t1_ms - commit_t0_ms)),
                            )
                            self._sync_counts_from_result(result)
                            rt.rolling_commit_index_next += 1
                            rt.rolling_processed_offset_ms = int(max(rt.rolling_processed_offset_ms, commit_t1_ms))
                            committed_this_poll = True
                            call_outcome = "commit"
                            self._maybe_trim_pcm_buffer()
                            if single_segment_forced_commit:
                                self._append_log(
                                    "rolling_single_segment_forced_commit",
                                    seq=seq,
                                    job_id=job_id,
                                    commit_t0_ms=int(commit_t0_ms),
                                    commit_t1_ms=int(commit_t1_ms),
                                    commit_duration_ms=int(max(0, commit_t1_ms - commit_t0_ms)),
                                )
                            if force_commit_repeats_applied:
                                rt.rolling_same_preview_repeats = 0
                                rt.rolling_last_preview_signature = ""
                                rt.rolling_last_preview_audio_end_ms = -1
                                rt.rolling_same_preview_audio_repeats = 0
                                rt.rolling_last_preview_text = ""
                                rt.rolling_last_preview_audio_end_fallback_ms = 0
                        except Exception as e:
                            call_outcome = "error"
                            call_error = f"{type(e).__name__}: {e}"
                            rt.finalization_state = "error"
                            self._append_log(
                                "rolling_commit_store_error",
                                seq=seq,
                                job_id=job_id,
                                error=f"{type(e).__name__}: {e}",
                            )

            if preview_text:
                if call_outcome != "commit":
                    call_outcome = "preview_only"
                rt.rolling_last_preview_text = str(preview_text or "")
                rt.rolling_last_preview_audio_end_fallback_ms = int(max(0, preview_audio_end_ms))
                try:
                    self.live_sessions.update_live_preview(
                        self.session_id,
                        text=preview_text,
                        preview_seq=int(max(0, seq)),
                        audio_end_ms=int(max(0, preview_audio_end_ms)),
                        append_to_existing=False,
                    )
                except Exception:
                    pass
                self._append_log(
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
                    rt.rolling_last_preview_signature = ""
                    rt.rolling_same_preview_repeats = 0
                    rt.rolling_last_preview_audio_end_ms = -1
                    rt.rolling_same_preview_audio_repeats = 0
            self._record_call_audit(
                seq=seq,
                job_id=job_id,
                call_t0_ms=t0_ms,
                call_t1_ms=t1_ms,
                segments_returned_count=segments_returned_count,
                outcome=call_outcome,
                error=call_error,
            )
            rt.rolling_last_applied_seq = int(max(rt.rolling_last_applied_seq, seq))
        else:
            _poll_status = dict(poll.status or {})
            _gpu_t = _safe_float(_poll_status.get("asr_timing_whisperx_transcribe_call_s"))
            if _gpu_t is not None:
                rt.rolling_gpu_proxy_transcribe_s += _gpu_t
            _gpu_p = _safe_float(_poll_status.get("asr_timing_whisperx_total_s"))
            if _gpu_p is not None:
                rt.rolling_gpu_proxy_pipeline_s += _gpu_p

            poll_state = str(poll.state or "").strip().lower()
            err = str(poll.error or f"asr_state:{poll_state}" or "asr_error")
            try:
                result = self.live_sessions.record_live_commit(
                    self.session_id,
                    chunk_index=int(max(0, rt.rolling_commit_index_next)),
                    t0_ms=int(t0_ms),
                    t1_ms=int(t1_ms),
                    text="",
                    segments=[],
                    state="error",
                    error=err,
                    reason="rolling_context_error",
                )
                self._sync_counts_from_result(result)
                rt.rolling_commit_index_next += 1
            except Exception:
                pass
            rt.rolling_processed_offset_ms = int(max(rt.rolling_processed_offset_ms, t1_ms))
            self._maybe_trim_pcm_buffer()
            rt.finalization_state = "error"
            rt.rolling_last_applied_seq = int(max(rt.rolling_last_applied_seq, seq))
            self._append_log("rolling_inference_error", seq=seq, job_id=job_id, error=err)
            self._record_call_audit(
                seq=seq,
                job_id=job_id,
                call_t0_ms=t0_ms,
                call_t1_ms=t1_ms,
                segments_returned_count=0,
                outcome="error",
                error=err,
            )

        if (rt.vad_gate is not None) and (rt.rolling_speech_gate_state == "flush"):
            flush_window_end_ms = int(max(t0_ms, t1_ms))
            self._commit_preview_tail_if_needed(include_recording_end=False, max_t1_ms=flush_window_end_ms)
            rt.rolling_processed_offset_ms = int(max(rt.rolling_processed_offset_ms, flush_window_end_ms))
            rt.rolling_last_submitted_t1_ms = int(max(rt.rolling_last_submitted_t1_ms, flush_window_end_ms))
            rt.rolling_speech_gate_recent_hits_mono = []
            rt.rolling_last_recent_speech_mono = 0.0
            rt.rolling_guardrail_metrics["speech_gate_silence_flush_count"] = int(
                max(0, int(rt.rolling_guardrail_metrics.get("speech_gate_silence_flush_count") or 0)) + 1
            )
            self._set_speech_gate_state(
                next_state="quiet",
                reason="flush_completed",
                now_mono=time.monotonic(),
                rearm_from_ms=flush_window_end_ms,
            )

        rt.rolling_inflight = None
        if rt.finalization_state not in {"error", "ready"}:
            rt.finalization_state = "recording" if str(rt.recording_state or "") == "recording" else "processing_chunks"
        await self._update_state_and_emit_result()

    async def _process_rolling(self, *, force_poll: bool = False, force_emit: bool = False) -> None:
        await self._poll_inference(force=force_poll)
        await self._enqueue_inference(force=force_emit)

    async def _drain_inflight_only(self, *, force_poll: bool = True) -> None:
        await self._poll_inference(force=force_poll)

    async def run(self) -> None:
        session_id = self.session_id
        websocket = self.websocket
        live_sessions = self.live_sessions
        _rooted_path = self.rooted_path_cb
        ctx = self._configure_context()
        rt = self.rt

        try:
            live_sessions.open_websocket(session_id)
        except KeyError:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="session_not_found")
            return
        except RuntimeError as e:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason=str(e))
            return

        await websocket.accept()
        try:
            session_snapshot = live_sessions.snapshot(session_id)
            snap_lang = _normalize_optional_language((session_snapshot or {}).get("asr_language"))
            if snap_lang is not None:
                rt.session_live_asr_language = snap_lang
        except Exception:
            rt.session_live_asr_language = ctx["LIVE_ASR_LANGUAGE"]
        if ctx["vad_enabled"]:
            try:
                rt.vad_gate = LiveSileroVadGate(
                    settings=ctx["vad_settings"],
                    sample_rate_hz=ctx["LIVE_AUDIO_SAMPLE_RATE_HZ"],
                )
            except Exception as e:
                try:
                    await websocket.close(code=status.WS_1011_INTERNAL_ERROR, reason="live_vad_init_failed")
                except Exception:
                    pass
                live_sessions.close_session(session_id, reason=f"live_vad_init_failed:{type(e).__name__}")
                return

        try:
            ready_payload = ready_event(
                session_id,
                message="Live websocket connected. Send binary PCM16 frames and JSON controls.",
                engine="rolling_context",
            )
            ready_payload["live_engine"] = ctx["LIVE_ENGINE"]
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
                    diarize_enabled=ctx["LIVE_DIARIZE_ENABLED"],
                    diarize_speaker_mode=ctx["LIVE_DIARIZE_SPEAKER_MODE"],
                    diarize_min_speakers=ctx["LIVE_DIARIZE_MIN_SPEAKERS"],
                    diarize_max_speakers=ctx["LIVE_DIARIZE_MAX_SPEAKERS"],
                )
                rt.recording_state = "recording"
                rt.recording_path = str(rec_snap.wav_path)
                rt.recording_bytes = int(rec_snap.bytes_written)
                rt.recording_duration_ms = int(rec_snap.duration_ms)
                rt.finalization_state = "recording"
                await self._update_state_and_emit_result(force_result=True)
                self._append_log(
                    "rolling_context_started",
                    recording=rec_snap.to_dict(),
                    config={
                        "poll_interval_ms": int(ctx["LIVE_ROLLING_POLL_INTERVAL_MS"]),
                        "min_infer_audio_ms": int(ctx["LIVE_ROLLING_MIN_INFER_AUDIO_MS"]),
                        "single_segment_commit_min_ms": int(ctx["single_segment_commit_min_ms"]),
                        "force_commit_repeats": int(ctx["force_commit_repeats"]),
                        "max_uncommitted_ms": int(ctx["max_uncommitted_ms"]),
                        "hard_clip_keep_tail_ms": int(ctx["hard_clip_keep_tail_ms"]),
                        "max_decode_window_ms": int(ctx["max_decode_window_ms"]),
                        "buffer_trim_threshold_ms": int(ctx["buffer_trim_threshold_ms"]),
                        "buffer_trim_drop_ms": int(ctx["buffer_trim_drop_ms"]),
                        "min_new_audio_ms": int(ctx["min_new_audio_ms"]),
                        "min_emit_interval_ms": int(ctx["LIVE_ROLLING_MIN_EMIT_INTERVAL_MS"]),
                        "pacing_base_emit_ms": int(max(1, ctx["pacing_base_emit_ms"])),
                        "pacing_effective_emit_ms": int(max(1, ctx["pacing_effective_emit_ms"])),
                        "pacing_startup_duration_ms": int(max(0, ctx["startup_duration_ms"])),
                        "pacing_startup_emit_ms": int(max(1, ctx["startup_emit_ms"])),
                        "pacing_startup_min_infer_audio_ms": int(max(0, ctx["startup_min_infer_audio_ms"])),
                        "pacing_startup_min_new_audio_ms": int(max(0, ctx["startup_min_new_audio_ms"])),
                        "pacing_phase_ms": int(max(0, ctx["pacing_phase_ms"])),
                        "vad_enabled": bool(ctx["vad_enabled"]),
                        "vad_threshold": float(ctx["vad_threshold"]),
                        "vad_max_speech_duration_s": float(ctx["vad_max_speech_duration_s"]),
                        "vad_min_speech_ms": int(ctx["vad_min_speech_ms"]),
                        "vad_hangover_ms": int(ctx["vad_hangover_ms"]),
                        "vad_whisperx_venv": str(ctx["vad_settings"].whisperx_venv or ""),
                        "speech_gate_silence_enter_ms": int(max(100, ctx["speech_gate_silence_enter_ms"])),
                        "speech_gate_rearm_hits": int(max(1, ctx["speech_gate_rearm_hits"])),
                        "speech_gate_rearm_window_ms": int(max(100, ctx["speech_gate_rearm_window_ms"])),
                        "speech_gate_force_commit_silence_ms": int(max(100, ctx["speech_gate_force_commit_silence_ms"])),
                        "language": rt.session_live_asr_language,
                        "diarize_enabled": bool(ctx["LIVE_DIARIZE_ENABLED"]),
                        "diarize_speaker_mode": str(ctx["LIVE_DIARIZE_SPEAKER_MODE"]),
                        "diarize_min_speakers": int(ctx["LIVE_DIARIZE_MIN_SPEAKERS"]),
                        "diarize_max_speakers": int(ctx["LIVE_DIARIZE_MAX_SPEAKERS"]),
                    },
                )
            except Exception as e:
                rt.recorder = None
                rt.chunk_bridge = None
                rt.recording_state = "error"
                rt.finalization_state = "error"
                rt.shadow_disabled_reason = f"rolling_init_failed:{type(e).__name__}"
                await self._update_state_and_emit_result(force_result=True)
                self._append_log("rolling_context_init_error", error=f"{type(e).__name__}: {e}")

            while True:
                try:
                    incoming = await asyncio.wait_for(websocket.receive(), timeout=ctx["poll_interval_s"])
                except asyncio.TimeoutError:
                    # Keep draining/polling inference even when no websocket frames arrive.
                    await self._process_rolling(force_poll=False, force_emit=False)
                    continue

                if incoming.get("type") == "websocket.disconnect":
                    rt.stop_reason = "client_disconnected"
                    break

                raw_bytes = incoming.get("bytes")
                if raw_bytes is not None:
                    snapshot = live_sessions.record_audio(session_id, byte_count=len(raw_bytes))
                    raw = bytes(raw_bytes or b"")
                    if (len(raw) % ctx["LIVE_AUDIO_SAMPLE_WIDTH_BYTES"]) != 0:
                        raw = raw[: len(raw) - (len(raw) % ctx["LIVE_AUDIO_SAMPLE_WIDTH_BYTES"])]
                    if rt.recorder is not None:
                        try:
                            rec_snap = rt.recorder.append_pcm16(raw)
                            rt.recording_bytes = int(rec_snap.bytes_written)
                            rt.recording_duration_ms = int(rec_snap.duration_ms)
                            rt.recording_path = str(rec_snap.wav_path)
                        except Exception as e:
                            rt.shadow_disabled_reason = f"recording_append_failed:{type(e).__name__}"
                            rt.recording_state = "error"
                            rt.finalization_state = "error"
                            self._append_log(
                                "rolling_recording_append_error",
                                error=f"{type(e).__name__}: {e}",
                                at_frame=int(snapshot.get("frames_received") or 0),
                            )
                            try:
                                rt.recorder.abort()
                            except Exception:
                                pass
                            rt.recorder = None
                            await self._update_state_and_emit_result()
                    if raw:
                        rt.rolling_pcm.extend(raw)

                    await self._process_rolling(force_poll=False, force_emit=False)

                    should_emit_stats = snapshot["frames_received"] == 1 or (snapshot["frames_received"] % 50) == 0
                    if should_emit_stats:
                        stats_payload = stats_event(
                            session_id,
                            bytes_received=snapshot["bytes_received"],
                            frames_received=snapshot["frames_received"],
                            controls_received=snapshot["controls_received"],
                            uptime_s=snapshot["age_s"],
                            live_engine=ctx["LIVE_ENGINE"],
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
                            live_shadow_disabled_reason=str(rt.shadow_disabled_reason or ""),
                            live_inflight=bool(rt.rolling_inflight is not None),
                            rolling_guardrails=dict(rt.rolling_guardrail_metrics),
                            rolling_unprocessed_audio_ms=int(max(0, int(rt.recording_duration_ms) - int(rt.rolling_processed_offset_ms))),
                            rolling_pcm_base_ms=int(max(0, rt.rolling_pcm_base_ms)),
                        )
                        try:
                            live_sessions.append_stats_log(session_id, stats_payload)
                        except Exception:
                            pass
                        await self._send_event(stats_payload)
                    continue

                raw_text = incoming.get("text")
                if raw_text is None:
                    await self._send_event(
                        error_event(
                            session_id,
                            code="invalid_frame",
                            message="Expected binary audio frame or JSON control message.",
                        )
                    )
                    continue

                control_type, obj, parse_err = parse_client_message(raw_text)
                if parse_err:
                    await self._send_event(error_event(session_id, code=parse_err, message="Invalid control message."))
                    continue

                live_sessions.record_control(session_id)

                if control_type == "ping":
                    await self._send_event(pong_event(session_id))
                    continue

                if control_type == "set_language":
                    try:
                        next_language = _parse_control_language((obj or {}).get("language"))
                    except ValueError as e:
                        await self._send_event(error_event(session_id, code="invalid_language", message=str(e)))
                        continue
                    snapshot = live_sessions.set_asr_language(
                        session_id,
                        asr_language=next_language,
                    )
                    rt.session_live_asr_language = _normalize_optional_language(snapshot.get("asr_language"))
                    if rt.session_live_asr_language is None:
                        rt.session_live_asr_language = ctx["LIVE_ASR_LANGUAGE"]
                    self._append_log(
                        "rolling_language_updated",
                        language=(rt.session_live_asr_language or ""),
                        requested=(next_language or "auto"),
                    )
                    await self._update_state_and_emit_result(force_result=True)
                    await self._send_event(control_ack_event(session_id, control_type="set_language", state=snapshot["state"]))
                    continue

                if control_type == "start":
                    snapshot = live_sessions.mark_state(session_id, state="listening")
                    rt.recording_state = "recording"
                    if rt.finalization_state not in {"error", "ready"}:
                        rt.finalization_state = "recording"
                    await self._update_state_and_emit_result()
                    await self._send_event(control_ack_event(session_id, control_type="start", state=snapshot["state"]))
                    continue

                if control_type == "pause":
                    snapshot = live_sessions.mark_state(session_id, state="paused")
                    rt.recording_state = "paused"
                    await self._update_state_and_emit_result()
                    await self._send_event(control_ack_event(session_id, control_type="pause", state=snapshot["state"]))
                    continue

                if control_type == "resume":
                    snapshot = live_sessions.mark_state(session_id, state="listening")
                    rt.recording_state = "recording"
                    if rt.finalization_state not in {"error", "ready"}:
                        rt.finalization_state = "recording"
                    await self._update_state_and_emit_result()
                    await self._send_event(control_ack_event(session_id, control_type="resume", state=snapshot["state"]))
                    continue

                if control_type == "stop":
                    rt.stop_reason = "client_stop"
                    live_result: dict[str, Any] = {}
                    if rt.recording_state in {"recording", "paused"}:
                        rt.recording_state = "finalizing"
                    if rt.finalization_state not in {"error", "ready"}:
                        rt.finalization_state = "finalizing"
                    await self._update_state_and_emit_result()
                    await self._process_rolling(force_poll=True, force_emit=True)
                    wait_deadline = time.monotonic() + max(0.0, ctx["LIVE_DRAIN_WAIT_S"])
                    while time.monotonic() < wait_deadline:
                        await self._drain_inflight_only(force_poll=True)
                        if rt.rolling_inflight is None:
                            break
                        await asyncio.sleep(min(0.1, ctx["poll_interval_s"]))

                    self._finalize_recording(reason=rt.stop_reason)
                    await self._drain_inflight_only(force_poll=True)
                    self._commit_preview_tail_if_needed()
                    if rt.finalization_state != "error":
                        rt.finalization_state = "ready"
                    await self._update_state_and_emit_result(force_result=True)
                    try:
                        live_result = self._archive_current_result(close_reason=rt.stop_reason)
                        rt.archived_result = True
                    except Exception:
                        live_result = {}
                    await self._emit_result_event(force=True)

                    await self._send_event(
                        ended_event(
                            session_id,
                            reason=rt.stop_reason,
                            transcript_revision=int(max(0, int(live_result.get("transcript_revision") or 0))),
                            final_segments_count=len(live_result.get("final_segments") or []),
                            final_transcript_url=_rooted_path(f"/demo/live/sessions/{session_id}/final"),
                        )
                    )
                    await websocket.close(code=status.WS_1000_NORMAL_CLOSURE)
                    rt.websocket_closed = True
                    break

        except WebSocketDisconnect:
            rt.stop_reason = "client_disconnected"
        except Exception as e:
            rt.stop_reason = "server_error"
            try:
                await self._send_event(
                    error_event(
                        session_id,
                        code="internal_error",
                        message=f"{type(e).__name__}: {e}",
                        fatal=True,
                    )
                )
            except Exception:
                pass
            if not rt.websocket_closed:
                try:
                    await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
                except Exception:
                    pass
        finally:
            self._finalize_recording(reason=rt.stop_reason)
            await self._drain_inflight_only(force_poll=True)
            if rt.stop_reason == "client_stop":
                wait_timeout = 0.0
            else:
                wait_timeout = ctx["LIVE_DRAIN_WAIT_S"]
            wait_deadline = time.monotonic() + max(0.0, float(wait_timeout))
            while time.monotonic() < wait_deadline:
                await self._drain_inflight_only(force_poll=True)
                remaining_ms = int(max(0, rt.recording_duration_ms - rt.rolling_processed_offset_ms))
                if (rt.rolling_inflight is None) and remaining_ms < ctx["LIVE_ROLLING_MIN_INFER_AUDIO_MS"]:
                    break
                await asyncio.sleep(min(0.1, ctx["poll_interval_s"]))

            self._commit_preview_tail_if_needed()
            if rt.finalization_state not in {"error", "ready"}:
                rt.finalization_state = "ready"
            await self._update_state_and_emit_result(force_result=True)

            if not rt.archived_result:
                try:
                    self._archive_current_result(close_reason=rt.stop_reason)
                except Exception:
                    pass
            await self._emit_result_event(force=True)

            live_sessions.close_session(session_id, reason=rt.stop_reason)
            if rt.recorder is not None and not rt.recording_finalized:
                try:
                    rt.recorder.abort()
                except Exception:
                    pass

async def run_live_session_ws_rolling_context(
    session_id: str,
    websocket: WebSocket,
    *,
    live_sessions: Any,
    rooted_path_cb: Callable[[str], str],
    config: Mapping[str, Any],
) -> None:
    session = RollingContextSession(
        session_id,
        websocket,
        live_sessions=live_sessions,
        rooted_path_cb=rooted_path_cb,
        config=config,
    )
    await session.run()
