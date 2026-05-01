from __future__ import annotations

import re
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from asr_pool_api import (
    ASRAudioFile,
    ASRCompletionEvent,
    ASRCompletionFeedReset,
    ASROutputSelection,
    ASRPoolClient,
    ASRPoolClientConfig,
    ASRPoolError,
    ASRRequestOptions,
    ASRRequestRouting,
    ASRSubmitRequest,
)
from live.runtime.util import _normalize_optional_language, _safe_float
from app.config.settings import get_setting


DEFAULT_SAMPLE_RATE_HZ = 16000
DEFAULT_CHANNELS = 1
DEFAULT_SAMPLE_WIDTH_BYTES = 2
_SPEAKER_PREFIX_RE = re.compile(
    r"^\s*\[?\s*((?:speaker[_ ]?\d+|spk[_ ]?\d+))\s*\]?\s*[:\-]",
    re.IGNORECASE,
)


def _repo_root() -> Path:
    # app/live/runtime/asr_bridge.py -> runtime -> live -> app -> repo root
    return Path(__file__).resolve().parents[3]


def _safe_session_id(session_id: str) -> str:
    text = str(session_id or "").strip()
    if not text:
        return "unknown"
    return "".join(ch if (ch.isalnum() or ch in {"-", "_", "."}) else "_" for ch in text)


def _srt_ts_to_ms(token: str) -> int:
    t = str(token or "").strip()
    if not t:
        return 0
    # "HH:MM:SS,mmm"
    parts = t.split(":")
    if len(parts) != 3:
        return 0
    h = int(parts[0] or 0)
    m = int(parts[1] or 0)
    sec_ms = parts[2].split(",")
    s = int(sec_ms[0] or 0)
    ms = int(sec_ms[1] or 0) if len(sec_ms) > 1 else 0
    return max(0, ((h * 3600 + m * 60 + s) * 1000) + ms)


def _parse_srt_segments(srt_text: str, *, t0_offset_ms: int = 0) -> list[dict[str, Any]]:
    text = str(srt_text or "").replace("\r\n", "\n").replace("\r", "\n")
    blocks = [blk.strip() for blk in text.split("\n\n") if blk.strip()]
    out: list[dict[str, Any]] = []
    seg_index = 0
    for blk in blocks:
        lines = [ln for ln in blk.split("\n") if ln.strip()]
        if len(lines) < 2:
            continue
        time_line_idx = 1 if "-->" in lines[1] else 0
        if "-->" not in lines[time_line_idx]:
            continue
        a, b = [p.strip() for p in lines[time_line_idx].split("-->", 1)]
        t0_ms = _srt_ts_to_ms(a) + int(max(0, t0_offset_ms))
        t1_ms = _srt_ts_to_ms(b) + int(max(0, t0_offset_ms))
        txt_lines = lines[time_line_idx + 1 :]
        seg_text = "\n".join(s.rstrip() for s in txt_lines if s.strip()).strip()
        if not seg_text:
            continue
        speaker = ""
        m = _SPEAKER_PREFIX_RE.match(seg_text)
        if m:
            speaker = str(m.group(1) or "").strip().upper().replace(" ", "_")
        seg_index += 1
        out.append(
            {
                "segment_id": f"s{seg_index:04d}",
                "text": seg_text,
                "t0_ms": int(max(0, t0_ms)),
                "t1_ms": int(max(t0_ms, t1_ms)),
                "speaker": speaker,
            }
        )
    return out


def _srt_to_plain_text(srt_text: str) -> str:
    segs = _parse_srt_segments(srt_text)
    return "\n".join(seg["text"] for seg in segs if str(seg.get("text") or "").strip())


@dataclass(frozen=True)
class EnqueuedChunkJob:
    session_id: str
    chunk_index: int
    job_id: str
    job_dir: str
    chunk_wav_path: str
    language: str | None
    initial_prompt_chars: int = 0
    initial_prompt_words: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": str(self.session_id),
            "chunk_index": int(self.chunk_index),
            "job_id": str(self.job_id),
            "job_dir": str(self.job_dir),
            "chunk_wav_path": str(self.chunk_wav_path),
            "language": self.language,
            "initial_prompt_chars": int(max(0, self.initial_prompt_chars)),
            "initial_prompt_words": int(max(0, self.initial_prompt_words)),
        }


@dataclass(frozen=True)
class ChunkJobPollResult:
    job_id: str
    state: str
    status: dict[str, Any]
    done: bool
    ok: bool
    error: str
    text: str
    srt_text: str
    segments: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": str(self.job_id),
            "state": str(self.state),
            "done": bool(self.done),
            "ok": bool(self.ok),
            "error": str(self.error),
            "text": str(self.text),
            "segments": [dict(seg) for seg in self.segments],
            "status": dict(self.status),
        }


class LiveChunkBatchBridge:
    def __init__(
        self,
        *,
        chunks_root: Path | None = None,
        sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ,
        channels: int = DEFAULT_CHANNELS,
        language: str | None = None,
        diarize_enabled: bool = False,
        diarize_speaker_mode: str = "fixed",
        diarize_min_speakers: int | None = 1,
        diarize_max_speakers: int | None = 4,
    ) -> None:
        self.chunks_root = (
            chunks_root if chunks_root is not None else (_repo_root() / "data" / "live" / "chunk_jobs")
        ).resolve()
        self.sample_rate_hz = int(max(1, sample_rate_hz))
        self.channels = int(max(1, channels))
        self.language = _normalize_optional_language(language)
        self.diarize_enabled = bool(diarize_enabled)
        mode = str(diarize_speaker_mode or "fixed").strip().lower()
        self.diarize_speaker_mode = mode if mode in {"none", "auto", "fixed"} else "fixed"
        try:
            self.diarize_min_speakers = None if diarize_min_speakers is None else int(max(1, int(diarize_min_speakers)))
        except Exception:
            self.diarize_min_speakers = None
        try:
            self.diarize_max_speakers = None if diarize_max_speakers is None else int(max(1, int(diarize_max_speakers)))
        except Exception:
            self.diarize_max_speakers = None
        if (
            self.diarize_min_speakers is not None
            and self.diarize_max_speakers is not None
            and self.diarize_max_speakers < self.diarize_min_speakers
        ):
            self.diarize_max_speakers = int(self.diarize_min_speakers)
        self._pool_client = ASRPoolClient(
            ASRPoolClientConfig(
                base_url=str(get_setting("asr_pool.base_url", "http://127.0.0.1:8090") or "http://127.0.0.1:8090"),
                token=str(get_setting("asr_pool.token", "") or ""),
            )
        )
        self._request_meta: dict[str, dict[str, Any]] = {}
        self._terminal_events: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._stream_stop = threading.Event()
        self._stream_thread: threading.Thread | None = None
        self._stream_consumer_id: str = ""
        self._stream_terminal_notify: Callable[[], None] | None = None
        self._feed_generation: int = 0

    def _request_id(self, *, safe_session_id: str, chunk_index: int, t0_ms: int, t1_ms: int) -> str:
        base = (
            f"live_{safe_session_id[:48]}_"
            f"{int(max(0, chunk_index)):06d}_"
            f"{int(max(0, t0_ms)):09d}_"
            f"{int(max(0, t1_ms)):09d}"
        )
        return base[:160]

    def _request_consumer_id(self, *, safe_session_id: str) -> str:
        return f"live_{safe_session_id[:96]}"

    def _with_timing_fields(self, *, status: dict[str, Any], timings: dict[str, Any]) -> dict[str, Any]:
        merged = dict(status or {})
        resolved_timings = dict(timings or {})
        merged["asr_timings"] = resolved_timings
        merged["asr_timing_runner_wall_s"] = _safe_float(resolved_timings.get("total_s"))
        merged["asr_timing_prepare_s"] = _safe_float(resolved_timings.get("prepare_s"))
        merged["asr_timing_load_audio_s"] = _safe_float(resolved_timings.get("load_audio_s"))
        merged["asr_timing_transcribe_call_s"] = _safe_float(resolved_timings.get("transcribe_call_s"))
        merged["asr_timing_transcribe_s"] = _safe_float(resolved_timings.get("transcribe_s"))
        merged["asr_timing_transcribe_overhead_s"] = _safe_float(resolved_timings.get("transcribe_overhead_s"))
        merged["asr_timing_align_s"] = _safe_float(resolved_timings.get("align_s"))
        merged["asr_timing_diarize_s"] = _safe_float(resolved_timings.get("diarize_s"))
        merged["asr_timing_finalize_s"] = _safe_float(resolved_timings.get("finalize_s"))
        merged["asr_timing_pool_ingest_s"] = _safe_float(resolved_timings.get("pool_ingest_s"))
        merged["asr_timing_pool_ingest_body_read_s"] = _safe_float(
            resolved_timings.get("pool_ingest_body_read_s")
        )
        merged["asr_timing_pool_ingest_multipart_parse_s"] = _safe_float(
            resolved_timings.get("pool_ingest_multipart_parse_s")
        )
        merged["asr_timing_pool_ingest_audio_write_s"] = _safe_float(
            resolved_timings.get("pool_ingest_audio_write_s")
        )
        merged["asr_timing_pool_ingest_submit_enqueue_s"] = _safe_float(
            resolved_timings.get("pool_ingest_submit_enqueue_s")
        )
        merged["asr_timing_pool_queue_wait_s"] = _safe_float(resolved_timings.get("pool_queue_wait_s"))
        merged["asr_timing_pool_wall_s"] = _safe_float(resolved_timings.get("pool_wall_s"))
        merged["asr_timing_pool_outside_runner_s"] = _safe_float(resolved_timings.get("pool_outside_runner_s"))
        merged["asr_timing_backend_wav_write_s"] = _safe_float(resolved_timings.get("backend_wav_write_s"))
        merged["asr_timing_backend_submit_s"] = _safe_float(resolved_timings.get("backend_submit_s"))
        merged["asr_timing_backend_result_collect_s"] = _safe_float(resolved_timings.get("backend_result_collect_s"))
        merged["asr_timing_backend_artifact_get_s"] = _safe_float(resolved_timings.get("backend_artifact_get_s"))
        merged["asr_timing_backend_srt_parse_s"] = _safe_float(resolved_timings.get("backend_srt_parse_s"))
        merged["asr_timing_backend_wall_s"] = _safe_float(resolved_timings.get("backend_wall_s"))
        merged["asr_timing_backend_outside_pool_s"] = _safe_float(resolved_timings.get("backend_outside_pool_s"))
        return merged

    def _pool_status_from_response(self, *, request_id: str, response: dict[str, Any]) -> dict[str, Any]:
        timings = dict(response.get("timings") or {})
        runtime_meta = dict(response.get("runtime") or {})
        status = {
            "state": "done",
            "phase": "done",
            "progress": 1.0,
            "message": "Done",
            "error": None,
            "asr_request_id": str(request_id),
            "asr_state": "completed",
            "asr_runtime": runtime_meta,
        }
        return self._with_timing_fields(status=status, timings=timings)

    def _augment_backend_status(
        self,
        *,
        status: dict[str, Any],
        meta: dict[str, Any],
        result_collect_s: float,
        artifact_get_s: float = 0.0,
        srt_parse_s: float = 0.0,
    ) -> dict[str, Any]:
        merged = dict(status or {})
        timings = dict(merged.get("asr_timings") or {})

        backend_wav_write_s = _safe_float(meta.get("backend_wav_write_s"))
        if backend_wav_write_s is not None:
            timings["backend_wav_write_s"] = round(float(backend_wav_write_s), 6)

        backend_submit_s = _safe_float(meta.get("backend_submit_s"))
        if backend_submit_s is not None:
            timings["backend_submit_s"] = round(float(backend_submit_s), 6)

        timings["backend_result_collect_s"] = round(max(0.0, float(result_collect_s)), 6)
        timings["backend_artifact_get_s"] = round(max(0.0, float(artifact_get_s)), 6)
        timings["backend_srt_parse_s"] = round(max(0.0, float(srt_parse_s)), 6)

        backend_started_mono = _safe_float(meta.get("backend_started_mono"))
        backend_wall_s: float | None = None
        if backend_started_mono is not None:
            backend_wall_s = max(0.0, float(time.monotonic()) - float(backend_started_mono))
            timings["backend_wall_s"] = round(float(backend_wall_s), 6)

        pool_wall_s = _safe_float(timings.get("pool_wall_s"))
        if backend_wall_s is not None and pool_wall_s is not None:
            timings["backend_outside_pool_s"] = round(max(0.0, float(backend_wall_s) - float(pool_wall_s)), 6)

        return self._with_timing_fields(status=merged, timings=timings)

    def _on_completion_stream_event(self, kind: str, payload: dict[str, Any]) -> None:
        event_kind = str(kind or "").strip().lower()
        data = dict(payload or {})
        notify: Callable[[], None] | None = None
        if event_kind == "completion":
            rid = str(data.get("request_id") or "").strip()
            state = str(data.get("state") or "").strip().lower()
            if not rid or state not in {"completed", "failed", "cancelled"}:
                return
            with self._lock:
                self._terminal_events[rid] = data
                notify = self._stream_terminal_notify
            if notify is not None:
                try:
                    notify()
                except Exception:
                    pass
            return
        if event_kind == "feed_reset":
            with self._lock:
                self._feed_generation = int(max(0, self._feed_generation + 1))
                notify = self._stream_terminal_notify
            if notify is not None:
                try:
                    notify()
                except Exception:
                    pass

    def start_completion_stream(self, *, session_id: str, on_terminal_event: Callable[[], None] | None = None) -> None:
        safe_id = _safe_session_id(session_id)
        consumer_id = self._request_consumer_id(safe_session_id=safe_id)
        with self._lock:
            running = self._stream_thread is not None and self._stream_thread.is_alive()
            if running:
                if self._stream_consumer_id != consumer_id:
                    raise RuntimeError("Completion stream already running for a different consumer_id")
                self._stream_terminal_notify = on_terminal_event
                return
            self._stream_consumer_id = str(consumer_id)
            self._stream_terminal_notify = on_terminal_event
            self._stream_stop = threading.Event()
            self._terminal_events.clear()
            self._feed_generation = 0
            stop_event = self._stream_stop

        def _run() -> None:
            try:
                for event in self._pool_client.iter_completions(
                    consumer_id=consumer_id,
                    since_seq=0,
                    stop_event=stop_event,
                ):
                    if isinstance(event, ASRCompletionEvent):
                        payload = event.status.to_dict()
                        payload["seq"] = int(event.seq)
                        payload["ts_utc"] = str(event.ts_utc)
                        self._on_completion_stream_event("completion", payload)
                    elif isinstance(event, ASRCompletionFeedReset):
                        self._on_completion_stream_event(
                            "feed_reset",
                            {
                                "old_feed_id": str(event.old_feed_id),
                                "new_feed_id": str(event.new_feed_id),
                            },
                        )
            except ASRPoolError:
                return

        thread = threading.Thread(
            target=_run,
            name=f"live-completions-{safe_id[:24]}",
            daemon=True,
        )
        with self._lock:
            self._stream_thread = thread
        thread.start()

    def stop_completion_stream(self) -> None:
        with self._lock:
            stop_event = self._stream_stop
            thread = self._stream_thread
            self._stream_thread = None
            self._stream_consumer_id = ""
            self._stream_terminal_notify = None
        stop_event.set()
        if thread is not None:
            thread.join(timeout=1.0)

    def has_terminal_result(self, job_id: str) -> bool:
        rid = str(job_id or "").strip()
        if not rid:
            return False
        with self._lock:
            if rid in self._terminal_events:
                return True
            meta = dict(self._request_meta.get(rid) or {})
            if not meta:
                return False
            submitted_feed_generation = int(max(0, int(meta.get("feed_generation") or 0)))
            return submitted_feed_generation < int(max(0, self._feed_generation))

    def enqueue_chunk_pcm16(
        self,
        *,
        session_id: str,
        chunk_index: int,
        t0_ms: int,
        t1_ms: int,
        pcm16le: bytes,
        language: str | None = None,
        asr_beam_size: int | None = None,
        asr_chunk_size: int | None = None,
        asr_backend: str | None = None,
        initial_prompt: str | None = None,
    ) -> EnqueuedChunkJob:
        backend_started_mono = time.monotonic()
        safe_id = _safe_session_id(session_id)
        idx = int(max(0, chunk_index))
        sess_dir = (self.chunks_root / safe_id).resolve()
        sess_dir.mkdir(parents=True, exist_ok=True)
        chunk_wav = (sess_dir / f"chunk_{idx:04d}_{int(max(0, t0_ms))}_{int(max(0, t1_ms))}.wav").resolve()

        raw = bytes(pcm16le or b"")
        if (len(raw) % DEFAULT_SAMPLE_WIDTH_BYTES) != 0:
            raw = raw[: len(raw) - 1]
        wav_write_started_mono = time.monotonic()
        with wave.open(str(chunk_wav), "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(DEFAULT_SAMPLE_WIDTH_BYTES)
            wf.setframerate(self.sample_rate_hz)
            wf.writeframes(raw)
        backend_wav_write_s = max(0.0, float(time.monotonic()) - float(wav_write_started_mono))

        prompt_text = str(initial_prompt or "").strip()
        prompt_words = len([tok for tok in prompt_text.split() if tok])
        resolved_language = _normalize_optional_language(language if language is not None else self.language)
        speaker_mode = self.diarize_speaker_mode if self.diarize_enabled else "none"
        if speaker_mode == "none":
            min_speakers = None
            max_speakers = None
        else:
            min_speakers = self.diarize_min_speakers
            max_speakers = self.diarize_max_speakers
        request_id = self._request_id(
            safe_session_id=safe_id,
            chunk_index=idx,
            t0_ms=int(max(0, t0_ms)),
            t1_ms=int(max(max(0, t0_ms), t1_ms)),
        )
        consumer_id = self._request_consumer_id(safe_session_id=safe_id)
        resolved_backend: str | None = None
        if asr_backend is not None:
            backend = str(asr_backend).strip().lower()
            if backend:
                resolved_backend = backend
        submit_request = ASRSubmitRequest(
            request_id=str(request_id),
            consumer_id=str(consumer_id),
            priority="interactive",
            routing=ASRRequestRouting(
                fairness_key=(str(session_id) or "live"),
            ),
            audio=ASRAudioFile(
                path=chunk_wav,
                duration_ms=int(max(1, max(max(0, t0_ms), t1_ms) - max(0, t0_ms))),
                format="wav",
                sample_rate_hz=int(self.sample_rate_hz),
                channels=int(self.channels),
            ),
            options=ASRRequestOptions(
                language=resolved_language,
                initial_prompt=(prompt_text or None),
                align_enabled=bool(get_setting("live.asr.align_enabled", False)),
                diarize_enabled=bool(self.diarize_enabled) and speaker_mode != "none",
                speaker_mode=speaker_mode,
                min_speakers=(int(max(1, min_speakers)) if speaker_mode == "fixed" and min_speakers is not None else None),
                max_speakers=(int(max(1, max_speakers)) if speaker_mode == "fixed" and max_speakers is not None else None),
                beam_size=(int(max(1, asr_beam_size)) if asr_beam_size is not None else None),
                chunk_size=(int(max(1, asr_chunk_size)) if asr_chunk_size is not None else None),
                asr_backend=resolved_backend,
            ),
            outputs=ASROutputSelection(
                text=False,
                segments=False,
                srt=True,
                srt_inline=True,
            ),
        )
        submit_started_mono = time.monotonic()
        try:
            submit_status = self._pool_client.submit_audio(submit_request)
        except ASRPoolError as e:
            raise RuntimeError(f"{e.code}: {e.message}") from e
        backend_submit_s = max(0.0, float(time.monotonic()) - float(submit_started_mono))
        accepted_request_id = str(submit_status.request_id or request_id).strip() or request_id
        with self._lock:
            feed_generation = int(max(0, self._feed_generation))
            self._request_meta[str(accepted_request_id)] = {
                "session_id": str(session_id),
                "consumer_id": str(consumer_id),
                "feed_generation": int(feed_generation),
                "backend_started_mono": float(backend_started_mono),
                "backend_wav_write_s": round(float(backend_wav_write_s), 6),
                "backend_submit_s": round(float(backend_submit_s), 6),
            }
        return EnqueuedChunkJob(
            session_id=str(session_id),
            chunk_index=idx,
            job_id=str(accepted_request_id),
            job_dir=str(sess_dir),
            chunk_wav_path=str(chunk_wav),
            language=resolved_language,
            initial_prompt_chars=len(prompt_text),
            initial_prompt_words=prompt_words,
        )

    def take_terminal_result(self, job_id: str, *, t0_offset_ms: int = 0) -> ChunkJobPollResult:
        rid = str(job_id or "").strip()
        if not rid:
            raise FileNotFoundError(f"Job not found: {job_id}")
        with self._lock:
            meta = dict(self._request_meta.get(rid) or {})
            terminal = dict(self._terminal_events.pop(rid, {}) or {})
            feed_generation = int(max(0, self._feed_generation))
        if not meta:
            raise FileNotFoundError(f"Job metadata missing for: {rid}")
        submitted_feed_generation = int(max(0, int(meta.get("feed_generation") or 0)))
        if not terminal:
            if submitted_feed_generation < feed_generation:
                err_code = "ASR_POOL_FEED_RESET"
                err_msg = "ASR pool completion feed reset detected while request was in-flight"
                status = {
                    "state": "error",
                    "phase": "error",
                    "progress": 1.0,
                    "message": f"{err_code}: {err_msg}",
                    "error": f"{err_code}: {err_msg}",
                    "asr_request_id": rid,
                    "asr_state": "feed_reset",
                }
                status = self._augment_backend_status(status=status, meta=meta, result_collect_s=0.0)
                with self._lock:
                    self._request_meta.pop(rid, None)
                return ChunkJobPollResult(
                    job_id=rid,
                    state="error",
                    status=status,
                    done=True,
                    ok=False,
                    error=f"{err_code}: {err_msg}",
                    text="",
                    srt_text="",
                    segments=[],
                )
            return ChunkJobPollResult(
                job_id=rid,
                state="queued",
                status={},
                done=False,
                ok=False,
                error="",
                text="",
                srt_text="",
                segments=[],
            )
        raw_state = str(terminal.get("state") or "").strip().lower()
        result_collect_started_mono = time.monotonic()
        if raw_state == "completed":
            artifact_get_s = 0.0
            srt_parse_s = 0.0
            srt_text = ""
            plain = ""
            segments: list[dict[str, Any]] = []
            response = dict(terminal.get("response") or {})
            result = dict(response.get("result") or {})
            if "srt_text" not in result:
                raise RuntimeError(f"Missing inline SRT in terminal response for request: {rid}")
            srt_parse_started_mono = time.monotonic()
            srt_text = str(result.get("srt_text") or "")
            segments = _parse_srt_segments(srt_text, t0_offset_ms=int(max(0, t0_offset_ms)))
            plain = "\n".join(str(seg.get("text") or "").strip() for seg in segments if str(seg.get("text") or "").strip())
            if not plain:
                plain = _srt_to_plain_text(srt_text)
            srt_parse_s = max(0.0, float(time.monotonic()) - float(srt_parse_started_mono))
            status = self._pool_status_from_response(request_id=rid, response=response)
            status = self._augment_backend_status(
                status=status,
                meta=meta,
                result_collect_s=max(0.0, float(time.monotonic()) - float(result_collect_started_mono)),
                artifact_get_s=artifact_get_s,
                srt_parse_s=srt_parse_s,
            )
            with self._lock:
                self._request_meta.pop(rid, None)
            return ChunkJobPollResult(
                job_id=rid,
                state="done",
                status=status,
                done=True,
                ok=True,
                error="",
                text=plain,
                srt_text=srt_text,
                segments=segments,
            )
        err_obj = dict(terminal.get("error") or {})
        err_code = str(err_obj.get("code") or "ASR_REMOTE_TERMINAL_ERROR")
        err_msg = str(err_obj.get("message") or f"ASR terminal state: {raw_state or 'unknown'}")
        status = {
            "state": "error",
            "phase": "error",
            "progress": 1.0,
            "message": f"{err_code}: {err_msg}",
            "error": f"{err_code}: {err_msg}",
            "asr_request_id": rid,
            "asr_state": raw_state,
        }
        response = dict(terminal.get("response") or {})
        if response:
            status = self._with_timing_fields(status=status, timings=dict(response.get("timings") or {}))
        status = self._augment_backend_status(
            status=status,
            meta=meta,
            result_collect_s=max(0.0, float(time.monotonic()) - float(result_collect_started_mono)),
        )
        with self._lock:
            self._request_meta.pop(rid, None)
        return ChunkJobPollResult(
            job_id=rid,
            state="error",
            status=status,
            done=True,
            ok=False,
            error=f"{err_code}: {err_msg}",
            text="",
            srt_text="",
            segments=[],
        )
