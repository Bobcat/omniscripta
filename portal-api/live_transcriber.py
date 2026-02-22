from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from live_engine_adapter import (
    DEFAULT_CHANNELS,
    DEFAULT_SAMPLE_RATE_HZ,
    EngineTranscriptEvent,
    create_live_engine_adapter,
)


def _bytes_to_ms(byte_count: int, *, sample_rate_hz: int, channels: int) -> int:
    bytes_per_second = max(1, int(sample_rate_hz) * int(channels) * 2)
    if byte_count <= 0:
        return 0
    ms = int(round((float(byte_count) * 1000.0) / float(bytes_per_second)))
    return max(1, ms)


@dataclass
class StableTranscriptEvent:
    kind: str  # "partial" | "final"
    text: str
    t0_ms: int
    t1_ms: int
    segment_id: str
    revision: int
    committed_chars: int
    committed_segments: int
    committed_until_ms: int


class LiveTranscriber:
    def __init__(
        self,
        *,
        session_id: str,
        sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ,
        channels: int = DEFAULT_CHANNELS,
        driver_override: str | None = None,
    ) -> None:
        self.session_id = str(session_id)
        self.sample_rate_hz = int(sample_rate_hz)
        self.channels = int(channels)

        self._engine = create_live_engine_adapter(
            session_id=self.session_id,
            sample_rate_hz=self.sample_rate_hz,
            channels=self.channels,
            driver_override=driver_override,
        )
        self._engine.start()

        self._decode_calls = 0
        self._decode_ms_total = 0.0
        self._decode_ms_last = 0.0
        self._audio_bytes_total = 0
        self._audio_ms_total = 0
        self._partials_emitted = 0
        self._finals_emitted = 0
        self._started_mono = time.monotonic()
        self._revision = 0
        self._segment_counter = 0
        self._committed_until_ms = 0
        self._final_segments: list[dict[str, Any]] = []
        self._partial_state: dict[str, Any] | None = None

    @property
    def driver_name(self) -> str:
        return str(getattr(self._engine, "driver_name", "unknown"))

    def _next_segment_id(self) -> str:
        self._segment_counter += 1
        return f"s{self._segment_counter:05d}"

    def _committed_text(self) -> str:
        return "\n".join(str(seg.get("text") or "") for seg in self._final_segments if str(seg.get("text") or "").strip())

    def _stabilize_events(self, events: list[EngineTranscriptEvent]) -> list[StableTranscriptEvent]:
        out: list[StableTranscriptEvent] = []
        for ev in events:
            kind = str(getattr(ev, "kind", "")).strip().lower()
            text = str(getattr(ev, "text", "") or "").strip()
            if not text:
                continue

            t0 = int(max(0, int(getattr(ev, "t0_ms", 0))))
            t1 = int(max(t0 + 1, int(getattr(ev, "t1_ms", t0 + 1))))

            if kind == "final":
                if t1 <= self._committed_until_ms:
                    continue
                t0 = max(t0, self._committed_until_ms)
                if t1 <= t0:
                    t1 = t0 + 1

                if self._final_segments:
                    prev = self._final_segments[-1]
                    if (
                        str(prev.get("text") or "").strip() == text
                        and int(prev.get("t0_ms") or 0) == t0
                        and int(prev.get("t1_ms") or 0) == t1
                    ):
                        continue

                seg_id = self._next_segment_id()
                segment = {
                    "segment_id": seg_id,
                    "text": text,
                    "t0_ms": int(t0),
                    "t1_ms": int(t1),
                }
                self._final_segments.append(segment)
                self._committed_until_ms = int(t1)
                self._partial_state = None
                self._revision += 1

                committed_text = self._committed_text()
                out.append(
                    StableTranscriptEvent(
                        kind="final",
                        text=text,
                        t0_ms=int(t0),
                        t1_ms=int(t1),
                        segment_id=seg_id,
                        revision=int(self._revision),
                        committed_chars=len(committed_text),
                        committed_segments=len(self._final_segments),
                        committed_until_ms=int(self._committed_until_ms),
                    )
                )
                continue

            if kind == "partial":
                if t1 <= self._committed_until_ms:
                    continue
                t0 = max(t0, self._committed_until_ms)
                if t1 <= t0:
                    t1 = t0 + 1

                sig = {"text": text, "t0_ms": int(t0), "t1_ms": int(t1)}
                if self._partial_state == sig:
                    continue
                self._partial_state = sig
                self._revision += 1
                committed_text = self._committed_text()
                out.append(
                    StableTranscriptEvent(
                        kind="partial",
                        text=text,
                        t0_ms=int(t0),
                        t1_ms=int(t1),
                        segment_id="tail",
                        revision=int(self._revision),
                        committed_chars=len(committed_text),
                        committed_segments=len(self._final_segments),
                        committed_until_ms=int(self._committed_until_ms),
                    )
                )
                continue

        return out

    def _record_decode(self, elapsed_ms: float, events: list[StableTranscriptEvent]) -> None:
        self._decode_calls += 1
        self._decode_ms_total += max(0.0, float(elapsed_ms))
        self._decode_ms_last = max(0.0, float(elapsed_ms))
        for ev in events:
            kind = str(getattr(ev, "kind", "")).strip().lower()
            if kind == "partial":
                self._partials_emitted += 1
            elif kind == "final":
                self._finals_emitted += 1

    def feed_audio(self, audio_bytes: bytes) -> list[StableTranscriptEvent]:
        size = len(audio_bytes or b"")
        if size <= 0:
            return []
        self._audio_bytes_total += size
        self._audio_ms_total += _bytes_to_ms(
            size,
            sample_rate_hz=self.sample_rate_hz,
            channels=self.channels,
        )

        t0 = time.monotonic()
        raw_events = self._engine.feed_audio(audio_bytes)
        events = self._stabilize_events(raw_events)
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        self._record_decode(elapsed_ms, events)
        return events

    def flush(self) -> list[StableTranscriptEvent]:
        t0 = time.monotonic()
        raw_events = self._engine.flush()
        events = self._stabilize_events(raw_events)
        if self._partial_state is not None:
            tail = dict(self._partial_state)
            self._partial_state = None
            text = str(tail.get("text") or "").strip()
            t0_ms = int(max(self._committed_until_ms, int(tail.get("t0_ms") or 0)))
            t1_ms = int(max(t0_ms + 1, int(tail.get("t1_ms") or (t0_ms + 1))))
            if text and t1_ms > self._committed_until_ms:
                seg_id = self._next_segment_id()
                self._final_segments.append(
                    {
                        "segment_id": seg_id,
                        "text": text,
                        "t0_ms": int(t0_ms),
                        "t1_ms": int(t1_ms),
                    }
                )
                self._committed_until_ms = int(t1_ms)
                self._revision += 1
                committed_text = self._committed_text()
                events.append(
                    StableTranscriptEvent(
                        kind="final",
                        text=text,
                        t0_ms=int(t0_ms),
                        t1_ms=int(t1_ms),
                        segment_id=seg_id,
                        revision=int(self._revision),
                        committed_chars=len(committed_text),
                        committed_segments=len(self._final_segments),
                        committed_until_ms=int(self._committed_until_ms),
                    )
                )
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        self._record_decode(elapsed_ms, events)
        return events

    def close(self) -> None:
        self._engine.close()

    def stats_snapshot(self) -> dict[str, Any]:
        engine_stats = self._engine.stats()
        audio_ms = max(0, int(self._audio_ms_total))
        decode_ms = max(0.0, float(self._decode_ms_total))
        rtf = (decode_ms / float(audio_ms)) if audio_ms > 0 else 0.0
        committed_text = self._committed_text()
        out: dict[str, Any] = {
            "engine_driver": self.driver_name,
            "decode_calls": int(self._decode_calls),
            "decode_ms_total": round(decode_ms, 3),
            "decode_ms_last": round(max(0.0, float(self._decode_ms_last)), 3),
            "rtf": round(max(0.0, rtf), 4),
            "audio_bytes": int(self._audio_bytes_total),
            "audio_ms": int(audio_ms),
            "partials_emitted": int(self._partials_emitted),
            "finals_emitted": int(self._finals_emitted),
            "engine_uptime_s": round(max(0.0, time.monotonic() - self._started_mono), 3),
            "revision": int(self._revision),
            "committed_until_ms": int(self._committed_until_ms),
            "committed_segments": len(self._final_segments),
            "committed_chars": len(committed_text),
            "has_partial_tail": bool(self._partial_state),
        }
        for key, val in engine_stats.items():
            out[str(key)] = val
        return out

    def transcript_snapshot(self) -> dict[str, Any]:
        return {
            "revision": int(self._revision),
            "committed_until_ms": int(self._committed_until_ms),
            "final_text": self._committed_text(),
            "final_segments": [
                {
                    "segment_id": str(seg.get("segment_id") or ""),
                    "text": str(seg.get("text") or ""),
                    "t0_ms": int(seg.get("t0_ms") or 0),
                    "t1_ms": int(seg.get("t1_ms") or 0),
                }
                for seg in self._final_segments
            ],
            "partial_tail": dict(self._partial_state) if self._partial_state else None,
        }
