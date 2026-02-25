from __future__ import annotations

import math
import struct
from collections import deque
from dataclasses import dataclass
from typing import Deque


DEFAULT_SAMPLE_RATE_HZ = 16000
DEFAULT_CHANNELS = 1
DEFAULT_SAMPLE_WIDTH_BYTES = 2
DEFAULT_FRAME_MS = 20


def _safe_int(value: int, default: int, *, min_value: int | None = None) -> int:
    try:
        out = int(value)
    except Exception:
        out = int(default)
    if min_value is not None:
        out = max(int(min_value), out)
    return out


def _bytes_to_ms(byte_count: int, *, sample_rate_hz: int, channels: int) -> int:
    bps = max(1, int(sample_rate_hz) * int(channels) * DEFAULT_SAMPLE_WIDTH_BYTES)
    if byte_count <= 0:
        return 0
    return int(round((float(byte_count) * 1000.0) / float(bps)))


def _pcm16_rms(frame_bytes: bytes) -> int:
    if not frame_bytes:
        return 0
    total = 0
    count = 0
    for (sample,) in struct.iter_unpack("<h", frame_bytes):
        total += int(sample) * int(sample)
        count += 1
    if count <= 0:
        return 0
    return int(round(math.sqrt(float(total) / float(count))))


@dataclass(frozen=True)
class LiveChunkerConfig:
    sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ
    channels: int = DEFAULT_CHANNELS
    frame_ms: int = DEFAULT_FRAME_MS
    energy_threshold: int = 500
    silence_threshold_ms: int = 1200
    max_chunk_ms: int = 20000
    min_chunk_ms: int = 800
    pre_roll_ms: int = 200
    post_roll_ms: int = 200


@dataclass(frozen=True)
class ClosedAudioChunk:
    chunk_index: int
    t0_ms: int
    t1_ms: int
    pcm16le: bytes
    reason: str  # "silence" | "max_duration" | "flush_tail"
    speech_frames: int
    silence_frames_tail: int

    @property
    def duration_ms(self) -> int:
        return max(0, int(self.t1_ms) - int(self.t0_ms))

    def to_dict(self) -> dict[str, object]:
        return {
            "chunk_index": int(self.chunk_index),
            "t0_ms": int(self.t0_ms),
            "t1_ms": int(self.t1_ms),
            "duration_ms": int(self.duration_ms),
            "pcm_bytes": int(len(self.pcm16le)),
            "reason": str(self.reason),
            "speech_frames": int(self.speech_frames),
            "silence_frames_tail": int(self.silence_frames_tail),
        }


class LiveAudioChunker:
    def __init__(self, *, config: LiveChunkerConfig | None = None) -> None:
        cfg = config or LiveChunkerConfig()
        self.config = LiveChunkerConfig(
            sample_rate_hz=_safe_int(cfg.sample_rate_hz, DEFAULT_SAMPLE_RATE_HZ, min_value=1),
            channels=_safe_int(cfg.channels, DEFAULT_CHANNELS, min_value=1),
            frame_ms=_safe_int(cfg.frame_ms, DEFAULT_FRAME_MS, min_value=5),
            energy_threshold=_safe_int(cfg.energy_threshold, 500, min_value=0),
            silence_threshold_ms=_safe_int(cfg.silence_threshold_ms, 1200, min_value=0),
            max_chunk_ms=_safe_int(cfg.max_chunk_ms, 20000, min_value=200),
            min_chunk_ms=_safe_int(cfg.min_chunk_ms, 800, min_value=0),
            pre_roll_ms=_safe_int(cfg.pre_roll_ms, 200, min_value=0),
            post_roll_ms=_safe_int(cfg.post_roll_ms, 200, min_value=0),
        )

        bytes_per_second = self.config.sample_rate_hz * self.config.channels * DEFAULT_SAMPLE_WIDTH_BYTES
        self._frame_size_bytes = max(2, int(round((bytes_per_second * self.config.frame_ms) / 1000.0)))
        if (self._frame_size_bytes % DEFAULT_SAMPLE_WIDTH_BYTES) != 0:
            self._frame_size_bytes += 1

        self._pending = bytearray()
        self._frames_seen = 0
        self._chunk_index_next = 0

        self._current_chunk_bytes: bytearray | None = None
        self._current_chunk_start_frame = 0
        self._current_chunk_speech_frames = 0
        self._current_chunk_tail_silence_frames = 0

        self._pre_roll_frames_max = max(0, int(math.ceil(self.config.pre_roll_ms / max(1, self.config.frame_ms))))
        self._pre_roll_frames: Deque[bytes] = deque(maxlen=self._pre_roll_frames_max)

    def feed_pcm16(self, audio_bytes: bytes) -> list[ClosedAudioChunk]:
        raw = bytes(audio_bytes or b"")
        if not raw:
            return []
        self._pending.extend(raw)
        out: list[ClosedAudioChunk] = []
        while len(self._pending) >= self._frame_size_bytes:
            frame = bytes(self._pending[: self._frame_size_bytes])
            del self._pending[: self._frame_size_bytes]
            out.extend(self._consume_frame(frame))
        return out

    def flush_tail(self) -> list[ClosedAudioChunk]:
        out: list[ClosedAudioChunk] = []
        if self._pending:
            # Zero-pad to a full analysis frame so we do not drop tail audio.
            pad_len = self._frame_size_bytes - len(self._pending)
            frame = bytes(self._pending) + (b"\x00" * max(0, pad_len))
            self._pending.clear()
            out.extend(self._consume_frame(frame))
        if self._current_chunk_bytes is not None and len(self._current_chunk_bytes) > 0:
            out.append(self._close_chunk(reason="flush_tail"))
        return out

    def snapshot(self) -> dict[str, int | bool]:
        active_frames = 0
        if self._current_chunk_bytes is not None:
            active_frames = len(self._current_chunk_bytes) // self._frame_size_bytes
        return {
            "chunk_index_next": int(self._chunk_index_next),
            "frames_seen": int(self._frames_seen),
            "pending_bytes": int(len(self._pending)),
            "chunk_open": bool(self._current_chunk_bytes is not None),
            "active_chunk_frames": int(active_frames),
            "active_chunk_duration_ms": int(active_frames * self.config.frame_ms),
            "pre_roll_frames_buffered": int(len(self._pre_roll_frames)),
        }

    def _consume_frame(self, frame: bytes) -> list[ClosedAudioChunk]:
        frame_index = int(self._frames_seen)
        self._frames_seen += 1

        rms = _pcm16_rms(frame)
        is_speech = rms >= self.config.energy_threshold

        out: list[ClosedAudioChunk] = []

        if self._current_chunk_bytes is None:
            if is_speech:
                self._open_chunk(frame_index=frame_index, first_frame=frame)
            else:
                if self._pre_roll_frames_max > 0:
                    self._pre_roll_frames.append(frame)
            return out

        # Chunk already open.
        self._current_chunk_bytes.extend(frame)
        if is_speech:
            self._current_chunk_speech_frames += 1
            self._current_chunk_tail_silence_frames = 0
        else:
            self._current_chunk_tail_silence_frames += 1

        chunk_duration_ms = self._active_chunk_duration_ms()
        silence_close_ms = self.config.silence_threshold_ms + self.config.post_roll_ms
        tail_silence_ms = self._current_chunk_tail_silence_frames * self.config.frame_ms

        if chunk_duration_ms >= self.config.max_chunk_ms:
            out.append(self._close_chunk(reason="max_duration"))
            if not is_speech and self._pre_roll_frames_max > 0:
                self._pre_roll_frames.append(frame)
            return out

        if (
            self._current_chunk_speech_frames > 0
            and chunk_duration_ms >= self.config.min_chunk_ms
            and tail_silence_ms >= silence_close_ms
        ):
            out.append(self._close_chunk(reason="silence"))
            return out

        return out

    def _open_chunk(self, *, frame_index: int, first_frame: bytes) -> None:
        prefix = list(self._pre_roll_frames)
        self._pre_roll_frames.clear()
        prefix_frames = len(prefix)

        self._current_chunk_start_frame = max(0, int(frame_index) - prefix_frames)
        buf = bytearray()
        for f in prefix:
            buf.extend(f)
        buf.extend(first_frame)
        self._current_chunk_bytes = buf

        self._current_chunk_speech_frames = 1
        self._current_chunk_tail_silence_frames = 0

    def _close_chunk(self, *, reason: str) -> ClosedAudioChunk:
        if self._current_chunk_bytes is None:
            raise RuntimeError("chunk_not_open")

        chunk_bytes = bytes(self._current_chunk_bytes)
        chunk_frames = len(chunk_bytes) // self._frame_size_bytes
        t0_ms = int(self._current_chunk_start_frame * self.config.frame_ms)
        t1_ms = int(t0_ms + (chunk_frames * self.config.frame_ms))

        chunk = ClosedAudioChunk(
            chunk_index=int(self._chunk_index_next),
            t0_ms=max(0, t0_ms),
            t1_ms=max(t0_ms, t1_ms),
            pcm16le=chunk_bytes,
            reason=str(reason),
            speech_frames=int(self._current_chunk_speech_frames),
            silence_frames_tail=int(self._current_chunk_tail_silence_frames),
        )
        self._chunk_index_next += 1

        self._current_chunk_bytes = None
        self._current_chunk_start_frame = 0
        self._current_chunk_speech_frames = 0
        self._current_chunk_tail_silence_frames = 0
        return chunk

    def _active_chunk_duration_ms(self) -> int:
        if self._current_chunk_bytes is None:
            return 0
        return _bytes_to_ms(
            len(self._current_chunk_bytes),
            sample_rate_hz=self.config.sample_rate_hz,
            channels=self.config.channels,
        )

