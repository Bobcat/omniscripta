from __future__ import annotations

import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path


DEFAULT_SAMPLE_RATE_HZ = 16000
DEFAULT_CHANNELS = 1
DEFAULT_SAMPLE_WIDTH_BYTES = 2


def _repo_root() -> Path:
    # portal-api/live_recordings.py -> portal-api -> repo root
    return Path(__file__).resolve().parents[1]


def _safe_session_id(session_id: str) -> str:
    text = str(session_id or "").strip()
    if not text:
        return "unknown"
    return "".join(ch if (ch.isalnum() or ch in {"-", "_", "."}) else "_" for ch in text)


def _bytes_to_ms(byte_count: int, *, sample_rate_hz: int, channels: int) -> int:
    bps = max(1, int(sample_rate_hz) * int(channels) * DEFAULT_SAMPLE_WIDTH_BYTES)
    if byte_count <= 0:
        return 0
    return int(round((float(byte_count) * 1000.0) / float(bps)))


@dataclass(frozen=True)
class LiveRecordingSnapshot:
    session_id: str
    state: str
    wav_path: str
    bytes_written: int
    duration_ms: int
    sample_rate_hz: int
    channels: int
    started_mono: float
    finalized_mono: float

    @property
    def elapsed_s(self) -> float:
        end_mono = self.finalized_mono if self.finalized_mono > 0 else time.monotonic()
        return max(0.0, float(end_mono - self.started_mono))

    def to_dict(self) -> dict[str, object]:
        return {
            "session_id": str(self.session_id),
            "state": str(self.state),
            "wav_path": str(self.wav_path),
            "bytes_written": int(self.bytes_written),
            "duration_ms": int(self.duration_ms),
            "sample_rate_hz": int(self.sample_rate_hz),
            "channels": int(self.channels),
            "elapsed_s": round(self.elapsed_s, 3),
        }


class LiveWavRecorder:
    def __init__(
        self,
        *,
        session_id: str,
        sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ,
        channels: int = DEFAULT_CHANNELS,
        recordings_root: Path | None = None,
    ) -> None:
        self.session_id = str(session_id)
        self.sample_rate_hz = int(max(1, sample_rate_hz))
        self.channels = int(max(1, channels))
        self.recordings_root = (
            recordings_root if recordings_root is not None else (_repo_root() / "data" / "live_recordings")
        ).resolve()

        safe_id = _safe_session_id(self.session_id)
        self._session_dir = (self.recordings_root / safe_id).resolve()
        self._wav_path = (self._session_dir / "recording.wav").resolve()

        self._lock = threading.Lock()
        self._writer: wave.Wave_write | None = None
        self._started_mono = 0.0
        self._finalized_mono = 0.0
        self._bytes_written = 0
        self._state = "idle"

    @property
    def wav_path(self) -> Path:
        return self._wav_path

    @property
    def session_dir(self) -> Path:
        return self._session_dir

    def start(self) -> LiveRecordingSnapshot:
        with self._lock:
            if self._state == "recording":
                return self.snapshot()
            if self._state == "finalized":
                raise RuntimeError("recording_already_finalized")

            self._session_dir.mkdir(parents=True, exist_ok=True)
            writer = wave.open(str(self._wav_path), "wb")
            writer.setnchannels(self.channels)
            writer.setsampwidth(DEFAULT_SAMPLE_WIDTH_BYTES)
            writer.setframerate(self.sample_rate_hz)

            self._writer = writer
            self._started_mono = time.monotonic()
            self._finalized_mono = 0.0
            self._bytes_written = 0
            self._state = "recording"
            return self._snapshot_locked()

    def append_pcm16(self, audio_bytes: bytes) -> LiveRecordingSnapshot:
        raw = bytes(audio_bytes or b"")
        if not raw:
            return self.snapshot()

        with self._lock:
            if self._state != "recording" or self._writer is None:
                raise RuntimeError("recording_not_active")

            # PCM16LE should be sample-aligned. Trim a stray byte instead of failing the session.
            if (len(raw) % DEFAULT_SAMPLE_WIDTH_BYTES) != 0:
                raw = raw[: len(raw) - 1]
            if raw:
                self._writer.writeframesraw(raw)
                self._bytes_written += len(raw)
            return self._snapshot_locked()

    def finalize(self) -> LiveRecordingSnapshot:
        with self._lock:
            if self._state == "finalized":
                return self._snapshot_locked()
            if self._state != "recording" or self._writer is None:
                raise RuntimeError("recording_not_active")

            try:
                self._writer.close()
            finally:
                self._writer = None
            self._finalized_mono = time.monotonic()
            self._state = "finalized"
            return self._snapshot_locked()

    def abort(self) -> LiveRecordingSnapshot:
        with self._lock:
            if self._writer is not None:
                try:
                    self._writer.close()
                except Exception:
                    pass
                finally:
                    self._writer = None
            if self._state != "finalized":
                self._state = "aborted"
                if self._finalized_mono <= 0 and self._started_mono > 0:
                    self._finalized_mono = time.monotonic()
            return self._snapshot_locked()

    def snapshot(self) -> LiveRecordingSnapshot:
        with self._lock:
            return self._snapshot_locked()

    def _snapshot_locked(self) -> LiveRecordingSnapshot:
        return LiveRecordingSnapshot(
            session_id=self.session_id,
            state=str(self._state),
            wav_path=str(self._wav_path),
            bytes_written=int(max(0, self._bytes_written)),
            duration_ms=int(max(0, _bytes_to_ms(self._bytes_written, sample_rate_hz=self.sample_rate_hz, channels=self.channels))),
            sample_rate_hz=int(self.sample_rate_hz),
            channels=int(self.channels),
            started_mono=float(self._started_mono),
            finalized_mono=float(self._finalized_mono),
        )

