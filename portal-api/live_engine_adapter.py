from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
import struct


DEFAULT_SAMPLE_RATE_HZ = 16000
DEFAULT_CHANNELS = 1
DEFAULT_SAMPLE_WIDTH_BYTES = 2
# Feed-loop drain should stay short to avoid stalling per incoming audio frame.
MAX_FEED_DRAIN_TIMEOUT_S = 0.02
MAX_INPUT_GAIN = 8.0


def _repo_root() -> Path:
    # portal-api/live_engine_adapter.py -> portal-api -> repo root
    return Path(__file__).resolve().parents[1]


def _safe_int(value: Any, default: int, *, min_value: int | None = None) -> int:
    try:
        out = int(value)
    except Exception:
        out = int(default)
    if min_value is not None:
        out = max(int(min_value), out)
    return out


def _safe_float(value: Any, default: float, *, min_value: float | None = None) -> float:
    try:
        out = float(value)
    except Exception:
        out = float(default)
    if min_value is not None:
        out = max(float(min_value), out)
    return out


def _clamp_float(value: float, *, min_value: float, max_value: float) -> float:
    return max(float(min_value), min(float(max_value), float(value)))


def _safe_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    s = str(value or "").strip().lower()
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _bytes_to_ms(byte_count: int, *, sample_rate_hz: int, channels: int) -> int:
    bps = max(1, int(sample_rate_hz) * int(channels) * DEFAULT_SAMPLE_WIDTH_BYTES)
    if byte_count <= 0:
        return 0
    ms = int(round((float(byte_count) * 1000.0) / float(bps)))
    return max(1, ms)


def _deep_merge(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    out = dict(dst)
    for key, val in src.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[str(key)] = _deep_merge(dict(out[str(key)]), val)
        else:
            out[str(key)] = val
    return out


def load_live_engine_config() -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "driver": "mock_stream",
        "sample_rate_hz": DEFAULT_SAMPLE_RATE_HZ,
        "channels": DEFAULT_CHANNELS,
        "mock_stream": {
            "partial_step_ms": 350,
            "final_step_ms": 1600,
        },
        "whisperlive_sidecar": {
            "ws_url": "ws://127.0.0.1:9090",
            "language": "nl",
            "task": "transcribe",
            "model": "medium",
            "input_gain": 1.0,
            "use_vad": True,
            "send_last_n_segments": 10,
            "no_speech_thresh": 0.45,
            "clip_audio": False,
            "same_output_threshold": 10,
            "enable_translation": False,
            "target_language": "fr",
            "open_timeout_s": 4.0,
            "ready_timeout_s": 3.0,
            "recv_timeout_s": 0.01,
            "flush_timeout_s": 1.0,
            "max_drain_messages": 12,
        },
    }

    service_path = Path(
        os.getenv("TRANSCRIBE_SERVICE_CONFIG")
        or str(_repo_root() / "config" / "service.json")
    )
    if not service_path.exists():
        return defaults

    try:
        raw = json.loads(service_path.read_text(encoding="utf-8"))
    except Exception:
        return defaults
    if not isinstance(raw, dict):
        return defaults

    live_raw = raw.get("live")
    if not isinstance(live_raw, dict):
        return defaults

    return _deep_merge(defaults, live_raw)


@dataclass
class EngineTranscriptEvent:
    kind: str  # "partial" | "final"
    text: str
    t0_ms: int
    t1_ms: int


class LiveEngineAdapter(Protocol):
    @property
    def driver_name(self) -> str:
        raise NotImplementedError

    def start(self) -> None:
        raise NotImplementedError

    def feed_audio(self, audio_bytes: bytes) -> list[EngineTranscriptEvent]:
        raise NotImplementedError

    def flush(self) -> list[EngineTranscriptEvent]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def stats(self) -> dict[str, Any]:
        raise NotImplementedError


class MockStreamEngine:
    _WORD_BANK = [
        "dit",
        "is",
        "een",
        "live",
        "transcriptie",
        "test",
        "zonder",
        "diarization",
        "met",
        "lage",
        "latency",
        "en",
        "stabiele",
        "tekst",
    ]

    def __init__(self, *, sample_rate_hz: int, channels: int, config: dict[str, Any]) -> None:
        self.sample_rate_hz = int(sample_rate_hz)
        self.channels = int(channels)
        self.partial_step_ms = _safe_int(config.get("partial_step_ms"), 350, min_value=100)
        self.final_step_ms = _safe_int(config.get("final_step_ms"), 1600, min_value=400)

        self._audio_ms_total = 0
        self._finalized_until_ms = 0
        self._next_partial_ms = self.partial_step_ms
        self._next_final_ms = self.final_step_ms
        self._segment_idx = 1
        self._last_partial_sig: tuple[str, int, int] | None = None
        self._partials_emitted = 0
        self._finals_emitted = 0

    @property
    def driver_name(self) -> str:
        return "mock_stream"

    def start(self) -> None:
        return None

    def _segment_text(self, idx: int) -> str:
        start = (int(idx) * 3) % len(self._WORD_BANK)
        words = [
            self._WORD_BANK[(start + n) % len(self._WORD_BANK)]
            for n in range(6)
        ]
        return " ".join(words)

    def _emit_due_events(self) -> list[EngineTranscriptEvent]:
        out: list[EngineTranscriptEvent] = []

        while self._audio_ms_total >= self._next_final_ms:
            t0 = int(self._finalized_until_ms)
            t1 = int(self._next_final_ms)
            text = self._segment_text(self._segment_idx)
            out.append(EngineTranscriptEvent(kind="final", text=text, t0_ms=t0, t1_ms=t1))
            self._finalized_until_ms = t1
            self._next_final_ms += self.final_step_ms
            self._segment_idx += 1
            self._last_partial_sig = None
            self._finals_emitted += 1

        if self._audio_ms_total >= self._next_partial_ms and self._audio_ms_total > self._finalized_until_ms:
            t0 = int(self._finalized_until_ms)
            t1 = int(self._audio_ms_total)
            text = f"{self._segment_text(self._segment_idx)} ..."
            sig = (text, t0, t1)
            if sig != self._last_partial_sig:
                out.append(EngineTranscriptEvent(kind="partial", text=text, t0_ms=t0, t1_ms=t1))
                self._last_partial_sig = sig
                self._partials_emitted += 1
            self._next_partial_ms += self.partial_step_ms

        return out

    def feed_audio(self, audio_bytes: bytes) -> list[EngineTranscriptEvent]:
        size = len(audio_bytes or b"")
        if size <= 0:
            return []
        self._audio_ms_total += _bytes_to_ms(
            size,
            sample_rate_hz=self.sample_rate_hz,
            channels=self.channels,
        )
        return self._emit_due_events()

    def flush(self) -> list[EngineTranscriptEvent]:
        if self._audio_ms_total <= self._finalized_until_ms:
            return []
        t0 = int(self._finalized_until_ms)
        t1 = int(self._audio_ms_total)
        text = self._segment_text(self._segment_idx)
        self._finalized_until_ms = t1
        self._segment_idx += 1
        self._last_partial_sig = None
        self._finals_emitted += 1
        return [EngineTranscriptEvent(kind="final", text=text, t0_ms=t0, t1_ms=t1)]

    def close(self) -> None:
        return None

    def stats(self) -> dict[str, Any]:
        return {
            "engine_driver": self.driver_name,
            "engine_audio_ms": int(self._audio_ms_total),
            "engine_partials": int(self._partials_emitted),
            "engine_finals": int(self._finals_emitted),
        }


class WhisperLiveSidecarEngine:
    def __init__(self, *, session_id: str, sample_rate_hz: int, channels: int, config: dict[str, Any]) -> None:
        self.session_id = str(session_id)
        self.sample_rate_hz = int(sample_rate_hz)
        self.channels = int(channels)

        self.ws_url = str(config.get("ws_url") or "ws://127.0.0.1:9090").strip()
        self.language = str(config.get("language") or "nl").strip()
        self.task = str(config.get("task") or "transcribe").strip()
        self.model = str(config.get("model") or "medium").strip()
        self.input_gain = _clamp_float(
            _safe_float(config.get("input_gain"), 1.0, min_value=0.1),
            min_value=0.1,
            max_value=MAX_INPUT_GAIN,
        )
        self.use_vad = _safe_bool(config.get("use_vad"), True)
        self.send_last_n_segments = _safe_int(config.get("send_last_n_segments"), 10, min_value=1)
        self.no_speech_thresh = _safe_float(config.get("no_speech_thresh"), 0.45, min_value=0.0)
        self.clip_audio = _safe_bool(config.get("clip_audio"), False)
        self.same_output_threshold = _safe_int(config.get("same_output_threshold"), 10, min_value=1)
        self.enable_translation = _safe_bool(config.get("enable_translation"), False)
        self.target_language = str(config.get("target_language") or "fr").strip()
        self.open_timeout_s = _safe_float(config.get("open_timeout_s"), 4.0, min_value=0.1)
        self.ready_timeout_s = _safe_float(config.get("ready_timeout_s"), 3.0, min_value=0.0)
        self.recv_timeout_s = _safe_float(config.get("recv_timeout_s"), 0.01, min_value=0.0)
        self.flush_timeout_s = _safe_float(config.get("flush_timeout_s"), 1.0, min_value=0.1)
        self.max_drain_messages = _safe_int(config.get("max_drain_messages"), 12, min_value=1)

        self._conn: Any | None = None
        self._ready = False
        self._audio_ms_total = 0
        self._rx_messages = 0
        self._rx_parse_errors = 0
        self._seen_finals: set[tuple[int, int, str]] = set()
        self._last_partial_sig: tuple[str, int, int] | None = None
        self._partials_emitted = 0
        self._finals_emitted = 0
        self._rx_unknown_messages = 0
        self._rx_unknown_keys: dict[str, int] = {}
        self._tx_sidecar_audio_bytes = 0

    @property
    def driver_name(self) -> str:
        return "whisperlive_sidecar"

    def _decode_packet(self, raw: Any) -> tuple[bool, list[EngineTranscriptEvent], str | None]:
        if raw is None:
            return False, [], None

        text_raw: str
        if isinstance(raw, bytes):
            try:
                text_raw = raw.decode("utf-8", errors="ignore")
            except Exception:
                self._rx_parse_errors += 1
                return False, [], None
        else:
            text_raw = str(raw)

        text_raw = text_raw.strip()
        if not text_raw:
            return False, [], None

        try:
            obj = json.loads(text_raw)
        except Exception:
            self._rx_parse_errors += 1
            return False, [], None
        if not isinstance(obj, dict):
            self._rx_parse_errors += 1
            return False, [], None

        status = str(obj.get("status") or "").strip().lower()
        if status in {"error", "failed"}:
            return False, [], str(obj.get("message") or obj.get("error") or "sidecar_error")

        message = str(obj.get("message") or "").strip()
        ready = message.upper() == "SERVER_READY"

        out: list[EngineTranscriptEvent] = []

        segments_any = obj.get("segments")
        if isinstance(segments_any, dict):
            segments_any = [segments_any]
        if segments_any is None:
            alt = obj.get("segment")
            if isinstance(alt, dict):
                segments_any = [alt]
            elif isinstance(alt, list):
                segments_any = alt
        if segments_any is None:
            alt_lines = obj.get("lines")
            if isinstance(alt_lines, list):
                segments_any = alt_lines
        if segments_any is None:
            data_obj = obj.get("data")
            if isinstance(data_obj, dict):
                ds = data_obj.get("segments")
                if isinstance(ds, dict):
                    segments_any = [ds]
                elif isinstance(ds, list):
                    segments_any = ds

        if isinstance(segments_any, list):
            last_partial: EngineTranscriptEvent | None = None
            for seg in segments_any:
                if not isinstance(seg, dict):
                    continue
                txt = str(seg.get("text") or "").strip()
                if not txt:
                    continue
                # Some servers emit seconds, others emit milliseconds.
                start_raw = seg.get("start")
                end_raw = seg.get("end")
                start_val = _safe_float(start_raw, self._audio_ms_total / 1000.0, min_value=0.0)
                end_val = _safe_float(end_raw, start_val + 0.2, min_value=start_val)
                if start_val > 1_000 or end_val > 1_000:
                    t0_ms = int(round(start_val))
                    t1_ms = int(round(end_val))
                else:
                    t0_ms = int(round(start_val * 1000.0))
                    t1_ms = int(round(end_val * 1000.0))
                if t1_ms <= t0_ms:
                    t1_ms = t0_ms + 1
                completed_raw = seg.get("completed", seg.get("final", seg.get("is_final", False)))
                if isinstance(completed_raw, str):
                    completed = completed_raw.strip().lower() in {"1", "true", "yes", "y"}
                else:
                    completed = bool(completed_raw)
                if completed:
                    key = (t0_ms, t1_ms, txt)
                    if key not in self._seen_finals:
                        self._seen_finals.add(key)
                        out.append(EngineTranscriptEvent(kind="final", text=txt, t0_ms=t0_ms, t1_ms=t1_ms))
                        self._finals_emitted += 1
                else:
                    last_partial = EngineTranscriptEvent(kind="partial", text=txt, t0_ms=t0_ms, t1_ms=t1_ms)
            if last_partial is not None:
                sig = (last_partial.text, last_partial.t0_ms, last_partial.t1_ms)
                if sig != self._last_partial_sig:
                    out.append(last_partial)
                    self._last_partial_sig = sig
                    self._partials_emitted += 1
            return ready, out, None

        txt = str(obj.get("text") or "").strip()
        if txt:
            t1_ms = int(max(1, self._audio_ms_total))
            t0_ms = int(max(0, t1_ms - 500))
            sig = (txt, t0_ms, t1_ms)
            if sig != self._last_partial_sig:
                out.append(EngineTranscriptEvent(kind="partial", text=txt, t0_ms=t0_ms, t1_ms=t1_ms))
                self._last_partial_sig = sig
                self._partials_emitted += 1

        if not out and not ready:
            self._rx_unknown_messages += 1
            for key in obj.keys():
                ks = str(key)
                self._rx_unknown_keys[ks] = int(self._rx_unknown_keys.get(ks, 0) + 1)

        return ready, out, None

    def _drain(self, *, timeout_s: float) -> list[EngineTranscriptEvent]:
        if self._conn is None:
            return []
        out: list[EngineTranscriptEvent] = []
        for _ in range(self.max_drain_messages):
            try:
                raw = self._conn.recv(timeout=timeout_s)
            except TimeoutError:
                break
            except Exception as e:
                raise RuntimeError(f"sidecar_receive_failed: {type(e).__name__}: {e}")
            self._rx_messages += 1
            ready, events, err = self._decode_packet(raw)
            if ready:
                self._ready = True
            if err:
                raise RuntimeError(err)
            out.extend(events)
        return out

    def start(self) -> None:
        if not self.ws_url:
            raise RuntimeError("Missing whisperlive_sidecar.ws_url")
        try:
            from websockets.sync.client import connect  # type: ignore
        except Exception as e:
            raise RuntimeError(f"websockets_dependency_missing: {type(e).__name__}: {e}")

        try:
            self._conn = connect(
                self.ws_url,
                open_timeout=self.open_timeout_s,
                close_timeout=1.0,
                max_size=None,
            )
        except Exception as e:
            raise RuntimeError(f"sidecar_connect_failed: {type(e).__name__}: {e}")

        hello = {
            "uid": self.session_id,
            "language": self.language,
            "task": self.task,
            "model": self.model,
            "use_vad": self.use_vad,
            "send_last_n_segments": self.send_last_n_segments,
            "no_speech_thresh": self.no_speech_thresh,
            "clip_audio": self.clip_audio,
            "same_output_threshold": self.same_output_threshold,
            "enable_translation": self.enable_translation,
            "target_language": self.target_language,
        }
        try:
            self._conn.send(json.dumps(hello), text=True)
        except Exception as e:
            self.close()
            raise RuntimeError(f"sidecar_handshake_failed: {type(e).__name__}: {e}")

        if self.ready_timeout_s <= 0:
            return

        deadline = time.monotonic() + self.ready_timeout_s
        while time.monotonic() < deadline and not self._ready:
            remaining = max(0.0, deadline - time.monotonic())
            timeout = min(0.25, remaining)
            _ = self._drain(timeout_s=timeout)

    def feed_audio(self, audio_bytes: bytes) -> list[EngineTranscriptEvent]:
        if self._conn is None:
            raise RuntimeError("sidecar_not_connected")
        size = len(audio_bytes or b"")
        if size <= 0:
            return []
        self._audio_ms_total += _bytes_to_ms(
            size,
            sample_rate_hz=self.sample_rate_hz,
            channels=self.channels,
        )
        # WhisperLive websocket server expects float32 PCM samples in binary frames.
        # Our live frontend contract uses PCM16LE mono, so we convert here.
        sidecar_audio = _pcm16le_to_float32le_bytes(
            audio_bytes,
            gain=self.input_gain,
        )
        if not sidecar_audio:
            return []
        try:
            self._conn.send(sidecar_audio, text=False)
            self._tx_sidecar_audio_bytes += len(sidecar_audio)
        except Exception as e:
            raise RuntimeError(f"sidecar_send_failed: {type(e).__name__}: {e}")
        # Guard against high recv_timeout_s values causing frame-by-frame backpressure.
        feed_timeout = min(float(self.recv_timeout_s), MAX_FEED_DRAIN_TIMEOUT_S)
        return self._drain(timeout_s=feed_timeout)

    def flush(self) -> list[EngineTranscriptEvent]:
        if self._conn is None:
            return []
        flush_errors: list[str] = []
        try:
            # Prefer binary control frame first; this matches the current sidecar
            # and avoids string/bytes type errors seen in some builds.
            self._conn.send(b"END_OF_AUDIO", text=False)
        except Exception as e:
            flush_errors.append(f"binary:{type(e).__name__}: {e}")
            try:
                # Fallback for sidecar variants that expect text control frame.
                self._conn.send("END_OF_AUDIO", text=True)
            except Exception as text_e:
                flush_errors.append(f"text:{type(text_e).__name__}: {text_e}")

        if flush_errors and len(flush_errors) >= 2:
            raise RuntimeError(f"sidecar_flush_signal_failed: {' | '.join(flush_errors)}")

        out: list[EngineTranscriptEvent] = []
        deadline = time.monotonic() + self.flush_timeout_s
        while time.monotonic() < deadline:
            timeout = min(0.2, max(0.0, deadline - time.monotonic()))
            try:
                batch = self._drain(timeout_s=timeout)
            except RuntimeError:
                break
            if not batch and timeout > 0:
                continue
            out.extend(batch)
        return out

    def close(self) -> None:
        if self._conn is None:
            return
        try:
            self._conn.close()
        except Exception:
            pass
        finally:
            self._conn = None

    def stats(self) -> dict[str, Any]:
        return {
            "engine_driver": self.driver_name,
            "engine_audio_ms": int(self._audio_ms_total),
            "engine_ready": bool(self._ready),
            "engine_rx_messages": int(self._rx_messages),
            "engine_rx_parse_errors": int(self._rx_parse_errors),
            "engine_rx_unknown_messages": int(self._rx_unknown_messages),
            "engine_rx_unknown_keys": dict(self._rx_unknown_keys),
            "engine_partials": int(self._partials_emitted),
            "engine_finals": int(self._finals_emitted),
            "engine_tx_sidecar_audio_bytes": int(self._tx_sidecar_audio_bytes),
        }


def create_live_engine_adapter(
    *,
    session_id: str,
    sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ,
    channels: int = DEFAULT_CHANNELS,
    driver_override: str | None = None,
) -> LiveEngineAdapter:
    live_cfg = load_live_engine_config()
    resolved_sr = _safe_int(live_cfg.get("sample_rate_hz"), sample_rate_hz, min_value=1)
    resolved_ch = _safe_int(live_cfg.get("channels"), channels, min_value=1)

    if driver_override is not None and str(driver_override).strip():
        driver = str(driver_override).strip().lower()
    else:
        driver = str(live_cfg.get("driver") or "mock_stream").strip().lower()
    if driver in {"mock", "mock_stream", "demo", "placeholder"}:
        return MockStreamEngine(
            sample_rate_hz=resolved_sr,
            channels=resolved_ch,
            config=live_cfg.get("mock_stream", {}) if isinstance(live_cfg.get("mock_stream"), dict) else {},
        )

    if driver in {"whisperlive", "whisperlive_sidecar"}:
        return WhisperLiveSidecarEngine(
            session_id=session_id,
            sample_rate_hz=resolved_sr,
            channels=resolved_ch,
            config=live_cfg.get("whisperlive_sidecar", {})
            if isinstance(live_cfg.get("whisperlive_sidecar"), dict)
            else {},
        )

    raise RuntimeError(f"Unsupported live engine driver: {driver}")


def _pcm16le_to_float32le_bytes(audio_bytes: bytes, *, gain: float = 1.0) -> bytes:
    raw = bytes(audio_bytes or b"")
    if not raw:
        return b""

    # Ensure 16-bit alignment.
    if len(raw) % 2 == 1:
        raw = raw[:-1]
    if not raw:
        return b""

    sample_count = len(raw) // 2
    samples = struct.unpack("<" + ("h" * sample_count), raw)
    g = _clamp_float(float(gain or 1.0), min_value=0.1, max_value=MAX_INPUT_GAIN)
    floats: list[float] = []
    for v in samples:
        x = (float(v) / 32768.0) * g
        if x > 1.0:
            x = 1.0
        elif x < -1.0:
            x = -1.0
        floats.append(x)
    return struct.pack("<" + ("f" * len(floats)), *floats)
