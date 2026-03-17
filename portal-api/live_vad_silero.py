from __future__ import annotations

import importlib
import logging
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


logger = logging.getLogger("live_vad_silero")


@dataclass(frozen=True)
class LiveSileroVadSettings:
    enabled: bool
    whisperx_venv: str | None
    threshold: float
    max_speech_duration_s: float
    min_speech_ms: int
    hangover_ms: int


def _candidate_site_packages_paths(venv_root: Path) -> list[Path]:
    major = int(sys.version_info.major)
    minor = int(sys.version_info.minor)
    preferred = venv_root / "lib" / f"python{major}.{minor}" / "site-packages"
    out: list[Path] = [preferred]
    py_root = venv_root / "lib"
    if py_root.exists():
        for p in sorted(py_root.glob("python*/site-packages")):
            if p != preferred:
                out.append(p)
    return out


def _append_whisperx_venv_site_packages(whisperx_venv: str | None) -> str:
    if not whisperx_venv:
        raise RuntimeError("live.rolling.vad.whisperx_venv is required when VAD is enabled")
    venv_root = Path(str(whisperx_venv)).expanduser().resolve()
    if not venv_root.exists():
        raise RuntimeError(f"whisperx_venv_not_found:{venv_root}")
    for candidate in _candidate_site_packages_paths(venv_root):
        if candidate.exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return candidate_str
    raise RuntimeError(f"site_packages_not_found_in_venv:{venv_root}")


class LiveSileroVadGate:
    _MODEL_LOCK = threading.Lock()
    _SHARED_TORCH: Any = None
    _SHARED_NP: Any = None
    _SHARED_VAD_MODEL: Any = None
    _SHARED_GET_SPEECH_TIMESTAMPS: Any = None
    _SHARED_SITE_PACKAGES_PATH: str = ""

    def __init__(self, *, settings: LiveSileroVadSettings, sample_rate_hz: int) -> None:
        self._settings = settings
        self._sample_rate_hz = int(sample_rate_hz)
        self._last_speech_mono = 0.0
        self._site_packages_path = ""
        self._torch: Any = None
        self._np: Any = None
        self._vad_model: Any = None
        self._get_speech_timestamps: Any = None
        self._checks = 0
        self._silence_checks = 0
        self._speech_checks = 0
        self._hangover_allows = 0

        if not bool(settings.enabled):
            raise RuntimeError("silero_vad_gate_disabled")
        if self._sample_rate_hz != 16000:
            raise RuntimeError(f"silero_requires_16000hz:got_{self._sample_rate_hz}")

        self._site_packages_path = _append_whisperx_venv_site_packages(settings.whisperx_venv)
        with LiveSileroVadGate._MODEL_LOCK:
            if LiveSileroVadGate._SHARED_VAD_MODEL is None:
                torch_mod = importlib.import_module("torch")
                np_mod = importlib.import_module("numpy")
                vad_model, vad_utils = torch_mod.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    model="silero_vad",
                    force_reload=False,
                    onnx=False,
                    trust_repo=True,
                )
                LiveSileroVadGate._SHARED_TORCH = torch_mod
                LiveSileroVadGate._SHARED_NP = np_mod
                LiveSileroVadGate._SHARED_VAD_MODEL = vad_model
                LiveSileroVadGate._SHARED_GET_SPEECH_TIMESTAMPS = vad_utils[0]
                LiveSileroVadGate._SHARED_SITE_PACKAGES_PATH = self._site_packages_path
        self._torch = LiveSileroVadGate._SHARED_TORCH
        self._np = LiveSileroVadGate._SHARED_NP
        self._vad_model = LiveSileroVadGate._SHARED_VAD_MODEL
        self._get_speech_timestamps = LiveSileroVadGate._SHARED_GET_SPEECH_TIMESTAMPS
        if not self._site_packages_path:
            self._site_packages_path = str(LiveSileroVadGate._SHARED_SITE_PACKAGES_PATH or "")
        logger.info(
            "live_vad_silero_ready threshold=%.3f min_speech_ms=%d hangover_ms=%d site_packages=%s",
            float(self._settings.threshold),
            int(self._settings.min_speech_ms),
            int(self._settings.hangover_ms),
            self._site_packages_path,
        )

    def should_enqueue_pcm16(
        self,
        pcm16le: bytes,
        *,
        now_mono: float | None = None,
        allow_hangover: bool = True,
    ) -> dict[str, Any]:
        now = float(now_mono if now_mono is not None else time.monotonic())
        self._checks += 1
        raw = bytes(pcm16le or b"")
        if not raw:
            self._silence_checks += 1
            return {
                "allow": False,
                "reason": "empty_audio",
                "speech_ms": 0,
                "segments_count": 0,
            }
        if (len(raw) % 2) != 0:
            raw = raw[: len(raw) - 1]
        if not raw:
            self._silence_checks += 1
            return {
                "allow": False,
                "reason": "empty_audio",
                "speech_ms": 0,
                "segments_count": 0,
            }

        pcm = self._np.frombuffer(raw, dtype=self._np.int16).astype(self._np.float32) / 32768.0
        timestamps = self._get_speech_timestamps(
            pcm,
            model=self._vad_model,
            sampling_rate=self._sample_rate_hz,
            threshold=float(max(0.0, min(1.0, float(self._settings.threshold)))),
            max_speech_duration_s=float(max(0.1, float(self._settings.max_speech_duration_s))),
        )
        speech_samples = 0
        for seg in (timestamps or []):
            start_i = int(max(0, int(seg.get("start") or 0)))
            end_i = int(max(start_i, int(seg.get("end") or start_i)))
            speech_samples += int(max(0, end_i - start_i))
        speech_ms = int((speech_samples * 1000) // max(1, self._sample_rate_hz))
        segments_count = int(len(timestamps or []))

        if speech_ms >= int(max(0, self._settings.min_speech_ms)):
            self._last_speech_mono = now
            self._speech_checks += 1
            return {
                "allow": True,
                "reason": "speech",
                "speech_ms": int(max(0, speech_ms)),
                "segments_count": segments_count,
            }

        if allow_hangover and self._last_speech_mono > 0.0 and int(self._settings.hangover_ms) > 0:
            elapsed_ms = int(max(0.0, now - self._last_speech_mono) * 1000.0)
            if elapsed_ms <= int(self._settings.hangover_ms):
                self._hangover_allows += 1
                return {
                    "allow": True,
                    "reason": "hangover",
                    "speech_ms": int(max(0, speech_ms)),
                    "segments_count": segments_count,
                    "elapsed_since_speech_ms": int(max(0, elapsed_ms)),
                    "hangover_ms": int(max(0, self._settings.hangover_ms)),
                }

        self._silence_checks += 1
        return {
            "allow": False,
            "reason": "silence",
            "speech_ms": int(max(0, speech_ms)),
            "segments_count": segments_count,
        }

    def config_payload(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "provider": "silero",
            "sample_rate_hz": int(self._sample_rate_hz),
            "threshold": float(max(0.0, min(1.0, float(self._settings.threshold)))),
            "max_speech_duration_s": float(max(0.1, float(self._settings.max_speech_duration_s))),
            "min_speech_ms": int(max(0, self._settings.min_speech_ms)),
            "hangover_ms": int(max(0, self._settings.hangover_ms)),
            "whisperx_venv": str(self._settings.whisperx_venv or ""),
            "site_packages": str(self._site_packages_path or ""),
        }

    def state_payload(self) -> dict[str, Any]:
        age_ms = None
        if self._last_speech_mono > 0.0:
            age_ms = int(max(0.0, time.monotonic() - self._last_speech_mono) * 1000.0)
        return {
            "checks": int(max(0, self._checks)),
            "speech_checks": int(max(0, self._speech_checks)),
            "silence_checks": int(max(0, self._silence_checks)),
            "hangover_allows": int(max(0, self._hangover_allows)),
            "last_speech_age_ms": age_ms,
        }
