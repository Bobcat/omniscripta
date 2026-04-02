from __future__ import annotations

from pathlib import Path
from typing import Any

from live.session.sessions import LiveSessionManager
from shared.app_config import get_bool, get_float, get_int, get_setting, get_str

_REPO_ROOT = Path(__file__).resolve().parents[2]
LIVE_RECORDINGS_ROOT = (_REPO_ROOT / "data" / "live_recordings").resolve()
LIVE_BENCHMARK_EXPORT_ROOT = (_REPO_ROOT / "data" / "live_benchmark_exports").resolve()


def _get_optional_setting_str(path: str) -> str | None:
    raw = get_setting(path, None)
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def _get_optional_setting_int(path: str, *, min_value: int | None = None) -> int | None:
    raw = get_setting(path, None)
    if raw is None:
        return None
    try:
        value = int(raw)
    except Exception:
        return None
    if min_value is not None:
        value = int(max(int(min_value), value))
    return value


ROOT_PATH = get_str("service.root_path", "/api")
LIVE_ENGINE = "rolling_context"
LIVE_SESSION_TTL_S = get_int("live.session_ttl_s", 900, min_value=60)
LIVE_SESSION_PRECONNECT_TTL_S = get_int("live.session_preconnect_ttl_s", 30, min_value=5)
LIVE_MAX_SESSIONS = get_int("live.max_sessions", 1, min_value=1)
LIVE_ARCHIVE_TTL_S = get_int("live.archive_ttl_s", 3600, min_value=60)
LIVE_MAX_ARCHIVES = get_int("live.max_archives", 256, min_value=1)
LIVE_AUDIO_SAMPLE_RATE_HZ = get_int("live.audio_sample_rate_hz", 16000, min_value=8000)
LIVE_AUDIO_CHANNELS = get_int("live.audio_channels", 1, min_value=1)
LIVE_AUDIO_SAMPLE_WIDTH_BYTES = 2
LIVE_AUDIO_BYTES_PER_SECOND = int(max(1, LIVE_AUDIO_SAMPLE_RATE_HZ * LIVE_AUDIO_CHANNELS * LIVE_AUDIO_SAMPLE_WIDTH_BYTES))
LIVE_DRAIN_WAIT_S = get_float("live.drain_wait_s", 20.0, min_value=0.0)
LIVE_POST_CLOSE_WAIT_S = get_float("live.post_close_wait_s", 60.0, min_value=0.0)
LIVE_ASR_LANGUAGE = _get_optional_setting_str("live.asr_language")
LIVE_ASR_BEAM_SIZE = _get_optional_setting_int("live.asr_beam_size", min_value=1)
LIVE_ASR_CHUNK_SIZE = _get_optional_setting_int("live.asr_chunk_size", min_value=1)
LIVE_ASR_BACKEND = _get_optional_setting_str("live.asr_backend")
LIVE_DIARIZE_ENABLED = get_bool("live.diarize_enabled", False)
LIVE_DIARIZE_SPEAKER_MODE = get_str("live.diarize_speaker_mode", "fixed")
LIVE_DIARIZE_MIN_SPEAKERS = get_int("live.diarize_min_speakers", 1, min_value=1)
LIVE_DIARIZE_MAX_SPEAKERS = get_int("live.diarize_max_speakers", 4, min_value=1)
LIVE_ROLLING_POLL_INTERVAL_MS = get_int("polling_intervals.live_rolling_poll_ms", 250, min_value=20)
LIVE_ROLLING_MIN_INFER_AUDIO_MS = get_int("live.rolling.min_infer_audio_ms", 1000, min_value=200)
LIVE_ROLLING_SINGLE_COMMIT_MIN_MS = max(
    LIVE_ROLLING_MIN_INFER_AUDIO_MS,
    get_int("live.rolling.single_segment_commit_min_ms", 12000, min_value=1000),
)
LIVE_ROLLING_FORCE_COMMIT_REPEATS = get_int("live.rolling.force_commit_repeats", 8, min_value=1)
LIVE_ROLLING_MAX_UNCOMMITTED_MS = max(
    LIVE_ROLLING_MIN_INFER_AUDIO_MS,
    get_int("live.rolling.max_uncommitted_ms", 15000, min_value=1000),
)
LIVE_ROLLING_HARD_CLIP_KEEP_TAIL_MS = max(
    LIVE_ROLLING_MIN_INFER_AUDIO_MS,
    get_int("live.rolling.hard_clip_keep_tail_ms", 5000, min_value=1000),
)
LIVE_ROLLING_MAX_DECODE_WINDOW_MS = max(
    LIVE_ROLLING_MIN_INFER_AUDIO_MS,
    get_int("live.rolling.max_decode_window_ms", 12000, min_value=1000),
)
LIVE_ROLLING_BUFFER_TRIM_THRESHOLD_MS = max(
    LIVE_ROLLING_MAX_DECODE_WINDOW_MS,
    get_int("live.rolling.buffer_trim_threshold_ms", 30000, min_value=5000),
)
LIVE_ROLLING_BUFFER_TRIM_DROP_MS = max(
    LIVE_ROLLING_MIN_INFER_AUDIO_MS,
    get_int("live.rolling.buffer_trim_drop_ms", 20000, min_value=1000),
)
LIVE_ROLLING_MIN_NEW_AUDIO_MS = get_int("live.rolling.min_new_audio_ms", LIVE_ROLLING_MIN_INFER_AUDIO_MS, min_value=0)
LIVE_ROLLING_MIN_EMIT_INTERVAL_MS = get_int(
    "polling_intervals.live_rolling_emit_min_ms",
    LIVE_ROLLING_POLL_INTERVAL_MS,
    min_value=0,
)
LIVE_ROLLING_PACING_BASE_EMIT_MS = get_int("live.rolling.pacing.base_emit_ms", 500, min_value=1)
LIVE_ROLLING_PACING_STARTUP_DURATION_MS = get_int("live.rolling.pacing.startup.duration_ms", 0, min_value=0)
LIVE_ROLLING_PACING_STARTUP_EMIT_MS = get_int(
    "live.rolling.pacing.startup.emit_ms",
    LIVE_ROLLING_PACING_BASE_EMIT_MS,
    min_value=1,
)
LIVE_ROLLING_PACING_STARTUP_MIN_INFER_AUDIO_MS = get_int(
    "live.rolling.pacing.startup.min_infer_audio_ms",
    LIVE_ROLLING_MIN_INFER_AUDIO_MS,
    min_value=0,
)
LIVE_ROLLING_PACING_STARTUP_MIN_NEW_AUDIO_MS = get_int(
    "live.rolling.pacing.startup.min_new_audio_ms",
    LIVE_ROLLING_MIN_NEW_AUDIO_MS,
    min_value=0,
)
LIVE_ROLLING_VAD_ENABLED = get_bool("live.rolling.vad.enabled", False)
LIVE_ROLLING_VAD_WHISPERX_VENV = _get_optional_setting_str("live.rolling.vad.whisperx_venv")
LIVE_ROLLING_VAD_THRESHOLD = get_float("live.rolling.vad.threshold", 0.35, min_value=0.0)
LIVE_ROLLING_VAD_MAX_SPEECH_DURATION_S = get_float("live.rolling.vad.max_speech_duration_s", 12.0, min_value=0.1)
LIVE_ROLLING_VAD_MIN_SPEECH_MS = get_int("live.rolling.vad.min_speech_ms", 120, min_value=0)
LIVE_ROLLING_VAD_HANGOVER_MS = get_int("live.rolling.vad.hangover_ms", 600, min_value=0)
LIVE_ROLLING_SPEECH_GATE_SILENCE_ENTER_MS = get_int("live.rolling.speech_gate.silence_enter_ms", 900, min_value=100)
LIVE_ROLLING_SPEECH_GATE_REARM_HITS = get_int("live.rolling.speech_gate.rearm_hits", 2, min_value=1)
LIVE_ROLLING_SPEECH_GATE_REARM_WINDOW_MS = get_int("live.rolling.speech_gate.rearm_window_ms", 500, min_value=100)
LIVE_ROLLING_SPEECH_GATE_FORCE_COMMIT_SILENCE_MS = get_int(
    "live.rolling.speech_gate.force_commit_silence_ms",
    1500,
    min_value=100,
)

LIVE_SESSIONS = LiveSessionManager(
    default_ttl_seconds=LIVE_SESSION_TTL_S,
    preconnect_ttl_seconds=LIVE_SESSION_PRECONNECT_TTL_S,
    max_sessions=LIVE_MAX_SESSIONS,
    archive_ttl_seconds=LIVE_ARCHIVE_TTL_S,
    max_archives=LIVE_MAX_ARCHIVES,
)


def rooted_path(path: str) -> str:
    p = str(path or "").strip()
    if not p.startswith("/"):
        p = "/" + p
    rp = str(ROOT_PATH or "").rstrip("/")
    if rp in {"", "/"}:
        return p
    return rp + p


LIVE_ROLLING_CONTEXT_CONFIG_KEYS = (
    "LIVE_ENGINE",
    "LIVE_AUDIO_SAMPLE_RATE_HZ",
    "LIVE_AUDIO_CHANNELS",
    "LIVE_AUDIO_SAMPLE_WIDTH_BYTES",
    "LIVE_AUDIO_BYTES_PER_SECOND",
    "LIVE_DRAIN_WAIT_S",
    "LIVE_POST_CLOSE_WAIT_S",
    "LIVE_ASR_LANGUAGE",
    "LIVE_ASR_BEAM_SIZE",
    "LIVE_ASR_CHUNK_SIZE",
    "LIVE_ASR_BACKEND",
    "LIVE_DIARIZE_ENABLED",
    "LIVE_DIARIZE_SPEAKER_MODE",
    "LIVE_DIARIZE_MIN_SPEAKERS",
    "LIVE_DIARIZE_MAX_SPEAKERS",
    "LIVE_ROLLING_POLL_INTERVAL_MS",
    "LIVE_ROLLING_MIN_INFER_AUDIO_MS",
    "LIVE_ROLLING_SINGLE_COMMIT_MIN_MS",
    "LIVE_ROLLING_FORCE_COMMIT_REPEATS",
    "LIVE_ROLLING_MAX_UNCOMMITTED_MS",
    "LIVE_ROLLING_HARD_CLIP_KEEP_TAIL_MS",
    "LIVE_ROLLING_MAX_DECODE_WINDOW_MS",
    "LIVE_ROLLING_BUFFER_TRIM_THRESHOLD_MS",
    "LIVE_ROLLING_BUFFER_TRIM_DROP_MS",
    "LIVE_ROLLING_MIN_NEW_AUDIO_MS",
    "LIVE_ROLLING_MIN_EMIT_INTERVAL_MS",
    "LIVE_ROLLING_PACING_BASE_EMIT_MS",
    "LIVE_ROLLING_PACING_STARTUP_DURATION_MS",
    "LIVE_ROLLING_PACING_STARTUP_EMIT_MS",
    "LIVE_ROLLING_PACING_STARTUP_MIN_INFER_AUDIO_MS",
    "LIVE_ROLLING_PACING_STARTUP_MIN_NEW_AUDIO_MS",
    "LIVE_ROLLING_VAD_ENABLED",
    "LIVE_ROLLING_VAD_WHISPERX_VENV",
    "LIVE_ROLLING_VAD_THRESHOLD",
    "LIVE_ROLLING_VAD_MAX_SPEECH_DURATION_S",
    "LIVE_ROLLING_VAD_MIN_SPEECH_MS",
    "LIVE_ROLLING_VAD_HANGOVER_MS",
    "LIVE_ROLLING_SPEECH_GATE_SILENCE_ENTER_MS",
    "LIVE_ROLLING_SPEECH_GATE_REARM_HITS",
    "LIVE_ROLLING_SPEECH_GATE_REARM_WINDOW_MS",
    "LIVE_ROLLING_SPEECH_GATE_FORCE_COMMIT_SILENCE_MS",
)


def live_engine_rolling_context_config() -> dict[str, Any]:
    module_globals = globals()
    return {key: module_globals[key] for key in LIVE_ROLLING_CONTEXT_CONFIG_KEYS}
