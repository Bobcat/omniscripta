from __future__ import annotations

import asyncio
import contextlib
import sys
import types
import unittest
from unittest import mock
from pathlib import Path
import tempfile
import json
from types import SimpleNamespace

from tests.portal_api import _bootstrap  # noqa: F401


if "fastapi" not in sys.modules:
    fastapi_stub = types.ModuleType("fastapi")
    fastapi_stub.WebSocket = object
    fastapi_stub.WebSocketDisconnect = Exception
    fastapi_stub.status = types.SimpleNamespace(
        WS_1000_NORMAL_CLOSURE=1000,
        WS_1008_POLICY_VIOLATION=1008,
        WS_1011_INTERNAL_ERROR=1011,
    )
    sys.modules["fastapi"] = fastapi_stub

if "asr_pool_api" not in sys.modules:
    asr_pool_api_stub = types.ModuleType("asr_pool_api")

    class _Dummy:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    asr_pool_api_stub.ASRAudioFile = _Dummy
    asr_pool_api_stub.ASRCompletionEvent = _Dummy
    asr_pool_api_stub.ASRCompletionFeedReset = _Dummy
    asr_pool_api_stub.ASROutputSelection = _Dummy
    asr_pool_api_stub.ASRPoolClient = _Dummy
    asr_pool_api_stub.ASRPoolClientConfig = _Dummy
    asr_pool_api_stub.ASRPoolError = Exception
    asr_pool_api_stub.ASRRequestOptions = _Dummy
    asr_pool_api_stub.ASRRequestRouting = _Dummy
    asr_pool_api_stub.ASRSubmitRequest = _Dummy
    sys.modules["asr_pool_api"] = asr_pool_api_stub

if "app" not in sys.modules:
    app_stub = types.ModuleType("app")
    app_stub.__path__ = []  # type: ignore[attr-defined]
    sys.modules["app"] = app_stub

if "app.config" not in sys.modules:
    app_config_pkg = types.ModuleType("app.config")
    app_config_pkg.__path__ = []  # type: ignore[attr-defined]
    sys.modules["app.config"] = app_config_pkg

if "app.config.settings" not in sys.modules:
    app_config_stub = types.ModuleType("app.config.settings")

    def _get_setting(_key: str, default=None):
        return default

    def _get_bool(_key: str, default: bool = False, **_kwargs) -> bool:
        return bool(default)

    def _get_float(_key: str, default: float = 0.0, **_kwargs) -> float:
        return float(default)

    def _get_int(_key: str, default: int = 0, **_kwargs) -> int:
        return int(default)

    def _get_str(_key: str, default: str = "", **_kwargs) -> str:
        return str(default)

    app_config_stub.get_setting = _get_setting
    app_config_stub.get_bool = _get_bool
    app_config_stub.get_float = _get_float
    app_config_stub.get_int = _get_int
    app_config_stub.get_str = _get_str
    sys.modules["app.config.settings"] = app_config_stub

from live import config as live_config
from live.results.exports import build_live_result_envelope
from live.runtime import ws_session
from live.session.manager import LiveSessionManager


class WebSocketSessionTests(unittest.TestCase):
    def _session(
        self,
        *,
        websocket: object | None = None,
        live_sessions: object | None = None,
        rooted_path_cb=None,
        config: dict[str, object] | None = None,
    ) -> ws_session.LiveWebSocketSession:
        return ws_session.LiveWebSocketSession(
            "session-1",
            websocket if websocket is not None else object(),  # type: ignore[arg-type]
            live_sessions=(live_sessions if live_sessions is not None else object()),
            rooted_path_cb=(rooted_path_cb if rooted_path_cb is not None else (lambda path: f"/api{path}")),
            config=(config if config is not None else {"LIVE_ENGINE": "rolling_context"}),
        )

    def test_public_entrypoint_delegates_to_session_class(self) -> None:
        captured: dict[str, object] = {}

        async def fake_run(self: ws_session.LiveWebSocketSession) -> None:
            captured["session"] = self

        websocket = object()
        live_sessions = object()
        rooted_path_cb = lambda path: path
        config = {"LIVE_ENGINE": "rolling_context"}

        with mock.patch.object(ws_session.LiveWebSocketSession, "run", new=fake_run):
            asyncio.run(
                ws_session.run_live_session_ws(
                    "session-1",
                    websocket,  # type: ignore[arg-type]
                    live_sessions=live_sessions,
                    rooted_path_cb=rooted_path_cb,
                    config=config,
                )
            )

        session = captured["session"]
        self.assertIsInstance(session, ws_session.LiveWebSocketSession)
        self.assertEqual(session.session_id, "session-1")
        self.assertIs(session.websocket, websocket)
        self.assertIs(session.live_sessions, live_sessions)
        self.assertIs(session.rooted_path_cb, rooted_path_cb)
        self.assertIs(session.config, config)

    def test_send_event_adds_sequence_when_available(self) -> None:
        class FakeWebSocket:
            def __init__(self) -> None:
                self.payloads: list[dict[str, object]] = []

            async def send_json(self, payload: dict[str, object]) -> None:
                self.payloads.append(dict(payload))

        class FakeSessions:
            def next_seq(self, session_id: str) -> int:
                assert session_id == "session-1"
                return 42

        websocket = FakeWebSocket()
        session = self._session(websocket=websocket, live_sessions=FakeSessions())

        asyncio.run(session._send_event({"type": "ready"}))

        self.assertEqual(websocket.payloads, [{"type": "ready", "seq": 42}])

    def test_send_event_skips_sequence_when_session_missing(self) -> None:
        class FakeWebSocket:
            def __init__(self) -> None:
                self.payloads: list[dict[str, object]] = []

            async def send_json(self, payload: dict[str, object]) -> None:
                self.payloads.append(dict(payload))

        class MissingSessions:
            def next_seq(self, session_id: str) -> int:
                raise KeyError("session_not_found")

        websocket = FakeWebSocket()
        session = self._session(websocket=websocket, live_sessions=MissingSessions())

        asyncio.run(session._send_event({"type": "ready"}))

        self.assertEqual(websocket.payloads, [{"type": "ready"}])

    def test_result_envelope_marks_recording_finalized_not_ready(self) -> None:
        recordings_root = live_config.LIVE_RECORDINGS_ROOT
        recordings_root.mkdir(parents=True, exist_ok=True)
        wav_path = recordings_root / "test_result_envelope_marks_recording_finalized_not_ready.wav"
        try:
            wav_path.write_bytes(b"RIFF")
            session = self._session()

            envelope = build_live_result_envelope(
                session_id=session.session_id,
                result_payload={
                    "finalization_state": "recording_finalized",
                    "final_segments": [{"text": "hello", "t0_ms": 0, "t1_ms": 100}],
                    "recording_path": str(wav_path),
                    "pc_events_count": 1,
                },
                rooted_path_cb=session.rooted_path_cb,
            )
        finally:
            with contextlib.suppress(FileNotFoundError):
                wav_path.unlink()

        self.assertEqual(envelope["live_engine"], "rolling_context")
        self.assertFalse(envelope["ready"])
        self.assertTrue(envelope["can_export_srt"])
        self.assertTrue(envelope["can_export_wav"])
        self.assertTrue(envelope["can_export_pc"])
        self.assertEqual(envelope["transcript_srt_url"], "/api/demo/live/sessions/session-1/transcript.srt")
        self.assertEqual(envelope["recording_wav_url"], "/api/demo/live/sessions/session-1/recording.wav")
        self.assertEqual(envelope["transcript_pc_url"], "/api/demo/live/sessions/session-1/transcript.pc")

    def test_result_envelope_keeps_payload_live_engine_when_present(self) -> None:
        session = self._session()

        envelope = build_live_result_envelope(
            session_id=session.session_id,
            result_payload={
                "live_engine": "other_engine",
                "finalization_state": "recording_finalized",
                "final_segments": [],
                "recording_path": "/definitely/missing.wav",
                "pc_events_count": 0,
            },
            rooted_path_cb=session.rooted_path_cb,
        )

        self.assertEqual(envelope["live_engine"], "other_engine")
        self.assertFalse(envelope["ready"])
        self.assertFalse(envelope["can_export_srt"])
        self.assertFalse(envelope["can_export_wav"])
        self.assertFalse(envelope["can_export_pc"])
        self.assertIsNone(envelope["transcript_srt_url"])
        self.assertIsNone(envelope["recording_wav_url"])
        self.assertIsNone(envelope["transcript_pc_url"])

    def test_configure_context_builds_runner_and_audio_format(self) -> None:
        config = live_config.live_engine_rolling_context_config()
        session = self._session(config=config)

        session._configure_context()
        runner = session.rt.runner

        self.assertIsNotNone(runner)
        self.assertEqual(runner.audio_format.bytes_per_second, config["LIVE_AUDIO_BYTES_PER_SECOND"])
        self.assertEqual(session._ctx["LIVE_AUDIO_SAMPLE_RATE_HZ"], config["LIVE_AUDIO_SAMPLE_RATE_HZ"])
        self.assertEqual(runner.settings.pacing.min_emit_interval_ms, config["LIVE_ROLLING_MIN_EMIT_INTERVAL_MS"])

    def test_configure_context_normalizes_rolling_settings_via_package(self) -> None:
        config = dict(live_config.live_engine_rolling_context_config())
        config.update(
            {
                "LIVE_ROLLING_MIN_INFER_AUDIO_MS": 800,
                "LIVE_ROLLING_SINGLE_COMMIT_MIN_MS": 200,
                "LIVE_ROLLING_FORCE_COMMIT_REPEATS": 0,
                "LIVE_ROLLING_MAX_DECODE_WINDOW_MS": 500,
                "LIVE_ROLLING_MAX_UNCOMMITTED_MS": 500,
                "LIVE_ROLLING_HARD_CLIP_KEEP_TAIL_MS": 300,
                "LIVE_ROLLING_BUFFER_TRIM_THRESHOLD_MS": 400,
                "LIVE_ROLLING_BUFFER_TRIM_DROP_MS": 200,
                "LIVE_ROLLING_MIN_NEW_AUDIO_MS": -10,
                "LIVE_ROLLING_MIN_EMIT_INTERVAL_MS": -20,
                "LIVE_ROLLING_PACING_BASE_EMIT_MS": 0,
                "LIVE_ROLLING_PACING_STARTUP_EMIT_MS": 0,
                "LIVE_ROLLING_PACING_STARTUP_MIN_INFER_AUDIO_MS": -1,
                "LIVE_ROLLING_PACING_STARTUP_MIN_NEW_AUDIO_MS": -1,
                "LIVE_ROLLING_VAD_THRESHOLD": 5.0,
                "LIVE_ROLLING_VAD_MAX_SPEECH_DURATION_S": 0.0,
                "LIVE_ROLLING_VAD_MIN_SPEECH_MS": -1,
                "LIVE_ROLLING_VAD_HANGOVER_MS": -1,
                "LIVE_ROLLING_SPEECH_GATE_SILENCE_ENTER_MS": 0,
                "LIVE_ROLLING_SPEECH_GATE_REARM_HITS": 0,
                "LIVE_ROLLING_SPEECH_GATE_REARM_WINDOW_MS": 0,
                "LIVE_ROLLING_SPEECH_GATE_FORCE_COMMIT_SILENCE_MS": 0,
            }
        )
        session = self._session(config=config)

        ctx = session._configure_context()
        runner = session.rt.runner

        self.assertIsNotNone(runner)
        self.assertEqual(runner.settings.rolling.single_segment_commit_min_ms, 800)
        self.assertEqual(runner.settings.rolling.force_commit_repeats, 1)
        self.assertEqual(runner.settings.rolling.max_decode_window_ms, 800)
        self.assertEqual(runner.settings.rolling.max_uncommitted_ms, 1600)
        self.assertEqual(runner.settings.rolling.hard_clip_keep_tail_ms, 800)
        self.assertEqual(runner.settings.rolling.buffer_trim_threshold_ms, 800)
        self.assertEqual(runner.settings.rolling.buffer_trim_drop_ms, 800)
        self.assertEqual(runner.settings.rolling.min_new_audio_ms, 0)
        self.assertEqual(runner.settings.pacing.min_emit_interval_ms, 0)
        self.assertEqual(runner.settings.pacing.policy.base_emit_ms, 1)
        self.assertEqual(runner.settings.pacing.policy.startup_emit_ms, 1)
        self.assertEqual(runner.settings.pacing.policy.startup_min_infer_audio_ms, 0)
        self.assertEqual(runner.settings.pacing.policy.startup_min_new_audio_ms, 0)
        self.assertEqual(runner.settings.vad.threshold, 1.0)
        self.assertEqual(runner.settings.vad.max_speech_duration_s, 0.1)
        self.assertEqual(runner.settings.vad.min_speech_ms, 0)
        self.assertEqual(runner.settings.vad.hangover_ms, 0)
        self.assertEqual(runner.settings.speech_gate.silence_enter_ms, 100)
        self.assertEqual(runner.settings.speech_gate.rearm_hits, 1)
        self.assertEqual(runner.settings.speech_gate.rearm_window_ms, 100)
        self.assertEqual(runner.settings.speech_gate.force_commit_silence_ms, 100)
        self.assertEqual(runner.engine_runtime_payload()["vad"]["config"]["threshold"], 1.0)


class RollingContextLivePathTests(unittest.TestCase):
    def test_run_processes_audio_to_committed_archive_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_rate_hz = 16000
            channels = 1
            sample_width_bytes = 2
            bytes_per_second = sample_rate_hz * channels * sample_width_bytes

            class FakeWebSocket:
                def __init__(self) -> None:
                    self._messages = [
                        {"bytes": b"\x00\x00" * (bytes_per_second // 5)},
                        {"text": json.dumps({"type": "stop"})},
                    ]
                    self.sent_payloads: list[dict[str, object]] = []
                    self.closed = False
                    self.close_code: int | None = None
                    self.close_reason: str | None = None
                    self.accepted = False

                async def accept(self) -> None:
                    self.accepted = True

                async def send_json(self, payload: dict[str, object]) -> None:
                    self.sent_payloads.append(dict(payload))

                async def receive(self) -> dict[str, object]:
                    if self._messages:
                        return self._messages.pop(0)
                    return {"type": "websocket.disconnect"}

                async def close(self, *, code: int, reason: str | None = None) -> None:
                    self.closed = True
                    self.close_code = int(code)
                    self.close_reason = reason

            class FakeRecorder:
                root = Path(tmpdir)

                def __init__(self, *, session_id: str, sample_rate_hz: int, channels: int) -> None:
                    self.session_id = session_id
                    self.sample_rate_hz = int(sample_rate_hz)
                    self.channels = int(channels)
                    self.sample_width_bytes = sample_width_bytes
                    self.wav_path = self.root / f"{session_id}.wav"
                    self.bytes_written = 0
                    self.wav_path.write_bytes(b"RIFF")

                def _snapshot(self) -> SimpleNamespace:
                    duration_ms = int((self.bytes_written * 1000) / (self.sample_rate_hz * self.channels * self.sample_width_bytes))
                    return SimpleNamespace(
                        wav_path=self.wav_path,
                        bytes_written=self.bytes_written,
                        duration_ms=duration_ms,
                        to_dict=lambda: {
                            "wav_path": str(self.wav_path),
                            "bytes_written": int(self.bytes_written),
                            "duration_ms": int(duration_ms),
                        },
                    )

                def start(self) -> SimpleNamespace:
                    return self._snapshot()

                def append_pcm16(self, raw: bytes) -> SimpleNamespace:
                    self.bytes_written += len(raw or b"")
                    return self._snapshot()

                def finalize(self) -> SimpleNamespace:
                    return self._snapshot()

                def abort(self) -> None:
                    pass

            class FakeChunkBridge:
                def __init__(self, **_kwargs) -> None:
                    self._results: dict[str, SimpleNamespace] = {}
                    self._callback = None

                def start_completion_stream(self, *, session_id: str, on_terminal_event) -> None:
                    self._callback = on_terminal_event

                def stop_completion_stream(self) -> None:
                    self._callback = None

                def enqueue_chunk_pcm16(
                    self,
                    *,
                    session_id: str,
                    chunk_index: int,
                    t0_ms: int,
                    t1_ms: int,
                    pcm16le: bytes,
                    **_kwargs,
                ) -> SimpleNamespace:
                    job_id = f"job-{chunk_index}"
                    self._results[job_id] = SimpleNamespace(
                        done=True,
                        ok=True,
                        state="ready",
                        text="hello world",
                        segments=[
                            {
                                "segment_id": "seg-1",
                                "text": "hello world",
                                "t0_ms": int(t0_ms),
                                "t1_ms": int(t1_ms),
                                "speaker": "",
                            }
                        ],
                        error="",
                        status={},
                    )
                    if self._callback is not None:
                        self._callback()
                    return SimpleNamespace(job_id=job_id)

                def has_terminal_result(self, job_id: str) -> bool:
                    return job_id in self._results

                def take_terminal_result(self, job_id: str, *, t0_offset_ms: int) -> SimpleNamespace:
                    return self._results.pop(job_id)

            manager = LiveSessionManager(
                default_ttl_seconds=900,
                preconnect_ttl_seconds=30,
                max_sessions=4,
                archive_ttl_seconds=3600,
                max_archives=8,
            )
            manager._stats_log_dir = Path(tmpdir) / "live_stats"
            created = manager.create_session()
            session_id = str(created["session_id"])

            websocket = FakeWebSocket()
            config = dict(live_config.live_engine_rolling_context_config())
            config.update(
                {
                    "LIVE_DRAIN_WAIT_S": 0.0,
                    "LIVE_POST_CLOSE_WAIT_S": 0.0,
                    "LIVE_ROLLING_MIN_INFER_AUDIO_MS": 200,
                    "LIVE_ROLLING_SINGLE_COMMIT_MIN_MS": 200,
                    "LIVE_ROLLING_FORCE_COMMIT_REPEATS": 2,
                    "LIVE_ROLLING_MAX_UNCOMMITTED_MS": 1000,
                    "LIVE_ROLLING_HARD_CLIP_KEEP_TAIL_MS": 400,
                    "LIVE_ROLLING_MAX_DECODE_WINDOW_MS": 1000,
                    "LIVE_ROLLING_BUFFER_TRIM_THRESHOLD_MS": 1000,
                    "LIVE_ROLLING_BUFFER_TRIM_DROP_MS": 400,
                    "LIVE_ROLLING_MIN_NEW_AUDIO_MS": 0,
                    "LIVE_ROLLING_MIN_EMIT_INTERVAL_MS": 0,
                    "LIVE_ROLLING_PACING_BASE_EMIT_MS": 1,
                    "LIVE_ROLLING_PACING_STARTUP_DURATION_MS": 0,
                    "LIVE_ROLLING_PACING_STARTUP_EMIT_MS": 1,
                    "LIVE_ROLLING_PACING_STARTUP_MIN_INFER_AUDIO_MS": 0,
                    "LIVE_ROLLING_PACING_STARTUP_MIN_NEW_AUDIO_MS": 0,
                    "LIVE_ROLLING_VAD_ENABLED": False,
                }
            )

            with (
                mock.patch.object(ws_session, "LiveWavRecorder", FakeRecorder),
                mock.patch.object(ws_session, "LiveChunkBatchBridge", FakeChunkBridge),
            ):
                asyncio.run(
                    ws_session.run_live_session_ws(
                        session_id,
                        websocket,  # type: ignore[arg-type]
                        live_sessions=manager,
                        rooted_path_cb=lambda path: f"/api{path}",
                        config=config,
                    )
                )

            result = manager.live_result_payload(session_id)
            pc_events = manager.live_pc_events(session_id)

            self.assertTrue(websocket.accepted)
            self.assertTrue(websocket.closed)
            self.assertEqual(result["final_segments_count"], 1)
            self.assertEqual(result["chunks_done"], 1)
            self.assertEqual(result["chunks_failed"], 0)
            self.assertEqual(result["transcript_revision"], 1)
            self.assertEqual(result["final_segments"][0]["text"], "hello world")
            self.assertEqual(pc_events, [{"kind": "c", "text": "hello world"}])
            stats_payload = next(payload for payload in websocket.sent_payloads if payload.get("type") == "stats")
            self.assertIn("rolling_guardrails", stats_payload)
            self.assertIn("vad_checks", stats_payload["rolling_guardrails"])
            self.assertTrue(any(payload.get("type") == "ended" for payload in websocket.sent_payloads))
if __name__ == "__main__":
    unittest.main()
