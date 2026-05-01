from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.portal_api import _bootstrap  # noqa: F401

from live.results.exports import live_pc_events_to_text
from live.session.manager import LiveSessionManager


class LiveSessionManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.manager = LiveSessionManager(
            default_ttl_seconds=900,
            preconnect_ttl_seconds=30,
            max_sessions=4,
            archive_ttl_seconds=3600,
            max_archives=8,
        )
        self.manager._stats_log_dir = Path(self._tmpdir.name) / "live_stats"

    def _create_session(self) -> str:
        created = self.manager.create_session()
        return str(created["session_id"])

    def test_record_live_commit_replaces_existing_row_and_preserves_optional_metadata(self) -> None:
        session_id = self._create_session()
        self.manager.update_live_state(
            session_id,
            recording_duration_ms=2400,
            finalization_state="recording",
        )
        self.manager.update_live_preview(
            session_id,
            text="preview text",
            preview_seq=1,
            audio_end_ms=900,
            append_to_existing=False,
        )
        self.manager.record_live_commit(
            session_id,
            chunk_index=0,
            t0_ms=0,
            t1_ms=1000,
            text="first commit",
            segments=[
                {
                    "segment_id": "seg-1",
                    "text": "first commit",
                    "t0_ms": 0,
                    "t1_ms": 1000,
                    "speaker": "SPEAKER_1",
                }
            ],
            state="ready",
            reason="rolling_context_commit",
            speech_frames=12,
            silence_frames_tail=3,
            chunk_duration_ms=1000,
        )
        self.manager.update_live_preview(
            session_id,
            text="second preview",
            preview_seq=2,
            audio_end_ms=1100,
            append_to_existing=False,
        )

        result = self.manager.record_live_commit(
            session_id,
            chunk_index=0,
            t0_ms=0,
            t1_ms=1100,
            text="updated commit",
            segments=[
                {
                    "segment_id": "seg-1b",
                    "text": "updated commit",
                    "t0_ms": 0,
                    "t1_ms": 1100,
                    "speaker": "SPEAKER_1",
                }
            ],
            state="ready",
            reason="",
            speech_frames=None,
            silence_frames_tail=None,
            chunk_duration_ms=None,
        )

        self.assertEqual(result["transcript_revision"], 2)
        self.assertEqual(result["chunks_total"], 1)
        self.assertEqual(result["chunks_done"], 1)
        self.assertEqual(result["chunks_failed"], 0)
        self.assertEqual(result["preview"]["text"], "")
        self.assertEqual(result["preview"]["preview_seq"], -1)
        self.assertEqual(result["final_segments_count"], 1)
        self.assertEqual(result["final_covered_ms"], 1100)

        row = result["chunk_results"][0]
        self.assertEqual(row["chunk_index"], 0)
        self.assertEqual(row["text"], "updated commit")
        self.assertEqual(row["reason"], "rolling_context_commit")
        self.assertEqual(row["speech_frames"], 12)
        self.assertEqual(row["silence_frames_tail"], 3)
        self.assertEqual(row["chunk_duration_ms"], 1000)

    def test_error_commit_keeps_preview_and_updates_failure_counts(self) -> None:
        session_id = self._create_session()
        self.manager.update_live_state(
            session_id,
            recording_duration_ms=1500,
            finalization_state="recording",
        )
        self.manager.update_live_preview(
            session_id,
            text="still previewing",
            preview_seq=7,
            audio_end_ms=1400,
            append_to_existing=False,
        )

        result = self.manager.record_live_commit(
            session_id,
            chunk_index=0,
            t0_ms=0,
            t1_ms=1400,
            text="",
            segments=[],
            state="error",
            error="boom",
            reason="rolling_context_error",
        )

        self.assertEqual(result["chunks_total"], 1)
        self.assertEqual(result["chunks_done"], 0)
        self.assertEqual(result["chunks_failed"], 1)
        self.assertEqual(result["preview"]["text"], "still previewing")
        self.assertEqual(result["preview"]["preview_seq"], 7)

    def test_retry_commit_reuses_row_and_preserves_existing_optional_fields(self) -> None:
        session_id = self._create_session()

        self.manager.record_live_commit(
            session_id,
            chunk_index=0,
            t0_ms=0,
            t1_ms=900,
            text="",
            segments=[],
            state="error",
            error="asr_failed",
            reason="rolling_context_error",
            speech_frames=7,
            chunk_duration_ms=900,
        )
        result = self.manager.record_live_commit(
            session_id,
            chunk_index=0,
            t0_ms=0,
            t1_ms=950,
            text="Hersteld",
            segments=[],
            state="ready",
            reason="",
            speech_frames=None,
            chunk_duration_ms=None,
        )

        self.assertEqual(result["chunks_done"], 1)
        self.assertEqual(result["chunks_failed"], 0)
        self.assertEqual(result["chunk_results_count"], 1)
        self.assertEqual(result["final_segments"][0]["text"], "Hersteld")
        self.assertEqual(result["chunk_results"][0]["reason"], "rolling_context_error")
        self.assertEqual(result["chunk_results"][0]["speech_frames"], 7)
        self.assertEqual(result["chunk_results"][0]["chunk_duration_ms"], 900)

    def test_live_pc_events_capture_preview_commit_and_clear_sequence(self) -> None:
        session_id = self._create_session()

        self.manager.update_live_preview(
            session_id,
            text="preview text",
            preview_seq=1,
            audio_end_ms=900,
            append_to_existing=False,
        )
        self.manager.record_live_commit(
            session_id,
            chunk_index=0,
            t0_ms=0,
            t1_ms=1000,
            text="committed line 1\ncommitted line 2",
            segments=[],
            state="ready",
            reason="rolling_context_commit",
        )

        pc_events = self.manager.live_pc_events(session_id)

        self.assertEqual(
            pc_events,
            [
                {"kind": "p", "text": "preview text"},
                {"kind": "c", "text": "committed line 1\ncommitted line 2"},
                {"kind": "p", "text": ""},
            ],
        )
        self.assertEqual(
            live_pc_events_to_text(pc_events),
            "p,preview text\nc,committed line 1 committed line 2\np,\n",
        )

    def test_archive_live_result_payload_preserves_active_result_contract(self) -> None:
        session_id = self._create_session()
        self.manager.update_live_state(
            session_id,
            recording_state="recording",
            recording_path="/tmp/test.wav",
            recording_duration_ms=2000,
            finalization_state="recording",
            asr_transcribe_s=1.25,
            asr_load_audio_s=0.75,
            asr_runner_wall_s=2.5,
            asr_pool_ingest_body_read_s=0.11,
            asr_pool_ingest_multipart_parse_s=0.22,
            asr_pool_ingest_audio_write_s=0.33,
            asr_pool_ingest_submit_enqueue_s=0.44,
        )
        self.manager.set_fixture_metadata(
            session_id,
            fixture_id="fixture-1",
            fixture_version="v2",
            fixture_test_mode="playback",
        )
        self.manager.set_asr_language(session_id, asr_language="nl")
        self.manager.set_live_engine_runtime(
            session_id,
            runtime={"vad": {"enabled": True, "config": {"hangover_ms": 600}, "state": {"checks": 7}}},
        )
        active_result = self.manager.record_live_commit(
            session_id,
            chunk_index=0,
            t0_ms=0,
            t1_ms=1200,
            text="archive me",
            segments=[
                {
                    "segment_id": "seg-archive",
                    "text": "archive me",
                    "t0_ms": 0,
                    "t1_ms": 1200,
                    "speaker": "",
                }
            ],
            state="ready",
            reason="rolling_context_commit",
        )

        self.manager.archive_transcript(
            session_id,
            close_reason="client_stop",
            final_segments=[{"segment_id": "legacy", "text": "legacy", "t0_ms": 0, "t1_ms": 100}],
            transcript_revision=1,
            recording_path="/tmp/test.wav",
            recording_bytes=0,
            recording_duration_ms=2000,
            chunks_total=1,
            chunks_done=1,
            chunks_failed=0,
            finalization_state="ready",
            batch_job_id="",
        )
        self.manager.close_session(session_id, reason="client_stop")

        archive_result = self.manager.live_result_payload(session_id)
        archive_pc_events = self.manager.live_pc_events(session_id)

        self.assertEqual(archive_result["source"], "archive")
        self.assertEqual(archive_result["preview"]["text"], "")
        self.assertEqual(archive_result["preview"]["preview_seq"], -1)
        self.assertEqual(archive_result["engine_runtime"]["engine_state"]["vad"]["state"]["checks"], 7)
        self.assertEqual(archive_result["engine_runtime"]["engine_state"]["vad"]["config"]["hangover_ms"], 600)
        self.assertEqual(archive_pc_events, [{"kind": "c", "text": "archive me"}])

        for key in (
            "live_engine",
            "recording_path",
            "recording_duration_ms",
            "chunks_total",
            "chunks_done",
            "chunks_failed",
            "chunk_results_count",
            "final_segments_count",
            "final_covered_ms",
            "fixture_id",
            "fixture_version",
            "fixture_test_mode",
            "asr_language",
            "asr_transcribe_s",
            "asr_load_audio_s",
            "asr_runner_wall_s",
            "asr_pool_ingest_body_read_s",
            "asr_pool_ingest_multipart_parse_s",
            "asr_pool_ingest_audio_write_s",
            "asr_pool_ingest_submit_enqueue_s",
            "pc_events_count",
        ):
            self.assertEqual(archive_result[key], active_result[key])

    def test_archive_transcript_inherits_session_metadata(self) -> None:
        session_id = self._create_session()
        self.manager.set_asr_language(session_id, asr_language="nl")
        self.manager.set_fixture_metadata(
            session_id,
            fixture_id="fixture-1",
            fixture_version="v2",
            fixture_test_mode="playback",
        )
        self.manager.record_live_commit(
            session_id,
            chunk_index=0,
            t0_ms=0,
            t1_ms=800,
            text="Hallo",
            segments=[],
            state="ready",
        )

        archive = self.manager.archive_transcript(
            session_id,
            close_reason="client_done",
            final_segments=[{"segment_id": "seg-1", "text": "Hallo", "t0_ms": 0, "t1_ms": 800, "speaker": ""}],
            transcript_revision=3,
            recording_path="/tmp/live.wav",
            recording_bytes=123,
            recording_duration_ms=800,
            chunks_total=1,
            chunks_done=1,
            chunks_failed=0,
            finalization_state="ready",
        )

        self.assertEqual(archive["live_engine"], "rolling_context")
        self.assertEqual(archive["fixture_id"], "fixture-1")
        self.assertEqual(archive["fixture_version"], "v2")
        self.assertEqual(archive["fixture_test_mode"], "playback")
        self.assertEqual(archive["asr_language"], "nl")
        self.assertEqual(archive["live_transcript_revision"], 1)
        self.assertEqual(archive["live_final_segments_count"], 1)
        self.assertEqual(archive["live_commit_results_count"], 1)

    def test_archive_preserves_live_pc_events_for_connected_session_past_ttl(self) -> None:
        session_id = self._create_session()
        self.manager.open_websocket(session_id)

        with self.manager._lock:
            self.manager._sessions[session_id].expires_unix = 1.0

        self.manager.record_live_commit(
            session_id,
            chunk_index=0,
            t0_ms=0,
            t1_ms=1000,
            text="long run commit",
            segments=[
                {
                    "segment_id": "seg-1",
                    "text": "long run commit",
                    "t0_ms": 0,
                    "t1_ms": 1000,
                    "speaker": "",
                }
            ],
            state="ready",
            reason="rolling_context_commit",
        )

        self.manager.archive_transcript(
            session_id,
            close_reason="client_stop",
            final_segments=[],
            transcript_revision=0,
            recording_path="/tmp/test.wav",
            recording_bytes=0,
            recording_duration_ms=1000,
            chunks_total=1,
            chunks_done=1,
            chunks_failed=0,
            finalization_state="ready",
            batch_job_id="",
        )
        self.manager.close_session(session_id, reason="client_stop")

        archive_result = self.manager.live_result_payload(session_id)
        archive_pc_events = self.manager.live_pc_events(session_id)

        self.assertEqual(archive_result["transcript_revision"], 1)
        self.assertEqual(archive_result["final_segments_count"], 1)
        self.assertEqual(archive_result["chunk_results_count"], 1)
        self.assertEqual(archive_result["pc_events_count"], 1)
        self.assertEqual(archive_pc_events, [{"kind": "c", "text": "long run commit"}])


if __name__ == "__main__":
    unittest.main()
