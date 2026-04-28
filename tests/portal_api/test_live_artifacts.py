from __future__ import annotations

import unittest
from unittest.mock import patch

from tests.portal_api import _bootstrap  # noqa: F401

from live.results.exports import build_live_result_envelope


class LiveArtifactsTests(unittest.TestCase):
    def test_build_live_result_envelope_uses_recording_guard_for_wav_export(self) -> None:
        result = {
            "finalization_state": "ready",
            "live_engine": "rolling_context",
            "final_segments": [{"text": "Hallo", "t0_ms": 0, "t1_ms": 1000}],
            "pc_events_count": 1,
            "recording_path": "/tmp/outside-root.wav",
        }

        with patch("live.results.exports.live_recording_wav_path_from_result", return_value=None):
            envelope = build_live_result_envelope(
                session_id="live_123",
                result_payload=result,
                rooted_path_cb=lambda path: f"/api{path}",
            )

        self.assertTrue(envelope["ready"])
        self.assertTrue(envelope["can_export_srt"])
        self.assertFalse(envelope["can_export_wav"])
        self.assertTrue(envelope["can_export_pc"])
        self.assertEqual(envelope["transcript_srt_url"], "/api/demo/live/sessions/live_123/transcript.srt")
        self.assertIsNone(envelope["recording_wav_url"])
        self.assertEqual(envelope["transcript_pc_url"], "/api/demo/live/sessions/live_123/transcript.pc")

    def test_build_live_result_envelope_keeps_recording_finalized_not_ready(self) -> None:
        result = {
            "finalization_state": "recording_finalized",
            "live_engine": "rolling_context",
            "final_segments": [],
            "pc_events_count": 0,
            "recording_path": "",
        }

        envelope = build_live_result_envelope(
            session_id="live_456",
            result_payload=result,
            rooted_path_cb=lambda path: path,
        )

        self.assertFalse(envelope["ready"])
        self.assertFalse(envelope["can_export_srt"])
        self.assertFalse(envelope["can_export_wav"])
        self.assertFalse(envelope["can_export_pc"])


if __name__ == "__main__":
    unittest.main()
