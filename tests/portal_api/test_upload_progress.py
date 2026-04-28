from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests.portal_api import _bootstrap  # noqa: F401

from upload._util import _normalize_speaker_mode, _read_json, _safe_float, _write_json_atomic
from upload.pipeline.progress_plan import build_prediction


class UploadUtilTests(unittest.TestCase):
    def test_normalize_speaker_mode_aliases(self) -> None:
        self.assertEqual(_normalize_speaker_mode("off"), "none")
        self.assertEqual(_normalize_speaker_mode("disabled"), "none")
        self.assertEqual(_normalize_speaker_mode("fixed"), "fixed")
        self.assertEqual(_normalize_speaker_mode("AUTO"), "auto")

    def test_safe_float_matches_existing_contract(self) -> None:
        self.assertEqual(_safe_float("1.5"), 1.5)
        self.assertEqual(_safe_float(2), 2.0)
        self.assertIsNone(_safe_float("nan-value"))

    def test_read_json_returns_empty_dict_for_non_object_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "payload.json"
            path.write_text("[1, 2, 3]\n", encoding="utf-8")
            self.assertEqual(_read_json(path), {})

    def test_write_json_atomic_round_trips_object_with_trailing_newline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "payload.json"
            _write_json_atomic(path, {"alpha": 1, "beta": "two"})
            self.assertEqual(_read_json(path), {"alpha": 1, "beta": "two"})
            self.assertTrue(path.read_text(encoding="utf-8").endswith("\n"))


class ProgressPlanTests(unittest.TestCase):
    def test_build_prediction_uses_matching_done_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_path = Path(tmpdir) / "runs.jsonl"
            rows = [
                {
                    "outcome": "done",
                    "hardware_key": "dc1-rtx5070ti-cuda",
                    "speaker_mode": "auto",
                    "snippet_seconds": 900,
                    "phase_seconds": {
                        "snipping": 4.0,
                        "whisperx_prepare": 2.0,
                        "whisperx_transcribe": 10.0,
                        "whisperx_align": 6.0,
                        "whisperx_diarize": 8.0,
                        "topics_prep": 3.0,
                        "llm_topics": 7.0,
                    },
                },
                {
                    "outcome": "done",
                    "hardware_key": "dc1-rtx5070ti-cuda",
                    "speaker_mode": "auto",
                    "snippet_seconds": 900,
                    "phase_seconds": {
                        "snipping": 6.0,
                        "whisperx_prepare": 4.0,
                        "whisperx_transcribe": 14.0,
                        "whisperx_align": 10.0,
                        "whisperx_diarize": 12.0,
                        "topics_prep": 5.0,
                        "llm_topics": 9.0,
                    },
                },
                {
                    "outcome": "done",
                    "hardware_key": "other-host",
                    "speaker_mode": "auto",
                    "snippet_seconds": 900,
                    "phase_seconds": {"snipping": 100.0},
                },
            ]
            with runs_path.open("w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row) + "\n")

            prediction = build_prediction(
                runs_path=runs_path,
                hardware_key="dc1-rtx5070ti-cuda",
                topics_enabled=True,
                speaker_mode="auto",
                snippet_seconds=900,
            )

            self.assertEqual(
                prediction.phase_expected_s,
                {
                    "snipping": 5.0,
                    "whisperx_prepare": 3.0,
                    "whisperx_transcribe": 12.0,
                    "whisperx_align": 8.0,
                    "whisperx_diarize": 10.0,
                    "llm_topics": 12.0,
                },
            )
            self.assertEqual(prediction.total_expected_s, 50.0)
            self.assertEqual(prediction.confidence, 0.1)
            self.assertEqual(prediction.hints, ["low_sample_n"])
            self.assertEqual(prediction.sample_count, 2)


if __name__ == "__main__":
    unittest.main()
