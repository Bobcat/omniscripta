from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests.portal_api import _bootstrap  # noqa: F401

from upload.topics.validate import validate_all_chunks


class TopicsValidateTests(unittest.TestCase):
    def _write_json(self, path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _write_chunk(
        self,
        *,
        root: Path,
        orig_stem: str,
        prompt_id: str,
        rows: list[dict],
        finish_reason: str,
    ) -> None:
        parsed_path = root / f"{orig_stem}_{prompt_id}_chunk_0001.json"
        response_path = root / f"{orig_stem}_{prompt_id}_chunk_0001_response.json"
        self._write_json(parsed_path, {"rows": rows})
        self._write_json(response_path, {"choices": [{"finish_reason": finish_reason}]})

    def test_salvages_small_start_drift_when_model_stopped_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            orig_stem = "sample"
            prompt_id = "topics_v1"
            manifest_path = root / "manifest.json"
            report_path = root / "report.json"
            self._write_json(
                manifest_path,
                {
                    "chunks": [
                        {
                            "index": 1,
                            "chunk_start": "00:00:00",
                            "chunk_end": "00:15:00",
                        }
                    ]
                },
            )
            self._write_chunk(
                root=root,
                orig_stem=orig_stem,
                prompt_id=prompt_id,
                finish_reason="stop",
                rows=[
                    {
                        "n": 1,
                        "topic_title": "Health Update",
                        "topic_description": "Medical discussion.",
                        "start_time": "00:00:09",
                        "end_time": "00:15:00",
                        "raw_line": "1 | Health Update | Medical discussion. | 00:00:09 | 00:15:00",
                    }
                ],
            )

            validate_all_chunks(
                manifest_path=manifest_path,
                parsed_dir=root,
                orig_stem=orig_stem,
                prompt_id=prompt_id,
                out_report_path=report_path,
            )

            report = json.loads(report_path.read_text(encoding="utf-8"))
            parsed = json.loads((root / f"{orig_stem}_{prompt_id}_chunk_0001.json").read_text(encoding="utf-8"))
            self.assertTrue(report["is_valid"])
            self.assertEqual(report["salvaged_chunks"], 1)
            self.assertTrue(report["chunks"][0]["salvaged"])
            self.assertEqual(parsed["rows"][0]["start_time"], "00:00:00")

    def test_salvages_small_end_drift_when_model_stopped_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            orig_stem = "sample"
            prompt_id = "topics_v1"
            manifest_path = root / "manifest.json"
            report_path = root / "report.json"
            self._write_json(
                manifest_path,
                {
                    "chunks": [
                        {
                            "index": 1,
                            "chunk_start": "00:15:01",
                            "chunk_end": "00:30:01",
                        }
                    ]
                },
            )
            self._write_chunk(
                root=root,
                orig_stem=orig_stem,
                prompt_id=prompt_id,
                finish_reason="stop",
                rows=[
                    {
                        "n": 1,
                        "topic_title": "Vibe Coding Bugs",
                        "topic_description": "Discussion of debugging issues.",
                        "start_time": "00:15:01",
                        "end_time": "00:29:57",
                        "raw_line": "1 | Vibe Coding Bugs | Discussion of debugging issues. | 00:15:01 | 00:29:57",
                    }
                ],
            )

            validate_all_chunks(
                manifest_path=manifest_path,
                parsed_dir=root,
                orig_stem=orig_stem,
                prompt_id=prompt_id,
                out_report_path=report_path,
            )

            report = json.loads(report_path.read_text(encoding="utf-8"))
            parsed = json.loads((root / f"{orig_stem}_{prompt_id}_chunk_0001.json").read_text(encoding="utf-8"))
            self.assertTrue(report["is_valid"])
            self.assertEqual(report["salvaged_chunks"], 1)
            self.assertTrue(report["chunks"][0]["salvaged"])
            self.assertEqual(parsed["rows"][0]["end_time"], "00:30:01")

    def test_does_not_salvage_truncated_length_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            orig_stem = "sample"
            prompt_id = "topics_v1"
            manifest_path = root / "manifest.json"
            report_path = root / "report.json"
            self._write_json(
                manifest_path,
                {
                    "chunks": [
                        {
                            "index": 1,
                            "chunk_start": "00:15:03",
                            "chunk_end": "00:30:09",
                        }
                    ]
                },
            )
            self._write_chunk(
                root=root,
                orig_stem=orig_stem,
                prompt_id=prompt_id,
                finish_reason="length",
                rows=[
                    {
                        "n": 1,
                        "topic_title": "Personal Health Advice Continued",
                        "topic_description": "Discussion continues.",
                        "start_time": "00:15:03",
                        "end_time": "00:26:34",
                        "raw_line": "1 | Personal Health Advice Continued | Discussion continues. | 00:15:03 | 00:26:34",
                    }
                ],
            )

            validate_all_chunks(
                manifest_path=manifest_path,
                parsed_dir=root,
                orig_stem=orig_stem,
                prompt_id=prompt_id,
                out_report_path=report_path,
            )

            report = json.loads(report_path.read_text(encoding="utf-8"))
            parsed = json.loads((root / f"{orig_stem}_{prompt_id}_chunk_0001.json").read_text(encoding="utf-8"))
            self.assertFalse(report["is_valid"])
            self.assertEqual(report["salvaged_chunks"], 0)
            self.assertFalse(report["chunks"][0].get("salvaged", False))
            self.assertEqual(parsed["rows"][0]["end_time"], "00:26:34")


if __name__ == "__main__":
    unittest.main()
