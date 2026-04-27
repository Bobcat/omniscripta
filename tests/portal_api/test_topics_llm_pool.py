from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from tests.portal_api import _bootstrap  # noqa: F401

from upload.topics.llm_pool import run_prompt_to_output_files


class _LlmPoolHandler(BaseHTTPRequestHandler):
    last_request_json: dict[str, object] | None = None

    def do_POST(self) -> None:  # noqa: N802
        content_length = int(self.headers.get("Content-Length") or "0")
        raw_body = self.rfile.read(content_length)
        type(self).last_request_json = json.loads(raw_body.decode("utf-8"))
        body = json.dumps({"output_text": "topic summary"}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        del format, args


class TopicsLlmPoolTests(unittest.TestCase):
    def test_run_prompt_to_output_files_posts_to_llm_pool_and_writes_artifacts(self) -> None:
        _LlmPoolHandler.last_request_json = None
        server = ThreadingHTTPServer(("127.0.0.1", 0), _LlmPoolHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                input_path = tmp_path / "chunk.txt"
                output_dir = tmp_path / "result"
                input_path.write_text("Hello world\n", encoding="utf-8")

                text_path = run_prompt_to_output_files(
                    base_url=f"http://127.0.0.1:{server.server_port}",
                    model="demo-model",
                    prompt_text="Summarize this transcript",
                    input_path=input_path,
                    output_dir=output_dir,
                    output_basename="chunk_0001",
                    decoding={"max_tokens": 12},
                    timeout_s=3.0,
                )

                self.assertEqual(text_path, (output_dir / "chunk_0001_text.txt").resolve())
                self.assertEqual(text_path.read_text(encoding="utf-8"), "topic summary\n")

                payload = json.loads((output_dir / "chunk_0001_payload.json").read_text(encoding="utf-8"))
                response = json.loads((output_dir / "chunk_0001_response.json").read_text(encoding="utf-8"))

                self.assertEqual(payload["model"], "demo-model")
                self.assertFalse(payload["stream"])
                self.assertEqual(payload["decoding"]["max_tokens"], 12)
                self.assertEqual(payload["decoding"]["top_k"], 1)
                self.assertIn("ATTACHMENTS:", payload["input"])
                self.assertEqual(response["output_text"], "topic summary")
                self.assertEqual(_LlmPoolHandler.last_request_json, payload)
        finally:
            server.shutdown()
            server.server_close()
