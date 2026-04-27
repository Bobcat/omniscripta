from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib import request

from upload._util import _write_json_atomic


_SUPPORTED_DECODING_KEYS = {
    "beam_size",
    "top_k",
    "top_p",
    "temperature",
    "repetition_penalty",
    "max_tokens",
    "stop",
}


def _default_decoding() -> dict[str, Any]:
    return {
        "max_tokens": 2048,
        "temperature": 0.01,
        "top_p": 1,
        "top_k": 1,
        "repetition_penalty": 1,
    }


def _merge_decoding(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in _SUPPORTED_DECODING_KEYS:
            merged[key] = value
    return merged


def _combined_prompt(*, prompt: str, src_name: str, src_text: str) -> str:
    text = str(src_text or "").rstrip("\n")
    inst = str(prompt or "").rstrip("\n")
    return (
        f"{inst}\n"
        f"ATTACHMENTS:\n"
        f"Name: {src_name}\n"
        f"Contents:\n"
        f"=====\n"
        f"{text}\n"
        f"=====\n"
    )


def _http_post_json(*, url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    with request.urlopen(req, timeout=max(1.0, float(timeout_s))) as resp:
        raw = resp.read()
    parsed = json.loads(raw.decode("utf-8", errors="replace"))
    if isinstance(parsed, dict):
        return parsed
    raise RuntimeError("Invalid JSON response: expected object")


def _extract_output_text(resp_json: dict[str, Any]) -> str:
    return str(resp_json.get("output_text") or "").rstrip("\n")


def run_prompt_to_output_files(
    *,
    base_url: str,
    model: str,
    prompt_text: str,
    input_path: Path,
    output_dir: Path,
    output_basename: str,
    decoding: dict[str, Any] | None = None,
    timeout_s: float,
) -> Path:
    src_text = input_path.read_text(encoding="utf-8", errors="replace").rstrip("\n")
    combined = _combined_prompt(
        prompt=prompt_text,
        src_name=input_path.name,
        src_text=src_text,
    )
    payload = {
        "model": str(model),
        "input": combined,
        "instructions": " ",
        "stream": False,
        "decoding": _merge_decoding(_default_decoding(), dict(decoding or {})),
    }
    url = f"{str(base_url or '').rstrip('/')}/v1/responses"

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = str(output_basename or input_path.stem).strip() or input_path.stem
    payload_path = (output_dir / f"{base_name}_payload.json").resolve()
    response_path = (output_dir / f"{base_name}_response.json").resolve()
    text_path = (output_dir / f"{base_name}_text.txt").resolve()

    _write_json_atomic(payload_path, payload)
    response_json = _http_post_json(url=url, payload=payload, timeout_s=timeout_s)
    _write_json_atomic(response_path, response_json)

    output_text = _extract_output_text(response_json)
    text_path.write_text(output_text + ("\n" if output_text else ""), encoding="utf-8")
    return text_path
