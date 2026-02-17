from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Optional
from urllib import request

from worker_status_io import _write_status


def _read_text(p: Path) -> str:
  return p.read_text(encoding="utf-8", errors="replace").rstrip("\n")


def _combined_prompt(*, prompt: str, src_name: str, src_text: str) -> str:
  # Mirrors your manual curl testing format.
  src_text = (src_text or "").rstrip("\n")
  prompt = (prompt or "").rstrip("\n")
  return (
    f"{prompt}\n"
    f"ATTACHMENTS:\n"
    f"Name: {src_name}\n"
    f"Contents:\n"
    f"=====\n"
    f"{src_text}\n"
    f"=====\n"
  )


def _tabby_url(base_url: str) -> str:
  base_url = base_url.rstrip("/")
  return f"{base_url}/v1/chat/completions"


def _default_generation_params() -> dict[str, Any]:
  # Mirrors your posted parameters (safe defaults).
  return {
    "max_tokens": 2048,
    "temperature": 0.01,
    "top_p": 1,
    "top_k": 1,
    "typical": 1,
    "min_p": 0,
    "tfs": 1,
    "top_a": 0,
    "smoothing_factor": 0,
    "repetition_penalty": 1,
    "penalty_range": 1024,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "dry_multiplier": 0,
    "mirostat_mode": 0,
    "xtc_threshold": 0.1,
    "xtc_probability": 0,
    "stream": False,
  }


def _build_payload(*, model: str, combined: str, generation: dict[str, Any]) -> dict[str, Any]:
  payload = {
    "model": model,
    "messages": [{"role": "user", "content": combined}],
  }
  payload.update(generation)
  if "stream" not in payload:
    payload["stream"] = False
  return payload


def _http_post_json(*, url: str, api_key: str, payload: dict[str, Any], timeout_s: int) -> dict[str, Any]:
  data = json.dumps(payload).encode("utf-8")
  req = request.Request(url, data=data, method="POST")
  req.add_header("Content-Type", "application/json")
  req.add_header("Authorization", f"Bearer {api_key}")
  with request.urlopen(req, timeout=timeout_s) as resp:
    raw = resp.read()
    return json.loads(raw.decode("utf-8", errors="replace"))


def _extract_content(resp_json: dict[str, Any]) -> str:
  try:
    return str(resp_json["choices"][0]["message"]["content"])
  except Exception:
    return ""


def run_topics_llm(
  *,
  job,
  manifest_path: Path,
  orig_stem: str,
  prompt_id: str,
  service_cfg: dict[str, Any],
  on_progress: Optional[Callable[[str], None]] = None,
) -> None:
  """
  Phase 41: Call Tabby (PC1) per chunk and write artifacts.

  Required config (in config/service.json):
    "topics": {
      "prompt_path": "/path/to/simple_prompt5.txt",
      "model": "matatonic_Mistral-Small-24B-Instruct-2501-4.0bpw-exl2",
      "generation": { ... optional overrides ... }
    },
    "tabby": {
      "base_url": "http://PC1_OR_TUNNEL:5001",
      "api_key_env": "TABBY_API_KEY",
      "timeout_s": 600,
      "retries": 2,
      "retry_sleep_s": 2
    }

  Outputs (per chunk) in job.result_dir:
    <orig_stem>_<prompt_id>_chunk_0001_payload.json
    <orig_stem>_<prompt_id>_chunk_0001_resp.json
    <orig_stem>_<prompt_id>_chunk_0001_raw.txt
  """
  topics_cfg = service_cfg.get("topics", {}) if isinstance(service_cfg, dict) else {}
  tabby_cfg = service_cfg.get("tabby", {}) if isinstance(service_cfg, dict) else {}

  prompt_path = Path(str(topics_cfg.get("prompt_path", "")))
  if not prompt_path.is_absolute():
    # Resolve relative paths from repo root so configs can use e.g. "prompts/simple_prompt5.txt".
    prompt_path = (Path(__file__).resolve().parents[1] / prompt_path).resolve()
  if not prompt_path.exists():
    raise RuntimeError("topics.prompt_path missing or not found in service.json")

  model = str(topics_cfg.get("model", "")).strip()
  if not model:
    raise RuntimeError("topics.model missing in service.json")

  generation = _default_generation_params()
  user_gen = topics_cfg.get("generation")
  if isinstance(user_gen, dict):
    generation.update(user_gen)

  base_url = str(tabby_cfg.get("base_url", "")).strip()
  if not base_url:
    raise RuntimeError("tabby.base_url missing in service.json")

  api_key_env = str(tabby_cfg.get("api_key_env", "TABBY_API_KEY"))
  api_key = os.getenv(api_key_env, "").strip()
  if not api_key:
    raise RuntimeError(f"Missing API key env var: {api_key_env}")

  timeout_s = int(tabby_cfg.get("timeout_s", 600))
  retries = int(tabby_cfg.get("retries", 2))
  retry_sleep_s = float(tabby_cfg.get("retry_sleep_s", 2))

  prompt = _read_text(prompt_path)
  url = _tabby_url(base_url)

  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  chunks = manifest.get("chunks") or []
  total = max(1, len(chunks))

  msg = "Calling LLM…"
  if on_progress:
    try:
      on_progress(msg)
    except Exception:
      pass
  _write_status(job.status_path, phase="topics", subphase="call", message=msg, topics_prompt_id=prompt_id)

  for ch in chunks:
    chunk_idx = int(ch["index"])
    chunk_file = job.result_dir / ch["filename"]
    chunk_text = _read_text(chunk_file)

    combined = _combined_prompt(prompt=prompt, src_name=chunk_file.name, src_text=chunk_text)
    payload = _build_payload(model=model, combined=combined, generation=generation)

    payload_path = job.result_dir / f"{orig_stem}_{prompt_id}_chunk_{chunk_idx:04d}_payload.json"
    resp_path = job.result_dir / f"{orig_stem}_{prompt_id}_chunk_{chunk_idx:04d}_resp.json"
    raw_path = job.result_dir / f"{orig_stem}_{prompt_id}_chunk_{chunk_idx:04d}_raw.txt"

    payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    msg = f"LLM chunk {chunk_idx}/{total}…"
    if on_progress:
      try:
        on_progress(msg)
      except Exception:
        pass
    _write_status(job.status_path, phase="topics", subphase="call", message=msg)

    last_err: Optional[str] = None
    for attempt in range(retries + 1):
      try:
        resp_json = _http_post_json(url=url, api_key=api_key, payload=payload, timeout_s=timeout_s)
        resp_path.write_text(json.dumps(resp_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        content = _extract_content(resp_json).rstrip("\n")
        raw_path.write_text(content + ("\n" if content else ""), encoding="utf-8")
        last_err = None
        break
      except Exception as e:
        last_err = f"{type(e).__name__}: {e}"
        if attempt < retries:
          time.sleep(retry_sleep_s)

    if last_err:
      msg = f"LLM failed: {last_err}"
      if on_progress:
        try:
          on_progress(msg)
        except Exception:
          pass
      _write_status(job.status_path, phase="topics", subphase="call", message=msg)
      raise RuntimeError(f"Tabby call failed for chunk {chunk_idx}: {last_err}")

  msg = "LLM calls complete."
  if on_progress:
    try:
      on_progress(msg)
    except Exception:
      pass
  _write_status(job.status_path, phase="topics", subphase="call", message=msg)
