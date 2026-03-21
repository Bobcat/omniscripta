from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import request

from llm_queue_fs import TaskPaths, claim_next_task, finish_task

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from shared.app_config import get_float, get_int, get_setting, get_str


def _repo_root() -> Path:
  # llm-worker/llm_worker_daemon.py -> llm-worker -> repo root
  return Path(__file__).resolve().parents[1]


def _utc_iso() -> str:
  return datetime.now(timezone.utc).isoformat()


def _resolve_path(raw_path: str) -> Path:
  p = Path(str(raw_path or "").strip())
  if not str(p):
    raise RuntimeError("Missing path")
  return p if p.is_absolute() else (_repo_root() / p).resolve()


def _write_json_atomic(path: Path, obj: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_suffix(path.suffix + ".tmp")
  tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
  os.replace(tmp, path)


def _read_json(path: Path) -> dict[str, Any]:
  raw = json.loads(path.read_text(encoding="utf-8"))
  if isinstance(raw, dict):
    return raw
  return {}


def _append_log(log_path: Path, message: str) -> None:
  line = f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] {message}\n"
  with log_path.open("a", encoding="utf-8") as f:
    f.write(line)


def _update_task_status(
  task: TaskPaths,
  *,
  state: str | None = None,
  phase: str | None = None,
  progress: float | None = None,
  message: str | None = None,
  error: str | None = None,
  started_at: str | None = None,
  finished_at: str | None = None,
  extra: dict[str, Any] | None = None,
) -> None:
  data = {}
  try:
    data = _read_json(task.status_path)
  except Exception:
    data = {"task_id": task.task_id}
  if state is not None:
    data["state"] = str(state)
  if phase is not None:
    data["phase"] = str(phase)
  if progress is not None:
    data["progress"] = max(0.0, min(1.0, float(progress)))
  if message is not None:
    data["message"] = str(message)
  if error is not None:
    data["error"] = str(error)
  if started_at is not None:
    data["started_at"] = str(started_at)
  if finished_at is not None:
    data["finished_at"] = str(finished_at)
  if isinstance(extra, dict):
    data.update(extra)
  _write_json_atomic(task.status_path, data)


def _default_generation() -> dict[str, Any]:
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


def _tabby_url(base_url: str) -> str:
  return f"{str(base_url or '').rstrip('/')}/v1/chat/completions"


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


def _http_post_json(*, url: str, api_key: str, payload: dict[str, Any], timeout_s: int) -> dict[str, Any]:
  body = json.dumps(payload).encode("utf-8")
  req = request.Request(url, data=body, method="POST")
  req.add_header("Content-Type", "application/json")
  req.add_header("Authorization", f"Bearer {api_key}")
  with request.urlopen(req, timeout=max(1, int(timeout_s))) as resp:
    raw = resp.read()
  parsed = json.loads(raw.decode("utf-8", errors="replace"))
  if isinstance(parsed, dict):
    return parsed
  raise RuntimeError("Invalid JSON response: expected object")


def _extract_text(resp_json: dict[str, Any]) -> str:
  try:
    return str(resp_json["choices"][0]["message"]["content"])
  except Exception:
    return ""


def _task_spec(task: TaskPaths) -> tuple[str, dict[str, Any]]:
  task_obj = _read_json(task.task_path)
  task_kind = str(task_obj.get("task_kind") or "").strip().lower()
  spec = task_obj.get("spec")
  if not isinstance(spec, dict):
    spec = {}
  return task_kind, dict(spec)


def _load_prompt_text(spec: dict[str, Any]) -> str:
  prompt_text = str(spec.get("prompt_text") or "").strip()
  if prompt_text:
    return prompt_text
  prompt_path_raw = str(spec.get("prompt_path") or "").strip()
  if not prompt_path_raw:
    raise RuntimeError("prompt_run missing prompt_text or prompt_path")
  prompt_path = _resolve_path(prompt_path_raw)
  if not prompt_path.exists():
    raise RuntimeError(f"Prompt path not found: {prompt_path}")
  return prompt_path.read_text(encoding="utf-8", errors="replace").rstrip("\n")


def _load_input_artifact(spec: dict[str, Any]) -> tuple[Path, str]:
  input_raw = str(spec.get("input_artifact") or "").strip()
  if not input_raw:
    raise RuntimeError("prompt_run missing input_artifact")
  input_path = _resolve_path(input_raw)
  if not input_path.exists():
    raise RuntimeError(f"Input artifact not found: {input_path}")
  return input_path, input_path.read_text(encoding="utf-8", errors="replace").rstrip("\n")


def _resolve_output_dir(task: TaskPaths, spec: dict[str, Any]) -> Path:
  output_raw = str(spec.get("output_dir") or "").strip()
  if not output_raw:
    return task.output_dir.resolve()
  p = _resolve_path(output_raw)
  p.mkdir(parents=True, exist_ok=True)
  return p


def _build_payload(*, model: str, combined: str, generation: dict[str, Any]) -> dict[str, Any]:
  payload: dict[str, Any] = {
    "model": model,
    "messages": [{"role": "user", "content": combined}],
  }
  payload.update(generation)
  if "stream" not in payload:
    payload["stream"] = False
  return payload


def _run_prompt_task(*, task: TaskPaths, spec: dict[str, Any]) -> dict[str, Any]:
  model = str(spec.get("model") or "").strip()
  if not model:
    model = get_str("llm.default_model", "").strip()
  if not model:
    raise RuntimeError("prompt_run missing model and llm.default_model is not set")

  prompt_text = _load_prompt_text(spec)
  input_path, input_text = _load_input_artifact(spec)
  combined = _combined_prompt(prompt=prompt_text, src_name=input_path.name, src_text=input_text)

  generation = _default_generation()
  cfg_gen = get_setting("llm.generation", {})
  if isinstance(cfg_gen, dict):
    generation.update(cfg_gen)
  spec_gen = spec.get("generation")
  if isinstance(spec_gen, dict):
    generation.update(spec_gen)

  cfg_tabby = get_setting("llm.tabby", {})
  if not isinstance(cfg_tabby, dict):
    cfg_tabby = {}
  base_url = str(spec.get("base_url") or cfg_tabby.get("base_url") or "http://127.0.0.1:5001").strip()
  if not base_url:
    raise RuntimeError("Missing llm.tabby.base_url")
  api_key_env = str(spec.get("api_key_env") or cfg_tabby.get("api_key_env") or "TABBY_API_KEY").strip()
  api_key = os.getenv(api_key_env, "").strip()
  if not api_key:
    raise RuntimeError(f"Missing API key env var: {api_key_env}")

  timeout_s = int(spec.get("timeout_s") or cfg_tabby.get("timeout_s") or 600)
  retries = int(spec.get("retries") or cfg_tabby.get("retries") or 2)
  retry_sleep_s = float(spec.get("retry_sleep_s") or cfg_tabby.get("retry_sleep_s") or 2.0)
  url = _tabby_url(base_url)
  payload = _build_payload(model=model, combined=combined, generation=generation)

  out_dir = _resolve_output_dir(task, spec)
  base_name = str(spec.get("output_basename") or task.task_id).strip() or task.task_id
  payload_path = (out_dir / f"{base_name}_payload.json").resolve()
  resp_path = (out_dir / f"{base_name}_response.json").resolve()
  text_path = (out_dir / f"{base_name}_text.txt").resolve()
  result_path = (out_dir / f"{base_name}_result.json").resolve()

  _write_json_atomic(payload_path, payload)

  last_err = ""
  response_json: dict[str, Any] | None = None
  for attempt in range(max(0, retries) + 1):
    try:
      response_json = _http_post_json(url=url, api_key=api_key, payload=payload, timeout_s=timeout_s)
      break
    except Exception as e:
      last_err = f"{type(e).__name__}: {e}"
      if attempt < retries:
        time.sleep(max(0.0, retry_sleep_s))
  if response_json is None:
    raise RuntimeError(f"LLM call failed: {last_err or 'unknown error'}")

  _write_json_atomic(resp_path, response_json)
  out_text = _extract_text(response_json).rstrip("\n")
  text_path.write_text(out_text + ("\n" if out_text else ""), encoding="utf-8")

  result_obj = {
    "ok": True,
    "task_id": task.task_id,
    "task_kind": "prompt_run",
    "model": model,
    "input_artifact": str(input_path),
    "output_text_path": str(text_path),
    "response_json_path": str(resp_path),
    "payload_json_path": str(payload_path),
  }
  _write_json_atomic(result_path, result_obj)
  return {
    "result_json_path": str(result_path),
    "response_json_path": str(resp_path),
    "payload_json_path": str(payload_path),
    "output_text_path": str(text_path),
    "model": model,
  }


def _process_task(task: TaskPaths) -> None:
  started_at = _utc_iso()
  _update_task_status(
    task,
    state="running",
    phase="running",
    progress=0.0,
    message="Task claimed",
    started_at=started_at,
    error="",
  )
  _append_log(task.log_path, "task_claimed")

  task_kind, spec = _task_spec(task)
  _append_log(task.log_path, f"task_kind={task_kind or 'unknown'}")
  if task_kind != "prompt_run":
    raise RuntimeError(f"Unsupported task_kind: {task_kind!r} (expected 'prompt_run')")

  _update_task_status(task, phase="call_llm", progress=0.5, message="Calling LLM")
  result = _run_prompt_task(task=task, spec=spec)
  _update_task_status(
    task,
    state="done",
    phase="done",
    progress=1.0,
    message="Done",
    finished_at=_utc_iso(),
    extra={"result": result},
  )
  _append_log(task.log_path, "task_done")
  finish_task(task, ok=True)


def _process_task_safe(task: TaskPaths) -> None:
  try:
    _process_task(task)
  except Exception as e:
    err_text = f"{type(e).__name__}: {e}"
    _append_log(task.log_path, f"task_error error={err_text}")
    _update_task_status(
      task,
      state="error",
      phase="error",
      progress=1.0,
      message=f"LLM worker error: {err_text}",
      error=err_text,
      finished_at=_utc_iso(),
    )
    finish_task(task, ok=False)


def main() -> int:
  poll_interval_s = get_float("llm.worker.poll_interval_s", 0.5, min_value=0.1)
  idle_log_interval_s = get_int("llm.worker.idle_log_interval_s", 30, min_value=1)
  last_idle_log = 0.0
  print(f"llm_worker started poll_interval_s={poll_interval_s:.2f}", flush=True)

  try:
    while True:
      task = claim_next_task()
      if task is None:
        now = time.monotonic()
        if (now - last_idle_log) >= float(idle_log_interval_s):
          last_idle_log = now
          print("llm_worker idle", flush=True)
        time.sleep(float(poll_interval_s))
        continue
      _process_task_safe(task)
  except KeyboardInterrupt:
    print("llm_worker stopping (keyboard interrupt)", flush=True)
    return 0


if __name__ == "__main__":
  raise SystemExit(main())
