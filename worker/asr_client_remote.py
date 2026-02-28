from __future__ import annotations

import json
import os
import random
import re
import secrets
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

ASR_SCHEMA_VERSION = "asr_v1"
_BLOB_REF_PREFIX = "fs://"
_SAFE_TOKEN_RE = re.compile(r"[^a-zA-Z0-9._-]+")
_cleanup_lock = threading.Lock()
_last_cleanup_monotonic = 0.0


def _env_str(name: str, default: str) -> str:
  return str(os.getenv(name, default) or default).strip()


def _env_int(name: str, default: int, *, min_value: int = 0) -> int:
  try:
    return max(min_value, int(str(os.getenv(name, str(default))).strip() or str(default)))
  except Exception:
    return max(min_value, int(default))


def _env_float(name: str, default: float, *, min_value: float = 0.0) -> float:
  try:
    return max(float(min_value), float(str(os.getenv(name, str(default))).strip() or str(default)))
  except Exception:
    return max(float(min_value), float(default))


def _env_bool(name: str, default: bool) -> bool:
  raw = str(os.getenv(name, "") or "").strip().lower()
  if not raw:
    return bool(default)
  if raw in {"1", "true", "yes", "on", "y"}:
    return True
  if raw in {"0", "false", "no", "off", "n"}:
    return False
  return bool(default)


def _build_error_response(
  *,
  request: dict[str, Any] | None,
  code: str,
  message: str,
  retryable: bool = False,
  details: dict[str, Any] | None = None,
) -> dict[str, Any]:
  req = dict(request or {})
  return {
    "schema_version": ASR_SCHEMA_VERSION,
    "request_id": str(req.get("request_id") or ""),
    "ok": False,
    "profile_id": str(req.get("profile_id") or ""),
    "resolved_options": dict(req.get("resolved_options") or {}),
    "error": {
      "code": str(code),
      "message": str(message),
      "retryable": bool(retryable),
      "details": dict(details or {}),
    },
    "warnings": [],
  }


def _repo_root() -> Path:
  return Path(__file__).resolve().parents[1]


def _blob_root() -> Path:
  raw = _env_str("TRANSCRIBE_ASR_BLOB_ROOT", str((_repo_root() / "data" / "asr_blobs").resolve()))
  p = Path(raw).expanduser()
  if not p.is_absolute():
    p = (_repo_root() / p).resolve()
  p.mkdir(parents=True, exist_ok=True)
  return p.resolve()


def _safe_token(value: str, *, fallback: str = "blob") -> str:
  s = _SAFE_TOKEN_RE.sub("_", str(value or "").strip())
  s = s.strip("._-")
  return s or fallback


def _validate_rel_blob_path(rel: str) -> Path:
  text = str(rel or "").strip().replace("\\", "/")
  while text.startswith("/"):
    text = text[1:]
  p = Path(text)
  if not p.parts or any(part in {"", ".", ".."} for part in p.parts):
    raise RuntimeError(f"Invalid blob_ref path: {rel!r}")
  return p


def _upload_local_path_as_blob_ref(*, local_path: Path, request_id: str) -> tuple[str, dict[str, Any]]:
  src = Path(str(local_path)).expanduser().resolve()
  if not src.exists() or not src.is_file():
    raise RuntimeError(f"Blob upload source missing: {src}")
  root = _blob_root()
  day = datetime.now(timezone.utc).strftime("%Y%m%d")
  safe_req = _safe_token(str(request_id or ""), fallback="req")
  suffix = "".join(ch for ch in str(src.suffix or "") if ch.isalnum() or ch in {".", "_", "-"}).lower()[:16]
  blob_name = f"{safe_req}_{secrets.token_hex(8)}{suffix}"
  rel = _validate_rel_blob_path((Path(day) / blob_name).as_posix())
  dst = (root / rel).resolve()
  if not dst.is_relative_to(root):
    raise RuntimeError("Resolved blob target escapes blob root")
  dst.parent.mkdir(parents=True, exist_ok=True)
  shutil.copyfile(src, dst)
  try:
    os.utime(dst, None)
  except Exception:
    pass
  blob_ref = f"{_BLOB_REF_PREFIX}{rel.as_posix()}"
  return blob_ref, {
    "blob_ref": blob_ref,
    "blob_rel": rel.as_posix(),
    "blob_path": str(dst),
    "bytes": int(dst.stat().st_size),
  }


def _cleanup_blob_store_if_due() -> None:
  interval_s = _env_int("TRANSCRIBE_ASR_BLOB_CLEANUP_INTERVAL_S", 120, min_value=0)
  ttl_s = _env_int("TRANSCRIBE_ASR_BLOB_TTL_S", 3600, min_value=0)
  max_scan = _env_int("TRANSCRIBE_ASR_BLOB_CLEANUP_MAX_SCAN_FILES", 5000, min_value=1)
  if interval_s <= 0 or ttl_s <= 0:
    return
  now_mono = time.monotonic()
  global _last_cleanup_monotonic
  with _cleanup_lock:
    if (now_mono - float(_last_cleanup_monotonic)) < float(interval_s):
      return
    _last_cleanup_monotonic = now_mono

  root = _blob_root()
  cutoff_unix = time.time() - float(ttl_s)
  scanned = 0
  try:
    for p in root.rglob("*"):
      if scanned >= max_scan:
        break
      if not p.is_file():
        continue
      scanned += 1
      try:
        st = p.stat()
      except Exception:
        continue
      if float(st.st_mtime) < cutoff_unix:
        try:
          p.unlink(missing_ok=True)
        except Exception:
          pass
  except Exception:
    return


def _pool_base_url() -> str:
  raw = _env_str("TRANSCRIBE_ASR_POOL_BASE_URL", "http://127.0.0.1:8090")
  return raw.rstrip("/")


def _priority_timeout_s(priority: str) -> int:
  p = str(priority or "").strip().lower()
  if p == "interactive":
    return _env_int("TRANSCRIBE_ASR_REMOTE_TIMEOUT_INTERACTIVE_S", 60, min_value=1)
  if p == "background":
    return _env_int("TRANSCRIBE_ASR_REMOTE_TIMEOUT_BACKGROUND_S", 420, min_value=1)
  return _env_int("TRANSCRIBE_ASR_REMOTE_TIMEOUT_NORMAL_S", 180, min_value=1)


def _poll_interval_s() -> float:
  return _env_float("TRANSCRIBE_ASR_REMOTE_POLL_INTERVAL_S", 0.2, min_value=0.05)


def _http_timeout_s() -> float:
  return _env_float("TRANSCRIBE_ASR_REMOTE_HTTP_TIMEOUT_S", 10.0, min_value=1.0)


def _retry_attempts() -> int:
  return _env_int("TRANSCRIBE_ASR_REMOTE_RETRY_ATTEMPTS", 3, min_value=1)


def _retry_base_delay_s() -> float:
  return _env_float("TRANSCRIBE_ASR_REMOTE_RETRY_BASE_DELAY_S", 0.2, min_value=0.0)


def _retry_max_delay_s() -> float:
  return _env_float("TRANSCRIBE_ASR_REMOTE_RETRY_MAX_DELAY_S", 2.0, min_value=0.05)


def _retry_jitter_s() -> float:
  return _env_float("TRANSCRIBE_ASR_REMOTE_RETRY_JITTER_S", 0.1, min_value=0.0)


def _json_or_empty(raw: bytes) -> dict[str, Any]:
  if not raw:
    return {}
  try:
    parsed = json.loads(raw.decode("utf-8", errors="replace"))
  except Exception:
    return {}
  return dict(parsed) if isinstance(parsed, dict) else {}


def _http_json_once(
  *,
  method: str,
  url: str,
  token: str,
  timeout_s: float,
  payload: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
  data = None
  if payload is not None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
  req = urlrequest.Request(url, data=data, method=str(method).upper())
  req.add_header("Content-Type", "application/json")
  if token:
    req.add_header("X-ASR-Token", token)
  try:
    with urlrequest.urlopen(req, timeout=float(timeout_s)) as resp:
      return int(getattr(resp, "status", 200) or 200), _json_or_empty(resp.read())
  except urlerror.HTTPError as e:
    return int(getattr(e, "code", 500) or 500), _json_or_empty(e.read())


def _retryable_http_status(status_code: int) -> bool:
  code = int(status_code)
  return code == 429 or code >= 500


def _backoff_sleep_s(*, retry_index: int, base_s: float, max_s: float, jitter_s: float) -> float:
  idx = max(0, int(retry_index))
  expo = float(base_s) * (2 ** idx)
  bounded = min(float(max_s), max(0.0, float(expo)))
  if float(jitter_s) > 0.0:
    bounded += random.uniform(0.0, float(jitter_s))
  return max(0.0, float(bounded))


def _http_json_with_retry(
  *,
  method: str,
  url: str,
  token: str,
  timeout_s: float,
  payload: dict[str, Any] | None = None,
  attempts: int,
  backoff_base_s: float,
  backoff_max_s: float,
  jitter_s: float,
) -> tuple[int, dict[str, Any], int]:
  max_attempts = max(1, int(attempts))
  last_exc: Exception | None = None
  for attempt in range(1, max_attempts + 1):
    try:
      status_code, body = _http_json_once(
        method=method,
        url=url,
        token=token,
        timeout_s=timeout_s,
        payload=payload,
      )
    except Exception as e:
      last_exc = e
      if attempt >= max_attempts:
        raise
      sleep_s = _backoff_sleep_s(
        retry_index=(attempt - 1),
        base_s=backoff_base_s,
        max_s=backoff_max_s,
        jitter_s=jitter_s,
      )
      if sleep_s > 0.0:
        time.sleep(sleep_s)
      continue

    if _retryable_http_status(status_code) and attempt < max_attempts:
      sleep_s = _backoff_sleep_s(
        retry_index=(attempt - 1),
        base_s=backoff_base_s,
        max_s=backoff_max_s,
        jitter_s=jitter_s,
      )
      if sleep_s > 0.0:
        time.sleep(sleep_s)
      continue
    return int(status_code), dict(body or {}), int(attempt)

  if last_exc is not None:
    raise last_exc
  return 500, {}, int(max_attempts)


def _terminal_response_from_lifecycle(
  *,
  request_payload: dict[str, Any],
  lifecycle: dict[str, Any],
  pool_base_url: str,
) -> dict[str, Any] | None:
  state = str(lifecycle.get("state") or "").strip().lower()
  if state in {"queued", "running"}:
    return None

  if state == "completed":
    response = dict(lifecycle.get("response") or {})
    if response:
      runtime = dict(response.get("runtime") or {})
      runtime["transport"] = "remote"
      runtime["pool_base_url"] = pool_base_url
      response["runtime"] = runtime
      return response
    return _build_error_response(
      request=request_payload,
      code="ASR_REMOTE_MISSING_RESPONSE",
      message="ASR pool completed without response payload",
      retryable=True,
      details={"state": state},
    )

  if state == "failed":
    response = dict(lifecycle.get("response") or {})
    if response:
      runtime = dict(response.get("runtime") or {})
      runtime["transport"] = "remote"
      runtime["pool_base_url"] = pool_base_url
      response["runtime"] = runtime
      return response
    err = dict(lifecycle.get("error") or {})
    return _build_error_response(
      request=request_payload,
      code=str(err.get("code") or "ASR_REMOTE_FAILED"),
      message=str(err.get("message") or "ASR pool request failed"),
      retryable=bool(err.get("retryable", True)),
      details=dict(err.get("details") or {}),
    )

  if state in {"cancel_requested", "cancelled"}:
    return _build_error_response(
      request=request_payload,
      code="ASR_REMOTE_CANCELLED",
      message=f"ASR pool request {state}",
      retryable=True,
      details={"state": state},
    )

  return _build_error_response(
    request=request_payload,
    code="ASR_REMOTE_UNKNOWN_STATE",
    message=f"ASR pool returned unknown lifecycle state: {state or '<empty>'}",
    retryable=True,
    details={"state": state},
  )


def transcribe_with_remote_pool(
  *,
  request_payload: dict[str, Any],
  on_lifecycle_update: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
  req = dict(request_payload or {})
  request_id = str(req.get("request_id") or "").strip()
  priority = str(req.get("priority") or "normal").strip().lower() or "normal"
  pool_base_url = _pool_base_url()
  token = _env_str("TRANSCRIBE_ASR_POOL_TOKEN", "")
  timeout_s = _priority_timeout_s(priority)
  poll_interval_s = _poll_interval_s()
  http_timeout_s = _http_timeout_s()
  retry_attempts = _retry_attempts()
  retry_base_delay_s = _retry_base_delay_s()
  retry_max_delay_s = max(retry_base_delay_s, _retry_max_delay_s())
  retry_jitter_s = _retry_jitter_s()
  blob_meta: dict[str, Any] | None = None
  submit_attempts_used = 1
  status_attempts_total = 0
  status_http_calls = 0
  cancel_attempts_used = 0

  if _env_bool("TRANSCRIBE_ASR_REMOTE_BLOB_ENABLED", True):
    audio = dict(req.get("audio") or {})
    local_path = str(audio.get("local_path") or "").strip()
    if local_path:
      try:
        blob_ref, blob_info = _upload_local_path_as_blob_ref(
          local_path=Path(local_path),
          request_id=(request_id or f"req_{int(time.time() * 1000)}"),
        )
        audio.pop("local_path", None)
        audio["blob_ref"] = str(blob_ref)
        req["audio"] = audio
        blob_meta = dict(blob_info or {})
        _cleanup_blob_store_if_due()
      except Exception as e:
        return _build_error_response(
          request=req,
          code="ASR_REMOTE_BLOB_UPLOAD_FAILED",
          message=f"Failed to upload audio blob for remote ASR: {type(e).__name__}: {e}",
          retryable=True,
          details={"exc_type": type(e).__name__, "request_id": request_id, "pool_base_url": pool_base_url},
        )

  def _annotate_remote_runtime(response: dict[str, Any]) -> dict[str, Any]:
    out = dict(response or {})
    runtime = dict(out.get("runtime") or {})
    runtime["remote_retry_attempts_config"] = int(retry_attempts)
    runtime["remote_submit_attempts"] = int(submit_attempts_used)
    runtime["remote_status_attempts_total"] = int(status_attempts_total)
    runtime["remote_status_http_calls"] = int(status_http_calls)
    if cancel_attempts_used > 0:
      runtime["remote_cancel_attempts"] = int(cancel_attempts_used)
    out["runtime"] = runtime
    return out

  submit_url = urlparse.urljoin(pool_base_url + "/", "asr/v1/requests")
  try:
    status_code, submit_body, submit_attempts_used = _http_json_with_retry(
      method="POST",
      url=submit_url,
      token=token,
      timeout_s=http_timeout_s,
      payload=req,
      attempts=retry_attempts,
      backoff_base_s=retry_base_delay_s,
      backoff_max_s=retry_max_delay_s,
      jitter_s=retry_jitter_s,
    )
  except Exception as e:
    return _build_error_response(
      request=req,
      code="ASR_REMOTE_SUBMIT_IO_FAILURE",
      message=f"ASR pool submit I/O failed: {type(e).__name__}: {e}",
      retryable=True,
      details={
        "pool_base_url": pool_base_url,
        "request_id": request_id,
        "attempts": int(retry_attempts),
        "http_timeout_s": float(http_timeout_s),
        "exc_type": type(e).__name__,
      },
    )
  if status_code not in {200, 202}:
    return _build_error_response(
      request=req,
      code=str(submit_body.get("code") or "ASR_REMOTE_SUBMIT_FAILED"),
      message=str(submit_body.get("message") or f"ASR pool submit failed with HTTP {status_code}"),
      retryable=bool(submit_body.get("retryable", True)),
      details={
        "http_status": int(status_code),
        "pool_base_url": pool_base_url,
        "request_id": request_id,
        "submit_attempts": int(submit_attempts_used),
        **dict(submit_body.get("details") or {}),
      },
    )

  terminal = _terminal_response_from_lifecycle(
    request_payload=req,
    lifecycle=submit_body,
    pool_base_url=pool_base_url,
  )
  if terminal is not None:
    if blob_meta is not None and isinstance(terminal, dict):
      runtime = dict(terminal.get("runtime") or {})
      runtime["blob_ref_used"] = True
      runtime["blob_ref"] = str(blob_meta.get("blob_ref") or "")
      terminal["runtime"] = runtime
    return _annotate_remote_runtime(terminal)

  rid = str(submit_body.get("request_id") or request_id or "").strip()
  if not rid:
    return _build_error_response(
      request=req,
      code="ASR_REMOTE_MISSING_REQUEST_ID",
      message="ASR pool submit response missing request_id",
      retryable=True,
      details={"pool_base_url": pool_base_url},
    )
  if callable(on_lifecycle_update):
    try:
      on_lifecycle_update(dict(submit_body or {}))
    except Exception:
      pass

  get_url = urlparse.urljoin(pool_base_url + "/", f"asr/v1/requests/{rid}")
  cancel_url = urlparse.urljoin(pool_base_url + "/", f"asr/v1/requests/{rid}/cancel")
  deadline = time.monotonic() + float(timeout_s)
  last_state = "queued"
  while time.monotonic() < deadline:
    time.sleep(poll_interval_s)
    try:
      status_code, body, status_attempts_used = _http_json_with_retry(
        method="GET",
        url=get_url,
        token=token,
        timeout_s=http_timeout_s,
        payload=None,
        attempts=retry_attempts,
        backoff_base_s=retry_base_delay_s,
        backoff_max_s=retry_max_delay_s,
        jitter_s=retry_jitter_s,
      )
      status_attempts_total += int(status_attempts_used)
      status_http_calls += 1
    except Exception as e:
      return _build_error_response(
        request=req,
        code="ASR_REMOTE_STATUS_IO_FAILURE",
        message=f"ASR pool status I/O failed: {type(e).__name__}: {e}",
        retryable=True,
        details={
          "request_id": rid,
          "pool_base_url": pool_base_url,
          "attempts": int(retry_attempts),
          "http_timeout_s": float(http_timeout_s),
          "exc_type": type(e).__name__,
        },
      )
    if status_code != 200:
      return _build_error_response(
        request=req,
        code=str(body.get("code") or "ASR_REMOTE_STATUS_FAILED"),
        message=str(body.get("message") or f"ASR pool status failed with HTTP {status_code}"),
        retryable=bool(body.get("retryable", True)),
        details={
          "http_status": int(status_code),
          "request_id": rid,
          "pool_base_url": pool_base_url,
          "status_attempts": int(status_attempts_used),
          **dict(body.get("details") or {}),
        },
      )
    last_state = str(body.get("state") or last_state)
    if callable(on_lifecycle_update):
      try:
        on_lifecycle_update(dict(body or {}))
      except Exception:
        pass
    terminal = _terminal_response_from_lifecycle(
      request_payload=req,
      lifecycle=body,
      pool_base_url=pool_base_url,
    )
    if terminal is not None:
      if blob_meta is not None and isinstance(terminal, dict):
        runtime = dict(terminal.get("runtime") or {})
        runtime["blob_ref_used"] = True
        runtime["blob_ref"] = str(blob_meta.get("blob_ref") or "")
        terminal["runtime"] = runtime
      return _annotate_remote_runtime(terminal)

  try:
    _status_ignored, _body_ignored, cancel_attempts_used = _http_json_with_retry(
      method="POST",
      url=cancel_url,
      token=token,
      timeout_s=http_timeout_s,
      payload={},
      attempts=retry_attempts,
      backoff_base_s=retry_base_delay_s,
      backoff_max_s=retry_max_delay_s,
      jitter_s=retry_jitter_s,
    )
  except Exception:
    cancel_attempts_used = int(retry_attempts)
  return _build_error_response(
    request=req,
    code="ASR_REMOTE_TIMEOUT",
    message=f"Timed out waiting for ASR pool response ({timeout_s}s)",
    retryable=True,
    details={
      "request_id": rid,
      "priority": priority,
      "pool_base_url": pool_base_url,
      "last_state": last_state,
      "timeout_s": int(timeout_s),
      "submit_attempts": int(submit_attempts_used),
      "status_attempts_total": int(status_attempts_total),
      "status_http_calls": int(status_http_calls),
      "cancel_attempts": int(cancel_attempts_used),
    },
  )
