from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from shared.asr.blob_store import cleanup_blob_store_if_due, upload_local_path_as_blob_ref
from shared.asr.schema import ASR_SCHEMA_VERSION
from shared.app_config import get_str, get_int, get_float, get_bool


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


def _pool_base_url() -> str:
  raw = get_str("asr_pool.base_url", "http://127.0.0.1:8090")
  return raw.rstrip("/")


def _http_timeout_s() -> float:
  return get_float("asr_remote.http_timeout_s", 10.0, min_value=1.0)


def _retry_attempts() -> int:
  return get_int("asr_remote.retry_attempts", 3, min_value=1)


def _retry_base_delay_s() -> float:
  return get_float("asr_remote.retry_base_delay_s", 0.2, min_value=0.0)


def _retry_max_delay_s() -> float:
  return get_float("asr_remote.retry_max_delay_s", 2.0, min_value=0.05)


def _retry_jitter_s() -> float:
  return get_float("asr_remote.retry_jitter_s", 0.1, min_value=0.0)


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


def _with_consumer_id(request_payload: dict[str, Any], *, consumer_id: str) -> dict[str, Any]:
  req = dict(request_payload or {})
  ctx = dict(req.get("context") or {})
  cid = str(consumer_id or "").strip()
  if cid:
    ctx["consumer_id"] = cid
  req["context"] = ctx
  return req


def _prepare_submit_payload(
  *,
  request_payload: dict[str, Any],
  consumer_id: str,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
  req = _with_consumer_id(dict(request_payload or {}), consumer_id=consumer_id)
  request_id = str(req.get("request_id") or "").strip()
  pool_base_url = _pool_base_url()
  blob_meta: dict[str, Any] | None = None
  if not get_bool("asr_remote.blob_enabled", True):
    return req, blob_meta, None
  audio = dict(req.get("audio") or {})
  local_path = str(audio.get("local_path") or "").strip()
  if not local_path:
    return req, blob_meta, None
  try:
    blob_ref, blob_info = upload_local_path_as_blob_ref(
      local_path=Path(local_path),
      request_id=(request_id or f"req_{int(time.time() * 1000)}"),
    )
    audio.pop("local_path", None)
    audio["blob_ref"] = str(blob_ref)
    req["audio"] = audio
    blob_meta = dict(blob_info or {})
    cleanup_blob_store_if_due()
    return req, blob_meta, None
  except Exception as e:
    return req, None, _build_error_response(
      request=req,
      code="ASR_REMOTE_BLOB_UPLOAD_FAILED",
      message=f"Failed to upload audio blob for remote ASR: {type(e).__name__}: {e}",
      retryable=True,
      details={"exc_type": type(e).__name__, "request_id": request_id, "pool_base_url": pool_base_url},
    )


def submit_remote_pool_request(
  *,
  request_payload: dict[str, Any],
  consumer_id: str,
) -> dict[str, Any]:
  req, _blob_meta, prep_error = _prepare_submit_payload(
    request_payload=request_payload,
    consumer_id=consumer_id,
  )
  if prep_error is not None:
    return {
      "ok": False,
      "request_id": str(req.get("request_id") or ""),
      "prepared_request": req,
      "error_response": prep_error,
      "submit_lifecycle": {},
      "http_status": 0,
    }

  pool_base_url = _pool_base_url()
  token = get_str("asr_pool.token", "")
  http_timeout_s = _http_timeout_s()
  retry_attempts = _retry_attempts()
  retry_base_delay_s = _retry_base_delay_s()
  retry_max_delay_s = max(retry_base_delay_s, _retry_max_delay_s())
  retry_jitter_s = _retry_jitter_s()
  request_id = str(req.get("request_id") or "").strip()
  submit_url = urlparse.urljoin(pool_base_url + "/", "asr/v1/requests")
  try:
    status_code, submit_body, attempts_used = _http_json_with_retry(
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
    return {
      "ok": False,
      "request_id": str(request_id),
      "prepared_request": req,
      "error_response": _build_error_response(
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
      ),
      "submit_lifecycle": {},
      "http_status": 0,
    }
  if status_code not in {200, 202}:
    return {
      "ok": False,
      "request_id": str(request_id),
      "prepared_request": req,
      "error_response": _build_error_response(
        request=req,
        code=str(submit_body.get("code") or "ASR_REMOTE_SUBMIT_FAILED"),
        message=str(submit_body.get("message") or f"ASR pool submit failed with HTTP {status_code}"),
        retryable=bool(submit_body.get("retryable", True)),
        details={
          "http_status": int(status_code),
          "pool_base_url": pool_base_url,
          "request_id": request_id,
          "submit_attempts": int(attempts_used),
          **dict(submit_body.get("details") or {}),
        },
      ),
      "submit_lifecycle": dict(submit_body or {}),
      "http_status": int(status_code),
    }

  rid = str(submit_body.get("request_id") or request_id or "").strip()
  return {
    "ok": True,
    "request_id": rid,
    "prepared_request": req,
    "submit_lifecycle": dict(submit_body or {}),
    "http_status": int(status_code),
  }


def fetch_remote_completions(
  *,
  consumer_id: str,
  since_seq: int,
  limit: int = 100,
) -> dict[str, Any]:
  pool_base_url = _pool_base_url()
  token = get_str("asr_pool.token", "")
  http_timeout_s = _http_timeout_s()
  retry_attempts = _retry_attempts()
  retry_base_delay_s = _retry_base_delay_s()
  retry_max_delay_s = max(retry_base_delay_s, _retry_max_delay_s())
  retry_jitter_s = _retry_jitter_s()
  query = urlparse.urlencode(
    {
      "consumer_id": str(consumer_id or ""),
      "since_seq": int(max(0, int(since_seq))),
      "limit": int(max(1, min(1000, int(limit)))),
    }
  )
  url = urlparse.urljoin(pool_base_url + "/", f"asr/v1/completions?{query}")
  try:
    status_code, body, _attempts_used = _http_json_with_retry(
      method="GET",
      url=url,
      token=token,
      timeout_s=http_timeout_s,
      payload=None,
      attempts=retry_attempts,
      backoff_base_s=retry_base_delay_s,
      backoff_max_s=retry_max_delay_s,
      jitter_s=retry_jitter_s,
    )
  except Exception as e:
    return {
      "ok": False,
      "status_code": 0,
      "body": {
        "code": "ASR_REMOTE_COMPLETIONS_IO_FAILURE",
        "message": f"ASR pool completions I/O failed: {type(e).__name__}: {e}",
        "retryable": True,
      },
    }
  return {
    "ok": bool(int(status_code) == 200),
    "status_code": int(status_code),
    "body": dict(body or {}),
  }


def fetch_remote_pending_status(
  *,
  consumer_id: str,
  request_ids: list[str],
  limit: int = 200,
) -> dict[str, Any]:
  pool_base_url = _pool_base_url()
  token = get_str("asr_pool.token", "")
  http_timeout_s = _http_timeout_s()
  retry_attempts = _retry_attempts()
  retry_base_delay_s = _retry_base_delay_s()
  retry_max_delay_s = max(retry_base_delay_s, _retry_max_delay_s())
  retry_jitter_s = _retry_jitter_s()
  clean_ids: list[str] = []
  seen: set[str] = set()
  for raw in list(request_ids or []):
    rid = str(raw or "").strip()
    if not rid or rid in seen:
      continue
    seen.add(rid)
    clean_ids.append(rid)
    if len(clean_ids) >= int(max(1, min(1000, int(limit)))):
      break
  query = urlparse.urlencode(
    {
      "consumer_id": str(consumer_id or ""),
      "limit": int(max(1, min(1000, int(limit)))),
      "request_id": clean_ids,
    },
    doseq=True,
  )
  url = urlparse.urljoin(pool_base_url + "/", f"asr/v1/pending-status?{query}")
  try:
    status_code, body, _attempts_used = _http_json_with_retry(
      method="GET",
      url=url,
      token=token,
      timeout_s=http_timeout_s,
      payload=None,
      attempts=retry_attempts,
      backoff_base_s=retry_base_delay_s,
      backoff_max_s=retry_max_delay_s,
      jitter_s=retry_jitter_s,
    )
  except Exception as e:
    return {
      "ok": False,
      "status_code": 0,
      "body": {
        "code": "ASR_REMOTE_PENDING_STATUS_IO_FAILURE",
        "message": f"ASR pool pending status I/O failed: {type(e).__name__}: {e}",
        "retryable": True,
      },
    }
  return {
    "ok": bool(int(status_code) == 200),
    "status_code": int(status_code),
    "body": dict(body or {}),
  }
