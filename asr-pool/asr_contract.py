from __future__ import annotations

from copy import deepcopy
import sys
from pathlib import Path
from typing import Any

from asr_options import AsrOptionsError, normalize_options
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from shared.asr.schema import (
  ASR_SCHEMA_VERSION,
  ASR_SCHEMA_VERSIONS_SUPPORTED,
)


class AsrRequestError(ValueError):
  def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None) -> None:
    super().__init__(message)
    self.code = str(code)
    self.details = dict(details or {})


def _normalize_outputs(raw: dict[str, Any] | None) -> dict[str, bool]:
  src = dict(raw or {})
  out = {
    "text": True,
    "segments": True,
    "srt": True,
    "srt_inline": False,
  }
  for key in list(out.keys()):
    if key in src and src[key] is not None:
      out[key] = bool(src[key])
  return out


def _validate_audio(audio: dict[str, Any] | None) -> dict[str, Any]:
  a = dict(audio or {})
  local_path = str(a.get("local_path") or "").strip()
  inline_base64 = str(a.get("inline_base64") or "").strip()
  blob_ref = str(a.get("blob_ref") or "").strip()

  provided = [name for name, val in (("local_path", local_path), ("inline_base64", inline_base64), ("blob_ref", blob_ref)) if val]
  if len(provided) != 1:
    raise AsrRequestError(
      "ASR_AUDIO_SOURCE_INVALID",
      "Exactly one audio source must be provided",
      details={"provided_sources": provided},
    )
  if "inline_base64" in provided:
    raise AsrRequestError(
      "ASR_AUDIO_SOURCE_NOT_IMPLEMENTED",
      "audio.inline_base64 is not supported in the current implementation",
      details={"provided_sources": provided},
    )
  if "local_path" in provided:
    a["local_path"] = local_path
  if "blob_ref" in provided:
    a["blob_ref"] = blob_ref
  return a


def _normalize_priority(value: Any) -> str:
  out = str(value or "normal").strip().lower() or "normal"
  if out not in {"interactive", "normal", "background"}:
    out = "normal"
  return out


def _normalize_routing(raw: dict[str, Any] | None) -> dict[str, Any]:
  src = dict(raw or {})
  allowed = {"fairness_key", "slot_affinity", "timeout_s"}
  for key in list(src.keys()):
    if key not in allowed:
      raise AsrRequestError(
        "ASR_UNKNOWN_ROUTING_KEY",
        f"Unknown ASR routing key: {key}",
        details={"routing_key": key, "allowed_routing_keys": sorted(allowed)},
      )
  out: dict[str, Any] = {}
  fairness_key = str(src.get("fairness_key") or "").strip()
  if fairness_key:
    out["fairness_key"] = fairness_key
  if "slot_affinity" in src and src.get("slot_affinity") is not None:
    try:
      out["slot_affinity"] = int(src.get("slot_affinity"))
    except Exception:
      raise AsrRequestError(
        "ASR_ROUTING_SLOT_AFFINITY_INVALID",
        "routing.slot_affinity must be an integer when provided",
        details={"slot_affinity": src.get("slot_affinity")},
      )
  if "timeout_s" in src and src.get("timeout_s") is not None:
    try:
      out["timeout_s"] = max(1, int(src.get("timeout_s")))
    except Exception:
      raise AsrRequestError(
        "ASR_ROUTING_TIMEOUT_INVALID",
        "routing.timeout_s must be an integer when provided",
        details={"timeout_s": src.get("timeout_s")},
      )
  return out


def prepare_request(raw_request: dict[str, Any]) -> dict[str, Any]:
  req = deepcopy(dict(raw_request or {}))
  schema_version = str(req.get("schema_version") or "").strip()
  if schema_version != ASR_SCHEMA_VERSION:
    raise AsrRequestError(
      "ASR_SCHEMA_UNSUPPORTED",
      f"Unsupported schema_version: {schema_version or '<missing>'}",
      details={"supported": list(ASR_SCHEMA_VERSIONS_SUPPORTED)},
    )
  request_id = str(req.get("request_id") or "").strip()
  if not request_id:
    raise AsrRequestError("ASR_REQUEST_ID_REQUIRED", "request_id is required")

  req["audio"] = _validate_audio(req.get("audio"))
  req["outputs"] = _normalize_outputs(req.get("outputs"))
  req["options"] = dict(req.get("options") or {})
  req["schema_version"] = ASR_SCHEMA_VERSION
  req["priority"] = _normalize_priority(req.get("priority"))
  req["consumer_id"] = str(req.get("consumer_id") or "").strip()
  req["routing"] = _normalize_routing(req.get("routing"))
  try:
    effective_options = normalize_options(req.get("options"))
  except AsrOptionsError as e:
    raise AsrRequestError(e.code, str(e), details=e.details) from e
  req["effective_options"] = effective_options
  return req


def build_error_response(
  *,
  request: dict[str, Any] | None,
  code: str,
  message: str,
  retryable: bool = False,
  details: dict[str, Any] | None = None,
  effective_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
  req = dict(request or {})
  return {
    "schema_version": ASR_SCHEMA_VERSION,
    "request_id": str(req.get("request_id") or ""),
    "ok": False,
    "effective_options": dict(
      effective_options
      if effective_options is not None
      else (req.get("effective_options") or {})
    ),
    "error": {
      "code": str(code),
      "message": str(message),
      "retryable": bool(retryable),
      "details": dict(details or {}),
    },
    "warnings": [],
  }
