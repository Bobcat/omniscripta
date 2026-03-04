from __future__ import annotations

from copy import deepcopy
import sys
from pathlib import Path
from typing import Any

from asr_profiles import AsrProfileError, resolve_profile_options
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from shared.asr.schema import ASR_SCHEMA_VERSION


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
    "word_timestamps": False,
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


def prepare_request(raw_request: dict[str, Any]) -> dict[str, Any]:
  req = deepcopy(dict(raw_request or {}))
  schema_version = str(req.get("schema_version") or "").strip()
  if schema_version != ASR_SCHEMA_VERSION:
    raise AsrRequestError(
      "ASR_SCHEMA_UNSUPPORTED",
      f"Unsupported schema_version: {schema_version or '<missing>'}",
      details={"expected": ASR_SCHEMA_VERSION},
    )
  request_id = str(req.get("request_id") or "").strip()
  if not request_id:
    raise AsrRequestError("ASR_REQUEST_ID_REQUIRED", "request_id is required")
  profile_id = str(req.get("profile_id") or "").strip()
  if not profile_id:
    raise AsrRequestError("ASR_PROFILE_REQUIRED", "profile_id is required")

  req["audio"] = _validate_audio(req.get("audio"))
  req["outputs"] = _normalize_outputs(req.get("outputs"))
  req["options"] = dict(req.get("options") or {})
  req["context"] = dict(req.get("context") or {})
  if "speculative_seq" in req["context"] or "speculative_audio_end_ms" in req["context"]:
    raise AsrRequestError(
      "ASR_CONTEXT_KEY_UNSUPPORTED",
      "Deprecated context keys are not supported; use preview_seq/preview_audio_end_ms",
      details={"deprecated_keys": ["speculative_seq", "speculative_audio_end_ms"]},
    )
  req["priority"] = str(req.get("priority") or "normal").strip().lower() or "normal"
  if req["priority"] not in {"interactive", "normal", "background"}:
    req["priority"] = "normal"

  try:
    resolved = resolve_profile_options(profile_id=profile_id, options=req.get("options"))
  except AsrProfileError as e:
    raise AsrRequestError(e.code, str(e), details=e.details) from e
  req["resolved_options"] = resolved
  return req


def build_error_response(
  *,
  request: dict[str, Any] | None,
  code: str,
  message: str,
  retryable: bool = False,
  details: dict[str, Any] | None = None,
  resolved_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
  req = dict(request or {})
  return {
    "schema_version": ASR_SCHEMA_VERSION,
    "request_id": str(req.get("request_id") or ""),
    "ok": False,
    "profile_id": str(req.get("profile_id") or ""),
    "resolved_options": dict(resolved_options if resolved_options is not None else (req.get("resolved_options") or {})),
    "error": {
      "code": str(code),
      "message": str(message),
      "retryable": bool(retryable),
      "details": dict(details or {}),
    },
    "warnings": [],
  }
