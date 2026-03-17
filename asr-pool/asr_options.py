from __future__ import annotations

from typing import Any


class AsrOptionsError(ValueError):
  def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None) -> None:
    super().__init__(message)
    self.code = str(code)
    self.details = dict(details or {})


_KNOWN_OPTION_KEYS = {
  "language",
  "align_enabled",
  "diarize_enabled",
  "speaker_mode",
  "min_speakers",
  "max_speakers",
  "beam_size",
  "initial_prompt",
  "latency_mode",
}


def _as_bool(value: Any) -> bool:
  if isinstance(value, bool):
    return value
  s = str(value or "").strip().lower()
  return s in {"1", "true", "yes", "on", "y"}


def normalize_options(options: dict[str, Any] | None) -> dict[str, Any]:
  opts = dict(options or {})
  for key in list(opts.keys()):
    if key not in _KNOWN_OPTION_KEYS:
      raise AsrOptionsError(
        "ASR_UNKNOWN_OPTION",
        f"Unknown ASR option: {key}",
        details={"option": key},
      )

  resolved: dict[str, Any] = {
    "align_enabled": False,
    "diarize_enabled": False,
    "speaker_mode": "none",
    "latency_mode": "default",
  }
  for key, value in opts.items():
    if value is None:
      continue
    resolved[key] = value

  if "language" in resolved and resolved["language"] is not None:
    resolved["language"] = str(resolved["language"]).strip().lower() or None

  if "speaker_mode" in resolved and resolved["speaker_mode"] is not None:
    speaker_mode = str(resolved["speaker_mode"]).strip().lower()
    if speaker_mode in {"off", "disabled", "no_speaker", "nospeaker", "no-speaker"}:
      speaker_mode = "none"
    if speaker_mode not in {"none", "auto", "fixed"}:
      speaker_mode = "auto"
    resolved["speaker_mode"] = speaker_mode

  if "align_enabled" in resolved and resolved["align_enabled"] is not None:
    resolved["align_enabled"] = _as_bool(resolved["align_enabled"])
  if "diarize_enabled" in resolved and resolved["diarize_enabled"] is not None:
    resolved["diarize_enabled"] = _as_bool(resolved["diarize_enabled"])

  for key in ("min_speakers", "max_speakers"):
    if key in resolved and resolved[key] is not None:
      try:
        resolved[key] = int(resolved[key])
      except Exception:
        resolved[key] = None

  if "beam_size" in resolved and resolved["beam_size"] is not None:
    try:
      resolved["beam_size"] = max(1, int(resolved["beam_size"]))
    except Exception:
      resolved["beam_size"] = 5

  if "latency_mode" in resolved and resolved["latency_mode"] is not None:
    latency_mode = str(resolved["latency_mode"]).strip().lower()
    if latency_mode not in {"low", "default"}:
      latency_mode = "default"
    resolved["latency_mode"] = latency_mode

  if "initial_prompt" in resolved and resolved["initial_prompt"] is not None:
    resolved["initial_prompt"] = str(resolved["initial_prompt"])

  return resolved
