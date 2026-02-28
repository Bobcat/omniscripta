from __future__ import annotations

from typing import Any


class AsrProfileError(ValueError):
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
  "initial_prompt",
  "timestamps_mode",
}


_PROFILES: dict[str, dict[str, Any]] = {
  "live_fast": {
    "defaults": {
      "align_enabled": False,
      "diarize_enabled": False,
      "speaker_mode": "none",
      "timestamps_mode": "segment",
    },
    "allowed_overrides": {
      "language",
      "align_enabled",
      "initial_prompt",
    },
  },
  "upload_full": {
    "defaults": {
      "align_enabled": True,
      "timestamps_mode": "segment",
    },
    "allowed_overrides": {
      "language",
      "speaker_mode",
      "min_speakers",
      "max_speakers",
      "diarize_enabled",
      "align_enabled",
      "initial_prompt",
      "timestamps_mode",
    },
  },
}


def _as_bool(value: Any) -> bool:
  if isinstance(value, bool):
    return value
  s = str(value or "").strip().lower()
  return s in {"1", "true", "yes", "on", "y"}


def _normalize_resolved_options(obj: dict[str, Any]) -> dict[str, Any]:
  out = dict(obj or {})
  if "language" in out and out["language"] is not None:
    out["language"] = str(out["language"]).strip().lower() or None
  if "speaker_mode" in out and out["speaker_mode"] is not None:
    sm = str(out["speaker_mode"]).strip().lower()
    if sm in {"off", "disabled", "no_speaker", "nospeaker", "no-speaker"}:
      sm = "none"
    if sm not in {"none", "auto", "fixed"}:
      sm = "auto"
    out["speaker_mode"] = sm
  if "align_enabled" in out and out["align_enabled"] is not None:
    out["align_enabled"] = _as_bool(out["align_enabled"])
  if "diarize_enabled" in out and out["diarize_enabled"] is not None:
    out["diarize_enabled"] = _as_bool(out["diarize_enabled"])
  for k in ("min_speakers", "max_speakers"):
    if k in out and out[k] is not None:
      try:
        out[k] = int(out[k])
      except Exception:
        out[k] = None
  if "timestamps_mode" in out and out["timestamps_mode"] is not None:
    tm = str(out["timestamps_mode"]).strip().lower()
    if tm not in {"segment", "word", "none"}:
      tm = "segment"
    out["timestamps_mode"] = tm
  if "initial_prompt" in out and out["initial_prompt"] is not None:
    out["initial_prompt"] = str(out["initial_prompt"])
  return out


def resolve_profile_options(*, profile_id: str, options: dict[str, Any] | None) -> dict[str, Any]:
  pid = str(profile_id or "").strip()
  if not pid:
    raise AsrProfileError("ASR_PROFILE_REQUIRED", "profile_id is required")
  prof = _PROFILES.get(pid)
  if prof is None:
    raise AsrProfileError(
      "ASR_UNKNOWN_PROFILE",
      f"Unknown ASR profile: {pid}",
      details={"profile_id": pid, "known_profiles": sorted(_PROFILES.keys())},
    )

  resolved = dict(prof.get("defaults") or {})
  allowed = set(prof.get("allowed_overrides") or set())
  opts = dict(options or {})

  for key, value in opts.items():
    if key not in _KNOWN_OPTION_KEYS:
      raise AsrProfileError(
        "ASR_UNKNOWN_OPTION",
        f"Unknown ASR option: {key}",
        details={"profile_id": pid, "option": key},
      )
    if value is None:
      continue
    if key not in allowed:
      raise AsrProfileError(
        "ASR_FORBIDDEN_OVERRIDE",
        f"Option not allowed for profile '{pid}': {key}",
        details={"profile_id": pid, "option": key},
      )
    resolved[key] = value

  return _normalize_resolved_options(resolved)
