from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter

router = APIRouter()

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _iso_utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_sensitive_key(name: str) -> bool:
    k = str(name or "").strip().lower()
    if not k or k.endswith("_env"):
        return False
    if k in {
        "token",
        "hf_token",
        "api_key",
        "apikey",
        "password",
        "secret",
        "access_token",
        "refresh_token",
        "authorization",
        "bearer_token",
    }:
        return True
    return (
        k.endswith("_token")
        or k.endswith("_api_key")
        or k.endswith("_apikey")
        or k.endswith("_password")
        or k.endswith("_secret")
    )


def _redact_sensitive(value: Any) -> Any:
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, child in value.items():
            if _is_sensitive_key(str(key)):
                out[str(key)] = "***REDACTED***"
            else:
                out[str(key)] = _redact_sensitive(child)
        return out
    if isinstance(value, list):
        return [_redact_sensitive(v) for v in value]
    return value


def _file_config_source(*, source_id: str, title: str, path: Path) -> Dict[str, Any]:
    exists = path.exists()
    size_bytes: int | None = None
    mtime_utc: str | None = None
    if exists:
        try:
            stat = path.stat()
            size_bytes = int(stat.st_size)
            mtime_utc = _iso_utc(float(stat.st_mtime))
        except Exception:
            size_bytes = None
            mtime_utc = None

    parse_ok = False
    data: Dict[str, Any] = {}
    error: str | None = None
    if not exists:
        error = "File not found"
    else:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                data = raw
                parse_ok = True
            else:
                error = "JSON root must be an object"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

    return {
        "id": source_id,
        "title": title,
        "path": str(path),
        "exists": exists,
        "size_bytes": size_bytes,
        "mtime_utc": mtime_utc,
        "parse_ok": parse_ok,
        "data": _redact_sensitive(data),
        "error": error,
    }


def _load_json_object(path: Path) -> Dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(raw) if isinstance(raw, dict) else {}


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(base or {})
    for key, value in (override or {}).items():
        if str(key).startswith("_"):
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge_dict(dict(out.get(key) or {}), value)
        else:
            out[key] = value
    return out


def _coerce_live_ui_settings(data: Dict[str, Any]) -> Dict[str, Any]:
    src = data if isinstance(data, dict) else {}
    defaults: Dict[str, Any] = {
        "transcript_presentation_mode": "segment_blocks_diarize_hard_v1",
        "speaker_labels_default_enabled": True,
        "transcript_format_rules": {
            "blockEverySegments": 3,
            "blockMinChars": 220,
            "blockMinWords": 35,
        },
    }
    out = _deep_merge_dict(defaults, src)
    rules = out.get("transcript_format_rules")
    if not isinstance(rules, dict):
        rules = dict(defaults["transcript_format_rules"])
    nrules: Dict[str, Any] = {}
    int_keys = {"blockEverySegments", "blockMinChars", "blockMinWords"}
    for k in defaults["transcript_format_rules"].keys():
        v = rules.get(k, defaults["transcript_format_rules"][k])
        if k in int_keys:
            try:
                nrules[k] = max(0, int(v))
            except Exception:
                nrules[k] = int(defaults["transcript_format_rules"][k])
    return {
        "transcript_presentation_mode": "segment_blocks_diarize_hard_v1",
        "speaker_labels_default_enabled": bool(out.get("speaker_labels_default_enabled", True)),
        "transcript_format_rules": nrules,
    }


def _load_ui_settings() -> Dict[str, Any]:
    config_dir = (_REPO_ROOT / "config").resolve()
    base_path = (config_dir / "ui_settings.json").resolve()
    local_path = (config_dir / "ui_settings.local.json").resolve()
    base_obj = _load_json_object(base_path) if base_path.exists() else {}
    local_obj = _load_json_object(local_path) if local_path.exists() else {}
    merged = _deep_merge_dict(base_obj, local_obj)
    live_obj = merged.get("live")
    return {
        "version": str(merged.get("version") or "ui_settings_v1"),
        "live": _coerce_live_ui_settings(live_obj if isinstance(live_obj, dict) else {}),
    }


@router.get("/health")
def health() -> Dict[str, bool]:
    return {"ok": True}


@router.get("/demo/settings")
def get_demo_settings() -> Dict[str, Any]:
    config_dir = (_REPO_ROOT / "config").resolve()
    settings_path = (config_dir / "settings.json").resolve()
    local_path = (config_dir / "local.json").resolve()
    return {
        "generated_at_utc": _iso_utc(datetime.now(timezone.utc).timestamp()),
        "sources": [
            _file_config_source(source_id="settings_json", title="settings.json", path=settings_path),
            _file_config_source(source_id="local_json", title="local.json", path=local_path),
        ],
    }


@router.get("/ui/settings")
def get_ui_settings() -> Dict[str, Any]:
    settings = _load_ui_settings()
    return {
        "generated_at_utc": _iso_utc(datetime.now(timezone.utc).timestamp()),
        "version": str(settings.get("version") or "ui_settings_v1"),
        "settings": {
            "live": dict(settings.get("live") or {}),
        },
    }
