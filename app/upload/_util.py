from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return dict(raw) if isinstance(raw, dict) else {}


def _write_json_atomic(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, path)


def _resolve_child_path(base_dir: Path, child: str | Path) -> Path | None:
    base_dir = Path(base_dir).resolve()
    candidate = (base_dir / child).resolve()
    try:
        candidate.relative_to(base_dir)
    except Exception:
        return None
    return candidate


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _normalize_speaker_mode(mode: Any) -> str:
    raw = str(mode or "auto").strip().lower()
    if raw in {"none", "off", "disabled", "no_speaker", "nospeaker", "no-speaker"}:
        return "none"
    if raw == "fixed":
        return "fixed"
    return "auto"


def _normalize_upload_language(language: Any) -> str:
    raw = str(language or "").strip().lower()
    if raw in {"", "auto", "detect", "detect_auto", "detect-automatic", "detect-automatically"}:
        return ""
    return raw


def _resolve_status_owner(*, key: str, default: str, service_cfg: dict[str, Any] | None = None) -> str:
    if isinstance(service_cfg, dict):
        status_owners = service_cfg.get("status_owners") or {}
        if isinstance(status_owners, dict):
            raw = str(status_owners.get(key) or "").strip()
            if raw:
                return raw
        return default
    from app.config.settings import get_str

    raw = str(get_str(f"upload.status_owners.{key}", default) or "").strip()
    return raw or default


def _append_log(path: Path, message: str) -> None:
    from datetime import datetime, timezone

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with path.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] COORD {message}\n")


def _hms_to_seconds(hms: str) -> int:
    hh, mm, ss = hms.split(":")
    return int(hh) * 3600 + int(mm) * 60 + int(ss)


def _seconds_to_hms(total_s: int) -> str:
    total_s = max(0, int(total_s))
    hh = total_s // 3600
    mm = (total_s % 3600) // 60
    ss = total_s % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def _topics_enabled_for_job(
    *,
    status: dict[str, Any] | None = None,
    job_cfg: dict[str, Any] | None = None,
    opts: dict[str, Any] | None = None,
    service_cfg: dict[str, Any] | None = None,
) -> bool | None:
    for source in (status, opts):
        if isinstance(source, dict):
            value = source.get("topics_enabled")
            if value is not None:
                return bool(value)
    if isinstance(job_cfg, dict):
        for section in ("upload", "options"):
            cfg = job_cfg.get(section) or {}
            if isinstance(cfg, dict):
                value = cfg.get("topics_enabled")
                if value is not None:
                    return bool(value)
    if isinstance(service_cfg, dict):
        topics_cfg = dict(service_cfg.get("topics") or {})
        return bool(topics_cfg.get("enabled", False))
    return None


def _topics_prompt_id(value: Any) -> str:
    raw = str(value or "").strip()
    return raw or "topics_v1"


def _topics_merged_filename(*, orig_stem: str, prompt_id: Any) -> str:
    safe_stem = Path(str(orig_stem or "").strip()).stem or "transcript"
    return f"{safe_stem}_{_topics_prompt_id(prompt_id)}_merged.json"
