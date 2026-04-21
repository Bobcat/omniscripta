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

