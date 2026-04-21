from __future__ import annotations

from typing import Any


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _normalize_optional_language(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None

