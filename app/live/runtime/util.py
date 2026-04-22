from __future__ import annotations

import re
from typing import Any

LIVE_ASR_LANGUAGE_ERROR = "language must be empty/auto or a short BCP-47 style code (e.g. 'en', 'nl', 'pt-br')"
_LIVE_ASR_LANGUAGE_RE = re.compile(r"^[a-z]{2,3}(?:[-_][a-z0-9]{2,8})?$")


def _safe_float(value: Any) -> float | None:
    try:
        return max(0.0, float(value))
    except Exception:
        return None


def _normalize_optional_language(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def parse_live_asr_language(value: Any) -> str | None:
    text = _normalize_optional_language(value)
    if text is None:
        return None
    normalized = str(text).lower()
    if normalized in {"auto", "default", "server-default", "server_default"}:
        return None
    if not _LIVE_ASR_LANGUAGE_RE.match(normalized):
        raise ValueError(LIVE_ASR_LANGUAGE_ERROR)
    return normalized
