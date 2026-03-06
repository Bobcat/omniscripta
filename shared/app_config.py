"""Application configuration loader.

Loads configuration from:
1. config/settings.json (defaults, in git)
2. config/local.json (workspace overrides, gitignored)
3. Environment variables for secret-like keys only

Later sources override earlier ones.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


def _repo_root() -> Path:
    """Find repository root (where this file lives)."""
    return Path(__file__).resolve().parents[1]


def _load_json_file(path: Path) -> dict[str, Any]:
    """Load JSON file if it exists, else empty dict."""
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, IOError):
        return {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge override into base."""
    result = dict(base)
    for key, value in override.items():
        if key.startswith("_"):
            continue  # Skip comments
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _get_env_value(name: str, default: Any = None) -> Any:
    """Get value from environment, with type inference."""
    raw = os.getenv(name)
    if raw is None:
        return default
    # Try bool first
    if raw.lower() in ("1", "true", "yes", "on"):
        return True
    if raw.lower() in ("0", "false", "no", "off"):
        return False
    # Try int
    try:
        return int(raw)
    except ValueError:
        pass
    # Try float
    try:
        return float(raw)
    except ValueError:
        pass
    # Return as string
    return raw


def load_config() -> dict[str, Any]:
    """Load full configuration from all sources."""
    config_dir = _repo_root() / "config"
    
    # 1. Load defaults
    settings = _load_json_file(config_dir / "settings.json")
    
    # 2. Merge local overrides
    local = _load_json_file(config_dir / "local.json")
    config = _deep_merge(settings, local)
    
    return config


# Singleton instance
_CONFIG: dict[str, Any] | None = None


def _is_secret_path(path: str) -> bool:
    leaf = str(path or "").split(".")[-1].strip().lower()
    if not leaf:
        return False
    if leaf in {
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
        leaf.endswith("_token")
        or leaf.endswith("_api_key")
        or leaf.endswith("_apikey")
        or leaf.endswith("_password")
        or leaf.endswith("_secret")
    )


def get_config() -> dict[str, Any]:
    """Get cached configuration (loads once)."""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = load_config()
    return _CONFIG


def get_setting(path: str, default: T | None = None) -> T | Any:
    """Get a setting by dot-notation path (e.g. 'asr_pool.runner_slots').

    Environment overrides are only applied for secret-like keys.
    """
    # Env overrides are restricted to secret-like keys.
    if _is_secret_path(path):
        env_name = "TRANSCRIBE_" + path.upper().replace(".", "_")
        env_value = _get_env_value(env_name)
        if env_value is not None:
            return env_value
    
    # Walk the config path
    cfg = get_config()
    keys = path.split(".")
    try:
        for key in keys:
            cfg = cfg[key]
        return cfg
    except (KeyError, TypeError):
        return default


def get_str(path: str, default: str = "") -> str:
    """Get string setting."""
    val = get_setting(path, default)
    return str(val) if val is not None else default


def get_int(path: str, default: int = 0, *, min_value: int | None = None) -> int:
    """Get integer setting with optional minimum."""
    val = get_setting(path, default)
    try:
        result = int(val)
    except (TypeError, ValueError):
        result = default
    if min_value is not None:
        result = max(min_value, result)
    return result


def get_float(path: str, default: float = 0.0, *, min_value: float | None = None) -> float:
    """Get float setting with optional minimum."""
    val = get_setting(path, default)
    try:
        result = float(val)
    except (TypeError, ValueError):
        result = default
    if min_value is not None:
        result = max(min_value, result)
    return result


def get_bool(path: str, default: bool = False) -> bool:
    """Get boolean setting."""
    val = get_setting(path, default)
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("1", "true", "yes", "on")
    return bool(val)


def get_list(path: str, default: list[Any] | None = None) -> list[Any]:
    """Get list setting."""
    val = get_setting(path, default or [])
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        # Comma-separated string
        return [x.strip() for x in val.split(",") if x.strip()]
    return default or []


def reload_config() -> dict[str, Any]:
    """Force reload of configuration."""
    global _CONFIG
    _CONFIG = load_config()
    return _CONFIG


# Legacy compatibility: helper to get env var with prefix
def env(name: str, default: T | None = None) -> T | Any:
    """Get environment variable (legacy helper)."""
    return _get_env_value(name, default)
