from __future__ import annotations

from typing import Any


def live_commit_rows_debug_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_rows = 0
    invalid_index_rows = 0
    by_index: dict[int, dict[str, Any]] = {}
    reason_counts: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        total_rows += 1
        try:
            idx = int(row.get("chunk_index"))
        except Exception:
            invalid_index_rows += 1
            continue
        # Payload contract is one row per chunk_index. If duplicates exist, keep the last one.
        by_index[idx] = row
    for row in by_index.values():
        reason = str(row.get("reason") or "").strip()
        if not reason:
            continue
        reason_counts[reason] = int(reason_counts.get(reason, 0) + 1)
    unique_rows = len(by_index)
    duplicate_rows = max(0, total_rows - invalid_index_rows - unique_rows)
    return {
        "chunk_reason_counts": dict(sorted(reason_counts.items(), key=lambda kv: kv[0])),
        "chunk_results_rows_count": int(max(0, total_rows)),
        "chunk_results_unique_count": int(max(0, unique_rows)),
        "chunk_results_duplicate_index_rows": int(max(0, duplicate_rows)),
        "chunk_results_invalid_index_rows": int(max(0, invalid_index_rows)),
    }


def build_live_session_manager_metrics_payload(
    *,
    active_sessions: list[Any],
    archived_sessions_count: int,
    now_mono: float,
    limits: dict[str, int],
) -> dict[str, Any]:
    states: dict[str, int] = {}
    connected = 0
    bytes_received = 0
    frames_received = 0
    controls_received = 0
    max_age_s = 0.0

    for sess in active_sessions:
        state = str(getattr(sess, "state", "") or "unknown")
        states[state] = int(states.get(state, 0) + 1)
        if bool(getattr(sess, "ws_connected", False)):
            connected += 1
        bytes_received += int(max(0, int(getattr(sess, "bytes_received", 0) or 0)))
        frames_received += int(max(0, int(getattr(sess, "frames_received", 0) or 0)))
        controls_received += int(max(0, int(getattr(sess, "controls_received", 0) or 0)))
        age_s = max(0.0, float(now_mono - float(getattr(sess, "created_monotonic", 0.0) or 0.0)))
        if age_s > max_age_s:
            max_age_s = age_s

    return {
        "active_sessions": len(active_sessions),
        "active_ws_connected": int(connected),
        "active_states": states,
        "active_max_age_s": round(max_age_s, 3),
        "active_bytes_received": int(bytes_received),
        "active_frames_received": int(frames_received),
        "active_controls_received": int(controls_received),
        "archived_sessions": int(max(0, archived_sessions_count)),
        "limits": {
            "max_sessions": int(limits["max_sessions"]),
            "default_ttl_seconds": int(limits["default_ttl_seconds"]),
            "preconnect_ttl_seconds": int(limits["preconnect_ttl_seconds"]),
            "archive_ttl_seconds": int(limits["archive_ttl_seconds"]),
            "max_archives": int(limits["max_archives"]),
        },
    }
