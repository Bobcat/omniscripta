from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict

from live.config import LIVE_BENCHMARK_EXPORT_ROOT, LIVE_RECORDINGS_ROOT


def _format_srt_timestamp(ms: int) -> str:
    total_ms = int(max(0, ms))
    hours = total_ms // 3_600_000
    rem = total_ms % 3_600_000
    minutes = rem // 60_000
    rem = rem % 60_000
    seconds = rem // 1000
    millis = rem % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def live_result_to_srt_text(result: dict[str, Any]) -> str:
    segments_any = result.get("final_segments")
    if not isinstance(segments_any, list):
        return ""
    rows: list[str] = []
    idx = 0
    for seg in segments_any:
        if not isinstance(seg, dict):
            continue
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        t0_ms = int(max(0, int(seg.get("t0_ms") or 0)))
        t1_ms = int(max(t0_ms + 1, int(seg.get("t1_ms") or (t0_ms + 1))))
        idx += 1
        rows.append(str(idx))
        rows.append(f"{_format_srt_timestamp(t0_ms)} --> {_format_srt_timestamp(t1_ms)}")
        rows.append(text)
        rows.append("")
    return "\n".join(rows).strip() + ("\n" if rows else "")


def live_result_to_plain_text(result: dict[str, Any]) -> str:
    segments_any = result.get("final_segments")
    if not isinstance(segments_any, list):
        return ""
    rows: list[str] = []
    for seg in segments_any:
        if not isinstance(seg, dict):
            continue
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        rows.append(text)
    return "\n".join(rows).strip()


def _normalize_pc_text(value: Any) -> str:
    text = str(value or "")
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ")


def live_pc_events_to_text(events: list[dict[str, Any]]) -> str:
    rows: list[str] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        kind = str(event.get("kind") or "").strip().lower()
        if kind not in {"p", "c"}:
            continue
        text = _normalize_pc_text(event.get("text"))
        if kind == "c" and not text:
            continue
        rows.append(f"{kind},{text}")
    return "".join(f"{row}\n" for row in rows)


def live_recording_wav_path_from_result(result: dict[str, Any]) -> Path | None:
    raw = str((result or {}).get("recording_path") or "").strip()
    if not raw:
        return None
    try:
        candidate = Path(raw).expanduser().resolve()
    except Exception:
        return None
    try:
        candidate.relative_to(LIVE_RECORDINGS_ROOT)
    except Exception:
        return None
    if candidate.suffix.lower() != ".wav":
        return None
    if not candidate.is_file():
        return None
    return candidate


def build_live_result_envelope(
    *,
    session_id: str,
    result_payload: dict[str, Any],
    rooted_path_cb: Callable[[str], str],
) -> dict[str, Any]:
    result = dict(result_payload or {})
    effective_engine = str(result.get("live_engine") or LIVE_ENGINE)
    result["live_engine"] = effective_engine

    final_segments = result.get("final_segments")
    has_segments = isinstance(final_segments, list) and any(isinstance(seg, dict) for seg in final_segments)
    has_recording_wav = live_recording_wav_path_from_result(result) is not None
    has_pc_replay = int(max(0, int(result.get("pc_events_count") or 0))) > 0

    finalization_state = str(result.get("finalization_state") or "").strip().lower()

    return {
        "session_id": str(session_id),
        "live_engine": effective_engine,
        "result": result,
        "ready": finalization_state in {"ready", "finalized"},
        "can_export_srt": bool(has_segments),
        "can_export_wav": bool(has_recording_wav),
        "can_export_pc": bool(has_pc_replay),
        "transcript_srt_url": rooted_path_cb(f"/demo/live/sessions/{session_id}/transcript.srt") if has_segments else None,
        "recording_wav_url": rooted_path_cb(f"/demo/live/sessions/{session_id}/recording.wav") if has_recording_wav else None,
        "transcript_pc_url": rooted_path_cb(f"/demo/live/sessions/{session_id}/transcript.pc") if has_pc_replay else None,
    }


def _iso_utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def safe_filename(name: str) -> str:
    return Path(name).name or "upload.bin"


def _write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)


def _autosave_live_benchmark_snapshot(
    *,
    session_id: str,
    artifact_name: str,
    envelope: Dict[str, Any],
    request_meta: Dict[str, Any] | None = None,
) -> None:
    safe_session_id = safe_filename(str(session_id or "session"))
    artifact = safe_filename(str(artifact_name or "benchmark"))
    now_ts = time.time()
    now_iso = _iso_utc(now_ts)
    record: Dict[str, Any] = {
        "saved_at_utc": now_iso,
        "saved_at_unix": round(float(now_ts), 6),
        "session_id": str(session_id or ""),
        "artifact_name": artifact_name,
        "request_meta": dict(request_meta or {}),
        "payload": envelope,
    }
    latest_path = (LIVE_BENCHMARK_EXPORT_ROOT / f"{safe_session_id}.{artifact}.latest.json").resolve()
    history_path = (LIVE_BENCHMARK_EXPORT_ROOT / f"{safe_session_id}.{artifact}.history.jsonl").resolve()
    _write_json_atomic(latest_path, record)
    _append_jsonl(history_path, record)


def try_autosave_live_benchmark_snapshot(
    *,
    session_id: str,
    artifact_name: str,
    envelope: Dict[str, Any],
    request_meta: Dict[str, Any] | None = None,
) -> None:
    try:
        _autosave_live_benchmark_snapshot(
            session_id=session_id,
            artifact_name=artifact_name,
            envelope=envelope,
            request_meta=request_meta,
        )
    except Exception as e:
        print(f"[live-benchmark-autosave] failed {artifact_name} session={session_id}: {type(e).__name__}: {e}")


def _read_json_file(path: Path) -> dict[str, Any] | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return raw if isinstance(raw, dict) else None


def _flatten_live_benchmark_record(record: dict[str, Any]) -> dict[str, Any]:
    payload = dict(record.get("payload") or {})
    quality = dict(payload.get("quality") or {})
    score = dict(quality.get("score") or {})
    run = dict(quality.get("run_metrics") or {})
    request_meta = dict(record.get("request_meta") or {})
    tuning = dict(request_meta.get("live_tuning_snapshot") or {})

    return {
        "session_id": str(record.get("session_id") or payload.get("session_id") or ""),
        "saved_at_utc": str(record.get("saved_at_utc") or ""),
        "fixture_id": str(payload.get("fixture_id") or ""),
        "fixture_test_mode": str(request_meta.get("fixture_test_mode") or ""),
        "fixture_version": str(request_meta.get("fixture_version") or ""),
        "score": score.get("upload_similarity_score"),
        "word_edit_distance": score.get("word_edit_distance"),
        "word_count_live": score.get("word_count_live"),
        "word_count_reference": score.get("word_count_reference"),
        "recording_duration_ms": int(run.get("recording_duration_ms") or 0),
        "transcript_revision": int(run.get("transcript_revision") or 0),
        "chunks_total": int(run.get("chunks_total") or 0),
        "chunks_done": int(run.get("chunks_done") or 0),
        "chunks_failed": int(run.get("chunks_failed") or 0),
        "chunks_pending": int(run.get("chunks_pending") or 0),
        "final_segments_count": int(run.get("final_segments_count") or 0),
        "chunk_reason_counts": (
            dict(run.get("chunk_reason_counts"))
            if isinstance(run.get("chunk_reason_counts"), dict)
            else {}
        ),
        "gpu_proxy_transcribe_total_s": run.get("gpu_proxy_transcribe_total_s"),
        "gpu_proxy_pipeline_total_s": run.get("gpu_proxy_pipeline_total_s"),
        "live_tuning_snapshot": tuning,
    }


def list_live_benchmark_exports(
    *,
    limit: int = 30,
    fixture_test_mode: str | None = None,
) -> list[dict[str, Any]]:
    safe_limit = int(max(1, min(200, int(limit))))
    wanted_mode = str(fixture_test_mode or "").strip().lower()
    rows: list[dict[str, Any]] = []

    for path in sorted(LIVE_BENCHMARK_EXPORT_ROOT.glob("*.final-quality.latest.json"), reverse=True):
        record = _read_json_file(path)
        if not record:
            continue
        flat = _flatten_live_benchmark_record(record)
        mode = str(flat.get("fixture_test_mode") or "").strip().lower()
        if wanted_mode and mode != wanted_mode:
            continue
        rows.append(flat)
        if len(rows) >= safe_limit:
            break
    return rows
