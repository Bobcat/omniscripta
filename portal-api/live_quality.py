from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    # portal-api/live_quality.py -> portal-api -> repo root
    return Path(__file__).resolve().parents[1]


FIXTURES_ROOT = (_repo_root() / "data" / "test_fixtures").resolve()

_SPEAKER_LABEL_RE = re.compile(r"\bspeaker[_ ]?\d+\s*:\s*", re.IGNORECASE)
_NON_WORD_RE = re.compile(r"[^a-z0-9\s]+", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


def normalize_transcript_text(text: str) -> str:
    s = str(text or "").strip().lower()
    if not s:
        return ""
    s = _SPEAKER_LABEL_RE.sub(" ", s)
    s = _NON_WORD_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def _tokenize_words(text: str) -> list[str]:
    normalized = normalize_transcript_text(text)
    return normalized.split(" ") if normalized else []


def _word_levenshtein(a_words: list[str], b_words: list[str]) -> int:
    if a_words == b_words:
        return 0
    if not a_words:
        return len(b_words)
    if not b_words:
        return len(a_words)

    # Keep the inner DP row small.
    if len(a_words) < len(b_words):
        a_words, b_words = b_words, a_words

    prev = list(range(len(b_words) + 1))
    for i, aw in enumerate(a_words, start=1):
        curr = [i]
        for j, bw in enumerate(b_words, start=1):
            cost = 0 if aw == bw else 1
            curr.append(
                min(
                    prev[j] + 1,       # deletion
                    curr[j - 1] + 1,   # insertion
                    prev[j - 1] + cost,  # substitution
                )
            )
        prev = curr
    return int(prev[-1])


def _parse_iso_utc(value: str) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except Exception:
        return None


def _stats_log_metrics(stats_log_path: str | Path) -> dict[str, Any]:
    path = Path(stats_log_path)
    out: dict[str, Any] = {
        "stop_to_ready_ms": None,
        "poll_error_count": 0,
        "chunk_error_count": 0,
    }
    if not path.exists():
        return out

    finalized_ts: datetime | None = None
    session_closed_ts: datetime | None = None
    poll_error_count = 0
    chunk_error_count = 0

    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    row = json.loads(raw)
                except Exception:
                    continue
                kind = str(row.get("kind") or row.get("type") or "").strip()
                if kind == "rolling_recording_finalized" and finalized_ts is None:
                    finalized_ts = _parse_iso_utc(str(row.get("ts_utc") or ""))
                elif kind == "session_closed":
                    session_closed_ts = _parse_iso_utc(str(row.get("ts_utc") or ""))
                elif kind == "rolling_inference_poll_error":
                    poll_error_count += 1
                elif kind in {
                    "rolling_inference_error",
                    "rolling_inference_enqueue_error",
                    "rolling_commit_store_error",
                    "rolling_context_init_error",
                    "rolling_recording_append_error",
                }:
                    chunk_error_count += 1
    except Exception:
        return out

    stop_to_ready_ms: int | None = None
    if finalized_ts is not None and session_closed_ts is not None:
        try:
            delta_ms = int(round((session_closed_ts - finalized_ts).total_seconds() * 1000.0))
            stop_to_ready_ms = max(0, delta_ms)
        except Exception:
            stop_to_ready_ms = None

    out["stop_to_ready_ms"] = stop_to_ready_ms
    out["poll_error_count"] = int(max(0, poll_error_count))
    out["chunk_error_count"] = int(max(0, chunk_error_count))
    return out


def _fixture_dir(fixture_id: str) -> Path:
    safe = str(fixture_id or "").strip()
    if not safe:
        raise FileNotFoundError("fixture_id_missing")
    # Convention-based lookup keeps the registry small for now.
    candidate = (FIXTURES_ROOT / safe).resolve()
    try:
        candidate.relative_to(FIXTURES_ROOT)
    except Exception:
        # Treat traversal/escape attempts the same as unknown fixtures.
        raise FileNotFoundError(f"fixture_not_found:{fixture_id}")
    return candidate


def load_fixture_reference(fixture_id: str) -> dict[str, Any]:
    fixture_dir = _fixture_dir(fixture_id)
    if not fixture_dir.exists():
        raise FileNotFoundError(f"fixture_not_found:{fixture_id}")

    ref_txt_path = fixture_dir / "reference.txt"
    if not ref_txt_path.exists():
        raise FileNotFoundError(f"fixture_reference_missing:{fixture_id}")

    ref_meta_path = fixture_dir / "reference_meta.json"
    meta: dict[str, Any] = {}
    if ref_meta_path.exists():
        try:
            meta_obj = json.loads(ref_meta_path.read_text(encoding="utf-8"))
            if isinstance(meta_obj, dict):
                meta = meta_obj
        except Exception:
            meta = {}

    ref_text = ref_txt_path.read_text(encoding="utf-8")
    return {
        "fixture_id": str(fixture_id),
        "fixture_dir": str(fixture_dir),
        "reference_txt_path": str(ref_txt_path),
        "reference_meta_path": str(ref_meta_path) if ref_meta_path.exists() else "",
        "reference_text": ref_text,
        "reference_meta": meta,
    }


def score_live_text_against_fixture(
    *,
    fixture_id: str,
    live_text: str,
    live_result: dict[str, Any] | None = None,
    stats_log_path: str | Path | None = None,
) -> dict[str, Any]:
    fixture = load_fixture_reference(fixture_id)
    ref_text = str(fixture.get("reference_text") or "")
    live_text = str(live_text or "")

    ref_norm = normalize_transcript_text(ref_text)
    live_norm = normalize_transcript_text(live_text)
    ref_words = ref_norm.split(" ") if ref_norm else []
    live_words = live_norm.split(" ") if live_norm else []

    dist = _word_levenshtein(live_words, ref_words)
    denom = max(1, len(ref_words))
    similarity = max(0.0, 1.0 - (float(dist) / float(denom)))
    score_100 = int(round(similarity * 100.0))
    score_100 = max(0, min(100, score_100))

    word_count_ratio: float | None = None
    if ref_words:
        word_count_ratio = round(float(len(live_words)) / float(len(ref_words)), 4)

    result = live_result if isinstance(live_result, dict) else {}
    chunk_rows = result.get("chunk_results") if isinstance(result.get("chunk_results"), list) else []
    asr_transcribe_time_total_s = 0.0
    asr_pipeline_time_total_s = 0.0
    for row in chunk_rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("state") or "") != "ready":
            continue
        try:
            if row.get("asr_transcribe_time_s") is not None:
                asr_transcribe_time_total_s += max(0.0, float(row.get("asr_transcribe_time_s")))
        except Exception:
            pass
        try:
            if row.get("asr_pipeline_time_s") is not None:
                asr_pipeline_time_total_s += max(0.0, float(row.get("asr_pipeline_time_s")))
        except Exception:
            pass

    recording_duration_ms = int(max(0, int(result.get("recording_duration_ms") or 0)))
    recording_s = float(recording_duration_ms) / 1000.0 if recording_duration_ms > 0 else 0.0
    asr_transcribe_pct_of_recording: float | None = None
    asr_pipeline_pct_of_recording: float | None = None
    if recording_s > 0:
        asr_transcribe_pct_of_recording = round((asr_transcribe_time_total_s / recording_s) * 100.0, 1)
        asr_pipeline_pct_of_recording = round((asr_pipeline_time_total_s / recording_s) * 100.0, 1)

    run_metrics: dict[str, Any] = {
        "finalization_state": str(result.get("finalization_state") or ""),
        "recording_duration_ms": recording_duration_ms,
        "chunks_total": int(max(0, int(result.get("chunks_total") or 0))),
        "chunks_done": int(max(0, int(result.get("chunks_done") or 0))),
        "chunks_failed": int(max(0, int(result.get("chunks_failed") or 0))),
        "chunks_pending": int(max(0, int(result.get("chunks_pending") or 0))),
        "chunk_reason_counts": (
            dict(result.get("chunk_reason_counts"))
            if isinstance(result.get("chunk_reason_counts"), dict)
            else {}
        ),
        "transcript_revision": int(max(0, int(result.get("transcript_revision") or 0))),
        "final_text_chars": len(live_text),
        "final_segments_count": int(max(0, int(result.get("final_segments_count") or 0))),
        "asr_transcribe_time_total_s": round(asr_transcribe_time_total_s, 3),
        "asr_pipeline_time_total_s": round(asr_pipeline_time_total_s, 3),
        "asr_transcribe_pct_of_recording": asr_transcribe_pct_of_recording,
        "asr_pipeline_pct_of_recording": asr_pipeline_pct_of_recording,
    }
    if stats_log_path:
        run_metrics.update(_stats_log_metrics(stats_log_path))

    return {
        "metric_version": "live_quality_v1",
        "fixture": {
            "fixture_id": str(fixture_id),
            "reference_txt_path": str(fixture.get("reference_txt_path") or ""),
            "reference_meta": fixture.get("reference_meta") if isinstance(fixture.get("reference_meta"), dict) else {},
        },
        "score": {
            "upload_similarity_score": int(score_100),
            "score_basis": "word_levenshtein_similarity_vs_fixture_reference",
            "word_edit_distance": int(max(0, dist)),
            "word_count_live": int(len(live_words)),
            "word_count_reference": int(len(ref_words)),
            "word_count_ratio_live_to_ref": word_count_ratio,
            "char_count_live": int(len(live_text)),
            "char_count_reference": int(len(ref_text)),
            "normalized_char_count_live": int(len(live_norm)),
            "normalized_char_count_reference": int(len(ref_norm)),
        },
        "run_metrics": run_metrics,
    }
