#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORT_ROOT = (REPO_ROOT / "data" / "live" / "benchmark_exports").resolve()


def _load_export(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _flatten_row(record: dict[str, Any]) -> dict[str, Any]:
    payload = dict(record.get("payload") or {})
    quality = dict(payload.get("quality") or {})
    score = dict(quality.get("score") or {})
    run = dict(quality.get("run_metrics") or {})
    request_meta = dict(record.get("request_meta") or {})
    tuning = dict(request_meta.get("live_tuning_snapshot") or {})
    asr = dict(tuning.get("asr") or {})
    timing = dict(tuning.get("timing") or {})
    rolling = dict(tuning.get("rolling") or {})
    vad = dict(rolling.get("vad") or {})
    speech_gate = dict(rolling.get("speech_gate") or {})

    reason_counts = dict(run.get("chunk_reason_counts") or {})
    reasons = ",".join(f"{key}={reason_counts[key]}" for key in sorted(reason_counts))

    recording_duration_ms = int(run.get("recording_duration_ms") or 0)
    return {
        "session_id": str(record.get("session_id") or payload.get("session_id") or ""),
        "saved_at_utc": str(record.get("saved_at_utc") or ""),
        "mode": str(request_meta.get("fixture_test_mode") or ""),
        "score": score.get("upload_similarity_score"),
        "revisions": int(run.get("transcript_revision") or 0),
        "chunks": int(run.get("chunks_total") or 0),
        "recording_s": round(recording_duration_ms / 1000.0, 2) if recording_duration_ms > 0 else "",
        "cs": asr.get("chunk_size"),
        "beam": asr.get("beam_size"),
        "backend": asr.get("backend"),
        "min_infer_ms": rolling.get("min_infer_audio_ms"),
        "min_new_ms": rolling.get("min_new_audio_ms"),
        "max_decode_ms": rolling.get("max_decode_window_ms"),
        "single_commit_ms": rolling.get("single_segment_commit_min_ms"),
        "force_repeats": rolling.get("force_commit_repeats"),
        "emit_min_ms": timing.get("emit_min_ms"),
        "vad_enabled": vad.get("enabled"),
        "vad_threshold": vad.get("threshold"),
        "vad_min_speech_ms": vad.get("min_speech_ms"),
        "gate_silence_ms": speech_gate.get("silence_enter_ms"),
        "gate_force_commit_ms": speech_gate.get("force_commit_silence_ms"),
        "reasons": reasons,
        "path": str(record.get("_path") or ""),
    }


def _iter_latest_exports() -> list[Path]:
    return sorted(EXPORT_ROOT.glob("*.final-quality.latest.json"))


def _pick_paths(session_ids: list[str], latest: int) -> list[Path]:
    paths = _iter_latest_exports()
    if session_ids:
        wanted = {sid.strip() for sid in session_ids if sid.strip()}
        return [path for path in paths if path.name.split(".final-quality.latest.json")[0] in wanted]
    if latest > 0:
        return paths[-latest:]
    return paths


def _print_tsv(rows: list[dict[str, Any]]) -> None:
    columns = [
        "session_id",
        "saved_at_utc",
        "mode",
        "score",
        "revisions",
        "chunks",
        "recording_s",
        "cs",
        "beam",
        "backend",
        "min_infer_ms",
        "min_new_ms",
        "max_decode_ms",
        "single_commit_ms",
        "force_repeats",
        "emit_min_ms",
        "vad_enabled",
        "vad_threshold",
        "vad_min_speech_ms",
        "gate_silence_ms",
        "gate_force_commit_ms",
        "reasons",
    ]
    print("\t".join(columns))
    for row in rows:
        print("\t".join("" if row.get(col) is None else str(row.get(col)) for col in columns))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print a compact TSV matrix for live benchmark exports.",
    )
    parser.add_argument("session_ids", nargs="*", help="Optional session ids to compare")
    parser.add_argument("--latest", type=int, default=12, help="Use the last N exports when no session ids are given")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of TSV")
    args = parser.parse_args()

    paths = _pick_paths(args.session_ids, args.latest)
    rows: list[dict[str, Any]] = []
    for path in paths:
        record = _load_export(path)
        record["_path"] = str(path)
        rows.append(_flatten_row(record))

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        _print_tsv(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
