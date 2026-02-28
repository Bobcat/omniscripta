from __future__ import annotations

import asyncio
import json
import secrets
import time
from asyncio.subprocess import PIPE
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from fastapi import HTTPException

from live_quality import load_fixture_reference, score_semilive_text_against_fixture
from speculative_quality import score_speculative_history_against_final, score_speculative_history_against_reference


_CONFIG: dict[str, Any] = {}
_RUNS: Dict[str, Dict[str, Any]] = {}
_TASKS: Dict[str, asyncio.Task[Any]] = {}


def configure(
    *,
    protocol_version: str,
    repo_root: Path,
    live_benchmark_export_root: Path,
    live_sessions: Any,
    rooted_path_cb: Any,
    autosave_snapshot_cb: Any,
    autosave_spec_trace_cb: Any,
    get_spec_timing_cb: Any,
    set_spec_timing_cb: Any,
    set_final_beam_override_cb: Any,
    live_audio_sample_width_bytes: int,
    live_audio_bytes_per_second: int,
    semilive_chunk_stop_wait_s: float,
    semilive_chunk_post_close_wait_s: float,
    ws_base_url: str,
) -> None:
    _CONFIG.update(
        {
            "protocol_version": str(protocol_version or "live_v1"),
            "repo_root": Path(repo_root).resolve(),
            "live_benchmark_export_root": Path(live_benchmark_export_root).resolve(),
            "live_sessions": live_sessions,
            "rooted_path_cb": rooted_path_cb,
            "autosave_snapshot_cb": autosave_snapshot_cb,
            "autosave_spec_trace_cb": autosave_spec_trace_cb,
            "get_spec_timing_cb": get_spec_timing_cb,
            "set_spec_timing_cb": set_spec_timing_cb,
            "set_final_beam_override_cb": set_final_beam_override_cb,
            "live_audio_sample_width_bytes": int(max(1, int(live_audio_sample_width_bytes))),
            "live_audio_bytes_per_second": int(max(1, int(live_audio_bytes_per_second))),
            "semilive_chunk_stop_wait_s": float(max(0.0, float(semilive_chunk_stop_wait_s))),
            "semilive_chunk_post_close_wait_s": float(max(0.0, float(semilive_chunk_post_close_wait_s))),
            "ws_base_url": str(ws_base_url or "").strip(),
        }
    )


def _cfg(name: str) -> Any:
    if name not in _CONFIG:
        raise RuntimeError(f"speculative_tuning_not_configured:{name}")
    return _CONFIG[name]


def _safe_filename(name: str) -> str:
    return Path(str(name or "")).name or "item"


def _iso_utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _new_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"specsweep_{ts}_{secrets.token_hex(4)}"


def _active_run_id() -> str:
    for run_id, run in _RUNS.items():
        if not isinstance(run, dict):
            continue
        status_val = str(run.get("status") or "").strip().lower()
        if status_val in {"queued", "running"}:
            return str(run_id)
    return ""


def _parse_int_list(
    raw: Any,
    *,
    default: list[int],
    min_value: int,
    max_value: int,
    field_name: str,
) -> list[int]:
    source: list[Any]
    if raw is None:
        source = list(default)
    elif isinstance(raw, str):
        source = [part.strip() for part in raw.split(",") if part.strip()]
    elif isinstance(raw, list):
        source = list(raw)
    else:
        raise HTTPException(status_code=400, detail=f"{field_name} must be a list of integers")

    out: list[int] = []
    seen: set[int] = set()
    for item in source:
        try:
            val = int(item)
        except Exception:
            raise HTTPException(status_code=400, detail=f"{field_name} contains a non-integer value")
        if val < min_value or val > max_value:
            raise HTTPException(status_code=400, detail=f"{field_name} values must be in range {min_value}..{max_value}")
        if val in seen:
            continue
        seen.add(val)
        out.append(int(val))
    if not out:
        raise HTTPException(status_code=400, detail=f"{field_name} is empty")
    return out


def _current_whisperx_beam_size() -> int:
    try:
        cfg_path = (_cfg("repo_root") / "config" / "whisperx.json").resolve()
        if cfg_path.exists():
            obj = json.loads(cfg_path.read_text(encoding="utf-8"))
            if isinstance(obj, dict) and obj.get("beam_size") is not None:
                return int(max(1, int(obj.get("beam_size"))))
    except Exception:
        pass
    return 5


def _normalize_combos(raw_combos: list[Any]) -> list[dict[str, int]]:
    out: list[dict[str, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for row in raw_combos:
        if not isinstance(row, dict):
            raise HTTPException(status_code=400, detail="combos must contain JSON objects")
        try:
            interval_ms = int(row.get("interval_ms"))
            window_ms = int(row.get("window_ms"))
            overlap_ms = int(row.get("overlap_ms"))
        except Exception:
            raise HTTPException(status_code=400, detail="each combo must include interval_ms, window_ms, overlap_ms integers")
        beam_size_raw = row.get("final_beam_size")
        if beam_size_raw is None:
            final_beam_size = int(max(1, _current_whisperx_beam_size()))
        else:
            try:
                final_beam_size = int(max(1, int(beam_size_raw)))
            except Exception:
                raise HTTPException(status_code=400, detail="final_beam_size must be an integer >=1")
        if interval_ms < 200 or interval_ms > 30000:
            raise HTTPException(status_code=400, detail="interval_ms out of range (200..30000)")
        if window_ms < 200 or window_ms > 60000:
            raise HTTPException(status_code=400, detail="window_ms out of range (200..60000)")
        if overlap_ms < 0 or overlap_ms > 20000:
            raise HTTPException(status_code=400, detail="overlap_ms out of range (0..20000)")
        if window_ms < interval_ms:
            raise HTTPException(status_code=400, detail="window_ms must be >= interval_ms")
        key = (int(interval_ms), int(window_ms), int(overlap_ms), int(final_beam_size))
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "interval_ms": int(interval_ms),
                "window_ms": int(window_ms),
                "overlap_ms": int(overlap_ms),
                "final_beam_size": int(final_beam_size),
            }
        )
    if not out:
        raise HTTPException(status_code=400, detail="combos is empty")
    return out


def _build_combos(*, interval_values: list[int], window_values: list[int], overlap_values: list[int]) -> list[dict[str, int]]:
    combos: list[dict[str, int]] = []
    for interval_ms in interval_values:
        for window_ms in window_values:
            if int(window_ms) < int(interval_ms):
                continue
            for overlap_ms in overlap_values:
                combos.append(
                    {
                        "interval_ms": int(interval_ms),
                        "window_ms": int(window_ms),
                        "overlap_ms": int(overlap_ms),
                    }
                )
    return _normalize_combos(combos)


def _persist_report(run_id: str) -> None:
    rid = _safe_filename(str(run_id or "").strip())
    if not rid:
        return
    run = _RUNS.get(rid)
    if not isinstance(run, dict):
        return
    envelope = {
        "protocol_version": _cfg("protocol_version"),
        "run_id": rid,
        "report": run,
    }
    _cfg("autosave_snapshot_cb")(
        session_id=rid,
        artifact_name="speculative-tuning",
        envelope=envelope,
        request_meta={"status": str(run.get("status") or "")},
    )


def _load_report_from_disk(run_id: str) -> Dict[str, Any] | None:
    rid = _safe_filename(str(run_id or "").strip())
    if not rid:
        return None
    p = (_cfg("live_benchmark_export_root") / f"{rid}.speculative-tuning.latest.json").resolve()
    if not p.exists():
        return None
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(rec, dict):
        return None
    payload = rec.get("payload")
    if not isinstance(payload, dict):
        return None
    report = payload.get("report")
    return dict(report) if isinstance(report, dict) else None


def _extract_sample_metrics(
    *,
    final_quality: dict[str, Any],
    speculative_quality: dict[str, Any],
    speculative_quality_vs_reference: dict[str, Any] | None,
) -> Dict[str, Any]:
    quality_obj = final_quality if isinstance(final_quality, dict) else {}
    score_obj = quality_obj.get("score") if isinstance(quality_obj.get("score"), dict) else {}
    run_metrics = quality_obj.get("run_metrics") if isinstance(quality_obj.get("run_metrics"), dict) else {}

    spec_obj = speculative_quality if isinstance(speculative_quality, dict) else {}
    spec_summary = spec_obj.get("summary") if isinstance(spec_obj.get("summary"), dict) else {}

    vs_ref_obj = speculative_quality_vs_reference if isinstance(speculative_quality_vs_reference, dict) else {}
    vs_ref_summary = vs_ref_obj.get("summary") if isinstance(vs_ref_obj.get("summary"), dict) else {}

    def _metric_mean(summary: dict[str, Any], key: str) -> float | None:
        row = summary.get(key) if isinstance(summary.get(key), dict) else {}
        if row.get("mean") is None:
            return None
        return round(_safe_float(row.get("mean"), 0.0), 6)

    return {
        "final_similarity_score": (
            int(score_obj.get("upload_similarity_score"))
            if score_obj.get("upload_similarity_score") is not None
            else None
        ),
        "final_word_edit_distance": (
            int(score_obj.get("word_edit_distance")) if score_obj.get("word_edit_distance") is not None else None
        ),
        "asr_combined_pipeline_pct_of_recording": (
            round(_safe_float(run_metrics.get("asr_combined_pipeline_pct_of_recording"), 0.0), 3)
            if run_metrics.get("asr_combined_pipeline_pct_of_recording") is not None
            else None
        ),
        "asr_combined_pipeline_time_total_s": (
            round(_safe_float(run_metrics.get("asr_combined_pipeline_time_total_s"), 0.0), 3)
            if run_metrics.get("asr_combined_pipeline_time_total_s") is not None
            else None
        ),
        "spec_windows_scored": (
            int(spec_summary.get("windows_scored")) if spec_summary.get("windows_scored") is not None else 0
        ),
        "spec_suffix_last_mean": _metric_mean(spec_summary, "suffix_last_word_similarity"),
        "spec_merged_last_mean": _metric_mean(spec_summary, "merged_last_word_similarity"),
        "spec_raw_last_mean": _metric_mean(spec_summary, "raw_last_word_similarity"),
        "spec_vs_ref_windows_aligned": (
            int(vs_ref_summary.get("windows_aligned")) if vs_ref_summary.get("windows_aligned") is not None else 0
        ),
        "spec_vs_ref_suffix_last_mean": _metric_mean(vs_ref_summary, "suffix_last_word_similarity"),
        "spec_vs_ref_merged_last_mean": _metric_mean(vs_ref_summary, "merged_last_word_similarity"),
        "spec_vs_ref_raw_last_mean": _metric_mean(vs_ref_summary, "raw_last_word_similarity"),
    }


def _aggregate_samples(samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metric_names = [
        "final_similarity_score",
        "asr_combined_pipeline_pct_of_recording",
        "spec_suffix_last_mean",
        "spec_merged_last_mean",
        "spec_raw_last_mean",
        "spec_vs_ref_suffix_last_mean",
        "spec_vs_ref_merged_last_mean",
        "spec_vs_ref_raw_last_mean",
    ]

    def _mean(values: list[float]) -> float | None:
        if not values:
            return None
        return round(sum(values) / float(len(values)), 6)

    buckets: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        cfg = sample.get("config") if isinstance(sample.get("config"), dict) else {}
        try:
            key = (
                int(cfg.get("interval_ms")),
                int(cfg.get("window_ms")),
                int(cfg.get("overlap_ms")),
                int(cfg.get("final_beam_size") or _current_whisperx_beam_size()),
            )
        except Exception:
            continue
        bucket = buckets.get(key)
        if bucket is None:
            bucket = {
                "config": {
                    "interval_ms": int(key[0]),
                    "window_ms": int(key[1]),
                    "overlap_ms": int(key[2]),
                    "final_beam_size": int(key[3]),
                },
                "samples_total": 0,
                "samples_ok": 0,
                "samples_error": 0,
                "metrics": {name: [] for name in metric_names},
            }
            buckets[key] = bucket
        bucket["samples_total"] = int(bucket["samples_total"] + 1)
        if str(sample.get("state") or "") == "ok":
            bucket["samples_ok"] = int(bucket["samples_ok"] + 1)
        else:
            bucket["samples_error"] = int(bucket["samples_error"] + 1)
        for name in metric_names:
            val = sample.get(name)
            if val is None:
                continue
            try:
                bucket["metrics"][name].append(float(val))
            except Exception:
                continue

    rows: list[dict[str, Any]] = []
    for _, bucket in buckets.items():
        metrics_avg: dict[str, float | None] = {}
        for name in metric_names:
            metrics_avg[f"{name}_avg"] = _mean(list(bucket["metrics"].get(name) or []))
        rows.append(
            {
                "config": dict(bucket["config"]),
                "samples_total": int(bucket["samples_total"]),
                "samples_ok": int(bucket["samples_ok"]),
                "samples_error": int(bucket["samples_error"]),
                **metrics_avg,
            }
        )

    def _metric_or_floor(row: dict[str, Any], key: str, floor: float = -1e9) -> float:
        if row.get(key) is None:
            return float(floor)
        return _safe_float(row.get(key), floor)

    rows.sort(
        key=lambda r: (
            -_metric_or_floor(r, "spec_vs_ref_merged_last_mean_avg"),
            -_metric_or_floor(r, "spec_merged_last_mean_avg"),
            -_metric_or_floor(r, "final_similarity_score_avg"),
        )
    )

    def _best_row(metric_name: str) -> dict[str, Any] | None:
        valid = [r for r in rows if r.get(metric_name) is not None]
        if not valid:
            return None
        return max(valid, key=lambda r: _safe_float(r.get(metric_name), -1.0))

    rankings: dict[str, Any] = {
        "best_spec_vs_ref_merged_last": _best_row("spec_vs_ref_merged_last_mean_avg"),
        "best_spec_merged_last": _best_row("spec_merged_last_mean_avg"),
        "best_final_similarity": _best_row("final_similarity_score_avg"),
    }
    return rows, rankings


async def _drain_ws_events(ws: Any, sink: list[dict[str, Any]]) -> None:
    while True:
        try:
            msg = await ws.recv()
        except Exception:
            return
        if isinstance(msg, bytes):
            continue
        try:
            obj = json.loads(str(msg))
        except Exception:
            continue
        if isinstance(obj, dict):
            sink.append(obj)


async def _run_fixture_audio_over_ws(*, session_id: str, pcm16le: bytes, frame_ms: int, realtime_factor: float) -> dict[str, Any]:
    try:
        import websockets  # type: ignore
    except Exception as e:
        raise RuntimeError(f"websockets_dependency_missing:{type(e).__name__}")

    base = str(_cfg("ws_base_url") or "").strip().rstrip("/")
    if not base or not (base.startswith("ws://") or base.startswith("wss://")):
        raise RuntimeError("invalid_TRANSCRIBE_LIVE_SWEEP_WS_BASE_URL")
    ws_url = f"{base}/demo/live/sessions/{_safe_filename(session_id)}/ws"

    sample_width = int(_cfg("live_audio_sample_width_bytes"))
    bytes_per_sec = int(_cfg("live_audio_bytes_per_second"))
    frame_ms_safe = int(max(20, min(500, int(frame_ms))))
    frame_bytes = int(max(sample_width, round((float(frame_ms_safe) / 1000.0) * bytes_per_sec)))
    if (frame_bytes % sample_width) != 0:
        frame_bytes += sample_width - (frame_bytes % sample_width)
    delay_s = float(frame_ms_safe) / 1000.0 / max(0.1, float(realtime_factor))

    events: list[dict[str, Any]] = []
    drain_task: asyncio.Task[Any] | None = None
    started_ts = time.time()
    frames_sent = 0
    bytes_sent = 0
    async with websockets.connect(
        ws_url,
        max_size=None,
        ping_interval=20.0,
        ping_timeout=20.0,
        close_timeout=30.0,
    ) as ws:
        drain_task = asyncio.create_task(_drain_ws_events(ws, events))
        await ws.send(json.dumps({"type": "start"}))
        for offset in range(0, len(pcm16le), frame_bytes):
            chunk = pcm16le[offset : offset + frame_bytes]
            if not chunk:
                continue
            if (len(chunk) % sample_width) != 0:
                chunk = chunk[: len(chunk) - (len(chunk) % sample_width)]
            if not chunk:
                continue
            await ws.send(chunk)
            frames_sent += 1
            bytes_sent += int(len(chunk))
            if delay_s > 0:
                await asyncio.sleep(delay_s)
        await asyncio.sleep(0.05)
        await ws.send(json.dumps({"type": "stop"}))
        wait_timeout_s = max(
            45.0,
            float(_cfg("semilive_chunk_stop_wait_s")) + float(_cfg("semilive_chunk_post_close_wait_s")) + 15.0,
        )
        try:
            await asyncio.wait_for(ws.wait_closed(), timeout=wait_timeout_s)
        except Exception:
            try:
                await ws.close()
            except Exception:
                pass
        if drain_task is not None:
            try:
                await asyncio.wait_for(drain_task, timeout=5.0)
            except Exception:
                drain_task.cancel()
    ended_ts = time.time()

    ended_events = [ev for ev in events if str(ev.get("type") or "") == "ended"]
    error_events = [ev for ev in events if str(ev.get("type") or "") == "error"]
    return {
        "ws_url": ws_url,
        "frames_sent": int(frames_sent),
        "bytes_sent": int(bytes_sent),
        "events_count": int(len(events)),
        "ended_events_count": int(len(ended_events)),
        "error_events_count": int(len(error_events)),
        "duration_s": round(max(0.0, ended_ts - started_ts), 3),
    }


async def _decode_audio_to_pcm16le_bytes(audio_path: Path) -> bytes:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(audio_path),
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        "16000",
        "pipe:1",
    ]
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=PIPE, stderr=PIPE)
    out, err = await proc.communicate()
    if proc.returncode != 0:
        detail = err.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg_decode_failed:{detail or proc.returncode}")
    pcm = bytes(out or b"")
    if not pcm:
        raise RuntimeError("ffmpeg_decode_empty_output")
    sample_width = int(_cfg("live_audio_sample_width_bytes"))
    rem = len(pcm) % sample_width
    if rem:
        pcm = pcm[: len(pcm) - rem]
    if not pcm:
        raise RuntimeError("ffmpeg_decode_invalid_alignment")
    return pcm


def _resolve_fixture_audio_path(fixture: Dict[str, Any]) -> Path:
    fixture_dir = Path(str(fixture.get("fixture_dir") or "")).resolve()
    meta = fixture.get("reference_meta") if isinstance(fixture.get("reference_meta"), dict) else {}
    clip_meta = meta.get("clip") if isinstance(meta.get("clip"), dict) else {}
    src_meta = meta.get("source") if isinstance(meta.get("source"), dict) else {}

    candidates_raw = [
        clip_meta.get("backend_clip_path"),
        src_meta.get("snippet_audio_path"),
        src_meta.get("orig_audio_upload_path"),
        clip_meta.get("frontend_clip_path"),
    ]
    for raw in candidates_raw:
        val = str(raw or "").strip()
        if not val:
            continue
        p = Path(val).expanduser()
        if not p.is_absolute():
            p = (fixture_dir / p).resolve()
        else:
            p = p.resolve()
        if p.is_file():
            return p

    for pattern in ("*.wav", "*.mp3", "*.m4a", "*.flac", "*.ogg", "*.opus"):
        for p in sorted(fixture_dir.glob(pattern)):
            if p.is_file():
                return p.resolve()
    raise FileNotFoundError(f"fixture_audio_missing:{fixture.get('fixture_id')}")


async def _wait_for_semilive_result_state(session_id: str, *, timeout_s: float = 180.0) -> tuple[dict[str, Any], bool]:
    deadline = time.monotonic() + max(1.0, float(timeout_s))
    last_result: dict[str, Any] = {}
    live_sessions = _cfg("live_sessions")
    while time.monotonic() < deadline:
        try:
            result = live_sessions.semilive_result_snapshot(session_id)
        except Exception:
            await asyncio.sleep(0.25)
            continue
        last_result = dict(result or {})
        state = str(last_result.get("finalization_state") or "").strip().lower()
        chunks_pending = int(max(0, int(last_result.get("chunks_pending") or 0)))
        if state in {"ready", "finalized"} and chunks_pending <= 0:
            return last_result, True
        if state in {"error"} and chunks_pending <= 0:
            return last_result, True
        await asyncio.sleep(0.25)
    return last_result, False


async def _run_sample(
    *,
    run_id: str,
    sample_index: int,
    repeat_index: int,
    config: dict[str, int],
    fixture_id: str,
    fixture_version: str,
    fixture_reference_text: str,
    pcm16le: bytes,
    stream_frame_ms: int,
    realtime_factor: float,
) -> dict[str, Any]:
    spec_defaults = dict(_cfg("get_spec_timing_cb")() or {})
    interval_ms = int(config.get("interval_ms") or spec_defaults.get("interval_ms") or 1800)
    window_ms = int(config.get("window_ms") or spec_defaults.get("window_ms") or 3000)
    overlap_ms = int(config.get("overlap_ms") or spec_defaults.get("overlap_ms") or 800)
    final_beam_size = int(config.get("final_beam_size") or _current_whisperx_beam_size())

    previous = _cfg("set_spec_timing_cb")(interval_ms=interval_ms, window_ms=window_ms, overlap_ms=overlap_ms)
    previous_beam = _cfg("set_final_beam_override_cb")(final_beam_size=final_beam_size)
    started_ts = time.time()
    session_id = ""
    live_sessions = _cfg("live_sessions")
    try:
        session = live_sessions.create_session()
        session_id = str(session.get("session_id") or "")
        if not session_id:
            raise RuntimeError("session_id_missing")
        live_sessions.set_fixture_metadata(
            session_id,
            fixture_id=fixture_id,
            fixture_version=fixture_version,
            fixture_test_mode="playback",
        )
        ws_run = await _run_fixture_audio_over_ws(
            session_id=session_id,
            pcm16le=pcm16le,
            frame_ms=stream_frame_ms,
            realtime_factor=realtime_factor,
        )
        semilive_result, ready = await _wait_for_semilive_result_state(session_id, timeout_s=240.0)
        if not ready:
            raise RuntimeError("semilive_result_timeout")

        final_quality = score_semilive_text_against_fixture(
            fixture_id=fixture_id,
            semilive_text=str(semilive_result.get("final_text") or ""),
            semilive_result=semilive_result,
            stats_log_path=live_sessions.stats_log_path(session_id),
        )
        speculative_history = live_sessions.semilive_speculative_history_snapshot(session_id)
        speculative_quality = score_speculative_history_against_final(
            speculative_history=speculative_history,
            verbose=False,
        )
        speculative_quality_vs_reference = score_speculative_history_against_reference(
            speculative_history=speculative_history,
            reference_text=fixture_reference_text,
            verbose=False,
        )

        final_envelope = {
            "protocol_version": _cfg("protocol_version"),
            "session_id": session_id,
            "fixture_id": fixture_id,
            "ready": True,
            "quality": final_quality,
        }
        _cfg("autosave_snapshot_cb")(
            session_id=session_id,
            artifact_name="final-quality",
            envelope=final_envelope,
        )
        _cfg("autosave_spec_trace_cb")(session_id)

        speculative_envelope = {
            "protocol_version": _cfg("protocol_version"),
            "session_id": session_id,
            "ready": True,
            "fixture_id": fixture_id,
            "speculative_quality": speculative_quality,
            "speculative_quality_vs_reference": speculative_quality_vs_reference,
        }
        _cfg("autosave_snapshot_cb")(
            session_id=session_id,
            artifact_name="speculative-quality",
            envelope=speculative_envelope,
            request_meta={"verbose": False, "fixture_id": fixture_id},
        )

        metrics = _extract_sample_metrics(
            final_quality=final_quality,
            speculative_quality=speculative_quality,
            speculative_quality_vs_reference=speculative_quality_vs_reference,
        )
        return {
            "run_id": str(run_id),
            "sample_index": int(sample_index),
            "repeat_index": int(repeat_index),
            "session_id": str(session_id),
            "state": "ok",
            "started_at_utc": _iso_utc(started_ts),
            "finished_at_utc": _iso_utc(time.time()),
            "duration_s": round(max(0.0, time.time() - started_ts), 3),
            "config": {
                "interval_ms": int(interval_ms),
                "window_ms": int(window_ms),
                "overlap_ms": int(overlap_ms),
                "final_beam_size": int(final_beam_size),
            },
            "finalization_state": str(semilive_result.get("finalization_state") or ""),
            "transcript_revision": int(max(0, int(semilive_result.get("transcript_revision") or 0))),
            "ws_run": ws_run,
            **metrics,
        }
    finally:
        if session_id:
            try:
                live_sessions.close_session(session_id, reason="speculative_tuning_sample_done")
            except Exception:
                pass
        _cfg("set_spec_timing_cb")(
            interval_ms=int(previous.get("interval_ms") or spec_defaults.get("interval_ms") or 1800),
            window_ms=int(previous.get("window_ms") or spec_defaults.get("window_ms") or 3000),
            overlap_ms=int(previous.get("overlap_ms") or spec_defaults.get("overlap_ms") or 800),
        )
        _cfg("set_final_beam_override_cb")(
            final_beam_size=(int(max(1, int(previous_beam))) if int(previous_beam) > 0 else None)
        )


async def _run_sweep(run_id: str) -> None:
    rid = _safe_filename(str(run_id or "").strip())
    run = _RUNS.get(rid)
    if not isinstance(run, dict):
        return

    run["status"] = "running"
    run["started_at_utc"] = _iso_utc(time.time())
    _persist_report(rid)

    try:
        req = run.get("request") if isinstance(run.get("request"), dict) else {}
        fixture_id = str(req.get("fixture_id") or "").strip()
        repeats = int(max(1, int(req.get("repeats") or 1)))
        combos = [dict(c) for c in (req.get("combos") or []) if isinstance(c, dict)]
        stream_frame_ms = int(max(20, int(req.get("stream_frame_ms") or 40)))
        realtime_factor = float(max(0.1, _safe_float(req.get("realtime_factor"), 1.0)))
        max_audio_ms = int(max(0, int(req.get("max_audio_ms") or 0)))

        fixture = load_fixture_reference(fixture_id)
        fixture_version = str((fixture.get("reference_meta") or {}).get("version") or "").strip()
        audio_path = _resolve_fixture_audio_path(fixture)
        pcm16le = await _decode_audio_to_pcm16le_bytes(audio_path)
        if max_audio_ms > 0:
            max_bytes = int(round((float(max_audio_ms) / 1000.0) * _cfg("live_audio_bytes_per_second")))
            if max_bytes > 0:
                sample_width = int(_cfg("live_audio_sample_width_bytes"))
                if (max_bytes % sample_width) != 0:
                    max_bytes -= max_bytes % sample_width
                pcm16le = pcm16le[:max_bytes] if max_bytes < len(pcm16le) else pcm16le
        if not pcm16le:
            raise RuntimeError("fixture_audio_decode_empty")

        run["fixture_audio"] = {
            "path": str(audio_path),
            "bytes": int(len(pcm16le)),
            "duration_ms": int(round((float(len(pcm16le)) / float(_cfg("live_audio_bytes_per_second"))) * 1000.0)),
            "stream_frame_ms": int(stream_frame_ms),
            "realtime_factor": float(realtime_factor),
        }
        _persist_report(rid)

        sample_index = 0
        for cfg in combos:
            for repeat_index in range(repeats):
                sample_index += 1
                run["progress"]["current_sample_index"] = int(sample_index)
                _persist_report(rid)
                started_ts = time.time()
                try:
                    sample = await _run_sample(
                        run_id=rid,
                        sample_index=int(sample_index),
                        repeat_index=int(repeat_index),
                        config=cfg,
                        fixture_id=fixture_id,
                        fixture_version=fixture_version,
                        fixture_reference_text=str(fixture.get("reference_text") or ""),
                        pcm16le=pcm16le,
                        stream_frame_ms=stream_frame_ms,
                        realtime_factor=realtime_factor,
                    )
                    run["samples"].append(sample)
                except Exception as e:
                    run["samples"].append(
                        {
                            "run_id": str(rid),
                            "sample_index": int(sample_index),
                            "repeat_index": int(repeat_index),
                            "state": "error",
                            "config": dict(cfg),
                            "error": f"{type(e).__name__}: {e}",
                            "duration_s": round(max(0.0, time.time() - started_ts), 3),
                            "finished_at_utc": _iso_utc(time.time()),
                        }
                    )
                    run["errors"].append(
                        {
                            "sample_index": int(sample_index),
                            "repeat_index": int(repeat_index),
                            "config": dict(cfg),
                            "error": f"{type(e).__name__}: {e}",
                        }
                    )
                run["progress"]["samples_completed"] = int(len(run["samples"]))
                run["progress"]["samples_failed"] = int(
                    len([s for s in run["samples"] if str(s.get("state") or "") != "ok"])
                )
                aggregate_rows, rankings = _aggregate_samples(run["samples"])
                run["aggregate_by_config"] = aggregate_rows
                run["rankings"] = rankings
                _persist_report(rid)

        failed_count = int(run["progress"].get("samples_failed") or 0)
        run["status"] = "completed_with_errors" if failed_count > 0 else "completed"
    except Exception as e:
        run["status"] = "failed"
        run["errors"].append({"scope": "run", "error": f"{type(e).__name__}: {e}"})
    finally:
        run["finished_at_utc"] = _iso_utc(time.time())
        run["progress"]["current_sample_index"] = int(run["progress"].get("samples_completed") or 0)
        _persist_report(rid)


async def start_run(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="payload must be a JSON object")
    fixture_id = str(payload.get("fixture_id") or "").strip()
    if not fixture_id:
        raise HTTPException(status_code=400, detail="fixture_id is required")

    repeats = _safe_int(payload.get("repeats"), 1)
    repeats = int(max(1, repeats))
    if repeats > 20:
        raise HTTPException(status_code=400, detail="repeats out of range (1..20)")

    explicit_combos = payload.get("combos")
    if explicit_combos is not None and not isinstance(explicit_combos, list):
        raise HTTPException(status_code=400, detail="combos must be a list of objects")

    defaults = dict(_cfg("get_spec_timing_cb")() or {})
    default_interval = int(max(200, int(defaults.get("interval_ms") or 1800)))
    default_window = int(max(default_interval, int(defaults.get("window_ms") or 3000)))
    default_overlap = int(max(0, int(defaults.get("overlap_ms") or 800)))

    if isinstance(explicit_combos, list) and explicit_combos:
        combos = _normalize_combos(explicit_combos)
    else:
        interval_vals = _parse_int_list(
            payload.get("interval_ms_values"),
            default=[default_interval],
            min_value=200,
            max_value=30000,
            field_name="interval_ms_values",
        )
        window_vals = _parse_int_list(
            payload.get("window_ms_values"),
            default=[default_window],
            min_value=200,
            max_value=60000,
            field_name="window_ms_values",
        )
        overlap_vals = _parse_int_list(
            payload.get("overlap_ms_values"),
            default=[default_overlap],
            min_value=0,
            max_value=20000,
            field_name="overlap_ms_values",
        )
        combos = _build_combos(
            interval_values=interval_vals,
            window_values=window_vals,
            overlap_values=overlap_vals,
        )

    beam_default = int(max(1, _current_whisperx_beam_size()))
    beam_vals = _parse_int_list(
        payload.get("final_beam_size_values"),
        default=[beam_default],
        min_value=1,
        max_value=20,
        field_name="final_beam_size_values",
    )
    combos_with_beam: list[dict[str, int]] = []
    for cfg in combos:
        for beam_size in beam_vals:
            row = dict(cfg)
            row["final_beam_size"] = int(max(1, int(beam_size)))
            combos_with_beam.append(row)
    combos = combos_with_beam

    if not combos:
        raise HTTPException(status_code=400, detail="no valid speculative timing combinations")

    combos_count = int(len(combos))
    sample_total = int(combos_count * repeats)
    if sample_total > 200:
        raise HTTPException(status_code=400, detail="too many samples requested (max 200)")

    stream_frame_ms = _safe_int(payload.get("stream_frame_ms"), 40)
    stream_frame_ms = int(max(20, min(500, stream_frame_ms)))
    realtime_factor = _safe_float(payload.get("realtime_factor"), 1.0)
    realtime_factor = float(max(0.1, min(20.0, realtime_factor)))
    max_audio_ms = _safe_int(payload.get("max_audio_ms"), 0)
    max_audio_ms = int(max(0, max_audio_ms))

    active_run_id = _active_run_id()
    if active_run_id:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "speculative_tuning_already_running",
                "run_id": active_run_id,
                "message": "A speculative tuning run is already active.",
            },
        )

    run_id = _new_run_id()
    now_ts = time.time()
    run: Dict[str, Any] = {
        "protocol_version": _cfg("protocol_version"),
        "run_id": run_id,
        "status": "queued",
        "created_at_utc": _iso_utc(now_ts),
        "created_at_unix": round(float(now_ts), 6),
        "started_at_utc": "",
        "finished_at_utc": "",
        "request": {
            "fixture_id": fixture_id,
            "repeats": int(repeats),
            "combos_count": int(combos_count),
            "samples_total": int(sample_total),
            "stream_frame_ms": int(stream_frame_ms),
            "realtime_factor": float(realtime_factor),
            "max_audio_ms": int(max_audio_ms),
            "final_beam_size_values": [int(v) for v in beam_vals],
            "combos": [dict(c) for c in combos],
        },
        "progress": {
            "samples_total": int(sample_total),
            "samples_completed": 0,
            "samples_failed": 0,
            "current_sample_index": 0,
        },
        "samples": [],
        "aggregate_by_config": [],
        "rankings": {},
        "errors": [],
    }
    _RUNS[run_id] = run
    _persist_report(run_id)

    task = asyncio.create_task(_run_sweep(run_id))
    _TASKS[run_id] = task
    task.add_done_callback(lambda _task, rid=run_id: _TASKS.pop(rid, None))

    return {
        "protocol_version": _cfg("protocol_version"),
        "run_id": run_id,
        "status": "queued",
        "samples_total": int(sample_total),
        "combos_count": int(combos_count),
        "repeats": int(repeats),
        "report_url": _cfg("rooted_path_cb")(f"/demo/live/speculative-tuning/report/{run_id}"),
    }


def get_report(run_id: str) -> dict[str, Any]:
    rid = _safe_filename(str(run_id or "").strip())
    if not rid:
        raise HTTPException(status_code=400, detail="run_id is required")
    report = _RUNS.get(rid)
    if report is None:
        report = _load_report_from_disk(rid)
    if report is None:
        raise HTTPException(status_code=404, detail="speculative tuning run not found")
    return {
        "protocol_version": _cfg("protocol_version"),
        "run_id": rid,
        "report": report,
    }

