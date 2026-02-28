from __future__ import annotations

import asyncio
import json
import secrets
import time
from asyncio.subprocess import PIPE
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

from fastapi import HTTPException

from live_quality import load_fixture_reference, score_semilive_text_against_fixture


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
    ws_base_url: str,
    live_audio_sample_width_bytes: int,
    live_audio_bytes_per_second: int,
    semilive_chunk_stop_wait_s: float,
    semilive_chunk_post_close_wait_s: float,
    asr_pool_status_url: str,
    asr_pool_token: str,
) -> None:
    _CONFIG.update(
        {
            "protocol_version": str(protocol_version or "live_v1"),
            "repo_root": Path(repo_root).resolve(),
            "live_benchmark_export_root": Path(live_benchmark_export_root).resolve(),
            "live_sessions": live_sessions,
            "rooted_path_cb": rooted_path_cb,
            "autosave_snapshot_cb": autosave_snapshot_cb,
            "ws_base_url": str(ws_base_url or "").strip(),
            "live_audio_sample_width_bytes": int(max(1, int(live_audio_sample_width_bytes))),
            "live_audio_bytes_per_second": int(max(1, int(live_audio_bytes_per_second))),
            "semilive_chunk_stop_wait_s": float(max(0.0, float(semilive_chunk_stop_wait_s))),
            "semilive_chunk_post_close_wait_s": float(max(0.0, float(semilive_chunk_post_close_wait_s))),
            "asr_pool_status_url": str(asr_pool_status_url or "").strip(),
            "asr_pool_token": str(asr_pool_token or "").strip(),
        }
    )


def _cfg(name: str) -> Any:
    if name not in _CONFIG:
        raise RuntimeError(f"asr_loadtest_not_configured:{name}")
    return _CONFIG[name]


def _safe_filename(name: str) -> str:
    return Path(str(name or "")).name or "item"


def _iso_utc(ts: float) -> str:
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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
    return f"asrload_{ts}_{secrets.token_hex(4)}"


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


def _parse_iso_to_unix(value: Any) -> float | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return float(datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp())
    except Exception:
        return None


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return round(float(values[0]), 3)
    p = max(0.0, min(1.0, float(q)))
    s = sorted(float(v) for v in values)
    pos = (len(s) - 1) * p
    lo = int(pos)
    hi = min(len(s) - 1, lo + 1)
    frac = pos - lo
    val = s[lo] if hi == lo else (s[lo] + ((s[hi] - s[lo]) * frac))
    return round(float(val), 3)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(float(sum(values) / float(len(values))), 6)


def _wait_summary(wait_values_ms: list[float]) -> dict[str, Any]:
    vals = [float(v) for v in wait_values_ms if v is not None]
    return {
        "n": int(len(vals)),
        "mean": _mean(vals),
        "p50": _percentile(vals, 0.50),
        "p95": _percentile(vals, 0.95),
        "p99": _percentile(vals, 0.99),
    }


def _http_json(
    *,
    method: str,
    url: str,
    token: str,
    timeout_s: float,
    payload: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    data = None
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urlrequest.Request(url, data=data, method=str(method).upper())
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("X-ASR-Token", token)
    try:
        with urlrequest.urlopen(req, timeout=float(timeout_s)) as resp:
            raw = resp.read()
            status_code = int(getattr(resp, "status", 200) or 200)
    except urlerror.HTTPError as e:
        raw = e.read()
        status_code = int(getattr(e, "code", 500) or 500)
    except Exception:
        return 0, {}
    if not raw:
        return status_code, {}
    try:
        obj = json.loads(raw.decode("utf-8", errors="replace"))
    except Exception:
        return status_code, {}
    return status_code, dict(obj) if isinstance(obj, dict) else {}


def _try_pool_status_snapshot() -> dict[str, Any]:
    status_url = str(_cfg("asr_pool_status_url") or "").strip()
    if not status_url:
        return {}
    token = str(_cfg("asr_pool_token") or "").strip()
    code, body = _http_json(
        method="GET",
        url=status_url,
        token=token,
        timeout_s=5.0,
        payload=None,
    )
    if code != 200 or not isinstance(body, dict):
        return {}
    return body


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
        artifact_name="asr-loadtest",
        envelope=envelope,
        request_meta={"status": str(run.get("status") or "")},
    )


def _load_report_from_disk(run_id: str) -> Dict[str, Any] | None:
    rid = _safe_filename(str(run_id or "").strip())
    if not rid:
        return None
    p = (_cfg("live_benchmark_export_root") / f"{rid}.asr-loadtest.latest.json").resolve()
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
        if state in {"ready", "finalized", "recording_finalized"} and chunks_pending <= 0:
            return last_result, True
        if state in {"error"} and chunks_pending <= 0:
            return last_result, True
        await asyncio.sleep(0.25)
    return last_result, False


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                s = str(line or "").strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    out.append(obj)
    except Exception:
        return []
    return out


def _extract_session_lane_metrics(stats_log_path: Path) -> dict[str, Any]:
    rows = _read_jsonl(stats_log_path)
    final_enqueued_ts: dict[int, float] = {}
    spec_enqueued_ts: dict[int, float] = {}
    final_waits: list[float] = []
    spec_waits: list[float] = []
    final_submit_attempts: list[int] = []
    spec_submit_attempts: list[int] = []
    final_blob_fetch_ms: list[float] = []
    spec_blob_fetch_ms: list[float] = []
    final_enqueued_count = 0
    spec_enqueued_count = 0
    final_ready_count = 0
    spec_ready_count = 0
    final_error_count = 0
    spec_error_count = 0

    for row in rows:
        kind = str(row.get("kind") or row.get("type") or "").strip()
        ts_unix = _parse_iso_to_unix(row.get("ts_utc"))
        if kind == "semilive_chunk_enqueued":
            chunk = row.get("chunk") if isinstance(row.get("chunk"), dict) else {}
            idx = _safe_int(chunk.get("chunk_index"), -1)
            if idx >= 0 and ts_unix is not None:
                final_enqueued_ts[int(idx)] = float(ts_unix)
            final_enqueued_count += 1
            continue
        if kind == "semilive_chunk_ready":
            idx = _safe_int(row.get("chunk_index"), -1)
            final_ready_count += 1
            if idx >= 0 and idx in final_enqueued_ts and ts_unix is not None:
                wait_ms = max(0.0, (float(ts_unix) - float(final_enqueued_ts[idx])) * 1000.0)
                final_waits.append(float(wait_ms))
            submit_attempts = row.get("remote_submit_attempts")
            if submit_attempts is not None:
                try:
                    final_submit_attempts.append(int(max(1, int(submit_attempts))))
                except Exception:
                    pass
            if row.get("blob_fetch_ms") is not None:
                try:
                    final_blob_fetch_ms.append(max(0.0, float(row.get("blob_fetch_ms"))))
                except Exception:
                    pass
            continue
        if kind == "semilive_chunk_error":
            final_error_count += 1
            continue
        if kind == "semilive_speculative_enqueued":
            seq = _safe_int(row.get("speculative_seq"), -1)
            if seq >= 0 and ts_unix is not None:
                spec_enqueued_ts[int(seq)] = float(ts_unix)
            spec_enqueued_count += 1
            continue
        if kind == "semilive_speculative_ready":
            seq = _safe_int(row.get("speculative_seq"), -1)
            spec_ready_count += 1
            if seq >= 0 and seq in spec_enqueued_ts and ts_unix is not None:
                wait_ms = max(0.0, (float(ts_unix) - float(spec_enqueued_ts[seq])) * 1000.0)
                spec_waits.append(float(wait_ms))
            submit_attempts = row.get("remote_submit_attempts")
            if submit_attempts is not None:
                try:
                    spec_submit_attempts.append(int(max(1, int(submit_attempts))))
                except Exception:
                    pass
            if row.get("blob_fetch_ms") is not None:
                try:
                    spec_blob_fetch_ms.append(max(0.0, float(row.get("blob_fetch_ms"))))
                except Exception:
                    pass
            continue
        if kind == "semilive_speculative_error":
            spec_error_count += 1
            continue

    final_retry_rate = None
    if final_submit_attempts:
        retries = sum(1 for n in final_submit_attempts if int(n) > 1)
        final_retry_rate = round(float(retries) / float(len(final_submit_attempts)), 6)
    spec_retry_rate = None
    if spec_submit_attempts:
        retries = sum(1 for n in spec_submit_attempts if int(n) > 1)
        spec_retry_rate = round(float(retries) / float(len(spec_submit_attempts)), 6)

    return {
        "final_wait_ms_values": [round(float(v), 3) for v in final_waits],
        "spec_wait_ms_values": [round(float(v), 3) for v in spec_waits],
        "final_wait_ms": _wait_summary(final_waits),
        "spec_wait_ms": _wait_summary(spec_waits),
        "interactive_wait_ms": _wait_summary(final_waits + spec_waits),
        "final_counts": {
            "enqueued": int(final_enqueued_count),
            "ready": int(final_ready_count),
            "errors": int(final_error_count),
            "submit_attempts_sampled": int(len(final_submit_attempts)),
        },
        "spec_counts": {
            "enqueued": int(spec_enqueued_count),
            "ready": int(spec_ready_count),
            "errors": int(spec_error_count),
            "submit_attempts_sampled": int(len(spec_submit_attempts)),
        },
        "final_error_rate": (
            round(float(final_error_count) / float(final_enqueued_count), 6) if final_enqueued_count > 0 else None
        ),
        "spec_error_rate": (
            round(float(spec_error_count) / float(spec_enqueued_count), 6) if spec_enqueued_count > 0 else None
        ),
        "final_retry_rate": final_retry_rate,
        "spec_retry_rate": spec_retry_rate,
        "blob_fetch_ms": _wait_summary(final_blob_fetch_ms + spec_blob_fetch_ms),
    }


def _aggregate_sample(sample: dict[str, Any]) -> dict[str, Any]:
    sessions = [dict(s) for s in (sample.get("sessions") or []) if isinstance(s, dict)]
    final_waits: list[float] = []
    spec_waits: list[float] = []
    blob_fetch_vals: list[float] = []
    final_scores: list[float] = []
    final_error_num = 0
    final_error_den = 0
    spec_error_num = 0
    spec_error_den = 0
    final_retry_num = 0
    final_retry_den = 0
    spec_retry_num = 0
    spec_retry_den = 0

    for sess in sessions:
        m = sess.get("lane_metrics") if isinstance(sess.get("lane_metrics"), dict) else {}
        for v in m.get("final_wait_ms_values") or []:
            try:
                final_waits.append(float(v))
            except Exception:
                pass
        for v in m.get("spec_wait_ms_values") or []:
            try:
                spec_waits.append(float(v))
            except Exception:
                pass
        quality = sess.get("final_quality") if isinstance(sess.get("final_quality"), dict) else {}
        score = quality.get("score") if isinstance(quality.get("score"), dict) else {}
        if score.get("upload_similarity_score") is not None:
            try:
                final_scores.append(float(score.get("upload_similarity_score")))
            except Exception:
                pass
        final_counts = m.get("final_counts") if isinstance(m.get("final_counts"), dict) else {}
        spec_counts = m.get("spec_counts") if isinstance(m.get("spec_counts"), dict) else {}
        final_error_num += int(max(0, int(final_counts.get("errors") or 0)))
        final_error_den += int(max(0, int(final_counts.get("enqueued") or 0)))
        spec_error_num += int(max(0, int(spec_counts.get("errors") or 0)))
        spec_error_den += int(max(0, int(spec_counts.get("enqueued") or 0)))
        if m.get("final_retry_rate") is not None:
            final_retry_den += int(max(0, int(final_counts.get("submit_attempts_sampled") or 0)))
            if final_counts.get("submit_attempts_sampled") is not None:
                try:
                    sampled = int(max(0, int(final_counts.get("submit_attempts_sampled") or 0)))
                    final_retry_num += int(round(float(m.get("final_retry_rate")) * float(sampled)))
                except Exception:
                    pass
        if m.get("spec_retry_rate") is not None:
            spec_retry_den += int(max(0, int(spec_counts.get("submit_attempts_sampled") or 0)))
            if spec_counts.get("submit_attempts_sampled") is not None:
                try:
                    sampled = int(max(0, int(spec_counts.get("submit_attempts_sampled") or 0)))
                    spec_retry_num += int(round(float(m.get("spec_retry_rate")) * float(sampled)))
                except Exception:
                    pass
        blob_fetch_row = m.get("blob_fetch_ms") if isinstance(m.get("blob_fetch_ms"), dict) else {}
        blob_p95 = blob_fetch_row.get("p95")
        if blob_p95 is not None:
            try:
                blob_fetch_vals.append(float(blob_p95))
            except Exception:
                pass

    metrics = {
        "sessions_total": int(len(sessions)),
        "sessions_ok": int(len([s for s in sessions if bool(s.get("ready"))])),
        "sessions_error": int(len([s for s in sessions if not bool(s.get("ready"))])),
        "final_similarity_score_avg": (round(sum(final_scores) / float(len(final_scores)), 6) if final_scores else None),
        "interactive_wait_ms": _wait_summary(final_waits + spec_waits),
        "final_wait_ms": _wait_summary(final_waits),
        "spec_wait_ms": _wait_summary(spec_waits),
        "final_error_rate": (round(float(final_error_num) / float(final_error_den), 6) if final_error_den > 0 else None),
        "spec_error_rate": (round(float(spec_error_num) / float(spec_error_den), 6) if spec_error_den > 0 else None),
        "final_retry_rate": (round(float(final_retry_num) / float(final_retry_den), 6) if final_retry_den > 0 else None),
        "spec_retry_rate": (round(float(spec_retry_num) / float(spec_retry_den), 6) if spec_retry_den > 0 else None),
        "blob_fetch_ms_p95": (_percentile(blob_fetch_vals, 0.95) if blob_fetch_vals else None),
        "pool_blob_fetch_ms_p95": None,
    }

    pool_status = sample.get("pool_status") if isinstance(sample.get("pool_status"), dict) else {}
    if pool_status.get("blob_fetch_ms_p95") is not None:
        try:
            metrics["pool_blob_fetch_ms_p95"] = round(float(pool_status.get("blob_fetch_ms_p95")), 3)
        except Exception:
            metrics["pool_blob_fetch_ms_p95"] = None

    return metrics


def _aggregate_by_concurrency(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[int, list[dict[str, Any]]] = {}
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        concurrency = _safe_int(sample.get("concurrency"), 0)
        if concurrency <= 0:
            continue
        buckets.setdefault(int(concurrency), []).append(sample)

    rows: list[dict[str, Any]] = []
    for concurrency, entries in sorted(buckets.items(), key=lambda kv: kv[0]):
        interactive_vals: list[float] = []
        final_vals: list[float] = []
        spec_vals: list[float] = []
        final_scores: list[float] = []
        final_error_num = 0
        final_error_den = 0
        spec_error_num = 0
        spec_error_den = 0
        final_retry_num = 0
        final_retry_den = 0
        spec_retry_num = 0
        spec_retry_den = 0
        blob_fetch_vals: list[float] = []

        for sample in entries:
            metrics = sample.get("metrics") if isinstance(sample.get("metrics"), dict) else {}
            for k, arr in (
                ("interactive_wait_ms", interactive_vals),
                ("final_wait_ms", final_vals),
                ("spec_wait_ms", spec_vals),
            ):
                row = metrics.get(k) if isinstance(metrics.get(k), dict) else {}
                p95 = row.get("p95")
                if p95 is not None:
                    try:
                        arr.append(float(p95))
                    except Exception:
                        pass
            if metrics.get("final_similarity_score_avg") is not None:
                try:
                    final_scores.append(float(metrics.get("final_similarity_score_avg")))
                except Exception:
                    pass
            # Rebuild weighted rates from per-session lane metrics
            sessions = [dict(s) for s in (sample.get("sessions") or []) if isinstance(s, dict)]
            for sess in sessions:
                lm = sess.get("lane_metrics") if isinstance(sess.get("lane_metrics"), dict) else {}
                fcounts = lm.get("final_counts") if isinstance(lm.get("final_counts"), dict) else {}
                scounts = lm.get("spec_counts") if isinstance(lm.get("spec_counts"), dict) else {}
                final_error_num += int(max(0, int(fcounts.get("errors") or 0)))
                final_error_den += int(max(0, int(fcounts.get("enqueued") or 0)))
                spec_error_num += int(max(0, int(scounts.get("errors") or 0)))
                spec_error_den += int(max(0, int(scounts.get("enqueued") or 0)))
                if lm.get("final_retry_rate") is not None:
                    sampled = int(max(0, int(fcounts.get("submit_attempts_sampled") or 0)))
                    final_retry_den += sampled
                    try:
                        final_retry_num += int(round(float(lm.get("final_retry_rate")) * float(sampled)))
                    except Exception:
                        pass
                if lm.get("spec_retry_rate") is not None:
                    sampled = int(max(0, int(scounts.get("submit_attempts_sampled") or 0)))
                    spec_retry_den += sampled
                    try:
                        spec_retry_num += int(round(float(lm.get("spec_retry_rate")) * float(sampled)))
                    except Exception:
                        pass
            blob_fetch = metrics.get("pool_blob_fetch_ms_p95")
            if blob_fetch is not None:
                try:
                    blob_fetch_vals.append(float(blob_fetch))
                except Exception:
                    pass
            blob_fetch_local = metrics.get("blob_fetch_ms_p95")
            if blob_fetch_local is not None:
                try:
                    blob_fetch_vals.append(float(blob_fetch_local))
                except Exception:
                    pass

        rows.append(
            {
                "concurrency": int(concurrency),
                "samples_total": int(len(entries)),
                "final_similarity_score_avg": (_mean(final_scores) if final_scores else None),
                "interactive_wait_ms_p95_avg": (_mean(interactive_vals) if interactive_vals else None),
                "final_wait_ms_p95_avg": (_mean(final_vals) if final_vals else None),
                "spec_wait_ms_p95_avg": (_mean(spec_vals) if spec_vals else None),
                "final_error_rate": (
                    round(float(final_error_num) / float(final_error_den), 6) if final_error_den > 0 else None
                ),
                "spec_error_rate": (
                    round(float(spec_error_num) / float(spec_error_den), 6) if spec_error_den > 0 else None
                ),
                "final_retry_rate": (
                    round(float(final_retry_num) / float(final_retry_den), 6) if final_retry_den > 0 else None
                ),
                "spec_retry_rate": (
                    round(float(spec_retry_num) / float(spec_retry_den), 6) if spec_retry_den > 0 else None
                ),
                "pool_blob_fetch_ms_p95_avg": (_mean(blob_fetch_vals) if blob_fetch_vals else None),
            }
        )
    return rows


async def _run_sample(
    *,
    run_id: str,
    sample_index: int,
    repeat_index: int,
    concurrency: int,
    fixture_id: str,
    fixture_version: str,
    pcm16le: bytes,
    stream_frame_ms: int,
    realtime_factor: float,
) -> dict[str, Any]:
    started_ts = time.time()
    sessions: list[dict[str, Any]] = []
    session_ids: list[str] = []
    live_sessions = _cfg("live_sessions")
    try:
        for _ in range(int(concurrency)):
            sess = live_sessions.create_session()
            sid = str(sess.get("session_id") or "")
            if not sid:
                raise RuntimeError("session_id_missing")
            session_ids.append(sid)
            live_sessions.set_fixture_metadata(
                sid,
                fixture_id=fixture_id,
                fixture_version=fixture_version,
                fixture_test_mode="playback",
            )

        ws_runs_raw = await asyncio.gather(
            *[
                _run_fixture_audio_over_ws(
                    session_id=sid,
                    pcm16le=pcm16le,
                    frame_ms=stream_frame_ms,
                    realtime_factor=realtime_factor,
                )
                for sid in session_ids
            ],
            return_exceptions=True,
        )
        for idx, item in enumerate(ws_runs_raw):
            if isinstance(item, Exception):
                raise RuntimeError(f"ws_run_failed[{idx}]:{type(item).__name__}:{item}")

        result_rows = await asyncio.gather(
            *[_wait_for_semilive_result_state(sid, timeout_s=300.0) for sid in session_ids],
            return_exceptions=True,
        )
        for idx, sid in enumerate(session_ids):
            result_row = result_rows[idx]
            if isinstance(result_row, Exception):
                sessions.append(
                    {
                        "session_id": sid,
                        "ready": False,
                        "error": f"{type(result_row).__name__}: {result_row}",
                        "finalization_state": "error",
                    }
                )
                continue
            semilive_result, ready = result_row
            semilive_result = dict(semilive_result or {})
            stats_log_path = Path(str(live_sessions.stats_log_path(sid) or "")).resolve()
            quality = score_semilive_text_against_fixture(
                fixture_id=fixture_id,
                semilive_text=str(semilive_result.get("final_text") or ""),
                semilive_result=semilive_result,
                stats_log_path=stats_log_path,
            )
            lane_metrics = _extract_session_lane_metrics(stats_log_path)
            sessions.append(
                {
                    "session_id": sid,
                    "ready": bool(ready),
                    "error": ("" if bool(ready) else "result_not_ready"),
                    "finalization_state": str(semilive_result.get("finalization_state") or ""),
                    "transcript_revision": int(max(0, int(semilive_result.get("transcript_revision") or 0))),
                    "final_quality": quality,
                    "lane_metrics": lane_metrics,
                }
            )

        sample = {
            "run_id": str(run_id),
            "sample_index": int(sample_index),
            "repeat_index": int(repeat_index),
            "concurrency": int(concurrency),
            "state": "ok",
            "started_at_utc": _iso_utc(started_ts),
            "finished_at_utc": _iso_utc(time.time()),
            "duration_s": round(max(0.0, time.time() - started_ts), 3),
            "fixture_id": str(fixture_id),
            "fixture_version": str(fixture_version),
            "sessions": sessions,
            "pool_status": _try_pool_status_snapshot(),
        }
        sample["metrics"] = _aggregate_sample(sample)
        return sample
    finally:
        for sid in session_ids:
            try:
                live_sessions.close_session(sid, reason="asr_loadtest_sample_done")
            except Exception:
                pass


async def _run_loadtest(run_id: str) -> None:
    rid = _safe_filename(str(run_id or "").strip())
    run = _RUNS.get(rid)
    if not isinstance(run, dict):
        return

    run["status"] = "running"
    run["started_at_utc"] = _iso_utc(time.time())
    _persist_report(rid)

    live_sessions = _cfg("live_sessions")
    max_sessions_prev = None
    try:
        req = run.get("request") if isinstance(run.get("request"), dict) else {}
        fixture_id = str(req.get("fixture_id") or "").strip()
        repeats = int(max(1, int(req.get("repeats") or 1)))
        concurrency_values = [int(v) for v in (req.get("concurrency_values") or [])]
        stream_frame_ms = int(max(20, int(req.get("stream_frame_ms") or 40)))
        realtime_factor = float(max(0.1, _safe_float(req.get("realtime_factor"), 1.0)))
        max_audio_ms = int(max(0, int(req.get("max_audio_ms") or 0)))
        auto_raise_capacity = bool(req.get("auto_raise_capacity", True))

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

        max_requested_concurrency = max(concurrency_values) if concurrency_values else 1
        if auto_raise_capacity:
            try:
                current_max = int(max(1, int(live_sessions.get_max_sessions())))
                target_max = int(max(current_max, max_requested_concurrency))
                if target_max != current_max:
                    max_sessions_prev = int(live_sessions.set_max_sessions(target_max))
                    run["capacity_override"] = {
                        "previous_max_sessions": int(current_max),
                        "temporary_max_sessions": int(target_max),
                    }
            except Exception as e:
                run["warnings"].append(f"capacity_override_failed:{type(e).__name__}:{e}")
        _persist_report(rid)

        sample_index = 0
        for concurrency in concurrency_values:
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
                        concurrency=int(concurrency),
                        fixture_id=fixture_id,
                        fixture_version=fixture_version,
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
                            "concurrency": int(concurrency),
                            "state": "error",
                            "error": f"{type(e).__name__}: {e}",
                            "duration_s": round(max(0.0, time.time() - started_ts), 3),
                            "finished_at_utc": _iso_utc(time.time()),
                        }
                    )
                    run["errors"].append(
                        {
                            "sample_index": int(sample_index),
                            "repeat_index": int(repeat_index),
                            "concurrency": int(concurrency),
                            "error": f"{type(e).__name__}: {e}",
                        }
                    )
                run["progress"]["samples_completed"] = int(len(run["samples"]))
                run["progress"]["samples_failed"] = int(
                    len([s for s in run["samples"] if str(s.get("state") or "") != "ok"])
                )
                run["aggregate_by_concurrency"] = _aggregate_by_concurrency(run["samples"])
                _persist_report(rid)

        failed_count = int(run["progress"].get("samples_failed") or 0)
        run["status"] = "completed_with_errors" if failed_count > 0 else "completed"
    except Exception as e:
        run["status"] = "failed"
        run["errors"].append({"scope": "run", "error": f"{type(e).__name__}: {e}"})
    finally:
        if max_sessions_prev is not None:
            try:
                live_sessions.set_max_sessions(int(max_sessions_prev))
            except Exception:
                pass
        run["finished_at_utc"] = _iso_utc(time.time())
        run["progress"]["current_sample_index"] = int(run["progress"].get("samples_completed") or 0)
        _persist_report(rid)


async def start_run(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="payload must be a JSON object")
    fixture_id = str(payload.get("fixture_id") or "").strip()
    if not fixture_id:
        raise HTTPException(status_code=400, detail="fixture_id is required")

    repeats = int(max(1, _safe_int(payload.get("repeats"), 1)))
    if repeats > 10:
        raise HTTPException(status_code=400, detail="repeats out of range (1..10)")

    concurrency_values = _parse_int_list(
        payload.get("concurrency_values"),
        default=[1, 2, 3],
        min_value=1,
        max_value=12,
        field_name="concurrency_values",
    )

    sample_total = int(len(concurrency_values) * repeats)
    if sample_total > 100:
        raise HTTPException(status_code=400, detail="too many samples requested (max 100)")

    stream_frame_ms = int(max(20, min(500, _safe_int(payload.get("stream_frame_ms"), 40))))
    realtime_factor = float(max(0.1, min(20.0, _safe_float(payload.get("realtime_factor"), 1.0))))
    max_audio_ms = int(max(0, _safe_int(payload.get("max_audio_ms"), 0)))
    auto_raise_capacity = bool(payload.get("auto_raise_capacity", True))

    active_run_id = _active_run_id()
    if active_run_id:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "asr_loadtest_already_running",
                "run_id": active_run_id,
                "message": "An ASR loadtest run is already active.",
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
            "fixture_id": str(fixture_id),
            "repeats": int(repeats),
            "concurrency_values": [int(v) for v in concurrency_values],
            "samples_total": int(sample_total),
            "stream_frame_ms": int(stream_frame_ms),
            "realtime_factor": float(realtime_factor),
            "max_audio_ms": int(max_audio_ms),
            "auto_raise_capacity": bool(auto_raise_capacity),
        },
        "progress": {
            "samples_total": int(sample_total),
            "samples_completed": 0,
            "samples_failed": 0,
            "current_sample_index": 0,
        },
        "samples": [],
        "aggregate_by_concurrency": [],
        "errors": [],
        "warnings": [],
    }
    _RUNS[run_id] = run
    _persist_report(run_id)

    task = asyncio.create_task(_run_loadtest(run_id))
    _TASKS[run_id] = task
    task.add_done_callback(lambda _task, rid=run_id: _TASKS.pop(rid, None))

    return {
        "protocol_version": _cfg("protocol_version"),
        "run_id": run_id,
        "status": "queued",
        "samples_total": int(sample_total),
        "repeats": int(repeats),
        "concurrency_values": [int(v) for v in concurrency_values],
        "report_url": _cfg("rooted_path_cb")(f"/demo/live/asr-loadtest/report/{run_id}"),
    }


def get_report(run_id: str) -> dict[str, Any]:
    rid = _safe_filename(str(run_id or "").strip())
    if not rid:
        raise HTTPException(status_code=400, detail="run_id is required")
    report = _RUNS.get(rid)
    if report is None:
        report = _load_report_from_disk(rid)
    if report is None:
        raise HTTPException(status_code=404, detail="asr loadtest run not found")
    return {
        "protocol_version": _cfg("protocol_version"),
        "run_id": rid,
        "report": report,
    }
