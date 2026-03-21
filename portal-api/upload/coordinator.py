from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jobs.queue_fs import (
    JobPaths,
    finish_job,
    job_paths_from_dir,
    move_job_to_queue_inbox,
    nudge_inbox,
)
from queue_roots import UPLOAD_PREP_QUEUE, UPLOAD_WORKER_QUEUE

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_LLM_DIR = (_REPO_ROOT / "llm-worker").resolve()
if str(_LLM_DIR) not in sys.path:
    sys.path.insert(0, str(_LLM_DIR))

from shared.app_config import get_bool, get_float, get_setting, get_str
from llm_queue_fs import init_task_in_inbox, DONE as LLM_DONE, ERROR as LLM_ERROR
from upload.progress_plan import DEFAULTS_SECONDS, build_prediction
from upload.snipping import make_snippet
from upload.status_io import _write_status
from upload.speaker_lines import make_speaker_lines_from_srt
from upload.chunk_speaker_lines import chunk_speaker_lines
from upload.topics_parse import parse_topics_raw_file
from upload.topics_validate import validate_all_chunks
from upload.topics_merge import merge_topics


def _read_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return dict(raw) if isinstance(raw, dict) else {}


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _append_log(path: Path, message: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with path.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] COORD {message}\n")


def _write_json_atomic(path: Path, obj: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _pick_srt_path(*, job: JobPaths, status: dict[str, Any]) -> Path | None:
    whisperx_dir = (job.dir / "whisperx").resolve()
    srt_name = str(status.get("srt_filename") or "").strip()
    if srt_name:
        p = (whisperx_dir / srt_name).resolve()
        if p.exists():
            return p
    if not whisperx_dir.exists():
        return None
    candidates = sorted(p for p in whisperx_dir.glob("*.srt") if p.is_file())
    if not candidates:
        return None
    return candidates[-1]


def _load_service_cfg() -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "snip": {
            "minutes_default": 15,
        },
        "topics": {
            "chunk_minutes": 15,
            "ctx_len": 16384,
            "ctx_safety": 0.85,
            "prompt_overhead_tokens_est": 1200,
            "token_estimator": "chars_div4",
            "enabled": True,
            "prompt_id": "topics_v1",
            "prompt_path": "portal-api/upload/prompts/simple_prompt5.txt",
            "model": "",
            "generation": {},
        },
        "tabby": {
            "base_url": "http://127.0.0.1:5001",
            "api_key_env": "TABBY_API_KEY",
            "timeout_s": 600,
            "retries": 2,
            "retry_sleep_s": 2.0,
        },
    }
    raw_snip = get_setting("snip", {})
    if isinstance(raw_snip, dict):
        merged_snip = dict(cfg["snip"])
        merged_snip.update(raw_snip)
        cfg["snip"] = merged_snip
    raw_topics = get_setting("topics", {})
    if isinstance(raw_topics, dict):
        merged = dict(cfg["topics"])
        merged.update(raw_topics)
        base_gen = dict(cfg["topics"].get("generation") or {})
        raw_gen = raw_topics.get("generation")
        if isinstance(raw_gen, dict):
            base_gen.update(raw_gen)
        merged["generation"] = base_gen
        cfg["topics"] = merged
    raw_tabby = get_setting("tabby", {})
    if isinstance(raw_tabby, dict):
        merged_tabby = dict(cfg["tabby"])
        merged_tabby.update(raw_tabby)
        cfg["tabby"] = merged_tabby
    return cfg


def _unique_hints(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    out: list[str] = []
    for raw in values:
        item = str(raw).strip()
        if item and item not in out:
            out.append(item)
    return out


def _resolve_cfg_path(path_value: str, *, fallback_rel: str) -> Path:
    raw = str(path_value or "").strip() or fallback_rel
    p = Path(raw)
    return p if p.is_absolute() else (_REPO_ROOT / p)


def _progress_runs_path() -> Path:
    raw = get_str("worker.progress_runs_path", "").strip()
    if raw:
        return _resolve_cfg_path(raw, fallback_rel="data/progress_db/runs_v1.jsonl")
    base = _resolve_cfg_path(get_str("worker.progress_db_dir", "data/progress_db"), fallback_rel="data/progress_db")
    return (base / "runs_v1.jsonl").resolve()


def _host_id() -> str:
    raw = get_str("worker.host_id", "").strip()
    if raw:
        return raw
    return socket.gethostname().split(".")[0]


def _hardware_key(host_id: str) -> str:
    raw = get_str("worker.hardware_key", "").strip()
    if raw:
        return raw
    if host_id == "dc1":
        return "dc1-rtx5070ti-cuda"
    if host_id == "dc2":
        return "dc2-rtx5090-cuda"
    return f"{host_id}-unknown"


def _normalize_speaker_mode(mode: Any) -> str:
    raw = str(mode or "auto").strip().lower()
    if raw in {"none", "off", "disabled", "no_speaker", "nospeaker", "no-speaker"}:
        return "none"
    if raw == "fixed":
        return "fixed"
    return "auto"


class _SnippingProgressTracker:
    def __init__(
        self,
        *,
        status_path: Path,
        prediction_total_s: float,
        snipping_expected_s: float,
        eta_confidence: float,
        eta_hints: list[str],
        message: str,
        started_at: str,
    ) -> None:
        self._status_path = status_path
        self._prediction_total_s = max(1.0, float(prediction_total_s))
        self._snipping_expected_s = max(0.1, float(snipping_expected_s))
        self._eta_confidence = max(0.0, float(eta_confidence))
        self._eta_hints = list(eta_hints)
        self._message = str(message or "").strip() or "Creating snippet…"
        self._started_at = str(started_at or "").strip() or datetime.now(timezone.utc).isoformat()
        self._started_mono = 0.0
        self._last_progress = 0.0
        self._snipping_band = min(0.99, max(0.01, self._snipping_expected_s / self._prediction_total_s))

    def _write(self) -> None:
        now_mono = time.monotonic()
        elapsed_s = max(0.0, now_mono - self._started_mono) if self._started_mono > 0.0 else 0.0
        phase_frac = min(0.995, max(0.0, elapsed_s / self._snipping_expected_s))
        progress = min(self._snipping_band * phase_frac, self._snipping_band * 0.995)
        progress = max(self._last_progress, progress)
        self._last_progress = progress
        remaining_after = max(0.0, self._prediction_total_s - elapsed_s)
        est_total = max(self._prediction_total_s, elapsed_s + remaining_after)
        _write_status(
            self._status_path,
            state="running",
            phase="snipping",
            progress=progress,
            started_at=self._started_at,
            message=self._message,
            progress_mode="predictive_v1",
            eta_total_s=round(est_total, 3),
            eta_remaining_s=round(max(1.0, remaining_after), 3),
            elapsed_s=round(elapsed_s, 3),
            eta_confidence=round(float(self._eta_confidence), 3),
            eta_hints=list(self._eta_hints),
        )

    def start(self) -> None:
        self._started_mono = time.monotonic()
        self._write()

    def heartbeat(self) -> None:
        self._write()

    def finish(self, *, snippet_filename: str, asr_input_relpath: str, actual_elapsed_s: float) -> None:
        remaining_after = max(0.0, self._prediction_total_s - self._snipping_expected_s)
        total_s = max(float(actual_elapsed_s), float(actual_elapsed_s) + remaining_after)
        _write_status(
            self._status_path,
            state="queued",
            phase="awaiting_asr",
            subphase="handoff",
            progress=min(0.99, self._snipping_band),
            started_at=self._started_at,
            message="Snippet ready; queued for ASR",
            asr_input_relpath=asr_input_relpath,
            snippet_filename=snippet_filename,
            timings_text=f"snipping={max(0.0, float(actual_elapsed_s)):.2f}s",
            progress_mode="predictive_v1",
            eta_total_s=round(total_s, 3),
            eta_remaining_s=round(max(0.0, remaining_after), 3),
            elapsed_s=round(max(0.0, float(actual_elapsed_s)), 3),
            eta_confidence=round(float(self._eta_confidence), 3),
            eta_hints=list(self._eta_hints),
        )


class _TopicsProgressTracker:
    """Upload-global progress owner for the remaining topics segment.

    The ASR worker owns progress through its own terminal completion.
    After the worker writes `done`, the coordinator keeps prep details as
    subphases and projects only the remaining topics segment on the top-level
    progress fields.
    """

    def __init__(self, *, status_path: Path, status: dict[str, Any]) -> None:
        self._status_path = status_path
        self._base_progress = 0.0
        self._base_elapsed_s = 0.0
        self._eta_confidence = 0.0
        self._eta_hints: list[str] = []
        self._current_message = "Running…"
        self._last_progress = self._base_progress
        self._topics_started_t = 0.0
        self._topics_total = 0
        self._topics_done = 0
        self._current_chunk_started_t = 0.0

        self._topics_expected_s = max(0.1, float(DEFAULTS_SECONDS.get("llm_topics", 8.5)))
        self._visible_remaining_band = max(0.0, 0.99 - self._base_progress)

    def seed_from_upload_prediction(
        self,
        *,
        base_progress: float,
        base_elapsed_s: float,
        topics_expected_s: float,
        eta_confidence: float,
        eta_hints: list[str],
        message: str,
    ) -> None:
        self._base_progress = max(0.0, min(0.99, float(base_progress)))
        self._base_elapsed_s = max(0.0, float(base_elapsed_s))
        self._eta_confidence = max(0.0, float(eta_confidence))
        self._eta_hints = list(eta_hints)
        self._current_message = str(message or "").strip() or "Running…"
        self._last_progress = self._base_progress
        self._topics_expected_s = max(0.1, float(topics_expected_s))
        self._visible_remaining_band = max(0.0, 0.99 - self._base_progress)

    def _topics_fraction(self, *, now_mono: float) -> float:
        if self._topics_started_t <= 0.0:
            return 0.0
        if self._topics_total <= 0:
            elapsed_total = max(0.0, now_mono - self._topics_started_t)
            return min(0.995, max(0.0, elapsed_total / self._topics_expected_s))

        done = max(0, min(int(self._topics_done), int(self._topics_total)))
        if done >= int(self._topics_total):
            return 0.995

        chunk_base = max(0.0, float(done) / float(self._topics_total))
        chunk_ceiling = min(0.995, float(done + 1) / float(self._topics_total))
        chunk_span = max(0.0001, chunk_ceiling - chunk_base)
        expected_chunk = max(0.1, float(self._topics_expected_s) / float(self._topics_total))
        elapsed_chunk = max(0.0, now_mono - self._current_chunk_started_t) if self._current_chunk_started_t > 0.0 else 0.0
        chunk_frac = min(0.995, max(0.0, float(elapsed_chunk) / float(expected_chunk)))
        return min(chunk_ceiling, chunk_base + (chunk_frac * chunk_span))

    def _write(self) -> None:
        now_mono = time.monotonic()
        elapsed_topics = max(0.0, now_mono - self._topics_started_t) if self._topics_started_t > 0.0 else 0.0
        est_elapsed = self._base_elapsed_s + elapsed_topics

        if self._topics_total > 0:
            expected_chunk = max(0.1, float(self._topics_expected_s) / float(self._topics_total))
            elapsed_chunk = max(0.0, now_mono - self._current_chunk_started_t) if self._current_chunk_started_t > 0.0 else 0.0
            projected_current_chunk = max(expected_chunk, elapsed_chunk)
            remaining_chunks = max(0, int(self._topics_total) - int(self._topics_done) - 1)
            est_remaining = max(0.0, projected_current_chunk - elapsed_chunk) + (remaining_chunks * expected_chunk)
        else:
            est_remaining = max(0.0, float(self._topics_expected_s) - elapsed_topics)

        est_remaining = max(3.0, est_remaining)
        est_total = max(est_elapsed, est_elapsed + est_remaining)

        fraction = self._topics_fraction(now_mono=now_mono)
        progress = self._base_progress + (max(0.0, fraction) * self._visible_remaining_band)
        progress = min(0.99, max(self._last_progress, progress))
        self._last_progress = progress

        _write_status(
            self._status_path,
            progress=progress,
            phase="topics",
            message=self._current_message,
            progress_mode="predictive_v1",
            eta_total_s=round(max(0.0, est_total), 3),
            eta_remaining_s=round(max(0.0, est_remaining), 3),
            elapsed_s=round(max(0.0, est_elapsed), 3),
            eta_confidence=round(float(self._eta_confidence), 3),
            eta_hints=list(self._eta_hints),
        )

    def start(self, *, message: str) -> None:
        self._topics_started_t = time.monotonic()
        self._current_chunk_started_t = self._topics_started_t
        self._topics_done = 0
        self._topics_total = 0
        self._current_message = str(message or "").strip() or "Topics: preparing"
        self._write()

    def set_message(self, message: str) -> None:
        self._current_message = str(message or "").strip() or self._current_message
        self._write()

    def update(self, *, done_count: int, total_count: int, message: str) -> None:
        safe_total = max(0, int(total_count))
        safe_done = max(0, min(int(done_count), safe_total))
        if safe_done != self._topics_done:
            self._current_chunk_started_t = time.monotonic()
        elif safe_total > 0 and self._current_chunk_started_t <= 0.0:
            self._current_chunk_started_t = time.monotonic()
        self._topics_total = safe_total
        self._topics_done = safe_done
        self._current_message = str(message or "").strip() or self._current_message
        self._write()

    def elapsed_total_s(self) -> float:
        elapsed_topics = max(0.0, time.monotonic() - self._topics_started_t) if self._topics_started_t > 0.0 else 0.0
        return max(0.0, self._base_elapsed_s + elapsed_topics)

    @property
    def eta_confidence(self) -> float:
        return float(self._eta_confidence)

    @property
    def eta_hints(self) -> list[str]:
        return list(self._eta_hints)


class UploadBatchCoordinator:
    def __init__(self) -> None:
        self._enabled = get_bool("upload_coordinator.enabled", True)
        self._poll_interval_s = max(0.1, float(get_float("upload_coordinator.poll_interval_s", 0.5, min_value=0.1)))
        self._idle_log_interval_s = max(
            1.0,
            float(get_float("upload_coordinator.idle_log_interval_s", 30.0, min_value=1.0)),
        )
        self._llm_wait_poll_s = max(0.1, float(get_float("upload_coordinator.llm_wait_poll_s", 0.5, min_value=0.1)))
        self._llm_wait_timeout_s = max(
            30.0,
            float(get_float("upload_coordinator.llm_wait_timeout_s", 1800.0, min_value=30.0)),
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_idle_log_mono = 0.0

    def start(self) -> None:
        if not self._enabled:
            print("upload_batch_coordinator disabled", flush=True)
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="upload-batch-coordinator",
            daemon=True,
        )
        self._thread.start()
        print(
            "upload_batch_coordinator started "
            f"poll_interval_s={self._poll_interval_s:.2f} "
            f"llm_wait_poll_s={self._llm_wait_poll_s:.2f}",
            flush=True,
        )

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=2.0)
        self._thread = None
        print("upload_batch_coordinator stopped", flush=True)

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            did_work = False
            try:
                did_work = self._scan_once()
            except Exception as e:
                print(f"upload_batch_coordinator scan_error={type(e).__name__}: {e}", flush=True)
            if not did_work:
                now = time.monotonic()
                if (now - self._last_idle_log_mono) >= self._idle_log_interval_s:
                    self._last_idle_log_mono = now
                    print("upload_batch_coordinator idle", flush=True)
            self._stop.wait(self._poll_interval_s)

    def _scan_once(self) -> bool:
        did_work = False
        if UPLOAD_PREP_QUEUE.inbox.exists():
            for job_dir in sorted(p for p in UPLOAD_PREP_QUEUE.inbox.iterdir() if p.is_dir() and not p.name.startswith(".tmp_")):
                if self._stop.is_set():
                    break
                try:
                    did_work = self._maybe_prepare_job(job_dir) or did_work
                except Exception as e:
                    print(f"upload_batch_coordinator inbox_scan_error job_id={job_dir.name} error={e!r}", flush=True)
        if UPLOAD_WORKER_QUEUE.done.exists():
            for job_dir in sorted(p for p in UPLOAD_WORKER_QUEUE.done.iterdir() if p.is_dir()):
                if self._stop.is_set():
                    break
                try:
                    did_work = self._maybe_process_job(job_dir) or did_work
                except Exception as e:
                    print(f"upload_batch_coordinator job_scan_error job_id={job_dir.name} error={e!r}", flush=True)
        return did_work

    def _acquire_lock(self, lock_path: Path) -> int | None:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            return None
        os.write(fd, f"{os.getpid()} {int(time.time())}\n".encode("utf-8"))
        return fd

    def _release_lock(self, lock_path: Path, fd: int | None) -> None:
        if fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass
        try:
            lock_path.unlink(missing_ok=True)
        except Exception:
            pass

    def _maybe_process_job(self, job_dir: Path) -> bool:
        job = job_paths_from_dir(job_dir, queue_root=UPLOAD_WORKER_QUEUE)
        if not job.status_path.exists() or not job.job_path.exists():
            return False

        status = _read_json(job.status_path)
        if str(status.get("state") or "").strip().lower() != "done":
            return False
        if str(status.get("phase") or "").strip().lower() != "done":
            return False
        if "topics_status" in status:
            return False

        lock_path = (job.dir / ".upload_coordinator.lock").resolve()
        fd = self._acquire_lock(lock_path)
        if fd is None:
            return False
        try:
            self._process_job_locked(job=job, status=status)
            return True
        finally:
            self._release_lock(lock_path, fd)

    def _maybe_prepare_job(self, job_dir: Path) -> bool:
        job = job_paths_from_dir(job_dir, queue_root=UPLOAD_PREP_QUEUE)
        if not job.status_path.exists() or not job.job_path.exists():
            return False

        status = _read_json(job.status_path)
        job_cfg = _read_json(job.job_path)
        phase = str(status.get("phase") or "").strip().lower()
        if phase not in {"upload", "snipping"}:
            return False

        lock_path = (job.dir / ".upload_coordinator.lock").resolve()
        fd = self._acquire_lock(lock_path)
        if fd is None:
            return False
        try:
            self._prepare_job_locked(job=job, status=status, job_cfg=job_cfg)
            return True
        finally:
            self._release_lock(lock_path, fd)

    def _prepare_job_locked(self, *, job: JobPaths, status: dict[str, Any], job_cfg: dict[str, Any]) -> None:
        service_cfg = _load_service_cfg()
        opts = dict(job_cfg.get("options") or {})
        orig_filename = str(job_cfg.get("orig_filename") or "").strip()
        if not orig_filename:
            raise RuntimeError("Missing orig_filename in upload job config")

        upload_dir = (job.dir / "upload").resolve()
        snippet_dir = (job.dir / "snippet").resolve()
        input_path = (upload_dir / orig_filename).resolve()
        if not input_path.exists():
            raise RuntimeError(f"Upload missing: {input_path}")

        topics_cfg = dict(service_cfg.get("topics") or {})
        snip_cfg = dict(service_cfg.get("snip") or {})
        snippet_seconds = int(opts.get("snippet_seconds") or int(snip_cfg.get("minutes_default", 15)) * 60)
        speaker_mode = _normalize_speaker_mode(opts.get("speaker_mode", "auto"))
        prediction = build_prediction(
            runs_path=_progress_runs_path(),
            hardware_key=_hardware_key(_host_id()),
            topics_enabled=bool(topics_cfg.get("enabled", False)),
            speaker_mode=speaker_mode,
            snippet_seconds=snippet_seconds,
        )
        disp = f"{snippet_seconds//60} min" if snippet_seconds > 0 and (snippet_seconds % 60) == 0 else f"{snippet_seconds} s"
        started_at = str(status.get("started_at") or "").strip() or datetime.now(timezone.utc).isoformat()
        tracker = _SnippingProgressTracker(
            status_path=job.status_path,
            prediction_total_s=prediction.total_expected_s,
            snipping_expected_s=float(prediction.phase_expected_s.get("snipping", DEFAULTS_SECONDS.get("snipping", 5.0))),
            eta_confidence=prediction.confidence,
            eta_hints=list(prediction.hints),
            message=f"Creating snippet ({disp})…",
            started_at=started_at,
        )
        _append_log(job.log_path, f"snipping_claimed snippet_seconds={snippet_seconds}")
        tracker.start()

        hb_stop = threading.Event()

        def _heartbeat() -> None:
            while not hb_stop.wait(0.5):
                try:
                    tracker.heartbeat()
                except Exception:
                    pass

        hb_thread = threading.Thread(target=_heartbeat, name=f"upload-snipping-{job.job_id}", daemon=True)
        hb_thread.start()
        t0 = time.monotonic()
        try:
            snippet_path = make_snippet(input_path, snippet_dir, seconds=snippet_seconds)
        except Exception as e:
            hb_stop.set()
            hb_thread.join(timeout=1.0)
            self._patch_status(
                job.status_path,
                state="error",
                phase="error",
                progress=1.0,
                finished_at=datetime.now(timezone.utc).isoformat(),
                message=f"Snipping failed: {e!r}",
                error=str(e),
            )
            finish_job(job, ok=False)
            raise
        hb_stop.set()
        hb_thread.join(timeout=1.0)
        elapsed_s = max(0.0, time.monotonic() - t0)
        snippet_relpath = str(snippet_path.relative_to(job.dir))
        tracker.finish(
            snippet_filename=snippet_path.name,
            asr_input_relpath=snippet_relpath,
            actual_elapsed_s=elapsed_s,
        )
        worker_job = {
            "input": {
                "audio_relpath": snippet_relpath,
                "duration_ms": int(max(1, snippet_seconds) * 1000),
                "format": str(snippet_path.suffix.lstrip(".") or "mp3"),
            },
            "request": {
                "language": str(opts.get("language", "nl") or "nl"),
                "speaker_mode": speaker_mode,
                "min_speakers": opts.get("min_speakers"),
                "max_speakers": opts.get("max_speakers"),
                "diarize_enabled": bool(speaker_mode != "none"),
                "align_enabled": True,
                "initial_prompt": opts.get("initial_prompt"),
                "priority": "background",
                "latency_mode": "default",
                "routing": {
                    "slot_affinity": 0,
                },
            },
            "outputs": {
                "status_relpath": "status.json",
                "srt_relpath": str(Path("whisperx") / f"{snippet_path.stem}.srt"),
            },
            "worker_features": {
                "write_status_json": True,
                "track_pending_status": True,
                "predictive_progress": True,
                "write_timings_text": True,
                "include_runtime_meta": True,
                "download_srt": True,
            },
        }
        _write_json_atomic(job.job_path, worker_job)
        moved_job = move_job_to_queue_inbox(job, dst_queue_root=UPLOAD_WORKER_QUEUE)
        try:
            (moved_job.dir / ".upload_coordinator.lock").unlink(missing_ok=True)
        except Exception:
            pass
        _append_log(moved_job.log_path, f"snipping_done snippet={snippet_path.name} seconds={elapsed_s:.3f}")
        _append_log(moved_job.log_path, "upload_handoff queued_for_upload_worker")
        nudge_inbox(UPLOAD_WORKER_QUEUE)

    def _patch_status(self, status_path: Path, **patch: Any) -> None:
        try:
            _write_status(status_path, **patch)
        except Exception:
            pass

    def _enqueue_topics_tasks(
        self,
        *,
        job: JobPaths,
        manifest_path: Path,
        orig_stem: str,
        prompt_id: str,
        service_cfg: dict[str, Any],
    ) -> dict[int, str]:
        topics_cfg = dict(service_cfg.get("topics") or {})
        tabby_cfg = dict(service_cfg.get("tabby") or {})
        result_dir = (job.dir / "result").resolve()
        prompt_path = Path(str(topics_cfg.get("prompt_path") or "").strip())
        if not prompt_path.is_absolute():
            prompt_path = (_REPO_ROOT / prompt_path).resolve()
        model = str(topics_cfg.get("model") or "").strip()
        generation = dict(topics_cfg.get("generation") or {})

        manifest = _read_json(manifest_path)
        chunks = manifest.get("chunks") or []
        task_ids: dict[int, str] = {}
        for ch in chunks:
            if not isinstance(ch, dict):
                continue
            idx = int(ch.get("index") or 0)
            chunk_name = str(ch.get("filename") or "").strip()
            if not chunk_name:
                continue
            chunk_path = (result_dir / chunk_name).resolve()
            if not chunk_path.exists():
                raise RuntimeError(f"Missing chunk input for llm task: {chunk_path}")
            out_base = f"{orig_stem}_{prompt_id}_chunk_{idx:04d}"
            task = init_task_in_inbox(
                task_kind="prompt_run",
                spec={
                    "owner_kind": "upload_audio",
                    "owner_job_id": job.job_id,
                    "prompt_id": prompt_id,
                    "model": model,
                    "prompt_path": str(prompt_path),
                    "generation": generation,
                    "base_url": str(tabby_cfg.get("base_url") or "").strip(),
                    "api_key_env": str(tabby_cfg.get("api_key_env") or "TABBY_API_KEY"),
                    "timeout_s": int(tabby_cfg.get("timeout_s") or 600),
                    "retries": int(tabby_cfg.get("retries") or 2),
                    "retry_sleep_s": float(tabby_cfg.get("retry_sleep_s") or 2.0),
                    "input_artifact": str(chunk_path),
                    "output_dir": str(result_dir),
                    "output_basename": out_base,
                },
            )
            task_ids[idx] = task.task_id
        return task_ids

    def _wait_topics_tasks(
        self,
        task_ids: dict[int, str],
        *,
        progress_tracker: _TopicsProgressTracker | None = None,
    ) -> dict[int, Path]:
        pending = dict(task_ids)
        outputs: dict[int, Path] = {}
        total = len(task_ids)
        deadline = time.monotonic() + self._llm_wait_timeout_s
        while pending and not self._stop.is_set():
            for idx, task_id in list(pending.items()):
                done_dir = (LLM_DONE / task_id).resolve()
                err_dir = (LLM_ERROR / task_id).resolve()
                if done_dir.exists():
                    status_path = (done_dir / "status.json").resolve()
                    status = _read_json(status_path) if status_path.exists() else {}
                    result = dict(status.get("result") or {})
                    text_path_raw = str(result.get("output_text_path") or "").strip()
                    if not text_path_raw:
                        raise RuntimeError(f"Missing output_text_path in llm task result: {task_id}")
                    text_path = Path(text_path_raw).resolve()
                    if not text_path.exists():
                        raise RuntimeError(f"LLM task output text missing: {text_path}")
                    outputs[idx] = text_path
                    pending.pop(idx, None)
                    continue
                if err_dir.exists():
                    status_path = (err_dir / "status.json").resolve()
                    status = _read_json(status_path) if status_path.exists() else {}
                    err = str(status.get("error") or "unknown llm worker error")
                    raise RuntimeError(f"LLM task failed task_id={task_id}: {err}")
            if progress_tracker is not None:
                progress_tracker.update(
                    done_count=len(outputs),
                    total_count=total,
                    message=f"Topics: waiting for {len(outputs)}/{total} llm tasks",
                )
            if not pending:
                break
            if time.monotonic() > deadline:
                raise RuntimeError(f"Timed out waiting for llm tasks: pending={len(pending)}")
            self._stop.wait(self._llm_wait_poll_s)
        if self._stop.is_set():
            raise RuntimeError("Coordinator stopping while waiting for llm tasks")
        return outputs

    def _process_job_locked(self, *, job: JobPaths, status: dict[str, Any]) -> None:
        _append_log(job.log_path, "topics_claimed phase=done")
        service_cfg = _load_service_cfg()
        result_dir = (job.dir / "result").resolve()
        topics_cfg = dict(service_cfg.get("topics") or {})
        topics_enabled = bool(topics_cfg.get("enabled", False))
        prompt_id = str(topics_cfg.get("prompt_id") or "topics_v1")
        coord_started_mono = time.monotonic()
        base_elapsed_s = max(0.0, float(_safe_float(status.get("elapsed_s")) or 0.0))
        progress_tracker = _TopicsProgressTracker(status_path=job.status_path, status=status) if topics_enabled else None
        if progress_tracker is not None:
            snip_cfg = dict(service_cfg.get("snip") or {})
            snippet_seconds = int(status.get("snippet_seconds") or int(snip_cfg.get("minutes_default", 15)) * 60)
            speaker_mode = _normalize_speaker_mode(status.get("speaker_mode", "auto"))
            prediction = build_prediction(
                runs_path=_progress_runs_path(),
                hardware_key=_hardware_key(_host_id()),
                topics_enabled=topics_enabled,
                speaker_mode=speaker_mode,
                snippet_seconds=snippet_seconds,
            )
            phase_order = ["snipping", "whisperx_prepare", "whisperx_transcribe", "whisperx_align"]
            if speaker_mode != "none":
                phase_order.append("whisperx_diarize")
            topics_phase = "llm_topics" if topics_enabled else "llm_topics_skipped"
            phase_order.append(topics_phase)
            completed_expected = sum(max(0.0, float(prediction.phase_expected_s.get(phase, 0.0))) for phase in phase_order[:-1])
            total_expected = max(1.0, float(prediction.total_expected_s))
            progress_tracker.seed_from_upload_prediction(
                base_progress=min(0.99, completed_expected / total_expected),
                base_elapsed_s=base_elapsed_s,
                topics_expected_s=float(prediction.phase_expected_s.get(topics_phase, DEFAULTS_SECONDS.get(topics_phase, 8.5))),
                eta_confidence=prediction.confidence,
                eta_hints=list(prediction.hints),
                message=str(status.get("message") or "Running…"),
            )
        self._patch_status(
            job.status_path,
            state="running",
            phase="topics",
            subphase="speaker_lines",
            message="Topics: generating speaker_lines",
        )

        orig_filename = str(status.get("orig_filename") or "").strip()
        orig_stem = Path(orig_filename).stem if orig_filename else "transcript"
        srt_path = _pick_srt_path(job=job, status=status)
        if srt_path is None:
            raise RuntimeError("No SRT found for topics handoff")

        speaker_lines_path, transcript_end_hms = make_speaker_lines_from_srt(
            job=job,
            srt_path=srt_path,
            orig_stem=orig_stem,
        )
        manifest_path = chunk_speaker_lines(
            job=job,
            speaker_lines_path=speaker_lines_path,
            orig_stem=orig_stem,
            service_cfg=service_cfg,
            transcript_end_hms=transcript_end_hms,
        )
        topics_status = "disabled"
        topics_warning = ""
        if topics_enabled:
            topics_status = "ok"
            try:
                progress_tracker.start(message="Topics: enqueueing llm tasks")
                self._patch_status(
                    job.status_path,
                    phase="topics",
                    subphase="queue",
                    message="Topics: enqueueing llm tasks",
                )
                task_ids = self._enqueue_topics_tasks(
                    job=job,
                    manifest_path=manifest_path,
                    orig_stem=orig_stem,
                    prompt_id=prompt_id,
                    service_cfg=service_cfg,
                )
                self._patch_status(
                    job.status_path,
                    phase="topics",
                    subphase="wait",
                    message=f"Topics: waiting for 0/{len(task_ids)} llm tasks",
                )
                progress_tracker.update(
                    done_count=0,
                    total_count=len(task_ids),
                    message=f"Topics: waiting for 0/{len(task_ids)} llm tasks",
                )
                chunk_outputs = self._wait_topics_tasks(task_ids, progress_tracker=progress_tracker)
                for idx, text_path in sorted(chunk_outputs.items()):
                    raw_path = (result_dir / f"{orig_stem}_{prompt_id}_chunk_{int(idx):04d}_raw.txt").resolve()
                    raw_text = text_path.read_text(encoding="utf-8", errors="replace")
                    raw_path.write_text(raw_text, encoding="utf-8")
                    parsed_path = (result_dir / f"{orig_stem}_{prompt_id}_chunk_{int(idx):04d}.json").resolve()
                    parse_topics_raw_file(raw_txt_path=raw_path, out_json_path=parsed_path)

                report_path = (result_dir / f"{orig_stem}_{prompt_id}_validation.json").resolve()
                validate_all_chunks(
                    manifest_path=manifest_path,
                    parsed_dir=result_dir,
                    orig_stem=orig_stem,
                    prompt_id=prompt_id,
                    out_report_path=report_path,
                )
                report = _read_json(report_path)
                if not bool(report.get("is_valid", False)):
                    topics_status = "validation_failed"
                    topics_warning = f"Topics validation failed: {report_path.name}"
                    _append_log(job.log_path, f"topics_nonfatal validation_failed report={report_path.name}")
                else:
                    merged_path = (result_dir / f"{orig_stem}_{prompt_id}_merged.json").resolve()
                    merge_topics(
                        manifest_path=manifest_path,
                        parsed_dir=result_dir,
                        orig_stem=orig_stem,
                        prompt_id=prompt_id,
                        out_merged_path=merged_path,
                    )
            except Exception as e:
                progress_tracker.set_message(f"Topics failed: {e}")
                topics_status = "failed"
                topics_warning = str(e)
                _append_log(job.log_path, f"topics_nonfatal error={e!r}")

        final_elapsed_s = (
            progress_tracker.elapsed_total_s()
            if progress_tracker is not None
            else (base_elapsed_s + max(0.0, time.monotonic() - coord_started_mono))
        )
        final_eta_confidence = (
            progress_tracker.eta_confidence
            if progress_tracker is not None
            else max(0.0, float(_safe_float(status.get("eta_confidence")) or 0.0))
        )
        final_eta_hints = (
            progress_tracker.eta_hints
            if progress_tracker is not None
            else _unique_hints(status.get("eta_hints"))
        )
        self._patch_status(
            job.status_path,
            state="done",
            phase="done",
            subphase="",
            progress=1.0,
            finished_at=datetime.now(timezone.utc).isoformat(),
            message="Done",
            progress_mode="predictive_v1",
            elapsed_s=round(final_elapsed_s, 3),
            eta_total_s=round(final_elapsed_s, 3),
            eta_remaining_s=0.0,
            eta_confidence=round(float(final_eta_confidence), 3),
            eta_hints=list(final_eta_hints),
            srt_filename=srt_path.name,
            speaker_lines_filename=speaker_lines_path.name,
            speaker_lines_manifest_filename=manifest_path.name,
            topics_status=topics_status,
            topics_warning=topics_warning,
        )
        _append_log(job.log_path, "topics_done")


UPLOAD_BATCH_COORDINATOR = UploadBatchCoordinator()
