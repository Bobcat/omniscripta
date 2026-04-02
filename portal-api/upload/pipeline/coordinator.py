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

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_LLM_DIR = (_REPO_ROOT / "llm-worker").resolve()
if str(_LLM_DIR) not in sys.path:
    sys.path.insert(0, str(_LLM_DIR))

from shared.app_config import get_bool, get_float, get_setting, get_str
from upload._util import _normalize_speaker_mode, _read_json, _write_json_atomic
from upload.pipeline.progress_plan import DEFAULTS_SECONDS, build_prediction
from upload.pipeline.snipping import make_snippet
from upload.status_io import _write_status
from upload.topics.flow import TopicsFlow


def _api_status_owner() -> str:
    return str(get_str("upload.status_owners.api", "api") or "api").strip() or "api"


def _asr_worker_batch_status_owner() -> str:
    raw = str(get_str("upload.status_owners.asr_worker_batch", "asr-worker-batch") or "").strip()
    return raw or "asr-worker-batch"


def _append_log(path: Path, message: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with path.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] COORD {message}\n")


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
        "status_owners": {
            "api": "api",
            "asr_worker_batch": "asr-worker-batch",
            "api_topics": "api-topics",
            "llm_worker": "llm-worker",
        },
    }
    raw_snip = get_setting("upload.snip", {})
    if isinstance(raw_snip, dict):
        merged_snip = dict(cfg["snip"])
        merged_snip.update(raw_snip)
        cfg["snip"] = merged_snip
    raw_topics = get_setting("upload.topics", {})
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
    raw_status_owners = get_setting("upload.status_owners", {})
    if isinstance(raw_status_owners, dict):
        merged_status_owners = dict(cfg["status_owners"])
        merged_status_owners.update(raw_status_owners)
        cfg["status_owners"] = merged_status_owners
    return cfg


def _resolve_cfg_path(path_value: str, *, fallback_rel: str) -> Path:
    raw = str(path_value or "").strip() or fallback_rel
    p = Path(raw)
    return p if p.is_absolute() else (_REPO_ROOT / p)


def _progress_runs_path() -> Path:
    raw = get_str("upload.worker.progress_runs_path", "").strip()
    if raw:
        return _resolve_cfg_path(raw, fallback_rel="data/progress_db/runs_v1.jsonl")
    base = _resolve_cfg_path(
        get_str("upload.worker.progress_db_dir", "data/progress_db"),
        fallback_rel="data/progress_db",
    )
    return (base / "runs_v1.jsonl").resolve()


def _host_id() -> str:
    raw = get_str("upload.worker.host_id", "").strip()
    if raw:
        return raw
    return socket.gethostname().split(".")[0]


def _hardware_key(host_id: str) -> str:
    raw = get_str("upload.worker.hardware_key", "").strip()
    if raw:
        return raw
    if host_id == "dc1":
        return "dc1-rtx5070ti-cuda"
    if host_id == "dc2":
        return "dc2-rtx5090-cuda"
    return f"{host_id}-unknown"


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
            status_owner=_api_status_owner(),
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
            status_owner=_asr_worker_batch_status_owner(),
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


class UploadBatchCoordinator:
    def __init__(self) -> None:
        self._poll_interval_s = max(
            0.1,
            float(get_float("upload.coordinator.poll_interval_s", 0.5, min_value=0.1)),
        )
        self._idle_log_interval_s = max(
            1.0,
            float(get_float("upload.coordinator.idle_log_interval_s", 30.0, min_value=1.0)),
        )
        self._llm_wait_poll_s = max(
            0.1,
            float(get_float("upload.coordinator.llm_wait_poll_s", 0.5, min_value=0.1)),
        )
        self._llm_wait_timeout_s = max(
            30.0,
            float(get_float("upload.coordinator.llm_wait_timeout_s", 1800.0, min_value=30.0)),
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_idle_log_mono = 0.0

    def start(self) -> None:
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

    def metrics_snapshot(self) -> dict[str, Any]:
        t = self._thread
        running = bool(t is not None and t.is_alive())
        return {
            "running": running,
            "poll_interval_s": float(self._poll_interval_s),
            "idle_log_interval_s": float(self._idle_log_interval_s),
            "llm_wait_poll_s": float(self._llm_wait_poll_s),
            "llm_wait_timeout_s": float(self._llm_wait_timeout_s),
        }

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
                status_owner=_api_status_owner(),
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
                "language": (
                    ""
                    if str(opts.get("language") or "").strip().lower() in {"", "auto", "detect", "detect_auto", "detect-automatic", "detect-automatically"}
                    else str(opts.get("language") or "").strip().lower()
                ),
                "speaker_mode": speaker_mode,
                "min_speakers": opts.get("min_speakers"),
                "max_speakers": opts.get("max_speakers"),
                "diarize_enabled": bool(speaker_mode != "none"),
                "align_enabled": opts.get("align_enabled", True),
                "initial_prompt": opts.get("initial_prompt"),
                "priority": "background",
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

    def _process_job_locked(self, *, job: JobPaths, status: dict[str, Any]) -> None:
        TopicsFlow(
            service_cfg=_load_service_cfg(),
            progress_runs_path=_progress_runs_path(),
            hardware_key=_hardware_key(_host_id()),
            llm_wait_poll_s=self._llm_wait_poll_s,
            llm_wait_timeout_s=self._llm_wait_timeout_s,
            stop_event=self._stop,
        ).run(job=job, status=status)


UPLOAD_BATCH_COORDINATOR = UploadBatchCoordinator()
