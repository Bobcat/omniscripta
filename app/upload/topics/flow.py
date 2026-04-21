from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from upload.jobs.queue_fs import JobPaths
from upload._util import _normalize_speaker_mode, _read_json, _safe_float
from upload.pipeline.progress_plan import DEFAULTS_SECONDS, build_prediction
from upload.status_io import _write_status
from upload.topics.chunk_speaker_lines import chunk_speaker_lines
from upload.topics.merge import merge_topics
from upload.topics.parse import parse_topics_raw_file
from upload.topics.speaker_lines import make_speaker_lines_from_srt
from upload.topics.validate import validate_all_chunks
from workers.llm.queue_fs import DONE as LLM_DONE, ERROR as LLM_ERROR, init_task_in_inbox


def _append_log(path: Path, message: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with path.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] COORD {message}\n")


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


def _unique_hints(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    out: list[str] = []
    for raw in values:
        item = str(raw).strip()
        if item and item not in out:
            out.append(item)
    return out


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _topics_enabled_for_job(
    *,
    status: dict[str, Any],
    job_cfg: dict[str, Any],
    service_cfg: dict[str, Any],
) -> bool:
    if "topics_enabled" in status:
        return bool(status.get("topics_enabled"))
    upload_cfg = job_cfg.get("upload") or {}
    if isinstance(upload_cfg, dict) and "topics_enabled" in upload_cfg:
        return bool(upload_cfg.get("topics_enabled"))
    topics_cfg = dict(service_cfg.get("topics") or {})
    return bool(topics_cfg.get("enabled", False))


def _append_progress_run_record(
    *,
    runs_path: Path,
    job_id: str,
    status: dict[str, Any],
    hardware_key: str,
    speaker_mode: str,
    topics_enabled: bool,
    snippet_seconds: int,
    base_elapsed_s: float,
    final_elapsed_s: float,
    topics_prep_elapsed_s: float,
    topics_llm_elapsed_s: float,
) -> None:
    asr_timings = dict(status.get("asr_timings") or {})
    asr_total_s = max(0.0, float(_safe_float(asr_timings.get("total_s")) or 0.0))
    phase_seconds = {
        "snipping": round(max(0.0, float(base_elapsed_s) - asr_total_s), 6),
        "whisperx_prepare": round(max(0.0, float(_safe_float(asr_timings.get("prepare_s")) or 0.0)), 6),
        "whisperx_transcribe": round(max(0.0, float(_safe_float(asr_timings.get("transcribe_s")) or 0.0)), 6),
        "whisperx_align": round(max(0.0, float(_safe_float(asr_timings.get("align_s")) or 0.0)), 6),
        "whisperx_diarize": round(max(0.0, float(_safe_float(asr_timings.get("diarize_s")) or 0.0)), 6),
        "whisperx_finalize": round(max(0.0, float(_safe_float(asr_timings.get("finalize_s")) or 0.0)), 6),
        "topics_prep": round(max(0.0, float(topics_prep_elapsed_s)), 6),
    }
    if topics_enabled:
        phase_seconds["llm_topics"] = round(max(0.0, float(topics_llm_elapsed_s)), 6)
    else:
        phase_seconds["llm_topics_skipped"] = round(max(0.0, float(topics_llm_elapsed_s)), 6)

    aligner_load_s = _safe_float(asr_timings.get("aligner_load_s"))
    diarizer_load_s = _safe_float(asr_timings.get("diarizer_load_s"))
    if aligner_load_s is not None:
        phase_seconds["aligner_load_s"] = round(max(0.0, float(aligner_load_s)), 6)
    if diarizer_load_s is not None:
        phase_seconds["diarizer_load_s"] = round(max(0.0, float(diarizer_load_s)), 6)

    record = {
        "schema_version": "1.0",
        "run_id": str(job_id),
        "job_id": str(job_id),
        "ts_start_utc": str(status.get("started_at") or "").strip() or datetime.now(timezone.utc).isoformat(),
        "ts_end_utc": str(status.get("finished_at") or "").strip() or datetime.now(timezone.utc).isoformat(),
        "host_id": "",
        "worker_instance": "",
        "snippet_seconds": int(max(1, int(snippet_seconds))),
        "topics_enabled": bool(topics_enabled),
        "speaker_mode": str(speaker_mode),
        "chunks_count": 1,
        "config_key": "",
        "hardware_key": str(hardware_key),
        "phase_seconds": phase_seconds,
        "wait_seconds": {},
        "total_seconds": round(max(0.0, float(final_elapsed_s)), 6),
        "outcome": "done",
        "error_text": "",
    }
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    with runs_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


class _TopicsProgressTracker:
    """Upload-global progress owner for the remaining topics segment.

    The ASR worker owns progress through its own terminal completion.
    After the worker writes `done`, the coordinator keeps prep details as
    subphases and projects only the remaining topics segment on the top-level
    progress fields.
    """

    def __init__(self, *, status_path: Path) -> None:
        self._status_path = status_path
        self._base_progress = 0.0
        self._base_elapsed_s = 0.0
        self._eta_confidence = 0.0
        self._eta_hints: list[str] = []
        self._current_message = "Running..."
        self._current_status_owner = "api-topics"
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
        self._current_message = str(message or "").strip() or "Running..."
        self._current_status_owner = "api-topics"
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
            status_owner=self._current_status_owner,
            message=self._current_message,
            progress_mode="predictive_v1",
            eta_total_s=round(max(0.0, est_total), 3),
            eta_remaining_s=round(max(0.0, est_remaining), 3),
            elapsed_s=round(max(0.0, est_elapsed), 3),
            eta_confidence=round(float(self._eta_confidence), 3),
            eta_hints=list(self._eta_hints),
        )

    def start(self, *, message: str, status_owner: str = "api-topics") -> None:
        self._topics_started_t = time.monotonic()
        self._current_chunk_started_t = self._topics_started_t
        self._topics_done = 0
        self._topics_total = 0
        self._current_message = str(message or "").strip() or "Topics: preparing"
        self._current_status_owner = str(status_owner or "api-topics").strip() or "api-topics"
        self._write()

    def set_message(self, message: str, *, status_owner: str | None = None) -> None:
        self._current_message = str(message or "").strip() or self._current_message
        if status_owner is not None:
            self._current_status_owner = str(status_owner or "api-topics").strip() or "api-topics"
        self._write()

    def update(self, *, done_count: int, total_count: int, message: str, status_owner: str | None = None) -> None:
        safe_total = max(0, int(total_count))
        safe_done = max(0, min(int(done_count), safe_total))
        if safe_done != self._topics_done:
            self._current_chunk_started_t = time.monotonic()
        elif safe_total > 0 and self._current_chunk_started_t <= 0.0:
            self._current_chunk_started_t = time.monotonic()
        self._topics_total = safe_total
        self._topics_done = safe_done
        self._current_message = str(message or "").strip() or self._current_message
        if status_owner is not None:
            self._current_status_owner = str(status_owner or "api-topics").strip() or "api-topics"
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


class TopicsFlow:
    def __init__(
        self,
        *,
        service_cfg: dict[str, Any],
        progress_runs_path: Path,
        hardware_key: str,
        llm_wait_poll_s: float,
        llm_wait_timeout_s: float,
        stop_event: threading.Event,
    ) -> None:
        self._service_cfg = dict(service_cfg or {})
        self._progress_runs_path = progress_runs_path
        self._hardware_key = str(hardware_key)
        self._llm_wait_poll_s = max(0.1, float(llm_wait_poll_s))
        self._llm_wait_timeout_s = max(30.0, float(llm_wait_timeout_s))
        self._stop = stop_event

    def _patch_status(self, status_path: Path, **patch: Any) -> None:
        try:
            _write_status(status_path, **patch)
        except Exception:
            pass

    def _status_owner(self, key: str, default: str) -> str:
        status_owners = self._service_cfg.get("status_owners") or {}
        if isinstance(status_owners, dict):
            raw = str(status_owners.get(key) or "").strip()
            if raw:
                return raw
        return default

    def _enqueue_topics_tasks(
        self,
        *,
        job: JobPaths,
        manifest_path: Path,
        orig_stem: str,
        prompt_id: str,
    ) -> dict[int, str]:
        topics_cfg = dict(self._service_cfg.get("topics") or {})
        result_dir = (job.dir / "result").resolve()
        prompt_path = Path(str(topics_cfg.get("prompt_path") or "").strip())
        if not prompt_path.is_absolute():
            prompt_path = (Path(__file__).resolve().parents[3] / prompt_path).resolve()
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

    def run(self, *, job: JobPaths, status: dict[str, Any]) -> None:
        _append_log(job.log_path, "topics_claimed phase=done")
        service_cfg = self._service_cfg
        result_dir = (job.dir / "result").resolve()
        topics_cfg = dict(service_cfg.get("topics") or {})
        snip_cfg = dict(service_cfg.get("snip") or {})
        try:
            job_cfg = _read_json(job.job_path)
        except Exception:
            job_cfg = {}
        topics_enabled = _topics_enabled_for_job(status=status, job_cfg=job_cfg, service_cfg=service_cfg)
        speaker_mode = _normalize_speaker_mode(status.get("speaker_mode", "auto"))
        snippet_seconds = _safe_int(status.get("snippet_seconds"))
        if snippet_seconds is None or snippet_seconds <= 0:
            duration_ms = _safe_int(((job_cfg.get("input") or {}).get("duration_ms")))
            if duration_ms is not None and duration_ms > 0:
                snippet_seconds = int((int(duration_ms) + 999) // 1000)
            else:
                snippet_seconds = int(snip_cfg.get("minutes_default", 15)) * 60
        prompt_id = str(topics_cfg.get("prompt_id") or "topics_v1")
        coord_started_mono = time.monotonic()
        base_elapsed_s = max(0.0, float(_safe_float(status.get("elapsed_s")) or 0.0))
        progress_tracker = _TopicsProgressTracker(status_path=job.status_path) if topics_enabled else None
        if progress_tracker is not None:
            prediction = build_prediction(
                runs_path=self._progress_runs_path,
                hardware_key=self._hardware_key,
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
                message=str(status.get("message") or "Running..."),
            )
        orig_filename = str(status.get("orig_filename") or "").strip()
        orig_stem = Path(orig_filename).stem if orig_filename else "transcript"
        srt_path = _pick_srt_path(job=job, status=status)
        if srt_path is None:
            raise RuntimeError("No SRT found for topics handoff")

        if not topics_enabled:
            final_elapsed_s = base_elapsed_s
            final_eta_confidence = max(0.0, float(_safe_float(status.get("eta_confidence")) or 0.0))
            final_eta_hints = _unique_hints(status.get("eta_hints"))
            self._patch_status(
                job.status_path,
                state="done",
                phase="done",
                subphase="",
                status_owner=self._status_owner("api_topics", "api-topics"),
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
                topics_enabled=topics_enabled,
                topics_status="disabled",
                topics_warning="",
            )
            try:
                _append_progress_run_record(
                    runs_path=self._progress_runs_path,
                    job_id=job.job_id,
                    status=_read_json(job.status_path),
                    hardware_key=self._hardware_key,
                    speaker_mode=speaker_mode,
                    topics_enabled=topics_enabled,
                    snippet_seconds=snippet_seconds,
                    base_elapsed_s=base_elapsed_s,
                    final_elapsed_s=final_elapsed_s,
                    topics_prep_elapsed_s=0.0,
                    topics_llm_elapsed_s=0.0,
                )
            except Exception as e:
                _append_log(job.log_path, f"progress_runs_nonfatal error={e!r}")
            _append_log(job.log_path, "topics_skipped disabled")
            _append_log(job.log_path, "topics_done")
            return

        self._patch_status(
            job.status_path,
            state="running",
            phase="topics",
            subphase="speaker_lines",
            message="Topics: generating speaker_lines",
            topics_enabled=topics_enabled,
        )

        topics_prep_started_mono = time.monotonic()
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
        topics_prep_elapsed_s = max(0.0, time.monotonic() - topics_prep_started_mono)
        topics_llm_started_mono = time.monotonic()

        topics_status = "disabled"
        topics_warning = ""
        if topics_enabled:
            topics_status = "ok"
            try:
                progress_tracker.start(
                    message="Topics: enqueueing llm tasks",
                    status_owner=self._status_owner("llm_worker", "llm-worker"),
                )
                self._patch_status(
                    job.status_path,
                    phase="topics",
                    subphase="queue",
                    status_owner=self._status_owner("llm_worker", "llm-worker"),
                    message="Topics: enqueueing llm tasks",
                )
                task_ids = self._enqueue_topics_tasks(
                    job=job,
                    manifest_path=manifest_path,
                    orig_stem=orig_stem,
                    prompt_id=prompt_id,
                )
                self._patch_status(
                    job.status_path,
                    phase="topics",
                    subphase="wait",
                    status_owner=self._status_owner("llm_worker", "llm-worker"),
                    message=f"Topics: waiting for 0/{len(task_ids)} llm tasks",
                )
                progress_tracker.update(
                    done_count=0,
                    total_count=len(task_ids),
                    message=f"Topics: waiting for 0/{len(task_ids)} llm tasks",
                    status_owner=self._status_owner("llm_worker", "llm-worker"),
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
                    salvaged_chunks = int(report.get("salvaged_chunks") or 0)
                    if salvaged_chunks > 0:
                        _append_log(
                            job.log_path,
                            f"topics_validation_salvaged chunks={salvaged_chunks} report={report_path.name}",
                        )
                    merged_path = (result_dir / f"{orig_stem}_{prompt_id}_merged.json").resolve()
                    merge_topics(
                        manifest_path=manifest_path,
                        parsed_dir=result_dir,
                        orig_stem=orig_stem,
                        prompt_id=prompt_id,
                        out_merged_path=merged_path,
                    )
            except Exception as e:
                progress_tracker.set_message(
                    f"Topics failed: {e}",
                    status_owner=self._status_owner("api_topics", "api-topics"),
                )
                topics_status = "failed"
                topics_warning = str(e)
                _append_log(job.log_path, f"topics_nonfatal error={e!r}")
        topics_llm_elapsed_s = max(0.0, time.monotonic() - topics_llm_started_mono)

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
            status_owner=self._status_owner("api_topics", "api-topics"),
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
            topics_enabled=topics_enabled,
        )
        try:
            _append_progress_run_record(
                runs_path=self._progress_runs_path,
                job_id=job.job_id,
                status=_read_json(job.status_path),
                hardware_key=self._hardware_key,
                speaker_mode=speaker_mode,
                topics_enabled=topics_enabled,
                snippet_seconds=snippet_seconds,
                base_elapsed_s=base_elapsed_s,
                final_elapsed_s=final_elapsed_s,
                topics_prep_elapsed_s=topics_prep_elapsed_s,
                topics_llm_elapsed_s=topics_llm_elapsed_s,
            )
        except Exception as e:
            _append_log(job.log_path, f"progress_runs_nonfatal error={e!r}")
        _append_log(job.log_path, "topics_done")
