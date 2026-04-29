from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from upload.jobs.status import _write_status
from upload._util import _safe_float
from upload.pipeline.progress_plan import DEFAULTS_SECONDS


def _unique_hints(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    out: list[str] = []
    for raw in values:
        item = str(raw).strip()
        if item and item not in out:
            out.append(item)
    return out


def status_elapsed_for_topics(status: dict[str, Any]) -> float:
    base_elapsed = max(0.0, float(_safe_float(status.get("elapsed_s")) or 0.0))
    asr_elapsed = max(0.0, float(_safe_float(status.get("asr_elapsed_s")) or 0.0))
    return max(0.0, base_elapsed + asr_elapsed)


def status_eta_confidence(status: dict[str, Any]) -> float:
    local = _safe_float(status.get("eta_confidence"))
    if local is not None:
        return max(0.0, float(local))
    asr_local = _safe_float(status.get("asr_eta_confidence"))
    if asr_local is not None:
        return max(0.0, float(asr_local))
    return 0.0


def status_eta_hints(status: dict[str, Any]) -> list[str]:
    return _unique_hints(status.get("eta_hints")) or _unique_hints(status.get("asr_eta_hints"))


class TopicsProgressTracker:
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
            state="running",
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
