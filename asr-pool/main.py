from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi import Depends, FastAPI, Header, Request
from fastapi.responses import JSONResponse


ROOT_PATH = os.getenv("TRANSCRIBE_ASR_POOL_ROOT_PATH", "")
app = FastAPI(root_path=ROOT_PATH)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _iso_utc(ts: float | None = None) -> str:
    if ts is None:
        ts = time.time()
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_hash(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _seconds_between_utc(start_utc: str | None, end_utc: str | None) -> float | None:
    try:
        if not start_utc or not end_utc:
            return None
        a = datetime.fromisoformat(str(start_utc).replace("Z", "+00:00"))
        b = datetime.fromisoformat(str(end_utc).replace("Z", "+00:00"))
        return max(0.0, float((b - a).total_seconds()))
    except Exception:
        return None


def _parse_utc_unix(value: str | None) -> float | None:
    try:
        if not value:
            return None
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return float(dt.timestamp())
    except Exception:
        return None


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.asr.blob_store import AsrBlobError, resolve_blob_ref_to_local_path
from asr_contract import (
    AsrRequestError,
    build_error_response,
    prepare_request,
)


def _cfg_int(name: str, default: int, *, min_value: int = 0) -> int:
    try:
        return max(min_value, int(str(os.getenv(name, str(default))).strip() or str(default)))
    except Exception:
        return max(min_value, int(default))


def _cfg_str(name: str, default: str) -> str:
    return str(os.getenv(name, default) or default).strip()


def _cfg_bool(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "") or "").strip().lower()
    if not raw:
        return bool(default)
    if raw in {"1", "true", "yes", "on", "y"}:
        return True
    if raw in {"0", "false", "no", "off", "n"}:
        return False
    return bool(default)


def _error(
    status_code: int,
    *,
    code: str,
    message: str,
    retryable: bool | None = None,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    payload: dict[str, Any] = {
        "code": str(code),
        "message": str(message),
    }
    if retryable is not None:
        payload["retryable"] = bool(retryable)
    if details:
        payload["details"] = dict(details)
    return JSONResponse(status_code=int(status_code), content=payload)


@dataclass
class _Record:
    request_id: str
    payload_hash: str
    request: dict[str, Any]
    profile_id: str
    priority: str
    live_lane: str
    queue_key: str
    state: str
    submitted_at_utc: str
    started_at_utc: str | None = None
    finished_at_utc: str | None = None
    stage: str | None = None
    stage_started_at_utc: str | None = None
    retryable: bool | None = None
    response: dict[str, Any] | None = None
    error: dict[str, Any] | None = None


class AsrPoolService:
    def __init__(self) -> None:
        self._runner_slots = _cfg_int("TRANSCRIBE_ASR_POOL_RUNNER_SLOTS", 2, min_value=1)
        self._queue_limits = {
            "interactive": _cfg_int("TRANSCRIBE_ASR_POOL_QUEUE_LIMIT_INTERACTIVE", 8, min_value=1),
            "normal": _cfg_int("TRANSCRIBE_ASR_POOL_QUEUE_LIMIT_NORMAL", 20, min_value=1),
            "background": _cfg_int("TRANSCRIBE_ASR_POOL_QUEUE_LIMIT_BACKGROUND", 50, min_value=1),
        }
        self._timeouts_s = {
            "interactive": _cfg_int("TRANSCRIBE_ASR_POOL_TIMEOUT_INTERACTIVE_S", 30, min_value=1),
            "normal": _cfg_int("TRANSCRIBE_ASR_POOL_TIMEOUT_NORMAL_S", 120, min_value=1),
            "background": _cfg_int("TRANSCRIBE_ASR_POOL_TIMEOUT_BACKGROUND_S", 300, min_value=1),
        }
        self._warm_start_enabled = _cfg_bool("TRANSCRIBE_ASR_POOL_WARM_START_ENABLED", True)
        self._warm_start_timeout_s = _cfg_int("TRANSCRIBE_ASR_POOL_WARM_START_TIMEOUT_S", 180, min_value=1)
        self._watchdog_enabled = _cfg_bool("TRANSCRIBE_ASR_POOL_WATCHDOG_ENABLED", True)
        self._watchdog_interval_s = max(
            0.2,
            float(_cfg_int("TRANSCRIBE_ASR_POOL_WATCHDOG_INTERVAL_MS", 2000, min_value=200)) / 1000.0,
        )
        self._watchdog_recover_timeout_s = _cfg_int(
            "TRANSCRIBE_ASR_POOL_WATCHDOG_RECOVER_TIMEOUT_S",
            30,
            min_value=1,
        )
        self._records_max = _cfg_int("TRANSCRIBE_ASR_POOL_RECORDS_MAX", 10000, min_value=100)
        self._records_ttl_s = {
            "completed": _cfg_int("TRANSCRIBE_ASR_POOL_RECORDS_TTL_COMPLETED_S", 900, min_value=10),
            "failed": _cfg_int("TRANSCRIBE_ASR_POOL_RECORDS_TTL_FAILED_S", 1800, min_value=10),
            "cancelled": _cfg_int("TRANSCRIBE_ASR_POOL_RECORDS_TTL_CANCELLED_S", 600, min_value=10),
        }
        self._records_prune_interval_s = _cfg_int("TRANSCRIBE_ASR_POOL_RECORDS_PRUNE_INTERVAL_S", 30, min_value=1)
        self._records_pruned_total = 0
        self._records_pruned_ttl_total = 0
        self._records_pruned_overflow_total = 0
        self._records_last_prune_utc = ""
        self._records_last_prune_reason = ""
        self._records_last_prune_count = 0
        self._work_root = (
            Path(_cfg_str("TRANSCRIBE_ASR_POOL_WORK_ROOT", str((_repo_root() / "data" / "asr_pool").resolve())))
            .expanduser()
            .resolve()
        )
        self._work_root.mkdir(parents=True, exist_ok=True)
        self._interactive_burst_max = _cfg_int("TRANSCRIBE_ASR_POOL_INTERACTIVE_BURST_MAX", 8, min_value=1)
        self._warm_clients: list[Any] = []
        try:
            from whisperx_runner_client import _AsrPoolWarmRunnerClient

            self._warm_clients = [_AsrPoolWarmRunnerClient() for _ in range(self._runner_slots)]
        except Exception:
            self._warm_clients = []

        self._records: dict[str, _Record] = {}
        self._queues: dict[str, deque[str]] = {
            "interactive": deque(),
            "normal": deque(),
            "background": deque(),
        }
        self._lock = asyncio.Lock()
        self._cond = asyncio.Condition(self._lock)
        self._tasks: list[asyncio.Task[None]] = []
        self._stopping = False
        self._interactive_burst_count = 0
        self._noninteractive_next = "normal"
        self._watchdog_restart_count: list[int] = [0 for _ in range(max(0, int(self._runner_slots)))]
        self._last_records_prune_mono = 0.0
        self._stage_poll_interval_s = max(0.05, float(_cfg_int("TRANSCRIBE_ASR_POOL_STAGE_POLL_MS", 150, min_value=50)) / 1000.0)

    async def start(self) -> None:
        should_prewarm = False
        async with self._lock:
            if self._tasks:
                return
            self._stopping = False
            for idx in range(self._runner_slots):
                task = asyncio.create_task(self._runner_loop(idx), name=f"asr-pool-runner-{idx}")
                self._tasks.append(task)
            should_prewarm = bool(self._warm_clients) and bool(self._warm_start_enabled)
            self._emit_event(
                "pool_started",
                runner_slots=int(self._runner_slots),
                executor_mode="warm_local",
                warm_start_enabled=bool(self._warm_start_enabled),
                watchdog_enabled=bool(self._watchdog_enabled),
                watchdog_interval_s=round(float(self._watchdog_interval_s), 3),
                watchdog_recover_timeout_s=int(self._watchdog_recover_timeout_s),
                records_max=int(self._records_max),
                records_ttl_completed_s=int(self._records_ttl_s["completed"]),
                records_ttl_failed_s=int(self._records_ttl_s["failed"]),
                records_ttl_cancelled_s=int(self._records_ttl_s["cancelled"]),
                interactive_burst_max=int(self._interactive_burst_max),
            )
            if bool(self._watchdog_enabled):
                watchdog_task = asyncio.create_task(self._watchdog_loop(), name="asr-pool-watchdog")
                self._tasks.append(watchdog_task)
        if should_prewarm:
            await self._prewarm_runners()

    async def stop(self) -> None:
        async with self._lock:
            self._stopping = True
            self._cond.notify_all()
            tasks = list(self._tasks)
            self._tasks.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for client in list(self._warm_clients):
            try:
                client.shutdown(reason="pool_shutdown")
            except Exception:
                continue
        self._emit_event("pool_stopped")

    async def _prewarm_runners(self) -> None:
        async def _run_one(slot_idx: int, client: Any) -> None:
            try:
                await asyncio.to_thread(self._prewarm_one_runner, slot_idx, client)
            except Exception as e:
                try:
                    print(f"asr_pool prewarm slot={slot_idx} failed: {type(e).__name__}: {e}", flush=True)
                except Exception:
                    pass

        coros = [_run_one(idx, client) for idx, client in enumerate(list(self._warm_clients))]
        if not coros:
            return
        try:
            await asyncio.wait_for(
                asyncio.gather(*coros, return_exceptions=True),
                timeout=float(self._warm_start_timeout_s),
            )
            self._emit_event(
                "pool_prewarm_done",
                slots=int(len(coros)),
                timeout_s=int(self._warm_start_timeout_s),
            )
        except asyncio.TimeoutError:
            try:
                print(
                    f"asr_pool prewarm timeout after {self._warm_start_timeout_s}s (slots={len(coros)})",
                    flush=True,
                )
            except Exception:
                pass
            self._emit_event(
                "pool_prewarm_timeout",
                slots=int(len(coros)),
                timeout_s=int(self._warm_start_timeout_s),
            )

    @staticmethod
    def _warm_client_health(client: Any) -> tuple[bool, int | None]:
        """
        Returns (alive, pid). A warm runner is alive if it has a process and poll() is None.
        """
        lock = getattr(client, "_lock", None)
        if lock is None:
            return False, None
        with lock:
            proc = getattr(client, "_proc", None)
            if proc is None:
                return False, None
            pid = int(getattr(proc, "pid", 0) or 0) or None
            try:
                alive = bool(proc.poll() is None)
            except Exception:
                alive = False
            return alive, pid

    @staticmethod
    def _recover_warm_client(slot_idx: int, client: Any, *, reason: str) -> dict[str, Any]:
        lock = getattr(client, "_lock", None)
        if lock is None:
            raise RuntimeError("warm_client_missing_lock")
        with lock:
            old_proc = getattr(client, "_proc", None)
            old_pid = int(getattr(old_proc, "pid", 0) or 0) or None
            if old_proc is not None:
                try:
                    if old_proc.poll() is not None:
                        shutdown = getattr(client, "_shutdown_locked", None)
                        if callable(shutdown):
                            shutdown(reason=f"watchdog_{reason}")
                except Exception:
                    pass
        prewarm = getattr(client, "prewarm", None)
        if callable(prewarm):
            prewarm()
        else:
            ensure = getattr(client, "_ensure_runner_locked", None)
            if callable(ensure):
                ensure()
            else:
                raise RuntimeError("warm_client_missing_prewarm_and_ensure")
        alive, new_pid = AsrPoolService._warm_client_health(client)
        if not alive:
            raise RuntimeError("watchdog_recovery_runner_not_alive")
        return {
            "slot_idx": int(slot_idx),
            "old_pid": old_pid,
            "new_pid": int(new_pid or 0) or None,
        }

    async def _watchdog_loop(self) -> None:
        while True:
            await asyncio.sleep(float(self._watchdog_interval_s))
            if self._stopping:
                return
            if not self._warm_clients:
                continue
            for slot_idx, client in enumerate(list(self._warm_clients)):
                if self._stopping:
                    return
                alive, pid = await asyncio.to_thread(self._warm_client_health, client)
                if alive:
                    continue
                self._emit_event(
                    "runner_watchdog_detected_unhealthy",
                    slot_idx=int(slot_idx),
                    pid=(int(pid) if pid is not None else None),
                )
                try:
                    rec = await asyncio.wait_for(
                        asyncio.to_thread(
                            self._recover_warm_client,
                            int(slot_idx),
                            client,
                            reason="unhealthy",
                        ),
                        timeout=float(self._watchdog_recover_timeout_s),
                    )
                    self._watchdog_restart_count[slot_idx] = int(self._watchdog_restart_count[slot_idx]) + 1
                    self._emit_event(
                        "runner_watchdog_recovered",
                        slot_idx=int(slot_idx),
                        old_pid=rec.get("old_pid"),
                        new_pid=rec.get("new_pid"),
                        restart_count=int(self._watchdog_restart_count[slot_idx]),
                    )
                except asyncio.TimeoutError:
                    self._emit_event(
                        "runner_watchdog_recover_timeout",
                        slot_idx=int(slot_idx),
                        timeout_s=int(self._watchdog_recover_timeout_s),
                    )
                except Exception as e:
                    self._emit_event(
                        "runner_watchdog_recover_failed",
                        slot_idx=int(slot_idx),
                        error=f"{type(e).__name__}: {e}",
                    )

    def _prewarm_one_runner(self, slot_idx: int, client: Any) -> None:
        try:
            prewarm = getattr(client, "prewarm", None)
            if callable(prewarm):
                prewarm()
                return
            ensure = getattr(client, "_ensure_runner_locked", None)
            if callable(ensure):
                lock = getattr(client, "_lock", None)
                if lock is not None:
                    with lock:
                        ensure()
                else:
                    ensure()
                return
            raise RuntimeError("warm client has no prewarm method")
        except Exception as e:
            raise RuntimeError(f"slot={slot_idx}: {type(e).__name__}: {e}") from e

    async def submit(self, raw_payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        try:
            prepared = prepare_request(raw_payload)
        except AsrRequestError as e:
            self._emit_event(
                "submit_rejected_validation",
                code=str(e.code),
                message=str(e),
            )
            return 400, {
                "code": str(e.code),
                "message": str(e),
                "retryable": False,
                "details": dict(e.details or {}),
            }

        request_id = str(prepared.get("request_id") or "").strip()
        payload_hash = _json_hash(prepared)
        priority = str(prepared.get("priority") or "normal").strip().lower() or "normal"
        live_lane = "single"
        queue_key = self._queue_key_for(priority=priority)

        async with self._lock:
            self._maybe_prune_records_unlocked(reason="submit", force=False)
            existing = self._records.get(request_id)
            if existing is not None:
                if existing.payload_hash != payload_hash:
                    self._emit_event(
                        "submit_conflict",
                        request_id=str(request_id),
                        priority=str(priority),
                    )
                    return 409, {
                        "code": "ASR_REQUEST_ID_CONFLICT",
                        "message": "request_id already exists with different payload",
                        "retryable": False,
                        "details": {"request_id": request_id},
                    }
                self._emit_event(
                    "submit_idempotent_hit",
                    request_id=str(request_id),
                    state=str(existing.state),
                )
                return 200, self._to_lifecycle(existing)

            if self._priority_depth(priority) >= int(self._queue_limits.get(priority, 1)):
                self._emit_event(
                    "submit_rejected_queue_full",
                    request_id=str(request_id),
                    priority=str(priority),
                    queue_depth=self._queue_depth_snapshot_unlocked(),
                    queue_limit=int(self._queue_limits.get(priority, 1)),
                )
                return 429, {
                    "code": "ASR_QUEUE_FULL",
                    "message": f"{priority} queue depth limit reached",
                    "retryable": True,
                    "details": {
                        "priority": priority,
                        "queue_depth": int(self._priority_depth(priority)),
                        "queue_limit": int(self._queue_limits.get(priority, 1)),
                    },
                }

            rec = _Record(
                request_id=request_id,
                payload_hash=payload_hash,
                request=prepared,
                profile_id=str(prepared.get("profile_id") or ""),
                priority=priority,
                live_lane=live_lane,
                queue_key=queue_key,
                state="queued",
                submitted_at_utc=_iso_utc(),
            )
            self._records[request_id] = rec
            self._queues[queue_key].append(request_id)
            queue_position = int(len(self._queues[queue_key]))
            self._emit_event(
                "submit_accepted",
                request_id=str(request_id),
                profile_id=str(rec.profile_id),
                priority=str(rec.priority),
                live_lane=str(rec.live_lane),
                queue_key=str(queue_key),
                queue_position=int(queue_position),
                queue_depth=self._queue_depth_snapshot_unlocked(),
            )
            # Wake all runner loops so slot-affinity constraints (e.g. upload_full -> slot 0)
            # cannot deadlock when a non-eligible slot was the only waiter notified.
            self._cond.notify_all()
            return 202, self._to_lifecycle(rec)

    async def get_request(self, request_id: str) -> tuple[int, dict[str, Any]]:
        rid = str(request_id or "").strip()
        async with self._lock:
            self._maybe_prune_records_unlocked(reason="get_request", force=False)
            rec = self._records.get(rid)
            if rec is None:
                return 404, {
                    "code": "ASR_REQUEST_NOT_FOUND",
                    "message": "request_id not found",
                    "retryable": False,
                    "details": {"request_id": rid},
                }
            return 200, self._to_lifecycle(rec)

    async def cancel(self, request_id: str) -> tuple[int, dict[str, Any]]:
        rid = str(request_id or "").strip()
        async with self._lock:
            self._maybe_prune_records_unlocked(reason="cancel", force=False)
            rec = self._records.get(rid)
            if rec is None:
                self._emit_event("cancel_not_found", request_id=str(rid))
                return 404, {
                    "code": "ASR_REQUEST_NOT_FOUND",
                    "message": "request_id not found",
                    "retryable": False,
                    "details": {"request_id": rid},
                }
            if rec.state == "queued":
                rec.state = "cancelled"
                rec.finished_at_utc = _iso_utc()
                self._remove_from_queue_unlocked(rid, rec.queue_key)
                self._emit_event(
                    "cancel_queued",
                    request_id=str(rec.request_id),
                    priority=str(rec.priority),
                    live_lane=str(rec.live_lane),
                    queue_depth=self._queue_depth_snapshot_unlocked(),
                )
            elif rec.state == "running":
                rec.state = "cancel_requested"
                self._emit_event(
                    "cancel_running",
                    request_id=str(rec.request_id),
                    priority=str(rec.priority),
                    live_lane=str(rec.live_lane),
                )
            else:
                self._emit_event(
                    "cancel_noop",
                    request_id=str(rec.request_id),
                    state=str(rec.state),
                )
            return 200, {
                "request_id": rec.request_id,
                "state": rec.state,
                "message": "cancel accepted",
            }

    async def pool_status(self) -> dict[str, Any]:
        async with self._lock:
            queued_interactive = self._priority_depth("interactive")
            queued_normal = self._priority_depth("normal")
            queued_background = self._priority_depth("background")
            running = sum(1 for rec in self._records.values() if rec.state in {"running", "cancel_requested"})
            return {
                "service": "asr-runtime-pool",
                "version": "1.0.0-skeleton",
                "now_utc": _iso_utc(),
                "slots_total": int(self._runner_slots),
                "slots_busy": int(running),
                "slots_available": int(max(0, self._runner_slots - running)),
                "slots_by_priority": {
                    "interactive": int(running),
                    "normal": 0,
                    "background": 0,
                },
                "queue_limits": dict(self._queue_limits),
                "queue_depth": {
                    "interactive": int(queued_interactive),
                    "normal": int(queued_normal),
                    "background": int(queued_background),
                },
                "request_timeouts_s": dict(self._timeouts_s),
                "queue_wait_ms_p95": {},
                "blob_fetch_ms_p95": None,
                "watchdog": {
                    "enabled": bool(self._watchdog_enabled),
                    "interval_s": round(float(self._watchdog_interval_s), 3),
                    "recover_timeout_s": int(self._watchdog_recover_timeout_s),
                    "restarts_by_slot": [int(v) for v in self._watchdog_restart_count],
                },
                "records": {
                    "count": int(len(self._records)),
                    "max": int(self._records_max),
                    "ttl_s": {
                        "completed": int(self._records_ttl_s["completed"]),
                        "failed": int(self._records_ttl_s["failed"]),
                        "cancelled": int(self._records_ttl_s["cancelled"]),
                    },
                    "prune_interval_s": int(self._records_prune_interval_s),
                    "pruned_total": int(self._records_pruned_total),
                    "pruned_ttl_total": int(self._records_pruned_ttl_total),
                    "pruned_overflow_total": int(self._records_pruned_overflow_total),
                    "last_prune_utc": str(self._records_last_prune_utc or ""),
                    "last_prune_reason": str(self._records_last_prune_reason or ""),
                    "last_prune_count": int(self._records_last_prune_count),
                },
                "scheduling_policy": {
                    "interactive_single_queue": True,
                    "interactive_burst_max": int(self._interactive_burst_max),
                    "fairness_mode": "burst_then_round_robin_noninteractive",
                },
            }

    def _queue_key_for(self, *, priority: str) -> str:
        p = str(priority or "normal").strip().lower() or "normal"
        if p == "interactive":
            return "interactive"
        if p == "background":
            return "background"
        return "normal"

    def _priority_depth(self, priority: str) -> int:
        p = str(priority or "").strip().lower()
        if p == "interactive":
            return int(len(self._queues["interactive"]))
        if p == "background":
            return int(len(self._queues["background"]))
        return int(len(self._queues["normal"]))

    def _queue_depth_snapshot_unlocked(self) -> dict[str, int]:
        return {
            "interactive": int(self._priority_depth("interactive")),
            "normal": int(self._priority_depth("normal")),
            "background": int(self._priority_depth("background")),
        }

    def _has_running_background_unlocked(self) -> bool:
        for rec in self._records.values():
            if str(rec.state) == "running" and str(rec.priority) == "background":
                return True
        return False

    def _emit_event(self, event: str, **fields: Any) -> None:
        payload: dict[str, Any] = {
            "ts_utc": _iso_utc(),
            "component": "asr_runtime_pool",
            "event": str(event),
        }
        payload.update({k: v for k, v in fields.items() if v is not None})
        try:
            print("ASR_POOL_EVENT " + json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
        except Exception:
            pass

    def _noninteractive_order_unlocked(self) -> list[str]:
        if str(self._noninteractive_next) == "background":
            return ["background", "normal"]
        return ["normal", "background"]

    def _dequeue_order_unlocked(self) -> list[str]:
        interactive_ready = self._priority_depth("interactive") > 0
        normal_ready = self._priority_depth("normal") > 0
        background_ready = self._priority_depth("background") > 0
        noninteractive_ready = normal_ready or background_ready
        prefer_noninteractive = (
            interactive_ready
            and noninteractive_ready
            and int(self._interactive_burst_count) >= int(self._interactive_burst_max)
        )
        if prefer_noninteractive:
            return self._noninteractive_order_unlocked() + ["interactive"]
        return ["interactive"] + self._noninteractive_order_unlocked()

    def _note_dequeue_key_unlocked(self, queue_key: str) -> None:
        key = str(queue_key or "").strip().lower()
        if key.startswith("interactive_"):
            self._interactive_burst_count = int(self._interactive_burst_count) + 1
            return
        self._interactive_burst_count = 0
        if key == "normal":
            self._noninteractive_next = "background"
        elif key == "background":
            self._noninteractive_next = "normal"

    def _remove_from_queue_unlocked(self, request_id: str, queue_key: str) -> None:
        q = self._queues.get(queue_key)
        if q is None:
            return
        try:
            q.remove(request_id)
        except ValueError:
            return

    def _to_lifecycle(self, rec: _Record) -> dict[str, Any]:
        queue_position = None
        if rec.state == "queued":
            q = self._queues.get(rec.queue_key)
            if q is not None:
                try:
                    queue_position = int(list(q).index(rec.request_id) + 1)
                except ValueError:
                    queue_position = None
        return {
            "request_id": rec.request_id,
            "state": rec.state,
            "profile_id": rec.profile_id,
            "priority": rec.priority,
            "queue_position": queue_position,
            "submitted_at_utc": rec.submitted_at_utc,
            "started_at_utc": rec.started_at_utc,
            "finished_at_utc": rec.finished_at_utc,
            "stage": rec.stage,
            "stage_started_at_utc": rec.stage_started_at_utc,
            "retryable": rec.retryable,
            "response": rec.response,
            "error": rec.error,
        }

    async def _poll_stage_updates(self, *, request_id: str, progress_path: Path, stop_event: asyncio.Event) -> None:
        last_stage = ""
        while True:
            if stop_event.is_set():
                break
            try:
                obj = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
                stage = str(obj.get("stage") or "").strip().lower()
                stage_ts = str(obj.get("ts_utc") or "").strip()
                if stage and stage != last_stage:
                    async with self._lock:
                        rec = self._records.get(str(request_id))
                        if rec is not None and rec.state in {"running", "cancel_requested"}:
                            rec.stage = str(stage)
                            rec.stage_started_at_utc = str(stage_ts or _iso_utc())
                    self._emit_event(
                        "request_stage",
                        request_id=str(request_id),
                        stage=str(stage),
                    )
                    last_stage = str(stage)
            except Exception:
                pass
            await asyncio.sleep(float(self._stage_poll_interval_s))

    def _prune_records_unlocked(self, *, reason: str) -> int:
        now_unix = time.time()
        pruned_ttl = 0
        pruned_overflow = 0

        removable_by_ttl: list[str] = []
        for rid, rec in self._records.items():
            state = str(rec.state or "").strip().lower()
            if state not in {"completed", "failed", "cancelled"}:
                continue
            ttl_s = int(self._records_ttl_s.get(state, 0))
            if ttl_s <= 0:
                continue
            ref_unix = _parse_utc_unix(rec.finished_at_utc) or _parse_utc_unix(rec.submitted_at_utc)
            if ref_unix is None:
                continue
            if (now_unix - ref_unix) >= float(ttl_s):
                removable_by_ttl.append(str(rid))
        for rid in removable_by_ttl:
            self._records.pop(rid, None)
            pruned_ttl += 1

        overflow = int(len(self._records) - int(self._records_max))
        if overflow > 0:
            terminal_rows: list[tuple[float, str]] = []
            for rid, rec in self._records.items():
                state = str(rec.state or "").strip().lower()
                if state not in {"completed", "failed", "cancelled"}:
                    continue
                ref_unix = _parse_utc_unix(rec.finished_at_utc) or _parse_utc_unix(rec.submitted_at_utc) or now_unix
                terminal_rows.append((float(ref_unix), str(rid)))
            terminal_rows.sort(key=lambda row: row[0])
            for _ref, rid in terminal_rows[:overflow]:
                if self._records.pop(rid, None) is not None:
                    pruned_overflow += 1

        pruned_total = int(pruned_ttl + pruned_overflow)
        if pruned_total > 0:
            self._records_pruned_total = int(self._records_pruned_total + pruned_total)
            self._records_pruned_ttl_total = int(self._records_pruned_ttl_total + pruned_ttl)
            self._records_pruned_overflow_total = int(self._records_pruned_overflow_total + pruned_overflow)
            self._records_last_prune_utc = _iso_utc(now_unix)
            self._records_last_prune_reason = str(reason or "")
            self._records_last_prune_count = int(pruned_total)
            self._emit_event(
                "records_pruned",
                reason=str(reason or ""),
                pruned_total=int(pruned_total),
                pruned_ttl=int(pruned_ttl),
                pruned_overflow=int(pruned_overflow),
                records_remaining=int(len(self._records)),
            )
        return int(pruned_total)

    def _maybe_prune_records_unlocked(self, *, reason: str, force: bool = False) -> int:
        now_mono = time.monotonic()
        if not force and (now_mono - float(self._last_records_prune_mono)) < float(self._records_prune_interval_s):
            return 0
        self._last_records_prune_mono = float(now_mono)
        return int(self._prune_records_unlocked(reason=str(reason or "periodic")))

    async def _dequeue_next_request_id(self, slot_idx: int) -> str:
        async with self._cond:
            while True:
                if self._stopping:
                    raise asyncio.CancelledError()
                for key in self._dequeue_order_unlocked():
                    # Keep heavy upload/background ASR single-flight across the pool.
                    if key == "background" and self._has_running_background_unlocked():
                        continue
                    queue = self._queues[key]
                    while queue:
                        rid = queue.popleft()
                        rec = self._records.get(rid)
                        if rec is None:
                            continue
                        if rec.state != "queued":
                            continue
                        # Upload full-profile requests may include align+diarize and can
                        # overflow VRAM if multiple warm slots each load aux models.
                        # Route these to slot 0 only; keep other traffic fully shared.
                        if (
                            key == "background"
                            and str(rec.profile_id or "") == "upload_full"
                            and int(slot_idx) != 0
                        ):
                            queue.appendleft(rid)
                            break
                        if key == "background" and self._has_running_background_unlocked():
                            queue.appendleft(rid)
                            break
                        self._note_dequeue_key_unlocked(key)
                        return rid
                await self._cond.wait()

    async def _runner_loop(self, slot_idx: int) -> None:
        while True:
            rid = await self._dequeue_next_request_id(slot_idx)
            progress_path = (self._work_root / str(rid) / f"slot_{slot_idx}" / "_progress.json").resolve()
            try:
                progress_path.parent.mkdir(parents=True, exist_ok=True)
                progress_path.unlink(missing_ok=True)
            except Exception:
                pass
            async with self._lock:
                rec = self._records.get(rid)
                if rec is None:
                    continue
                if rec.state != "queued":
                    continue
                rec.state = "running"
                rec.started_at_utc = _iso_utc()
                rec.stage = "dispatch"
                rec.stage_started_at_utc = rec.started_at_utc
                request = dict(rec.request)
                timeout_s = int(self._timeouts_s.get(rec.priority, 120))
                queue_wait_s = _seconds_between_utc(rec.submitted_at_utc, rec.started_at_utc)
                self._emit_event(
                    "request_started",
                    request_id=str(rec.request_id),
                    profile_id=str(rec.profile_id),
                    priority=str(rec.priority),
                    live_lane=str(rec.live_lane),
                    queue_key=str(rec.queue_key),
                    slot_idx=int(slot_idx),
                    timeout_s=int(timeout_s),
                    queue_wait_s=(round(float(queue_wait_s), 3) if queue_wait_s is not None else None),
                    queue_depth=self._queue_depth_snapshot_unlocked(),
                )

            stage_stop = asyncio.Event()
            stage_task = asyncio.create_task(
                self._poll_stage_updates(
                    request_id=str(rid),
                    progress_path=progress_path,
                    stop_event=stage_stop,
                ),
                name=f"asr-pool-stage-{slot_idx}-{rid}",
            )
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        self._execute_request,
                        request=request,
                        slot_idx=slot_idx,
                        progress_path=progress_path,
                    ),
                    timeout=float(timeout_s),
                )
            except asyncio.TimeoutError:
                response = build_error_response(
                    request=request,
                    code="ASR_REQUEST_TIMEOUT",
                    message=f"ASR request exceeded timeout ({timeout_s}s)",
                    retryable=True,
                    details={"timeout_s": int(timeout_s)},
                )
            except Exception as e:
                response = build_error_response(
                    request=request,
                    code="ASR_RUNTIME_FAILURE",
                    message=f"ASR runtime failure: {type(e).__name__}: {e}",
                    retryable=True,
                    details={"exc_type": type(e).__name__},
                )
            finally:
                stage_stop.set()
                try:
                    await stage_task
                except Exception:
                    pass

            ok = bool(response.get("ok", False))
            async with self._lock:
                rec2 = self._records.get(rid)
                if rec2 is None:
                    continue
                rec2.finished_at_utc = _iso_utc()
                rec2.stage = "completed" if ok else "failed"
                rec2.stage_started_at_utc = rec2.finished_at_utc
                if rec2.state == "cancel_requested":
                    rec2.state = "cancelled"
                    rec2.response = None
                    rec2.error = None
                    rec2.retryable = None
                    rec2.stage = "cancelled"
                else:
                    rec2.response = dict(response)
                    rec2.error = dict(response.get("error") or {}) if not ok else None
                    rec2.retryable = bool((rec2.error or {}).get("retryable", False)) if not ok else None
                    rec2.state = "completed" if ok else "failed"
                runtime_meta = dict((rec2.response or {}).get("runtime") or {})
                err = dict(rec2.error or {})
                exec_s = _seconds_between_utc(rec2.started_at_utc, rec2.finished_at_utc)
                self._emit_event(
                    "request_finished",
                    request_id=str(rec2.request_id),
                    profile_id=str(rec2.profile_id),
                    priority=str(rec2.priority),
                    live_lane=str(rec2.live_lane),
                    slot_idx=int(slot_idx),
                    state=str(rec2.state),
                    ok=bool(ok),
                    retryable=(None if rec2.retryable is None else bool(rec2.retryable)),
                    execution_s=(round(float(exec_s), 3) if exec_s is not None else None),
                    error_code=(str(err.get("code")) if err else None),
                    error_message=(str(err.get("message")) if err else None),
                    runner_kind=(str(runtime_meta.get("runner_kind")) if runtime_meta else None),
                    runner_reused=(runtime_meta.get("runner_reused") if runtime_meta else None),
                    transport=(str(runtime_meta.get("transport")) if runtime_meta else None),
                )
                self._maybe_prune_records_unlocked(reason="request_finished", force=False)

    def _execute_request(self, *, request: dict[str, Any], slot_idx: int, progress_path: Path | None = None) -> dict[str, Any]:
        req = dict(request or {})
        audio = dict(req.get("audio") or {})
        local_path = str(audio.get("local_path") or "").strip()
        blob_ref = str(audio.get("blob_ref") or "").strip()
        blob_fetch_ms: float | None = None
        if not local_path and blob_ref:
            blob_t0 = time.monotonic()
            try:
                resolved = resolve_blob_ref_to_local_path(blob_ref)
            except AsrBlobError as e:
                return build_error_response(
                    request=req,
                    code=e.code,
                    message=str(e),
                    retryable=e.retryable,
                    details=e.details,
                )
            blob_fetch_ms = max(0.0, (time.monotonic() - blob_t0) * 1000.0)
            audio["local_path"] = str(resolved)
            req["audio"] = audio
        request = req
        request_id = str(request.get("request_id") or f"req_{int(time.time())}")
        job_root = (self._work_root / request_id / f"slot_{slot_idx}").resolve()
        out_dir = (job_root / "whisperx").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        job = SimpleNamespace(
            whisperx_dir=out_dir,
        )
        if slot_idx < 0 or slot_idx >= len(self._warm_clients):
            return build_error_response(
                request=request,
                code="ASR_POOL_RUNNER_INVALID_SLOT",
                message=f"Warm runner slot out of range: {slot_idx}",
                retryable=True,
                details={"slot_idx": int(slot_idx), "slots": int(len(self._warm_clients))},
            )
        client = self._warm_clients[slot_idx]
        try:
            resp = dict(client.transcribe(job=job, request=request, progress_path=progress_path) or {})
            if blob_fetch_ms is not None and isinstance(resp, dict):
                runtime = dict(resp.get("runtime") or {})
                runtime["blob_fetch_ms"] = round(float(blob_fetch_ms), 3)
                resp["runtime"] = runtime
            return resp
        except Exception as e:
            return build_error_response(
                request=request,
                code="ASR_POOL_WARM_EXECUTOR_FAILURE",
                message=f"Warm executor failed: {type(e).__name__}: {e}",
                retryable=True,
                details={"slot_idx": int(slot_idx), "exc_type": type(e).__name__},
            )


POOL = AsrPoolService()
ASR_TOKEN = _cfg_str("TRANSCRIBE_ASR_POOL_TOKEN", "")


def _auth_guard(x_asr_token: str | None = Header(default=None)) -> None:
    if not ASR_TOKEN:
        return
    presented = str(x_asr_token or "").strip()
    if presented != ASR_TOKEN:
        raise RuntimeError("unauthorized")


@app.on_event("startup")
async def _startup() -> None:
    await POOL.start()


@app.on_event("shutdown")
async def _shutdown() -> None:
    await POOL.stop()


@app.post("/asr/v1/requests")
async def submit_asr_request(raw_payload: dict[str, Any], _auth: None = Depends(_auth_guard)) -> JSONResponse:
    status_code, body = await POOL.submit(raw_payload if isinstance(raw_payload, dict) else {})
    return JSONResponse(status_code=int(status_code), content=body)


@app.get("/asr/v1/requests/{request_id}")
async def get_asr_request(request_id: str, _auth: None = Depends(_auth_guard)) -> JSONResponse:
    status_code, body = await POOL.get_request(request_id)
    return JSONResponse(status_code=int(status_code), content=body)


@app.post("/asr/v1/requests/{request_id}/cancel")
async def cancel_asr_request(request_id: str, _auth: None = Depends(_auth_guard)) -> JSONResponse:
    status_code, body = await POOL.cancel(request_id)
    return JSONResponse(status_code=int(status_code), content=body)


@app.get("/asr/v1/pool")
async def get_asr_pool_status(_req: Request, _auth: None = Depends(_auth_guard)) -> JSONResponse:
    body = await POOL.pool_status()
    return JSONResponse(status_code=200, content=body)


@app.exception_handler(RuntimeError)
async def _runtime_error_handler(_request: Request, exc: RuntimeError) -> JSONResponse:
    if "unauthorized" in str(exc).lower():
        return _error(401, code="ASR_UNAUTHORIZED", message="Invalid X-ASR-Token", retryable=False)
    return _error(500, code="ASR_INTERNAL_ERROR", message=str(exc), retryable=True)
