#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return bool(default)
    raw = str(value).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _parse_int(value: str | None, default: int, *, min_value: int | None = None) -> int:
    try:
        out = int(str(value).strip()) if value is not None else int(default)
    except Exception:
        out = int(default)
    if min_value is not None and out < min_value:
        out = int(min_value)
    return int(out)


def _safe_read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _kind_for_job_dir(job_dir: Path) -> str:
    job_json = _safe_read_json(job_dir / "job.json")
    if isinstance(job_json, dict):
        raw = str(job_json.get("job_kind") or "").strip().lower()
        if raw:
            return raw
    status_json = _safe_read_json(job_dir / "status.json")
    if isinstance(status_json, dict):
        raw = str(status_json.get("job_kind") or "").strip().lower()
        if raw:
            return raw
    return ""


@dataclass(frozen=True)
class JobEntry:
    state: str
    path: Path
    mtime: float
    job_kind: str


def _collect_entries(*, state_dir: Path, state: str) -> list[JobEntry]:
    out: list[JobEntry] = []
    if not state_dir.exists() or not state_dir.is_dir():
        return out
    for child in state_dir.iterdir():
        if not child.is_dir():
            continue
        if child.name.startswith("."):
            continue
        try:
            st = child.stat()
            mtime = float(st.st_mtime)
        except Exception:
            mtime = 0.0
        out.append(
            JobEntry(
                state=str(state),
                path=child,
                mtime=mtime,
                job_kind=_kind_for_job_dir(child),
            )
        )
    return out


def _must_be_child_of(parent: Path, child: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def _log(event: str, **fields: Any) -> None:
    row = {"event": str(event), "ts_unix": time.time()}
    row.update(fields)
    print(json.dumps(row, ensure_ascii=True), flush=True)


def main() -> int:
    env_jobs_base = (os.getenv("TRANSCRIBE_JOBS_BASE") or "").strip()
    default_jobs_base = (Path(env_jobs_base) if env_jobs_base else (_repo_root() / "data" / "demo_jobs")).resolve()
    default_enabled = _parse_bool(os.getenv("TRANSCRIBE_JOBS_JANITOR_ENABLED"), False)
    default_dry_run = _parse_bool(os.getenv("TRANSCRIBE_JOBS_JANITOR_DRY_RUN"), True)
    default_allow_nonstandard_base = _parse_bool(
        os.getenv("TRANSCRIBE_JOBS_JANITOR_ALLOW_NONSTANDARD_BASE"),
        False,
    )
    default_min_age_s = _parse_int(os.getenv("TRANSCRIBE_JOBS_JANITOR_MIN_AGE_S"), 3600, min_value=0)
    default_max_per_state = _parse_int(os.getenv("TRANSCRIBE_JOBS_JANITOR_MAX_PER_STATE"), 3000, min_value=0)
    default_live_kinds = (os.getenv("TRANSCRIBE_JOBS_JANITOR_LIVE_KINDS") or "live_chunk").strip()
    default_verbose_items = _parse_bool(os.getenv("TRANSCRIBE_JOBS_JANITOR_VERBOSE_ITEMS"), False)
    default_verbose_items_max = _parse_int(os.getenv("TRANSCRIBE_JOBS_JANITOR_VERBOSE_ITEMS_MAX"), 20, min_value=0)

    parser = argparse.ArgumentParser(description="Cleanup demo_jobs done/error directories for live chunk jobs only.")
    parser.add_argument("--jobs-base", default=str(default_jobs_base))
    parser.add_argument("--enabled", action="store_true", default=bool(default_enabled))
    parser.add_argument("--dry-run", action="store_true", default=bool(default_dry_run))
    parser.add_argument("--allow-nonstandard-base", action="store_true", default=bool(default_allow_nonstandard_base))
    parser.add_argument("--min-age-s", type=int, default=int(default_min_age_s))
    parser.add_argument("--max-per-state", type=int, default=int(default_max_per_state))
    parser.add_argument("--live-kinds", default=str(default_live_kinds))
    parser.add_argument("--verbose-items", action="store_true", default=bool(default_verbose_items))
    parser.add_argument("--verbose-items-max", type=int, default=int(default_verbose_items_max))
    args = parser.parse_args()

    jobs_base = Path(str(args.jobs_base)).expanduser().resolve()
    enabled = bool(args.enabled)
    dry_run = bool(args.dry_run)
    allow_nonstandard_base = bool(args.allow_nonstandard_base)
    min_age_s = int(max(0, int(args.min_age_s)))
    max_per_state = int(max(0, int(args.max_per_state)))
    live_kinds = {x.strip().lower() for x in str(args.live_kinds).split(",") if x.strip()}
    verbose_items = bool(args.verbose_items)
    verbose_items_max = int(max(0, int(args.verbose_items_max)))

    if "upload_audio" in live_kinds:
        _log("fatal_invalid_config", reason="upload_audio_not_allowed_in_live_kinds")
        return 2

    # Safety guard: by default only allow base path ending with demo_jobs.
    if (jobs_base.name != "demo_jobs") and (not allow_nonstandard_base):
        _log(
            "fatal_invalid_base",
            jobs_base=str(jobs_base),
            reason="basename_not_demo_jobs",
        )
        return 2

    if not enabled:
        _log("janitor_disabled", jobs_base=str(jobs_base))
        return 0

    now = time.time()
    total_deleted = 0
    total_candidates = 0
    states = ("done", "error")

    for state in states:
        state_dir = (jobs_base / state).resolve()
        if not _must_be_child_of(jobs_base, state_dir):
            _log("state_dir_guard_failed", state=state, state_dir=str(state_dir))
            return 2
        entries = _collect_entries(state_dir=state_dir, state=state)

        live_entries = [e for e in entries if e.job_kind in live_kinds]
        # Phase 1: TTL-based removal candidates.
        ttl_candidates = [
            e for e in live_entries if (now - float(e.mtime)) >= float(min_age_s)
        ]

        # Phase 2: Max-count retention for live entries (keep newest max_per_state).
        overflow_candidates: list[JobEntry] = []
        if max_per_state > 0 and len(live_entries) > max_per_state:
            sorted_by_mtime_desc = sorted(live_entries, key=lambda e: float(e.mtime), reverse=True)
            keep = set(e.path for e in sorted_by_mtime_desc[:max_per_state])
            overflow_candidates = [e for e in live_entries if e.path not in keep]

        candidates_by_path: dict[Path, JobEntry] = {}
        for e in ttl_candidates:
            candidates_by_path[e.path] = e
        for e in overflow_candidates:
            candidates_by_path[e.path] = e
        candidates = list(candidates_by_path.values())
        total_candidates += len(candidates)

        _log(
            "state_scan",
            state=state,
            state_dir=str(state_dir),
            total_dirs=len(entries),
            live_dirs=len(live_entries),
            ttl_candidates=len(ttl_candidates),
            overflow_candidates=len(overflow_candidates),
            selected_candidates=len(candidates),
            dry_run=bool(dry_run),
            min_age_s=int(min_age_s),
            max_per_state=int(max_per_state),
        )

        items_logged = 0
        deleted_in_state = 0
        for entry in sorted(candidates, key=lambda e: float(e.mtime)):
            if not _must_be_child_of(state_dir, entry.path):
                _log(
                    "skip_guard_failed",
                    state=state,
                    path=str(entry.path),
                    reason="candidate_not_under_state_dir",
                )
                continue
            if dry_run:
                if verbose_items and items_logged < verbose_items_max:
                    _log(
                        "dry_run_delete",
                        state=state,
                        path=str(entry.path),
                        job_kind=str(entry.job_kind),
                        age_s=int(max(0, now - float(entry.mtime))),
                    )
                    items_logged += 1
                continue
            try:
                shutil.rmtree(entry.path)
                total_deleted += 1
                deleted_in_state += 1
                if verbose_items and items_logged < verbose_items_max:
                    _log(
                        "deleted",
                        state=state,
                        path=str(entry.path),
                        job_kind=str(entry.job_kind),
                        age_s=int(max(0, now - float(entry.mtime))),
                    )
                    items_logged += 1
            except Exception as e:
                _log(
                    "delete_failed",
                    state=state,
                    path=str(entry.path),
                    job_kind=str(entry.job_kind),
                    error=f"{type(e).__name__}: {e}",
                )

        _log(
            "state_apply",
            state=state,
            selected_candidates=len(candidates),
            deleted_in_state=int(deleted_in_state),
            verbose_items=bool(verbose_items),
            verbose_items_logged=int(items_logged),
            verbose_items_max=int(verbose_items_max),
        )

    _log(
        "janitor_done",
        jobs_base=str(jobs_base),
        candidates_total=int(total_candidates),
        deleted_total=int(total_deleted),
        dry_run=bool(dry_run),
        live_kinds=sorted(live_kinds),
        min_age_s=int(min_age_s),
        max_per_state=int(max_per_state),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
