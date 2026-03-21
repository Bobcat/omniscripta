#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.app_config import get_bool, get_int, get_str


def _repo_root() -> Path:
    return _REPO_ROOT


@dataclass(frozen=True)
class JobEntry:
    state: str
    path: Path
    mtime: float


def _collect_entries(*, state_dir: Path, state: str) -> list[JobEntry]:
    out: list[JobEntry] = []
    if not state_dir.exists() or not state_dir.is_dir():
        return out
    for child in state_dir.iterdir():
        if not child.is_dir() or child.name.startswith("."):
            continue
        try:
            mtime = float(child.stat().st_mtime)
        except Exception:
            mtime = 0.0
        out.append(JobEntry(state=str(state), path=child, mtime=mtime))
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
    cfg_jobs_base = get_str("jobs.live_worker_base", "data/jobs/live_worker").strip()
    default_jobs_base = (
        Path(cfg_jobs_base) if Path(cfg_jobs_base).is_absolute() else (_repo_root() / cfg_jobs_base)
    ).resolve()
    default_enabled = bool(get_bool("janitor.enabled", False))
    default_dry_run = bool(get_bool("janitor.dry_run", True))
    default_allow_nonstandard_base = bool(get_bool("janitor.allow_nonstandard_base", False))
    default_min_age_s = int(get_int("janitor.min_age_s", 3600, min_value=0))
    default_max_per_state = int(get_int("janitor.max_per_state", 3000, min_value=0))
    default_verbose_items = bool(get_bool("janitor.verbose_items", False))
    default_verbose_items_max = int(get_int("janitor.verbose_items_max", 20, min_value=0))

    parser = argparse.ArgumentParser(description="Cleanup live worker done/error directories.")
    parser.add_argument("--jobs-base", default=str(default_jobs_base))
    parser.add_argument("--enabled", action="store_true", default=bool(default_enabled))
    parser.add_argument("--dry-run", action="store_true", default=bool(default_dry_run))
    parser.add_argument("--allow-nonstandard-base", action="store_true", default=bool(default_allow_nonstandard_base))
    parser.add_argument("--min-age-s", type=int, default=int(default_min_age_s))
    parser.add_argument("--max-per-state", type=int, default=int(default_max_per_state))
    parser.add_argument("--verbose-items", action="store_true", default=bool(default_verbose_items))
    parser.add_argument("--verbose-items-max", type=int, default=int(default_verbose_items_max))
    args = parser.parse_args()

    jobs_base = Path(str(args.jobs_base)).expanduser().resolve()
    enabled = bool(args.enabled)
    dry_run = bool(args.dry_run)
    allow_nonstandard_base = bool(args.allow_nonstandard_base)
    min_age_s = int(max(0, int(args.min_age_s)))
    max_per_state = int(max(0, int(args.max_per_state)))
    verbose_items = bool(args.verbose_items)
    verbose_items_max = int(max(0, int(args.verbose_items_max)))

    if (jobs_base.name != default_jobs_base.name) and (not allow_nonstandard_base):
        _log(
            "fatal_invalid_base",
            jobs_base=str(jobs_base),
            reason=f"basename_not_{default_jobs_base.name}",
        )
        return 2

    if not enabled:
        _log("janitor_disabled", jobs_base=str(jobs_base))
        return 0

    now = time.time()
    total_deleted = 0
    total_candidates = 0

    for state in ("done", "error"):
        state_dir = (jobs_base / state).resolve()
        if not _must_be_child_of(jobs_base, state_dir):
            _log("state_dir_guard_failed", state=state, state_dir=str(state_dir))
            return 2
        entries = _collect_entries(state_dir=state_dir, state=state)

        ttl_candidates = [e for e in entries if (now - float(e.mtime)) >= float(min_age_s)]
        overflow_candidates: list[JobEntry] = []
        if max_per_state > 0 and len(entries) > max_per_state:
            sorted_by_mtime_desc = sorted(entries, key=lambda e: float(e.mtime), reverse=True)
            keep = set(e.path for e in sorted_by_mtime_desc[:max_per_state])
            overflow_candidates = [e for e in entries if e.path not in keep]

        candidates_by_path: dict[Path, JobEntry] = {}
        for entry in ttl_candidates:
            candidates_by_path[entry.path] = entry
        for entry in overflow_candidates:
            candidates_by_path[entry.path] = entry
        candidates = list(candidates_by_path.values())
        total_candidates += len(candidates)

        _log(
            "state_scan",
            state=state,
            state_dir=str(state_dir),
            total_dirs=len(entries),
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
                        age_s=int(max(0, now - float(entry.mtime))),
                    )
                    items_logged += 1
            except Exception as e:
                _log(
                    "delete_failed",
                    state=state,
                    path=str(entry.path),
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
        min_age_s=int(min_age_s),
        max_per_state=int(max_per_state),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
