from __future__ import annotations

import json
import os
import secrets
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _repo_root() -> Path:
    # portal-api/queue_fs.py -> portal-api -> repo root
    return Path(__file__).resolve().parents[1]


def _jobs_base() -> Path:
    # Allow override for e.g. local dev: TRANSCRIBE_JOBS_BASE=/tmp/demo_jobs
    raw = (os.getenv("TRANSCRIBE_JOBS_BASE") or "").strip()
    if raw:
        p = Path(raw)
        return p if p.is_absolute() else (_repo_root() / p)

    return _repo_root() / "data" / "demo_jobs"


BASE = _jobs_base()
INBOX = BASE / "inbox"
RUNNING = BASE / "running"
DONE = BASE / "done"
ERROR = BASE / "error"


@dataclass(frozen=True)
class JobPaths:
    job_id: str
    dir: Path
    upload_dir: Path
    snippet_dir: Path
    whisperx_dir: Path
    result_dir: Path
    status_path: Path
    job_path: Path
    log_path: Path


def _utc_stamp() -> str:
    # Lexicografisch sorteerbaar (FIFO op naam)
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def new_job_id() -> str:
    # Voorbeeld: job_20260104T100156Z_a1b2c3d4
    return f"job_{_utc_stamp()}_{secrets.token_hex(4)}"


def init_job_in_inbox(
    *,
    orig_filename: str,
    options: Dict[str, Any],
    job_kind: str = "upload_audio",
    upload_src_path: str | Path | None = None,
) -> JobPaths:
    """
    Maakt een jobfolder in INBOX via:
      1) schrijven naar een hidden tmp dir
      2) atomic rename naar definitieve dirnaam
    """
    INBOX.mkdir(parents=True, exist_ok=True)

    job_id = new_job_id()
    final_dir = INBOX / job_id
    tmp_dir = INBOX / f".tmp_{job_id}"

    # Zorg dat een oude tmp niet blijft hangen
    if tmp_dir.exists():
        raise RuntimeError(f"Temp dir already exists: {tmp_dir}")
    if final_dir.exists():
        raise RuntimeError(f"Job dir already exists: {final_dir}")

    # Maak structuur in tmp
    upload_dir = tmp_dir / "upload"
    snippet_dir = tmp_dir / "snippet"
    whisperx_dir = tmp_dir / "whisperx"
    result_dir = tmp_dir / "result"
    for d in (upload_dir, snippet_dir, whisperx_dir, result_dir):
        d.mkdir(parents=True, exist_ok=True)

    status_path = tmp_dir / "status.json"
    job_path = tmp_dir / "job.json"
    log_path = tmp_dir / "worker.log"

    # Initial status
    status = {
        "job_id": job_id,
        "job_kind": str(job_kind or "upload_audio"),
        "state": "queued",
        "phase": "upload",
        "progress": 0.0,
        "message": "Queued",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "started_at": None,
        "finished_at": None,
        "error": None,
        # handig: wijs straks naar de daadwerkelijke output naam
        "orig_filename": orig_filename,
        "language": options.get("language"),
        "speaker_mode": options.get("speaker_mode"),
        "expected_speakers": options.get("expected_speakers"),
        "min_speakers": options.get("min_speakers"),
        "max_speakers": options.get("max_speakers"),
        "snippet_filename": None,
        "srt_filename": None,
    }
    _write_json_atomic(status_path, status)

    # Job config (wat worker nodig heeft)
    job = {
        "job_id": job_id,
        "job_kind": str(job_kind or "upload_audio"),
        "orig_filename": orig_filename,
        "options": options,
    }
    _write_json_atomic(job_path, job)

    # Maak lege log
    log_path.write_text("", encoding="utf-8")

    # Optional: stage upload into tmp job dir before publish.
    # This prevents worker race conditions where a claimed job has no upload file yet.
    if upload_src_path is not None:
        src = Path(str(upload_src_path)).resolve()
        if not src.exists():
            raise FileNotFoundError(f"Upload source missing: {src}")
        dst = (upload_dir / str(orig_filename)).resolve()
        shutil.copy2(src, dst)

    # Atomic publish: tmp -> final
    os.replace(tmp_dir, final_dir)

    return JobPaths(
        job_id=job_id,
        dir=final_dir,
        upload_dir=final_dir / "upload",
        snippet_dir=final_dir / "snippet",
        whisperx_dir=final_dir / "whisperx",
        result_dir=final_dir / "result",
        status_path=final_dir / "status.json",
        job_path=final_dir / "job.json",
        log_path=final_dir / "worker.log",
    )


def _write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


RUNNING = BASE / "running"

def claim_next_job() -> JobPaths | None:
    """
    Pakt de oudste job uit INBOX en verplaatst atomically naar RUNNING.
    Returnt JobPaths of None als de queue leeg is.
    """
    RUNNING.mkdir(parents=True, exist_ok=True)

    # Alleen echte job dirs, geen tmp
    candidates = sorted(
        p for p in INBOX.iterdir()
        if p.is_dir() and not p.name.startswith(".tmp_")
    )
    if not candidates:
        return None

    for job_dir in candidates:
        target = RUNNING / job_dir.name
        try:
            os.replace(job_dir, target)  # atomic claim
        except FileNotFoundError:
            continue  # was net weggehaald
        except OSError:
            continue  # race/perm issue; probeer volgende

        return JobPaths(
            job_id=target.name,
            dir=target,
            upload_dir=target / "upload",
            snippet_dir=target / "snippet",
            whisperx_dir=target / "whisperx",
            result_dir=target / "result",
            status_path=target / "status.json",
            job_path=target / "job.json",
            log_path=target / "worker.log",
        )

    return None


DONE = BASE / "done"
ERROR = BASE / "error"

def finish_job(job: JobPaths, *, ok: bool) -> Path:
    """
    Verplaatst job uit RUNNING naar DONE of ERROR.
    Returnt de nieuwe jobdir Path.
    """
    DONE.mkdir(parents=True, exist_ok=True)
    ERROR.mkdir(parents=True, exist_ok=True)

    src = job.dir
    dst_base = DONE if ok else ERROR
    dst = dst_base / src.name
    os.replace(src, dst)
    return dst
