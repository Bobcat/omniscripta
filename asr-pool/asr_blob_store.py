from __future__ import annotations

import os
import re
import secrets
import shutil
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_BLOB_REF_PREFIX = "fs://"
_SAFE_TOKEN_RE = re.compile(r"[^a-zA-Z0-9._-]+")
_cleanup_lock = threading.Lock()
_last_cleanup_monotonic = 0.0


@dataclass
class AsrBlobError(RuntimeError):
  code: str
  message: str
  retryable: bool = False
  details: dict[str, Any] | None = None

  def __post_init__(self) -> None:
    super().__init__(str(self.message))
    self.code = str(self.code or "ASR_BLOB_ERROR")
    self.retryable = bool(self.retryable)
    self.details = dict(self.details or {})


def _repo_root() -> Path:
  # worker/asr_blob_store.py -> worker -> repo root
  return Path(__file__).resolve().parents[1]


def _env_str(name: str, default: str) -> str:
  return str(os.getenv(name, default) or default).strip()


def _env_int(name: str, default: int, *, min_value: int = 0) -> int:
  try:
    return max(min_value, int(str(os.getenv(name, str(default))).strip() or str(default)))
  except Exception:
    return max(min_value, int(default))


def _blob_root() -> Path:
  raw = _env_str("TRANSCRIBE_ASR_BLOB_ROOT", str((_repo_root() / "data" / "asr_blobs").resolve()))
  p = Path(raw).expanduser()
  if not p.is_absolute():
    p = (_repo_root() / p).resolve()
  p.mkdir(parents=True, exist_ok=True)
  return p.resolve()


def _safe_token(value: str, *, fallback: str = "blob") -> str:
  s = _SAFE_TOKEN_RE.sub("_", str(value or "").strip())
  s = s.strip("._-")
  return s or fallback


def _validate_rel_blob_path(rel: str) -> Path:
  text = str(rel or "").strip().replace("\\", "/")
  while text.startswith("/"):
    text = text[1:]
  if not text:
    raise AsrBlobError(
      code="ASR_BLOB_REF_INVALID",
      message="Empty blob_ref payload",
      retryable=False,
      details={"blob_ref_rel": rel},
    )
  p = Path(text)
  parts = p.parts
  if not parts:
    raise AsrBlobError(
      code="ASR_BLOB_REF_INVALID",
      message="Empty blob_ref path",
      retryable=False,
      details={"blob_ref_rel": rel},
    )
  for part in parts:
    if part in {"", ".", ".."}:
      raise AsrBlobError(
        code="ASR_BLOB_REF_INVALID",
        message="blob_ref path contains invalid traversal tokens",
        retryable=False,
        details={"blob_ref_rel": rel},
      )
  return p


def upload_local_path_as_blob_ref(*, local_path: str | Path, request_id: str) -> tuple[str, dict[str, Any]]:
  src = Path(str(local_path)).expanduser().resolve()
  if not src.exists() or not src.is_file():
    raise AsrBlobError(
      code="ASR_BLOB_UPLOAD_SOURCE_NOT_FOUND",
      message=f"Blob upload source missing: {src}",
      retryable=False,
      details={"local_path": str(src)},
    )
  root = _blob_root()
  day = datetime.now(timezone.utc).strftime("%Y%m%d")
  safe_req = _safe_token(str(request_id or ""), fallback="req")
  suffix = "".join(ch for ch in str(src.suffix or "") if ch.isalnum() or ch in {".", "_", "-"}).lower()[:16]
  blob_name = f"{safe_req}_{secrets.token_hex(8)}{suffix}"
  rel = Path(day) / blob_name
  rel = _validate_rel_blob_path(rel.as_posix())
  dst = (root / rel).resolve()
  if not dst.is_relative_to(root):
    raise AsrBlobError(
      code="ASR_BLOB_PATH_ESCAPE",
      message="Resolved blob target escapes blob root",
      retryable=False,
      details={"blob_root": str(root), "dst": str(dst)},
    )
  dst.parent.mkdir(parents=True, exist_ok=True)
  # Use copyfile (not copy2): we want fresh mtime on blob files for TTL cleanup.
  shutil.copyfile(src, dst)
  try:
    os.utime(dst, None)
  except Exception:
    pass
  blob_ref = f"{_BLOB_REF_PREFIX}{rel.as_posix()}"
  meta = {
    "blob_ref": blob_ref,
    "blob_rel": rel.as_posix(),
    "blob_path": str(dst),
    "bytes": int(dst.stat().st_size),
  }
  return blob_ref, meta


def resolve_blob_ref_to_local_path(blob_ref: str) -> Path:
  raw = str(blob_ref or "").strip()
  if not raw:
    raise AsrBlobError(
      code="ASR_BLOB_REF_REQUIRED",
      message="Missing audio.blob_ref",
      retryable=False,
    )
  if not raw.startswith(_BLOB_REF_PREFIX):
    raise AsrBlobError(
      code="ASR_BLOB_REF_UNSUPPORTED",
      message=f"Unsupported blob_ref scheme: {raw.split(':', 1)[0] or '<none>'}",
      retryable=False,
      details={"blob_ref": raw},
    )
  rel_raw = raw[len(_BLOB_REF_PREFIX):]
  rel = _validate_rel_blob_path(rel_raw)
  root = _blob_root()
  p = (root / rel).resolve()
  if not p.is_relative_to(root):
    raise AsrBlobError(
      code="ASR_BLOB_PATH_ESCAPE",
      message="Resolved blob_ref escapes blob root",
      retryable=False,
      details={"blob_ref": raw, "blob_root": str(root), "resolved": str(p)},
    )
  if not p.exists() or not p.is_file():
    raise AsrBlobError(
      code="ASR_BLOB_NOT_FOUND",
      message=f"blob_ref not found: {raw}",
      retryable=False,
      details={"blob_ref": raw, "resolved": str(p)},
    )
  return p


def cleanup_blob_store_if_due() -> None:
  interval_s = _env_int("TRANSCRIBE_ASR_BLOB_CLEANUP_INTERVAL_S", 120, min_value=0)
  ttl_s = _env_int("TRANSCRIBE_ASR_BLOB_TTL_S", 3600, min_value=0)
  max_scan = _env_int("TRANSCRIBE_ASR_BLOB_CLEANUP_MAX_SCAN_FILES", 5000, min_value=1)
  if interval_s <= 0 or ttl_s <= 0:
    return
  now_mono = time.monotonic()
  global _last_cleanup_monotonic
  with _cleanup_lock:
    if (now_mono - float(_last_cleanup_monotonic)) < float(interval_s):
      return
    _last_cleanup_monotonic = now_mono

  root = _blob_root()
  cutoff_unix = time.time() - float(ttl_s)
  scanned = 0
  removed = 0
  try:
    for p in root.rglob("*"):
      if scanned >= max_scan:
        break
      if not p.is_file():
        continue
      scanned += 1
      try:
        st = p.stat()
      except Exception:
        continue
      if float(st.st_mtime) >= cutoff_unix:
        continue
      try:
        p.unlink(missing_ok=True)
        removed += 1
      except Exception:
        continue
    if removed > 0:
      # Best effort prune for empty nested day folders.
      for d in sorted((x for x in root.glob("*") if x.is_dir()), reverse=True):
        try:
          d.rmdir()
        except Exception:
          continue
  except Exception:
    # Cleanup must never break ASR path.
    return
