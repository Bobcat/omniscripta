from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from worker_status_io import _write_status


_LINE_RE = re.compile(r'^\(\s*[^,]+,\s*(\d{2}:\d{2}:\d{2})(?:\s*-\s*\d{2}:\d{2}:\d{2})?\s*\)\s*(.*)$')


def _hms_to_seconds(hms: str) -> int:
  hh, mm, ss = hms.split(":")
  return int(hh) * 3600 + int(mm) * 60 + int(ss)


def _seconds_to_hms(sec: int) -> str:
  sec = max(0, int(sec))
  hh = sec // 3600
  mm = (sec % 3600) // 60
  ss = sec % 60
  return f"{hh:02d}:{mm:02d}:{ss:02d}"


def _estimate_tokens(chars: int, *, method: str) -> int:
  # crude but stable: chars/4 is a common approximation
  if method == "chars_div4":
    return max(1, chars // 4)
  return max(1, chars // 4)


def chunk_speaker_lines(
  *,
  job,
  speaker_lines_path: Path,
  orig_stem: str,
  service_cfg: dict[str, Any],
  transcript_end_hms: str,
) -> Path:
  """
  Phase 32: chunk speaker_lines into ~N-minute chunks, plus manifest.

  Outputs (in job.result_dir):
    <orig_stem>_speaker_lines_chunk_0001.txt
    <orig_stem>_speaker_lines_chunk_0002.txt
    ...
    <orig_stem>_speaker_lines_manifest.json

  Chunking policy:
    - target duration: topics.chunk_minutes (default 25)
    - plus context safety check:
        token_est(line_chars_sum) + prompt_overhead_tokens_est <= ctx_len * ctx_safety

  Chunk boundaries (important for later full-coverage validation):
    - chunk_start = start timestamp of first line in chunk
    - chunk_end   = start timestamp of first line in *next* chunk
    - last chunk chunk_end = transcript_end (from SRT cue end timestamps)

  NOTE (future): overlapping chunk boundaries could be added later.
  """
  _write_status(job.status_path, phase="postprocess", subphase="chunk_speaker_lines", message="Chunking speaker_lines…")

  topics_cfg = service_cfg.get("topics", {}) if isinstance(service_cfg, dict) else {}
  chunk_minutes = int(topics_cfg.get("chunk_minutes", 25))
  ctx_len = int(topics_cfg.get("ctx_len", 16384))
  ctx_safety = float(topics_cfg.get("ctx_safety", 0.85))
  overhead = int(topics_cfg.get("prompt_overhead_tokens_est", 1200))
  token_method = str(topics_cfg.get("token_estimator", "chars_div4"))

  limit_tokens = int(ctx_len * ctx_safety)
  target_span = max(1, chunk_minutes * 60)

  # Parse speaker_lines
  raw_lines = speaker_lines_path.read_text(encoding="utf-8", errors="replace").splitlines()
  parsed: list[tuple[int, str]] = []  # (start_seconds, raw_line)
  for ln in raw_lines:
    m = _LINE_RE.match(ln)
    if not m:
      continue
    parsed.append((_hms_to_seconds(m.group(1)), ln))

  if not parsed:
    raise RuntimeError(f"No parseable speaker_lines in {speaker_lines_path}")

  chunks: list[dict[str, Any]] = []
  chunk_lines: list[str] = []
  chunk_start_sec: Optional[int] = None
  chars_sum = 0

  def flush_chunk(chunk_index: int, *, end_sec: int) -> None:
    nonlocal chunk_lines, chunk_start_sec, chars_sum
    if not chunk_lines or chunk_start_sec is None:
      return

    fname = f"{orig_stem}_speaker_lines_chunk_{chunk_index:04d}.txt"
    out_path = job.result_dir / fname
    out_path.write_text("\n".join(chunk_lines) + "\n", encoding="utf-8")

    token_est = _estimate_tokens(chars_sum, method=token_method) + overhead

    chunks.append({
      "index": chunk_index,
      "filename": fname,
      "line_count": len(chunk_lines),
      "chunk_start": _seconds_to_hms(chunk_start_sec),
      "chunk_end": _seconds_to_hms(end_sec),
      "token_est": int(token_est),
      "chars_sum": int(chars_sum),
    })

    chunk_lines = []
    chunk_start_sec = None
    chars_sum = 0

  chunk_index = 1

  for sec, raw in parsed:
    if chunk_start_sec is None:
      chunk_start_sec = sec
      chunk_lines = [raw]
      chars_sum = len(raw) + 1
      continue

    span_if_added = sec - chunk_start_sec
    chars_if_added = chars_sum + len(raw) + 1
    token_if_added = _estimate_tokens(chars_if_added, method=token_method) + overhead

    # Cut if we'd exceed token safety, OR we've reached target span.
    # When cutting, current line becomes the FIRST line of the next chunk,
    # so the previous chunk_end is this line's start time.
    if token_if_added > limit_tokens or span_if_added >= target_span:
      flush_chunk(chunk_index, end_sec=sec)
      chunk_index += 1
      chunk_start_sec = sec
      chunk_lines = [raw]
      chars_sum = len(raw) + 1
      continue

    chunk_lines.append(raw)
    chars_sum = chars_if_added

  # Flush last chunk using transcript end time (from SRT)
  transcript_end_sec = _hms_to_seconds(transcript_end_hms)
  flush_chunk(chunk_index, end_sec=transcript_end_sec)

  manifest_path = job.result_dir / f"{orig_stem}_speaker_lines_manifest.json"
  manifest = {
    "orig_stem": orig_stem,
    "speaker_lines_filename": speaker_lines_path.name,
    "transcript_end": transcript_end_hms,
    "chunking": {
      "chunk_minutes": chunk_minutes,
      "ctx_len": ctx_len,
      "ctx_safety": ctx_safety,
      "prompt_overhead_tokens_est": overhead,
      "token_estimator": token_method,
      "limit_tokens": limit_tokens,
      "target_span_seconds": target_span,
    },
    "chunks": chunks,
  }
  manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

  _write_status(
    job.status_path,
    phase="postprocess",
    subphase="chunk_speaker_lines",
    message=f"Chunking done: {len(chunks)} chunks",
    speaker_lines_manifest_filename=manifest_path.name,
    speaker_lines_chunk_count=len(chunks),
  )
  return manifest_path
