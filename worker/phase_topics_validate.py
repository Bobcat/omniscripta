from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TopicsValidationError(Exception):
  message: str


def _hms_to_seconds(hms: str) -> int:
  hh, mm, ss = hms.split(":")
  return int(hh) * 3600 + int(mm) * 60 + int(ss)


def validate_chunk_rows(*, rows: list[dict[str, Any]], chunk_start: str, chunk_end: str) -> list[str]:
  """
  Returns list of error strings (empty => valid).
  Strict v1 checks:
    - n sequence must be 1..N
    - timestamps parseable, start<=end
    - chronological order (start nondecreasing)
    - contiguity (end_i == start_{i+1})
    - full coverage (first.start == chunk_start; last.end == chunk_end)
  """
  errs: list[str] = []
  if not rows:
    return ["no_rows"]

  # n sequence check (as output format requires n)
  expected_n = 1
  for r in rows:
    n = r.get("n")
    if n != expected_n:
      errs.append(f"bad_n_sequence: expected {expected_n}, got {n}")
      break
    expected_n += 1

  # Parse times and basic ordering
  times = []
  for idx, r in enumerate(rows, start=1):
    s = r.get("start_time")
    e = r.get("end_time")
    try:
      s_sec = _hms_to_seconds(s)
      e_sec = _hms_to_seconds(e)
    except Exception:
      errs.append(f"bad_timestamp_format_row_{idx}: {s}..{e}")
      continue
    if e_sec < s_sec:
      errs.append(f"end_before_start_row_{idx}: {s}..{e}")
    times.append((s_sec, e_sec))

  if not times:
    return errs or ["no_parseable_timestamps"]

  # Chronological + contiguity
  for i in range(len(times) - 1):
    s_i, e_i = times[i]
    s_next, e_next = times[i + 1]
    if s_next < s_i:
      errs.append(f"non_chronological: row_{i+2}_start {s_next} < row_{i+1}_start {s_i}")
      break
    if e_i != s_next:
      errs.append(f"non_contiguous: row_{i+1}_end {e_i} != row_{i+2}_start {s_next}")
      break

  # Coverage
  try:
    chunk_start_sec = _hms_to_seconds(chunk_start)
    chunk_end_sec = _hms_to_seconds(chunk_end)
  except Exception:
    errs.append(f"bad_chunk_bounds: {chunk_start}..{chunk_end}")
    return errs

  first_start = times[0][0]
  last_end = times[-1][1]

  if first_start != chunk_start_sec:
    errs.append(f"coverage_start_mismatch: first_start {first_start} != chunk_start {chunk_start_sec}")
  if last_end != chunk_end_sec:
    errs.append(f"coverage_end_mismatch: last_end {last_end} != chunk_end {chunk_end_sec}")

  return errs


def validate_all_chunks(
  *,
  manifest_path: Path,
  parsed_dir: Path,
  orig_stem: str,
  prompt_id: str,
  out_report_path: Path,
) -> Path:
  """
  Validate parsed per-chunk JSON outputs against the manifest chunk bounds.

  parsed files expected:
    <orig_stem>_<prompt_id>_chunk_0001.json
    ...

  Writes validation report JSON.
  """
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  chunks = manifest.get("chunks") or []

  report_chunks: list[dict[str, Any]] = []
  ok_all = True

  for ch in chunks:
    idx = int(ch["index"])
    chunk_start = ch["chunk_start"]
    chunk_end = ch["chunk_end"]
    parsed_path = parsed_dir / f"{orig_stem}_{prompt_id}_chunk_{idx:04d}.json"

    if not parsed_path.exists():
      ok_all = False
      report_chunks.append({
        "index": idx,
        "ok": False,
        "errors": [f"missing_parsed_file: {parsed_path.name}"],
      })
      continue

    parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
    rows = parsed.get("rows") or []

    errs = validate_chunk_rows(rows=rows, chunk_start=chunk_start, chunk_end=chunk_end)
    ok = (len(errs) == 0)
    ok_all = ok_all and ok

    report_chunks.append({
      "index": idx,
      "ok": ok,
      "errors": errs,
      "chunk_start": chunk_start,
      "chunk_end": chunk_end,
      "parsed_file": parsed_path.name,
      "row_count": len(rows),
    })

  report = {
    "orig_stem": orig_stem,
    "prompt_id": prompt_id,
    "is_valid": ok_all,
    "chunks": report_chunks,
  }

  out_report_path.parent.mkdir(parents=True, exist_ok=True)
  out_report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
  return out_report_path
