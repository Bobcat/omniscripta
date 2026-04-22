from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from upload._util import _hms_to_seconds, _seconds_to_hms


BOUNDARY_SALVAGE_TOLERANCE_S = 10


@dataclass
class TopicsValidationError(Exception):
  message: str
def _row_to_raw_line(row: dict[str, Any]) -> str:
  return (
    f"{int(row.get('n') or 0)} | "
    f"{str(row.get('topic_title') or '').strip()} | "
    f"{str(row.get('topic_description') or '').strip()} | "
    f"{str(row.get('start_time') or '').strip()} | "
    f"{str(row.get('end_time') or '').strip()}"
  )


def _read_finish_reason(
  *,
  parsed_dir: Path,
  orig_stem: str,
  prompt_id: str,
  chunk_index: int,
) -> str:
  response_path = parsed_dir / f"{orig_stem}_{prompt_id}_chunk_{chunk_index:04d}_response.json"
  if not response_path.exists():
    return ""
  try:
    payload = json.loads(response_path.read_text(encoding="utf-8"))
  except Exception:
    return ""
  choices = payload.get("choices") or []
  if not isinstance(choices, list) or not choices:
    return ""
  first = choices[0] if isinstance(choices[0], dict) else {}
  return str(first.get("finish_reason") or "").strip()


def _only_boundary_mismatch_errors(errs: list[str]) -> bool:
  if not errs:
    return False
  for err in errs:
    if not (
      err.startswith("coverage_start_mismatch:")
      or err.startswith("coverage_end_mismatch:")
    ):
      return False
  return True


def _maybe_salvage_boundary_rows(
  *,
  rows: list[dict[str, Any]],
  chunk_start: str,
  chunk_end: str,
  errs: list[str],
  finish_reason: str,
) -> tuple[list[dict[str, Any]] | None, list[str]]:
  if finish_reason == "length":
    return None, []
  if not rows or not _only_boundary_mismatch_errors(errs):
    return None, []

  try:
    chunk_start_sec = _hms_to_seconds(chunk_start)
    chunk_end_sec = _hms_to_seconds(chunk_end)
    first_start_sec = _hms_to_seconds(str(rows[0].get("start_time") or ""))
    first_end_sec = _hms_to_seconds(str(rows[0].get("end_time") or ""))
    last_start_sec = _hms_to_seconds(str(rows[-1].get("start_time") or ""))
    last_end_sec = _hms_to_seconds(str(rows[-1].get("end_time") or ""))
  except Exception:
    return None, []

  start_delta = first_start_sec - chunk_start_sec
  end_delta = last_end_sec - chunk_end_sec
  if start_delta == 0 and end_delta == 0:
    return None, []
  if start_delta and abs(start_delta) > BOUNDARY_SALVAGE_TOLERANCE_S:
    return None, []
  if end_delta and abs(end_delta) > BOUNDARY_SALVAGE_TOLERANCE_S:
    return None, []
  if start_delta and chunk_start_sec > first_end_sec:
    return None, []
  if end_delta and chunk_end_sec < last_start_sec:
    return None, []

  corrected = [dict(row) for row in rows]
  notes: list[str] = []
  if start_delta:
    corrected[0]["start_time"] = _seconds_to_hms(chunk_start_sec)
    corrected[0]["raw_line"] = _row_to_raw_line(corrected[0])
    notes.append(f"clamped_start {first_start_sec}->{chunk_start_sec}")
  if end_delta:
    corrected[-1]["end_time"] = _seconds_to_hms(chunk_end_sec)
    corrected[-1]["raw_line"] = _row_to_raw_line(corrected[-1])
    notes.append(f"clamped_end {last_end_sec}->{chunk_end_sec}")
  return corrected, notes


def validate_chunk_rows(*, rows: list[dict[str, Any]], chunk_start: str, chunk_end: str) -> list[str]:
  errs: list[str] = []
  if not rows:
    return ["no_rows"]

  expected_n = 1
  for r in rows:
    n = r.get("n")
    if n != expected_n:
      errs.append(f"bad_n_sequence: expected {expected_n}, got {n}")
      break
    expected_n += 1

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

  for i in range(len(times) - 1):
    s_i, e_i = times[i]
    s_next, _e_next = times[i + 1]
    if s_next < s_i:
      errs.append(f"non_chronological: row_{i+2}_start {s_next} < row_{i+1}_start {s_i}")
      break
    if e_i != s_next:
      errs.append(f"non_contiguous: row_{i+1}_end {e_i} != row_{i+2}_start {s_next}")
      break

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
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  chunks = manifest.get("chunks") or []

  report_chunks: list[dict[str, Any]] = []
  ok_all = True
  salvaged_chunks = 0
  for ch in chunks:
    idx = int(ch["index"])
    chunk_start = ch["chunk_start"]
    chunk_end = ch["chunk_end"]
    parsed_path = parsed_dir / f"{orig_stem}_{prompt_id}_chunk_{idx:04d}.json"
    if not parsed_path.exists():
      ok_all = False
      report_chunks.append({"index": idx, "ok": False, "errors": [f"missing_parsed_file: {parsed_path.name}"]})
      continue

    parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
    rows = parsed.get("rows") or []
    finish_reason = _read_finish_reason(
      parsed_dir=parsed_dir,
      orig_stem=orig_stem,
      prompt_id=prompt_id,
      chunk_index=idx,
    )
    errs_before_salvage = validate_chunk_rows(rows=rows, chunk_start=chunk_start, chunk_end=chunk_end)
    salvage_notes: list[str] = []
    salvaged = False
    corrected_rows, candidate_notes = _maybe_salvage_boundary_rows(
      rows=rows,
      chunk_start=chunk_start,
      chunk_end=chunk_end,
      errs=errs_before_salvage,
      finish_reason=finish_reason,
    )
    if corrected_rows is not None:
      candidate_errs = validate_chunk_rows(rows=corrected_rows, chunk_start=chunk_start, chunk_end=chunk_end)
      if not candidate_errs:
        parsed_out = dict(parsed)
        parsed_out["rows"] = corrected_rows
        parsed_meta = parsed_out.get("meta")
        parsed_meta_out = dict(parsed_meta) if isinstance(parsed_meta, dict) else {}
        parsed_meta_out["boundary_salvage"] = {
          "tolerance_seconds": BOUNDARY_SALVAGE_TOLERANCE_S,
          "finish_reason": finish_reason,
          "errors_before_salvage": list(errs_before_salvage),
          "notes": list(candidate_notes),
        }
        parsed_out["meta"] = parsed_meta_out
        parsed_path.write_text(json.dumps(parsed_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        rows = corrected_rows
        salvage_notes = list(candidate_notes)
        salvaged = True
        salvaged_chunks += 1
    errs = validate_chunk_rows(rows=rows, chunk_start=chunk_start, chunk_end=chunk_end)
    ok = len(errs) == 0
    ok_all = ok_all and ok
    chunk_report = {
      "index": idx,
      "ok": ok,
      "errors": errs,
      "chunk_start": chunk_start,
      "chunk_end": chunk_end,
      "parsed_file": parsed_path.name,
      "row_count": len(rows),
      "finish_reason": finish_reason,
    }
    if salvaged:
      chunk_report["salvaged"] = True
      chunk_report["errors_before_salvage"] = list(errs_before_salvage)
      chunk_report["salvage_notes"] = list(salvage_notes)
    report_chunks.append(chunk_report)

  report = {
    "orig_stem": orig_stem,
    "prompt_id": prompt_id,
    "is_valid": ok_all,
    "salvaged_chunks": salvaged_chunks,
    "chunks": report_chunks,
  }
  out_report_path.parent.mkdir(parents=True, exist_ok=True)
  out_report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
  return out_report_path
