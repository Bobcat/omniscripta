from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def merge_topics(
  *,
  manifest_path: Path,
  parsed_dir: Path,
  orig_stem: str,
  prompt_id: str,
  out_merged_path: Path,
) -> Path:
  """
  Merge per-chunk parsed topic rows into one list, renumbering n from 1..N.
  Assumes validation already passed.

  Reads:
    <orig_stem>_<prompt_id>_chunk_0001.json, ...
  Writes:
    <orig_stem>_<prompt_id>_merged.json
  """
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  chunks = manifest.get("chunks") or []

  merged_rows: list[dict[str, Any]] = []
  for ch in chunks:
    idx = int(ch["index"])
    parsed_path = parsed_dir / f"{orig_stem}_{prompt_id}_chunk_{idx:04d}.json"
    parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
    rows = parsed.get("rows") or []
    merged_rows.extend(rows)

  # Renumber n
  for i, r in enumerate(merged_rows, start=1):
    r["n"] = i

  out = {
    "orig_stem": orig_stem,
    "prompt_id": prompt_id,
    "rows": merged_rows,
  }

  out_merged_path.parent.mkdir(parents=True, exist_ok=True)
  out_merged_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
  return out_merged_path
