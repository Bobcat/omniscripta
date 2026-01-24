from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# Strict row format (single line):
# n | topic title | topic short description | start_time | end_time
# Example:
# 1 | Pivoting and Iteration | ... | 00:25:03 | 00:25:43
#
# v1 deliberate choice:
# - Disallow '|' inside fields (title/description). If present, the row won't match -> rejected.
_ROW_RE = re.compile(
    r"""^\s*
    (?P<n>\d+)\s*\|\s*
    (?P<title>[^|]+?)\s*\|\s*
    (?P<descr>[^|]+?)\s*\|\s*
    (?P<start>\d{2}:\d{2}:\d{2})\s*\|\s*
    (?P<end>\d{2}:\d{2}:\d{2})
    \s*$""",
    re.VERBOSE,
)


@dataclass
class TopicsParseError(Exception):
  message: str


def parse_topics_rows_from_text(raw_text: str) -> list[dict[str, Any]]:
  """
  Extract strict topic rows from raw LLM output.
  Ignores all non-matching lines (debug, markdown, politeness, headers, separators).

  Returns list of rows:
    {
      "n": int,
      "topic_title": str,
      "topic_description": str,
      "start_time": "HH:MM:SS",
      "end_time": "HH:MM:SS"
    }

  Raises TopicsParseError if zero rows found.
  """
  rows: list[dict[str, Any]] = []
  for line in (raw_text or "").splitlines():
    m = _ROW_RE.match(line)
    if not m:
      continue

    n = int(m.group("n"))
    title = m.group("title").strip()
    descr = m.group("descr").strip()
    start = m.group("start")
    end = m.group("end")

    if not title or not descr:
      raise TopicsParseError(f"Empty title/description in row: {line}")

    rows.append({
      "n": n,
      "topic_title": title,
      "topic_description": descr,
      "start_time": start,
      "end_time": end,
      "raw_line": line.rstrip("\n"),
    })

  if not rows:
    raise TopicsParseError("No valid topic rows found (expected: n | title | description | HH:MM:SS | HH:MM:SS).")

  return rows


def parse_topics_raw_file(*, raw_txt_path: Path, out_json_path: Path, meta: dict[str, Any] | None = None) -> Path:
  raw = raw_txt_path.read_text(encoding="utf-8", errors="replace")
  rows = parse_topics_rows_from_text(raw)

  payload: dict[str, Any] = {"rows": rows}
  if meta:
    payload["meta"] = meta

  out_json_path.parent.mkdir(parents=True, exist_ok=True)
  out_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
  return out_json_path
