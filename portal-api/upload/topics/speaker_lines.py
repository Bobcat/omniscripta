from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from upload.status_io import _write_status


_SPK_RE = re.compile(r"^\s*\[?(SPEAKER_\d+)\]?\s*:?\s*(.*)\s*$", re.IGNORECASE)
_TS_RE = re.compile(r"^\s*(\d{2}:\d{2}:\d{2}),\d{3}\s*-->\s*(\d{2}:\d{2}:\d{2}),\d{3}\s*$")


def _hms_to_seconds(hms: str) -> int:
  hh, mm, ss = hms.split(":")
  return int(hh) * 3600 + int(mm) * 60 + int(ss)


def _derive_speaker_and_text(joined: str) -> tuple[str, str]:
  m = _SPK_RE.match(joined)
  if not m:
    return ("SPEAKER_UNKNOWN", joined)
  spk = m.group(1).upper()
  txt = m.group(2).strip()
  return (spk, txt)


@dataclass
class _Cue:
  start_hms: str
  end_hms_raw: str
  spk: str
  txt: str


def make_speaker_lines_from_srt(*, job, srt_path: Path, orig_stem: str) -> tuple[Path, str]:
  result_dir = (job.dir / "result").resolve()
  out_path = result_dir / f"{orig_stem}_speaker_lines.txt"
  _write_status(job.status_path, phase="topics", subphase="speaker_lines", message="Generating speaker_lines…")

  lines = srt_path.read_text(encoding="utf-8", errors="replace").splitlines()

  cues: list[_Cue] = []
  last_end_hms: str | None = None
  i = 0
  while i < len(lines):
    while i < len(lines) and not lines[i].strip():
      i += 1
    if i >= len(lines):
      break

    i += 1
    if i >= len(lines):
      break
    ts_line = lines[i].strip()
    i += 1

    m_ts = _TS_RE.match(ts_line)
    if not m_ts:
      while i < len(lines) and lines[i].strip():
        i += 1
      continue

    start_hms = m_ts.group(1)
    end_hms_raw = m_ts.group(2)
    last_end_hms = end_hms_raw

    text_parts: list[str] = []
    while i < len(lines) and lines[i].strip():
      text_parts.append(lines[i].strip())
      i += 1
    if not text_parts:
      continue

    joined = " ".join(text_parts)
    spk, txt = _derive_speaker_and_text(joined)
    cues.append(_Cue(start_hms=start_hms, end_hms_raw=end_hms_raw, spk=spk, txt=txt))

  out_lines: list[str] = []
  for idx, cue in enumerate(cues):
    if idx < len(cues) - 1:
      end_hms = cues[idx + 1].start_hms
    else:
      end_hms = cue.end_hms_raw
    if _hms_to_seconds(end_hms) < _hms_to_seconds(cue.start_hms):
      end_hms = cue.start_hms
    out_lines.append(f"({cue.spk}, {cue.start_hms}-{end_hms}) {cue.txt}".rstrip())

  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")

  transcript_end_hms = last_end_hms or "00:00:00"
  _write_status(
    job.status_path,
    phase="topics",
    subphase="speaker_lines",
    message=f"speaker_lines written: {out_path.name}",
    speaker_lines_filename=out_path.name,
    transcript_end=transcript_end_hms,
  )
  return out_path, transcript_end_hms
