from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from worker_status_io import _write_status


_SPK_RE = re.compile(r"^\s*\[?(SPEAKER_\d+)\]?\s*:?\s*(.*)\s*$", re.IGNORECASE)
_TS_RE = re.compile(r"^\s*(\d{2}:\d{2}:\d{2}),\d{3}\s*-->\s*(\d{2}:\d{2}:\d{2}),\d{3}\s*$")


def _hms_to_seconds(hms: str) -> int:
  hh, mm, ss = hms.split(":")
  return int(hh) * 3600 + int(mm) * 60 + int(ss)


def _seconds_to_hms(sec: int) -> str:
  sec = max(0, int(sec))
  hh = sec // 3600
  mm = (sec % 3600) // 60
  ss = sec % 60
  return f"{hh:02d}:{mm:02d}:{ss:02d}"


def _derive_speaker_and_text(joined: str) -> tuple[str, str]:
  """
  WhisperX SRT text lines often look like:
    [SPEAKER_01]: Hello ...
  We keep normalization minimal:
    - parse speaker if present
    - otherwise speaker="UNKNOWN" and keep full text
  """
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
  """
  Phase 31: SRT -> speaker_lines (flat text), 1 SRT cue => 1 output line.

  Output file (in job.result_dir):
    <orig_stem>_speaker_lines.txt

  Format:
    (SPEAKER_XX, HH:MM:SS-HH:MM:SS) <joined cue text>

  Important detail (v4):
    - SRT has millisecond precision, but our speaker_lines timestamps are HH:MM:SS only.
      If we take the cue's end time by truncating ms, you can get artificial 1-second gaps like:
        end=00:08:22 (from 00:08:22,999) and next start=00:08:23 (from 00:08:23,000)
      This later makes LLM topic boundaries fail strict "contiguous" validation.

    - Therefore we set each cue's end time (except the last) to the *next cue's start time*
      at HH:MM:SS precision. The last cue keeps its own SRT end time (truncated to HH:MM:SS).
      This yields a second-accurate representation that is contiguous by construction.

  Rules (as agreed):
    - No extra normalization
    - Multi-line cue text is joined with a single space
  """
  out_path = job.result_dir / f"{orig_stem}_speaker_lines.txt"
  _write_status(job.status_path, phase="postprocess", subphase="speaker_lines", message="Generating speaker_lines…")

  lines = srt_path.read_text(encoding="utf-8", errors="replace").splitlines()

  cues: list[_Cue] = []
  last_end_hms: str | None = None
  i = 0
  while i < len(lines):
    # Skip leading blank lines
    while i < len(lines) and not lines[i].strip():
      i += 1
    if i >= len(lines):
      break

    # Cue index line (may be missing, but WhisperX includes it)
    i += 1  # idx_line

    # Timestamp line
    if i >= len(lines):
      break
    ts_line = lines[i].strip()
    i += 1

    m_ts = _TS_RE.match(ts_line)
    if not m_ts:
      # Not an SRT cue; skip until next blank line
      while i < len(lines) and lines[i].strip():
        i += 1
      continue

    start_hms = m_ts.group(1)  # HH:MM:SS
    end_hms_raw = m_ts.group(2)  # HH:MM:SS (SRT end truncated to seconds)
    last_end_hms = end_hms_raw

    # Text lines until blank
    text_parts: list[str] = []
    while i < len(lines) and lines[i].strip():
      text_parts.append(lines[i].strip())
      i += 1

    if not text_parts:
      continue

    joined = " ".join(text_parts)
    spk, txt = _derive_speaker_and_text(joined)

    cues.append(_Cue(start_hms=start_hms, end_hms_raw=end_hms_raw, spk=spk, txt=txt))

  # Emit speaker_lines with contiguous second-precision end times
  out_lines: list[str] = []
  for idx, cue in enumerate(cues):
    if idx < len(cues) - 1:
      end_hms = cues[idx + 1].start_hms
    else:
      end_hms = cue.end_hms_raw

    # Safety: avoid end < start due to any weirdness
    if _hms_to_seconds(end_hms) < _hms_to_seconds(cue.start_hms):
      end_hms = cue.start_hms

    out_lines.append(f"({cue.spk}, {cue.start_hms}-{end_hms}) {cue.txt}".rstrip())

  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")

  transcript_end_hms = last_end_hms or "00:00:00"
  _write_status(
    job.status_path,
    phase="postprocess",
    subphase="speaker_lines",
    message=f"speaker_lines written: {out_path.name}",
    speaker_lines_filename=out_path.name,
    transcript_end=transcript_end_hms,
  )
  return out_path, transcript_end_hms
