from __future__ import annotations

import json
import subprocess
from pathlib import Path


def _ffprobe_json(input_path: Path, *, show_entries: str) -> dict:
  cmd = [
    "ffprobe",
    "-v",
    "error",
    "-select_streams",
    "a:0",
    "-show_entries",
    show_entries,
    "-of",
    "json",
    str(input_path),
  ]
  cp = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
  return json.loads(cp.stdout or "{}")


def _is_mp3_probably_cbr(input_path: Path) -> bool:
  """
  Conservative CBR detection for fast-path stream copy.
  We only allow copy when codec is MP3 and packet sizes in the first seconds
  look nearly constant (typical for CBR).
  """
  try:
    meta = _ffprobe_json(input_path, show_entries="stream=codec_name")
    streams = meta.get("streams") or []
    if not streams:
      return False
    codec_name = str((streams[0] or {}).get("codec_name") or "").strip().lower()
    if codec_name != "mp3":
      return False
  except Exception:
    return False

  try:
    # Read only a short front slice to keep probing cheap.
    cp = subprocess.run(
      [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_packets",
        "-show_entries",
        "packet=size",
        "-of",
        "csv=p=0",
        "-read_intervals",
        "%+20",
        str(input_path),
      ],
      check=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      text=True,
    )
    sizes: list[int] = []
    for raw in (cp.stdout or "").splitlines():
      s = raw.strip()
      if not s:
        continue
      try:
        val = int(s)
      except ValueError:
        continue
      if val > 0:
        sizes.append(val)

    if len(sizes) < 80:
      return False
    mn = min(sizes)
    mx = max(sizes)
    if mn <= 0:
      return False

    unique_count = len(set(sizes))
    spread_ratio = mx / mn
    return unique_count <= 3 and spread_ratio <= 1.15
  except Exception:
    return False


def _make_snippet(input_path: Path, out_dir: Path, seconds: int) -> Path:
  """
  Make first N seconds snippet.
  Output is mp3 for editor friendliness and smaller size.

  Note: filename currently contains `_snippet_5min` even when `seconds` differs.
  Kept as-is to avoid changing existing behavior/UI expectations.
  """
  out_dir.mkdir(parents=True, exist_ok=True)
  suffix = f"{seconds//60}min" if seconds > 0 and (seconds % 60) == 0 else f"{seconds}s"
  out_path = out_dir / f"{input_path.stem}_snippet_{suffix}.mp3"

  if _is_mp3_probably_cbr(input_path):
    cmd = [
      "ffmpeg",
      "-y",
      "-i",
      str(input_path),
      "-t",
      str(int(seconds)),
      "-vn",
      "-map",
      "0:a:0",
      "-c",
      "copy",
      str(out_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Optional future hardening point: run a quick ffprobe sanity check on out_path
    # (duration/timestamps) before returning.
    return out_path

  cmd = [
    "ffmpeg",
    "-y",
    "-i",
    str(input_path),
    "-t",
    str(int(seconds)),
    "-vn",
    "-acodec",
    "libmp3lame",
    "-b:a",
    "128k",
    str(out_path),
  ]
  subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  return out_path
