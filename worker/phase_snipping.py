from __future__ import annotations

import subprocess
from pathlib import Path


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
