from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any


PHASE_ORDER_BASE = [
  "snipping",
  "whisperx_prepare",
  "whisperx_transcribe",
  "whisperx_align",
  "whisperx_diarize",
  "postprocess",
]

AUDIO_PHASES = {"whisperx_transcribe", "whisperx_align", "whisperx_diarize"}

DEFAULTS_SECONDS = {
  "snipping": 5.0,
  "whisperx_prepare": 2.0,
  "whisperx_transcribe": 12.0,
  "whisperx_align": 8.0,
  "whisperx_diarize": 12.0,
  "postprocess": 0.5,
  "llm_topics": 8.0,
  "llm_topics_skipped": 0.0,
}


@dataclass(frozen=True)
class ProgressPrediction:
  phase_expected_s: dict[str, float]
  total_expected_s: float
  confidence: float
  hints: list[str]
  sample_count: int


def _safe_float(v: Any) -> float | None:
  try:
    return float(v)
  except Exception:
    return None


def _phase_name_for_topics(topics_enabled: bool) -> str:
  return "llm_topics" if topics_enabled else "llm_topics_skipped"


def phase_order_for_job(*, topics_enabled: bool) -> list[str]:
  return [*PHASE_ORDER_BASE, _phase_name_for_topics(topics_enabled)]


def _iter_done_records(runs_path: Path) -> list[dict[str, Any]]:
  if not runs_path.exists():
    return []
  rows: list[dict[str, Any]] = []
  with runs_path.open("r", encoding="utf-8") as f:
    for line in f:
      s = line.strip()
      if not s:
        continue
      try:
        obj = json.loads(s)
      except Exception:
        continue
      if str(obj.get("outcome", "")) != "done":
        continue
      if not isinstance(obj.get("phase_seconds"), dict):
        continue
      rows.append(obj)
  return rows


def build_prediction(
  *,
  runs_path: Path,
  hardware_key: str,
  topics_enabled: bool,
  snippet_seconds: int,
) -> ProgressPrediction:
  phase_order = phase_order_for_job(topics_enabled=topics_enabled)
  snippet_seconds = max(1, int(snippet_seconds))

  all_done = _iter_done_records(runs_path)
  candidates = [r for r in all_done if str(r.get("hardware_key", "")) == str(hardware_key)]

  hints: list[str] = []
  n = len(candidates)
  if n == 0:
    hints.append("cold_start")
  if n < 5:
    hints.append("low_sample_n")

  # For audio-dependent phases, model by seconds-per-audio-second.
  expected: dict[str, float] = {}
  used_defaults = False

  # Snippet-range extrapolation hint (based on matching hardware only).
  if n > 0:
    snips = [int(r.get("snippet_seconds", 0)) for r in candidates if int(r.get("snippet_seconds", 0)) > 0]
    if snips and (snippet_seconds < min(snips) or snippet_seconds > max(snips)):
      hints.append("extrapolated_snippet_length")

  for phase in phase_order:
    if phase in AUDIO_PHASES:
      rates: list[float] = []
      for r in candidates:
        sec = _safe_float((r.get("phase_seconds") or {}).get(phase))
        snip = _safe_float(r.get("snippet_seconds"))
        if sec is None or snip is None or snip <= 0:
          continue
        if sec < 0:
          continue
        rates.append(sec / snip)
      if rates:
        exp = median(rates) * snippet_seconds
      else:
        exp = DEFAULTS_SECONDS.get(phase, 1.0) * (snippet_seconds / 900.0)
        used_defaults = True
      expected[phase] = max(0.0, float(exp))
      continue

    vals: list[float] = []
    for r in candidates:
      sec = _safe_float((r.get("phase_seconds") or {}).get(phase))
      if sec is None or sec < 0:
        continue
      vals.append(sec)
    if vals:
      exp = median(vals)
    else:
      exp = DEFAULTS_SECONDS.get(phase, 1.0)
      used_defaults = True
    expected[phase] = max(0.0, float(exp))

  if used_defaults:
    hints.append("phase_defaults")

  total = sum(expected.get(p, 0.0) for p in phase_order)
  # Confidence is sample-count based for now; capped so UI can still show uncertainty.
  confidence = min(0.95, max(0.05, n / 20.0))

  # keep stable order, no duplicates
  ordered_hints: list[str] = []
  for h in ("cold_start", "low_sample_n", "extrapolated_snippet_length", "phase_defaults"):
    if h in hints and h not in ordered_hints:
      ordered_hints.append(h)

  return ProgressPrediction(
    phase_expected_s={k: round(v, 6) for k, v in expected.items()},
    total_expected_s=round(max(1.0, float(total)), 6),
    confidence=round(float(confidence), 6),
    hints=ordered_hints,
    sample_count=n,
  )
