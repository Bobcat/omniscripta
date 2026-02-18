from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any


def _cleanup_torch(torch_mod: Any) -> None:
  gc.collect()
  try:
    if torch_mod.cuda.is_available():
      torch_mod.cuda.empty_cache()
  except Exception:
    pass


def _as_positive_int(value: Any) -> int | None:
  try:
    v = int(value)
  except Exception:
    return None
  if v <= 0:
    return None
  return v


def _apply_torch_thread_tuning(
  torch_mod: Any,
  *,
  torch_num_threads: int | None,
  torch_num_interop_threads: int | None,
) -> dict[str, Any]:
  errors: list[str] = []
  if torch_num_threads is not None:
    try:
      torch_mod.set_num_threads(int(torch_num_threads))
    except Exception as e:
      errors.append(f"set_num_threads_failed:{e!r}")
  if torch_num_interop_threads is not None:
    try:
      torch_mod.set_num_interop_threads(int(torch_num_interop_threads))
    except Exception as e:
      errors.append(f"set_num_interop_threads_failed:{e!r}")

  effective_threads: int | None = None
  effective_interop: int | None = None
  try:
    effective_threads = int(torch_mod.get_num_threads())
  except Exception as e:
    errors.append(f"get_num_threads_failed:{e!r}")
  try:
    effective_interop = int(torch_mod.get_num_interop_threads())
  except Exception as e:
    errors.append(f"get_num_interop_threads_failed:{e!r}")

  return {
    "requested": {
      "torch_num_threads": torch_num_threads,
      "torch_num_interop_threads": torch_num_interop_threads,
    },
    "effective": {
      "torch_num_threads": effective_threads,
      "torch_num_interop_threads": effective_interop,
    },
    "errors": errors,
  }


def _is_likely_hf_gated_none_pipeline_error(exc: Exception) -> bool:
  # whisperx.diarize currently chains `.to(device)` directly after
  # Pipeline.from_pretrained(...). When Hugging Face access is denied for a
  # gated/private model, from_pretrained may yield None and trigger this shape.
  msg = str(exc).lower()
  return isinstance(exc, AttributeError) and ("nonetype" in msg) and ("to" in msg)


def _run(args_obj: dict[str, Any], out_json: Path) -> int:
  import whisperx
  from whisperx.utils import get_writer
  import torch

  snippet_path = Path(str(args_obj["snippet_path"]))
  whisperx_out_dir = Path(str(args_obj["whisperx_out_dir"]))
  whisperx_out_dir.mkdir(parents=True, exist_ok=True)

  language = str(args_obj.get("language", "nl") or "nl")
  speaker_mode = str(args_obj.get("speaker_mode", "auto") or "auto").strip().lower()
  if speaker_mode in {"none", "off", "disabled", "no_speaker", "nospeaker", "no-speaker"}:
    speaker_mode = "none"
  min_speakers = args_obj.get("min_speakers")
  max_speakers = args_obj.get("max_speakers")

  model_name = str(args_obj.get("model", "large-v3") or "large-v3")
  device = str(args_obj.get("device", "cuda") or "cuda")
  compute_type = str(args_obj.get("compute_type", "float16") or "float16")
  batch_size = int(args_obj.get("batch_size", 3))
  chunk_size = int(args_obj.get("chunk_size", 30))
  beam_size = int(args_obj.get("beam_size", 5))
  align_model = str(args_obj.get("align_model") or "").strip() or None
  diarize_model = str(args_obj.get("diarize_model") or "").strip() or None
  torch_num_threads = _as_positive_int(args_obj.get("torch_num_threads"))
  torch_num_interop_threads = _as_positive_int(args_obj.get("torch_num_interop_threads"))

  durations: dict[str, float] = {}

  thread_info = _apply_torch_thread_tuning(
    torch,
    torch_num_threads=torch_num_threads,
    torch_num_interop_threads=torch_num_interop_threads,
  )
  thread_info["env"] = {
    "OMP_NUM_THREADS": (os.getenv("OMP_NUM_THREADS") or "").strip(),
    "MKL_NUM_THREADS": (os.getenv("MKL_NUM_THREADS") or "").strip(),
  }

  print(f"INFO whisperx_file={getattr(whisperx, '__file__', '<unknown>')}", flush=True)
  print(
    f"INFO config model={model_name} device={device} compute={compute_type} batch={batch_size} chunk={chunk_size} beam={beam_size} language={language} speaker_mode={speaker_mode}",
    flush=True,
  )
  print(f"INFO thread_tuning={json.dumps(thread_info, ensure_ascii=False, sort_keys=True)}", flush=True)

  print("STAGE transcribe", flush=True)
  t0 = time.monotonic()
  asr_model = whisperx.load_model(
    model_name,
    device=device,
    compute_type=compute_type,
    language=language,
    asr_options={"beam_size": beam_size},
    vad_options={"chunk_size": chunk_size},
  )
  audio = whisperx.load_audio(str(snippet_path))
  result = asr_model.transcribe(
    audio,
    batch_size=batch_size,
    chunk_size=chunk_size,
    print_progress=False,
    verbose=False,
  )
  durations["transcribe"] = time.monotonic() - t0
  print(f"TIMING transcribe {durations['transcribe']:.6f}", flush=True)
  print(f"INFO transcribe_segments={len(result.get('segments') or [])}", flush=True)

  del asr_model
  _cleanup_torch(torch)

  print("STAGE align", flush=True)
  t0 = time.monotonic()
  align_language = str(result.get("language") or language or "en")
  aligner, align_meta = whisperx.load_align_model(align_language, device, model_name=align_model)
  aligned: dict[str, Any]
  if aligner is not None and len(result.get("segments") or []) > 0:
    aligned = whisperx.align(
      result["segments"],
      aligner,
      align_meta,
      audio,
      device,
      return_char_alignments=False,
      print_progress=False,
    )
  else:
    aligned = {"segments": result.get("segments") or []}
  aligned["language"] = str(align_meta.get("language") or align_language)
  durations["align"] = time.monotonic() - t0
  print(f"TIMING align {durations['align']:.6f}", flush=True)
  print(f"INFO aligned_segments={len(aligned.get('segments') or [])}", flush=True)

  del aligner
  _cleanup_torch(torch)

  if speaker_mode == "none":
    print("INFO diarize_skipped speaker_mode=none", flush=True)
  else:
    from whisperx.diarize import DiarizationPipeline

    print("STAGE diarize", flush=True)
    t0 = time.monotonic()
    hf_token = (os.getenv("HF_TOKEN") or "").strip() or None
    if not hf_token:
      print("WARN HF_TOKEN is not set; diarization model loading may fail", flush=True)
    diarize_kwargs: dict[str, Any] = {}
    if speaker_mode == "fixed":
      if min_speakers is not None:
        diarize_kwargs["min_speakers"] = int(min_speakers)
      if max_speakers is not None:
        diarize_kwargs["max_speakers"] = int(max_speakers)
    try:
      diarize_pipe = DiarizationPipeline(
        model_name=diarize_model,
        use_auth_token=hf_token,
        device=device,
      )
    except Exception as e:
      if _is_likely_hf_gated_none_pipeline_error(e):
        model_ref = str(diarize_model or "pyannote/speaker-diarization-3.1")
        raise RuntimeError(
          f"Could not load diarization model '{model_ref}'. "
          "Hugging Face access is likely missing for this gated/private model. "
          f"Accept model terms at https://hf.co/{model_ref} and ensure HF_TOKEN has read access."
        ) from e
      raise
    diarize_df = diarize_pipe(str(snippet_path), **diarize_kwargs)
    aligned = whisperx.assign_word_speakers(diarize_df, aligned)
    durations["diarize"] = time.monotonic() - t0
    print(f"TIMING diarize {durations['diarize']:.6f}", flush=True)

    del diarize_pipe
    _cleanup_torch(torch)

  print("STAGE finalize", flush=True)
  t0 = time.monotonic()
  writer = get_writer("srt", str(whisperx_out_dir))
  writer_args = {"highlight_words": False, "max_line_count": None, "max_line_width": None}
  writer(aligned, str(snippet_path), writer_args)
  durations["finalize"] = time.monotonic() - t0
  print(f"TIMING finalize {durations['finalize']:.6f}", flush=True)

  srt_path = whisperx_out_dir / f"{snippet_path.stem}.srt"
  if not srt_path.exists():
    srts = sorted(whisperx_out_dir.glob("*.srt"), key=lambda p: p.stat().st_mtime)
    if not srts:
      raise RuntimeError(f"No .srt produced in {whisperx_out_dir}")
    srt_path = srts[-1]

  out_obj = {
    "srt_path": str(srt_path),
    "timings": {k: round(v, 6) for k, v in durations.items()},
    "transcribe_segments": int(len(result.get("segments") or [])),
    "aligned_segments": int(len(aligned.get("segments") or [])),
  }
  out_json.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
  print(f"INFO srt_path={srt_path}", flush=True)
  return 0


def main() -> int:
  parser = argparse.ArgumentParser(description="Run WhisperX pipeline via Python import and write SRT + metadata JSON.")
  parser.add_argument("--args-json", required=True)
  parser.add_argument("--out-json", required=True)
  ns = parser.parse_args()

  args_path = Path(ns.args_json)
  out_path = Path(ns.out_json)
  args_obj = json.loads(args_path.read_text(encoding="utf-8"))
  return _run(args_obj, out_path)


if __name__ == "__main__":
  raise SystemExit(main())
