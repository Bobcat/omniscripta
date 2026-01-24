from __future__ import annotations

import json
import time
from pathlib import Path
from datetime import datetime, timezone

from queue_fs import claim_next_job, finish_job
from worker_status_io import _append_log, _utc_iso, _write_status
from phase_snipping import _make_snippet
from phase_speaker_lines import make_speaker_lines_from_srt
from phase_chunk_speaker_lines import chunk_speaker_lines
from phase_topics_llm import run_topics_llm
from phase_topics_parse import parse_topics_raw_file
from phase_topics_validate import validate_all_chunks
from phase_topics_merge import merge_topics
from phase_whisperx import P_SNIP_DONE, _load_server_config, _run_whisperx_streaming


SLEEP_IDLE_SECONDS = 2.0

SERVICE_CONFIG_PATH = "/srv/transcribe/config/service.json"


def _load_service_config() -> dict:
  # Minimal defaults; override via /srv/transcribe/config/service.json if present
  cfg = {
    "snip": {
      "minutes_default": 5
    },
    "topics": {
      "chunk_minutes": 25,
      "ctx_len": 16384,
      "ctx_safety": 0.85,
      "prompt_overhead_tokens_est": 1200,
      "token_estimator": "chars_div4"
    }
  }
  try:
    p = Path(SERVICE_CONFIG_PATH)
    if p.exists():
      data = json.loads(p.read_text(encoding="utf-8"))
      if isinstance(data, dict):
        # shallow-ish merge for now
        cfg.update(data)
  except Exception:
    # Keep defaults if config is invalid/unreadable
    pass
  return cfg


def main() -> int:
  print("worker_daemon started")
  while True:
    job = claim_next_job()
    if not job:
      time.sleep(SLEEP_IDLE_SECONDS)
      continue

    try:
      now = _utc_iso()
      _write_status(
        job.status_path,
        state="running",
        phase="snipping",
        progress=0.0,
        started_at=now,
        message="Starting job…",
      )

      job_cfg = json.loads(job.job_path.read_text(encoding="utf-8"))
      opts = job_cfg.get("options", {}) or {}
      orig_filename = job_cfg.get("orig_filename")
      service_cfg = _load_service_config()
      snip_cfg = (service_cfg.get("snip") or {}) if isinstance(service_cfg, dict) else {}
      default_min = int(snip_cfg.get("minutes_default", 5))
      if opts.get("snippet_seconds") is not None:
        snippet_seconds = int(opts.get("snippet_seconds"))
      else:
        snippet_seconds = int(default_min * 60)
      language = str(opts.get("language", "nl") or "nl")
      speaker_mode = str(opts.get("speaker_mode", "auto") or "auto")
      min_speakers = opts.get("min_speakers")
      max_speakers = opts.get("max_speakers")

      input_path = job.upload_dir / orig_filename
      if not input_path.exists():
        raise RuntimeError(f"Upload missing: {input_path}")

      disp = f"{snippet_seconds//60} min" if snippet_seconds > 0 and (snippet_seconds % 60) == 0 else f"{snippet_seconds} s"
      _write_status(job.status_path, phase="snipping", progress=0.0, message=f"Creating snippet ({disp})…")
      snippet_path = _make_snippet(input_path, job.snippet_dir, seconds=snippet_seconds)
      _write_status(
        job.status_path,
        phase="snipping",
        progress=P_SNIP_DONE,
        snippet_filename=snippet_path.name,
        message=f"Snippet created: {snippet_path.name}",
      )

      cfg = _load_server_config()
      _append_log(job.log_path, f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] WORKER service_cfg={json.dumps(service_cfg, ensure_ascii=False)}")
      _append_log(job.log_path, f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] WORKER whisperx_cfg={json.dumps(cfg, ensure_ascii=False)}")

      _write_status(job.status_path, phase="whisperx_prepare", progress=P_SNIP_DONE, message="Preparing WhisperX…")

      srt_path = _run_whisperx_streaming(
        job=job,
        snippet_path=snippet_path,
        whisperx_out_dir=job.whisperx_dir,
        snippet_seconds=snippet_seconds,
        language=language,
        speaker_mode=speaker_mode,
        min_speakers=(int(min_speakers) if min_speakers is not None else None),
        max_speakers=(int(max_speakers) if max_speakers is not None else None),
        cfg=cfg,
      )
      # Phase 31: SRT -> speaker_lines
      orig_stem = Path(orig_filename).stem if orig_filename else "transcript"
      _write_status(job.status_path, phase="postprocess", subphase="speaker_lines", progress=0.94, message="Generating speaker_lines…")
      speaker_lines_path, transcript_end_hms = make_speaker_lines_from_srt(job=job, srt_path=srt_path, orig_stem=orig_stem)

      # Phase 32: chunk speaker_lines + manifest
      _write_status(job.status_path, phase="postprocess", subphase="chunk_speaker_lines", progress=0.96, message="Chunking speaker_lines…")
      manifest_path = chunk_speaker_lines(job=job, speaker_lines_path=speaker_lines_path, orig_stem=orig_stem, service_cfg=service_cfg, transcript_end_hms=transcript_end_hms)


      # Phase 40: topics (optional; disabled by default)
      topics_cfg = service_cfg.get("topics", {}) if isinstance(service_cfg, dict) else {}
      topics_enabled = bool(topics_cfg.get("enabled", False))
      prompt_id = str(topics_cfg.get("prompt_id", "topics_v1"))

      if topics_enabled:
        # 41) Call LLM (stub for now: only writes payloads; no raw output)
        run_topics_llm(job=job, manifest_path=manifest_path, orig_stem=orig_stem, prompt_id=prompt_id, service_cfg=service_cfg)

        # 42) Parse (expects *_raw.txt files to exist; since stub doesn't write them, this will be a no-op for now)
        # When PC1 call is implemented, it will write:
        #   <orig_stem>_<prompt_id>_chunk_0001_raw.txt
        # ... and then this loop will produce parsed JSON per chunk.
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for ch in (manifest.get("chunks") or []):
          idx = int(ch["index"])
          raw_path = job.result_dir / f"{orig_stem}_{prompt_id}_chunk_{idx:04d}_raw.txt"
          parsed_path = job.result_dir / f"{orig_stem}_{prompt_id}_chunk_{idx:04d}.json"
          if raw_path.exists():
            parse_topics_raw_file(raw_txt_path=raw_path, out_json_path=parsed_path)

        # 43) Validate
        report_path = job.result_dir / f"{orig_stem}_{prompt_id}_validation.json"
        validate_all_chunks(
          manifest_path=manifest_path,
          parsed_dir=job.result_dir,
          orig_stem=orig_stem,
          prompt_id=prompt_id,
          out_report_path=report_path,
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not report.get("is_valid", False):
          raise RuntimeError(f"Topics validation failed: {report_path.name}")

        # 44) Merge
        merged_path = job.result_dir / f"{orig_stem}_{prompt_id}_merged.json"
        merge_topics(
          manifest_path=manifest_path,
          parsed_dir=job.result_dir,
          orig_stem=orig_stem,
          prompt_id=prompt_id,
          out_merged_path=merged_path,
        )
      else:
        _write_status(job.status_path, phase="postprocess", subphase="chunk_speaker_lines", message="Topics disabled; skipping.")

      # Finalize
      _write_status(
        job.status_path,
        state="done",
        phase="done",
        progress=1.0,
        finished_at=_utc_iso(),
        message="Done",
        srt_filename=srt_path.name,
        speaker_lines_filename=speaker_lines_path.name,
        speaker_lines_manifest_filename=manifest_path.name,
      )
      finish_job(job, ok=True)
      print(f"Done {job.job_id}")

    except Exception as e:
      _write_status(
        job.status_path,
        state="error",
        phase="error",
        progress=1.0,
        message=f"Worker error: {e!r}",
        finished_at=_utc_iso(),
        error=str(e),
      )
      finish_job(job, ok=False)
      print(f"Error {job.job_id}: {e!r}")

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
