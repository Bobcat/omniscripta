# Decision Note: Progress Database Schema (Timing-First)

Date: 2026-02-16  
Status: Draft (agreed baseline for implementation)

## Goal

Store recent run timings to derive predictive progress mapping later, without relying on WhisperX callbacks/stdout detail.

## Storage Model

- Runtime files (local, ignored by git): `data/progress_db/`
- Versioned schema/docs (in git): `docs/progress/`

## Run Record (JSONL)

Each line in `runs_v1.jsonl` is one completed run.

Required fields:

- `schema_version` (string)
- `run_id` (string)
- `job_id` (string)
- `content_hash_sha256` (string, sha256 of uploaded source file)
- `ts_start_utc` (ISO-8601 string)
- `ts_end_utc` (ISO-8601 string)
- `host_id` (string)
- `worker_instance` (string)
- `snippet_seconds` (number)
- `topics_enabled` (boolean)
- `chunks_count` (integer)
- `config_key` (string)
- `hardware_key` (string)
- `phase_seconds` (object: phase -> seconds)
- `wait_seconds` (object: wait category -> seconds)
- `total_seconds` (number)
- `outcome` (`done` or `error`)
- `error_text` (string, empty on success)

## Hardware Profiles

`hardware_profiles_v1.json` maps `hardware_key` to hardware/runtime metadata:

- `host_id`
- `gpu_name`
- `gpu_vram_gb`
- `cuda_version`
- `torch_version`
- `whisperx_version`
- `notes`

## Notes

- Keep only last `N` runs per `config_key` later (rolling window).
- `wait_seconds` is reserved for future semaphore waits (`whisperx_slot`, `llm_slot`).
- `phase_seconds` should represent processing time, not queue wait time.
