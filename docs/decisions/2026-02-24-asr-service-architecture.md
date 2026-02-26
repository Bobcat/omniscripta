# Decision Note: Shared ASR Service Architecture (Profiles + Overrides)

Date: 2026-02-24  
Status: Accepted (design direction)

## Context

The codebase now has two ASR-consuming use cases:

- `live_chunk` (semi-live, chunked, latency-sensitive)
- `upload_audio` (batch upload, richer post-processing, less latency-sensitive)

At the moment, chunk jobs are already split at worker dispatch level (`job_kind`), but WhisperX still runs as a one-shot subprocess per job. This causes per-chunk overhead and backlog growth for longer live sessions.

## Problem

We need a scalable ASR execution model that:

- reuses loaded models (GPU/CPU) across requests
- supports multiple use cases without coupling their orchestration logic
- can later scale by hardware profile / worker pool / scheduler policy

## Decision

Use a **shared ASR service layer** with a **single generic request/response contract** and **profile-based defaults**, while keeping use-case orchestration separate.

### Separation of concerns

1. **Use-case orchestrators (not shared)**

- Live orchestrator: chunking, tail flush, chunk result merge, finalization, UI progress
- Upload orchestrator: file lifecycle, optional diarization/speakers/topics/LLM, batch UX

2. **ASR service interface (shared)**

- Transcribe audio requests using a generic contract (`asr_v1`)
- Resolve options from:
  - `profile_id` defaults
  - limited `overrides`
- Return normalized text/segments/artifact metadata/timings

3. **ASR execution backend (shared)**

- Persistent runner process(es) with warm model(s)
- Scheduling / capacity policy per hardware
- Transport can evolve (IPC first, HTTP/gRPC later) without changing the contract

## Why one contract (not separate contracts per use case)

We expect multiple use cases with overlapping ASR needs. Separate contracts would increase duplication and versioning burden.

One contract with:

- `profile_id` (e.g. `live_fast`, `upload_full`)
- whitelisted `overrides` (e.g. `align_enabled`)

gives us:

- one stable interface
- profile-specific policy and safety
- easier migration to a persistent service

## Why profiles matter (not just arbitrary params)

Pure parameter bags are too loose and hard to operate safely.

Profiles provide:

- known-good defaults per use case
- policy enforcement (what can/can't be overridden)
- easier hardware scheduling and capacity planning

Examples:

- `live_fast`: low-latency defaults, usually no align/diarize
- `upload_full`: richer outputs, align enabled, possibly diarization

## Transport decision (for now)

The contract is transport-agnostic.

Initial implementation should prefer **local IPC** (persistent local process) over FastAPI/HTTP:

- lower complexity
- easier process lifecycle / GPU ownership
- fewer moving parts while validating performance gains

FastAPI/gRPC remains a future transport/deployment option once the contract and scheduling model are proven.

## Scaling model (future)

The scalable unit is **not** one process per user stream.

Instead:

- a small number of persistent ASR runner processes per hardware/GPU/profile
- multiple orchestrators submit requests to that pool
- scheduler/policy decides admission and priority

This avoids duplicating model loads in GPU memory per user/session.

## Near-term implications

1. `live_chunk` should be the first user of a persistent ASR runner (warm model optimization).
2. `upload_audio` can stay on the current one-shot path initially.
3. Once stable, `upload_audio` can migrate to the same ASR service contract/backend.

## Current Implementation Note (Important)

The current "persistent warm runner" is **per worker process**, not a shared machine-wide ASR service yet.

- `worker` process -> local IPC -> **that worker's own** warm runner child process
- multiple workers do **not** automatically share one warm runner/model instance
- therefore, multiple workers can still multiply VRAM usage until a shared ASR service/pool is introduced

This is an intentional intermediate step:

- it solves per-chunk cold-start overhead in the current single-worker path
- it does not yet solve multi-worker model sharing/scheduling

## Related Notes

- `docs/decisions/2026-02-24-semilive-concurrency-and-spool-retention.md`
- `docs/decisions/2026-02-16-worker-parallelisatie.md`
- `docs/decisions/opus_4_6_semi_live_thoughts.md`
