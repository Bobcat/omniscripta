# Decision Note: Worker Parallelization Strategy

Date: 2026-02-16  
Status: Accepted (for future implementation)

## Context

The backend currently uses a filesystem queue (`inbox` -> `running` -> `done/error`) with atomic claim semantics.  
Future requirements include parallel job handling while protecting limited GPU resources:

- WhisperX capacity target: max 2 concurrent runs (dc1, RTX 5070 Ti)
- LLM/topic capacity target: max 3 concurrent runs (dc2, RTX 5090)

## Decision

Use multiple worker service instances (`transcribe-worker-dev@N`) for queue throughput, plus explicit phase-level concurrency limits:

- Global semaphore `whisperx_slots = 2`
- Global semaphore `llm_slots = 3`

Each claimed job remains single-owner, but phases that consume constrained GPU resources must acquire/release a slot.

## Why This Approach

- Keeps operations simple with systemd templates (`@1`, `@2`, ...)
- Avoids complex in-process pool lifecycle management
- Matches existing queue/claim model with minimal architectural change
- Gives deterministic caps per expensive phase, independent of worker count

## Implementation Plan (Later)

1. Enable multiple worker instances as needed (`transcribe-worker-dev@1..@N`).
2. Add a small semaphore helper used by worker phases:
   - acquire before WhisperX phase
   - release after WhisperX phase
   - acquire before LLM/topics phase
   - release after LLM/topics phase
3. Expose slot wait status in `status.json` (e.g., `phase=queue_wait_whisperx` / `queue_wait_llm`).

## Semaphore Backend Choice

Initial target (single host worker execution): file-based locks/semaphore state on local disk.  
If workers later run across multiple hosts: migrate semaphore backend to Redis or Postgres advisory locks.

## Out of Scope for This Note

- No code changes in this note
- No immediate switch to process-pool inside one worker service
