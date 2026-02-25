# Decision Note: Semi-Live Concurrency and Spool/Retention

Date: 2026-02-24  
Status: Accepted (near-term guidance)

## Context

The live recording flow has been moved away from WhisperLive to a semi-live chunked approach:

- browser audio -> live recorder/chunker
- chunk jobs -> filesystem queue (`inbox/running/done/error`)
- worker -> WhisperX -> SRT
- portal merges chunk results into one live session transcript

Recent changes introduced explicit `job_kind` values:

- `upload_audio`
- `live_chunk`

This allows separate worker pipelines while still sharing the same queue backend and worker service.

Relevant code (current):

- `portal-api/queue_fs.py` (`job_kind` in `job.json` / `status.json`)
- `portal-api/live_chunk_transcribe.py` (enqueues `job_kind="live_chunk"`)
- `worker/worker_daemon.py` (dispatch on `job_kind`)
- `worker/pipeline_live_chunk.py` (lean live chunk path)

## Problem 1: Concurrency / Resource Use

### Current behavior with 1 worker service

With a single worker instance (e.g. `transcribe-worker-dev@1`) jobs are processed sequentially:

- upload jobs and live chunk jobs share the same `inbox`
- one worker claims one job at a time
- no parallel WhisperX inference happens

Implication:

- no immediate GPU concurrency/OOM risk from multiple worker processes
- no GPU lock/semaphore is required yet

### What still matters with 1 worker

Even with one worker, there is a QoS/latency issue:

- long upload jobs can delay live chunk jobs
- many live chunk jobs can delay upload jobs

This is a scheduling/fairness issue, not a GPU concurrency issue.

### What changes when >1 workers are enabled

If multiple worker instances are started, the queue model allows parallel processing:

- workers claim different jobs atomically
- `upload_audio` and `live_chunk` can run at the same time
- WhisperX models may load concurrently on the same GPU

Risks:

- GPU VRAM pressure / OOM
- worse throughput due to contention
- unpredictable latency for live chunks

## Decision (Concurrency)

### Near-term (single worker)

Do not implement a resource-concurrency layer yet.

Reason:

- unnecessary complexity with one worker
- no real parallel GPU work to control yet

### Next step before multi-worker rollout

Before enabling multiple worker instances for production-like usage:

1. Keep `job_kind` split (already started).
2. Add explicit resource policy for WhisperX phases.
3. Add scheduling policy (or separate queues) for `upload_audio` vs `live_chunk`.

### Recommended future resource policy

Start conservative:

- global WhisperX concurrency = `1` (per host/GPU)

This can be enforced with a file-based semaphore/lock and later increased only after measurement.

Reference: aligns with the earlier parallelization note in
`docs/decisions/2026-02-16-worker-parallelisatie.md`, but semilive introduces a new QoS dimension (`live_chunk` responsiveness).

## Problem 2: Spool / Retention Growth

### Current behavior

Each live chunk is currently a normal queue job:

- one chunk -> one job directory
- lifecycle: `inbox -> running -> done/error`
- each job directory contains multiple subdirectories (`upload`, `snippet`, `whisperx`, `result`)

Additionally, semilive chunk enqueue currently stores a session-side chunk WAV copy under:

- `data/live_chunk_jobs/<session_id>/...`

Result:

- many directories and files for long sessions
- inode growth
- duplicate storage (session chunk copy + queue upload copy)

### Why this is acceptable short-term

It enabled fast reuse of the existing queue/worker infrastructure and reduced implementation risk.

### Why it should be improved

Long recordings can generate many chunk jobs. Keeping all successful chunk job dirs indefinitely is not useful once their transcript has been merged into the live session result.

## Decision (Spool / Retention)

### Near-term cleanup policy (recommended next)

Treat `live_chunk` jobs as ephemeral spool items.

After a chunk result has been successfully polled and merged into the live session:

- delete successful `done/<job_id>` live chunk job dirs
- optionally delete corresponding session-side chunk WAV file

Keep only session-level truth:

- full recording (`wav`)
- merged semilive transcript (`txt/srt/segments`)
- compact chunk status metadata in `live_sessions`

### Failure retention policy

Keep only a limited amount of failure data for debugging:

- keep failed chunk jobs for the last `N` chunks per session (e.g. `N=3`)
- or keep failures only when debug mode is enabled

This avoids losing debuggability while preventing unbounded growth.

### Medium-term storage improvements

1. Minimize `live_chunk` job layout

For `job_kind=live_chunk`, create only the directories actually needed by that pipeline (likely `upload`, `whisperx`, maybe `result`).

2. Remove duplicate chunk WAV storage

Instead of writing the chunk WAV to `data/live_chunk_jobs/...` and copying it into the queue job:

- write directly into the job `upload/` directory
- keep session-side copies only when debug capture is enabled

3. Add periodic cleanup

Add a simple maintenance task to purge old semilive spool leftovers (e.g. abandoned chunk files, stale debug chunk dirs).

## Scheduling and Retention Strategy (combined view)

The important separation is:

- `resource control`: protect GPU (needed when multi-worker)
- `scheduling/QoS`: ensure live chunk responsiveness (can matter even with one worker)
- `retention`: avoid spool directory explosion (matters now)

These should be implemented independently.

## Implementation Order (recommended)

1. Finish worker pipeline split (`upload_audio` vs `live_chunk`) without hidden branching.
2. Add semilive spool cleanup for successful chunk jobs (ephemeral retention).
3. Reduce duplicate chunk WAV storage.
4. Add minimal `live_chunk` job layout.
5. Add scheduling policy / queue separation if needed.
6. Add resource semaphore/lock before running multiple workers on one GPU.

## Non-Goals (for this note)

- No immediate multi-worker enablement
- No persistent warm-model ASR worker yet
- No queue backend replacement (still filesystem queue)

## Open Questions (track later)

1. Should `live_chunk` and `upload_audio` get separate inboxes, or is one inbox + priority claiming enough?
2. How much failed chunk evidence should be retained by default for support/debugging?
3. Should semilive spool cleanup happen in the portal after merge, in the worker after done, or in a dedicated janitor task?
4. When warm-model optimization is introduced, should it be a dedicated `live_chunk` worker service/template?
