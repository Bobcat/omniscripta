# Omniscripta redesign
## Live/upload worker split, submit+reap workers, pool-side live supersede, and portal-api live-state ownership

## Purpose

This document captures **all design changes currently known to be needed** to make the Omniscripta ASR pipeline more responsive for live transcription while preserving the current safety and VRAM constraints for upload transcription.

The intended ownership split is:

- the **live worker should remain deliberately dumb**
- the **ASR pool owns queued live-request replacement**
- the **portal-api owns live sequencing, freshness selection, and transcript application logic**

That split is cleaner than making the live worker itself track rich per-session live state.

---

## Scope and target outcome

The desired end state is:

- **live** and **upload** no longer compete inside one blocking worker loop
- workers keep polling the inbox for new jobs instead of blocking on one request result
- workers **submit quickly** to the ASR pool and **reap results later**
- the **live worker stays simple** and does not own session ordering semantics
- the **ASR pool** can treat newer queued live requests as replacements for older queued live requests from the same live session
- the **portal-api** keeps the real live session state and decides which finished live jobs still matter
- the rolling live path in `portal-api` no longer artificially serializes live inference to one in-flight chunk
- upload behavior remains conservative and VRAM-aware
- the system keeps its current strengths:
  - simple filesystem-based inbox
  - pool-owned scheduling and VRAM policy
  - WhisperX execution isolated in ASR-pool runner subprocesses
  - operational simplicity

This redesign does **not** require a full asyncio rewrite of the existing worker in phase 1.

---

## High-level redesign themes

There are **four** main redesign themes.

### 1. Split live and upload worker service instances

Do not keep one worker process that tries to serve both workloads fairly from one blocking processing loop.

Instead, keep **one worker codebase** but run it as **two differently configured service instances**:

- `worker-live`
- `worker-upload`

This gives runtime separation without immediately forcing separate codebases.

### 2. Workers should keep polling inbox jobs and reap results later

Workers should stop using the current pattern:

- claim one job
- submit it to ASR-pool
- block on per-request polling until the pool finishes
- only then claim another job

Instead, workers should move to:

- keep polling inbox for new jobs
- claim only allowed job kinds
- submit quickly to ASR-pool
- record only minimal outstanding bookkeeping needed for reaping/finalization
- continue claiming new jobs until mode-specific limits are reached
- separately reap terminal results from the pool and finalize worker jobs

This is the key change that removes the current “one outstanding request per worker” bottleneck.

### 3. The ASR pool should replace older **pending** live requests with newer ones for the same session

For live workloads, newer requests may contain fresher context and may render older queued work obsolete.

The pool should therefore understand that multiple queued live requests for the same `(live_session_id, live_lane)` are not always independent work items. It should keep the latest relevant queued request and internally supersede older queued requests.

This should apply to **queued/pending** requests, not running requests, in the first design phase.

### 4. The portal-api rolling live path must stop enforcing single-inflight live inference and must own live result selection

This is the major end-to-end dependency that has to be included in the redesign.

Right now the live rolling path in `portal-api` effectively tracks a single in-flight inference item and polls that one item. If another live inference opportunity comes in while that one is still outstanding, the path does not continue to feed fresh work into the system in the way the new worker/pool design would need.

In the redesigned system, `portal-api` must also own:

- per-session job/request registry
- sequencing metadata such as chunk index or preview sequence
- deciding which done jobs still matter
- ignoring stale completions that have been overtaken by newer work

So the redesign is not complete unless the rolling live producer path is also updated.

---

## Current architecture summary

## Portal API

Today `portal-api` is responsible for at least two distinct kinds of upstream production into the worker queue:

- **upload path**: creates ordinary upload jobs
- **live rolling path**: creates `live_chunk` jobs

The rolling live path currently keeps internal state around a single in-flight inference item and periodically polls job status. This means live work is already being serialized at the API-layer orchestration level before worker/pool improvements can help.

### Consequence

If the worker and pool are redesigned but the portal-api live rolling logic remains single-inflight, live responsiveness will still be capped by the portal layer.

## Worker

The current worker daemon is effectively one-job-at-a-time.

Its current shape is roughly:

1. claim oldest job from inbox
2. process that job fully
3. if the job needs remote ASR, submit to ASR-pool
4. poll for that one request result until terminal
5. finalize the worker job
6. only then go back to inbox claim

This means a single worker instance usually has only one outstanding ASR-pool request at a time.

### Consequence

The worker is the current practical bottleneck for overlapping submission.

## ASR pool

The ASR pool already has several useful mechanisms:

- request validation and profile resolution
- priority separation such as interactive vs background
- warm runners per slot
- background throttling / single-flight behavior
- special routing of `upload_full` background work to slot 0 for VRAM safety
- request record tracking

However, the pool currently treats requests primarily as independent entries keyed by `request_id`. It does not yet have first-class semantics for:

- “this is a live request”
- “this live request belongs to session X and lane Y”
- “this newer queued request replaces that older queued request”

---

## Design goals

## Goal 1: lower live end-to-end latency

Live should feel more responsive because new live chunks can keep flowing into the system instead of waiting behind the previous end-to-end request chain.

## Goal 2: improve freshness and likely quality for live

If a newer live request supersedes an older queued one for the same session, it is usually better to process the newer one rather than spend time on stale pending work.

This may improve perceived quality because newer requests can carry better context and more complete audio windows.

## Goal 3: preserve safe upload behavior

Uploads should not accidentally become more VRAM-aggressive or bypass current pool rules.

The pool’s slot-0 background policy for `upload_full` should remain authoritative.

## Goal 4: keep the redesign operationally manageable

Prefer the smallest change set that provides the main gains:

- one worker codebase
- multiple worker service instances
- explicit worker modes
- no phase-1 requirement to rewrite the worker into a fully async architecture

## Goal 5: put logic where it belongs

- the **worker** should be simple orchestration glue
- the **pool** should own live queue replacement
- the **portal-api** should own live sequencing and transcript application

That separation is cleaner than trying to make the worker smart about live session semantics.

---

## Why one worker is no longer the right orchestration boundary

A single worker could theoretically be retained if it learned to:

- claim both live and upload jobs
- decide which to prioritize
- limit outstanding requests per lane
- reap completions
- prevent upload jobs from crowding out live
- manage lane-specific policies like supersede vs no-supersede

That is possible, but it turns one worker into an internal scheduler.

Using two worker service instances gives simpler and more reliable separation:

- the live worker owns live submission/reap behavior
- the upload worker owns upload submission/reap behavior
- each can have different outstanding limits
- logs and metrics are clearer
- tuning is easier
- the scheduling guarantee is more explicit

This is why the redesign should prefer **two worker services from one codebase** over one all-in-one worker loop.

---

## Worker split design

## Recommendation

Keep one implementation, but run it as two services:

- `worker-live`
- `worker-upload`

Suggested config field:

- `worker.mode = live | upload | all`

`all` can still exist for development/testing, but production should move toward separate mode-specific instances.

## Why not two separate codebases now?

Because the current worker already imports logic for multiple paths. Splitting code physically right away is not necessary to get the runtime behavior we want.

It is acceptable in phase 1 that the live worker binary still technically contains code for uploads/topics/etc. The important thing now is runtime behavior, not immediate codebase purity.

A future cleanup can introduce:

- `worker_live.py`
- `worker_upload.py`

as thin entrypoints over shared modules.

## Claiming must become mode-aware

This is critical.

It is not enough to start two worker services if both still claim jobs blindly from the same shared inbox.

The claiming logic must respect job kind.

### Required change

Either:

- extend queue claiming so a worker can claim only matching job kinds, or
- physically split inboxes

### Preferred approach

Start with a shared inbox and add claim filtering, for example:

- `claim_next_job(job_kind_filter="live_chunk")`
- `claim_next_job(job_kind_filter="upload_audio")`

Why this is preferred first:

- smallest conceptual change
- preserves existing queue structure
- avoids immediately touching every producer path
- keeps the queue simple

### Alternative later

If queue behavior becomes more complex, separate inboxes may still be introduced later.

---

## Worker redesign: from blocking per-job polling to submit + reap

## Current problem

Today the worker claims a job and then blocks until the pool request finishes. During that time the same worker does not go back to the inbox.

This is the main place where parallelism currently stops.

## New worker model

Each worker should have two responsibilities:

1. **submission**
   - keep polling the inbox
   - claim jobs of the allowed kind
   - submit them quickly to ASR-pool
   - record only enough local state to finalize the right worker job later
   - return immediately to claim more work, subject to mode-specific limits

2. **completion reaping**
   - periodically ask the pool for terminal results
   - map those completions back to worker jobs
   - finalize worker job directories and status files
   - remove terminal items from outstanding state

These are responsibilities, not a required threading model. They may be implemented as one simple main loop that alternates between:

- reaping terminal results
- claiming/submitting new jobs
- sleeping briefly

The key shift is:

- no more “claim one job and stay inside that call stack until pool finishes”
- instead: “claim, submit, remember minimally, continue”

## Upload worker behavior

The upload worker should use a strict limit:

- `max_outstanding_requests = 1`

This gives the architectural cleanup of submit+reap while preserving conservative upload behavior.

Important: this does **not** increase upload ASR throughput by itself, because the pool still constrains `upload_full` background work.

## Live worker behavior

The live worker should remain intentionally simple.

It should:

- claim live jobs
- submit them to the pool
- keep only minimal mapping needed for finalization and local recovery if needed
- reap completions and materialize result files/status
- not own per-session sequencing semantics
- not need to understand whether an older live job has been logically overtaken

In other words, the live worker should **pump jobs into the pool and materialize results back out**, not act as the authority on which live result should win.

## Worker intelligence budget

### Upload worker

The upload worker needs a little state because it enforces `max_outstanding_requests = 1`.

### Live worker

The live worker should need as little state as possible. Ideally:

- `request_id == job_id`
- completion returns `request_id`
- worker can finalize the matching job directory directly

That means the worker does **not** need rich per-session live tracking.

---

## Replace request-id result polling with completion reaping

## Current behavior

The worker currently polls specific request IDs one by one.

This works for one outstanding request, but is awkward once a worker may have several outstanding requests or when the design deliberately wants workers to keep submitting while results come back later.

## Proposed pool API addition

Add a completion-oriented endpoint such as:

- `GET /asr/v1/completions?consumer=<id>&since_seq=<n>`

Possible response:

```json
{
  "next_seq": 104,
  "events": [
    {
      "seq": 101,
      "request_id": "job_abc",
      "state": "completed",
      "profile_id": "live_fast",
      "context": {
        "live_session_id": "sess_1",
        "live_chunk_index": 17,
        "live_lane": "final"
      },
      "result_ref": "...",
      "error": ""
    }
  ]
}
```

## Why a completion feed is better

It supports the desired worker model naturally:

- workers can have several requests outstanding
- one reap loop can process all terminal updates
- live and upload can share the same completion mechanism
- request-id polling becomes optional rather than fundamental

## Consumer scoping

Do not make the completion feed totally global by default.

Choose **consumer identity** now and keep it stable across restarts.

Initial recommendation:

- `consumer_id = "worker-live@1"`
- `consumer_id = "worker-upload@1"`

This is clearer than scoping by profile or session and is simple for implementation. It avoids one worker seeing events it never submitted while remaining stable enough for service-level recovery logic.

## Terminal states relevant to workers

Worker-facing completion reaping needs to surface terminal states that matter for worker finalization and outstanding cleanup, such as:

- `completed`
- `failed`
- `cancelled` if that is externally relevant
- `superseded`

A `superseded` result should be treated as terminal. The worker does not need to understand live-session semantics, but it does need to materialize a terminal worker job state so that the job does not remain outstanding forever.

---

## Pool-side live supersede semantics

## Why the pool should own this

The pool is already the authoritative owner of:

- request records
- queue membership
- running vs queued state
- scheduling order

That makes it the correct place to decide whether a newly submitted live request makes an older **queued** request obsolete.

## Core rule

For a given live key:

- `(live_session_id, live_lane)`

only the newest queued request should remain pending.

### Meaning of pending

In this proposal, **pending** means:

- accepted by pool
- queued
- not yet running

This rule must **not** forcibly replace a running request in phase 1.

## Initial policy

For each `(live_session_id, live_lane)` key:

- allow at most **1 running** request
- allow at most **1 queued latest** request
- when a newer request arrives, replace any older queued request for that same key
- leave any currently running request untouched

## Ordering key

Use `live_chunk_index` as the freshness/version key.

If a new request arrives with:

- lower index than known latest: reject or ignore as stale
- equal index: treat as duplicate or idempotent retry
- higher index: it may supersede the older queued request

## Worker visibility and terminal materialization

The pool should internally know that an older queued live request was superseded.

The worker still needs to hear about `superseded` as a terminal outcome, even if it stays dumb about live-session semantics. Otherwise the matching worker job can remain outstanding indefinitely.

So:

- **pool internal state/diagnostics** should include `superseded`
- **worker-facing completion reaping** should also surface `superseded` as a terminal outcome
- the worker should simply finalize the matching job as terminal without trying to interpret session ordering

## Result expectations

A superseded request should:

- become terminal
- produce no ASR payload result
- be materialized as a terminal worker job outcome, for example in `done/` with `status.state = "superseded"`
- remain available for diagnostics/metrics/admin inspection if useful

No separate `superseded/` directory is needed in the initial design.

---

## Portal-api must own live session sequencing and result selection

## Why portal-api should own this

Portal-api already owns the live session abstraction and is the natural place for:

- transcript merge logic
- preview/final transcript application
- offset progression
- deciding whether a completed live job is still relevant

It is therefore cleaner for portal-api, not the worker, to keep the authoritative live-session registry.

## Portal-api live registry

For each live session, portal-api should track at least:

- `job_id` / `request_id`
- `live_chunk_index`
- `preview_seq` or equivalent sequence number if used
- `live_lane`
- relevant timing metadata such as `t0_ms` / `t1_ms`
- whether a result is still wanted, already applied, ignored, or stale

The portal-api does not need to poll every request ID individually if worker job completion or pool completion reaping already materializes the done/error state into the job workspace.

## What portal-api should do with done jobs

Portal-api should be able to decide:

- this completed live job is still the newest relevant one → process it
- this completed live job has been overtaken by a newer one → ignore it
- this completed live job may still be useful for preview but not for commit → apply selectively

This is where freshness and correctness policy belongs.

## Important consequence

The system should no longer rely on the assumption that every finished live job must be consumed just because it completed successfully.

Portal-api should be allowed to ignore stale completed live jobs.

---

## Portal-api rolling live path must stop single-inflight gating

## Current issue

The rolling live engine currently behaves like it has a single live inference slot in flight. That means it stops emitting new live work while one earlier job is still outstanding.

## Consequence

If only the worker and pool are changed, live may still remain serialized because portal-api will continue to emit at most one live chunk job at a time.

## Required redesign

The rolling path must stop using a single inflight slot as its core model.

Instead, it should track at least:

- newest submitted chunk index per session
- newest interesting completed chunk index per session
- zero or more outstanding live jobs

### Minimal viable redesign

Keep the live rolling code simple by tracking:

- `latest_submitted_seq`
- `latest_applied_seq`
- registry of submitted job IDs / request IDs with sequence metadata

Then portal-api can continue to emit fresh live work even if an earlier chunk is still unfinished.

### Compatibility flag

During rollout, a config flag can keep the old behavior available:

- `LIVE_ROLLING_REQUIRE_SINGLE_INFLIGHT = true|false`

But if such a flag exists, the implementation should actually honor it rather than merely reporting it in diagnostics.

---

## Completion ordering: what is and is not assumed

The system should be careful about **what kind of ordering it assumes**.

## Within one `(live_session_id, live_lane)`

The initial design intentionally chooses a conservative rule:

- at most one running request per `(live_session_id, live_lane)`
- queued replacement only for non-running requests

Under that rule, successful ASR completions for a single `(session, lane)` should normally remain monotonic in chunk order.

In other words, if chunk 17 is already running, chunk 18 should not also be running for the same `(session, lane)` in phase 1; chunk 18 can only be queued and may replace an older queued request.

This is a deliberate simplification, not an ideal maximum-throughput rule. If a free slot exists, allowing chunk 18 to run concurrently could eventually be beneficial, but that would require more complex ordering and application logic. The document should preserve this as a future optimization opportunity rather than treating the phase-1 rule as fundamental.

## Across terminal events generally

However, **terminal event order in general still must not be treated as equivalent to submit order**, because:

- other sessions may interleave
- other lanes may interleave
- internal supersede events may happen for queued requests
- error/cancel behavior may interleave differently from successful completions

## Practical rule

Portal-api should not depend on “whatever finished most recently is always the next thing to apply globally”.

Portal-api should instead apply explicit per-session/per-lane sequencing rules based on stored metadata.

---

## Upload behavior in the new design

The upload worker should follow the same basic worker pattern as the live worker:

- claim matching jobs
- submit quickly to the pool
- reap terminal results later
- finalize the matching worker job

The main differences are policy differences, not architectural differences:

- no live supersede semantics
- `max_outstanding_requests = 1` initially
- requests still submit as `background`
- the pool still enforces slot-0 affinity and background single-flight for `upload_full`

So the worker-side pattern remains aligned across live and upload, while the pool applies different queue policy only where needed.

---

## Suggested implementation plan

## Phase 1: low-risk worker split

1. add worker mode config: `live`, `upload`, optional `all`
2. add `claim_next_job(job_kind_filter=...)`
3. run two service instances from the same worker codebase
4. keep current worker blocking behavior briefly if necessary for a transitional step

Outcome:

- live and upload no longer compete for the same worker loop
- immediate scheduling clarity

## Phase 2: worker submit + reap model

1. add outstanding request tracking in worker
2. implement both worker responsibilities:
   - submit newly claimed jobs promptly
   - reap terminal pool results and finalize jobs
3. keep upload worker limit at `1`
4. keep live worker intentionally simple and only as smart as needed for submit/reap/finalization

The implementation does not need two concurrent loops. One simple main loop that alternates between reaping, claiming/submitting, and sleeping briefly is acceptable.

Outcome:

- workers no longer stop polling inbox while awaiting one result

## Phase 3: pool completion feed

1. add completion event store or completion cursor mechanism in pool
2. expose completion endpoint
3. move worker from request-id polling to completion reaping

Outcome:

- cleaner worker logic for several outstanding requests
- no need for one poll loop per request

## Phase 4: pool-side live supersede

1. identify live requests by `(live_session_id, live_lane, live_chunk_index)`
2. when new live request is submitted:
   - reject stale older-than-latest requests
   - supersede older queued request for same key
   - keep running request unchanged for now
3. surface `superseded` as a terminal completion state as well as in diagnostics

Outcome:

- stale queued live work stops accumulating
- workers can clear superseded jobs from outstanding state cleanly

## Phase 5: portal-api live engine redesign

1. remove single `rolling_inflight` bottleneck
2. allow more than one outstanding live job per session
3. maintain a per-session registry of submitted jobs/requests and sequencing metadata
4. define explicit emission criteria for when a new live chunk may be sent while others are still in flight
5. update preview/commit logic to tolerate stale completions and selectively ignore them
6. guard final transcript application by sequence and timestamp sanity

The emission criteria must stay consistent with the live worker's configured `max_outstanding_requests`, otherwise portal-api can overfill the inbox or create pointless backlog. The exact policy still needs to be chosen, but the design must include one. Examples include:

- per-session maximum outstanding live jobs
- minimum additional audio/context before emitting another chunk
- minimum spacing between emissions
- skipping emission if the next candidate chunk adds too little beyond already submitted work

Outcome:

- live end-to-end path can finally exploit the worker/pool changes without flooding the inbox

---

## Correctness and safety concerns

## Supersede safety

Only queued, not running, live requests should be superseded initially.

Why:

- running cancellation is more complex
- current system does not appear to support true hard interruption of active ASR work cleanly
- queued replacement already captures most of the freshness benefit

## Session identity

Pool-side supersede must rely on stable metadata already present in live requests:

- `live_session_id`
- `live_lane`
- `live_chunk_index`

These fields should be treated as required for any request using live supersede behavior.

## Portal-api stale-result handling

Portal-api must explicitly support the idea that a completed live job may be valid as a computation result but no longer wanted as a transcript update.

That is a feature, not an error.

## Diagnostics

Add metrics and logs for:

- live requests submitted
- live requests completed
- live requests superseded
- stale live requests rejected
- upload requests submitted
- outstanding counts per worker
- outstanding counts per session/lane where helpful
- portal-api emitted jobs vs ignored stale completions
- end-to-end live latency percentiles

---

## Final target state

### Live path

- portal-api may have more than one live chunk outstanding per session
- live worker keeps polling inbox and submitting live jobs quickly
- live worker reaps terminal completions from pool and materializes job outputs/status, including `superseded` when relevant
- pool keeps only newest queued live request per `(session, lane)`
- portal-api keeps the authoritative session registry and decides which done jobs to process or ignore

### Upload path

- upload worker keeps polling inbox
- upload worker may submit one upload request and reap later
- pool still serializes upload-heavy background work according to VRAM policy

### System outcome

- better live freshness
- less stale queued live work
- less worker self-blocking
- minimal disruption to upload safety model
- cleaner ownership boundaries between portal-api, worker, and pool

---

## Open questions

1. What should the initial live outstanding limit be?
2. Should stale live submissions be rejected at pool submit time or accepted and immediately marked superseded?
3. What exact portal-api emission policy should control when a new live chunk is sent while others remain in flight?
4. How much stale-result tolerance should portal-api allow for preview vs final transcript application?
5. Should the worker persist minimal outstanding state for crash recovery, or is in-memory plus directory scan enough for phase 1?

---

## Practical bottom line

The redesign is sound, but the final correct division of responsibilities is:

- **worker:** simple submit + reap orchestration
- **ASR pool:** queue policy and queued live-request replacement
- **portal-api:** live session registry, sequencing, freshness, and transcript application

That is the cleanest version of the design currently known.
