# 2026-04-25 Live ASR Wall Timing Breakdown

Date: 2026-04-25  
Scope: dev live path only (`omniscripta` + local `asr-pool-dev`)

## Goal

Pin down where live ASR wall time is spent before changing transport or IPC design.

This note is based on actual instrumentation and two concrete live runs:

- initial analysis run: `live_20260425T181642Z_07c8bbe1`
- validation run after extra pool instrumentation: `live_20260425T184000Z_dd0999c8`
- benchmark exports:
  [live_20260425T181642Z_07c8bbe1.final-quality.latest.json](/home/gunnar/projects/omniscripta/data/live/benchmark_exports/live_20260425T181642Z_07c8bbe1.final-quality.latest.json)
  [live_20260425T184000Z_dd0999c8.final-quality.latest.json](/home/gunnar/projects/omniscripta/data/live/benchmark_exports/live_20260425T184000Z_dd0999c8.final-quality.latest.json)

## Important Reading Rule

All timing totals below are cumulative over all ASR requests in the live session.

They are not one single end-to-end wall clock for the whole 120s recording.

For this run:

- recording duration: `120.0s`
- final chunks: `16`
- pool requests: `176`

## Timing Layers

The current live timing model is:

- `backend_wall_s`
  - measured in [asr_bridge.py](/home/gunnar/projects/omniscripta/app/live/runtime/asr_bridge.py:274)
  - backend-local wall time from before WAV write to terminal result collected
- `pool_wall_s`
  - measured in [service.py](/home/gunnar/projects/asr-pool-dev/app/pool/service.py:794)
  - pool wall time from FastAPI ingest to terminal request state
- `runner_wall_s`
  - currently the runner `total_s`, exposed into live as `runner wall`
  - measured in [server.py](/home/gunnar/projects/asr-pool-dev/app/whisperx/server.py:355)
- `transcribe_call_s`
  - pure ASR decode call inside the runner
- `load_audio_s`
  - audio load/decode inside the runner before the model call

The current live rollup/export wiring is:

- backend merge into live result:
  [asr_bridge.py](/home/gunnar/projects/omniscripta/app/live/runtime/asr_bridge.py:221)
- rolling sums in live session runtime:
  [ws_session.py](/home/gunnar/projects/omniscripta/app/live/runtime/ws_session.py:79)
- benchmark export fields:
  [fixture_scoring.py](/home/gunnar/projects/omniscripta/app/live/quality/fixture_scoring.py:122)

Derived deltas:

- `runner_non_transcribe = runner_wall - transcribe`
- `pool_non_runner = pool_wall - runner_wall`
- `backend_non_pool = backend_wall - pool_wall`

The intended hierarchy is:

- `backend_wall >= pool_wall >= runner_wall >= transcribe`

## New Warm-Runner IPC Metrics

Additional warm-runner IPC timings are now measured here:

- client-side request/response IPC:
  [client.py](/home/gunnar/projects/asr-pool-dev/app/whisperx/client.py:208)
  [client.py](/home/gunnar/projects/asr-pool-dev/app/whisperx/client.py:276)
- runner-side request read:
  [server.py](/home/gunnar/projects/asr-pool-dev/app/whisperx/server.py:473)

Current extra keys:

- `warm_runner_request_read_s`
- `warm_runner_payload_write_s`
- `warm_runner_dispatch_s`
- `warm_runner_response_wait_s`
- `warm_runner_response_poll_lag_s`
- `warm_runner_response_read_s`
- `warm_runner_response_write_s`
- `pool_stage_poller_join_s`

## Example Run: `live_20260425T181642Z_07c8bbe1`

### Top-Level Totals

From the live benchmark export:

```text
Backend wall:      79.093s
Pool wall:         77.384s
Runner wall:       55.703s
Transcribe call:   48.529s
```

Breakdowns:

```text
Backend:
  wav write        0.053s
  submit           2.108s
  result collect   0.917s
  non-pool         1.710s

Pool:
  ingest           1.722s
  queue            0.059s
  non-runner      21.681s

Runner:
  load_audio       7.064s
  non-transcribe   7.174s
```

Average per pool request (`176` requests):

```text
Backend wall             449.4ms/request
Pool wall                439.7ms/request
Pool non-runner          123.2ms/request
Runner wall              316.5ms/request
Transcribe call          275.7ms/request
Load audio                40.1ms/request
```

### Warm-Runner IPC Totals

Aggregated from the `176` terminal pool request records:

```text
warm_runner_request_read_s       0.013806s total   0.078ms/request
warm_runner_payload_write_s      0.016256s total   0.092ms/request
warm_runner_dispatch_s           0.003657s total   0.021ms/request
warm_runner_response_wait_s     60.443719s total 343.430ms/request
warm_runner_response_poll_lag_s  4.596892s total  26.119ms/request
warm_runner_response_read_s      0.025155s total   0.143ms/request
```

Important note:

- `warm_runner_response_wait_s` is not additive with `runner_wall_s`
- it already includes the runner execution time

So the useful comparison is:

```text
warm_runner_response_wait_s - runner_wall_s = 4.740s total
```

That is close to:

```text
warm_runner_response_poll_lag_s = 4.597s total
```

Conclusion: response polling in the warm-runner client is real, but it is not the main explanation for `pool_non_runner`.

## Validation Run: `live_20260425T184000Z_dd0999c8`

This second run was used to validate the two added timings:

- `warm_runner_response_write_s`
- `pool_stage_poller_join_s`

Top-level totals:

```text
Backend wall:      83.456s
Pool wall:         81.661s
Runner wall:       57.648s
Transcribe call:   50.235s
```

Breakdowns:

```text
Backend:
  wav write        0.052s
  submit           2.280s
  result collect   0.987s
  non-pool         1.794s

Pool:
  ingest           1.885s
  queue            0.068s
  non-runner      24.013s

Runner:
  load_audio       7.293s
  non-transcribe   7.412s
```

Warm-runner and pool orchestration totals from the `184` terminal pool requests:

```text
warm_runner_request_read_s       0.012004s total   0.065ms/request
warm_runner_payload_write_s      0.017811s total   0.097ms/request
warm_runner_dispatch_s           0.004615s total   0.025ms/request
warm_runner_response_read_s      0.044323s total   0.241ms/request
warm_runner_response_write_s     0.012801s total   0.070ms/request
warm_runner_response_poll_lag_s  4.848894s total  26.353ms/request
pool_stage_poller_join_s        16.757669s total  91.074ms/request
```

That leaves only a small residual:

```text
pool_non_runner
- pool_ingest
- pool_queue
- warm_runner_request_read
- warm_runner_payload_write
- warm_runner_dispatch
- warm_runner_response_read
- warm_runner_response_write
- warm_runner_response_poll_lag
- pool_stage_poller_join
= 0.362s total
```

Conclusion from the validation run:

- `pool_stage_poller_join_s` is the dominant part of `pool_non_runner`
- warm-runner response polling is the second clear pool-side cost
- response write itself is negligible
- file IPC read/write cost is negligible

## What Is Directly Measured vs Inferred

### Directly measured

Measured facts from this run:

- `backend_non_pool` is small in dev: `1.710s` total
- `pool_non_runner` is large: `21.681s` total
- `runner_non_transcribe` is significant: `7.174s` total
- `load_audio_s` is almost all of `runner_non_transcribe`
- warm-runner request payload write/read/dispatch/readback are tiny
- warm-runner response poll lag is meaningful: `4.597s` total

### Inferred

The main remaining inference is now architectural, not numerical:

- the large `pool_stage_poller_join_s` cost exists because:
  - stage poll interval defaults to `150ms` in [service.py](/home/gunnar/projects/asr-pool-dev/app/pool/service.py:125)
  - a per-request stage poll task is created in [service.py](/home/gunnar/projects/asr-pool-dev/app/pool/service.py:913)
  - after runner completion, `_runner_loop()` still does `await stage_task` in [service.py](/home/gunnar/projects/asr-pool-dev/app/pool/service.py:948)

That design explanation is still an inference, but the timing itself is now directly measured.

## Current Bottleneck Reading

On dev, the main latency buckets now look like this:

1. `pool_non_runner`
   - `24.013s` total in the validation run
   - currently the biggest bucket
   - now directly explained mainly by:
     - `pool_stage_poller_join_s = 16.758s`
     - `warm_runner_response_poll_lag_s = 4.849s`
     - `pool_ingest_s = 1.885s`
2. `load_audio_s`
   - `7.293s` total
   - about `39.6ms/request`
   - this is the main runner-side cost outside the pure transcribe call
3. `backend_non_pool`
   - `1.794s` total in dev
   - not the main local bottleneck

## Dev vs Prod Caveat

This note is based on dev, where backend and pool run on the same machine.

That means:

- `backend_non_pool` is probably understated versus prod
- prod also pays for remote backend <-> pool submit and artifact fetch

So:

- `load_audio_s` is still representative
- pool-internal orchestration costs are still representative
- backend transport costs must later be checked again on prod or prod-like topology

## Confirmed Findings

After the validation run:

1. `pool_stage_poller_join_s` is the main pool-side overhead.
2. `warm_runner_response_poll_lag_s` is the second pool-side overhead.
3. `warm_runner_response_write_s` is negligible.
4. file IPC read/write itself is negligible.
5. `load_audio_s` remains the main runner-side overhead outside the pure transcribe call.

Optional extra measurement only if needed later:

1. `pool_terminalize_s`
   - measure the terminalization block after the stage task has ended
   - purpose:
     confirm that record update + completion append stay small

## Proposed Optimization Order

1. Remove or redesign the stage-poller tail first.
   - this is now the largest confirmed pool-side cost

2. Then address warm-runner response polling.
   - this is the next confirmed pool-side cost

3. Then address `load_audio_s`:
   - add a fast path for known live PCM WAV input
   - avoid `whisperx.load_audio()` for this input shape

4. Only after that revisit warm-runner IPC transport more broadly:
   - file IPC itself is not free
   - but the measurements show it is not yet the main cost center

5. Re-validate backend transport on prod-like topology:
   - especially submit and artifact fetch

## Out of Scope for This Note

- no transport redesign yet
- no SSE transcript redesign yet

## Addendum 2026-04-26

This addendum records what happened after the main note above was written.

### Extra Metrics Added

Two additional timings were added in `asr-pool-dev`:

1. `warm_runner_response_write_s`
   - measured around response file materialization in
     [server.py](/home/gunnar/projects/asr-pool-dev/app/whisperx/server.py:545)
2. `pool_stage_poller_join_s`
   - measured from `stage_stop.set()` until `await stage_task` returns in
     [service.py](/home/gunnar/projects/asr-pool-dev/app/pool/service.py:948)

These were validated first with:

- session: `live_20260425T184000Z_dd0999c8`

That validation run confirmed:

- `pool_stage_poller_join_s` was the dominant part of `pool_non_runner`
- `warm_runner_response_poll_lag_s` was the second-largest pool-side cost
- `warm_runner_response_write_s` was negligible

### Fix Applied

The stage poller was then changed to stop interruptibly on `stop_event` instead of always sleeping out the full poll interval.

Implementation:

- [service.py](/home/gunnar/projects/asr-pool-dev/app/pool/service.py:837)

Conceptually:

- old behavior:
  - poll progress
  - `await asyncio.sleep(stage_poll_interval)`
  - if stop is requested during sleep, wait for sleep to finish first
- new behavior:
  - poll progress
  - `await stop_event.wait()` with timeout
  - if stop is requested, exit immediately

This keeps the same polling architecture, but removes the measured completion tail.

### Post-Fix Run

The first full live run after that patch was:

- session: `live_20260426T074124Z_04b5a3c8`
- benchmark export:
  [live_20260426T074124Z_04b5a3c8.final-quality.latest.json](/home/gunnar/projects/omniscripta/data/live/benchmark_exports/live_20260426T074124Z_04b5a3c8.final-quality.latest.json)

Totals from that run:

```text
Backend wall:      65.447s
Pool wall:         64.186s
Runner wall:       57.456s
Transcribe call:   50.198s
Pool non-runner:    6.730s
Backend non-pool:   1.260s
```

Aggregated pool-side breakdown from the `182` terminal pool requests:

```text
pool_ingest_s                 1.772407s total   9.739ms/request
pool_queue_wait_s             0.065170s total   0.358ms/request
warm_runner_response_poll_lag_s
                              4.450924s total  24.456ms/request
pool_stage_poller_join_s      0.006749s total   0.037ms/request
```

Small IPC timings remained negligible:

```text
warm_runner_request_read_s    0.013074s total
warm_runner_payload_write_s   0.019711s total
warm_runner_dispatch_s        0.004319s total
warm_runner_response_read_s   0.044295s total
warm_runner_response_write_s  0.012476s total
```

### Measured Impact

Comparing the validation run before the fix (`live_20260425T184000Z_dd0999c8`) to the first run after the fix (`live_20260426T074124Z_04b5a3c8`):

```text
pool_non_runner:        24.013s ->  6.730s   (-17.283s, -72.0%)
pool_wall:              81.661s -> 64.186s   (-17.475s, -21.4%)
pool_stage_poller_join: 16.758s ->  0.0067s  (-16.751s, ~-100%)
response_poll_lag:       4.849s ->  4.451s   (-0.398s,  -8.2%)
runner_wall:            57.648s -> 57.456s   (-0.192s,  -0.3%)
transcribe:             50.235s -> 50.198s   (-0.037s,  -0.1%)
```

Interpretation:

- almost the entire improvement came from removing the stage-poller tail
- runner work itself stayed essentially unchanged
- the next clear pool-side bottleneck is now `warm_runner_response_poll_lag_s`

### Updated Priority After the Fix

With the stage-poller tail removed, the next priorities are now:

1. `warm_runner_response_poll_lag_s`
   - remaining pool-side overhead
2. `load_audio_s`
   - remaining runner-side overhead outside pure transcribe
3. only after that, broader IPC redesign if still justified

## Addendum 2026-04-26 (Later)

This addendum records two more changes made after the stage-poller fix:

1. reducing warm-runner response polling from `50ms` to `25ms`
2. removing the live artifact `GET` path by switching to inline SRT in terminal completions

### Warm Runner Poll Interval: 50ms -> 25ms

After the stage-poller tail was removed, the next remaining pool-side cost was still
`warm_runner_response_poll_lag_s`.

One important detail: changing the fallback default in code was not enough on its own.
The effective runtime value stayed at `50ms` until
[`config/settings.json`](/home/gunnar/projects/asr-pool-dev/config/settings.json:49)
was updated.

Baseline run before the real config change:

- session: `live_20260426T082459Z_16a3d5d4`

Validation run after the real config change:

- session: `live_20260426T083630Z_b727b7fd`
- benchmark export:
  [live_20260426T083630Z_b727b7fd.final-quality.latest.json](/home/gunnar/projects/omniscripta/data/live/benchmark_exports/live_20260426T083630Z_b727b7fd.final-quality.latest.json)

Measured impact:

```text
pool_non_runner:              7.582s -> 4.743s
warm_runner_response_poll_lag 5.238s -> 2.406s
response_poll_lag/request:   28.5ms -> 13.1ms
```

Interpretation:

- the `25ms` setting did what it was expected to do
- remaining response poll lag was cut by roughly half
- this was a useful but smaller win than the stage-poller fix

### Remote dc1 -> dc2 Measurement and Inline SRT

To measure the real remote return path instead of the local dev path, the dev
Omniscripta backend on `dc1` was pointed at the pool on `dc2` via the existing
`127.0.0.1:8090 -> dc2:18090` SSH tunnel.

At the same time, two backend-side timings were added:

- `backend_artifact_get_s`
- `backend_srt_parse_s`

These were first measured in:

- session: `live_20260426T090855Z_42568c69`
- benchmark export:
  [live_20260426T090855Z_42568c69.final-quality.latest.json](/home/gunnar/projects/omniscripta/data/live/benchmark_exports/live_20260426T090855Z_42568c69.final-quality.latest.json)

That remote run showed:

```text
backend_result_collect_total_s  10.173s
backend_artifact_get_total_s    10.093s
backend_srt_parse_total_s        0.025s
```

Interpretation:

- the extra artifact `GET` was effectively the whole `result_collect` cost
- local SRT parsing was negligible

The live backend was then changed to request inline SRT in terminal completions
instead of doing a separate artifact download:

- `ASROutputSelection(..., srt_inline=True)` in
  [asr_bridge.py](/home/gunnar/projects/omniscripta/app/live/runtime/asr_bridge.py:479)
- terminal result handling now reads `response.result.srt_text` directly in
  [asr_bridge.py](/home/gunnar/projects/omniscripta/app/live/runtime/asr_bridge.py:577)

Validation run after that change:

- session: `live_20260426T092150Z_784e6a45`
- benchmark export:
  [live_20260426T092150Z_784e6a45.final-quality.latest.json](/home/gunnar/projects/omniscripta/data/live/benchmark_exports/live_20260426T092150Z_784e6a45.final-quality.latest.json)

Measured impact:

```text
backend_result_collect: 10.173s -> 0.021s
backend_artifact_get:   10.093s -> 0.000s
backend_srt_parse:       0.025s -> 0.012s
backend_wall:           68.781s -> 58.026s
backend_outside_pool:   14.043s -> 4.219s
```

Interpretation:

- the live artifact `GET` path was a real remote bottleneck
- switching to inline SRT removed that bottleneck almost completely
- this was a larger real-world win than the `50ms -> 25ms` poll reduction

### What Now Looks Like the Next Optimization Candidates

After the stage-poller fix, the `25ms` poll change, and inline SRT, the remaining
first suspects are now:

1. submit / POST transport cost
   - still large on the remote dc1 -> dc2 path
   - in `live_20260426T092150Z_784e6a45`:
     - `asr_backend_submit_total_s = 14.369s`
     - about `78.95ms/request`
2. `load_audio_s`
   - still the clearest runner-side non-transcribe cost
   - in the same run:
     - `load_audio_s = 6.560s` total
     - about `36ms/request`

These are the remaining candidates most likely to reduce:

```text
backend_wall - runner_transcribe
```

### UX Stop Line

There is also an important stopping condition now.

The production user experience is not only:

- backend -> pool -> runner

It also includes:

- browser -> backend live transport
- the Cloudflare tunnel on the browser-facing path
- backend -> browser return transport
- frontend apply / render cadence

So from this point onward, more backend/pool optimization is no longer guaranteed
to produce noticeable browser UX improvement.

That means:

- `submit / POST transport`
- `load_audio_s`

are the next technical suspects, but they should be weighed against end-to-end
browser-visible latency before further optimization work is prioritized.
