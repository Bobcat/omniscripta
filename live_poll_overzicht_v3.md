# Live Poll Overzicht (v3)

Datum: 2026-03-08
Scope: rolling live pad (browser -> portal-api -> worker -> asr-pool)

## 1. Upstream polls (browser naar portal-api)

| # | Component | Poll target | Trigger | Cadans | Opmerking |
|---|---|---|---|---|---|
| 1 | `LiveView.startResultPolling()` | `GET /api/demo/live/sessions/{session_id}/result` | start/pause/resume/stop/ended/error flows | **250 ms** (`setInterval`) | `intervalMs` argument wordt nu niet gebruikt; cadence is hardcoded 250 ms. |

## 2. Downstream polls (portal-api/worker naar backend lagen)

| # | Component | Poll target | Trigger | Cadans | Opmerking |
|---|---|---|---|---|---|
| 1 | `live_engine_rolling_context._poll_inference()` | Job status via `chunk_bridge.poll_job(job_id)` | while websocket sessie actief is en inflight jobs bestaan | `polling_intervals.live_rolling_poll_ms` (nu **120 ms**) | Pollt job status uit `inbox/running/done/error` job dirs. |
| 2 | `live_engine_rolling_context` drain loop | `_drain_inflight_only(force_poll=True)` | bij `stop` en in `finally` | sleep `min(0.1, poll_interval_s)` + timeout `live.drain_wait_s` (nu **20s**) | Force-poll tot inflight leeg of timeout. |
| 3 | `worker._run_live_worker_submit_reap()` | `GET /asr/v1/completions?consumer_id=...&since_seq=...` | elke worker hoofdloop iteratie | `polling_intervals.asr_remote_completions_poll_s` (nu **0.2s**) | Live-loop reap’t snel zowel bij inflight als bij nieuwe queue-items. |
| 4 | `worker._run_live_worker_submit_reap()` | `claim_next_job(job_kind_filter="live_chunk")` | elke worker hoofdloop iteratie | per loop | Filesystem queue claim-poll op `inbox`. |
| 5 | `asr_pool._poll_stage_updates()` | progress json op disk | per running request | `polling_intervals.asr_pool_stage_poll_ms` (nu **150 ms**) | Alleen interne stage/progress polling in pool. |
| 6 | `asr_pool._watchdog_loop()` | warm runner health check | continu als watchdog enabled | `polling_intervals.asr_pool_watchdog_poll_ms` (nu **2000 ms**) | Interne health poll + recovery. |

## 3. Completion feed polling contract

- Worker pollt `GET /asr/v1/completions` met:
  - `consumer_id` (live: `worker-live@<instance>`)
  - `since_seq`
  - `limit`
- Pool antwoordt met:
  - `events[]`
  - `next_seq`
- Feed is in-memory (v3), zonder durable cursor recovery na pool restart.

## 4. Effectieve dev-config waarden (huidige omgeving)

- `polling_intervals.live_rolling_poll_ms = 120`
- `polling_intervals.live_rolling_emit_min_ms = 120`
- `live.drain_wait_s = 20.0`
- `worker.live.max_outstanding_requests = 1`
- `polling_intervals.asr_remote_completions_poll_s = 0.2` (wel als sleep cadence gebruikt bij pending live requests)
- `polling_intervals.asr_pool_stage_poll_ms = 150`
- `polling_intervals.asr_pool_watchdog_poll_ms = 2000`
- `asr_pool.watchdog_recover_timeout_s = 60`
- `asr_pool.completions.max_events = default (20000)`

## 5. Niet-polling maar wel cadence-relevant

- Live audio/control transport is websocket push/pull (geen HTTP polling).
- Server verstuurt `stats` events periodiek op frame-count voorwaarden (eerste frame en daarna per 50 frames).
