# Polling, Latency-Gevoeligheid en Verantwoordelijkheden (Worker vs ASR Pool)

Datum: 2026-03-10  
Scope: huidige backend (`portal-api`, `worker`, `asr-pool`) en richting voor herstructurering.

## 1. Overgebleven polling-keys (`config/settings.json`)

| Key | Huidige waarde | Waar gebruikt | Classificatie | Waarom |
|---|---:|---|---|---|
| `live_rolling_poll_ms` | `120` | `portal-api/live_engine_rolling_context.py` | `Latency-gevoelig (direct)` | Bepaalt hoe snel inflight live chunks opnieuw gepolld worden op klaar/nieuw resultaat. |
| `live_rolling_emit_min_ms` | `120` | `portal-api/live_engine_rolling_context.py` | `Latency-gevoelig (conditioneel)` | Throttle op nieuwe infer-enqueue momenten; relevant als dit limiterend is t.o.v. andere guardrails. |
| `asr_pool_watchdog_poll_ms` | `2000` | `asr-pool/main.py` | `Niet latency-gevoelig (normaal pad)` | Alleen health/recovery cadence; geen directe transcript-latency op gezonde requests. |
| `asr_pool_stage_poll_ms` | `150` | `asr-pool/main.py` | `Progress-latency` | Hoe snel stage-updates (`transcribing/aligning/...`) in pool-records komen. |
| `asr_pool_warm_response_poll_s` | `0.05` | `asr-pool/whisperx_runner_client.py` | `Latency-gevoelig (direct)` | Poll op runner response file; voegt kleine wachttijd toe op request/prewarm-response oppakken. |
| `asr_remote_pending_status_poll_s` | `1.0` | `worker/worker_daemon.py` (upload pad) | `Progress-latency` | Hoe snel worker stage/status uit pool ophaalt voor upload progress UX. |
| `asr_pool_records_prune_s` | `30` | `asr-pool/main.py` | `Niet latency-gevoelig` | Housekeeping van record map, geen directe impact op transcript snelheid. |
| `asr_blob_cleanup_s` | `120` | `shared/asr/blob_store.py` | `Niet latency-gevoelig` | Temp blob cleanup cadence, vooral disk/ops. |

## 2. Belangrijke nuance: `#4` en `#6` zitten in dezelfde keten

- `asr_pool_stage_poll_ms` (`#4`) bepaalt hoe snel stage in pool-records wordt geactualiseerd.
- `asr_remote_pending_status_poll_s` (`#6`) bepaalt hoe snel worker die stage ophaalt en toont.
- Voor stage/progress UX tellen ze samen.
- Voor terminal completion push naar worker zijn ze niet de primaire trigger.

## 3. Wat ligt op het echte "latency-gevoelige pad"?

Voor het gevoel van "snappy transcript updates" wegen vooral:

1. `live_rolling_poll_ms` (live inflight poll cadence)
2. `asr_pool_warm_response_poll_s` (runner response detectie)
3. `live_rolling_emit_min_ms` (alleen als limiterend in combinatie met overige guardrails)

Minder relevant voor transcript-latency, wel voor UX/ops:

1. `asr_pool_stage_poll_ms` en `asr_remote_pending_status_poll_s` (fase/progress tekst)
2. `asr_pool_watchdog_poll_ms`, `asr_pool_records_prune_s`, `asr_blob_cleanup_s` (stabiliteit/ops)

## 4. Gerelateerde cadansknoppen buiten `polling_intervals`

Dit zijn geen polling-keys, maar ze beïnvloeden latencygevoel vaak sterker dan meerdere poll-keys:

1. `live.rolling.min_infer_audio_ms`
2. `live.rolling.min_new_audio_ms`
3. `worker.live.max_outstanding_requests`
4. `worker_events.coordinator_tick_s`

## 5. Richting voor herstructurering verantwoordelijkheden

Doel: minder verborgen polling, duidelijkere eigenaarschap per component, betere schaalbaarheid per fase.

### 5.1 Worker

1. Worker orkestreert expliciet faseketen: `transcribe -> align -> diarize -> finalize`.
2. Worker bepaalt statusmessages per fase (in plaats van afgeleide stage-inferentie uit pool).
3. Worker beheert flow control per fase (`max_outstanding` per fase in toekomst).

### 5.2 ASR Pool

1. ASR pool wordt executor + scheduler (slots, queue, health, cancellation).
2. `profile` wordt uit ASR pool gehaald; worker stuurt expliciete request-params per fase.
3. Pool publiceert events, geen business-beslissingen over profielkeuze.

### 5.3 Eventing i.p.v. stage-polls

1. Stage-overgangen als push-event aanbieden (SSE/event stream) i.p.v. afhankelijkheid op `#4 + #6` keten.
2. Worker consumeert stage/completion events en publiceert eenduidige UX-status naar frontend.
3. Dit maakt progress-latency voorspelbaar en verlaagt polling-complexiteit.

## 6. Consequentie voor toekomstige e2e-optimalisatie

Prioriteit voor voelbare latency-winst:

1. Eerst: echte latency-pad tuning (`live_rolling_poll_ms`, `asr_pool_warm_response_poll_s`, plus non-poll guardrails).
2. Daarna: progress-pad opschonen (`#4/#6` vervangen door push-eventing).
3. Parallel: verantwoordelijkheidsscheiding (worker orchestration, pool execution, profiel uit pool).

