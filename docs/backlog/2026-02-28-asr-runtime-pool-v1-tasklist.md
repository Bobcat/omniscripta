# ASR Runtime Pool v1 - Tasklist per component

Status: Ready-to-implement checklist

## Gebruik

- Deze lijst is een uitvoeringsvertaling van:
  - `docs/backlog/2026-02-28-asr-runtime-pool-v1-implementation-plan.md`
  - `docs/decisions/2026-02-28-asr-runtime-pool-v1-adr.md`
  - `docs/contracts/asr_runtime_pool_v1.openapi.yaml`
- Volgorde: eerst "Sprint 1", daarna "Sprint 2+".

## Sprint 1 (minimum pad naar werkende pool)

### Component: ASR gateway/pool service

- [ ] Nieuwe service directory + process entrypoint toevoegen (`asr_pool_main.py` of equivalent).
- [ ] Config laden voor:
  - `runner_slots`
  - queue limits (`8/20/50`)
  - request timeouts (`30/120/300`)
  - token auth secret
- [ ] Endpoints implementeren:
  - `POST /asr/v1/requests`
  - `GET /asr/v1/requests/{request_id}`
  - `POST /asr/v1/requests/{request_id}/cancel`
  - `GET /asr/v1/pool`
- [ ] Lifecycle state machine implementeren:
  - `queued|running|cancel_requested|cancelled|completed|failed`
- [ ] Idempotency op `request_id` implementeren.
- [ ] 409 bij payload-conflict op bestaand `request_id`.
- [ ] 429 met code `ASR_QUEUE_FULL` bij queue limit overschrijding.

### Component: Scheduler

- [ ] Priority queues implementeren (`interactive`, `normal`, `background`).
- [ ] Final-first policy binnen `interactive` implementeren op basis van metadata:
  - `context.live_lane=final|speculative`.
- [ ] Fairness/anti-starvation regel toevoegen voor `normal/background`.
- [ ] Queue metrics bijhouden per priority.

### Component: Runner slot manager

- [ ] Warm runner process wrapper bouwen per slot.
- [ ] Slot lease/claim/release mechanisme toevoegen.
- [ ] Watchdog toevoegen:
  - process liveness
  - auto-restart bij crash
- [ ] Soft-cancel gedrag afdwingen:
  - `running` request niet hard killen
  - resultaat negeren bij `cancel_requested`.

### Component: Worker integratie

- [ ] Nieuwe remote ASR client module toevoegen.
- [ ] Feature flag `TRANSCRIBE_ASR_TRANSPORT=local|remote` verwerken in backendselectie.
- [ ] `local` fallback behouden zonder regressie.
- [ ] `priority` en `context.live_lane` doorzetten in request.
- [ ] `X-ASR-Token` header meesturen.

### Component: Security/ops (v1)

- [ ] Service intern binden (loopback of intern segment).
- [ ] Shared token validatie op alle ASR endpoints.
- [ ] Timeouts + keepalive settings voor worker -> gateway client instellen.
- [ ] Basis service unit toevoegen (`transcribe-asr-pool-dev.service` of equivalent).

### Component: Observability

- [ ] Metrics/diagnostics endpoint payload voor `/asr/v1/pool` vullen met:
  - `slots_total|busy|available`
  - `queue_limits`
  - `queue_depth`
  - `request_timeouts_s`
  - `queue_wait_ms_p95`
  - `scheduling_policy.interactive_final_first`
- [ ] Structured logs per request lifecycle transition toevoegen.

### Sprint 1 exit criteria

- [ ] 1 live stream werkt end-to-end in `remote` mode.
- [ ] `local` rollback werkt via feature flag.
- [ ] Queue limit, timeout en final-first policy aantoonbaar actief.
- [ ] Geen hard crashes bij runner restart scenario.

## Sprint 2+ (remote-ready en schaalvalidatie)

### Component: Audio transport (`blob_ref`)

- [ ] Blob uploader in worker toevoegen voor chunk audio.
- [ ] `audio.blob_ref` pad in ASR pool implementeren.
- [ ] Temporary supportpad voor `audio.local_path` gecontroleerd behouden.
- [ ] Cleanup policy voor tijdelijke blobs vastleggen.

### Component: Load en SLO validatie

- [ ] Loadtest scripts maken voor 1/2/3+ simultane live streams.
- [ ] Meten:
  - `interactive` wait/latency p50/p95/p99
  - final-vs-speculative wachttijd
  - error/retry rates
  - blob fetch latency p95
- [ ] Validatie tegen targets:
  - `blob_fetch_p95 < 100ms`
  - geen starvation van final chunks

### Component: Hardening

- [ ] Recovery bij service restart documenteren en testen.
- [ ] Retentiebeleid voor in-memory request history + evt. persistent store beslissen.
- [ ] Tokenrotatiepad voor `X-ASR-Token` toevoegen.

## Open vragen voor implementatie start

- [ ] Exacte modulelocatie voor nieuwe ASR pool service in repo.
- [ ] Keuze dev poort voor pool service.
- [ ] Keuze blob backend voor dev (`minio`/filesystem abstraction/etc.).
- [ ] Retry policy worker -> gateway (attempts/backoff/jitter).
