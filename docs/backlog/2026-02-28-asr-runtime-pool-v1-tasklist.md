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

- [x] Nieuwe service directory + process entrypoint toegevoegd (`asr-pool/main.py`).
- [x] Config laden voor:
  - `runner_slots`
  - queue limits (`8/20/50`)
  - request timeouts (`30/120/300`)
  - token auth secret
- [x] Endpoints implementeren:
  - `POST /asr/v1/requests`
  - `GET /asr/v1/requests/{request_id}`
  - `POST /asr/v1/requests/{request_id}/cancel`
  - `GET /asr/v1/pool`
- [x] Lifecycle state machine implementeren:
  - `queued|running|cancel_requested|cancelled|completed|failed`
- [x] Idempotency op `request_id` implementeren.
- [x] 409 bij payload-conflict op bestaand `request_id`.
- [x] 429 met code `ASR_QUEUE_FULL` bij queue limit overschrijding.

### Component: Scheduler

- [x] Priority queues implementeren (`interactive`, `normal`, `background`).
- [x] Final-first policy binnen `interactive` implementeren op basis van metadata:
  - `context.live_lane=final|speculative`.
- [x] Fairness/anti-starvation regel toevoegen voor `normal/background`.
- [x] Queue metrics bijhouden per priority.

### Component: Runner slot manager

- [x] Warm runner process wrapper bouwen per slot.
- [x] Slot lease/claim/release mechanisme toevoegen.
- [x] Watchdog toevoegen:
  - process liveness
  - auto-restart bij crash
- [x] Soft-cancel gedrag afdwingen:
  - `running` request niet hard killen
  - resultaat negeren bij `cancel_requested`.

### Component: Worker integratie

- [x] Nieuwe remote ASR client module toevoegen.
- [x] Live chunk ASR transport in worker op remote-only gezet.
- [x] `priority` en `context.live_lane` doorzetten in request.
- [x] `X-ASR-Token` header meesturen.

### Component: Security/ops (v1)

- [x] Service intern binden (loopback of intern segment).
- [x] Shared token validatie op alle ASR endpoints.
- [x] Timeouts + keepalive settings voor worker -> gateway client instellen.
- [x] Basis service unit toevoegen (`transcribe-asr-pool-dev.service` of equivalent).

### Component: Observability

- [x] Metrics/diagnostics endpoint payload voor `/asr/v1/pool` vullen met:
  - `slots_total|busy|available`
  - `queue_limits`
  - `queue_depth`
  - `request_timeouts_s`
  - `queue_wait_ms_p95`
  - `scheduling_policy.interactive_final_first`
- [x] Structured logs per request lifecycle transition toevoegen.

### Sprint 1 exit criteria

- [x] 1 live stream werkt end-to-end in `remote` mode.
- [x] Queue limit, timeout en final-first policy aantoonbaar actief.
- [x] Geen hard crashes bij runner restart scenario.

## Sprint 2+ (remote-ready en schaalvalidatie)

### Component: Audio transport (`blob_ref`)

- [x] Blob uploader in worker toevoegen voor chunk audio.
- [x] `audio.blob_ref` pad in ASR pool implementeren.
- [x] Temporary supportpad voor `audio.local_path` gecontroleerd behouden.
- [x] Cleanup policy voor tijdelijke blobs vastleggen.

### Component: Load en SLO validatie

- [x] Loadtest scripts maken voor 1/2/3+ simultane live streams.
- [x] Meten:
  - `interactive` wait/latency p50/p95/p99
  - final-vs-speculative wachttijd
  - error/retry rates
  - blob fetch latency p95
- [ ] Validatie tegen targets:
  - `blob_fetch_p95 < 100ms`
  - geen starvation van final chunks

### Component: Hardening

- [ ] Recovery bij service restart documenteren en testen.
- [x] Retentiebeleid voor in-memory request history + evt. persistent store beslissen (v1: in-memory only + TTL/max-record pruning, geen persistent store).
- [ ] Tokenrotatiepad voor `X-ASR-Token` toevoegen.

## Open vragen voor implementatie start

- [x] Exacte modulelocatie voor nieuwe ASR pool service in repo (`asr-pool/main.py`).
- [x] Keuze dev poort voor pool service (`8090`).
- [x] Keuze blob backend voor dev (`filesystem abstraction` via `fs://` blob_ref op gedeelde root).
- [x] Retry policy worker -> gateway (attempts/backoff/jitter; retry op netwerkfouten + HTTP 429/5xx met bounded exponential backoff).
