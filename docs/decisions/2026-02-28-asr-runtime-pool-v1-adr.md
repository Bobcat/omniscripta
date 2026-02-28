# Decision Note: ASR Runtime Pool v1 (Worker-Mediated Gateway)

Date: 2026-02-28  
Status: Accepted (phase-0 baseline)

## Context

De huidige live ASR flow gebruikt process-local warm runners per worker. Dit werkt voor single-worker, maar schaalt beperkt:

- GPU-resources worden niet centraal beheerd
- meerdere gelijktijdige live streams concurreren onvoorspelbaar
- remote ASR executie (andere machine) wordt bemoeilijkt door `audio.local_path` afhankelijkheid

Het doel is om naar een centrale ASR runtime pool te gaan met 1 logisch ingangspunt en centrale scheduling.

## Decision

### 1) Caller-pad v1

Voor v1 blijft de bestaande job-orchestratie behouden:

- `Portal API` enqueued filesystem jobs
- `Worker` claimt jobs
- `Worker` stuurt ASR requests naar de centrale ASR gateway (remote mode), met local fallback via feature flag

Dit voorkomt een grote architecturele breuk in de eerste migratiestap.

### 2) Transport en API

- v1 gebruikt HTTP/JSON RPC met OpenAPI contract
- Contractinhoud blijft `asr_v1` (geen tweede inhoudelijk ASR contract)
- OpenAPI definieert lifecycle endpoints:
  - `POST /asr/v1/requests`
  - `GET /asr/v1/requests/{request_id}`
  - `POST /asr/v1/requests/{request_id}/cancel`
  - `GET /asr/v1/pool`

### 3) Runner slots

- v1 runner slots zijn persistente warm runners
- Start met `runner_slots=2`
- Opschalen naar `3` alleen na load- en VRAM-validatie

### 4) Scheduling policy

- Prioriteit: `interactive > normal > background`
- Fairness vereist (anti-starvation)
- Optionele capaciteitspartitie:
  - `interactive_reserved_slots=1`
  - overige slots shared
- Queue limieten (v1 defaults):
  - `interactive_max_queue_depth=8`
  - `normal_max_queue_depth=20`
  - `background_max_queue_depth=50`

Speculative lane nuance in v1:

- Geen vierde priority-level in v1.
- Zowel final als speculative live requests gebruiken `interactive`.
- Binnen `interactive` geldt final-first scheduling:
  - final chunks mogen niet geblokkeerd worden door speculative bursts.
  - onderscheid via request metadata/context (`live_lane=final|speculative`).

### 5) Idempotency en cancel

- `request_id` is idempotency key
- Zelfde `request_id` + zelfde payload: bestaand request/resultaat teruggeven
- Zelfde `request_id` + andere payload: `409` conflict
- Cancel voor `running` is v1 soft-cancel:
  - markeer `cancel_requested`
  - runner laat huidige inference aflopen
  - resultaat wordt niet meer als actief resultaat geconsumeerd

### 6) Audio source evolutie

- `audio.local_path` blijft tijdelijk ondersteund voor same-host migratie
- `audio.blob_ref` wordt verplicht pad voor remote ASR executie

### 7) Security v1

- Service alleen intern bereikbaar (loopback of intern netwerksegment)
- Shared secret header (`X-ASR-Token`) tussen worker en gateway
- mTLS is expliciet v2-eis voor multi-node

### 8) State recovery

Minimaal vereist in v1:

- execution timeout per priority class:
  - `interactive=30s`
  - `normal=120s`
  - `background=300s`
- runner watchdog + auto-restart bij crash
- stuck running request detectie en cleanup pad

## Alternatives Considered

### A) Direct `Portal API -> ASR` in v1

Niet gekozen voor v1; te grote verandering tegelijk (queue + orchestration + ASR transport).

### B) Drie losse ASR services laten kiezen door workers

Niet gekozen; scheduling/policy hoort centraal, niet in workers.

### C) gRPC-first in v1

Niet gekozen; hogere adoptiecomplexiteit. HTTP/JSON is sneller valideerbaar in huidige stack.

### D) Hard kill op cancel-running

Niet gekozen; verhoogt kans op runner-instabiliteit en verlies van warm state.

### E) On-demand subprocess slots

Niet gekozen voor interactive v1; cold-start latency te hoog.

## Consequences

### Positief

- Incrementele migratie met rollback via feature flag
- Centrale resource- en schedulingcontrole
- Heldere route naar remote ASR nodes

### Negatief / trade-offs

- Worker blijft in v1 nog tussenlaag
- Tijdelijke dual-path complexiteit (`local` + `remote`)
- In-memory queue in pool vereist expliciete crash/recovery afspraken

## Rollout Plan

1. Fase 0: ADR + OpenAPI afronden en reviewen
2. Fase 1/2: pool skeleton + scheduler + recovery
3. Fase 3: worker remote client onder feature flag
4. Fase 4: `blob_ref` audio transport
5. Fase 5: load-validatie en SLO check

## Rollback Plan

- Zet `TRANSCRIBE_ASR_TRANSPORT=local`
- Worker gebruikt bestaande local backend pad
- Centrale pool kan uitgezet worden zonder breuk in live flow

## Links

- Backlog plan: `docs/backlog/2026-02-28-asr-runtime-pool-v1-implementation-plan.md`
- ASR contract: `docs/contracts/asr_service_v1.md`
- HTTP binding (OpenAPI): `docs/contracts/asr_runtime_pool_v1.openapi.yaml`
