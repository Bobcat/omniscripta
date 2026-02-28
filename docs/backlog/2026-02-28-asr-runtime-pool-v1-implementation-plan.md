# Implementatieplan: ASR Runtime Pool v1

Status: Draft voor review (nog niet geïmplementeerd)

## Fase 0 artifacts

- ADR:
  - `docs/decisions/2026-02-28-asr-runtime-pool-v1-adr.md`
- OpenAPI binding:
  - `docs/contracts/asr_runtime_pool_v1.openapi.yaml`
- Basis contract (transport-agnostisch):
  - `docs/contracts/asr_service_v1.md`
- Uitvoeringschecklist:
  - `docs/backlog/2026-02-28-asr-runtime-pool-v1-tasklist.md`

## Doel

Schaalbare live transcriptie mogelijk maken met meerdere gelijktijdige streams, zonder dat elke pipeline-worker zelf een WhisperX subprocess en GPU-allocatie beheert.

Concreet:

- 1 logisch ASR-ingangspunt voor workers/API
- centrale queue/scheduler met duidelijke prioriteiten
- runner-slots centraal beheerd (in plaats van process-local warm runners)
- voorbereid op ASR executie op andere machine(s)

## Huidige situatie (samengevat)

- `LiveSessionManager` heeft een harde sessielimiet via `TRANSCRIBE_LIVE_MAX_SESSIONS` (default `1`).
- `live_chunk` gebruikt een process-local warm runner in de worker.
- Queue-claiming in worker is sequentieel.
- ASR contract (`asr_v1`) bestaat al, maar audio input is praktisch nog `local_path`-gebaseerd.

Gevolg:

- beperkte schaalbaarheid
- GPU-beheer niet centraal
- remote ASR-machine lastig door filesystem-koppeling

## Architectuurdoel v1

### Logisch model

- Worker/API -> `ASR Gateway` (1 endpoint) -> `ASR Runtime Pool` (N runner slots)
- Workers kiezen geen specifieke runner/node.
- Scheduler in de pool bepaalt slottoewijzing.

### Beslissing: wie stuurt ASR requests

Voor v1 blijft de huidige keten bewust bestaan:

- `Portal API` enqueued jobs in de bestaande filesystem queue.
- `Worker daemon` blijft de actor die `live_chunk` jobs claimt.
- Worker stuurt vervolgens de ASR request naar de centrale ASR gateway (remote mode), of naar local backend (fallback mode).

Rationale:

- kleinste migratiestap zonder directe breuk in huidige job-orchestratie
- rollback blijft eenvoudig via feature flag
- direct `Portal API -> ASR` kan later als v2-vereenvoudiging worden uitgewerkt

### Prioriteiten

- `interactive`: live UX-kritisch
- `normal`: standaard
- `background`: batch/laagste prioriteit

### Queue limieten (v1 defaults)

- `interactive_max_queue_depth=8`
- `normal_max_queue_depth=20`
- `background_max_queue_depth=50`
- Overschrijding geeft `429` met expliciete backpressure-foutcode.

### Capaciteitspolicy (v1)

- Start conservatief met `runner_slots=2`.
- Na meting optioneel naar `runner_slots=3`.
- Optioneel partitioneren:
  - `interactive_reserved_slots=1`
  - overige slots shared.

### Timeout defaults (v1)

- `interactive_request_timeout_s=30`
- `normal_request_timeout_s=120`
- `background_request_timeout_s=300`

### Runner slot type (v1)

- Runner slots zijn in v1 **persistente warm runners** (geen on-demand cold start voor interactive pad).
- Voor v1 geen mixed backend per priority binnen dezelfde pool; dat kan later als uitbreiding.
- Bij VRAM druk geldt: slots omlaag in config, niet dynamisch oversubscriben.

## Contractkeuze

### Contractbron

- `asr_v1` blijft de canonieke request/response-contractlaag.
- Geen tweede contractvariant introduceren.

### API-vorm

- HTTP/JSON RPC met OpenAPI (v1).
- Later eventueel gRPC intern, maar niet in eerste migratiestap.

### Vereiste audio-aanpassing

- `audio.local_path` is niet voldoende voor remote executie.
- `audio.blob_ref` moet first-class worden (of equivalent object storage referentie).

## Scope v1

- Nieuwe ASR pool service met queue + scheduler + runner slots.
- Worker remote client met feature flag.
- Endpoints:
  - `POST /asr/v1/requests`
  - `GET /asr/v1/requests/{request_id}`
  - `POST /asr/v1/requests/{request_id}/cancel`
  - `GET /asr/v1/pool`
- Idempotency op `request_id`.
- Backpressure en duidelijke retryable errors.

### Idempotency en cancel semantiek (v1)

- Dubbele submit met hetzelfde `request_id`:
  - zelfde payload -> bestaand request/resultaat teruggeven (idempotent gedrag)
  - conflicterende payload -> `409` met expliciete foutcode
- Cancel op queued request:
  - status wordt `cancelled` en request wordt niet uitgevoerd
- Cancel op running request:
  - v1: markeer als `cancel_requested`; runner draait af
  - resultaat wordt niet meer doorgezet als actief resultaat (soft-cancel)
- Cancel na completion:
  - no-op met succesvolle response
- Client timeout/disconnect:
  - request blijft doorlopen tenzij expliciet gecanceld

### Speculative lane scheduling (v1)

- Speculative en final live chunks blijven beide onder `interactive`.
- Binnen `interactive` geldt extra ordering policy:
  - final chunks krijgen voorrang op speculative chunks.
- In v1 geen vierde priority-level; onderscheid via request metadata/context.

## Non-goals v1

- Geen volledige multi-region / geo-routing.
- Geen complexe user-tier billing/quotas in dezelfde scheduler.
- Geen harde dependency op gRPC in v1.
- Geen grote herbouw van live transcript UI-flow.

## Fasering

### Fase 0 — Design freeze (ADR + OpenAPI)

Deliverables:

- ADR met:
  - context en probleemstelling
  - keuze voor centrale pool en contract
  - alternatieven en trade-offs
  - rollout + rollback
- OpenAPI spec voor bovengenoemde endpoints.
- Besluit vastleggen voor:
  - callerpad (`Worker -> ASR Gateway`, niet direct `Portal API -> ASR`)
  - warm runner slot type
  - idempotency/cancel semantiek

Definition of Done:

- ADR geaccepteerd.
- OpenAPI contract door team gereviewd.

### Fase 1 — ASR pool service skeleton

Deliverables:

- Nieuwe service met health + pool status endpoint.
- Persistente runner-slot manager (nog zonder geavanceerde fairness).
- Basis queue model en request lifecycle state.
- Minimale security voor v1:
  - bind op intern/loopback adres
  - shared secret header tussen worker en gateway

Definition of Done:

- Service start/stopt betrouwbaar.
- Requests kunnen geaccepteerd en afgewerkt worden met 1 slot.

### Fase 2 — Scheduler en capaciteitspolicy

Deliverables:

- Priority scheduling (`interactive > normal > background`).
- Fairness-bescherming (voorkom starvation).
- Configurabele slotcount en optionele `interactive` reservatie.
- Backpressure gedrag (`429` of expliciete queue rejection met retry hints).
- State recovery basis:
  - request execution timeout per priority (`30s/120s/300s` defaults)
  - runner watchdog + auto-restart bij crash
  - stuck running requests detecteren en afhandelen
- Queue depth limits per priority afdwingen (`8/20/50` defaults).
- Binnen `interactive`: final-first policy t.o.v. speculative requests.

Definition of Done:

- Onder load behoudt `interactive` lage wachttijd.
- `background` wordt niet oneindig uitgehongerd.
- Crash/herstartgedrag is aantoonbaar voor queue + runner slots.

### Fase 3 — Worker integratie met feature flag

Deliverables:

- Nieuwe remote ASR client in worker.
- Feature flag:
  - `TRANSCRIBE_ASR_TRANSPORT=local|remote`
- Fallback pad naar bestaande local mode.

Definition of Done:

- Live chunk pipeline draait end-to-end via remote mode.
- Rollback naar local mode werkt zonder codewijziging.

### Fase 4 — Remote audio transport (`blob_ref`)

Deliverables:

- Blob/object storage pad voor audio chunks.
- Worker uploadt audio en stuurt `blob_ref`.
- ASR pool leest blob en verwerkt zonder shared filesystem.

Definition of Done:

- ASR service kan op aparte machine draaien.
- Geen functionele afhankelijkheid meer op `audio.local_path`.

Opmerking planning:

- Fase 3 en Fase 4 mogen parallel lopen zolang contractcompatibiliteit behouden blijft.

### Fase 5 — Observability en load-validatie

Deliverables:

- Metrics:
  - queue depth per priority
  - queue wait p50/p95/p99
  - runner utilization
  - error/retry rates
  - blob fetch latency p50/p95/p99
- Loadtests met meerdere gelijktijdige live streams.

Definition of Done:

- Meetrapport beschikbaar.
- SLO/latency en stabiliteit binnen afgesproken marges.
- Blob fetch latency voldoet aan target (initieel: `blob_fetch_p95 < 100ms` voor live chunks).

## Reviewchecklist (voor externe review)

- Is `asr_v1` contract voldoende stabiel en compleet voor remote uitvoering?
- Zijn idempotency, cancel en retry-semantiek eenduidig?
- Is backpressure expliciet en veilig (geen onbeperkte queuegroei)?
- Zijn queue depth limieten per priority expliciet en operationeel logisch?
- Is starvation voorkomen bij mixed workload?
- Zijn failure modes afgedekt:
  - runner crash
  - timeout
  - blob unavailable
  - node outage
- Is security/transport duidelijk:
  - interne endpoint only
  - auth tussen worker en ASR gateway
  - TLS/mTLS beleid
- Is het callerpad expliciet en haalbaar zonder queue-architectuurbreuk?
- Is `cancel_running` bewust soft-cancel (niet hard kill) en operationeel acceptabel?
- Is rollback simpel genoeg (`remote` -> `local`)?
- Is de migratie incrementeel uitvoerbaar zonder lange freeze?

## Risico's en mitigatie

- VRAM pieken hoger dan verwacht:
  - start met 2 slots, pas later opschalen.
- Queue regressie bij live UX:
  - priority + fairness + loadtests als gate.
- Speculative burst verdringt final chunks:
  - final-first policy binnen `interactive` + monitor op wait-time.
- Remote storage latency:
  - kleine blobs, korte paden, retries met bounded backoff.
- Extra operationele complexiteit:
  - eerst single-host pool, daarna pas multi-node.
- In-memory queueverlies bij process crash:
  - acceptabel in v1 voor live pad, maar documenteren en monitoren.

## Acceptatiecriteria v1

- Meerdere gelijktijdige live streams zonder crash of systemische backlog.
- Centrale ASR capaciteit is observeerbaar en begrensd.
- `interactive` latency aantoonbaar beter dan `normal/background`.
- Worker kan via feature flag veilig terug naar local mode.
- Architectuur klaar voor volgende stap: multi-node ASR pools.

## Securitypad (v1 -> v2)

- v1:
  - interne binding + shared secret header
- v2 (multi-node):
  - mTLS tussen clients/gateway/nodes
  - service identity en policy-based routing

## Volgende stap na review

- Reviewfeedback verwerken.
- Fase 0 tasks aanmaken in issue/backlog tracker.
- Daarna implementatie starten vanaf Fase 1.
