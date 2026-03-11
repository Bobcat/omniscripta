# Worker Event-Driven Submit+Reap Tasklist

Datum: 2026-03-10  
Scope: `worker`, `asr-pool` (geen frontend, geen portal-api live-engine wijziging in deze scope)

## Voortgang
- Klaar: Fase 1 / Taak 1.1, 1.2, 1.3
- Klaar: Fase 2 / Taak 2.1, 2.2, 2.3
- Klaar: Fase 3 / Taak 3.1, 3.2
- Klaar (code): Fase 4 / Taak 4.1, 4.2
- Klaar (code): Fase 5 / Taak 5.1, 5.2, 5.3
- Klaar (code + runtime smoke): Fase 6 / Taak 6.1, 6.2
- Klaar (runtime upload): Fase 7 / Taak 7.1 (upload normaal + restart tijdens in-flight + nieuwe job na restart)
- Klaar (runtime live): Fase 7 / Taak 7.1 (live normaal + restart tijdens in-flight + nieuwe job na restart)
- Klaar (runtime meting): Fase 7 / Taak 7.2 (latency validatie live+upload)
- Klaar (stabilisatie): feed-reset cursor/reconnect bugfix in `asr-pool` + `worker` live path

## Doel
- Worker volledig event-driven maken zonder poll-cadence op:
- `inbox` claim
- `asr/v1/completions`
- v3-invariant behouden: worker blokkeert niet op individuele ASR-requests.
- Ontwerp future-proof maken voor `max_outstanding > 1` (live en upload).

## Niet-doel
- Geen recovery van in-flight requests na asr-pool restart (blijft expliciet buiten scope).
- Geen wijziging aan transcript merge/dedup logica.
- Geen aanpassing aan frontend of websocket protocol.

## Fase 1 - ASR Pool SSE Completion Stream

### Taak 1.1 - SSE endpoint toevoegen
- Bestand: `asr-pool/main.py`
- Voeg endpoint toe: `GET /asr/v1/completions/stream`
- Query params:
- `consumer_id` (required)
- `since_seq` (default 0)
- `heartbeat_s` (default uit settings; bounded)
- Event types:
- `meta` (feed_id, consumer_id, since_seq, next_seq)
- `completion` (zelfde event body als huidige `/asr/v1/completions`)
- `heartbeat` (periodiek, keepalive)

Acceptatiecriteria:
- Met open stream en zonder completions komt heartbeat periodiek binnen.
- Bij nieuwe completion verschijnt direct een `completion` SSE event zonder client polling.
- `consumer_id` ontbreekt geeft 400 met duidelijke foutcode.

### Taak 1.2 - Condition-driven wakeup gebruiken
- Hergebruik bestaande `asyncio.Condition` in pool-service (geen busy loop).
- Stream wacht op nieuwe completion events of heartbeat timeout.

Acceptatiecriteria:
- Geen extra poll-loop in asr-pool voor dit endpoint.
- CPU blijft stabiel idle met actieve SSE clients.

### Taak 1.3 - Feed reset contract op stream
- `feed_id` blijft onderdeel van stream meta.
- Na pool restart krijgt client nieuwe `feed_id`.

Acceptatiecriteria:
- Worker kan restart detecteren op basis van `feed_id` wissel.
- Bestaand v3-gedrag blijft mogelijk: pending failen, verder met nieuwe jobs.

## Fase 2 - Worker Event Infrastructure

### Taak 2.1 - Eventmodel introduceren
- Bestand: `worker/worker_daemon.py` (of nieuw module `worker/event_loop.py`)
- Definieer eventtypes:
- `INBOX_DIRTY`
- `COMPLETION_EVENT`
- `FEED_RESET`
- `SUBMIT_RESULT`
- `TICK`
- `SHUTDOWN`

Acceptatiecriteria:
- Centrale queue (`queue.Queue`) is enige ingang voor state mutaties.
- Geen directe state-mutatie vanuit helper threads.

### Taak 2.2 - Centrale coordinator loop
- Implementeer 1 centrale loop die:
- events consumeert
- pending state bijwerkt
- scheduler/refill draait: `while pending < max_outstanding: claim+submit`

Acceptatiecriteria:
- Geen codepad dat op individueel request wacht.
- Zelfde loop werkt correct voor `max_outstanding=1` en `>1`.

### Taak 2.3 - SubmitWorkerThread
- Doel: blocking HTTP submit uit coordinator thread halen.
- Input queue: submit opdrachten.
- Output event: `SUBMIT_RESULT`.

Acceptatiecriteria:
- Trage submit/retries blokkeren inbox/completion event verwerking niet.
- Backpressure: submit queue is begrensd op redelijke capaciteit.

## Fase 3 - Inbox Event Source (inotify)

### Taak 3.1 - Inbox watcher thread
- Nieuw bestand: `worker/inbox_watch.py` (of equivalent).
- Watch op `data/demo_jobs/inbox`:
- `IN_MOVED_TO`
- `IN_CREATE`
- `IN_CLOSE_WRITE`
- Push `INBOX_DIRTY` naar centrale event queue (met debounce).

Acceptatiecriteria:
- Nieuwe job in inbox triggert direct scheduler wakeup zonder periodic sleep.
- Geen regressie bij burst van veel inbox events.

### Taak 3.2 - Fallback gedrag
- Als inotify init faalt, worker start niet stilzwijgend “half”.
- Fail-fast met duidelijke startup fout.

Acceptatiecriteria:
- Broken watcher wordt zichtbaar in logs en service state.

## Fase 4 - Completion Event Source (SSE)

### Taak 4.1 - Worker SSE client
- Bestand: `worker/asr_client_remote.py` (nieuwe streaming helper).
- Verantwoordelijkheden:
- verbinden
- SSE parsing
- reconnect met backoff
- `since_seq` bijhouden
- `FEED_RESET` signaleren bij `feed_id` wijziging

Acceptatiecriteria:
- Bij netwerk hiccup reconnectt stream automatisch.
- Geen dubbele finalisatie van dezelfde completion (idempotent verwerkt).

### Taak 4.2 - Feed reset handling (v3-beleid)
- Bij `FEED_RESET`: fail alle pending jobs met huidige v3-foutsemantiek.
- Cursor resetten, vervolgens doorgaan met nieuwe jobs.

Acceptatiecriteria:
- Na asr-pool restart:
- lopende jobs worden niet gerecovered
- nieuwe jobs blijven direct verwerkbaar

## Fase 5 - Live en Upload Integratie

### Taak 5.1 - Live loop migreren
- Vervang huidige `completions poll + sleep` in live workerpad.
- Live pad gebruikt event reactor + submit worker + inbox watcher + SSE.

Acceptatiecriteria:
- `worker.live.max_outstanding_requests=1` blijft functioneel gelijk aan huidig gedrag.
- Geen regressie in live submit+reap flow.

### Taak 5.2 - Upload loop migreren
- Uploadpad idem event reactor (niet blokkerend per request).
- `pending_status` kan timer-driven blijven voor progress UX.

Acceptatiecriteria:
- Upload worker blijft nieuwe jobs aannemen terwijl andere requests in-flight zijn.
- Klaar voor opschalen naar meerdere asr-pool slots.

### Taak 5.3 - Upload outstanding configureerbaar maken
- Nieuwe setting introduceren:
- `worker.upload.max_outstanding_requests`
- Default defensief (bijv. 1), runtime schaalbaar naar >1.

Acceptatiecriteria:
- Met waarde 2+ wordt effectief parallel submit+reap gedraaid (zonder per-request block).

## Fase 6 - Config, Logging, Ops

### Taak 6.1 - Polling keys opruimen
- Verwijder afhankelijkheid van `polling_intervals.asr_remote_completions_poll_s` in worker event-pad.
- Voeg nieuwe event-stream keys toe in settings namespace.

Acceptatiecriteria:
- Worker functioneert zonder completions poll-interval key.

### Taak 6.2 - Observability
- Log counters toevoegen:
- inbox events ontvangen
- sse reconnects
- feed resets
- submits gestart/klaar/mislukt
- scheduler refill cycli

Acceptatiecriteria:
- Run-time gedrag is uit logs direct te verklaren bij incidenten.

## Fase 7 - Validatie en Rollout

### Taak 7.1 - Technische validatie
- Scenario's:
- live run normaal
- upload run normaal
- asr-pool restart tijdens in-flight
- backlog/burst jobs
- `max_outstanding=1` en `>1`

Acceptatiecriteria:
- Alle scenario's slagen volgens v3-semantiek.
- Geen blokkade op individuele requests.

### Taak 7.2 - Latency validatie
- Meet:
- inbox->submit start latency
- completion->worker finalize latency
- completion->portal zichtbaar latency (end-to-end)

Acceptatiecriteria:
- Duidelijke daling t.o.v. huidige poll-cadence baseline.

## Beslissingen (vastgelegd)

### 1) SSE wire format
- Endpoint: `GET /asr/v1/completions/stream?consumer_id=...&since_seq=...&heartbeat_s=...`
- SSE events:
- `meta` (eerste event met `feed_id`, `consumer_id`, `since_seq`, `next_seq`)
- `completion` (payload semantisch gelijk aan huidige `/asr/v1/completions` rows)
- `heartbeat` (periodiek keepalive met `feed_id` en `next_seq`)
- Feed-reset detectie gebeurt via `feed_id` wijziging.

### 2) Settings namespace
- Nieuwe namespace: `worker_events`.
- Event-driven worker parameters worden hier gecentraliseerd (SSE, inbox debounce, loop tick).

### 3) Worker code-structuur
- Modulair maar beperkt:
- `worker/worker_daemon.py` (entrypoint + wiring)
- `worker/event_loop.py` (centrale coordinator/state machine)
- `worker/inbox_watch.py` (inotify watcher)
- `worker/asr_client_remote.py` (SSE completion reader helper)

## Harde Scope-afspraak (geen fallback-paden)
- Completions in worker gaan in deze scope uitsluitend via SSE stream.
- Geen fallback naar completion polling (`/asr/v1/completions`) in runtime pad.
- Geen compat-laag of “voor de zekerheid” dode codepaden toevoegen.
