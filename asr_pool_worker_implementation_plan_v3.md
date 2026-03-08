# Implementatieplan v3

## Scope en harde keuzes

Dit plan volgt strikt de afgesproken v3-scope.

- `live_lane` blijft hardcoded `"single"` (geen multi-lane abstrahering).
- Worker split wordt direct ingevoerd: `worker-live` en `worker-upload`.
- Upload worker blijft conservatief: `max_outstanding_requests = 1`.
- Stale live submit gedrag in pool: `accept + superseded`.
- Completion feed v1 is in-memory.
- Restart-recovery en re-request gedrag worden nu niet gebouwd; alleen expliciete `TODO` comments.

## Werkafspraken (bindend)

- Maak de kleinste verdedigbare wijziging.
- Houd diffs strak gescoped.
- Laat ongerelateerde files ongemoeid.
- Voeg geen fallback- of compatibiliteitslagen toe tenzij expliciet gevraagd.
- Laat geen dode codepaden staan "voor de zekerheid".
- Focus strikt op v3-scope.

## Niet-doelen v3

- Geen pool restart-recovery mechanisme.
- Geen persistent completion cursor-opslag.
- Geen live multi-lane support.
- Geen extra backward compatibility flags/lagen.

## Technische aanpak per component

## 1) Worker split en mode-aware claim

- Voeg `worker.mode` toe met waarden `live` of `upload`.
- Claiming wordt job-kind aware:
  - `live` claimt alleen `job_kind=live_chunk`.
  - `upload` claimt alleen `job_kind=upload_audio`.
- Draai twee service instances met mode-specifieke env/config.

## 2) Worker submit + reap model

- Verwijder blocking per-request poll pad als hoofdflow.
- Worker doet in een eenvoudige loop:
  - reap terminal completions
  - claim + submit nieuwe jobs zolang limiet niet bereikt
  - korte sleep
- Upload mode houdt limiet op `1` outstanding.
- Live mode krijgt eigen outstanding limiet (startwaarde laag, bijvoorbeeld `2`).

## 3) ASR pool completion feed (in-memory)

- Voeg endpoint toe:
  - `GET /asr/v1/completions?consumer_id=...&since_seq=...&limit=...`
- Eventlog is in-memory met monotone `seq`.
- Feed levert alleen terminal events voor worker-reaping:
  - `completed`
  - `failed`
  - `cancelled`
  - `superseded`
- Voeg expliciete `TODO` comments toe voor restart-recovery en re-request semantics.

## 4) ASR pool live supersede (single lane)

- Supersede is alleen voor live requests (`source_kind=live_chunk`) en lane `"single"`.
- Sleutel: `(live_session_id, live_lane="single")` met `live_chunk_index` als freshness key.
- Regelset v1:
  - Running request wordt nooit gesuperseded.
  - Oudere queued request wordt `superseded` door nieuwere queued request.
  - Stale submit wordt geaccepteerd en meteen terminal `superseded`.

## 5) Portal-api rolling live zonder single-inflight bottleneck

- Verwijder hard single-inflight gating in rolling orchestration.
- Houd per sessie registry bij voor meerdere outstanding live jobs.
- Portal-api beslist welke completions nog relevant zijn en mag stale completions negeren.
- Transcript apply blijft sequenced op eigen registry-metadata, niet op globale completion-volgorde.

## Tasklist (uitvoerbaar)

Status bijgewerkt op: 2026-03-08

## Review follow-up (2026-03-08)

- [x] Critique punt 1: completion feed scan geoptimaliseerd (reverse-scan + early-break op `since_seq`) om lock-contention bij pollen te verlagen.
- [x] Critique punt 2 getriageerd: upload polling blijft bewust sync/blocking in v3; TODO-comment toegevoegd voor latere submit+reap-unificatie.
- [x] Critique punt 3 getriageerd: `superseded` lifecycle-state blijft expliciete v3-keuze; verdere lifecycle-vereenvoudiging buiten scope.

## Fase 0 - Voorbereiding en guardrails

- [x] Maak in code comments expliciet dat `live_lane` hardcoded `"single"` blijft in v3.
- [x] Leg in comments vast dat restart-recovery bewust niet in v3 zit (`TODO`).
- [x] Verwijder/vermijd oude fallback paden die niet meer gebruikt worden.

## Fase 1 - Worker split

- [x] Introduceer `worker.mode` config (`live|upload`).
- [x] Maak claim functie mode-aware op `job_kind`.
- [x] Voeg/actualiseer dev systemd units voor aparte live/upload worker instances.
- [x] Valideer dat live en upload niet meer door dezelfde worker-claimloop lopen.

## Fase 2 - Submit + reap in worker

- [x] Bouw worker main loop om naar submit+reap model. (live + upload hebben nu beide een aparte submit/reap-loop)
- [x] Houd worker outstanding map minimaal en expliciet. (live pending-map + upload pending-map met `max_outstanding=1`)
- [x] Upload mode hard limiteren op `max_outstanding_requests=1`.
- [x] Live mode limiteren op lage startwaarde (v3 veilig begin).
- [x] Finalize job status/directories uitsluitend op terminal completion events.
- [x] Upload wait-progress hersteld zonder blocking: completion-reap blijft leidend, heartbeat + batch pending-status alleen voor tussentijdse UI-status.
- [x] Upload ASR-fase timings worden incrementeel zichtbaar tijdens wait (bij stage-overgang direct vorige fase-tijd wegschrijven).

## Fase 3 - Pool completion feed v1

- [x] Voeg in-memory completion event store toe met seq.
- [x] Voeg `GET /asr/v1/completions` endpoint toe met `consumer_id`, `since_seq`, `limit`.
- [x] Voeg `GET /asr/v1/pending-status` endpoint toe voor batch stage/status snapshots van worker-pending requests.
- [x] Zorg dat terminal transitions completion events emitten.
- [x] Voeg `TODO` comments toe voor restart-recovery/re-request gedrag.

## Fase 4 - Pool live supersede v1

- [x] Parse live metadata uit request context: `live_session_id`, `live_chunk_index`, `live_lane`.
- [x] Forceer lane op `"single"` in supersede pad.
- [x] Implementeer queued-only supersede.
- [x] Implementeer stale submit => terminal `superseded` (accept+superseded).
- [x] Zorg dat `superseded` zichtbaar is in completion feed.

## Fase 5 - Portal rolling redesign

- [x] Verwijder single-inflight gating uit rolling enqueue/poll loop.
- [x] Voeg per-session outstanding registry toe.
- [x] Pas apply-logica aan zodat stale completions veilig genegeerd kunnen worden.
- [x] Behoud commit/preview correctheid op basis van expliciete sequence/index checks.

## Fase 6 - Opschonen en afronden

- [x] Verwijder niet-gebruikte oude codepaden na migratie.
- [x] Controleer dat geen compat/fallback lagen zijn achtergebleven.
- [x] Werk operationele notities bij met v3 beperkingen en bekende TODO's.

## Validatiechecklist

- [x] Live blijft responsief terwijl upload jobs lopen.
- [x] Upload pad blijft VRAM-conservatief en functioneel correct.
- [x] Supersede werkt voor queued live requests binnen lane `"single"`.
- [x] Worker reapt `completed|failed|cancelled|superseded` correct.
- [x] Portal negeert stale live completions zonder transcript corruptie.
- [x] Bekende beperking na pool restart is gedocumenteerd (geen recovery in v3).
- [x] Na ASR-pool restart detecteert worker feed-reset (`feed_id`), faalt alleen pre-restart pending jobs en kan direct weer nieuwe jobs verwerken.

Laatste parallel smoke-run (2026-03-08):
- upload job `job_20260308T100033Z_8ffc3251` bleef tijdens live run in `running/whisperx_wait` en eindigde daarna op `done`.
- live session `live_20260308T100033Z_a9ea0f06` was responsief (`ready_ms=20`, `ended_ms=4454`, finalization `ready`).
