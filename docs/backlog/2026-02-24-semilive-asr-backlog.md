# Semilive + ASR Backlog (Working)

## Status

Working backlog (ordered draft)

## Gerelateerde uitwerking

- Concrete uitvoering voor schaalbare centrale ASR pool:
  - `docs/backlog/2026-02-28-asr-runtime-pool-v1-implementation-plan.md`

## Doel

Compact overzicht van wat we al weten dat we willen, maar nog niet hebben, voor:

- semilive live recording UX/kwaliteit
- gedeelde ASR-architectuur (live + upload)
- robuustheid/ops/schaalbaarheid

Bron van waarheid voor gedrag blijft de code. Dit document is alleen voor prioritering en scope.

## Huidige basis (wat al werkt)

- `Live recording` draait semilive chunked (geen WhisperLive in user-flow)
- `live_chunk` gebruikt persistente warme ASR runner (`persistent_local`)
- warm-runner winst gevalideerd (fase 6)
- `upload_audio` gebruikt voor WhisperX-stap dezelfde ASR contract/backend-laag (fase 7), met behoud van bestaande upload-orchestratie
- Huidige warme ASR runner is **per worker process** (process-local IPC), nog niet gedeeld tussen meerdere workers
- Fixture test-harness op Live page:
  - playback fixture run
  - inject fixture run (zonder speaker->mic loopback)
  - fixture quality score (`Upload Similarity Score`)
- Eerste kwaliteitspad staat aan:
  - chunk overlap (pre-roll)
  - conservatieve final-text dedup (guarded)
  - dynamische `initial_prompt` tail (WhisperX CLI-compatibele route)
- LiveView exports:
  - `TXT`
  - `SRT`
  - `WAV` (integrale opname)

## Backlog (bekend gewenst, nog niet af)

### Kwaliteit / transcript

- Kwaliteitsbenchmark verder aanscherpen:
  - fixture-runs in rustige tijdsloten / mediaanvergelijking
  - inject-mode als primaire A/B benchmark
  - chunkgrens-annotatie (`reason=silence|max_duration|flush_tail`) meenemen in analyse
- Boundary artefact mitigatie (vervolg op huidige overlap+dedup):
  - dedup/merge verfijnen (nu conservatieve exact-word overlap)
  - segment/SRT boundary-dedup
  - later eventueel local-agreement-achtige merge
- Prompt conditioning (vervolg op dynamische tail):
  - statische termenlijst (keywords/hotwords) vanuit UI/config
  - combinatie statisch + dynamisch promptbeleid
- Noise/low-signal chunk filtering vóór enqueue (hallucinatie-achtige mini-chunks verminderen)
- Beter merge-beleid voor chunkresultaten (nu vooral concat; later overlap-aware)
- Chunker/VAD tuning:
  - thresholds (energy, silence)
  - min/max chunk duur
  - eventueel later robuustere VAD dan energy-based
- Post-stop final pass over volledige opname (kwaliteit van eindtranscript verhogen)

### Robuustheid / runtime

- Poll-race hardening rond chunk `status.json` (retry op `JSONDecodeError` of bredere read-consistency)
- Persistent runner hardening:
  - 1 retry vóór one-shot fallback bij transient errors
  - betere foutclassificatie (timeout/crash/io)
  - optioneel heartbeat/liveness voor snellere hang-detectie
- Persistent runner cache hygiene:
  - aligner-cache limiet/LRU (later vooral relevant bij multi-language)
  - `/tmp` init-file cleanup bij crash

### Ops / spool / retentie

- `live_chunk` jobdirs ephemeral maken na succesvolle merge in live sessie (spool-inodegroei beperken)
- Dubbele chunk-audio-opslag verminderen (sessie-opslag + queue job artefacten)
- Minimalere queue-job layout voor `live_chunk` (minder directories/files per chunk)
- Retentiebeleid voor chunkjob-failures (bijv. alleen laatste `N`, of debug-only)

### UX / product

- Time-to-first-transcript verlagen (zonder extra UI-complexiteit):
  - runner prewarm bij `Start`
  - eventueel startup-only kortere eerste chunk-policy
  - doel: sneller eerste zichtbare tekst zonder extra technische UI-labels
- Handoff vanuit live resultaat:
  - `Open in Upload audio`
  - `Open in editor`
- Heldere UI-status/progress voor semilive verwerking (verfijnen, niet fundamenteel)

### ASR-contract / architectuur

- `asr_v1` backend support voor echte `result.text` / `result.segments` (nu fail-fast als gevraagd)
- `diarize_enabled` end-to-end netjes ondersteunen via gedeelde ASR backendlaag (voor upload use-cases)
- Upload-orchestratie verder opschonen / extractie naar aparte pipelinefile (`pipeline_upload_audio.py`)
- ASR service pool generaliseren (meerdere persistente runners / routing per profiel/hardware)
- Transportkeuze later expliciteren (IPC blijft nu prima; FastAPI/gRPC pas als nodig)

### Schaalbaarheid / scheduling (later)

- Warm-runner sharing expliciet ontwerpen:
  - huidige warm runner is per-worker, niet gedeeld
  - voorkomen van VRAM-multiplicatie bij meerdere workers
  - route naar centrale ASR service/pool
- Resource/scheduling policy voor multi-worker / multi-user:
  - prioriteit/fairness `live_chunk` vs `upload_audio`
  - expliciete GPU concurrency policy/lock bij >1 worker
- Capaciteitsmodel per hardwareprofiel:
  - max gelijktijdige live streams
  - impact van gelijktijdige uploads
- Observability/metrics voor ASR-laag:
  - runner reuse rate
  - fallback rate
  - queue wait
  - latency percentiles

## Voorgestelde volgorde (eerste versie)

### Blok A — Kwaliteit eerst (zichtbaar voor gebruiker)

1. Kwaliteitsbenchmark aanscherpen (inject-mode + mediaanvergelijking + chunkgrens-analyse)
2. Boundary artefact mitigatie (vervolg: betere dedup/merge, SRT/segmentgrenzen)
3. Prompt conditioning vervolg (statische termen + gecombineerd promptbeleid)
4. Noise/low-signal chunk filtering
5. Chunker/VAD tuning (klein, gericht)

Opmerking:
- Punt 1 en 2 horen functioneel bij elkaar. Eerst meten waar de fouten vallen, dan dedup/merge gericht verfijnen.

### Blok B — Ops/retentie en kleine robuustheid

6. Spool/retentie cleanup (`live_chunk` ephemeral jobdirs, minder dubbele artefacts)
7. Poll-race hardening (`status.json` read-race)
8. Persistent runner hardening (retry + foutclassificatie; heartbeat alleen als nodig)

### Blok C — Productkwaliteit eindresultaat

9. Post-stop final pass over volledige opname
10. Handoff-features vanuit LiveView

### Blok D — Architectuurvervolg

11. `result.text` / `result.segments` echt ondersteunen in ASR backend
12. Upload-orchestratie verder opschonen (`pipeline_upload_audio.py`)
13. Diarization via gedeelde ASR backendlaag verder uitwerken

### Blok E — Schaalbaarheid (pas wanneer nodig)

14. Resource/scheduling policy (multi-worker / fairness)
15. GPU concurrency policy + capaciteitsmodel
16. ASR service pool generaliseren (meerdere persistente runners)

## Niet nu (bewuste de-prioritering)

- Nieuwe “mooie” docs over implementatiedetails als de code zelf al duidelijk is
- Grote scheduler/resource-complexiteit zolang er effectief `1` worker-instance draait
- Transportwissel naar FastAPI/gRPC vóórdat de ASR service interface inhoudelijk verder gestabiliseerd is
