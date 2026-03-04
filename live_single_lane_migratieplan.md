# Live single-lane migratieplan (dual-lane verwijdering)

Doel: alle codepaden voor dual/speculative lane verwijderen in zowel backend (`transcribe-dev`) als frontend (`omniscripta-app`), zonder fallbacks.

## Stappen
1. Inventaris + kill-list opstellen (beide repo's).
2. Backend engine hard-cut naar single-lane.
3. Backend speculative datamodel verwijderen.
4. Backend speculative endpoints verwijderen.
5. Backend quality/metrics opschonen naar single-lane.
6. Worker/ASR-pool lane cleanup.
7. Frontend speculative state en rendering verwijderen.
8. Frontend cards/API-contract update (single-lane).
9. Config/env cleanup (dual/spec keys eruit).
10. E2E regressie + afronding.

## Voortgang
- Stap 1: afgerond
- Stap 2: afgerond
- Stap 3: afgerond
- Stap 4: afgerond
- Stap 5: afgerond
- Stap 6: afgerond
- Stap 7: afgerond
- Stap 8: afgerond
- Stap 9: afgerond
- Stap 10: afgerond (code + smokechecks; manual fixture gate nog door gebruiker)

## Manual gates
- Na elke stap: korte fixture test + expliciete GO.

## Stap 1 - Inventaris (in uitvoering)
Te leveren output:
- Concrete lijst van bestanden, functies en velden die dual/spec-only zijn.
- Voorstel wat hard verwijderd wordt vs. hernoemd naar single-lane equivalent.

## Stap 1 - Inventaris (opgeleverd)

### A. Backend (`/home/gunnar/projects/transcribe-dev`)

Hard verwijderen (dual/spec-only):
- `portal-api/live_engine_chunked_dual.py` (volledig bestand)
- `portal-api/speculative_quality.py` (volledig bestand)
- `portal-api/speculative_tuning_runner.py` (volledig bestand)
- `portal-api/asr_loadtest_runner.py` (speculative tuning/loadtest paden)
- `portal-api/main.py`:
  - `TRANSCRIBE_LIVE_ENGINE` parse/switch
  - route naar `run_live_session_ws_chunked_dual`
  - endpoint `/demo/live/sessions/{session_id}/speculative-quality`
  - speculative tuning endpoints
- `live_quality.py`:
  - alle `asr_speculative_*` metrics
  - alle `asr_combined_*` metrics
  - speculative job-id aggregatie vanuit statslog
- `config/service.json`:
  - `live.engine=chunked_dual` en dual/spec configblokken die niet meer gelden

Herwerken naar single-lane:
- `portal-api/live_engine_rolling_context.py`:
  - naamgeving `speculative_*` vervangen door `preview_*` (zelfde functionaliteit)
- `portal-api/live_sessions.py`:
  - speculative history/windows/open-window structuren verwijderen
  - resultmodel reduceren naar `final_text` + `preview`
- `portal-api/live_chunk_transcribe.py`:
  - lane metadata normaliseren naar single lane
- `worker/pipeline_live_chunk.py`:
  - `live_lane in {final,speculative}` vervangen door vaste single-lane metadata
- `asr-pool/main.py`:
  - queues `interactive_final` + `interactive_speculative` samenvoegen naar 1 interactive queue
  - `speculative_mode`/`live_lane` branch-logic verwijderen

### B. Frontend (`/var/www/omniscripta-app`)

Hard verwijderen (dual/spec-only):
- `js/services/LiveSessionService.js`:
  - `fetchSpeculativeQuality(...)`
- `js/components/LiveView.js`:
  - speculative quality state:
    - `speculativeQualityEnvelope`
    - `speculativeQualityRows`
    - `speculativeQualitySummaryText`
    - `speculativeQualityInFlight`
  - speculative quality pipeline:
    - `formatSpeculativeQualitySummary(...)`
    - `_appendSpeculativeQualityRow(...)`
    - `_formatSpeculativeQualityTablesText(...)`
    - `applySemiliveSpeculativeQualityEnvelope(...)`
    - `refreshSemiliveSpeculativeQuality(...)`
  - benchmarkregels:
    - `ASR speculative ...`
    - `ASR combined ...`
  - rolling-vs-dual branching in FE (wordt overbodig als backend single-lane is)
- `css/style.css`:
  - `.live-final-text-speculative`

Herwerken naar single-lane:
- `js/components/LiveView.js`:
  - transcript rendering met `final + preview` i.p.v. `final + speculative`
  - quality card enkel final/single-lane metrics

### C. Contract-impact (verwacht)
- API output wordt eenvoudiger:
  - geen speculative benchmark endpoint
  - geen speculative/combined quality velden
  - `result` bevat alleen single-lane preview + final
- FE hoeft geen engine-detectie/fallbacks meer te doen.

### D. Stap 1 gate
GO-criterium:
- akkoord op bovenstaande kill-list en op contract-impact.
