# Patchplan: Speculative Lane v1 (Semilive)

Status: Draft patchplan (nog niet geïmplementeerd)

## Doel
Snellere, tijdelijke preview-tekst tonen tijdens live opname via een `speculative` lane, zonder impact op het definitieve (`final`) transcriptpad.

## Belangrijke guardrail (bevroren final baseline)
De `final` semilive lane blijft voorlopig op de huidige A1-tuning:

- `MAX_MS=12000`
- `SILENCE_MS=1200`
- `ENERGY_THRESHOLD=12`
- `PRE_ROLL_MS=800`
- `DEDUP_ENABLED=1`
- `INITIAL_PROMPT_ENABLED=1`

Speculative lane mag **geen** wijzigingen aan deze final lane forceren en mag:

- `final_text` / `final_segments` niet beïnvloeden
- exports (`TXT`, `SRT`, `WAV`) niet beïnvloeden
- fixture score / benchmark niet beïnvloeden

## UX-richting (v1)
Speculative tekst tonen als een tijdelijke suffix achter het finale transcript:

- lichtgrijs
- italic
- UI-only (niet gemerged in final transcript state)

Bij nieuwe final chunk update:

- speculative suffix wissen/vervangen
- finale tekst blijft bron van waarheid

## Feature flags (default uit)
Toe te voegen in `portal-api/main.py` (env-gestuurd):

- `TRANSCRIBE_LIVE_SEMILIVE_SPECULATIVE_ENABLED=0`
- `TRANSCRIBE_LIVE_SEMILIVE_SPECULATIVE_INTERVAL_MS=1800`
- `TRANSCRIBE_LIVE_SEMILIVE_SPECULATIVE_WINDOW_MS=3000`
- `TRANSCRIBE_LIVE_SEMILIVE_SPECULATIVE_OVERLAP_MS=800`
- `TRANSCRIBE_LIVE_SEMILIVE_SPECULATIVE_MAX_STALENESS_MS=1200`
- `TRANSCRIBE_LIVE_SEMILIVE_SPECULATIVE_REQUIRE_NO_FINAL_PENDING=1`

## Heuristiek (v1)
- Final lane heeft altijd prioriteit
- Speculative lane alleen als:
  - recording loopt
  - geen final pending chunks (guardrail)
  - geen speculative job in flight
  - interval verstreken
- Max 1 speculative job tegelijk
- Speculative resultaat droppen als het te oud/stale is

## Bestand-voor-bestand patchstappen

### 1. `portal-api/main.py` (scheduler + lane orchestration)
Voeg speculative lane runtime state toe in de live WS-handler:

- `speculative_in_flight`
- `speculative_last_emit_mono`
- `speculative_seq`
- counters:
  - `speculative_enqueued`
  - `speculative_shown`
  - `speculative_dropped_busy`
  - `speculative_dropped_stale`
- timestamps/metrics:
  - `time_to_first_speculative_ms`

Voeg helperfuncties toe:

- `_maybe_enqueue_speculative_job()`
- `_poll_speculative_jobs()`

Bron voor speculative audio:

- recent opnamebuffer / recent PCM venster
- los van de final chunkerlogica (final lane niet aanpassen)

### 2. `portal-api/live_chunk_transcribe.py` (job metadata passthrough)
Hergebruik `enqueue_chunk_pcm16(...)`, maar voeg optionele metadata toe in job `options`:

- `live_lane="speculative"`
- `speculative_seq`
- `speculative_audio_end_ms`

`poll_job(...)` hoeft in v1 inhoudelijk niet aangepast te worden.

### 3. `portal-api/live_sessions.py` (speculative preview state + metrics)
Breid `LiveSession` uit met speculative preview fields:

- `semilive_speculative_text`
- `semilive_speculative_seq`
- `semilive_speculative_audio_end_ms`
- `semilive_speculative_updated_unix`

En speculative counters/metrics:

- `speculative_enqueued`
- `speculative_shown`
- `speculative_dropped_busy`
- `speculative_dropped_stale`
- `time_to_first_speculative_ms`

Nieuwe helpers:

- `update_semilive_speculative_preview(...)`
- `clear_semilive_speculative_preview(...)`
- `update_semilive_speculative_metrics(...)`

Expose in `semilive_result_snapshot`:

- `speculative_preview`
- `speculative_metrics`

### 4. `worker/pipeline_live_chunk.py` (metadata/telemetry only)
Laat de worker lane metadata doorzetten naar `status.json`:

- `live_lane`
- `speculative_seq` (indien aanwezig)

Geen behaviorwijziging in de ASR pipeline voor v1.

Belangrijk:

- `initial_prompt` voor speculative lane in v1 uit laten (API-side niet meesturen)

### 5. `portal-api/live_quality.py` (benchmark metrics, geen score-impact)
Breid `run_metrics` uit met speculative lane observability:

- `time_to_first_speculative_ms`
- speculative counters (`shown`, `dropped_busy`, `dropped_stale`)

Scoreberekening blijft ongewijzigd.

### 6. `/var/www/omniscripta-app/js/components/LiveView.js` (UI rendering)
Speculative preview tonen als suffix achter final transcript:

- lichtgrijs + italic (via CSS class)
- alleen tonen als `speculative_preview.text` vers genoeg is
- niet tonen als suffix al exact in final transcript-einde voorkomt (simpele dedup-check)

Bij nieuwe final transcript update:

- speculative suffix resetten/vervangen

### 7. `/var/www/omniscripta-app/css/style.css` (styling)
Voeg styling toe voor speculative suffix:

- kleur: lichtgrijs
- `font-style: italic`
- optioneel lagere opacity

### 8. `/var/www/omniscripta-app/js/services/LiveSessionService.js` (waarschijnlijk geen wijziging)
V1 gebruikt bestaande `/result` polling, dus vermoedelijk geen codewijziging nodig.

## Implementatievolgorde (veilig)
1. Backend-only scheduler + sessiestate + metrics (zonder UI)
2. Dev-test met flag aan:
   - final lane mag niet regressief zijn
   - speculative counters moeten bewegen
3. Frontend speculative suffix UI (dev)
4. Fixture benchmark uitbreiden met speculative metrics (indien nog nodig)

## Must-pass criteria (v1)
- Final lane (A1) blijft gelijk qua:
  - inject score
  - chunk reasons
  - errors/finalization
- Speculative lane:
  - veroorzaakt geen zichtbare backloggroei in final lane
  - mag resultaten droppen zonder schade
  - toont soms preview (UX winst)

## Bewuste non-goals (v1)
- Geen speculative -> final merge
- Geen speculative tekst in exports of quality score
- Geen parallelle workers/runners
- Geen scheduler herontwerp
- Geen prompt conditioning voor speculative lane

## Vervolgideeën (na v1, expliciet nog niet in scope)

### A. Speculative `initial_prompt` via `frozen_final_tail` (veiliger dan dynamisch speculative prompt)
Doel:
- speculative lane wél context geven, zonder feedback-loops vanuit speculative output zelf

Idee:
- gebruik **niet** de speculative output als promptbron
- bouw in plaats daarvan een `initial_prompt` uit de **laatst gecommitte final transcripttekst**
- “freeze” die prompt totdat de **volgende final chunk** binnenkomt

Praktisch model:
- nieuwe optie/mode (later), bijvoorbeeld:
  - `speculative_initial_prompt_mode = off | frozen_final_tail`
- bij elke `final ready`:
  - bereken nieuwe frozen prompt uit actuele merged `final_text`
  - tail bv. `20–30` woorden, met char-cap (`300–400`)
- alle speculative chunks tot de volgende final gebruiken exact dezelfde frozen prompt

Waarom dit aantrekkelijk is:
- minder kans op repetition-loops dan speculative-dynamische prompts
- wel continuïteit over boundaries
- promptbron blijft “authoritative” (`final` lane)

### B. UX-gerichtere `stale`-definitie (op basis van laatst gecommitte final coverage)
Huidige v1-heuristiek:
- speculative wordt stale beoordeeld t.o.v. **huidige live opnamepositie** (`recording_duration_ms`)

Probleem:
- een speculative resultaat kan technisch “achterlopen” op de live cursor,
  maar voor de gebruiker nog steeds een nuttige voortzetting zijn van de
  **laatst zichtbare final tekst**

Vervolgidee:
- beoordeel speculative “stale” (of “wissen”) primair t.o.v. de
  **laatst gecommitte final chunk coverage** i.p.v. alleen t.o.v. live cursor

Benadering:
- houd `last_final_covered_ms` bij (bijv. `t1_ms` van laatste final chunk)
- speculative preview met `speculative_audio_end_ms` is nog relevant zolang:
  - `speculative_audio_end_ms > last_final_covered_ms`
- wis/drop speculative preview zodra een nieuwe final hem inhoudelijk/tijdmatig inhaalt
  - bij benadering: `last_final_covered_ms >= speculative_audio_end_ms` (evt. met marge)

Belangrijk onderscheid:
- `drop before show` (backend guardrail)
- `clear after show` (UI wist speculative suffix zodra final hem inhaalt)

Waarom dit UX-matig beter kan zijn:
- sluit aan op wat de gebruiker daadwerkelijk ziet (final transcript), niet op een onzichtbare “live cursor”
- maakt speculative preview waarschijnlijk bruikbaarder zonder het final transcriptpad te veranderen

### C. Newline-heuristiek tussen final transcript en speculative suffix (UI-only)
Observatie:
- speculative preview kan natuurlijker aanvoelen als die dezelfde “regelstructuur” volgt als final chunk updates
- eerste eenvoudige UX-keuze: de eerste speculative preview na een final op een **nieuwe regel** starten

Later (buiten v1-scope) mogelijk verfijnen met heuristieken op basis van:
- punctuation / zinsgrenzen
- chunk reason (`silence` vs `max_duration`)
- overlap/coverage t.o.v. final tail
- uiteindelijke minimalistische UX-keuzes

Belangrijk:
- dit is UI-only
- geen impact op `final_text`, exports of fixture benchmark score
