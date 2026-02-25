# Fase 6 Validatie: `live_chunk` Warm Runner (Persistent Local ASR)

## Status

Accepted (dev baseline)

## Samenvatting

De persistente lokale ASR-runner (`persistent_local`) voor `live_chunk` jobs is gevalideerd op meerdere sessies.

Besluit:

- `TRANSCRIBE_LIVE_CHUNK_WARM_ENABLED` blijft standaard **aan** voor `live_chunk`
- one-shot backend blijft als fallback-pad behouden

## Wat gemeten is

Vergelijking tussen:

- pre-warm (`oneshot_subprocess`, align al uit voor `live_chunk`)
- post-warm (`persistent_local`, align uit)

Meetpunten:

- gemiddelde chunkverwerkingstijd
- runner reuse (`runner_reused`)
- `Stop -> ready` (van `semilive_recording_finalized` naar `session_closed`)
- chunk errors / poll errors

## Gebruikte sessies

Pre-warm (one-shot):

- `live_20260224T093111Z_691a4b91` (contract-first one-shot sanity; `ready`)
- `live_20260224T085000Z_612ab80d` (one-shot align-off run met poll-race error; gebruikt als context, niet als hoofdbenchmark)

Post-warm (persistent):

- `live_20260224T094258Z_a6db5f1b` (korte validatie)
- `live_20260224T094945Z_83b3e50f` (lange run ~9m15s)
- `live_20260224T101740Z_077af7b1` (fase-5 telemetrie sanity)

## Resultaten (kerncijfers)

### Pre-warm baseline (one-shot, align uit)

Sessie `live_20260224T093111Z_691a4b91`:

- `job_count=5`
- `runner_kind=oneshot_subprocess` voor alle chunks
- gemiddelde chunk `total_s` (alle chunks): **~5.27s**
- gemiddelde chunk `total_s` (chunks 2..N): **~5.25s**
- `Stop -> ready`: **~2.36s**

### Post-warm (persistent local runner, align uit)

Sessie `live_20260224T094258Z_a6db5f1b`:

- `job_count=10`
- `runner_kind=persistent_local` voor alle chunks
- `runner_reused`: `false` voor chunk 0, daarna `true` voor chunks 1..9
- gemiddelde chunk `total_s` (alle chunks): **~0.88s**
- gemiddelde chunk `total_s` (chunks 2..N): **~0.45s**
- `Stop -> ready`: **~1.52s**

Sessie `live_20260224T094945Z_83b3e50f` (lange run):

- `job_count=31`
- `runner_kind=persistent_local` voor alle chunks
- `runner_reused`: `false` voor chunk 0, daarna `true` voor chunks 1..30
- gemiddelde chunk `total_s` (alle chunks): **~0.63s**
- gemiddelde chunk `total_s` (chunks 2..N): **~0.50s**
- `chunks_failed=0`, `chunks_pending=0`, `finalization_state=ready`
- `Stop -> ready`: **~1.78s**

Sessie `live_20260224T101740Z_077af7b1` (fase-5 sanity):

- `job_count=7`
- `runner_reused` status-telemetrie bevestigd in chunk `status.json`
- `asr_warm_fallback_used=false` in alle chunkjobs
- `Stop -> ready`: **~1.02s**

## Conclusie

De winst van de warme runner is duidelijk en groot:

- chunkverwerkingstijd daalt van ongeveer **5.25s** (one-shot) naar ongeveer **0.45–0.50s** (warme runner, na eerste chunk)
- geen relevante backlog meer aan het eind van de lange run
- `Stop -> ready` blijft laag (~1–2s) ook bij langere sessies

Dit valideert de architectuurkeuze:

- `live_chunk` via persistente lokale ASR-runner
- one-shot backend als fallback

## Bekende nuances

- De eerste chunk per sessie heeft nog steeds warmup-kosten (model load / `runner_reused=false`)
- Een oudere one-shot run (`live_20260224T085000Z_612ab80d`) had een `semilive_chunk_poll_error` (`JSONDecodeError`) door een status-poll race; dit staat los van de warm-runner winst
- Kwaliteitsissues aan chunkgrenzen (boundary artefacts) blijven een aparte verbeterlijn en zijn niet opgelost door deze fase

## Vervolg (fase 7)

- `upload_audio` orchestratie migreren naar dezelfde ASR interface/backend-laag (orchestratie blijft apart)
- resource/scheduling policy pas toevoegen zodra multi-worker / multi-user nodig wordt
