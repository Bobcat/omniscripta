# Semi-Live Transcriptie: Analyse en Aanbevelingen

Date: 2026-02-24
Status: Advies / brainstorm

## Context

Het WhisperLive sidecar-pad (ws://127.0.0.1:9090) bleek onbruikbaar voor langere recordings. Na ~30s liep de sidecar vast door full-context re-inference per frame (5–9s stalls, bevestigd door 9 stall-logfiles in `data/live_stats/`). Dit is een architecturele beperking van WhisperLive, geen bug.

De codebase is herwerkt naar een **semi-live chunked** aanpak die de bestaande WhisperX batch worker hergebruikt.

## Huidige architectuur

```
Browser Mic → AudioWorklet → PCM16 16kHz → WebSocket
  → LiveWavRecorder (volledige WAV opslag)
  → LiveAudioChunker (RMS VAD, splits op stilte of max duur)
    → LiveChunkBatchBridge (chunk → WAV → inbox/ queue)
      → Bestaande WhisperX Worker (transcriptie)
        → Poll status.json + .srt → merge in sessie
```

Modules: `live_chunker.py`, `live_chunk_transcribe.py`, `live_recordings.py`, `live_sessions.py`.

## Waarom sequentieel prima is

WhisperX transcribeert ~100x sneller dan realtime (RTF ~0.01). Een chunk van 10s kost ~100ms inference. Parallelle workers zijn pas nodig bij meerdere gelijktijdige sessies.

Latency-breakdown per chunk:

| Component | Tijd |
|---|---|
| Chunk vol (wachten op stilte/max duur) | 1–20s |
| Filesystem overhead | ~50ms |
| WhisperX inference | ~50–200ms |
| Poll interval | 0.75s (huidig) |
| **Totaal** | **~2–21s**, gedomineerd door chunk-opbouw |

## Overlap voor chunk boundary artefacten

### Whisper context-behoefte

Whisper verwerkt audio in blokken van 30s (attention window). De encoder ziet het complete mel-spectrogram in één keer; woorden aan het einde van de input hebben minder rechtercontext.

- Woorden met ≥2s audio erna: vrijwel altijd stabiel.
- Woorden in de laatste ~1–2s: kunnen veranderen als er meer context zou zijn.
- Stilte ≥1s na een woord: woord is al stabiel.

### Aanbevolen overlap: 2–3 seconden

- Chunk N eindigt op `t=15s`, chunk N+1 begint op `t=13s` (2s overlap).
- Beide transcriberen het stuk 13–15s.
- Bij mergen: neem de versie van chunk N (daar zit dat stuk in het midden met meer context, niet aan de rand).
- De overlap is vooral nodig bij `max_duration` sluitingen (midden in spraak). Bij `silence` sluitingen (1.2s stilte) is het laatste woord al stabiel.

### Dedup-strategie

Timestamp-overlap dedup is betrouwbaarder dan fuzzy text matching. Als segment N.einde en segment N+1.begin dezelfde `t0_ms..t1_ms` range overlappen, neem het segment met de langere context (= uit de chunk waar het niet aan de rand zit).

## Aanbevelingen

### Latency verlagen

1. **Poll-interval naar ~250ms** — De huidige 0.75s is conservatief. WhisperX is in ~100ms klaar, dan wacht je 0.75s op poll. Met 250ms direct merkbaar sneller, nauwelijks extra load (alleen `stat()` + JSON read).

2. **Silence threshold verlagen naar ~800ms** — Huidige 1200ms is conservatief. Elke ms eerder chunk sluiten = eerder in de queue.

3. **(Optioneel) Directe notificatie ipv polling** — Worker schrijft een markerbestand of unix socket notificatie. Complexity is hoger, pollen werkt prima bij huidige schaal.

### Transcriptie-kwaliteit

4. **Prompt conditioning per chunk (hoogste prioriteit)** — Whisper `initial_prompt` parameter: geef de laatste ~50 woorden van chunk N-1 mee als prompt voor chunk N. Voordelen:
   - Betere continuïteit in stijl en vocabulaire.
   - Minder hallucinatie op korte chunks.
   - Betere handling van eigennamen die eerder herkend zijn.
   - Vereist kleine aanpassing in job options + worker.

5. **Noise-chunk filter (hoge prioriteit)** — Met `energy_threshold: 12` (zeer laag) komen zuiver-ruis chunks door. Whisper hallucineert op stille input ("Thank you for watching", "Subscribe to my channel"). Filter chunks met 0 speech frames of zeer lage gemiddelde RMS vóór enqueue.

6. **Language locking** — Huidige `language: "en"` is goed. Zonder expliciete taal doet Whisper auto-detect per chunk, wat bij korte chunks (< 3s) regelmatig fout gaat. Overweeg taalknop in de LiveView UI.

7. **Overlap met context-aware dedup** — Zie bovenstaande sectie. 2–3s overlap, prefer-inner-context strategie.

### Efficiëntie

8. **Skip topics/LLM fase voor chunk jobs** — De worker doet mogelijk topic-extractie via Tabby LLM per job. Voor live chunks alleen WhisperX + SRT doen. De `live_chunk_mode: True` in job options is er al — zorg dat de worker hierop reageert.

9. **Chunk WAV cleanup** — `data/live_chunk_jobs/` groeit bij elke sessie. Cleanup na succesvolle transcriptie, of periodieke purge.

### UX

10. **Progressieve tekst-reveal** — Chunk-resultaten met subtiele fade-in animatie tonen. Voelt vloeiender dan een blok tekst dat ineens verschijnt.

11. **Chunk progress indicator** — Niet als nummers (te developer), maar als dunne voortgangsbalk of pulserend icoon dat aangeeft "tekst komt eraan".

12. **"Bezig met verwerken..." na stop** — Na stoppen verwerkt de backend nog lopende chunks (max 20s). Toon duidelijke status met spinner, zodat de gebruiker weet dat het bezig is.

## Quick-wins prioritering

| # | Actie | Impact |
|---|---|---|
| 1 | Prompt conditioning (#4) | Grootste kwaliteitsverbetering |
| 2 | Noise-chunk filter (#5) | Voorkomt hallucinaties + onnodig werk |
| 3 | Poll-interval 250ms (#1) | Direct merkbare latency-winst |
| 4 | Overlap 2–3s + dedup (#7) | Elimineert boundary artefacten |
| 5 | Skip topics voor chunks (#8) | Snellere verwerking, minder GPU gebruik |

## Eindgebruiker perspectief

Wat een eindgebruiker (interview/meeting) wil:
- Start / pause / stop met duidelijke visuele indicatie.
- Opname beschikbaar als bestand (WAV/mp3) → ✅ WAV recorder aanwezig.
- .srt meteen beschikbaar na stop → haalbaar met huidige pad, mits finalisatie-wachttijd getoond.
- Geen extra handelingen → semi-live flow werkt, finalisatie is niet instant maar ~20s acceptabel.
- Live tekst zien verschijnen → per chunk (2–7s vertraging), niet woord-voor-woord.
- (Brainstorm) LLM advies op lopende conversatie → vereist lopende merged transcript, architectureel mogelijk.
