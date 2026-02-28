# Speculative Lane v2: Kwaliteitsverbeteringen

Status: Concept (vervolgideeën op de huidige speculative lane v1)

## Context

De speculative lane v1 is geïmplementeerd en draait. De **final lane** levert goede kwaliteit, maar de **speculative lane** heeft merkbaar lagere kwaliteit. Dit document beschrijft twee concrete verbeteringen die de speculative output significant kunnen verbeteren.

### Oorzaakanalyse: waarom is de speculative kwaliteit laag?

De huidige speculative lane stuurt Whisper per call een **vast, kort audiosegment** van ~3 seconden (`WINDOW_MS=3000`). De window wordt als volgt berekend:

```python
# main.py, regel 524-527
semilive_speculative_effective_window_ms = max(
    LIVE_SEMILIVE_SPECULATIVE_WINDOW_MS,                              # 3000
    LIVE_SEMILIVE_SPECULATIVE_INTERVAL_MS + LIVE_SEMILIVE_SPECULATIVE_OVERLAP_MS,  # 1800 + 800 = 2600
)
# → effectief: 3000ms vast
```

In `_maybe_enqueue_speculative_job()` (regel 736-750) wordt dit vaste venster gebruikt om de audio te selecteren:

```python
want_bytes = round((float(semilive_speculative_effective_window_ms) / 1000.0) * LIVE_AUDIO_BYTES_PER_SECOND)
pcm = bytes(semilive_speculative_recent_pcm[-use_bytes:])  # laatste 3s uit de rolling buffer
```

Whisper krijgt dus elke 1.8 seconden een **disjunct stukje** van 3 seconden, met 800ms overlap. Dit geeft weinig context.

Ter vergelijking: WhisperLive (`collabora/WhisperLive`) geeft Whisper **alle audio sinds het laatst bevestigde segment** — een continu groeiend venster dat typisch 5-20 seconden is. Dit levert significant betere kwaliteit omdat Whisper meer context heeft om woorden en zinsbouw correct in te vullen. WhisperLive's aanpak is echter instabiel bij langlopende sessies (buffer groeit onbeperkt, stalling). De onderstaande verbeteringen combineren WhisperLive's kwaliteitsvoordeel met de stabiliteit van de huidige chunked architectuur.

### Getest en afgewezen: frozen final tail prompt

De `SPECULATIVE_INITIAL_PROMPT_ENABLED` flag (standaard `1`, regel 121-122) stuurt een initial prompt mee op basis van de laatste final tekst. In de praktijk bleek dit de speculative kwaliteit juist te **verslechteren**. De oorzaak is dat bij slechts 3 seconden audio de prompt een buitenproportioneel grote invloed heeft op de decoder, waardoor Whisper tekst genereert die "mooi aansluit" op de prompt in plaats van eerlijk te transcriberen wat er klinkt.

---

## Verbetering 1: Groeiende overlap ("micro-WhisperLive")

### Idee

In plaats van een vast venster van 3 seconden, stuur je **alle audio sinds de laatste final chunk** naar Whisper. Het venster groeit automatisch met de tijd, en reset bij elke nieuwe final chunk.

### Huidig gedrag

```
Final chunk klaar (t=0)
  → spec call (t=1.8s): audio [0.0 → 3.0s]   (3s vast venster)
  → spec call (t=3.6s): audio [0.8 → 3.8s]   (3s vast venster, 800ms overlap)
  → spec call (t=5.4s): audio [2.6 → 5.6s]   (3s vast venster, 800ms overlap)
Final chunk klaar (t=8s) → reset
```

### Nieuw gedrag

```
Final chunk klaar (t=0)
  → spec call (t=1.8s): audio [0.0 → 1.8s]   (1.8s)
  → spec call (t=3.6s): audio [0.0 → 3.6s]   (3.6s — groeiend)
  → spec call (t=5.4s): audio [0.0 → 5.4s]   (5.4s — groeiend)
  → spec call (t=7.2s): audio [0.0 → 7.2s]   (7.2s — groeiend)
Final chunk klaar (t=8s) → reset
```

### Implementatie

De wijziging zit op één plek: in `_maybe_enqueue_speculative_job()` in `main.py`.

**Stap 1: Houd het tijdstip bij van de laatste final chunk.**

Voeg een variabele toe in de WebSocket-handler scope (naast de bestaande speculative state, regel ~518):

```python
semilive_speculative_last_final_end_ms = 0
```

Update deze waarde bij het vastleggen van een final chunk resultaat (de plek waar `semilive_chunk_results` wordt bijgewerkt):

```python
semilive_speculative_last_final_end_ms = int(max(0, t1_ms_of_final_chunk))
```

**Stap 2: Vervang de vaste window door de dynamische groeiende overlap.**

In `_maybe_enqueue_speculative_job()`, vervang de `want_bytes` berekening (regel 736-741):

```python
# OUD: vast venster
want_bytes = round((float(semilive_speculative_effective_window_ms) / 1000.0) * LIVE_AUDIO_BYTES_PER_SECOND)

# NIEUW: alles sinds laatste final chunk
time_since_final_ms = max(0, int(semilive_recording_duration_ms) - int(semilive_speculative_last_final_end_ms))
want_bytes = int(max(
    LIVE_AUDIO_SAMPLE_WIDTH_BYTES,
    round((float(time_since_final_ms) / 1000.0) * LIVE_AUDIO_BYTES_PER_SECOND),
))
```

**Stap 3: Vergroot het rolling buffer maximum.**

De huidige `semilive_speculative_recent_pcm_max_bytes` is gebaseerd op `effective_window_ms + 1000ms` (regel 530-533). Dit moet groot genoeg zijn voor de langstmogelijke groei: `LIVE_SEMILIVE_CHUNK_MAX_MS` (= 12000ms). Wijzig:

```python
# OUD
semilive_speculative_recent_pcm_max_bytes = round(
    ((semilive_speculative_effective_window_ms + 1000) / 1000.0) * LIVE_AUDIO_BYTES_PER_SECOND
)

# NIEUW: buffer moet de volledige inter-final-chunk periode aankunnen
semilive_speculative_recent_pcm_max_bytes = round(
    ((float(LIVE_SEMILIVE_CHUNK_MAX_MS) + 2000) / 1000.0) * LIVE_AUDIO_BYTES_PER_SECOND
)
# → max ~14s buffer (12s max chunk + 2s marge) = ~448KB bij 16kHz mono PCM16
```

### Begrensde GPU-kosten

De audio groeit maximaal tot `LIVE_SEMILIVE_CHUNK_MAX_MS` (12 seconden), want daarna vuurt de final lane een chunk af en reset de speculative buffer. Whisper verwerkt 12 seconden audio op een moderne GPU in ~1-2 seconden. De GPU-kosten per speculative call groeien wel per iteratie, maar zijn begrensd en voorspelbaar.

### Cold-start mitigatie: pre-roll uit de vorige final chunk

De eerste speculative call na een final chunk reset heeft maar ~1.8 seconden audio — het zwakste moment. Door 1-2 seconden audio-overlap met het einde van de vorige final chunk als pre-roll mee te sturen, krijgt Whisper direct akoestische context bij die eerste call. Dit is effectief de audio-variant van de frozen final tail prompt, maar dan met het echte geluid in plaats van tekst.

```
Final chunk klaar (t=0), speculative pre-roll = 1500ms

  → spec call 1 (t=1.8s): audio [-1.5 → 1.8s]  = 3.3s (met pre-roll)
  → spec call 2 (t=3.6s): audio [-1.5 → 3.6s]  = 5.1s (pre-roll + groei)
  → spec call 3 (t=5.4s): audio [-1.5 → 5.4s]  = 6.9s
```

Naarmate de buffer groeit, wordt de pre-roll relatief steeds kleiner. Maar bij die cruciale eerste call maakt het een groot verschil: 3.3 seconden context in plaats van 1.8.

Implementatie: bewaar de laatste `SPECULATIVE_PRE_ROLL_MS` bytes van de vorige final chunk audio. Prepend deze aan `semilive_speculative_recent_pcm` bij elke reset. De pre-roll wordt automatisch onderdeel van het groeiende venster — geen extra logica nodig.

### Verwachte kwaliteitsverbetering

- Eerste call: vergelijkbaar met huidige kwaliteit (weinig audio)
- Latere calls: significant beter — Whisper heeft 5-10 seconden context in plaats van 3
- Output wordt stabieler: elke call hertranscribeert dezelfde audio vanaf het begin, correcties zijn vloeiend
- Geen overlap-dedup meer nodig voor de speculative lane (de stukjes worden niet meer aan elkaar geplakt)

---

## Verbetering 2: Silero VAD als speculative gate

### Idee

Sla speculative calls over wanneer er geen spraak gedetecteerd wordt. Dit bespaart GPU-tijd en voorkomt dat Whisper nonsens-transcripties genereert van stilte, achtergrondgeluid, of typgeluid.

### Huidige situatie

De final lane gebruikt een simpele energy-gebaseerde stiltedetectie (`ENERGY_THRESHOLD=12`). De speculative lane heeft helemaal geen stiltedetectie — hij vuurt elke `INTERVAL_MS` milliseconden ongeacht of er spraak is.

Energy-gebaseerde detectie herkent alleen **volume**, niet **spraak**. Een tikkende klok, airco, of tikken op een toetsenbord worden als "geluid" gezien en triggeren onterecht een chunk. Dit verklaart mede de huidige matige kwaliteit van de speculative lane.

### Silero VAD

[Silero VAD](https://github.com/snakers4/silero-vad) is een klein neuraal netwerk (~2MB) specifiek getraind om menselijke spraak te onderscheiden van niet-spraak. Het wordt ook door WhisperLive (in `whisper_live/vad.py`) en WhisperX intern gebruikt.

Kenmerken:
- **Modelgrootte**: ~2MB (ONNX of PyTorch)
- **Inferentietijd**: <1ms per 512 samples (~32ms audio bij 16kHz)
- **Nauwkeurigheid**: aanzienlijk beter dan energy-based detectie
- **Geen GPU nodig**: draait prima op CPU

### Implementatie

**Stap 1: Silero VAD model laden bij startup.**

In `main.py` (of een aparte `vad.py` module):

```python
import torch

_silero_vad_model = None

def _get_silero_vad():
    global _silero_vad_model
    if _silero_vad_model is None:
        _silero_vad_model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
        )
    return _silero_vad_model
```

**Stap 2: Voeg een voice-check toe aan `_maybe_enqueue_speculative_job()`.**

Vóór de `want_bytes` berekening (rond regel 733), voeg toe:

```python
# Controleer of de recente audio spraak bevat
if LIVE_SEMILIVE_SPECULATIVE_VAD_ENABLED:
    recent_tail_bytes = min(len(semilive_speculative_recent_pcm), int(LIVE_AUDIO_BYTES_PER_SECOND * 1.5))
    if recent_tail_bytes >= LIVE_AUDIO_BYTES_PER_SECOND * 0.3:
        tail_pcm = bytes(semilive_speculative_recent_pcm[-recent_tail_bytes:])
        audio_tensor = torch.frombuffer(tail_pcm, dtype=torch.int16).float() / 32768.0
        vad_model = _get_silero_vad()
        confidence = vad_model(audio_tensor, LIVE_AUDIO_SAMPLE_RATE_HZ).item()
        if confidence < 0.5:
            # Geen spraak gedetecteerd — sla deze speculative call over
            semilive_speculative_last_emit_mono = now_mono  # voorkom burst bij volgende interval
            return
```

**Stap 3: Feature flag.**

```python
LIVE_SEMILIVE_SPECULATIVE_VAD_ENABLED = str(
    os.getenv("TRANSCRIBE_LIVE_SEMILIVE_SPECULATIVE_VAD_ENABLED", "0")
).strip().lower() in {"1", "true", "yes"}
```

### Verwachte impact

- **GPU-besparing**: In een typisch gesprek met pauzes tussen zinnen wordt 30-50% van de speculative calls overgeslagen.
- **Kwaliteitsverbetering**: Whisper krijgt alleen audio met spraak, waardoor het minder vaak hallucinaties produceert over stilte of ruis.
- **Combinatie met groeiende overlap**: De VAD voorkomt dat de groeiende buffer gevuld raakt met stilte — alleen daadwerkelijke spraak triggert een speculative call.

---

## Implementatievolgorde

1. **Groeiende overlap eerst** — dit is de grootste kwaliteitsverbetering en de eenvoudigste wijziging (3 plekken in `main.py`).
2. **Silero VAD daarna** — extra optimalisatie voor GPU-gebruik en ruisonderdrukking. Kan achter een feature flag en onafhankelijk getest worden.
3. **Heroverweeg initial prompt** — zodra de groeiende overlap actief is (Whisper heeft dan 5-10s audio), kan de frozen final tail prompt opnieuw worden getest. Met meer audio-context is de balans tussen prompt en akoestisch bewijs anders, en zou de prompt wél positief kunnen uitpakken.

## Feature flags uiteindelijk

| Flag | Default | Beschrijving |
|---|---|---|
| `TRANSCRIBE_LIVE_SEMILIVE_SPECULATIVE_GROWING_OVERLAP` | `0` | Dynamisch groeiend venster i.p.v. vast WINDOW_MS |
| `TRANSCRIBE_LIVE_SEMILIVE_SPECULATIVE_VAD_ENABLED` | `0` | Silero VAD gate voor speculative calls |
| `TRANSCRIBE_LIVE_SEMILIVE_SPECULATIVE_INITIAL_PROMPT_ENABLED` | `1` | Initial prompt (opnieuw testen na groeiende overlap) |
