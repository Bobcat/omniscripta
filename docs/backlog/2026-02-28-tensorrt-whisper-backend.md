# Backlog: TensorRT Whisper Backend voor ASR Pool

Status: Gepland (v2/v3, ná ASR Pool v1 + observability)

## Doel

Whisper inference versnellen via NVIDIA TensorRT-LLM als alternatief voor de huidige FasterWhisper/CTranslate2 backend. Hierdoor kan dezelfde GPU meer gelijktijdige live streams bedienen.

## Verwachte impact

| Metric | FasterWhisper (huidig) | TensorRT (verwacht) |
|---|---|---|
| Inference 12s audio (large-v3) | ~1.5-2s | ~0.7-1s |
| GPU-bezetting per stream | ~35-50% | ~15-25% |
| Max gelijktijdige streams (1 GPU) | 2-3 | 4-6 |

## Voorwaarden

- ASR Pool v1 draait (Fase 1-5 af)
- Meetdata beschikbaar uit Fase 5 (basislijn FasterWhisper performance)
- Runner-slot architectuur ondersteunt meerdere backend-typen

## Scope

- TensorRT-LLM compilatie pipeline voor Whisper large-v3
- Nieuwe runner-backend in de ASR pool (naast bestaande FasterWhisper)
- Configureerbaar per runner-slot (niet alle slots hoeven dezelfde engine te draaien)
- A/B vergelijking: kwaliteit TensorRT vs FasterWhisper output valideren

## Risico's

- **Compilatiecomplexiteit**: model moet vooraf gecompileerd worden voor specifieke GPU-architectuur (compute capability)
- **Modelupdates**: bij nieuwe Whisper-versie moet opnieuw gecompileerd worden
- **Kwaliteitsverschil**: TensorRT kan door kwantisering marginaal andere output geven — moet gevalideerd worden
- **Timestamp-support**: TensorRT-variant in WhisperLive retourneert geen timestamps per segment — vereist extra werk voor alignment

## Referentie

- WhisperLive TensorRT backend: `whisper_live/backend/trt_backend.py`
- NVIDIA TensorRT-LLM Whisper: https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/whisper
