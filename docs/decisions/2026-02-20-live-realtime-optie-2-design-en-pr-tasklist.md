# Decision Note: Live Realtime Transcriptie (Optie 2)

Date: 2026-02-20
Status: In uitvoering

## Implementatiestatus

- PR-01: Gereed en handmatig gevalideerd op 2026-02-20 (session create/ws control flow/cleanup OK).
- PR-02: Gereed en handmatig gevalideerd op 2026-02-20 (LiveView UI + mock eventflow OK; WS proxy-limitatie verwacht tot PR-06).
- PR-03: Gereed en handmatig gevalideerd op 2026-02-20 (partial/final/stats/pong/ended + session cleanup 404 OK).
- PR-04: Handmatig gevalideerd op 2026-02-21 voor UI/capture-control flow (mic capture + PCM16 streaming + start/pause/resume/stop); echte mic-transcriptie op specifieke Bluetooth headset/dongle combinatie blijft hardware-/profielafhankelijk.
- PR-05: Handmatig gevalideerd op 2026-02-21 (backend stabilisatie + richer eventcontract + final transcript endpoint; gevalideerd via mock flow en backend smoke-tests).
- PR-06: Handmatig gevalideerd op 2026-02-21 (dev proxy + systemd service; WS pass-through + host-forwarding via dev URL/tunnel OK).
- PR-07: In uitvoering (backend hardening/metrics/capacity-policy geimplementeerd; frontend editor-handoff + volledige QA matrix nog open).

## Context

Doel is een nieuwe sidebar-functie "Live recording" met zo realtime mogelijke transcriptie en live tekstweergave in het rechterpaneel van de SPA.

Bestaande situatie:
- Frontend werkt nu met upload + polling.
- Backend werkt nu met batch jobs (file-based queue + worker).
- Er bestaat nog geen streaming endpoint of microfoonflow.
- Dev frontend proxy ondersteunt nu geen WebSocket upgrade.

## Besluit

We implementeren echte realtime transcriptie via een nieuw live-pad naast de bestaande batch-pipeline:

1. Nieuwe LiveView in de frontend met microfoonbediening en live tekstvak.
2. Nieuwe live sessie-API met WebSocket voor audio in + transcript events uit.
3. Realtime transcriptie-engine via adapterlaag (start met WhisperLive sidecar).
4. Stabilisatie van partial/final tekst op de server.
5. WebSocket-capabele dev proxy, zodat dev URL/tunnel gelijk kan blijven.

De batch upload/editor flow blijft onaangetast.

## Architectuur

Frontend:
- Sidebar item `live`.
- Nieuwe `LiveView` met:
- `Start`, `Pause`, `Resume`, `Stop`.
- Status (connected/listening/processing/error).
- Rechterpaneel met `final` tekst + aparte `partial` regel.
- Optioneel na stop: "Open in editor".

Backend:
- Nieuwe live sessie manager in `portal-api`.
- HTTP endpoint voor sessie creatie.
- WebSocket endpoint per sessie.
- Audio ingest queue per sessie.
- Realtime decode worker per sessie.
- Transcript state machine (partial/final commit).

Engine:
- Adapterlaag zodat implementatie wisselbaar is.
- Startdriver: WhisperLive sidecar.
- Later mogelijk: native faster-whisper driver onder hetzelfde adaptercontract.

Infra:
- Dev frontend proxy vervangen of uitbreiden voor WS pass-through.
- Systemd dev service bijwerken naar nieuwe proxy-implementatie.

## Protocol Contract (MVP)

HTTP:
- `POST /api/demo/live/sessions`
- Response: `session_id`, `ws_url`, basisconfig.

WebSocket:
- `GET /api/demo/live/sessions/{session_id}/ws`
- Binary frames: PCM16 mono 16kHz LE.
- Client control JSON: `start`, `pause`, `resume`, `stop`, `ping`.
- Server JSON events: `ready`, `partial`, `final`, `stats`, `error`, `ended`.

Event velden minimaal:
- `type`
- `session_id`
- `seq`
- `text`
- `t0_ms`
- `t1_ms`

## Realtime kwaliteitsstrategie

Doel is lage latency met stabiele output:
- Sliding window decode met overlap.
- Partial updates op hoge frequentie.
- Final commit met stabiliteitsdrempel (bijv. meerdere consistente hypotheses).
- Geen diarization/topics in live pad.
- Optionele finalize pass naar SRT bij stoppen.

## Resource en veiligheid

Resource policy:
- Start met maximaal 1 actieve live sessie.
- Nieuwe sessie weigeren met expliciete foutcode als slot bezet is.

Veiligheid:
- Sessies hebben TTL en cleanup.
- Input validatie voor control messages.
- Backpressure op inbound audio queue.
- Graceful close bij client disconnect.

## PR-Level Tasklist

### PR-01: Live API skeleton + protocol

Scope:
- Introduceer sessiecreatie endpoint.
- Introduceer WebSocket endpoint met handshake.
- Voeg protocol types en event helpers toe.
- Nog geen echte ASR; tijdelijke echo/placeholder events.

Hoofdbestanden:
- `portal-api/main.py`
- `portal-api/live_protocol.py` (nieuw)
- `portal-api/live_sessions.py` (nieuw)

Acceptatie:
- Frontend kan sessie openen en WS verbinden.
- `ready` en `ended` events werken.
- Disconnect cleanup werkt zonder process leak.

### PR-02: Frontend LiveView shell + sidebar route

Scope:
- Nieuwe sidebar actie `live`.
- Nieuwe `LiveView` met controls, status, rechter tekstvak.
- WS client lifecycle zonder echte microfoonstream (test met mock control messages).

Hoofdbestanden:
- `/var/www/omniscripta-app/index.html`
- `/var/www/omniscripta-app/js/app.js`
- `/var/www/omniscripta-app/js/components/LiveView.js` (nieuw)
- `/var/www/omniscripta-app/css/style.css`
- `/var/www/omniscripta-app/css/layout.css`

Acceptatie:
- Navigatie naar LiveView werkt op desktop en mobiel.
- UI reageert correct op `ready/error/ended`.
- Geen regressie in upload/editor/settings routes.

### PR-03: Realtime engine adapter + WhisperLive driver

Scope:
- Voeg engine adaptercontract toe.
- Implementeer WhisperLive driver.
- Koppel live sessieworker aan adapter.
- Basic stats events (`lag`, `decode`, `rtf`).

Hoofdbestanden:
- `portal-api/live_engine_adapter.py` (nieuw)
- `portal-api/live_transcriber.py` (nieuw)
- `portal-api/live_sessions.py`
- `config/service.json` (live config keys)

Acceptatie:
- Inkomende audio levert `partial` events.
- Stop levert nette flush en `ended`.
- Bij engine failure volgt expliciete `error` event.

### PR-04: Browser microfoon capture + PCM streaming

Scope:
- Microfoon capture via `AudioWorklet` (primary).
- Fallback pad indien worklet niet beschikbaar.
- Encode naar PCM16 mono 16kHz en stream via WS binary frames.
- Start/pause/resume/stop bediening volledig end-to-end.

Hoofdbestanden:
- `/var/www/omniscripta-app/js/components/LiveView.js`
- `/var/www/omniscripta-app/js/services/LiveAudioService.js` (nieuw)
- `/var/www/omniscripta-app/js/services/LiveSessionService.js` (nieuw)

Acceptatie:
- Spraak verschijnt als partial tekst binnen doel-latency.
- Pause/resume stopt en hervat audioframes correct.
- Permission denied flow toont bruikbare foutmelding.

### PR-05: Transcript stabilisatie + final commit model

Scope:
- Server-side stabilisatielogica voor partial naar final.
- Sequence numbering en dedupe.
- Beter eventcontract voor UI merge.
- Optie om finale transcript op stop op te vragen.

Hoofdbestanden:
- `portal-api/live_transcriber.py`
- `portal-api/live_protocol.py`
- `portal-api/main.py`

Acceptatie:
- Final tekst is stabiel en groeit monotonic.
- Partial overschrijft alleen lopende tail.
- UI hoeft niet te "springen" bij herzieningen.

### PR-06: Dev proxy WebSocket support + service update

Scope:
- Maak dev frontend proxy WS-capabel of vervang met WS-capabele reverse proxy.
- Houd bestaande dev test URL op poort 8010 in stand.
- Update systemd dev frontend service.

Hoofdbestanden:
- `deploy/dev_frontend_proxy.py`
- `deploy/systemd/transcribe-frontend-dev.service`

Acceptatie:
- `ws://127.0.0.1:8010/api/...` werkt via proxy.
- HTTP API routes blijven werken.
- Geen regressie in static file serving.

### PR-07: Hardening, metrics, QA, editor handoff

Scope:
- Metrics/logging uitbreiden.
- Session limits en timeout policy finaliseren.
- End-to-end tests + handmatige testmatrix.
- Knop "Open in editor" na stop (live transcript als SRT laden).

Hoofdbestanden:
- `portal-api/live_sessions.py`
- `portal-api/live_transcriber.py`
- `/var/www/omniscripta-app/js/components/LiveView.js`
- `/var/www/omniscripta-app/js/editor.js` (alleen handoff integratie)

Acceptatie:
- 30 min sessie zonder memory leak of zombie worker.
- Heldere foutcodes voor busy/timeout/disconnect.
- Overname naar editor werkt met bruikbare transcript output.

Huidige invulling (2026-02-20):
- Capacity-policy default aangescherpt naar 1 actieve live sessie (overridable via `TRANSCRIBE_LIVE_MAX_SESSIONS`).
- Nieuw metrics endpoint `GET /api/demo/live/metrics` voor actieve/archived sessie counters + limieten.
- Session create bij capacity geeft nu expliciete `429` detail met `code` en `message`.
- Nog te doen binnen PR-07: frontend knop "Open in editor" + volledige handmatige QA matrix en soak-run verslag.

Known issue / tijdelijke afbakening (2026-02-21):
- Bij live opname in een Chromium-tab kan dezelfde browser-instance lokaal minder responsief worden in andere tabs.
- Dit wordt voorlopig behandeld als lokaal client-performance issue (renderer/load), geen blocker voor server-side webservice gedrag.
- Acceptatie voor MVP blijft gericht op functionele correctness over devices/sessies; performance-optimalisatie van deze lokale browserinteractie volgt als aparte hardening-actie.

Known issue / hardware-afhankelijke afbakening (2026-02-21):
- Specifieke Bluetooth headset-configuraties (o.a. EPOS ADAPT 560 zonder originele dongle, via generieke BT-dongle) kunnen intermitterende of zeer lage microfooninput leveren in Linux browser capture.
- Dit is beoordeeld als audio profiel/driver/hardware pad issue (client-side), niet als blocker voor backend/proxy implementatie of protocolflow.
- Voor functionele validatie van live transcriptie wordt tijdelijk `mock_stream` ondersteund via URL-toggle (`?live_recording_mock=1`) en kan echte transcriptie getest worden met andere mic-hardware (USB/laptopmic/proprietary dongle).

Werkend profiel (2026-02-23, dev validatie op Linux + Chromium + Nedis USB mic):
- Echte live transcriptie werkt betrouwbaar met `?live_recording_mock=0&live_audio_dsp=0` (browser DSP uit).
- Root cause van eerdere slechte transcriptie/hallucinaties bleek vooral browser capture DSP (`echoCancellation`, `noiseSuppression`, `autoGainControl`) in combinatie met deze testsetup/hardware.
- PipeWire/Chromium routing is gevalideerd op actieve Nedis-source (`alsa_input.usb-C-Media_Electronics_Inc._nedis_MICCU100BK-00.mono-fallback`, state `RUNNING` tijdens opname).
- `arecord` lokale mic-test (`S16_LE`, `16kHz`, mono) klonk helder; daarmee is microfoonhardware + OS inputpad bevestigd.
- Dev sidecar tuning die voor deze setup aantoonbaar beter werkt (tijdelijk "working profile", niet generieke default):
- `live.whisperlive_sidecar.language = "en"`
- `live.whisperlive_sidecar.input_gain = 8.0`
- `live.whisperlive_sidecar.use_vad = false`
- `live.whisperlive_sidecar.no_speech_thresh = 0.45`
- `live.whisperlive_sidecar.recv_timeout_s = 0.02`
- `live.whisperlive_sidecar.flush_timeout_s = 45.0`
- `live.whisperlive_sidecar.max_drain_messages = 64`
- Bij cleanup/generalizatie deze waarden niet stilzwijgend terugzetten zonder her-test op dezelfde hardware; markeer dit expliciet als hardware-/browser-afhankelijk profiel.

## Acceptatiecriteria op productniveau

MVP is geslaagd als:
- Eerste partial tekst gemiddeld binnen 1.5s verschijnt.
- Tijdens spraak minstens elke 500ms update binnenkomt.
- Na stilte wordt binnen 2.0s final tekst gecommit.
- Bestaande upload/editor flow functioneel onveranderd blijft.

## Uitrolvolgorde

1. PR-01
2. PR-02
3. PR-03
4. PR-04
5. PR-05
6. PR-06
7. PR-07

## Open beslissingen (voor start implementatie)

1. Engine start: WhisperLive sidecar bevestigen als MVP-driver.
2. Proxypad: bestaande Python proxy upgraden of vervangen door aparte reverse proxy.
3. Stop-flow: direct "Open in editor" standaard aan of optioneel via aparte knop.
