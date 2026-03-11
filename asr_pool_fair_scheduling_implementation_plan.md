# ASR Pool Fair Scheduling Implementatieplan

Doel: meerdere live-sessies eerlijk schedulen wanneer ze ASR-slots delen, zodat "winner-takes-most" gedrag verdwijnt en kwaliteit gelijkmatiger blijft per sessie.

## Statusupdate 2026-03-11
- Geïmplementeerd:
  - Fase 1 (session-aware fairness state in asr-pool)
  - Fase 2 (submit pad gekoppeld aan fairness state, incl. non-live pseudo-session)
  - Fase 3 (fair dequeue: round-robin over interactive sessies, FIFO binnen sessie)
- Gevalideerd met synthetische smoke:
  - 2 sessies, 2 requests per sessie (gegroepeerd submit-order A,A,B,B):
    - start-order: `A1, B1, A2, B2`
  - 4 sessies, 2 requests per sessie (gegroepeerd submit-order A,A,B,B,C,C,D,D):
    - start-order: `A1, B1, C1, D1, A2, B2, C2, D2`
- Conclusie: fairness-gedrag is bevestigd (geen FIFO-per-session dominantie meer in `interactive` queue).

### Fase 6 validatie - echte 4x inject fixture run (dev)
- Uitvoering: 4 gelijktijdige inject-runs van fixture `panel_120s_v1` (sessies `S1..S4`), op 1 ASR-slot.
- Sessies:
  - `live_20260311T225528Z_719e8af2`
  - `live_20260311T225528Z_31eb026b`
  - `live_20260311T225528Z_9e20bd6f`
  - `live_20260311T225528Z_a9cab808`
- Resultaten:
  - scores: `90, 91, 91, 92` (mean `91`, spread `2`)
  - `asr_transcribe_pct_of_recording`: `7.5 - 7.7`
  - `chunks_done/chunks_total`: overal `16/16`
  - `chunks_failed`: overal `0`
  - `hard_clip_dropped_audio_ms`: overal `0`
- Conclusie: onder 4 gelijktijdige sessies is geen winner-takes-most patroon zichtbaar; sessieresultaten liggen dicht bij elkaar.

## Scope en uitgangspunten
- Bottleneck-first: fairness wordt in `asr-pool` geïmplementeerd (niet primair in worker of portal-api).
- Geen fallback-/compatibiliteitslagen toevoegen.
- Geen wijziging aan protocol tussen worker en asr-pool nodig.
- Huidige submit+reap architectuur blijft behouden.
- Live lane blijft hardcoded `single`.

## Fase 0 - Baseline en meetkader
### Taak 0.1 - Meetset vastleggen
- Leg baseline vast met:
  - 2 gelijktijdige inject fixtures op 1 slot
  - 4 gelijktijdige inject fixtures op 1 slot
- Per sessie noteren:
  - fixture score
  - asr_transcribe_pct
  - chunks_done/chunks_total
  - hard_clip_dropped_audio_ms

Acceptatiecriteria:
- Baseline meetnotitie is aanwezig in een markdown-bestand.
- Voor elke sessie zijn de 4 kernmetrics ingevuld.

## Fase 1 - Fairness datastructuren in asr-pool
### Taak 1.1 - Session-aware queue bookkeeping toevoegen
- In `asr-pool/main.py` datastructuren toevoegen voor live interactive fairness, bijvoorbeeld:
  - `live_rr_order` (ring/list met actieve `live_session_id`)
  - `live_rr_cursor` (index/pointer)
  - `live_session_pending_count`
  - `live_session_request_ids` (of equivalente mapping)

Acceptatiecriteria:
- Datastructuren bestaan en initialiseren correct bij pool start.
- Geen functionele regressie voor upload/background paden.

## Fase 2 - Submit pad koppelen aan fairness state
### Taak 2.1 - Live interactive submits registreren
- Bij `submit` van live interactive request:
  - `live_session_id` in fairness state registreren
  - pending count verhogen
  - session in RR-order opnemen als nieuw

Acceptatiecriteria:
- Elke geaccepteerde live request is terug te vinden in session bookkeeping.
- Pending count per sessie klopt bij oplopende load.

## Fase 3 - Fair dequeue algoritme
### Taak 3.1 - Round-robin selectie voor live interactive
- Dequeue van interactive live requests wijzigen van puur FIFO naar round-robin over sessies.
- Binnen een sessie FIFO behouden.
- Non-live interactive requests expliciet beleid geven:
  - ofwel apart pad
  - ofwel als eigen pseudo-session

Acceptatiecriteria:
- Bij 4 actieve sessies wordt beurtverdeling zichtbaar cyclisch in logs.
- Geen structurele dominantie van 1 sessie bij gelijke inputbelasting.

## Fase 4 - State lifecycle correctheid
### Taak 4.1 - Consistentie bij terminal transitions
- Bij `completed|failed|cancelled`:
  - pending count verlagen
  - request uit per-session mapping verwijderen
  - lege sessie uit RR-order verwijderen
- Bij cancel van queued requests hetzelfde pad gebruiken.

Acceptatiecriteria:
- Geen stale session ids in RR-order.
- Pending counts kunnen niet negatief worden.
- Queue leeg => RR-order leeg.

## Fase 5 - Observability en diagnose
### Taak 5.1 - Fairness events/logging
- Logs/events uitbreiden met fairness velden, bijv.:
  - `rr_pick_session_id`
  - `rr_cursor_before/after`
  - `session_pending_before/after`
  - `queue_depth_per_session` (samengevat)

Acceptatiecriteria:
- Uit logs is de keuze van volgende request per dequeue uitlegbaar.
- Bij incidentanalyse kan starvation direct worden vastgesteld.

## Fase 6 - Validatie op gedrag
### Taak 6.1 - Reproduceer testscenario's
- Herhaal baseline scenario's uit Fase 0.
- Vergelijk voor/na op fairness en kwaliteit.

Acceptatiecriteria:
- Geen 90/50/50/50 patroon meer bij 4 sessies op 1 slot.
- Scores liggen zichtbaar dichter bij elkaar.
- Hard clips niet structureel geconcentreerd op dezelfde sessie.

## Fase 7 - Documentatie en rollout
### Taak 7.1 - Docs en runbook updaten
- Werk relevante docs bij met:
  - fairness policy
  - nieuwe loggingvelden
  - testprocedure

Acceptatiecriteria:
- Documentatie reflecteert daadwerkelijke runtime logica.
- Team kan fairness gedrag reproduceren en controleren met runbook-stappen.

## Open beslissingen (kort)
1. Algoritme: RR of DRR
- Voorstel nu: RR (laag risico, kleinste verdedigbare stap).
- Later uitbreiden naar DRR als request-kosten sterk variëren.

2. Non-live interactive requests in fairness model
- Voorstel nu: als aparte pseudo-session zodat live fairness niet breekt.

3. Feature flag
- Voorstel nu: geen flag (directe vervanging), conform afgesproken "geen fallback paden".

## Risico's
- Foutieve state cleanup kan session-ring vervuilen.
- Onvolledige observability maakt tuning lastig.
- Fairness zonder goede cancel/terminal verwerking kan alsnog starvation veroorzaken.

## Niet in scope
- Worker-side fairness scheduler.
- SSE/protocol wijzigingen.
- Herontwerp van ASR-profielen of slot-allocation policy.
