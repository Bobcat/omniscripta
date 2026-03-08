# Code Review: ASR Pool Worker Redesign v3 - Fase 2 (Upload Submit/Reap)

Ik heb de code wijzigingen voor Fase 2 (en deels Fase 6) grondig bekeken. De wijzigingen komen overeen met de actiepunten in het plan: `asr_pool_worker_implementation_plan_v3.md` is netjes bijgewerkt en de logica in de code reflecteert de doelen.

Hieronder vind je mijn bevindingen en kritische review van deze iteratie.

## Wat gaat er goed (Positieve bevindingen)

1. **Heldere scheiding van worker modes:** De toevoeging van `_run_live_worker_submit_reap` en `_run_upload_worker_submit_reap` als losse `main` entrypunten op basis van `TRANSCRIBE_WORKER_MODE` is een grote verbetering. Het voorkomt enorme `if/else` blokken in de hoofdloop en maakt de code overzichtelijker.
2. **Succesvolle integratie van Submit/Reap voor uploads:** Het upload pad gebruikt nu succesvol een `pending` map. In plaats van een lange blokkerende request of legacy waithulpen (die terecht verwijderd zijn), leest de worker nu het `/asr/v1/completions` endpoint uit om terminal events (zoals `done` of `error`) binnen te halen.
3. **Code opschoning:** Het verwijderen van het parallelle fallback pad en verouderde "wait on request" helpers vereenvoudigt de `worker_daemon.py` aanzienlijk.
4. **Faserings-behoud in Finalize:** Het is goed om te zien dat de originele fasering (zoals het aanmaken van `speaker_lines`, chunking manifest en LLM topics validatie) grotendeels herbruikbaar bleek en succesvol naar `_finalize_upload_job_terminal` is gemigreerd.

## Kritische Opmerkingen / Aandachtspunten

Hoewel de huidige opzet prima werkt onder de huidige requirements (`max_outstanding=1`), zijn er nog een paar aandachtspunten met het oog op eventuele toekomstige schaalbaarheid:

1. **Synchrone LLM/Postprocessing taken in de Finalize-fase:**
   De afronding van een upload taak gebeurt in `_finalize_upload_job_terminal`. Daarbinnen worden functies aangeroepen zoals (het genereren van speaker_lines, LLM/topics processing en validaties). Dit zijn synchrone operaties. Omdat dit vanuit de hoofdloop (`_run_upload_worker_submit_reap`) draait, blokkeert dit de verdere afhandeling van de completion feed. 
   - **Gevolg:** Zolang `max_outstanding=1` is, is dit geen probleem (de worker mag nergens anders mee bezig zijn).
   - **Risico:** Mocht dit later worden opgeschaald (`max_outstanding > 1`), dan blockt één upload taak in de LLM-fase direct het subitten of in ontvangst nemen van een compleet andere upload taak. Houd dit in het achterhoofd voor toekomstige iteraties.
   
2. **`time.sleep()` Idle-loop in de Submit/Reap:**
   Net als bij de e eerdere live-worker, maakt de upload worker gebruik van een basis `time.sleep(...)` wanneer de `completions` feed via HTTP is uitgelezen en er geen werk is verzet (`did_work == False`). Ook hier is synchroon pollen via HTTP relatief zwaar vergeleken met websockets of Redis pub/sub. Je hebt dit al als v3-compromis genoteerd, dus dit is puur een confirmatie dat ik zie dat dit nog steeds de gekozen architectuur is.

3. **Geen lokale error-cleanup voor pending jobs:**
   Bij een ongecontroleerde crash van de upload worker, verdwijnen lopende job items uit de local memory `pending` list, terwijl deze in de ASR Pool nog bezig kunnen zijn of al ge-reapt. Een TODO ("`v3-followup: add restart recovery / re-request reconciliation for pending upload jobs`") is hiervoor netjes geplaatst.

## Conclusie

De code voor Fase 2 voldoet volledig aan de eisen uit het actieplan en is logisch ingedeeld. De gekozen oplossingen passen binnen de `v3` constraints (zoals HTTP polling integratie met `asr-pool`). 

**Status:** Goedgekeurd (Approved). We kunnen deze patches samen met het bijgewerkte plan als productieklaar beschouwen voor v3.
