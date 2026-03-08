# Code Review: ASR Pool Worker Redesign v3 - Fase 0 & Fase 6 (Cleanup/Afronding)

Ik heb de iteratie voor Fase 0 (Voorbereiding en guardrails) en Fase 6 (Opschonen en afronden) grondig bekeken. De wijzigingen komen prachtig overeen met de vastgestelde v3-scope en de actiepunten uit het plan `asr_pool_worker_implementation_plan_v3.md` zijn nauwgezet gevolgd.

Hieronder vind je mijn bevindingen en kritische review van deze afrondende stap.

## Wat gaat er goed (Positieve bevindingen)

1. **Grondige opschoning van oude code (Phase 6):** 
   - De bestanden `worker/phase_whisperx.py` en `worker/pipeline_live_chunk.py` zijn succesvol verwijderd. Dit ruimt de codebase aanzienlijk op en verwijdert dode code die na de migratie naar de v3 submit/reap pool archtectuur niet meer nodig was.
   - De oude fallback- / compatibiliteitslagen (zoals `done_fallback_srt` en `done_empty_chunk` statussen via speaker_lines validatiehacks) in `live_chunk_transcribe.py` zijn verwijderd wat de logica aanzienlijk eenvoudiger maakt.

2. **Expliciete Guardrails & Documentatie (Phase 0):**
   - De `TODO(v3-followup)` comments voor "restart recovery" zijn netjes geplaatst in zowel `asr-pool/main.py` als `worker/worker_daemon.py`.
   - In de code is goed gedocumenteerd (en hardcoded afgedwongen) dat `live_lane` op `"single"` blijft staan (onder andere met het expliciete commentaar `# v3 scope: one hardcoded live lane.`).
   
3. **Consistente Opschoning Systemen:**
   - De wijziging aan `deploy/jobs_janitor.py` (toevoegen van de `"superseded"` TTL cleanup configuratie via `asr_pool.records.ttl_superseded_s`) sluit naadloos aan op de architectonische keuzes in Fase 4 zodat ook deze events netjes opgeruimd worden.
   - Config-veranderingen in `config/settings.json` (waaronder de vervanging van `require_single_inflight` met een concretere limiet instelling `max_outstanding_per_session`) reflecteren de nieuwe mogelijkheden van het portal-design (Fase 5) accuraat in de gehele context.

## Kritische Opmerkingen / Aandachtspunten

Deze fase was in essentie een cleanup-operatie passend bij de gestelde afspraken. De focus lag op het verwijderen van overblijfsels en er zijn geen grote nieuwe constructies geïntroduceerd. 

Een paar kleine aandachtspunten (voor in de toekomst, géén blockers voor deze PR):
- **Jobs Janitor completeness:** Zorg dat eventueel achtergebleven job directory cleanup via de janitor correct matcht op `superseded` in zowel de actieve db rows en eventueel de job files op disk, of monitor even op de live-omgeving of deze inactieve/superseded runs daar niet als weesbestanden (orphans) achterblijven na verloop van tijd.
- **Dode properties configuratie:** Dubbelcheck de komende tijd of backend deployments niet per ongeluk omvallen over obsoleet geworden JSON-settings in custom configuration configs (als ze ontbreken ten opzichte van default settings).

## Conclusie

Fase 0 en Fase 6 zijn strak en netjes afgerond en laten de codebase in een duidelijk betere staat (+ cleaner architecture) achter vergeleken met de start van deze redesign v3. 

**Status:** Goedgekeurd (Approved). De backend afronding is hiermee succesvol.
