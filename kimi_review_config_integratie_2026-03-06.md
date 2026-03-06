# Code Review: Config Integratie (service.json & whisperx.json)

**Datum:** 2026-03-06  
**Reviewer:** Kimi  
**Scope:** Integratie van service.json en whisperx.json in settings.json/local.json

---

## Samenvatting

De collega heeft uitstekend werk geleverd. De integratie is technisch correct uitgevoerd met goede aandacht voor backwards compatibility en namespace organisatie.

**Status:** ✅ **LGTM** (Looks Good To Me) met minimale suggesties

---

## Positieve Punten

### 1. Schone Namespace Organisatie
- `asr_pool.whisperx.*` voor ASR model configuratie
- `snip.*`, `topics.*`, `tabby.*` overgenomen uit service.json
- Logische groepering gerelateerde instellingen

### 2. Goede Default Waarden
Alle defaults in `settings.json` zijn consistent met de oorspronkelijke waarden uit:
- `service.json` (snip, topics, tabby secties)
- `whisperx.json` (ASR model configuratie)

### 3. Thread Configuratie Structuur
De threads zijn netjes genest onder `asr_pool.whisperx.threads`:
```json
"threads": {
  "omp": 8,
  "mkl": 8,
  "torch": 8,
  "torch_interop": 1
}
```
Dit is leesbaarder dan de platte structuur in de oude whisperx.json.

### 4. Venv Pad in Config
`asr_pool.whisperx.venv` is toegevoegd - dit was eerder impliciet. Goed voor expliciete configuratie.

### 5. Code Aanpassingen
- `whisperx_runner_env.py` gebruikt nu correct de nieuwe config loader
- `_load_server_config()` leest uit `asr_pool.whisperx.*` namespace
- Worker daemon gebruikt `get_str()` helpers

---

## Bevindingen (Minimaal)

### 1. **Low**: Dubbele `work_root` verwijdering gemist
**Bestand:** `settings.json:43`

`asr_pool.work_root` staat al in settings.json (regel 43), maar eerder stond er ook een `paths.work_root` sectie die inmiddels is verwijderd. Dit is correct opgelost.

**Status:** ✅ Opgelost

---

### 2. **Low**: `prompt_path` aangepast
**Bestand:** `settings.json:148`

Oorspronkelijk in service.json:
```json
"prompt_path": "/srv/transcribe/prompts/simple_prompt5.txt"
```

Nu in settings.json:
```json
"prompt_path": "prompts/simple_prompt5.txt"
```

Dit is een functionele wijziging - het pad is relatief geworden. Zorg dat dit werkt in de deployment (pad wordt relatief aan repo root).

**Verdict:** ✅ Acceptabel - relatieve paden zijn beter voor portability

---

### 3. **Low**: `live_chunk_backend` default
**Bestand:** `settings.json:83`

Default is "whisperx", maar de code in `whisperx_runner_env.py:57-58` valt terug op "whisperx" als de waarde ongeldig is. Dit is defensief geprogrammeerd.

**Status:** ✅ Goed

---

## Code Review Specifieke Bestanden

### `asr-pool/whisperx_runner_env.py`

**Goed:**
- Import van `shared.app_config` helpers ✅
- `get_str()`/`get_int()`/`get_bool()` gebruik ✅
- Graceful fallback voor threads configuratie ✅
- `_load_server_config()` retourneert dict in verwacht formaat ✅

**Opmerking:**
De functie `_load_server_config()` wordt aangeroepen vanuit `whisperx_runner_client.py` en `whisperx_runner_server.py`. Beide bestanden hebben nu toegang tot de config via deze module.

### `worker/worker_daemon.py`

**Goed:**
- Gebruikt nu `get_str()` voor pad configuratie ✅
- `_resolve_cfg_path()` helper is schoon ✅
- `_load_service_config()` nog steeds beschikbaar voor topics/snip config ✅

---

## Test Aanbevelingen

1. **Restart services** en controleer of waarden correct laden:
   ```bash
   systemctl --user restart transcribe-asr-pool-dev.service
   curl -s http://127.0.0.1:18090/asr/v1/pool | jq '.slots_total'
   # Zou 1 moeten zijn (uit local.json)
   ```

2. **Test whisperx config loading**:
   ```bash
   # In whisperx venv
   python3 -c "from whisperx_runner_env import _load_server_config; print(_load_server_config())"
   ```

3. **Test topics pipeline**:
   - Upload een test audio
   - Controleer of topics extractie werkt (gebruikt `topics.*` config)

---

## Conclusie

De integratie is **goed uitgevoerd**. Alle configuratie is nu centraal beheerd via:
- `config/settings.json` - defaults
- `config/local.json` - workspace overrides
- `~/.config/transcribe/dev.env` - alleen secrets

De oude files `service.json` en `whisperx.json` kunnen worden verwijderd na goedkeuring van deze review.

**Aanbevolen acties:**
1. ✅ Merge deze wijzigingen
2. 🔄 Test op dev omgeving
3. 🗑️ Verwijder oude `config/service.json` en `config/whisperx.json`
4. 🔄 Update documentatie

---

**Reviewer:** Kimi  
**Datum:** 2026-03-06  
**Status:** ✅ Approved
