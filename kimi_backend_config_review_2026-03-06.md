# Code Review: Kimi backend config-migratie

Datum: 2026-03-06  
Scope: actuele unstaged changes in deze backend worktree

## Beoordeelde wijzigingen
- `.gitignore`
- `shared/app_config.py` (nieuw)
- `config/settings.json` (nieuw)
- `config/local.json.example` (nieuw)
- `portal-api/main.py`
- `portal-api/queue_fs.py`
- `asr-pool/main.py`
- `worker/asr_client_remote.py`
- `worker/pipeline_live_chunk.py`
- `worker/worker_daemon.py`

## Findings (op ernst)

### 1) **High**: dev kan onbedoeld naar live ASR-pool routen
- Bestanden:
  - `config/settings.json:38`
  - `worker/asr_client_remote.py:79`
- Probleem:
  - Default `asr_pool.base_url` staat op `http://127.0.0.1:8090` (live).
  - `worker/asr_client_remote.py` leest primair deze config key.
  - Als `dev.env` wordt opgeschoond (alleen secrets) en `config/local.json` ontbreekt, gaat de dev worker naar live ASR.
- Impact:
  - Cross-environment verkeer, onbedoelde live load/vervuiling.
- Advies:
  - Voor dev-safe default `asr_pool.base_url` in `settings.json` naar `http://127.0.0.1:18090`, of hard fail als running mode dev is en local override ontbreekt.

### 2) **High**: legacy env-compatibiliteit is deels gebroken door nieuwe naammapping
- Bestanden:
  - `shared/app_config.py:107`
  - `portal-api/main.py:40, 47-90`
  - `asr-pool/main.py:139-168`
- Probleem:
  - `get_setting()` mapt path `x.y.z` naar env `TRANSCRIBE_X_Y_Z`.
  - Dat matcht niet altijd bestaande legacy env-namen (bijv. `TRANSCRIBE_ROOT_PATH`, `TRANSCRIBE_ASR_POOL_TIMEOUT_INTERACTIVE_S`, etc.).
  - In meerdere modules is expliciete fallback op legacy env verwijderd.
- Impact:
  - Bestaande unit env kan stilzwijgend niet meer doorwerken.
- Advies:
  - Voeg expliciete alias mapping toe voor legacy keys die operationeel nog bestaan, of migreer alle unit env’s atomisch plus verificatie.

### 3) **Medium**: config key-mismatch maakt work_root instelling effectief dood
- Bestanden:
  - `asr-pool/main.py:175`
  - `config/settings.json:111-115`
  - `config/local.json.example:26-28`
- Probleem:
  - Code leest `asr_pool.work_root`, maar config gebruikt `paths.work_root`.
  - Daardoor wordt `paths.work_root` nergens toegepast.
- Impact:
  - Verwachte configuratie werkt niet; pad valt terug op default.
- Advies:
  - Kies 1 canonieke key en gebruik die consequent in code + settings + example.

### 4) **Medium**: voorbeeldconfig bevat niet-bestaande key voor rolling poll interval
- Bestand:
  - `config/local.json.example:18`
- Probleem:
  - Voorbeeld gebruikt `live.rolling_poll_interval_ms`.
  - Code verwacht `live.rolling.poll_interval_ms` (`portal-api/main.py:63`).
- Impact:
  - Operators denken dat override actief is terwijl die genegeerd wordt.
- Advies:
  - Corrigeer example naar geneste structuur `live.rolling.poll_interval_ms`.

### 5) **Medium**: queue base configuratie hangt aan `janitor.jobs_base`
- Bestand:
  - `portal-api/queue_fs.py:27`
- Probleem:
  - Job queue root wordt nu gelezen uit janitor namespace.
  - Functioneel werkt dit, maar semantisch koppelt het queue pad en janitor-instellingen hard aan elkaar.
- Impact:
  - Moeilijker onderhoud/naamgeving; grotere kans op verkeerde aanname.
- Advies:
  - Gebruik dedicated key, bv. `jobs.base`, en laat janitor daar expliciet naar verwijzen.

### 6) **Low**: dode legacy helpers in `asr-pool/main.py`
- Bestand:
  - `asr-pool/main.py:72-95`
- Probleem:
  - `_cfg_int/_cfg_str/_cfg_bool` zijn niet meer in gebruik.
- Impact:
  - Ruis; verwarring over echte configuratiepad.
- Advies:
  - Verwijderen of direct inzetten als expliciete legacy-aliaslaag.

### 7) **Low**: “secrets only in env” doel wordt niet afgedwongen
- Bestand:
  - `shared/app_config.py:106-110`
- Probleem:
  - Alle `TRANSCRIBE_*` env vars overrulen config, niet alleen secrets.
- Impact:
  - Toevallige envs kunnen gedrag onverwacht wijzigen.
- Advies:
  - Beperk env-overrides tot allowlist (tokens/keys), of documenteer expliciet dat env altijd hoogste prioriteit heeft.

## Positieve punten
- Config centralisatie is duidelijker dan verspreide `os.getenv`.
- Type helpers (`get_int/get_float/get_bool`) verminderen parse-fouten.
- `config/local.json` staat netjes in `.gitignore`.
- Syntax check van gewijzigde Python-bestanden is groen (`python3 -m py_compile`).

## Aanbevolen vervolgstappen (kort)
1. Fix High finding #1 (dev-safe routing) direct.
2. Beslis over legacy env-policy en implementeer #2 (alias of harde migratie).
3. Repareer key-mismatches (#3, #4) in dezelfde patch.
4. Eventueel cleanup (#5-#7) in vervolgpacth.
