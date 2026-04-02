---
description: Canonical workflow for Omniscripta frontend + Transcribe backend (live and dev)
---

# Omniscripta + Transcribe Workflow

## Quickstart (Daily)

1. If frontend changed, deploy to dev static target:
```bash
/var/www/omniscripta-app/deploy-dev.sh
```

2. Verify server-side dev services on `dc1`:
```bash
systemctl --user --no-pager status transcribe-api-dev.service transcribe-asr-pool-dev.service transcribe-frontend-dev.service asr-worker-live-dev@1.service asr-worker-batch-dev@1.service llm-worker-dev@1.service
```

3. If you changed backend code and want the full dev stack fresh:
```bash
~/projects/transcribe-dev/deploy/dev_restart_all.sh
```

4. On your other computer, use one of these URLs:
- Live backend path: `http://localhost:8080/index.html`
- Dev backend path: `http://127.0.0.1:18010/index.html`

## 1) Repositories and Roles

1. **Frontend source repo**: `/var/www/omniscripta-app`
   - Edit UI/UX and editor logic here (`index.html`, `js/*`, `css/*`).
   - Deploy scripts live here: `deploy.sh` (live), `deploy-dev.sh` (dev static target).

2. **Live repos on `dc1`**
   - `/srv/transcribe` - production Omniscripta backend/API, LLM worker, deploy scripts, static files.
   - `/srv/asr-pool` - production ASR pool repo checkout.
   - `/srv/asr-worker` - production ASR worker repo checkout.
   - Treat all three as live environment; avoid direct development changes there.

3. **Backend dev worktree (persistent)**: `~/projects/transcribe-dev`
   - Safe API/LLM/backend development environment.
   - Separate data/config paths from live.

4. **Standalone ASR pool repo**: `~/projects/asr-pool-dev`
   - Source for dev ASR pool service (`transcribe-asr-pool-dev.service`).
   - Keep ASR pool runtime/config changes here.

5. **Standalone ASR worker repo**: `~/projects/asr-worker-dev`
   - Source for dev worker services (`asr-worker-live-dev@1.service`, `asr-worker-batch-dev@1.service`).
   - Keep ASR worker runtime/config changes here.

## 2) Environments

### Live
- Omniscripta root: `/srv/transcribe`
- ASR pool root: `/srv/asr-pool`
- ASR worker root: `/srv/asr-worker`
- API service: `transcribe-api.service`
- ASR pool service: `transcribe-asr-pool.service` (127.0.0.1:8090)
- Worker service: `asr-worker-live.service` (ops on 127.0.0.1:28110)
- Worker service: `asr-worker-batch.service` (ops on 127.0.0.1:28111)
- LLM worker service: `llm-worker.service`
- API/LLM env file: `/etc/transcribe/transcribe.env`
- ASR pool env file: `/etc/asr-pool/asr-pool.env`
- ASR worker env file: `/etc/asr-worker/asr-worker.env`
- Tabby tunnel service: `transcribe-tabby-tunnel.service`
- Frontend static target: `/srv/transcribe/static` (served by nginx)
- Worker queue root: `/srv/transcribe/data/jobs/upload_worker`
- API `/ops` on live uses the local systemd drop-in `/etc/systemd/system/transcribe-api.service.d/ops-env.conf` so it points at pool `:8090` and worker ops `:28110` / `:28111`.
- Note: live ASR pool currently uses `/srv/asr-pool/config/local.json` to point the WhisperX runner at the shared `~/whisperx/.venv`.

### Dev
- API/LLM backend root: `~/projects/transcribe-dev`
- ASR pool backend root: `~/projects/asr-pool-dev`
- ASR worker backend root: `~/projects/asr-worker-dev`
- API service: `transcribe-api-dev.service` (127.0.0.1:8001)
- ASR pool service: `transcribe-asr-pool-dev.service` (127.0.0.1:18090)
- Worker service template: `asr-worker-live-dev@.service`
- Worker service template: `asr-worker-batch-dev@.service`
- LLM worker service template: `llm-worker-dev@.service`
- Janitor timer: `transcribe-demo-jobs-janitor-dev.timer`
- Frontend proxy service: `transcribe-frontend-dev.service` (127.0.0.1:8010)
- API/LLM/frontend env file: `~/.config/transcribe/dev.env`
- ASR pool env file: `~/.config/asr-pool/asr-pool.env`
- ASR worker env file: `~/.config/asr-worker/asr-worker.dev.env`
- Dev frontend static target: `~/projects/transcribe-dev/static`
- Note: dev API currently runs `uvicorn` from `/srv/transcribe/portal-api/.venv`, and `llm-worker-dev@.service` currently runs from `/srv/transcribe/worker/.venv`.

### Shared transport path coupling
- Canonical ASR pool transport module path (dev): `~/projects/asr-pool-dev/asr_pool_transport.py`
- Canonical ASR pool transport module path (live): `/srv/asr-pool/asr_pool_transport.py`
- `asr-worker` imports this module by path-coupled lookup order:
  1. `ASR_POOL_REPO_ROOT` (if set)
  2. `~/projects/asr-pool-dev`
  3. `/srv/asr-pool`
- Operational consequence:
  - Keep the ASR pool repo checkout present and up-to-date on hosts where API/worker Python runs, even if the pool service itself runs elsewhere.
- Deploy order consequence (when changing `asr_pool_transport.py`):
  1. Deploy/sync ASR pool repo checkout on consumer hosts first.
  2. Then restart consumer services (worker now; API later when it starts using this module).

## 3) Frontend Test Paths (both active)

1. **Live-backend test path**
   - URL on your other computer: `http://localhost:8080/index.html`
   - This path talks to the **live backend**.

2. **Dev-backend test path**
   - URL on your other computer: `http://127.0.0.1:18010/index.html`
   - This path talks to the **dev backend**.

## 4) Canonical Stack Control Scripts

From `~/projects/transcribe-dev`:

1. Dev full start:
```bash
~/projects/transcribe-dev/deploy/dev_start_all.sh
```

2. Dev full restart:
```bash
~/projects/transcribe-dev/deploy/dev_restart_all.sh
```

3. Live full start:
```bash
~/projects/transcribe-dev/deploy/live_start_all.sh
```

4. Live full restart:
```bash
~/projects/transcribe-dev/deploy/live_restart_all.sh
```

These scripts are the current canonical service sets and start order.

## 5) Deploy Scripts

From `/var/www/omniscripta-app`:

1. Live deploy:
```bash
/var/www/omniscripta-app/deploy.sh
```
- Builds frontend bundle and deploys to `/srv/transcribe/static`.

2. Dev deploy:
```bash
/var/www/omniscripta-app/deploy-dev.sh
```
- Builds frontend bundle and deploys to `~/projects/transcribe-dev/static`.

## 6) Required systemd Services

### On server (`dc1`) - user services

```bash
systemctl --user enable --now transcribe-api-dev.service
systemctl --user enable --now transcribe-asr-pool-dev.service
systemctl --user enable --now transcribe-frontend-dev.service
systemctl --user enable --now asr-worker-live-dev@1.service
systemctl --user enable --now asr-worker-batch-dev@1.service
systemctl --user enable --now llm-worker-dev@1.service
```

### On server (`dc1`) - live services (system scope)

```bash
sudo systemctl enable --now transcribe-api.service
sudo systemctl enable --now transcribe-asr-pool.service
sudo systemctl enable --now asr-worker-live.service
sudo systemctl enable --now asr-worker-batch.service
sudo systemctl enable --now llm-worker.service
sudo systemctl enable --now transcribe-tabby-tunnel.service
```

## 7) Tunnel Services on Your Other Computer

Tunnels run on your other computer, not on `dc1`.
If you want tunnel auto-start after reboot, create user services there.

### A) Live path tunnel service (localhost:8080 -> dc1:8080)

Create `~/.config/systemd/user/ssh-tunnel-transcribe-live-frontend.service`:

```ini
[Unit]
Description=SSH tunnel: local 8080 -> dc1 127.0.0.1:8080
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=/usr/bin/ssh -N \
  -L 8080:127.0.0.1:8080 gunnar@dc1 \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3
Restart=always
RestartSec=2

[Install]
WantedBy=default.target
```

### B) Dev path tunnel service (localhost:18010 -> dc1:8010)

Create `~/.config/systemd/user/ssh-tunnel-transcribe-dev-frontend.service`:

```ini
[Unit]
Description=SSH tunnel: local 18010 -> dc1 127.0.0.1:8010
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=/usr/bin/ssh -N \
  -L 18010:127.0.0.1:8010 gunnar@dc1 \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3
Restart=always
RestartSec=2

[Install]
WantedBy=default.target
```

### Enable both tunnel services on your other computer

```bash
systemctl --user daemon-reload
systemctl --user enable --now ssh-tunnel-transcribe-live-frontend.service
systemctl --user enable --now ssh-tunnel-transcribe-dev-frontend.service
```

Optional (if you need user services before interactive login):

```bash
sudo loginctl enable-linger "$USER"
```

### C) Observability tunnel services

These are also user services on your other computer, never on `dc1`.

1. **Dev observability tunnel**
   - Unit name: `transcribe-observability-tunnel.service`
   - Local ports:
     - `8001` -> `dc1:8001` (API `/ops`)
     - `18090` -> `dc1:18090` (ASR pool `/ops`)
     - `18110` -> `dc1:18110` (worker live `/ops`)
     - `18111` -> `dc1:18111` (worker batch `/ops`)

2. **Live observability tunnel**
   - Unit name: `transcribe-observability-live-tunnel.service`
   - Local ports:
     - `8000` -> `dc1:8000` (API `/ops`)
     - `8090` -> `dc1:8090` (ASR pool `/ops`)
     - `28110` -> `dc1:28110` (worker live `/ops`)
     - `28111` -> `dc1:28111` (worker batch `/ops`)
   - Live uses `28110` and `28111` intentionally because dev and live worker ops pages both run on `dc1`.

3. Tracked unit files for these observability tunnels live in:
   - `~/projects/transcribe-dev/deploy/systemd/transcribe-observability-tunnel.service`
   - `~/projects/transcribe-dev/deploy/systemd/transcribe-observability-live-tunnel.service`

## 8) Environment Files and Secrets

### Live

1. Omniscripta API / LLM worker / janitor:
   - `/etc/transcribe/transcribe.env`
   - This is also where live Tabby-related values live.

2. ASR pool:
   - `/etc/asr-pool/asr-pool.env`

3. ASR workers:
   - `/etc/asr-worker/asr-worker.env`

### Dev

1. Omniscripta API / frontend proxy / LLM worker / janitor:
   - `~/.config/transcribe/dev.env`

2. ASR pool:
   - `~/.config/asr-pool/asr-pool.env`

3. ASR workers:
   - `~/.config/asr-worker/asr-worker.dev.env`

4. Dev key quick set for the transcribe-side env:
```bash
mkdir -p ~/.config/transcribe && printf 'TABBY_API_KEY=VUL_HIER_DE_KEY_IN\n' > ~/.config/transcribe/dev.env && chmod 600 ~/.config/transcribe/dev.env && systemctl --user restart asr-worker-live-dev@1.service asr-worker-batch-dev@1.service llm-worker-dev@1.service
```

## 9) Golden Rules

1. Build frontend in `/var/www/omniscripta-app`, never in `/srv/transcribe/static`.
2. Do API/LLM backend development in `~/projects/transcribe-dev`, not in `/srv/transcribe`.
3. Do ASR pool development in `~/projects/asr-pool-dev`, not in `/srv/asr-pool`.
4. Do ASR worker development in `~/projects/asr-worker-dev`, not in `/srv/asr-worker`.
5. Use `deploy.sh` only for live release.
6. Use `deploy-dev.sh` for dev-backend testing.
7. Keep both frontend tunnel paths available (`8080` live and `18010` dev) for fast A/B validation.
8. Keep dev and live observability tunnels separate; live worker ops use `28110` and `28111` on purpose.
9. In dev repos, do all development on `dev/*` branches (`dev/transcribe-dev`, `dev/asr-pool-dev`, `dev/asr-worker-dev`); use `main` only for controlled merge/release steps.
10. If you temporarily check out `main` in a dev repo for merge/release work, switch back to the matching `dev/*` branch before continuing development.
