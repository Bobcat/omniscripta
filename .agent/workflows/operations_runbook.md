---
description: Canonical workflow for Omniscripta frontend + Transcribe backend (prod and dev)
---

# Omniscripta + Transcribe Workflow

## Quickstart (Daily)

1. If frontend changed, deploy to dev static target:
```bash
/var/www/omniscripta-app/deploy-dev.sh
```

2. Verify server-side dev services on `dc1`:
```bash
systemctl --user --no-pager status transcribe-api-dev.service transcribe-asr-pool-dev.service transcribe-frontend-dev.service asr-worker-batch-dev@1.service llm-worker-dev@1.service
```

3. If you changed backend code and want the full dev stack fresh:
```bash
~/projects/transcribe-dev/deploy/dev_restart_all.sh
```

4. If you changed tracked unit files and want repo truth installed:
```bash
~/projects/transcribe-dev/deploy/install-dev-user-units.sh
```

5. On your other computer, use one of these URLs:
- Prod backend path: `http://localhost:8080/index.html`
- Dev backend path: `http://127.0.0.1:18010/index.html`

## 1) Repositories and Roles

1. **Frontend source repo**: `/var/www/omniscripta-app`
   - Edit UI/UX and editor logic here (`index.html`, `js/*`, `css/*`).
   - Deploy scripts are in this repo: `deploy.sh` (prod), `deploy-dev.sh` (dev static target).

2. **Prod repos on `dc1`**
   - `/srv/transcribe` - production Omniscripta backend/API, LLM worker, deploy scripts, static files.
   - `/srv/asr-worker` - production ASR worker repo checkout.
   - Prod ASR pool runtime is remote on `dc2`, not a local process on `dc1`.
   - Treat these as prod environment; avoid direct development changes there.

3. **Backend dev worktree (persistent)**: `~/projects/transcribe-dev`
   - Safe API/LLM/backend development environment.
   - Separate data/config paths from prod.

4. **Standalone ASR pool repo**: `~/projects/asr-pool-dev`
   - Source for dev ASR pool service (`transcribe-asr-pool-dev.service`).
   - Keep ASR pool runtime/config changes here.

5. **Standalone ASR worker repo**: `~/projects/asr-worker-dev`
   - Source for the dev batch worker service (`asr-worker-batch-dev@1.service`).
   - Keep ASR worker runtime/config changes here.

6. **Standalone ASR pool client repo**: `~/projects/asr-pool-api-dev`
   - Source for the shared dev `asr_pool_api` client library used by backend consumers.
   - Keep shared client API and transport-wrapping changes here.

## 2) Environments

### Prod
- Omniscripta root: `/srv/transcribe`
- ASR pool runtime: remote on `dc2`
- ASR worker root: `/srv/asr-worker`
- API service: `transcribe-api.service`
- Worker service: `asr-worker-batch.service` (ops on 127.0.0.1:28111)
- LLM worker service: `llm-worker.service`
- API/LLM env file: `/etc/transcribe/transcribe.env`
- ASR pool env file: `/etc/asr-pool/asr-pool.env`
- ASR worker env file: `/etc/asr-worker/asr-worker.env`
- Tabby tunnel service: `transcribe-tabby-tunnel.service`
- Frontend static target: `/srv/transcribe/static` (served by nginx)
- Worker queue root: `/srv/transcribe/data/jobs/upload_worker`
- API `/ops` on prod uses the local systemd drop-in `/etc/systemd/system/transcribe-api.service.d/ops-env.conf` so it points at the prod pool access path on `:8090` and worker ops `:28111`.
- Note: on `dc1`, prod consumers talk to the prod ASR pool through the configured dc1->dc2 access path; `dc1` does not run the pool process locally.

### Dev
- API/LLM backend root: `~/projects/transcribe-dev`
- ASR pool backend root: `~/projects/asr-pool-dev`
- ASR worker backend root: `~/projects/asr-worker-dev`
- ASR pool client root: `~/projects/asr-pool-api-dev`
- API service: `transcribe-api-dev.service` (127.0.0.1:8001)
- ASR pool service: `transcribe-asr-pool-dev.service` (127.0.0.1:18090)
- Worker service template: `asr-worker-batch-dev@.service`
- LLM worker service template: `llm-worker-dev@.service`
- Janitor timer: `transcribe-demo-jobs-janitor-dev.timer`
- Frontend proxy service: `transcribe-frontend-dev.service` (127.0.0.1:8010)
- API/LLM/frontend env file: `~/.config/transcribe/dev.env`
- ASR pool env file: `~/.config/asr-pool/asr-pool.env`
- ASR worker env file: `~/.config/asr-worker/asr-worker.dev.env`
- Dev frontend static target: `~/projects/transcribe-dev/static`
- Note: dev API currently runs `uvicorn` from `/srv/transcribe/portal-api/.venv`, and `llm-worker-dev@.service` currently runs from `/srv/transcribe/worker/.venv`.
- Dev API and worker services resolve `asr_pool_api` from `~/projects/asr-pool-api-dev/src` via service-level `PYTHONPATH`.

### Shared client library
- Dev consumers now import `asr_pool_api` from the dedicated dev repo `~/projects/asr-pool-api-dev`.
- Dev resolution is service-scoped, not code-scoped:
  1. `transcribe-api-dev.service` sets `PYTHONPATH=~/projects/asr-pool-api-dev/src`
  2. `asr-worker-batch-dev@.service` sets `PYTHONPATH=~/projects/asr-pool-api-dev/src`
- Operational consequence:
  - You can iterate on `asr-pool-api-dev` without modifying prod services or prod consumer checkouts.
- Note:
  - Prod consumers still use their existing runtime wiring until a separate prod rollout migrates them to `/srv/asr-pool-api`.

## 3) Frontend Test Paths (both active)

1. **Prod-backend test path**
   - URL on your other computer: `http://localhost:8080/index.html`
   - This path talks to the **prod backend**.

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

3. Prod full start:
```bash
~/projects/transcribe-dev/deploy/live_start_all.sh
```

4. Prod full restart:
```bash
~/projects/transcribe-dev/deploy/live_restart_all.sh
```

5. Install repo-backed dev user units:
```bash
~/projects/transcribe-dev/deploy/install-dev-user-units.sh
```

6. Install repo-backed prod system units on `dc1`:
```bash
~/projects/transcribe-dev/deploy/install-prod-system-units.sh
```

These scripts are the current canonical stack-control and unit-sync entry points.
Current gap: `transcribe-asr-pool-dev.service` is not yet installed by `install-dev-user-units.sh` because `asr-pool-dev` does not currently track a matching source file under `deploy/systemd/`.
Current scope limit: `install-prod-system-units.sh` installs the `dc1` prod consumer units only; it does not manage the remote prod ASR pool runtime on `dc2`.

## 5) Deploy Scripts

From `/var/www/omniscripta-app`:

1. Prod deploy:
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
systemctl --user enable --now asr-worker-batch-dev@1.service
systemctl --user enable --now llm-worker-dev@1.service
```

### On server (`dc1`) - prod services (system scope)

```bash
sudo systemctl enable --now transcribe-api.service
sudo systemctl enable --now asr-worker-batch.service
sudo systemctl enable --now llm-worker.service
sudo systemctl enable --now transcribe-tabby-tunnel.service
```

## 7) Tunnel Services on Your Other Computer

Tunnels run on your other computer, not on `dc1`.
If you want tunnel auto-start after reboot, create user services there.

### A) Prod path tunnel service (localhost:8080 -> dc1:8080)

Create `~/.config/systemd/user/ssh-tunnel-transcribe-live-frontend.service`:

```ini
[Unit]
Description=SSH tunnel: local 8080 -> dc1 127.0.0.1:8080 (prod path)
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
     - `18111` -> `dc1:18111` (worker batch `/ops`)

2. **Prod observability tunnel**
   - Unit name: `transcribe-observability-live-tunnel.service`
   - Local ports:
     - `8000` -> `dc1:8000` (API `/ops`)
     - `8090` -> `dc1:8090` (ASR pool `/ops`)
     - `28111` -> `dc1:28111` (worker batch `/ops`)

3. Tracked unit files for these observability tunnels are in:
   - `~/projects/transcribe-dev/deploy/systemd/transcribe-observability-tunnel.service`
   - `~/projects/transcribe-dev/deploy/systemd/transcribe-observability-live-tunnel.service`

## 8) Environment Files and Secrets

### Prod

1. Omniscripta API / LLM worker / janitor:
   - `/etc/transcribe/transcribe.env`
   - This is also where prod Tabby-related values are stored.

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
mkdir -p ~/.config/transcribe && printf 'TABBY_API_KEY=VUL_HIER_DE_KEY_IN\n' > ~/.config/transcribe/dev.env && chmod 600 ~/.config/transcribe/dev.env && systemctl --user restart asr-worker-batch-dev@1.service llm-worker-dev@1.service
```

## 9) Golden Rules

1. Build frontend in `/var/www/omniscripta-app`, never in `/srv/transcribe/static`.
2. Do API/LLM backend development in `~/projects/transcribe-dev`, not in `/srv/transcribe`.
3. Do ASR pool development in `~/projects/asr-pool-dev`, not on the prod runtime host/process.
4. Do ASR worker development in `~/projects/asr-worker-dev`, not in `/srv/asr-worker`.
5. Use `deploy.sh` only for prod release.
6. Use `deploy-dev.sh` for dev-backend testing.
7. Keep both frontend tunnel paths available (`8080` prod and `18010` dev) for fast A/B validation.
8. Keep dev and prod observability tunnels separate; batch worker ops use `18111` and `28111` on purpose.
9. In dev repos, do all development on `dev/*` branches (`dev/transcribe-dev`, `dev/asr-pool-dev`, `dev/asr-worker-dev`); use `main` only for controlled merge/release steps.
10. If you temporarily check out `main` in a dev repo for merge/release work, switch back to the matching `dev/*` branch before continuing development.
