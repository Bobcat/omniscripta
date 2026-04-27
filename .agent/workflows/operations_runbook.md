---
description: Canonical workflow for Omniscripta frontend + Transcribe backend (prod and dev)
---

# Omniscripta + Transcribe Workflow

## Quickstart (Daily)

1. If frontend changed, deploy to dev static target:
```bash
/var/www/omniscripta-ui/deploy/deploy-dev.sh
```

2. Verify server-side dev services on `dc1`:
```bash
systemctl --user --no-pager status omniscripta-api-dev.service asr-pool-dev.service omniscripta-frontend-dev.service asr-worker-dev@1.service
```

3. If you changed backend code and want the full dev stack fresh:
```bash
~/projects/omniscripta/deploy/dev_restart_all.sh
```

4. If you changed tracked unit files and want repo truth installed:
```bash
~/projects/omniscripta/deploy/install-dev-user-units.sh
```

5. On your other computer, use one of these URLs:
- Prod backend path: `http://localhost:8080/index.html`
- Dev backend path: `http://127.0.0.1:18010/index.html`

## 1) Repositories and Roles

1. **Frontend source repo**: `/var/www/omniscripta-ui`
   - Edit UI/UX and editor logic here (`index.html`, `js/*`, `css/*`).
   - Deploy scripts are in this repo under `deploy/`: `deploy/deploy.sh` (prod), `deploy/deploy-dev.sh` (dev static target).

2. **Prod repos on `dc1`**
   - `/srv/omniscripta` - production Omniscripta backend/API, LLM worker, deploy scripts, static files.
   - `/srv/realtime-asr-engine` - production checkout of the shared live ASR engine package used by the current dev branch rollout.
   - `/srv/asr-worker` - production ASR worker repo checkout.
   - Prod ASR pool runtime is remote on `dc2`, not a local process on `dc1`.
   - Treat these as prod environment; avoid direct development changes there.

3. **Backend dev worktree (persistent)**: `~/projects/omniscripta`
   - Safe API/LLM/backend development environment.
   - Separate data/config paths from prod.

4. **Standalone ASR pool repo**: `~/projects/asr-pool-dev`
   - Source for dev ASR pool service (`asr-pool-dev.service`).
   - Keep ASR pool runtime/config changes here.

5. **Standalone ASR worker repo**: `~/projects/asr-worker-dev`
   - Source for the dev batch worker service (`asr-worker-dev@1.service`).
   - Keep ASR worker runtime/config changes here.

6. **Standalone ASR pool client repo**: `~/projects/asr-pool-api-dev`
   - Source for the shared dev `asr_pool_api` client library used by backend consumers.
   - Keep shared client API and transport-wrapping changes here.

7. **Shared live ASR engine package repo**: `~/projects/realtime-asr-engine`
   - Source for the shared `realtime_asr_engine` package used by the current dev Omniscripta live engine.
   - The matching prod rollout target is `/srv/realtime-asr-engine`.

## 2) Environments

### Prod
- Omniscripta root: `/srv/omniscripta`
- Shared live ASR engine package root for the current dev rollout target: `/srv/realtime-asr-engine`
- ASR pool runtime: remote on `dc2`
- ASR worker root: `/srv/asr-worker`
- API service: `omniscripta-api.service`
- Worker service: `asr-worker.service` (ops on 127.0.0.1:28111)
- LLM worker service: `llm-worker.service`
- ASR pool access tunnel on `dc1`: `asr-pool-dc2-tunnel.service`
- API/LLM env file: `/etc/transcribe/transcribe.env`
- ASR pool env file: `/etc/asr-pool/asr-pool.env`
- ASR worker env file: `/etc/asr-worker/asr-worker.env`
- Frontend static target: `/srv/omniscripta/static` (served by nginx)
- Worker queue root: `/srv/omniscripta/data/upload/jobs/worker`
- API `/ops` on prod uses the local systemd drop-in `/etc/systemd/system/omniscripta-api.service.d/ops-env.conf` so it points at the prod pool access path on `:8090` and worker ops `:28111`.
- Note: on `dc1`, prod consumers talk to the prod ASR pool through the configured dc1->dc2 access path; `dc1` does not run the pool process locally.
- Note: current prod `main` still runs the legacy internal layout. Installing the repo-backed prod units from the current dev branch only becomes valid after the prod Omniscripta checkout has been promoted to the `app/` + `workers/llm` layout and `/srv/realtime-asr-engine` exists as a sibling checkout.

### Dev
- API/LLM backend root: `~/projects/omniscripta`
- ASR pool backend root: `~/projects/asr-pool-dev`
- ASR worker backend root: `~/projects/asr-worker-dev`
- ASR pool client root: `~/projects/asr-pool-api-dev`
- API service: `omniscripta-api-dev.service` (127.0.0.1:8001)
- ASR pool service: `asr-pool-dev.service` (127.0.0.1:18090)
- Worker service template: `asr-worker-dev@.service`
- Frontend proxy service: `omniscripta-frontend-dev.service` (127.0.0.1:8010)
- API/LLM/frontend env file: `~/.config/transcribe/dev.env`
- ASR pool env file: `~/.config/asr-pool-dev/dev.env`
- ASR worker env file: `~/.config/asr-worker/asr-worker.dev.env`
- Dev frontend static target: `~/projects/omniscripta/static`
- Note: dev API code now lives under `~/projects/omniscripta/app`, and the live engine resolves `realtime_asr_engine` from the sibling checkout at `~/projects/realtime-asr-engine/src`. The dev API service still uses the legacy venv at `/srv/omniscripta/portal-api/.venv`.
- Dev API and ASR worker services resolve `asr_pool_api` from `~/projects/asr-pool-api-dev/src` via service-level `PYTHONPATH`.

### Shared client library
- Dev consumers now import `asr_pool_api` from the dedicated dev repo `~/projects/asr-pool-api-dev`.
- Dev resolution is service-scoped, not code-scoped:
  1. `omniscripta-api-dev.service` sets `PYTHONPATH=~/projects/asr-pool-api-dev/src`
  2. `asr-worker-dev@.service` sets `PYTHONPATH=~/projects/asr-pool-api-dev/src`
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

From `~/projects/omniscripta`:

1. Dev full restart:
```bash
~/projects/omniscripta/deploy/dev_restart_all.sh
```

2. Prod full restart:
```bash
~/projects/omniscripta/deploy/live_restart_all.sh
```

3. Install repo-backed dev user units:
```bash
~/projects/omniscripta/deploy/install-dev-user-units.sh
```

4. Install repo-backed prod system units on `dc1`:
```bash
~/projects/omniscripta/deploy/install-prod-system-units.sh
```

These scripts are the current canonical stack-control and unit-sync entry points.
Current scope limit: `install-prod-system-units.sh` installs the `dc1` prod consumer units only; it does not manage the remote prod ASR pool runtime on `dc2`.

## 5) Deploy Scripts

From `/var/www/omniscripta-ui`:

1. Prod deploy:
```bash
/var/www/omniscripta-ui/deploy/deploy.sh
```
- Builds frontend bundle and deploys to `/srv/omniscripta/static`.

2. Dev deploy:
```bash
/var/www/omniscripta-ui/deploy/deploy-dev.sh
```
- Builds frontend bundle and deploys to `~/projects/omniscripta/static`.

## 6) Required systemd Services

### On server (`dc1`) - user services

```bash
systemctl --user enable --now omniscripta-api-dev.service
systemctl --user enable --now asr-pool-dev.service
systemctl --user enable --now omniscripta-frontend-dev.service
systemctl --user enable --now asr-worker-dev@1.service
```

### On server (`dc1`) - prod services (system scope)

```bash
sudo systemctl enable --now omniscripta-api.service
sudo systemctl enable --now asr-worker.service
sudo systemctl enable --now llm-worker.service
sudo systemctl enable --now asr-pool-dc2-tunnel.service
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

## 8) Environment Files and Secrets

### Prod

1. Omniscripta API / LLM worker:
   - `/etc/transcribe/transcribe.env`

2. ASR pool:
   - `/etc/asr-pool/asr-pool.env`

3. ASR workers:
   - `/etc/asr-worker/asr-worker.env`

### Dev

1. Omniscripta API / frontend proxy / LLM worker:
   - `~/.config/transcribe/dev.env`

2. ASR pool:
   - `~/.config/asr-pool/asr-pool.env`

3. ASR workers:
   - `~/.config/asr-worker/asr-worker.dev.env`

## 9) Golden Rules

1. Build frontend in `/var/www/omniscripta-ui`, never in `/srv/omniscripta/static`.
2. Do API/LLM backend development in `~/projects/omniscripta`, not in `/srv/omniscripta`.
3. Do ASR pool development in `~/projects/asr-pool-dev`, not on the prod runtime host/process.
4. Do ASR worker development in `~/projects/asr-worker-dev`, not in `/srv/asr-worker`.
5. Use `deploy/deploy.sh` only for prod release.
6. Use `deploy/deploy-dev.sh` for dev-backend testing.
7. Keep both frontend tunnel paths available (`8080` prod and `18010` dev) for fast A/B validation.
8. In dev repos, do all development on `dev/*` branches (`dev/omniscripta`, `dev/asr-pool-dev`, `dev/asr-worker-dev`); use `main` only for controlled merge/release steps.
9. If you temporarily check out `main` in a dev repo for merge/release work, switch back to the matching `dev/*` branch before continuing development.
