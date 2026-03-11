#!/usr/bin/env bash
set -euo pipefail

# Restart the full dev stack in a stable order and wait for key HTTP endpoints.
# This guarantees fresh processes for all involved services.

API_UNIT="transcribe-api-dev.service"
ASR_UNIT="transcribe-asr-pool-dev.service"
UPLOAD_WORKER_UNIT="transcribe-worker-upload-dev@1.service"
LIVE_WORKER_UNIT="transcribe-worker-live-dev@1.service"
FRONTEND_UNIT="transcribe-frontend-dev.service"

API_HEALTH_URL="http://127.0.0.1:8001/health"
ASR_POOL_URL="http://127.0.0.1:18090/asr/v1/pool"
FRONTEND_URL="http://127.0.0.1:8010/index.html"

wait_for_http() {
  local url="$1"
  local timeout_s="$2"
  local started_s now_s elapsed_s
  started_s="$(date +%s)"
  while true; do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    now_s="$(date +%s)"
    elapsed_s="$((now_s - started_s))"
    if (( elapsed_s >= timeout_s )); then
      echo "Timeout while waiting for $url (${timeout_s}s)" >&2
      return 1
    fi
    sleep 1
  done
}

echo "[dev-restart] Restarting API + ASR pool..."
systemctl --user restart "$API_UNIT" "$ASR_UNIT"

echo "[dev-restart] Waiting for API health..."
wait_for_http "$API_HEALTH_URL" 60

echo "[dev-restart] Waiting for ASR pool readiness (warm startup may take time)..."
wait_for_http "$ASR_POOL_URL" 240

echo "[dev-restart] Restarting workers..."
systemctl --user restart "$UPLOAD_WORKER_UNIT" "$LIVE_WORKER_UNIT"

echo "[dev-restart] Restarting frontend proxy..."
systemctl --user restart "$FRONTEND_UNIT"

echo "[dev-restart] Waiting for frontend..."
wait_for_http "$FRONTEND_URL" 30

echo
echo "[dev-restart] Service status:"
for unit in \
  "$API_UNIT" \
  "$ASR_UNIT" \
  "$UPLOAD_WORKER_UNIT" \
  "$LIVE_WORKER_UNIT" \
  "$FRONTEND_UNIT"; do
  state="$(systemctl --user is-active "$unit" || true)"
  printf "  - %-35s %s\n" "$unit" "$state"
done

echo
echo "[dev-restart] OK"
