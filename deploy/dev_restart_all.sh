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
ASR_POOL_READY_TIMEOUT_S=90

ts() {
  date +"%H:%M:%S"
}

log() {
  echo "[dev-restart][$(ts)] $*"
}

wait_for_http() {
  local url="$1"
  local timeout_s="$2"
  local label="${3:-$1}"
  local started_s now_s elapsed_s
  started_s="$(date +%s)"
  while true; do
    if curl -fsS "$url" >/dev/null 2>&1; then
      now_s="$(date +%s)"
      elapsed_s="$((now_s - started_s))"
      log "$label ready after ${elapsed_s}s"
      return 0
    fi
    now_s="$(date +%s)"
    elapsed_s="$((now_s - started_s))"
    if (( elapsed_s >= timeout_s )); then
      log "Timeout while waiting for $label (${timeout_s}s)"
      return 1
    fi
    sleep 1
  done
}

print_status() {
  local maxlen=0 unit state
  for unit in "$@"; do
    if (( ${#unit} > maxlen )); then
      maxlen=${#unit}
    fi
  done
  for unit in "$@"; do
    state="$(systemctl --user is-active "$unit" || true)"
    printf "  - %-*s  %s\n" "$maxlen" "$unit" "$state"
  done
}

log "Restarting API + ASR pool..."
systemctl --user restart "$API_UNIT" "$ASR_UNIT"

log "Waiting for API health..."
wait_for_http "$API_HEALTH_URL" 60 "API health"

log "Waiting for ASR pool readiness (warm startup may take time)..."
wait_for_http "$ASR_POOL_URL" "$ASR_POOL_READY_TIMEOUT_S" "ASR pool readiness"

log "Restarting workers..."
systemctl --user restart "$UPLOAD_WORKER_UNIT" "$LIVE_WORKER_UNIT"

log "Restarting frontend proxy..."
systemctl --user restart "$FRONTEND_UNIT"

log "Waiting for frontend..."
wait_for_http "$FRONTEND_URL" 30 "Frontend"

echo
echo "[dev-restart] Service status:"
print_status \
  "$API_UNIT" \
  "$ASR_UNIT" \
  "$UPLOAD_WORKER_UNIT" \
  "$LIVE_WORKER_UNIT" \
  "$FRONTEND_UNIT"

echo
echo "[dev-restart] OK"
