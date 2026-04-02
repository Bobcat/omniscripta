#!/usr/bin/env bash
set -euo pipefail

# Restart the full prod backend stack in a stable order and wait for key HTTP endpoints.
# If not run as root, this script uses sudo for system-level units.

API_UNIT="transcribe-api.service"
# ASR_UNIT="transcribe-asr-pool.service"
BATCH_WORKER_UNIT="asr-worker-batch.service"
LLM_WORKER_UNIT="llm-worker.service"
TABBY_TUNNEL_UNIT="transcribe-tabby-tunnel.service"

API_HEALTH_URL="http://127.0.0.1:8000/health"
ASR_POOL_URL="http://127.0.0.1:8090/asr/v1/pool"
API_READY_TIMEOUT_S="${API_READY_TIMEOUT_S:-20}"
ASR_POOL_READY_TIMEOUT_S="${ASR_POOL_READY_TIMEOUT_S:-120}"
WAIT_POLL_INTERVAL_S="${WAIT_POLL_INTERVAL_S:-0.5}"
CURL_CONNECT_TIMEOUT_S="${CURL_CONNECT_TIMEOUT_S:-1}"
CURL_MAX_TIME_S="${CURL_MAX_TIME_S:-2}"

ts() {
  date +"%H:%M:%S"
}

log() {
  echo "[live-restart][$(ts)] $*"
}

systemctl_live() {
  if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
    systemctl "$@"
  else
    sudo systemctl "$@"
  fi
}

wait_for_http() {
  local url="$1"
  local timeout_s="$2"
  local label="${3:-$1}"
  local ok_codes="${4:-200}"
  local started_s now_s elapsed_s
  local http_code
  started_s="$(date +%s)"
  while true; do
    http_code="$(
      curl -sS -o /dev/null -w "%{http_code}" \
        --connect-timeout "$CURL_CONNECT_TIMEOUT_S" \
        --max-time "$CURL_MAX_TIME_S" \
        "$url" 2>/dev/null || true
    )"
    if [[ ",$ok_codes," == *",$http_code,"* ]]; then
      now_s="$(date +%s)"
      elapsed_s="$((now_s - started_s))"
      log "$label ready after ${elapsed_s}s (http=$http_code)"
      return 0
    fi
    now_s="$(date +%s)"
    elapsed_s="$((now_s - started_s))"
    if (( elapsed_s >= timeout_s )); then
      log "Timeout while waiting for $label (${timeout_s}s)"
      return 1
    fi
    sleep "$WAIT_POLL_INTERVAL_S"
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
    state="$(systemctl_live is-active "$unit" || true)"
    printf "  - %-*s  %s\n" "$maxlen" "$unit" "$state"
  done
}

log "Restarting API..."
systemctl_live restart "$API_UNIT"
# systemctl_live restart "$API_UNIT" "$ASR_UNIT"

log "Waiting for API health..."
wait_for_http "$API_HEALTH_URL" "$API_READY_TIMEOUT_S" "API health" "200"

log "Waiting for ASR pool readiness via the dc1 prod access path..."
wait_for_http "$ASR_POOL_URL" "$ASR_POOL_READY_TIMEOUT_S" "ASR pool readiness" "200,401"

log "Restarting batch worker + tabby tunnel..."
systemctl_live restart "$BATCH_WORKER_UNIT" "$LLM_WORKER_UNIT" "$TABBY_TUNNEL_UNIT"

echo
echo "[live-restart] Service status:"
print_status \
  "$API_UNIT" \
  "$BATCH_WORKER_UNIT" \
  "$LLM_WORKER_UNIT" \
  "$TABBY_TUNNEL_UNIT"

echo
echo "[live-restart] OK"
