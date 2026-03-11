#!/usr/bin/env bash
set -euo pipefail

# Restart the full live backend stack in a stable order and wait for key HTTP endpoints.
# If not run as root, this script uses sudo for system-level units.

API_UNIT="transcribe-api.service"
ASR_UNIT="transcribe-asr-pool.service"
UPLOAD_WORKER_UNIT="transcribe-worker-upload.service"
LIVE_WORKER_UNIT="transcribe-worker-live.service"
TABBY_TUNNEL_UNIT="transcribe-tabby-tunnel.service"

API_HEALTH_URL="http://127.0.0.1:8000/health"
ASR_POOL_URL="http://127.0.0.1:8090/asr/v1/pool"

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

echo "[live-restart] Restarting API + ASR pool..."
systemctl_live restart "$API_UNIT" "$ASR_UNIT"

echo "[live-restart] Waiting for API health..."
wait_for_http "$API_HEALTH_URL" 60

echo "[live-restart] Waiting for ASR pool readiness (warm startup may take time)..."
wait_for_http "$ASR_POOL_URL" 240

echo "[live-restart] Restarting workers + tabby tunnel..."
systemctl_live restart "$UPLOAD_WORKER_UNIT" "$LIVE_WORKER_UNIT" "$TABBY_TUNNEL_UNIT"

echo
echo "[live-restart] Service status:"
for unit in \
  "$API_UNIT" \
  "$ASR_UNIT" \
  "$UPLOAD_WORKER_UNIT" \
  "$LIVE_WORKER_UNIT" \
  "$TABBY_TUNNEL_UNIT"; do
  state="$(systemctl_live is-active "$unit" || true)"
  printf "  - %-35s %s\n" "$unit" "$state"
done

echo
echo "[live-restart] OK"
