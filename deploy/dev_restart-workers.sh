#!/usr/bin/env bash
set -euo pipefail

BATCH_WORKER_UNIT="asr-worker-batch-dev@1.service"
LIVE_WORKER_UNIT="asr-worker-live-dev@1.service"
LLM_WORKER_UNIT="llm-worker-dev@1.service"

ts() {
  date +"%H:%M:%S"
}

log() {
  echo "[dev-restart-workers][$(ts)] $*"
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

log "Restarting dev workers..."
systemctl --user restart "$BATCH_WORKER_UNIT" "$LIVE_WORKER_UNIT" "$LLM_WORKER_UNIT"

echo
echo "[dev-restart-workers] Service status:"
print_status \
  "$BATCH_WORKER_UNIT" \
  "$LIVE_WORKER_UNIT" \
  "$LLM_WORKER_UNIT"

echo
echo "[dev-restart-workers] OK"
