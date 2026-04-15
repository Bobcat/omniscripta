#!/usr/bin/env bash
set -euo pipefail

TRANSCRIBE_REPO_ROOT="${TRANSCRIBE_REPO_ROOT:-/srv/transcribe}"
ASR_WORKER_REPO_ROOT="${ASR_WORKER_REPO_ROOT:-/srv/asr-worker}"
UNIT_DIR="/etc/systemd/system"

if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
  SUDO=""
else
  SUDO="sudo"
fi

install_unit() {
  local src="$1"
  local dst_name="${2:-$(basename "$src")}"
  if [[ ! -f "$src" ]]; then
    printf 'missing source unit: %s\n' "$src" >&2
    exit 1
  fi
  $SUDO install -m 0644 "$src" "$UNIT_DIR/$dst_name"
  printf 'installed %s\n' "$UNIT_DIR/$dst_name"
}

install_unit "$TRANSCRIBE_REPO_ROOT/deploy/systemd/transcribe-api.service"
install_unit "$TRANSCRIBE_REPO_ROOT/deploy/systemd/llm-worker.service"
install_unit "$ASR_WORKER_REPO_ROOT/deploy/systemd/asr-worker-batch.service"

$SUDO systemctl daemon-reload
printf 'reloaded systemd units\n'
