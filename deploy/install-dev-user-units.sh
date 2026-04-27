#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRANSCRIBE_REPO_ROOT="${TRANSCRIBE_REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
ASR_WORKER_REPO_ROOT="${ASR_WORKER_REPO_ROOT:-$HOME/projects/asr-worker-dev}"
ASR_POOL_REPO_ROOT="${ASR_POOL_REPO_ROOT:-$HOME/projects/asr-pool-dev}"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

install_unit() {
  local src="$1"
  local dst_name="${2:-$(basename "$src")}"
  if [[ ! -f "$src" ]]; then
    printf 'missing source unit: %s\n' "$src" >&2
    exit 1
  fi
  install -m 0644 "$src" "$UNIT_DIR/$dst_name"
  printf 'installed %s\n' "$UNIT_DIR/$dst_name"
}

mkdir -p "$UNIT_DIR"

install_unit "$TRANSCRIBE_REPO_ROOT/deploy/systemd/omniscripta-api-dev.service"
install_unit "$TRANSCRIBE_REPO_ROOT/deploy/systemd/omniscripta-frontend-dev.service"
install_unit "$ASR_WORKER_REPO_ROOT/deploy/systemd/asr-worker-dev@.service"

if [[ -f "$ASR_POOL_REPO_ROOT/deploy/systemd/asr-pool-dev.service" ]]; then
  install_unit "$ASR_POOL_REPO_ROOT/deploy/systemd/asr-pool-dev.service"
else
  printf 'note: skipped asr-pool-dev.service; no tracked source file at %s\n' \
    "$ASR_POOL_REPO_ROOT/deploy/systemd/asr-pool-dev.service" >&2
fi

systemctl --user daemon-reload
printf 'reloaded user systemd units\n'
