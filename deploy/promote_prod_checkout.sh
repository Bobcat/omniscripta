#!/usr/bin/env bash
set -euo pipefail

TRANSCRIBE_REPO_ROOT="${TRANSCRIBE_REPO_ROOT:-/srv/omniscripta}"
TARGET_REF="${1:-origin/main}"
REMOTE_NAME="${REMOTE_NAME:-origin}"

ALLOWED_DIRTY_PATHS=(
  "static/app.bundle.js"
  "static/index.html"
  "static/app/index.html"
)

ts() {
  date +"%H:%M:%S"
}

log() {
  echo "[prod-promote][$(ts)] $*"
}

die() {
  echo "[prod-promote][$(ts)] ERROR: $*" >&2
  exit 1
}

if [[ ! -d "$TRANSCRIBE_REPO_ROOT/.git" && ! -f "$TRANSCRIBE_REPO_ROOT/.git" ]]; then
  die "missing git checkout at $TRANSCRIBE_REPO_ROOT"
fi

if ! git -C "$TRANSCRIBE_REPO_ROOT" remote get-url "$REMOTE_NAME" >/dev/null 2>&1; then
  die "missing git remote '$REMOTE_NAME' in $TRANSCRIBE_REPO_ROOT"
fi

log "Fetching $REMOTE_NAME..."
git -C "$TRANSCRIBE_REPO_ROOT" fetch "$REMOTE_NAME"

TARGET_COMMIT="$(git -C "$TRANSCRIBE_REPO_ROOT" rev-parse --verify "$TARGET_REF^{commit}")"
CURRENT_COMMIT="$(git -C "$TRANSCRIBE_REPO_ROOT" rev-parse --verify HEAD)"

mapfile -t DIRTY_PATHS < <(git -C "$TRANSCRIBE_REPO_ROOT" status --porcelain | sed 's/^...//')

UNEXPECTED_DIRTY=()
for path in "${DIRTY_PATHS[@]}"; do
  [[ -n "$path" ]] || continue
  allowed=0
  for allowed_path in "${ALLOWED_DIRTY_PATHS[@]}"; do
    if [[ "$path" == "$allowed_path" ]]; then
      allowed=1
      break
    fi
  done
  if (( ! allowed )); then
    UNEXPECTED_DIRTY+=("$path")
  fi
done

if (( ${#UNEXPECTED_DIRTY[@]} > 0 )); then
  {
    echo "refusing to promote with unexpected local changes in $TRANSCRIBE_REPO_ROOT:"
    printf '  - %s\n' "${UNEXPECTED_DIRTY[@]}"
  } >&2
  exit 1
fi

if (( ${#DIRTY_PATHS[@]} > 0 )); then
  mapfile -t STATIC_TARGET_DIFFS < <(
    git -C "$TRANSCRIBE_REPO_ROOT" diff --name-only "$CURRENT_COMMIT..$TARGET_COMMIT" -- "${ALLOWED_DIRTY_PATHS[@]}"
  )
  if (( ${#STATIC_TARGET_DIFFS[@]} > 0 )); then
    {
      echo "refusing to promote because the target commit also changes generated static files:"
      printf '  - %s\n' "${STATIC_TARGET_DIFFS[@]}"
      echo
      echo "restore those files first, run this script again, then redeploy the frontend."
    } >&2
    exit 1
  fi
fi

if [[ "$CURRENT_COMMIT" == "$TARGET_COMMIT" ]]; then
  log "Checkout already at $TARGET_COMMIT; nothing to do."
  exit 0
fi

log "Promoting $TRANSCRIBE_REPO_ROOT from $CURRENT_COMMIT to $TARGET_COMMIT ($TARGET_REF)..."
git -C "$TRANSCRIBE_REPO_ROOT" switch --detach "$TARGET_COMMIT"

log "Now at $(git -C "$TRANSCRIBE_REPO_ROOT" rev-parse --verify HEAD)"
