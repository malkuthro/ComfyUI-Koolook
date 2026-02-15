#!/usr/bin/env bash
set -euo pipefail

ROOT=/opt/bootstrap/Runpod_Comfy
WORKDIR="${COMFYUI_WORKDIR:-/workspace/ComfyUI}"
LOCK="$ROOT/image/active/models.json"

if [[ ! -f "$LOCK" ]]; then
  echo "[models] lock file not found: $LOCK"
  exit 0
fi

COUNT=$(jq '.models | length' "$LOCK")
for ((i=0; i<COUNT; i++)); do
  ENABLED=$(jq -r ".models[$i].enabled // true" "$LOCK")
  [[ "$ENABLED" != "true" ]] && continue

  NAME=$(jq -r ".models[$i].name" "$LOCK")
  URL=$(jq -r ".models[$i].url" "$LOCK")
  DEST_REL=$(jq -r ".models[$i].dest" "$LOCK")
  SHA256=$(jq -r ".models[$i].sha256 // \"\"" "$LOCK")

  DEST="$WORKDIR/$DEST_REL"
  mkdir -p "$(dirname "$DEST")"

  if [[ -f "$DEST" ]]; then
    echo "[models] exists: $NAME ($DEST_REL), skipping"
    continue
  fi

  echo "[models] downloading: $NAME"
  curl -L --fail "$URL" -o "$DEST"

  if [[ -n "$SHA256" ]]; then
    echo "$SHA256  $DEST" | sha256sum -c -
  fi
done

echo "[models] done"
