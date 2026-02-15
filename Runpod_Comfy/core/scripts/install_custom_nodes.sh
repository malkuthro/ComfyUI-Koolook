#!/usr/bin/env bash
set -euo pipefail

ROOT=/opt/bootstrap/Runpod_Comfy
WORKDIR="${COMFYUI_WORKDIR:-/workspace/ComfyUI}"
NODES_DIR="$WORKDIR/custom_nodes"
LOCK="$ROOT/image/active/custom_nodes.lock.json"

mkdir -p "$NODES_DIR"

if [[ ! -f "$LOCK" ]]; then
  echo "[custom_nodes] lock file not found: $LOCK"
  exit 0
fi

COUNT=$(jq '.nodes | length' "$LOCK")
for ((i=0; i<COUNT; i++)); do
  REPO=$(jq -r ".nodes[$i].repo" "$LOCK")
  REF=$(jq -r ".nodes[$i].ref" "$LOCK")
  TARGET=$(jq -r ".nodes[$i].target_dir" "$LOCK")

  [[ -z "$TARGET" || "$TARGET" == "null" ]] && TARGET=$(basename "$REPO" .git)

  DST="$NODES_DIR/$TARGET"

  if [[ ! -d "$DST/.git" ]]; then
    echo "[custom_nodes] cloning $REPO -> $DST"
    git clone "$REPO" "$DST"
  fi

  pushd "$DST" >/dev/null
  git fetch --all --tags || true
  git checkout "$REF"

  if [[ -f requirements.txt ]]; then
    python3 -m pip install -r requirements.txt || true
  fi

  popd >/dev/null
done

echo "[custom_nodes] done"
