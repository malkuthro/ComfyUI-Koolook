#!/usr/bin/env bash
set -euo pipefail

ROOT=/opt/bootstrap/Runpod_Comfy
WORKDIR="${COMFYUI_WORKDIR:-/workspace/ComfyUI}"
LOCK="$ROOT/image/active/comfyui.lock"
mkdir -p "$WORKDIR"

lock_get() {
  local key="$1"
  grep -E "^${key}=" "$LOCK" | head -n1 | cut -d= -f2-
}

COMFYUI_REPO="$(lock_get COMFYUI_REPO)"
COMFYUI_REF="$(lock_get COMFYUI_REF)"

if [[ -z "$COMFYUI_REPO" || -z "$COMFYUI_REF" ]]; then
  echo "[bootstrap] invalid lock file: missing COMFYUI_REPO or COMFYUI_REF"
  exit 1
fi

if [[ ! -f "$WORKDIR/main.py" ]]; then
  echo "[bootstrap] Cloning ComfyUI from $COMFYUI_REPO"
  git clone "$COMFYUI_REPO" "$WORKDIR"
fi

cd "$WORKDIR"

echo "[bootstrap] Checking out ComfyUI ref: $COMFYUI_REF"
git fetch --all --tags || true
git checkout "$COMFYUI_REF"

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

"$ROOT/core/scripts/install_custom_nodes.sh"
"$ROOT/core/scripts/download_models.sh"

echo "[bootstrap] done"
