#!/usr/bin/env bash
set -euo pipefail

ROOT=/opt/bootstrap/Runpod_Comfy
WORKDIR="${COMFYUI_WORKDIR:-/workspace/ComfyUI}"
mkdir -p "$WORKDIR"

source "$ROOT/config/comfyui.lock"

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

"$ROOT/scripts/install_custom_nodes.sh"
"$ROOT/scripts/download_models.sh"

echo "[bootstrap] done"
