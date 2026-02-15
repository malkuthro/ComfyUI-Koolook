#!/usr/bin/env bash
set -euo pipefail

ROOT=/opt/bootstrap/Runpod_Comfy
WORKDIR="${COMFYUI_WORKDIR:-/workspace/ComfyUI}"
PORT="${COMFYUI_PORT:-8188}"
JUPYTER_PORT="${JUPYTER_PORT:-8888}"
JUPYTER_TOKEN="${JUPYTER_TOKEN:-runpod}"

"$ROOT/core/scripts/bootstrap.sh"

if [[ -f "$ROOT/image/active/comfyui.lock" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/image/active/comfyui.lock"
fi

if [[ "${ENABLE_JUPYTER:-true}" == "true" ]]; then
  echo "[jupyter] starting on 0.0.0.0:${JUPYTER_PORT}"
  jupyter lab \
    --ip=0.0.0.0 \
    --port "$JUPYTER_PORT" \
    --no-browser \
    --allow-root \
    --IdentityProvider.token="$JUPYTER_TOKEN" \
    --ServerApp.root_dir="$WORKDIR" \
    >/tmp/jupyter.log 2>&1 &
fi

cd "$WORKDIR"

exec python3 main.py --listen 0.0.0.0 --port "$PORT"
