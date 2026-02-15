#!/usr/bin/env bash
set -euo pipefail

ROOT=/opt/bootstrap/Runpod_Comfy
WORKDIR="${COMFYUI_WORKDIR:-/workspace/ComfyUI}"
PORT="${COMFYUI_PORT:-8188}"
JUPYTER_PORT="${JUPYTER_PORT:-8888}"
JUPYTER_TOKEN="${JUPYTER_TOKEN:-}"

"$ROOT/core/scripts/bootstrap.sh"

LOCK="$ROOT/image/active/comfyui.lock"
ENABLE_JUPYTER_LOCK="true"
if [[ -f "$LOCK" ]]; then
  ENABLE_JUPYTER_LOCK=$(grep -E '^ENABLE_JUPYTER=' "$LOCK" | head -n1 | cut -d= -f2-)
  [[ -z "$ENABLE_JUPYTER_LOCK" ]] && ENABLE_JUPYTER_LOCK="true"
fi

if [[ "$ENABLE_JUPYTER_LOCK" == "true" ]]; then
  if ! command -v jupyter >/dev/null 2>&1; then
    echo "[jupyter] WARNING: ENABLE_JUPYTER=true but jupyter binary not found (likely built with ENABLE_JUPYTER=false)"
  else
    if [[ -z "$JUPYTER_TOKEN" ]]; then
      JUPYTER_TOKEN="$(python3 -c 'import secrets; print(secrets.token_hex(16))')"
      echo "[jupyter] generated token: $JUPYTER_TOKEN"
    fi

    echo "[jupyter] starting on 0.0.0.0:${JUPYTER_PORT}"
    jupyter lab \
      --ip=0.0.0.0 \
      --port "$JUPYTER_PORT" \
      --no-browser \
      --allow-root \
      --IdentityProvider.token="$JUPYTER_TOKEN" \
      --ServerApp.root_dir="$WORKDIR" \
      >/tmp/jupyter.log 2>&1 &

    JUPYTER_PID=$!
    sleep 2
    if ! kill -0 "$JUPYTER_PID" 2>/dev/null; then
      echo "[jupyter] WARNING: failed to start, check /tmp/jupyter.log"
    fi
  fi
fi

cd "$WORKDIR"

exec python3 main.py --listen 0.0.0.0 --port "$PORT"
