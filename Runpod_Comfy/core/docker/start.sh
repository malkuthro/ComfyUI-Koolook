#!/usr/bin/env bash
set -euo pipefail

ROOT=/opt/bootstrap/Runpod_Comfy
WORKDIR="${COMFYUI_WORKDIR:-/workspace/ComfyUI}"
PORT="${COMFYUI_PORT:-8188}"

"$ROOT/core/scripts/bootstrap.sh"

cd "$WORKDIR"

exec python3 main.py --listen 0.0.0.0 --port "$PORT"
