# Runpod_Comfy

Production-oriented scaffold for running ComfyUI on RunPod with pinned versions.

## Goals

- Reproducible ComfyUI runtime
- Clear version pinning for ComfyUI + custom nodes
- Optional model bootstrap from URLs
- Easy promotion from test to stable

## Structure

- `docker/` → image/runtime entrypoint
- `config/` → lock files and model manifests
- `scripts/` → bootstrap installers
- `docs/` → requirements and references

## Quick Start

### 1) Pin ComfyUI version

Edit `config/comfyui.lock`:

```ini
COMFYUI_REPO=https://github.com/comfyanonymous/ComfyUI.git
COMFYUI_REF=master
```

Set `COMFYUI_REF` to a commit SHA for stability.

### 2) Add custom nodes

Edit `config/custom_nodes.lock.json`.

### 3) Add models (optional)

Edit `config/models.json`.

### 4) Build image

```bash
docker build -t comfyui-koolook:runpod-v1 -f Runpod_Comfy/docker/Dockerfile .
```

### 5) Run locally (smoke test)

```bash
docker run --rm -it -p 8188:8188 \
  -e COMFYUI_WORKDIR=/workspace/ComfyUI \
  comfyui-koolook:runpod-v1
```

## Runtime env vars

- `COMFYUI_PORT` (default: `8188`)
- `COMFYUI_WORKDIR` (default: `/workspace/ComfyUI`)
- `RUNPOD_VOLUME_ROOT` (default: `/workspace`)
- `HF_TOKEN` (optional for gated HuggingFace models)
- `CIVITAI_TOKEN` (optional)

## Notes

- This scaffold is intentionally conservative and explicit.
- For production users, pin refs to commits/tags before release.
