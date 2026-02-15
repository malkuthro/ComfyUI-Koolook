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

### 1) Create profile files

Copy and edit these files:

- `profiles/runpod.<profile-id>.yaml`
- `profiles/nodes.<profile-id>.yaml`
- `profiles/models.<profile-id>.yaml`
- Optional: `profiles/urls.<profile-id>.txt`

You can start from the `*.example.yaml` files.

### 2) Compile profile into lock files

```bash
Runpod_Comfy/tools/compile_profile.sh <profile-id>
```

This generates:

- `config/comfyui.lock`
- `config/custom_nodes.lock.json`
- `config/models.json`

### 3) Build image

```bash
BASE_IMAGE=$(grep '^BASE_IMAGE=' Runpod_Comfy/config/comfyui.lock | cut -d= -f2-)
docker build --build-arg BASE_IMAGE="$BASE_IMAGE" \
  -t comfyui-koolook:runpod-v1 -f Runpod_Comfy/docker/Dockerfile .
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
