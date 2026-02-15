# Runpod_Comfy Setup Guide

This package is split into 3 readable zones:

- `setup/` → what you edit (recipes + helper tools)
- `image/` → generated outputs (build artifacts + active locks) + deploy runbook
- `core/` → runtime internals (Docker + bootstrap scripts + technical references)

## Quick Start

### 1) Create a recipe

```bash
Runpod_Comfy/setup/tools/new_profile.sh upscaler-v1
```

Creates:
- `Runpod_Comfy/setup/recipes/upscaler-v1/runpod.yaml`
- `Runpod_Comfy/setup/recipes/upscaler-v1/nodes.yaml`
- `Runpod_Comfy/setup/recipes/upscaler-v1/models.yaml`
- `Runpod_Comfy/setup/recipes/upscaler-v1/urls.txt`

In `runpod.yaml`, feature defaults are ON:
- `features.enable_jupyter: true`
- `features.enable_file_tools: true`
Set either to `false` to disable.

### 2) Compile recipe

```bash
Runpod_Comfy/setup/tools/compile_profile.sh --id upscaler-v1
```

Generates into:
- `Runpod_Comfy/image/builds/upscaler-v1/comfyui.lock`
- `Runpod_Comfy/image/builds/upscaler-v1/custom_nodes.lock.json`
- `Runpod_Comfy/image/builds/upscaler-v1/models.json`

By default compile also **activates** to:
- `Runpod_Comfy/image/active/comfyui.lock`
- `Runpod_Comfy/image/active/custom_nodes.lock.json`
- `Runpod_Comfy/image/active/models.json`

### 3) Build image

```bash
BASE_IMAGE=$(grep '^BASE_IMAGE=' Runpod_Comfy/image/active/comfyui.lock | cut -d= -f2-)
IMAGE_TAG=$(grep '^RUNPOD_IMAGE_TAG=' Runpod_Comfy/image/active/comfyui.lock | cut -d= -f2-)
ENABLE_JUPYTER=$(grep '^ENABLE_JUPYTER=' Runpod_Comfy/image/active/comfyui.lock | cut -d= -f2-)
ENABLE_FILE_TOOLS=$(grep '^ENABLE_FILE_TOOLS=' Runpod_Comfy/image/active/comfyui.lock | cut -d= -f2-)

docker build \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  --build-arg ENABLE_JUPYTER="$ENABLE_JUPYTER" \
  --build-arg ENABLE_FILE_TOOLS="$ENABLE_FILE_TOOLS" \
  -t comfyui-koolook:${IMAGE_TAG} \
  -f Runpod_Comfy/core/docker/Dockerfile .
```

By default both `ENABLE_JUPYTER` and `ENABLE_FILE_TOOLS` are `true`.

## Iterate safely

- Minor tweaks: keep same recipe id, bump `image_tag` in `runpod.yaml`
- Major changes: clone to new id (`upscaler-v2`, `wan22-v1`)
- Optional no-activate compile:

```bash
Runpod_Comfy/setup/tools/compile_profile.sh --id upscaler-v1 --no-activate
```

## Deploy

RunPod deploy runbook:
- `Runpod_Comfy/image/DEPLOY_RUNPOD.md`

## Compiler args

```bash
Runpod_Comfy/setup/tools/compile_profile.sh --id <recipe-id> [--out <dir>] [--image-tag <tag>] [--no-activate]
```
