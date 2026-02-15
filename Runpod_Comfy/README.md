# Runpod_Comfy

Production-oriented scaffold for running ComfyUI on RunPod with pinned versions.

## Structure

- `base/` → stable scaffold area (reserved for core runtime layout)
- `admin_recipes/` → **your editable inputs** (one recipe per image variant)
- `builds/` → generated lock outputs per recipe
- `config/` → active lock set consumed by runtime scripts
- `docker/` → image/runtime entrypoint
- `scripts/` → bootstrap installers
- `tools/` → recipe helpers (`new_profile.sh`, `compile_profile.sh`)

## Quick Start

### 1) Create a recipe

```bash
Runpod_Comfy/tools/new_profile.sh upscaler-v1
```

This creates:

- `admin_recipes/upscaler-v1/runpod.yaml`
- `admin_recipes/upscaler-v1/nodes.yaml`
- `admin_recipes/upscaler-v1/models.yaml`
- `admin_recipes/upscaler-v1/urls.txt` (optional helper)

### 2) Compile recipe

```bash
Runpod_Comfy/tools/compile_profile.sh --id upscaler-v1
```

Generated outputs:

- `builds/upscaler-v1/comfyui.lock`
- `builds/upscaler-v1/custom_nodes.lock.json`
- `builds/upscaler-v1/models.json`

By default, compile also **activates** this recipe by copying outputs to `config/`.

### 3) Build image

```bash
BASE_IMAGE=$(grep '^BASE_IMAGE=' Runpod_Comfy/config/comfyui.lock | cut -d= -f2-)
IMAGE_TAG=$(grep '^RUNPOD_IMAGE_TAG=' Runpod_Comfy/config/comfyui.lock | cut -d= -f2-)
docker build --build-arg BASE_IMAGE="$BASE_IMAGE" \
  -t comfyui-koolook:${IMAGE_TAG} -f Runpod_Comfy/docker/Dockerfile .
```

### 4) Iterate safely

- Minor tweaks: keep same recipe folder, bump `image_tag` in `runpod.yaml`.
- Major changes: clone recipe to new id (`upscaler-v2`, `wan22-v1`).
- You can compile without activation using `--no-activate`.

## Compiler arguments

```bash
Runpod_Comfy/tools/compile_profile.sh --id <recipe-id> [--out <dir>] [--image-tag <tag>] [--no-activate]
```

Backward compatible:

```bash
Runpod_Comfy/tools/compile_profile.sh <recipe-id>
```
