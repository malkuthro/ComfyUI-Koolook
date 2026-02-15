# Deploy on RunPod

## 1) Build and push image

Use GitHub Container Registry or Docker Hub.

Example (GHCR):

```bash
docker build -t ghcr.io/malkuthro/comfyui-koolook:runpod-v1 -f Runpod_Comfy/core/docker/Dockerfile .
docker push ghcr.io/malkuthro/comfyui-koolook:runpod-v1
```

## 2) Create RunPod template

- Container image: `ghcr.io/malkuthro/comfyui-koolook:runpod-v1`
- Exposed HTTP port: `8188`
- Volume mount: `/workspace` (recommended)

## 3) Optional env vars

- `COMFYUI_PORT=8188`
- `COMFYUI_WORKDIR=/workspace/ComfyUI`
- `HF_TOKEN=...` (if needed)
- `CIVITAI_TOKEN=...` (if needed)

## 4) First boot expectations

On first start:
- ComfyUI is cloned at pinned ref from `config/comfyui.lock`
- custom nodes are installed from `config/custom_nodes.lock.json`
- models are downloaded from `config/models.json` (enabled entries only)

Subsequent starts reuse existing files from volume storage.
