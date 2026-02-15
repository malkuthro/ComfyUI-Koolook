# Deploy on RunPod

## 1) Build and push image

Use GitHub Container Registry or Docker Hub.

Example (GHCR):

```bash
BASE_IMAGE=$(grep '^BASE_IMAGE=' Runpod_Comfy/image/active/comfyui.lock | cut -d= -f2-)
IMAGE_NAME=$(grep '^IMAGE_NAME=' Runpod_Comfy/image/active/comfyui.lock | cut -d= -f2-)
IMAGE_TAG=$(grep '^RUNPOD_IMAGE_TAG=' Runpod_Comfy/image/active/comfyui.lock | cut -d= -f2-)
ENABLE_JUPYTER=$(grep '^ENABLE_JUPYTER=' Runpod_Comfy/image/active/comfyui.lock | cut -d= -f2-)
ENABLE_FILE_TOOLS=$(grep '^ENABLE_FILE_TOOLS=' Runpod_Comfy/image/active/comfyui.lock | cut -d= -f2-)

docker build \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  --build-arg ENABLE_JUPYTER="$ENABLE_JUPYTER" \
  --build-arg ENABLE_FILE_TOOLS="$ENABLE_FILE_TOOLS" \
  -t ${IMAGE_NAME}:${IMAGE_TAG} \
  -f Runpod_Comfy/core/docker/Dockerfile .

docker push ${IMAGE_NAME}:${IMAGE_TAG}
```

## 2) Create RunPod template

- Container image: `ghcr.io/malkuthro/comfyui-koolook:runpod-v1`
- Exposed HTTP ports: `8188` (ComfyUI), `8888` (JupyterLab)
- Volume mount: `/workspace` (recommended)

## 3) Optional env vars

- `COMFYUI_PORT=8188`
- `JUPYTER_PORT=8888`
- `JUPYTER_TOKEN=...` (default in image is `runpod`)
- `COMFYUI_WORKDIR=/workspace/ComfyUI`
- `HF_TOKEN=...` (if needed)
- `CIVITAI_TOKEN=...` (if needed)

## 4) First boot expectations

On first start:
- ComfyUI is cloned at pinned ref from `image/active/comfyui.lock`
- custom nodes are installed from `image/active/custom_nodes.lock.json`
- models are downloaded from `image/active/models.json` (enabled entries only)
- JupyterLab starts automatically when `ENABLE_JUPYTER=true`

Subsequent starts reuse existing files from volume storage.
