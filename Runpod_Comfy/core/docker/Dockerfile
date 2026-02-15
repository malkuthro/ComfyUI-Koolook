ARG BASE_IMAGE=runpod/pytorch:2.6.0-py3.11-cuda12.4.1-devel-ubuntu22.04
ARG ENABLE_JUPYTER=true
ARG ENABLE_FILE_TOOLS=true
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    COMFYUI_PORT=8188 \
    JUPYTER_PORT=8888 \
    COMFYUI_WORKDIR=/workspace/ComfyUI \
    RUNPOD_VOLUME_ROOT=/workspace

WORKDIR /opt/bootstrap

ARG ENABLE_JUPYTER
ARG ENABLE_FILE_TOOLS

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates jq \
    && if [ "$ENABLE_FILE_TOOLS" = "true" ]; then apt-get install -y --no-install-recommends zip unzip p7zip-full rsync; fi \
    && rm -rf /var/lib/apt/lists/*

RUN if [ "$ENABLE_JUPYTER" = "true" ]; then python3 -m pip install --no-cache-dir jupyterlab; fi

COPY Runpod_Comfy /opt/bootstrap/Runpod_Comfy

RUN chmod +x /opt/bootstrap/Runpod_Comfy/core/scripts/*.sh /opt/bootstrap/Runpod_Comfy/core/docker/start.sh

EXPOSE 8188 8888

ENTRYPOINT ["/opt/bootstrap/Runpod_Comfy/core/docker/start.sh"]
