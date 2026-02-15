ARG BASE_IMAGE=runpod/pytorch:2.6.0-py3.11-cuda12.4.1-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    COMFYUI_PORT=8188 \
    COMFYUI_WORKDIR=/workspace/ComfyUI \
    RUNPOD_VOLUME_ROOT=/workspace

WORKDIR /opt/bootstrap

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates jq \
    && rm -rf /var/lib/apt/lists/*

COPY Runpod_Comfy /opt/bootstrap/Runpod_Comfy

RUN chmod +x /opt/bootstrap/Runpod_Comfy/scripts/*.sh /opt/bootstrap/Runpod_Comfy/docker/start.sh

EXPOSE 8188

ENTRYPOINT ["/opt/bootstrap/Runpod_Comfy/docker/start.sh"]
