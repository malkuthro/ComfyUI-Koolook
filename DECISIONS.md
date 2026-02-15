# DECISIONS

## 2026-02-15 — Profile-first RunPod build flow
- We use profile files as inputs and generate lock files as build/runtime truth.
- Profile inputs:
  - `Runpod_Comfy/profiles/runpod.<id>.yaml`
  - `Runpod_Comfy/profiles/nodes.<id>.yaml`
  - `Runpod_Comfy/profiles/models.<id>.yaml`
  - Optional `Runpod_Comfy/profiles/urls.<id>.txt`
- Compiler output:
  - `Runpod_Comfy/config/comfyui.lock`
  - `Runpod_Comfy/config/custom_nodes.lock.json`
  - `Runpod_Comfy/config/models.json`
- Docker base image is taken from generated `comfyui.lock` (`BASE_IMAGE` build arg).

## 2026-02-15 — Work continuity model
- Planning/serious execution in OpenClaw web dashboard.
- Telegram for quick pings.
- Repo files (`TODO/WORKLOG/DECISIONS/WORKFLOW`) are continuity anchor across devices.
