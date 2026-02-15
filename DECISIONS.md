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
- Goal: normal updates (new node repo/model URL/version bumps) should require editing config files only, not Python/Docker logic.
- Expected operator UX:
  - Edit 3–4 simple profile files with real values.
  - Run compiler.
  - Build/push image.
  - Deploy and smoke-test on RunPod.

## 2026-02-15 — Work continuity model
- Planning/serious execution in OpenClaw web dashboard.
- Telegram for quick pings.
- Repo files (`TODO/WORKLOG/DECISIONS/WORKFLOW`) are continuity anchor across devices.
