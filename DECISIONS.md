# DECISIONS

## 2026-02-15 — Profile-first RunPod build flow
- We use profile files as inputs and generate lock files as build/runtime truth.
- Profile/recipe inputs:
  - `Runpod_Comfy/setup/recipes/<id>/runpod.yaml`
  - `Runpod_Comfy/setup/recipes/<id>/nodes.yaml`
  - `Runpod_Comfy/setup/recipes/<id>/models.yaml`
  - Optional `Runpod_Comfy/setup/recipes/<id>/urls.txt`
- Compiler output:
  - `Runpod_Comfy/image/builds/<id>/comfyui.lock`
  - `Runpod_Comfy/image/builds/<id>/custom_nodes.lock.json`
  - `Runpod_Comfy/image/builds/<id>/models.json`
  - Activated to `Runpod_Comfy/image/active/*`
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

## 2026-02-15 — Recipe architecture for long-term readability
- Use a 3-zone layout inside `Runpod_Comfy/`:
  - `setup/` (operator-facing edits/tools; keep lightweight)
  - `image/` (generated outputs + active locks + deploy runbook)
  - `core/` (runtime internals + technical references)
- Keep one recipe folder per image variant under `setup/recipes/<id>/`.
- Compiler supports explicit args (`--id`, `--out`, `--image-tag`, `--no-activate`).
- Default outputs go to `image/builds/<recipe-id>/` and can be activated to `image/active/`.
