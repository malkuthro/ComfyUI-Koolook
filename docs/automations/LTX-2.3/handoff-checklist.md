# Handoff checklist — LTX 2.3 loop

Use this when picking up the loop on a different machine, or handing over to someone else. Target time-to-first-card: ~5 minutes.

## 1. Repo + branch

- [ ] Clone `ComfyUI-Koolook`.
- [ ] Check out the branch carrying the loop tooling — currently `investigate/ltx-director-4k-transitions`. (Eventually expected to merge to `main`.)
- [ ] Read [`README.md`](README.md) (this folder) and [`../CONVENTIONS.md`](../CONVENTIONS.md).

## 2. Python + Pillow

- [ ] Python 3.11+ available. On Windows: typically `C:\Python313\python.exe`.
- [ ] `pip install Pillow` (needed by `scripts/make_card.py`).
- [ ] Quick check: `python -c "from PIL import Image; print('ok')"`.

## 3. ComfyUI install

- [ ] LTX 2.3 model (`ltx-2.3-22b-dev-fp8` or equivalent), its CLIP (Gemma 3 12B), video VAE, audio VAE, distilled LoRA — see the **Base · model** section of any recent `_AI/card.png` for exact filenames.
- [ ] Custom nodes installed:
  - `WhatDreamsCost-ComfyUI` (provides `LTXDirector`, `LTXDirectorGuide`).
  - `comfyui-kjnodes` (`VisualizeSigmasKJ`, `ImageConcatMulti`).
  - `comfyui_layerstyle` (`LayerUtility: TextImage V2`, `LayerUtility: ImageBlend`).
  - `was-node-suite-comfyui` (`Text Multiline`).
  - `rgthree-comfy` (helpful for finding nodes by ID).
  - `pythongosssss/ComfyUI-Custom-Scripts` (optional — workflow auto-save).
- [ ] An LTX-2.3 workflow with the three required Text Multiline nodes (see `../CONVENTIONS.md` §2): `Working_Folder_PATH`, `OVERLAY - INFO`, `OVERLAY - FEEDBACK`.

## 4. Working folder

- [ ] Decide a path for this project (e.g. `<job>/ComfyUI-working-folder/`). Folder must exist.
- [ ] Put the path in **two places**, identical:
  - In ComfyUI: the `Working_Folder_PATH` Text Multiline node in the workflow.
  - In the repo: copy `.env.example` to `.env` (gitignored) and set `KOLOOK_AUTOMATIONS_WORK_DIR=<your path>`. Forward slashes work on Windows too. The `Working_Folder_PATH` node wins if both are set and differ.

## 5. First card

- [ ] In ComfyUI, save the workflow into the working folder via `Workflow → Save (API Format)`.
- [ ] (Optional) Render once so an MP4 lands in the folder.
- [ ] Trigger the agent skill: type `/make-card` (or "make card").
- [ ] Expect: a new `_AI/` subfolder appears in the working folder, containing `card.png` and `iterations.md`. The PNG is shown inline in chat.

If the card doesn't appear or fields are wrong: see "Troubleshooting" below.

## 6. Iterate

For each run:
1. Tweak one knob in ComfyUI (Phase 2 denoise, scheduler, etc.).
2. Update `OVERLAY - INFO` so the new state is captured.
3. Save the workflow + queue the render.
4. After watching the video, fill `OVERLAY - FEEDBACK` with notes + score lines (`motion: 4/5`).
5. `/make-card`. Drop `_AI/card.png` next to the video in your NLE.
6. Compare with the previous card. Repeat.

## 7. Promote findings

When a knob's behaviour is stable across runs, promote it from `_AI/iterations.md` (inside the working folder) into [`findings.md`](findings.md) (this repo). That keeps the locked-in stuff in the repo where it belongs, rather than buried in the per-project log.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Card shows `Run ?` | JSON filename lacks both `v01`-style and `_00001` patterns | Rename the saved JSON to match either pattern. |
| Resolution wrong on card | Selector INTConstant + Switch chain disagrees with what was rendered | See `CONVENTIONS.md` §3 — the card reflects the *workflow's* setting. Re-save after flipping the selector. |
| `Render time —` always | JSON and MP4 saved by ComfyUI's metadata-bundle have identical mtimes | Expected. The duration can't be derived from filesystem timestamps in this mode — left blank to avoid lying. |
| Outcome scores all `?/5` | `OVERLAY - FEEDBACK` doesn't contain score lines | Add lines like `motion: 4/5`. Case-insensitive; `/5` optional. |
| Card writes to wrong folder | `Working_Folder_PATH` node and `KOLOOK_AUTOMATIONS_WORK_DIR` env var are out of sync | Align both to the same path; the JSON node wins if present. |
| `python` not found on Windows | Microsoft Store stub | Use absolute `C:\Python313\python.exe` (already wired in `.claude/launch.json`). |

## Off-boarding (if you stop in the middle)

- **In the repo** — commit + push any `findings.md` updates and investigation-doc edits. Note in the final commit message what the last knob being swept was and what the next obvious step is.
- **In the working folder** — the JSON, MP4, and `_AI/` subfolder stay where they are (per-project, not in the repo). Make sure the path is in your `.env` so it's recoverable on the next machine.
- The next person reads this file, then `findings.md`, then the latest row of `_AI/iterations.md` in the working folder, then continues.
