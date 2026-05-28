# Handoff checklist — LTX 2.3 audio-lipsync automation

Use this when picking up this automation on a different machine, or handing
over to someone else. Target time-to-first-render: ~5 minutes after the
sibling [`../base-1step/handoff-checklist.md`](../base-1step/handoff-checklist.md)
prerequisites are met.

The base-1step handoff covers shared prerequisites (Python + Pillow + the
working folder + Text Multiline nodes + most custom-node packs). Do that
first, then come back here.

## 1. Extra ComfyUI deps

This automation drives a modified `LTXDirector`, so beyond the base-1step
custom-node list you also need:

- [ ] **`WhatDreamsCost-ComfyUI` installed.** The Koolook fork only ports
  `LTXDirector`; the companion `LTXDirectorGuide`, `LTXKeyframer`,
  `MultiImageLoader`, `LTXSequencer`, `SpeechLengthCalculator`,
  `LoadAudioUI`, `LoadVideoUI` come from upstream. Install via
  ComfyUI-Manager or `git clone https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI` into `custom_nodes/`.
- [ ] **The Koolook fork is loaded automatically.** No extra step — it ships
  inside this repo at
  [`../../../../forks/whatdreamscost_koolook/`](../../../../forks/whatdreamscost_koolook/)
  and is wired into [`../../../../__init__.py`](../../../../__init__.py). On
  ComfyUI startup you should see both `LTX Director` (upstream) and
  `LTX Director (Koolook v1.3.2)` in the node picker.

## 2. Workflow file

- [ ] Save the audio-lipsync workflow as
  `<workflow root>/LTX-23-audio_tests_v01.json` (or whatever filename your
  working folder uses). The Director node must be
  **`LTX Director (Koolook v1.3.2)`** (registered ID
  `LTXDirector__koolook_v1_3_2`), not the upstream `LTX Director`.
- [ ] On the Koolook Director, the `relay_overrides` widget sits at the
  bottom of the node. Start with an empty string for a regression check
  against upstream-default behaviour, then iterate.

## 3. First render

- [ ] Queue once. Confirm in chat with the maintainer what the audio /
  sync / motion looked like.
- [ ] Agent will capture the snapshot into `runs/run-001_<label>/` and
  append a row to [`runs/log.md`](runs/log.md).

If the Director node still shows the upstream `LTXDirector` after loading
the workflow:

- Check the ComfyUI startup log for a `[Koolook] node registry skipped` or
  `from .forks.whatdreamscost_koolook import …` ImportError — that means
  the fork didn't load (transitive import failure). Run pytest at the repo
  root to see the failure outside ComfyUI.
- Verify `pip show -f WhatDreamsCost-ComfyUI` or equivalent — the Koolook
  fork imports through `comfy_api`, which only exists in a real ComfyUI
  process.

## 4. Iteration

See [`README.md`](README.md) (this folder) for the iteration loop and
[`runs/LOOP.md`](runs/LOOP.md) for the per-render protocol.

## Off-boarding

- Commit + push `findings.md` updates and any fork-code edits you made
  this session.
- The per-project working folder (workflow JSON + MP4 + `_AI/`) stays
  where it is, outside the repo. Make sure
  `KOLOOK_AUTOMATIONS_WORK_DIR` in `.env` points at it so the next
  machine picks up the same iterations log.
