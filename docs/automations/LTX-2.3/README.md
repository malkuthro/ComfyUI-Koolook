# LTX 2.3 — evaluation loop

AI-assisted iteration loop for evaluating **Lightricks LTX 2.3** video generation behaviour. Currently focused on the WhatDreamsCost `LTXDirector` node — multi-segment prompts, keyframe transitions, audio sync, and high-resolution behaviour.

The shared loop pattern + node contract is in [`../CONVENTIONS.md`](../CONVENTIONS.md). Anything LTX-specific is here.

## What this loop is for

- Sweep one knob at a time (Phase 2 denoise, scheduler shape, resolution selector, etc.), render at the chosen seed, capture a tracking card next to the video.
- Build a body of evidence over multiple runs about *what each knob actually does at this model's training boundaries*.
- Promote stable findings to [`findings.md`](findings.md) so they don't get re-discovered.

The original investigation that produced this loop: [`../../investigations/ltx-director-4k-transitions.md`](../../investigations/ltx-director-4k-transitions.md).

## The loop (per iteration)

1. **In ComfyUI** — tweak one knob. Update the `OVERLAY - INFO` Text Multiline so the change is recorded in the JSON. After watching the rendered video, fill in the `OVERLAY - FEEDBACK` Text Multiline with observations + score lines (`motion: 4/5`, `sync: 5/5`, `sharp: 4/5`).
2. **Save workflow** — `Workflow → Save (API Format)` into the working folder (path from `Working_Folder_PATH` node).
3. **Render** — queue. MP4 lands in the same working folder, same basename.
4. **`/make-card`** — agent reads the newest JSON, writes `card.png` next to the video, appends a row to `iterations.md`.
5. **Composite** — drop `card.png` next to the video in your NLE (or wire `LoadImage` + `ImageConcatMulti` inside ComfyUI for a baked side-by-side).

Steps 1–3 are manual in ComfyUI. Step 4 is one command. Step 5 is one drag-and-drop per video.

## Workflow contract

The LTX 2.3 workflow JSON is read by `scripts/make_card.py`. The script handles both formats (raw editor JSON and the metadata-bundle JSON ComfyUI saves alongside output PNGs).

Fields extracted directly from nodes (no manual entry):

| Card field | Source node(s) | Notes |
|---|---|---|
| Format / Resolution | `LTXDirector.custom_width × custom_height` | Traced through `easy anythingIndexSwitch` if width/height are wired from a selector. Width-to-label map: 3744→4K, 3840→4K, 2560→2K, 1920→HD, 1280→HD. |
| Mult | First `LTXDirectorGuide.scale_by` whose `latent` input is fed by `LTXDirector` | Phase 1 guide. |
| Scheduler / Steps / Denoise | `BasicScheduler` for the chain whose denoise is `1.0` | Phase 1 scheduler. |
| Sampler | `KSamplerSelect` (locked to `euler` per findings). | Same chain. |
| CFG | `CFGGuider.cfg` | Locked to `1.0`. |
| Seed | `RandomNoise.noise_seed` | Held at `12 (fixed)` for reproducibility while sweeping. |
| ε (epsilon) | `LTXDirector.epsilon` widget. | Paper-default `0.001`. |
| Audio src | `LTXDirector.use_custom_audio` widget. | `model-gen` when False, `custom` when True. |
| Duration / FPS | `LTXDirector.duration_frames` × `1/frame_rate`. | |
| Segments | `LTXDirector.timeline_data` JSON → segments array. | Each segment's `[x] prompt · [x] audio · [x] keyframe` is checked. |
| Output / Render time | Newest video in working folder + filename match. | Render time is `mtime(video) − mtime(json)` if they don't share a basename; "—" otherwise (the metadata-bundle save writes both at end-of-render). |

Free-text fields (filled by you in the workflow):

| Card field | Source | Notes |
|---|---|---|
| Δ from base | `OVERLAY - INFO` → `BASE (notes):` section | What's different about this run. Optional. |
| Feedback | `OVERLAY - FEEDBACK` (non-score lines) | Post-render observations. |
| Outcome scores | `OVERLAY - FEEDBACK` (lines like `motion: 4/5`) | Each axis 0–5; missing axes show as `?/5`. |

## Active investigation

The 4K dissolve / audio drift problem. See [`findings.md`](findings.md) for what's already settled and [`../../investigations/ltx-director-4k-transitions.md`](../../investigations/ltx-director-4k-transitions.md) for the running narrative.

Open hypotheses being tested:
- Phase 2 denoise = 0.80 is too aggressive for "preserve stage 1" semantics; sweep down toward 0.30.
- LTX 2.3 prior weakens at 4K → 5K, producing dissolves in inter-keyframe motion.
- `LTXVLatentUpsampler [BETA]` may inject mouth-region detail that breaks audio sync.

## Where to find everything

| Thing | Path |
|---|---|
| Card renderer | [`scripts/make_card.py`](../../../scripts/make_card.py) |
| Watcher | [`scripts/watch_cards.py`](../../../scripts/watch_cards.py) |
| `/make-card` skill | [`.claude/skills/make-card/`](../../../.claude/skills/make-card/) |
| Cross-cutting conventions | [`../CONVENTIONS.md`](../CONVENTIONS.md) |
| Original investigation narrative | [`../../investigations/ltx-director-4k-transitions.md`](../../investigations/ltx-director-4k-transitions.md) |
| Working folder | `$KOLOOK_AUTOMATIONS_WORK_DIR` (set in `.env` — see `.env.example`) |
| Handoff to another machine / person | [`handoff-checklist.md`](handoff-checklist.md) |
