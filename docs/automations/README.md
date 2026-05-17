# Automations

AI-assisted iteration loops for evaluating generative models inside ComfyUI.

Each subfolder is one model under study. The loop pattern is the same across all of them; the model-specific knobs, conventions, and findings live in the subfolder.

> **Want the visual walkthrough?** → [`index.html`](index.html) presents the setup and loop as an HTML overview.
>
> **Already know the loop?** → [`CHEATSHEET.md`](CHEATSHEET.md) (one page, dense bullets).

## Where things live

### In the repo

```
ComfyUI-Koolook/
├── .env                                     gitignored — your machine's config
│                                            (KOLOOK_AUTOMATIONS_WORK_DIR=…)
├── .env.example                             committed template — documents the var
├── .claude/skills/make-card/
│   └── SKILL.md                             the `/make-card` slash command
├── scripts/
│   ├── make_card.py                         renders a card PNG from workflow JSON
│   └── watch_cards.py                       optional polling watcher
└── docs/automations/
    ├── README.md                            you are here
    ├── CONVENTIONS.md                       cross-cutting rules
    └── LTX-2.3/                             one folder per model under study
        ├── README.md                        loop entry point for this model
        ├── handoff-checklist.md             5-minute onboarding
        └── findings.md                      locked-in conclusions
```

### Per-project working folder (lives outside the repo)

Pointed at by `KOLOOK_AUTOMATIONS_WORK_DIR` in `.env`. One folder per project. Holds:

```
<working folder>/                            e.g. JOBS/<client>/<scene>/ComfyUI-working-folder
├── <workflow>.json                          ← user content — ComfyUI workflow export
├── <workflow>.mp4                           ← user content — rendered video
├── …more JSONs/MP4s as you iterate
└── _AI/                                     ← AI-managed — written by make_card.py
    ├── card.png                             stable tracking card (overwritten each run)
    └── iterations.md                        append-only log, one row per render
```

The `_AI/` split lets you wipe agent artifacts without touching user content (or vice-versa). You can delete `_AI/` to reset the history; you can delete the JSONs and MP4s without losing the iteration log.

## What lives here

| Subfolder | Status | Scope |
|---|---|---|
| [`LTX-2.3/`](LTX-2.3/README.md) | active | Lightricks LTX 2.3 video diffusion — Director node behaviour, 4K transitions, audio sync |

## The pattern (every loop looks like this)

1. **One working folder per project** holds everything for one iteration: workflow JSON + rendered video at the root (user content), plus an `_AI/` subfolder for the agent-managed card + iterations log.
2. **Three named Text Multiline nodes** inside the ComfyUI workflow form the data contract:
   - `Working_Folder_PATH` — authoritative output path.
   - `OVERLAY - INFO` — the run params summary + free-text `BASE (notes):` for what changed this run.
   - `OVERLAY - FEEDBACK` — observations on the rendered video + score lines (`motion: 4/5`, `sync: 5/5`, `sharp: 4/5`).
3. **The `/make-card` skill** (or `scripts/make_card.py` directly) reads the latest JSON in the working folder, extracts every parameter, renders `_AI/card.png`, and appends a row to `_AI/iterations.md`. Cards travel alongside the video into the NLE for side-by-side comparison.
4. **Findings get promoted** from per-run notes into the model's `findings.md` when a knob's behaviour is stable enough to lock in.

Full cross-cutting contract: [`CONVENTIONS.md`](CONVENTIONS.md).

## Starting a new loop (new model)

1. Pick a folder name — short, lowercase, hyphen-separated (e.g. `wan-2.1`, `audio-ldm-3`).
2. Copy `LTX-2.3/README.md` and `LTX-2.3/handoff-checklist.md` to the new folder as templates.
3. Replace LTX-specific node names, parameter sets, and finding placeholders with the new model's.
4. The cross-cutting tooling (`scripts/make_card.py`, `.claude/skills/make-card/`) is model-agnostic — no changes needed there unless the new model needs different fields extracted.
