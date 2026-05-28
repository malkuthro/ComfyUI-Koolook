# Automations

AI-managed ComfyUI iteration loops. Each automation module is one
self-contained workflow: own ComfyUI session, own workflow JSON, own
narrative, own findings — typically focused on one specific generation
task (a single-stage render path, an audio-driven lip-sync, an upscale
recipe, etc.). The same diffusion model can host several modules side
by side; one folder per **task**, not one per model.

The agent's role inside each module is the same: watch the maintainer's
workflow edits, score feedback, render tracking cards via `/make-card`,
and — when the task needs it — modify code in `forks/` or `scripts/`.

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
├── forks/                                   ← any modified upstream node code an
│                                            automation needs (e.g. whatdreamscost_koolook/)
└── docs/automations/
    ├── README.md                            you are here
    ├── CONVENTIONS.md                       cross-cutting loop contract
    ├── CHEATSHEET.md                        one-page reminder
    └── LTX-2.3/                             grouping by diffusion model
        ├── base-1step/                       ← one automation module
        │   ├── README.md                     loop entry point for this task
        │   ├── handoff-checklist.md
        │   ├── findings.md                   locked-in conclusions for THIS task
        │   ├── backstory/                    narrative that produced the module
        │   └── example-cards/                rendered cards from past runs
        └── audio-lipsync/                    ← another automation module, same model
            ├── README.md
            ├── handoff-checklist.md
            ├── findings.md
            ├── backstory/
            └── runs/                         per-render snapshots in the repo
                ├── LOOP.md                   per-task iteration protocol
                └── log.md                    rolling table of runs
```

The model-name folder (`LTX-2.3/`) is **only a grouping** — it doesn't
carry shared docs of its own. Each automation under it is independent
(own README, own findings). Model-architecture facts that genuinely
apply to every module on that model can live in any one module's
`findings.md` and be cross-linked from siblings (or lifted to a shared
file later if a third module materialises and the duplication is real).

### Per-project working folder (lives outside the repo)

Pointed at by `KOLOOK_AUTOMATIONS_WORK_DIR` in `.env`. One folder per
project. Holds:

```
<working folder>/                            e.g. JOBS/<client>/<scene>/ComfyUI-working-folder
├── <workflow>.json                          ← user content — ComfyUI workflow export
├── <workflow>.mp4                           ← user content — rendered video
├── …more JSONs/MP4s as you iterate
└── _AI/                                     ← AI-managed — written by make_card.py
    ├── card.png                             stable tracking card (overwritten each run)
    └── iterations.md                        append-only log, one row per render
```

The `_AI/` split lets you wipe agent artifacts without touching user
content (or vice-versa). You can delete `_AI/` to reset the history;
you can delete the JSONs and MP4s without losing the iteration log.

For automations that ship in-repo `runs/` snapshots (the
audio-lipsync module does this — because the iteration touches
fork code, not just widgets), the snapshots live inside the module
folder. The `_AI/iterations.md` outside-the-repo log is still the
authoritative running record; the in-repo `runs/log.md` is the
condensed cross-machine summary.

## What lives here

| Module | Model | Status | Task |
|---|---|---|---|
| [`LTX-2.3/base-1step/`](LTX-2.3/base-1step/README.md) | LTX 2.3 | active | Single-stage render: brief once, sample 8 steps, no Phase-2 upscale. Model-generated audio. |
| [`LTX-2.3/audio-lipsync/`](LTX-2.3/audio-lipsync/README.md) | LTX 2.3 | active | Audio-file-driven lip-sync. Modified `LTXDirector` (Prompt-Relay paper σ + `relay_overrides` widget). Ships a fork at [`forks/whatdreamscost_koolook/`](../../forks/whatdreamscost_koolook/). |

## The pattern (every module looks like this)

1. **One working folder per project** holds everything for the iteration: workflow JSON + rendered video at the root (user content), plus an `_AI/` subfolder for the agent-managed card + iterations log.
2. **Three named Text Multiline nodes** inside the ComfyUI workflow form the data contract:
   - `Working_Folder_PATH` — authoritative output path.
   - `OVERLAY - INFO` — the run params summary + free-text `BASE (notes):` for what changed this run.
   - `OVERLAY - FEEDBACK` — observations on the rendered video + score lines (`motion: 4/5`, `sync: 5/5`, `sharp: 4/5`).
3. **The `/make-card` skill** (or `scripts/make_card.py` directly) reads the latest JSON in the working folder, extracts every parameter, renders `_AI/card.png`, and appends a row to `_AI/iterations.md`. Cards travel alongside the video into the NLE for side-by-side comparison.
4. **Code modifications go in `forks/`** when a module needs to alter an upstream node. The agent edits the fork file in the repo; the maintainer runs `dev-sync` to copy it into the live install. See [`../../forks/README.md`](../../forks/README.md).
5. **Findings get promoted** from per-run notes into the module's `findings.md` when a knob's behaviour is stable enough to lock in (≥ 3 confirming runs).

Full cross-cutting contract: [`CONVENTIONS.md`](CONVENTIONS.md).

## Starting a new automation module

1. Pick a folder name — short, lowercase, hyphen-separated, descriptive of the *task* (e.g. `LTX-2.3/two-stage-upscale/`, `wan-2.1/i2v-from-still/`). The model-name grouping is optional but recommended once there are 2+ modules on the same model.
2. Copy `LTX-2.3/base-1step/README.md` and `handoff-checklist.md` to the new folder as templates.
3. Rewrite the *task scope* paragraph at the top. Replace task-specific node names, parameter sets, and finding placeholders with the new module's.
4. Add a row to the "What lives here" table above.
5. If the module needs to modify upstream node code, run the
   [`add-external-fork`](../../.claude/skills/add-external-fork/) skill
   first to register the fork properly; then the module can reference
   `forks/<name>_koolook/`.
6. The cross-cutting tooling (`scripts/make_card.py`, `.claude/skills/make-card/`) is module-agnostic — no changes needed there unless the new module needs different fields extracted from its workflow.
