# Cheat sheet

For someone who already knows the loop. Full docs: [`README.md`](README.md), [`CONVENTIONS.md`](CONVENTIONS.md), [`LTX-2.3/README.md`](LTX-2.3/README.md).

---

## ComfyUI

Three Text Multiline nodes drive everything. Match by title (case-insensitive).

| Node title | Edit when | Contents |
|---|---|---|
| `OVERLAY - INFO` | Before render | Free-form summary. The card pulls the `BASE (notes):` section as "Δ from baseline." |
| `OVERLAY - FEEDBACK` | After watching the video | Observations + score lines (see syntax below). |
| `Working_Folder_PATH` | Only when switching projects | One absolute path, forward slashes. |

**Score syntax** inside `OVERLAY - FEEDBACK`:

```
motion: 4/5
sync: 5/5
sharp: 4/5
```

Case-insensitive · `/5` optional · integer 0–5 · non-matching lines stay as feedback text.

**Workflow save:** `Workflow → Save (API Format)` → save into the working folder. The metadata-bundle JSON ComfyUI writes (`<name>_<seq>.json`) works natively.

---

## Scripts

| Command | What it does |
|---|---|
| `/make-card` | Auto-discover newest JSON in working folder → render card → show inline. |
| `python scripts/make_card.py` | Same as `/make-card`, from a terminal. |
| `python scripts/make_card.py <wf.json>` | Render a specific JSON. |
| `python scripts/make_card.py <wf.json> <out.png>` | Render to a custom output path. |
| `python scripts/watch_cards.py` | Hands-free background poller — re-renders the card on every JSON save. |

On Windows, replace `python` with `C:/Python313/python.exe` if the Microsoft Store stub gets in the way.

---

## Folder structure

### Repo-side (where the tools and docs live)

```
ComfyUI-Koolook/
├── .env                                gitignored — your config
│   KOLOOK_AUTOMATIONS_WORK_DIR=…
├── .env.example                        committed — template
├── .claude/skills/make-card/
│   └── SKILL.md                        the /make-card command
├── scripts/
│   ├── make_card.py                    card renderer
│   └── watch_cards.py                  optional file watcher
└── docs/automations/
    ├── README.md                       overview + start-here
    ├── CHEATSHEET.md                   this file
    ├── CONVENTIONS.md                  full contract
    └── LTX-2.3/                        one folder per model
        ├── README.md
        ├── handoff-checklist.md
        └── findings.md
```

### Working-folder-side (per project — pointed at by `KOLOOK_AUTOMATIONS_WORK_DIR`)

```
<working folder>/
├── <workflow>.json                     ← user content — save here from ComfyUI
├── <workflow>.mp4                      ← user content — ComfyUI writes here
├── …more JSONs/MP4s as you iterate
└── _AI/                                ← agent-managed
    ├── card.png                        stable name — overwritten each run
    └── iterations.md                   append-only log, one row per render
```

Wipe `_AI/` to reset the agent's tracking history without touching renders. Wipe the JSONs/MP4s without losing the iteration log.

---

## Conventions

- **One JSON = one iteration.** Don't overwrite the same JSON across substantively different runs.
- **Card filename is stable.** Wire your NLE to `_AI/card.png` once.
- **Seed = 12 (fixed)** while sweeping any other knob. Vary seed only after a setting stabilises.
- **Scheduler = `linear_quadratic` (8 steps)** — locked-in finding. See [`LTX-2.3/findings.md`](LTX-2.3/findings.md).
- **Promote** a hypothesis to a finding when it's stable across ≥ 3 runs in `_AI/iterations.md`.
- **Run label** on the card: prefers `v01`-style; falls back to `_00001` sequence numbers; else `?`.
- **Switching projects:** update `Working_Folder_PATH` node + `.env`. Node wins on mismatch.

---

## Quick-fail diagnostics

| Card says… | Means… |
|---|---|
| `Run ?` | Filename has neither `v<N>` nor `_<NNNNN>`. Rename or accept the `?`. |
| `Render time —` | JSON and MP4 mtimes are identical (ComfyUI bundle save). Expected — not a bug. |
| `Outcome … ?/5` | `OVERLAY - FEEDBACK` has no `motion:` / `sync:` / `sharp:` lines yet. |
| Card writes to wrong folder | `Working_Folder_PATH` and `.env` disagree — node wins; align them. |
| `KOLOOK_AUTOMATIONS_WORK_DIR not set` | Copy `.env.example` to `.env`, fill the var. |
