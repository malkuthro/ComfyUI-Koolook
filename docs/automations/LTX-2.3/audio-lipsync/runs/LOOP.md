# LTX 2.3 audio-lipsync — per-render iteration protocol

**Automation:** [LTX 2.3 audio-lipsync](../README.md)
**Tracking issue:** [malkuthro/ComfyUI-Koolook#177](https://github.com/malkuthro/ComfyUI-Koolook/issues/177)
**Started:** 2026-05-28

---

## The loop

| Step | Who | What |
|---|---|---|
| 1 | Maintainer | Either edit `relay_overrides` JSON on the **`LTX Director (Koolook)`** node in ComfyUI, swap to upstream **`LTX Director`** for an original-vs-Koolook comparison, OR edit a `forks/whatdreamscost_koolook/versions/v1_3_9/*.py` source in this repo + run **`dev-sync-audio`**, then restart ComfyUI manually. |
| 2 | Maintainer | **Save** workflow → overwrites the current workflow file at the working folder. |
| 3 | Maintainer | Queue render in ComfyUI. |
| 4 | Maintainer | Report result in chat (verbal feedback — sync state, motion state, prompt adherence). |
| 5 | Agent | Append a row to [`log.md`](log.md). |
| 6 | Agent | Snapshot the current state into a matching `run-NNN_<label>/` folder under `runs/`. |
| 7 | Agent | Confirm what was captured (`log.md` row + folder if one was made). |

**Save-before-render** is the anchor: the workflow file on disk at submission time = exactly what produced the render being described. No copy-paste, no version drift.

**Why this retention policy.** For this fork-touching loop, every `loop-audio`
capture is kept as a reproducibility snapshot: workflow JSON,
relay_overrides, fork state, and notes. The rolling [`log.md`](log.md)
stays the skim view; the folders carry the evidence.

## Saved-run snapshot contents

Each `loop-audio` capture folder looks like:

```
run-NNN_<label>/
├── runNNN_workflow.json    ← redacted copy of the working-folder workflow at submission
├── relay_overrides.txt     ← RELAY_OVERRIDES multiline body; marked inert when upstream Director is active
├── patch_state.txt         ← MAIN SHA + dev-sync SHA + whether forks/.../v1_3_9/*.py differs from MAIN
├── metadata.json           ← structured run/setup/director/repo metadata
├── notes.md                ← maintainer feedback + SETUP variables + director state
└── card.png                ← stable archive card
```

The archived workflow keeps the Comfy graph shape but redacts absolute
workstation paths before it is committed. The `relay_overrides` value is
also stored inside `runNNN_workflow.json` when the
Koolook Director is active, but the `.txt` copy makes it diff-friendly in the
run folder. On upstream `LTXDirector` comparison runs, the `.txt` file keeps
the same note body and explicitly marks that value as inert.

When card rendering is enabled, the loop also writes a delivery copy into the
setup output folder's `cards/` subfolder as
`cards/<Output name>_runNNN_card.png`. The run folder keeps only the stable
archive `card.png`; the long filename belongs to the external delivery copy
so the run snapshot stays compact.

The visual card shows the original copied workflow stem (without `.json`)
under BASE/RUN, not the predictable archive filename. Technical details live
in `metadata.json` and the PNG metadata. The Director section shows both the
active flavor and the lock pin tag; for upstream comparison runs that pin tag
is read automatically from the live `WhatDreamsCost-ComfyUI/pyproject.toml`
beside `$KOLOOK_COMFYUI_DEV_PATH`.

Re-render an existing run through the script, not by copying files manually:

```powershell
.\.venv\Scripts\python.exe scripts\make_card_audio.py docs\automations\LTX-2.3\audio-lipsync\runs\run-NNN_<label>
```

That scripted path updates `card.png`, rewrites `metadata.json`, refreshes
`notes.md`, and replaces the external
`cards/<Output name>_runNNN_card.png` delivery copy. Use `--no-delivery`
when refreshing committed evidence without touching the render drive.

## Trigger detection

| Maintainer chat input | Agent action |
|---|---|
| Says **`loop-audio`** after a saved render | Append the `log.md` row and create `run-NNN_<label>/` with the full snapshot. |
| Says "no log" / "don't log this" before capture | Skip the command — pure scratch render. |

The maintainer can override the generated label with `loop-audio --label ...`
when a run needs a specific name.

## Run labels (when a run gets saved)

Auto-generated from key knob state + feedback summary, e.g.:

- `run-001_no-override_baseline-fork-only`
- `run-002_vstr10_no-change`
- `run-003_vstr25_some-motion-recovered`
- `run-004_vstr10_astr0.5_sync-partial`
- `run-005_codechange_sigma-bumped_sync-clean`

`log.md` row numbers increment on every render; saved-folder NNN reuses that row number, so `run-003_…` corresponds to row 3 in `log.md`. The maintainer can override any label by stating it explicitly.

## Syncing fork-code edits to the live install — `dev-sync-audio`

When the iteration step requires editing a `forks/whatdreamscost_koolook/`
source (not just the widget), run the **scoped** sync for this module —
not the full `dev-sync`:

```
python scripts/sync_to_dev_audio.py
# or just say "dev-sync-audio" in chat — the agent runs the script
```

What it does:
- Copies only `forks/whatdreamscost_koolook/`,
  `web/whatdreamscost_koolook/`, and the root `__init__.py` to
  `$KOLOOK_COMFYUI_DEV_PATH` (read from `.env`).
- Removes the stale pre-v1.3.9 `web/whatdreamscost_koolook_v1_3_2/`
  folder if present, because leaving both web extensions loaded mounts
  two identical timeline editors on legacy workflows.
- Leaves the rest of the live install (radiance fork, the root `k_*.py`
  nodes, the sidebar `web/` bundle, `video_formats/`) untouched.
- Copies files only. Restart ComfyUI manually after Python changes so
  custom-node modules re-import.
- Writes the scope tag into `<target>/web/_dev_build.json` so the
  Kforge Labs sidebar footer shows which build is live.

Flags mirror `dev-sync`: `--dry-run`, `--init`, `--verbose`,
`--scope "<≤10-word change>"`.

User-initiated only — same rule as `dev-sync` (see project `CLAUDE.md`).
Never automatic. Never on commit, never on session end.

Widget-only changes (editing `relay_overrides` JSON on the canvas) do
**not** require sync or restart — just queue the next render.

Why not full `dev-sync`? Because it `rmtree`s the entire dest `forks/`
tree and re-copies both Koolook forks. For this module's iteration
loop, that churns `forks/radiance_koolook/` for no reason. Scoping to
`forks/whatdreamscost_koolook/` keeps the radiance side of the live
install stable across audio-lipsync iterations.
