# LTX 2.3 audio-lipsync — per-render iteration protocol

**Automation:** [LTX 2.3 audio-lipsync](../README.md)
**Tracking issue:** [malkuthro/ComfyUI-Koolook#177](https://github.com/malkuthro/ComfyUI-Koolook/issues/177)
**Started:** 2026-05-28

---

## The loop

| Step | Who | What |
|---|---|---|
| 1 | Maintainer | Either edit `relay_overrides` JSON on the **`LTX Director (Koolook)`** node in ComfyUI, OR edit a `forks/whatdreamscost_koolook/versions/v1_3_9/*.py` source in this repo + run **`dev-sync-audio`** (auto-restarts Comfy). |
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
├── workflow.json           ← copy of the working-folder workflow at submission
├── relay_overrides.txt     ← the LTX Director (Koolook) node's relay_overrides widget value
├── patch_state.txt         ← MAIN SHA + dev-sync SHA + whether forks/.../v1_3_9/*.py differs from MAIN
└── notes.md                ← maintainer's verbal feedback + agent's one-line interp
```

The `relay_overrides` value is also stored inside `workflow.json` (Koolook
LTXDirector node widget), but the `.txt` copy makes it diff-friendly in the
run folder.

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
- Auto-triggers a ComfyUI-Manager reboot so the Python module re-imports
  (custom-node `.py` files load once at server start; without a restart
  the new code stays invisible). Use `--no-restart` to opt out.
- Writes the scope tag into `<target>/web/_dev_build.json` so the
  Kforge Labs sidebar footer shows which build is live.

Flags mirror `dev-sync`: `--dry-run`, `--init`, `--verbose`,
`--scope "<≤10-word change>"`, `--no-restart`, `--restart-url`.

User-initiated only — same rule as `dev-sync` (see project `CLAUDE.md`).
Never automatic. Never on commit, never on session end.

Widget-only changes (editing `relay_overrides` JSON on the canvas) do
**not** require sync or restart — just queue the next render.

Why not full `dev-sync`? Because it `rmtree`s the entire dest `forks/`
tree and re-copies both Koolook forks. For this module's iteration
loop, that churns `forks/radiance_koolook/` for no reason. Scoping to
`forks/whatdreamscost_koolook/` keeps the radiance side of the live
install stable across audio-lipsync iterations.
