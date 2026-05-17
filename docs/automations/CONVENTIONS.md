# Conventions

Cross-cutting rules that every loop under `docs/automations/<model>/` shares. Edit here when something changes globally; per-model docs reference these.

## 1. The working folder

**One folder = one project = everything for one iteration.**

Structure:
```
<working folder>/
  <workflow>.json              ← user content (ComfyUI export)
  <workflow>.mp4               ← user content (rendered video)
  _AI/                         ← AI-managed — safe to keep / commit separately
    card.png                    ← tracking card overlay
    iterations.md               ← append-only log
```

- **`_AI/` subfolder** holds everything the agent writes. Wiping it never destroys user content; wiping the parent never touches the AI's history. Either side can be cleaned independently.
- Path lives in two places, kept in sync:
  - **Inside the workflow** — a Text Multiline node titled `Working_Folder_PATH` (also matches `OUT_working_folder`, `Work_Folder`, etc. — case-insensitive title contains "working_folder").
  - **In `.env`** (repo root, gitignored) — `KOLOOK_AUTOMATIONS_WORK_DIR=<path>`. See `.env.example` for the template.
- The `Working_Folder_PATH` node wins if both are present; the env var is the fallback used by `/make-card` to find the latest JSON when invoked without an argument.
- When switching projects: update the node, save the workflow into the new folder, update `KOLOOK_AUTOMATIONS_WORK_DIR` in `.env`. No script or doc edits.

## 2. The Text Multiline node contract

Three named nodes drive the card. Naming is matched by *title containing* (case-insensitive); exact strings below are recommended.

### `Working_Folder_PATH`
- Body: one line, the absolute working folder path (forward slashes preferred).
- Read by: `scripts/make_card.py` to determine where to write `_AI/card.png` and `_AI/iterations.md`.

### `OVERLAY - INFO`
- Body: a free-form summary of this run, plus a section marked `BASE (notes):` or `BASE:` whose content captures *Δ from baseline*.
- Example body:
  ```
  Phase 1
  Format: 2K
  Mult: 1
  Scheduler: linear_quadratic
  Steps: 8
  Denoise: 1

  BASE (notes):
  Changed Audio for each segment to match the duration
  ```
- The card extracts the `BASE (notes):` block as the "Δ from base" text. The other lines are informational — already captured from the actual workflow nodes.

### `OVERLAY - FEEDBACK`
- Body: post-render observations + score lines.
- Score syntax: `motion: 4/5`, `sync 5/5`, `sharp: 3`, `sharpness: 4` — case-insensitive, `/5` optional, integer 0–5.
- Non-score lines are kept as the feedback text on the card.
- Example body:
  ```
  Teeth are missing in some frames
  Very subtle loss of quality on Hi-speed movements
  motion: 4/5
  sync: 5/5
  sharp: 4/5
  ```

## 3. The card

- **Filename: `_AI/card.png`** inside the working folder. Stable — always overwrites. NLE points at it once.
- **Width: 540 px.** Vertical, sits beside a 16:9 video.
- **Sections:** Header → Phase 1 → Base · notes → Base · locked → Base · scene → Post-render (render time + output + feedback + outcome scores).
- **Palette:** mirrors `docs/designs/snapshot-dialogs.html` (panel `#151515`, panel-soft `#1a1a1f`, amber `#ffb84d`, sky `#6db4ff`, green `#7bcf80`).

Layout + extraction lives in `scripts/make_card.py`. Iterate on that file, not on per-model docs.

## 4. Render duration

Sourced from ComfyUI's own log (`<comfyui-root>/user/comfyui.log`), not from disk timestamps. The script pairs the latest `saving images: 100%` line with the next `Prompt executed in X` and shows that duration — wall-time for the actual render that wrote frames. Disk-based fallbacks (EXR sequence span, MP4 mtime delta) are used only if the log is unavailable.

Log path resolution:
1. `KOLOOK_COMFYUI_LOG` env var if set (explicit override).
2. Inferred from `KOLOOK_COMFYUI_DEV_PATH` → `<comfyui-root>/user/comfyui.log`.

## 5. The iterations log

- **File: `_AI/iterations.md`** inside the working folder.
- Append-only Markdown table. Newest at bottom.
- One row per call to `/make-card`. Dedupe key: `(json_filename, json_mtime)` — re-running on the same iteration won't double-log.
- Hidden HTML-comment fingerprint at end of each row enables the dedupe.
- Columns: `# · When · Run · Format · Denoise · JSON · Video · Δ from base · Feedback`.

## 6. The `/make-card` skill

- Trigger: `/make-card`, `card`, or `make card`.
- Reads `KOLOOK_AUTOMATIONS_WORK_DIR` from `.env` (repo root) for the working folder.
- Globs newest `*.json` in that folder; runs `scripts/make_card.py` against it.
- Writes `_AI/card.png` + appends `_AI/iterations.md` row.
- Returns the absolute path of the PNG + shows it inline.

## 7. The watcher (optional)

- `scripts/watch_cards.py` — leave running in a terminal during a session.
- Polls the working folder every 2 s.
- Re-renders `_AI/card.png` automatically whenever the JSON in the folder is modified.
- For hands-free operation alongside ComfyUI's "auto-save workflow on queue" setting (pythongosssss custom-scripts).

## 8. The `loop` filename marker

Files whose basename contains `loop` (case-insensitive) are treated as **post-card outputs** — the composited card+video that ComfyUI writes back to the working folder *after* `/make-card` has produced the card. Auto-discovery and the watcher both skip these:

- `LTX-Director_loop_00001.mp4`, `LTX-Director_loop_00001.json` → ignored.
- `LTX-Director_00002.mp4`, `LTX-Director_00002.json` → picked up normally.

Reason: without the skip, the script's "newest JSON in the folder" picks up its own downstream output, generating a stale card pointing at the previous iteration. Naming the post-card pass with `loop` in the basename keeps the auto-discovery clean.

If you need to force-render a card for a `loop`-named JSON, invoke the script explicitly: `python scripts/make_card.py <wf_loop.json>`.

## 9. Run identification

The card title's "Run" label is auto-derived from the JSON stem:
1. **`v01` / `v04` style** wins if present — gives `Run v04`.
2. Otherwise falls back to a **5-digit sequence number** like `_00007_` — gives `Run #7`.
3. Otherwise `Run ?`.

ComfyUI's metadata bundle JSONs (`<name>_<seq>.json`) work natively with #2.

## 10. Tools — where they live (and don't move)

| Tool | Path | Why here |
|---|---|---|
| Card renderer | `scripts/make_card.py` | All project scripts under `scripts/`. |
| File watcher | `scripts/watch_cards.py` | Same. |
| Skill | `.claude/skills/make-card/` | Claude Code expects skills under `.claude/skills/`. |
| Working folder env var | `.env` → `KOLOOK_AUTOMATIONS_WORK_DIR` | Project convention; see `.env.example`. |
| Per-project artifacts | `<working folder>/` | One path for everything. |
| Loop docs | `docs/automations/<model>/` | This folder. |
