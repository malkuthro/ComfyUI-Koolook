---
name: make-card
description: Render the LTX Director experiment-tracking card PNG from the latest workflow JSON in the working folder. One-shot. Reads the working folder path from the `KOLOOK_AUTOMATIONS_WORK_DIR` environment variable (loaded from `.env` at the repo root). Use when the maintainer types `/make-card`, `card`, or `make card` after exporting a workflow JSON from ComfyUI alongside a rendered video. Skill picks the newest `*.json` in the folder, runs `scripts/make_card.py`, writes `card.png` next to the JSON, and shows the PNG inline.
---

# /make-card

Per-iteration card renderer for the LTX Director investigation
(branch `investigate/ltx-director-4k-transitions`).

## Behaviour

1. Run a single command:
   ```
   python scripts/make_card.py
   ```
   With no args, the script auto-loads `.env` (worktree first, then main
   repo via git common-dir if running from a worktree), reads
   `KOLOOK_AUTOMATIONS_WORK_DIR`, finds the newest `*.json` in that
   folder, and writes `card.png` into the same folder.

   On Windows, resolve `python` to `C:/Python313/python.exe` if a plain
   `python` invocation fails.

2. End the turn by Reading the rendered PNG so it shows inline in the
   chat. Always include the absolute path of the PNG on the line above
   the inline image. No other prose unless something failed.

## Failure modes

- **`KOLOOK_AUTOMATIONS_WORK_DIR` unset** — tell the maintainer to add
  it to `.env` (see `.env.example` for the template).
- **Working folder missing on disk** — surface the path and tell the
  maintainer to update `KOLOOK_AUTOMATIONS_WORK_DIR` or create the
  directory.
- **No JSON in folder** — tell the maintainer to export the workflow
  JSON from ComfyUI into that folder.
- **Multiple JSONs with same mtime** — pick alphabetically last (matches
  `v01`/`v02`/`v03` versioning when timestamps tie).
- **`make_card.py` errors** — surface stderr verbatim; do not silently
  fall back to a stale PNG.

## Maintainer notes

- The script reads `OVERLAY - INFO` and `OVERLAY - FEEDBACK` Text Multiline
  nodes from the workflow JSON for the card's note boxes + outcome scores.
  Keep those multiline nodes in the workflow and update them per render
  before saving the JSON. See `automations/CONVENTIONS.md` for the full
  contract.
- To switch projects, change `KOLOOK_AUTOMATIONS_WORK_DIR` in `.env`. No
  code or skill edits needed.
- Card layout + extraction live in `scripts/make_card.py`. Iterate on
  that file, not on this skill.
