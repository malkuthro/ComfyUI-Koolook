---
name: make-card
description: Render the LTX Director experiment-tracking card PNG from the latest workflow JSON in the working folder. One-shot. Reads the working folder path from `.claude/skills/make-card/work-folder.txt`. Use when the maintainer types `/make-card`, `card`, or `make card` after exporting a workflow JSON from ComfyUI alongside a rendered video. Skill picks the newest `*.json` in the folder, runs `scripts/make_card.py`, writes `<stem>_card.png` next to the JSON, and shows the PNG inline.
---

# /make-card

Per-iteration card renderer for the LTX Director investigation
(branch `investigate/ltx-director-4k-transitions`).

## Behaviour

1. Read the working folder path from
   `.claude/skills/make-card/work-folder.txt` (one line, no trailing slash needed).
2. Glob `*.json` in that folder; pick the one with the most recent
   `mtime`. If none, abort with a clear error pointing the maintainer
   at the working folder.
3. Run:
   ```
   python scripts/make_card.py <latest.json> <latest.stem>_card.png
   ```
   Resolve `python` to `C:/Python313/python.exe` on Windows if a plain
   `python` invocation fails.
4. Save the PNG **into the same working folder** so the rendered video,
   workflow JSON, and card live side-by-side per iteration.
5. End the turn by Reading the rendered PNG so it shows inline in the
   chat. Always include the absolute path of the PNG on the line above
   the inline image. No other prose unless something failed.

## Failure modes

- **Working folder missing** — surface the path and tell the maintainer
  to update `work-folder.txt`.
- **No JSON in folder** — tell the maintainer to export the workflow
  JSON from ComfyUI into that folder.
- **Multiple JSONs with same mtime** — pick alphabetically last (matches
  `v01`/`v02`/`v03` versioning when timestamps tie).
- **`make_card.py` errors** — surface stderr verbatim; do not silently
  fall back to a stale PNG.

## Maintainer notes

- The script reads `OVERLAY - INFO` and `OVERLAY - FEEDBACK` Text Multiline
  nodes from the workflow JSON for the card's note boxes. Keep those
  multiline nodes in the workflow and update them per render before
  saving the JSON.
- To change the working folder (e.g. switching projects), edit
  `.claude/skills/make-card/work-folder.txt`. No need to touch this
  SKILL.md.
- Card layout + extraction live in `scripts/make_card.py`. Iterate on
  that file, not on the skill.
