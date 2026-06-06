# Visual harness for snapshot dialogs redesign (issue #137)

In-tree, dev-only harness pages that mount real `web/sidebar/` modules
in isolation so the agent (or maintainer) can verify each redesigned
dialog renders correctly against [`docs/designs/snapshot-dialogs.html`](../snapshot-dialogs.html)
without needing a live ComfyUI install.

**Status:** development-only. Not loaded by ComfyUI; not shipped to the
Comfy Registry archive (the registry scanner walks `__init__.py`-rooted
exports and `WEB_DIRECTORY = "./web"`, neither of which touches
`docs/designs/_harness/`).

## How it works

Each harness page is a standalone HTML file that:

1. Calls `ensureStyle()` from [`web/sidebar/constants.js`](../../../web/sidebar/constants.js)
   to inject the live sidebar CSS into the harness document.
2. Imports the real component under test from `web/sidebar/modals.js`
   (or wherever it lives) via ES-module imports — *no copying, no
   redefining*. What the harness renders is what ComfyUI renders.
3. Stubs only the integration surface — the `getSettings` /
   `browseDirectories` / `createBrowseDirectory` capability functions
   that the dialog receives as constructor args. The stubs use canned
   filesystem-shaped data so navigation, new-folder, etc. all work
   end-to-end inside the harness without a Python server beyond the
   static file host.
4. Exposes manual triggers (buttons in the harness header) for the
   states the matching mockup card illustrates (default, expanded,
   new-folder-input, error, …). The agent screenshots each state and
   diffs against the corresponding mockup card.

## Running

From the worktree root, start the static file server:

```bash
python3 -m http.server 8765
```

…or, when running under the Claude Code harness, the equivalent
`design-server` preset in [`.claude/launch.json`](../../../.claude/launch.json)
is launched via the preview tool.

Then open one of the harness pages alongside the mockup:

- Folder picker (mockup §6): http://127.0.0.1:8765/docs/designs/_harness/folder-picker.html
- Mockup reference (§6):     http://127.0.0.1:8765/docs/designs/snapshot-dialogs.html#
- Archive cleanup:          http://127.0.0.1:8765/docs/designs/_harness/archive-cleanup.html
- Publish setup dialog:     http://127.0.0.1:8765/docs/designs/_harness/publish-setup.html

`archive-cleanup.html` is not tied to a design mockup; it mounts the real
sidebar panel against canned workflow-store data so Archive row labels and
the Archive-folder context menu can be screenshot-checked without a live
ComfyUI install.

The harness pages auto-trigger the default state on load; the buttons
in the header cycle through the other states the mockup illustrates.

## When to add a new harness page

Whenever a commit's assertion lives at the rendered surface (CSS,
layout, wrapper structure) and the matching `docs/designs/` mockup
card has more than one state. One harness page per dialog component is
enough — additional states are buttons inside the same page, not
separate files.

## When NOT to use this

This is *component-level* verification — it confirms the redesigned
dialog renders against the mockup. It does **not** replace the
integration path:

- **Live ComfyUI behaviour** (snapshot save round-trip, server
  endpoints under real load, theme variables provided by Comfy's
  frontend) → still verified via `dev-sync` against a live install,
  per `docs/maintainers/dev-iteration-loop.md`.
- **End-to-end flow across multiple dialogs** (Save → Load round-trip
  picking up a file written in the previous step) → still verified via
  `dev-sync`.

Use the harness to confirm visual parity with the mockup; use
`dev-sync` to confirm integration with the rest of the sidebar.
