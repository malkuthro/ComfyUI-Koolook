# Visual harness — agent-driven UI verification

The in-tree visual harness mounts the real `web/sidebar/` modules in
isolation so an agent (or maintainer) can verify rendered UI against a
[`docs/designs/`](../designs/) mockup *without* running a full ComfyUI
install. Component-level visual parity becomes a self-contained loop:
edit → reload → screenshot → diff.

This complements (does not replace) the dev-sync integration loop in
[`dev-iteration-loop.md`](dev-iteration-loop.md). See *When to use the
harness vs. dev-sync* below.

## Files

| Path | Role |
|---|---|
| [`.claude/launch.json`](../../.claude/launch.json) | Defines the `design-server` preview config — `python3 -m http.server 8765` from the worktree root. The Claude Code preview tool reads this; running by hand from a terminal works equivalently. |
| [`docs/designs/_harness/`](../designs/_harness/) | One HTML file per dialog under verification. Each is a standalone page that mounts the corresponding component from `web/sidebar/`. |
| `web/sidebar/<module>.js` | Source-of-truth components. The harness imports these directly — no copies, no redefining. |
| `docs/designs/<feature>.html` | The locked design spec. Open side-by-side with the harness page for visual diff. |

## How a harness page is wired

Each page in `docs/designs/_harness/`:

1. Imports `ensureStyle` from `web/sidebar/constants.js` and calls it.
   That injects the live sidebar stylesheet so what renders is byte-
   for-byte the same CSS ComfyUI loads.
2. Imports the component under test from its real location (e.g.
   `showFolderPicker` from `web/sidebar/modals.js`). All transitive
   imports resolve naturally via the static server — no module
   stubbing needed for code that doesn't touch ComfyUI globals.
3. Stubs **only** the capability functions the component receives as
   constructor args (`browseDirectories`, `createBrowseDirectory`,
   etc.). The stubs use canned filesystem-shaped data so navigation
   round-trips end-to-end inside the harness with no Python beyond
   the static file host.
4. Auto-mounts the default state on page load. The page header
   carries buttons that cycle through the additional states the
   matching mockup card illustrates (default / new-folder-input /
   empty / error / …). One button per mockup-illustrated state.
5. Cache-busts the live-module imports with `?bust=` + load time so
   editing `modals.js` while the harness is open picks up the fresh
   code on `window.location.reload()` instead of serving the browser's
   module cache. This is dev-only — the harness directory is never
   shipped (filtered out of the Comfy Registry publish, ignored by the
   sidebar loader).

## Running

From the worktree root:

```bash
python3 -m http.server 8765
```

Then open both pages in the same browser tab group:

- Harness: `http://127.0.0.1:8765/docs/designs/_harness/<feature>.html`
- Mockup:  `http://127.0.0.1:8765/docs/designs/<feature>.html`

Edit → save → switch to the harness tab → `⌘R` / `Ctrl-R`. The harness
imports re-fetch from the static server; the new code renders.

When invoked via the Claude Code preview tool, the equivalent flow is
`preview_start({ name: "design-server" })` followed by `preview_eval`
(navigation) and `preview_screenshot`. Resize the viewport to at least
1000 × 700 via `preview_resize` so the picker / dialogs aren't
collapsed into the harness panel's narrow default.

## Adding a harness page for a new dialog

1. Pick the matching mockup card in `docs/designs/<feature>.html`.
   Read the surrounding annotations — they specify state names
   (default, hover, error, …) that become the harness buttons.
2. Copy the structure of an existing harness page (e.g.
   `folder-picker.html`) and adapt:
   - Replace the imported component with yours.
   - Replace the canned stubs with the capability surface your
     component expects.
   - Add one trigger button per state the mockup shows. Auto-mount
     the default state on load.
3. Verify the harness renders matching the mockup before opening the
   PR. Note in the PR description "visual diff vs. mockup section N"
   for each cycle that touched rendered output.

## When to use the harness vs. dev-sync

**Use the harness for** any change whose assertion lives at the rendered
surface and can be exercised in isolation:

- Single-dialog layouts, state transitions inside one component, CSS
  changes, copy changes, icon swaps.
- Stubs cover all the integration surface the component touches.
- Iteration loop is ~5 seconds (edit → reload → screenshot).

**Use dev-sync for** anything that needs a full ComfyUI environment:

- Cross-dialog flows (Save dialog opens → Load dialog reflects the
  written file in the next session).
- Theme variables that come from Comfy's frontend chrome (light/dark
  toggle, custom palette overrides).
- End-to-end snapshot library round-trip against a real preset folder
  on disk.
- Any sidebar interaction with ComfyUI's canvas, graph state, or
  keyboard manager.

A common pattern: harness during implementation to nail the layout,
dev-sync once before opening the PR to confirm it still looks right
under real Comfy chrome.

## What the harness does NOT prove

- That the component's integration with the rest of the sidebar still
  works. (Use dev-sync.)
- That theme tokens resolve the same way under real Comfy theming.
  The harness fakes `--comfy-menu-bg` etc. with dark-theme defaults;
  Comfy's actual values may differ slightly.
- That the component renders the same way at narrow widths or on
  mobile breakpoints. Test those explicitly via `preview_resize`.

If your change might touch any of the above, follow harness
verification with a dev-sync pass before requesting review.

## Pitfalls

- **Backticks in CSS comments break the entire sidebar.** The CSS in
  [`web/sidebar/constants.js`](../../web/sidebar/constants.js) is a JS
  template literal. A literal backtick inside a comment closes the
  template early, the rest of the file becomes invalid syntax, and the
  whole sidebar fails to load. Use plain quotes in comments, never
  backticks. (There's a sentinel comment in `constants.js` calling this
  out — read it before adding new CSS.)
- **Stale module cache.** Without the harness's cache-busting query
  string, the browser may serve the previous version of the module
  even after a reload. The harness handles this automatically; if you
  hand-write a one-off harness, mirror the `?bust=` + `Date.now()`
  pattern.
- **The harness is dev-only.** Do not import harness HTML or its stubs
  from any code under `web/` or `koolook_routes.py`. The harness lives
  under `docs/designs/_harness/` precisely so it's outside the Comfy
  Registry archive scanner's walk.
