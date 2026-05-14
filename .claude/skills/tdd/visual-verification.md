# Visual verification — ComfyUI-Koolook

The secondary procedure when a test cycle's assertion lives at the *test
surface* (className strings, inline styles, structural wrappers) but its
correctness lives at the *user-facing surface* (rendered pixels in the
running ComfyUI sidebar, or in a standalone HTML mockup).

This file complements, doesn't replace, the test cycle. Trigger when
the cycle's assertion is anything in
[SKILL.md § "When test green isn't enough"](SKILL.md).

## Why this matters here specifically

ComfyUI runs in a browser tab inside a host app, with its own theming
and chrome. The sidebar is layered onto a UI surface this project
doesn't fully control — Comfy's frontend renderer can change a margin
or a class precedence between releases, breaking how the sidebar looks
without changing any of our code. Test-green-only verification can't
catch that: only seeing the running result against the mockup can.

Same applies to:

- **Snapshot dialogs** — modal + sub-modals layered on Comfy's dialog
  stack.
- **Workflows / Picks rows** — inside a Comfy sidebar tab, subject to
  Comfy's row chrome.
- **Folder picker** — a custom modal sitting on top of Comfy's modal
  layer.

## Trigger conditions

Apply when the cycle's assertion is any of:

- A CSS class string (`hidden`, `selected`, theme tokens like
  `--sky`/`--green`/`--amber`)
- An inline `style={}` value
- A layout property (`position`, `padding`, `margin`, `z-index`,
  `top`/`bottom`/`left`/`right`)
- Section / wrapper structure ("recovery section auto-expands under the
  clicked preset", "badge on top, meta line below")
- Anything in the AC that says "looks like", "matches the mockup",
  "scoped to", or "above/below/beside"

If unsure whether the cycle qualifies, it qualifies.

## The procedure — runtime sidebar work

For changes to `web/koolook_sidebar.js` or any module under
`web/sidebar/`:

1. **Dev-sync the change** —
   `python scripts/sync_to_dev.py --scope "<10-word summary>"`. Copies
   the worktree's runtime files into `KOLOOK_COMFYUI_DEV_PATH` (a live
   ComfyUI install set per-machine in `.env`).
2. **Restart ComfyUI.** The sidebar's footer (`dev <sha> · <time>` +
   italic scope from `_dev_build.json`) confirms the sync landed. If
   you don't see the new scope, the sync didn't apply — diagnose before
   continuing.
3. **Navigate to the affected feature** through the normal UI route —
   Kforge Labs sidebar, Snapshot dialog, Folder picker, etc. Reach the
   state the mockup illustrates. Don't fake the state from a stub or
   devtools tweak; the chrome around the feature is part of what you're
   verifying.
4. **Take a screenshot.** Use the harness's preview screenshot tool, or
   a native screenshot of the actual browser window.
5. **Diff against the design source**, in priority order:
   - `docs/designs/<feature>.html` — the locked design spec (e.g.
     [`snapshot-dialogs.html`](../../../docs/designs/snapshot-dialogs.html))
   - The matching `docs/designs/<feature>-redesign.md` scope sketch
   - Inline mockups in the issue body
6. **Resolve contradictions in favour of the design source.** If the
   AC's English description says "above the recovery section" but the
   mockup card shows the row *inside* the recovery section's nested
   panel, the mockup wins. The English is a pointer; the design page
   is the spec.

## The procedure — standalone HTML mockups

When iterating on a mockup itself in `docs/designs/*.html` (not
implementation), the audience is the maintainer's stated intent rather
than a higher-up design doc — but the self-check rule still applies:
the maintainer is never the first set of eyes on the rendered result.

Two render options that don't require new package installs (per the
project's supply-chain rules):

- **Harness preview panel** — auto-reloads on every edit. Default for
  agent-driven iteration.
- **`python3 -m http.server 8080 --directory docs/designs/`** — open
  `http://127.0.0.1:8080/<file>.html` in any real browser. Full
  devtools, OS zoom, etc. Requires `⌘R` after each save (no auto-
  reload).

If you want auto-reload in a real browser, `live-server` works but
needs an explicit audit + install pass — the auto-classifier blocks
`npx --yes live-server` because it pulls a package without audit.

## What "diff against the mockup" actually means

It's not pixel-equality. It's a structured check against the mockup's
explicit decisions:

- **Spatial:** is the element in the same relative position (above /
  below / inside / outside the same container)?
- **Hierarchy:** does the DOM nesting match the mockup's nesting? (A
  row "inside the recovery panel" must be a child of that panel, not a
  sibling.)
- **States:** every state the mockup shows (default, hover, selected,
  expanded, error, in-progress) must render correctly in the
  implementation, not just the default state.
- **Chrome:** padding, border, background, divider lines, badge
  positioning — the mockup's small decisions are decisions. If you
  deviate, document the reason; don't silently change them.
- **Text:** labels and button text match the mockup exactly. The
  mockup is authoritative for copy.

If the implementation deviates on any of these and you intend to keep
the deviation, note it explicitly in the PR description with a
rationale. The maintainer can accept or reject; what they can't do is
notice a silent deviation later.

## Don't ship without this

A green CI on `ruff check .` + `python -m compileall ...` + `bandit` +
`tools/preflight_release.py` does **not** prove a frontend feature
renders correctly. Those gates verify syntax, lint, security, and node-
registration consistency — none of them open a browser.

If a mockup exists for the change you're implementing, the maintainer
should never be the first set of eyes on the rendered result. Visual
verification is a hard gate, not optional polish.

## Tiebreaker rule (one line)

**Design source > AC English. The English points at the design; the
design is the spec.**
