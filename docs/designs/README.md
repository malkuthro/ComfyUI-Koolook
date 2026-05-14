# Design Proposals

Small, reviewable UI mockups for decisions that benefit from visual comparison
before they land in the runtime sidebar.

| File | Purpose |
|---|---|
| [`sidebar-icon-proposals.html`](sidebar-icon-proposals.html) | Side-by-side icon direction mockup for the Kforge Labs sidebar action rows |
| [`sidebar-icon-proposals.png`](sidebar-icon-proposals.png) | Rendered preview of the current icon proposal state |
| [`snapshot-dialogs-redesign.md`](snapshot-dialogs-redesign.md) | Scope sketch + Mermaid flow for consolidating Settings cog into Save / Load dialogs (+ autosave-newest fix) |
| [`snapshot-dialogs.html`](snapshot-dialogs.html) | Side-by-side mockups: current vs. proposed Snapshot toolbar, Save dialog, Load dialog, and autosave YES/NO modal |

## How these mockups are used

These files are the **visual spec** for any implementing PR. When code
lands that implements one of these designs, the agent doing the work
must:

1. Render the implementation in a real browser — `dev-sync` to the live
   ComfyUI install, or a local static server (e.g.
   `python3 -m http.server`) for standalone HTML.
2. Screenshot every state shown in the corresponding mockup (default,
   hover, expanded, selected, error, …).
3. Visually diff the screenshots against the mockup before requesting
   review. Either match the design or document the deviation with a
   rationale.

The maintainer is never the first reviewer of rendered output — visual
verification is a precondition for review, not something to ask the
maintainer to do. See the [project CLAUDE.md](../../CLAUDE.md) under
*"Visual verification for design-driven implementation"* for the full
rule.
