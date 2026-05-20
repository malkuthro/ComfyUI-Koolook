# Snapshot merge / branch review — design sketch

Working sketch for collaborative snapshot editing without silent
overwrites. Two maintainers work from a shared starting point, end up
with diverging snapshots, then review and selectively merge each
other's work.

Pixel mockup lives in [`snapshot-merge.html`](snapshot-merge.html).
Discussion-only — no issue cut yet, no implementation.

## The shape of the problem

A Kforge Labs snapshot is already a structured JSON object (`web/sidebar/snapshot.js`):

```js
{
  kind: "koolook-snapshot",
  version: 1,
  name: "Wan22 VFX kit",
  exportedAt: "2026-05-14T09:12:00.000Z",
  picks: ["easy_pattern", "GetImageSizeAndCount", ...],   // a set of node IDs
  workflows: { directories: { "VFX": { workflows: {...}, directories: {...} } } }
}
```

The two halves merge with very different rules:

- **`picks`** is a set of strings. Set difference is the entire diff:
  *in mine and not theirs* / *in theirs and not mine* / *in both*.
  No three-way conflict possible.
- **`workflows`** is a recursive folder tree of opaque graph JSON.
  Folder adds/removes merge cleanly by path. *File* edits on both sides
  are unmergeable at the graph level — workflows are arbitrary node
  graphs, line-merging them is unsafe. The v1 answer is "keep both,
  rename theirs."

This split is the load-bearing insight: the merge UI is essentially
two flat diff lists (picks set diff, workflow tree diff) plus a small
conflict pane for the rare "same workflow edited on both sides" case.

## Storage model (v1 → v3 escalation)

The mockup is intentionally **storage-agnostic**. The same UI surface
works at every level of the ladder:

| Version | Storage | Sharing | Auth |
|---|---|---|---|
| **v1** | Plain JSON files in the snapshot library. Each save adds `lineage_id` + `parent_hash` + `author` to the envelope. | File-by-file: send `Wan22 VFX kit (Alice).json` over Discord / Drive / email; drop it into your `koolook-presets/` folder. | None — file trust matches the current "open this workflow JSON" trust model. |
| **v2** | Snapshot library *is* a git repo. Saves are commits; sharing is push/pull to a remote (GitHub, Gitea, self-hosted). | `git pull`, surfaced in the Load dialog as branches. | Inherited from git remote (SSH key / PAT). |
| **v3** | Koolook backend service: snapshots indexed by lineage, hosted diff/merge endpoints, per-branch URLs. | Discover branches in-app; "request review" notifications. | Real accounts. |

**Start at v1.** It buys ~80% of the user's "review and click to merge"
experience without committing to git plumbing or a server. v2/v3 plug
in underneath the *same* review dialog — that's the design constraint.

## Merge flow

```mermaid
flowchart TD
  A[Alice exports her snapshot<br/>(envelope carries lineage_id + parent_hash)] --> B[Sends file to you]
  B --> C[You drop file into koolook-presets/<br/>OR click Import branch…]
  C --> D{Same lineage_id<br/>as a local preset?}
  D -- No --> E[Import as a new, unrelated preset<br/>(no merge UI, no diff)]
  D -- Yes --> F[Attach as a branch child<br/>of the matching preset row]
  F --> G[User clicks Review… on the branch row]
  G --> H[Review dialog opens with<br/>side-by-side header + diff sections]
  H --> I{Each row decided?}
  I -- Adds ticked, removes left,<br/>conflicts resolved --> J[Click Merge selected]
  J --> K{Target?}
  K -- Save into mine --> L[Write merged snapshot over current preset<br/>(pre_load_*.json autosave written first)]
  K -- Save as new branch --> M[Write merged snapshot under a new name<br/>(yours stays untouched)]
  L --> N[markStateSaved · green dot]
  M --> N
```

The pre-merge autosave (`pre_load_*.json` in `<preset>_autosave/`) is
the safety net — same recovery surface as a regular Load, exposed
through the existing Recovery dropdown from PR #144. **No new recovery
mechanism needed.**

## Decisions baked into the mockup

- **Adds default ON, removes default OFF.** The common case is
  "pick up the new stuff Alice added"; deleting picks you kept on
  purpose is rarer and should be deliberate.
- **Conflicts default to "keep both."** Workflow graphs aren't safely
  text-mergeable. Preserving Alice's variant under a renamed file
  (`comp_v02 (Alice).json`) gives the user a no-data-loss path and
  lets them resolve in the canvas afterwards.
- **Folder-level toggle cascades to children.** Ticking
  `Color science/` ticks both child files. Unticking a child puts the
  folder into a partial state. Matches the model the Comfy frontend
  workflow tree already uses.
- **The Load dialog grows a `▾` disclosure** under any preset row
  that has incoming branches matching its `lineage_id`. Same visual
  pattern as the recovery row from PR #144 — no new container.
- **Branches discovered by filename or sidecar.** v1: scan
  `koolook-presets/` for files whose envelope `lineage_id` matches a
  local preset. v2: branches live in git refs. v3: branches come back
  from the service API. Same dialog, three transports.

## Out of scope (for this sketch)

- **Transport.** Drag-drop / paste / git pull / service fetch — all
  decided later; the review UI doesn't care which.
- **Identity.** `author` is whatever the file says. Real auth is v2+.
- **Three-way graph merge** of conflicting workflows. v1 ships
  "keep both, rename theirs."
- **Multi-branch merge.** One incoming branch per review pass.
- **Live co-editing / CRDT.** Different feature entirely.
- **History tree view.** Could be a separate surface later; not part
  of the merge flow.

## Open questions before scoping

1. **Lineage seed.** When does a snapshot acquire a `lineage_id`?
   Options: (a) on first save (random UUID), (b) on first "Share…"
   action, (c) every save (default). Picks the answer to: "what
   counts as 'the same snapshot' vs 'two unrelated things'?"
2. **Discovery surface.** Should branches in the same library
   auto-attach, or does the user explicitly click <em>Import
   branch…</em> for each? Auto-attach is friendlier but mixes
   user-trusted and untrusted files in one list. Explicit import is
   safer but adds a step.
3. **Where does the conflict pane live?** Inline below the workflows
   tree (mockup §6, scrollable) or as a focused sub-modal that opens
   per-conflict? Inline keeps everything on one screen; sub-modal
   lets us show a bigger preview of the two workflow graphs.
4. **Cross-library merge.** Does the dialog work if Yours and Theirs
   live in *different* library folders (e.g. each maintainer has their
   own folder)? The schema doesn't care, but the discovery story
   needs to handle it.
5. **Version drift.** What happens if Theirs is `version: 2` and
   Yours can only read `version: 1`? Today's import path rejects
   newer versions outright. Merge should refuse with a clear message
   ("update plugin to merge this branch") rather than silently drop
   fields.
