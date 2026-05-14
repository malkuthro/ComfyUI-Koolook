# Snapshot dialogs — redesign sketch

Working sketch for consolidating the Snapshot Settings cog into the Save
and Load dialogs, plus the related "load the newest autosave, whichever
file it is" behaviour.

Pixel mockups live in [`snapshot-dialogs.html`](snapshot-dialogs.html).
Implementation tracked in [issue #137](https://github.com/malkuthro/ComfyUI-Koolook/issues/137).

## What changes

1. **Toolbar:** the `pi pi-cog` Settings button is removed from the Snapshot
   row. The row becomes: status indicator · Load · Quick Save · Save.
2. **Save dialog:** library path inlined as a top-of-body info row
   ("Saved to · Open folder ↗" with leaf folder name + full path).
   `Save to…` in the bottom command bar opens a folder picker that
   persists the library path via `saveSettings`. The primary Save
   button cycles through four states (Default / Dirty / In progress /
   Done) per mockup section 5.
3. **Load dialog:** mirrors the Save dialog — same "Loaded from · Open
   folder ↗" top row, `Load from…` in the bottom command bar opening
   the same folder picker.
4. **Autosave-newer affordance:** the YES/NO choice modal is **dropped**.
   Instead, when a preset has a newer autosave (server row-augment
   reports `latestAutosaveMtime > mtime`), clicking that preset row
   in the Load dialog expands a single scoped recovery row directly
   beneath it. The row carries a kind badge (Pre-load / Periodic) and
   the timestamp + meta of the newest entry across `periodic.json`
   and every `pre_load_*.json` in `<preset>_autosave/`. The user
   chooses by clicking: re-click the named row to load named, click
   the recovery row to restore the autosave.
5. **Inline delete confirm:** the per-preset `×` button no longer
   opens a confirm modal; instead it outlines the target row red and
   transforms the Close button to *"Yes — delete '\<name\>'"* (danger
   styling). A second click commits; Escape cancels and reverts.

## Load flow (post-redesign)

```mermaid
flowchart TD
  A[Click preset row] --> B{latestAutosaveMtime &gt; mtime<br/>(server row-augment)?}
  B -- No --> C[Direct load<br/>(pre-load auto-save written first)]
  B -- Yes --> D[Expand scoped recovery row<br/>under the clicked preset]
  D --> E{User clicks…}
  E -- Named preset row again --> C
  E -- Scoped recovery row --> F[Restore newest autosave<br/>(periodic.json or pre_load_*.json)]
  E -- Elsewhere / Esc --> G[Collapse scoped recovery]
  C --> H[markStateSaved · green dot]
  F --> I[markStateAutosaved · blue dot]
```

Compared to the pre-#137 design (a YES/NO modal that interrupted the
flow), the inline scoped row keeps both options visible alongside the
preset list. The user picks by clicking the row they want — no extra
modal layer, no Esc-to-cancel ambiguity, no double-confirm.

## Out of scope (deliberately, for this PR)

- Multi-library support
- Per-snapshot library binding
- Bulk migrate-presets-to-new-folder action
- Recent-libraries dropdown

The single-library design is documented intent ([curated-sidebar.md
§ library directory](../maintainers/curated-sidebar.md)). The redesign
only consolidates surface, it does not change the model.
