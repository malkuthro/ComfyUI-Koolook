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
   The earlier scoped-row click design is also superseded. Final
   decision: when a preset has a newer autosave (server row-augment
   reports `latestAutosaveMtime > mtime`), selecting that preset keeps
   the user in the same Load dialog, changes the header to
   *"Auto-save is newer than the saved version"*, opens the bottom
   Recovery auto-saves section for that preset only, and shows explicit
   footer choices: *"NO - load saved"* and *"YES - load latest"*. The
   recovery row carries a kind badge (Pre-load / Periodic) and the
   timestamp + meta of the newest entry across `periodic.json` and
   every `pre_load_*.json` in `<preset>_autosave/`, but row clicks
   select only; footer buttons commit.
5. **Inline delete confirm:** the per-preset `×` button no longer
   opens a confirm modal; instead it outlines the target row red and
   transforms the Close button to *"Yes"* (danger styling). A second
   click commits; Escape or clicking another row cancels and reverts.
   Delete buttons stay visually neutral until hover.

## Load flow (post-redesign)

```mermaid
flowchart TD
  A[Click preset row] --> B{latestAutosaveMtime &gt; mtime<br/>(server row-augment)?}
  B -- No --> C[Select row<br/>footer action becomes Load]
  B -- Yes --> D[Change title + open bottom Recovery section<br/>for this preset only]
  D --> E{User clicks…}
  E -- NO - load saved --> F[Load named snapshot<br/>(pre-load auto-save written first)]
  E -- YES - load latest --> G[Restore newest autosave<br/>(periodic.json or pre_load_*.json)]
  E -- Another row / Esc --> H[Cancel scoped recovery selection]
  C --> I[User clicks Load]
  I --> F
  F --> J[markStateSaved · green dot]
  G --> K[markStateAutosaved · blue dot]
```

Compared to the pre-#137 design (a YES/NO modal that interrupted the
flow), the final one-window design keeps both options visible in the
Load dialog but makes the commit action explicit in the footer. Row
clicks select; footer buttons load. This supersedes the intermediate
"click the named row again / click the scoped row" design because that
proved too easy to miss during review.

## Superseded notes from the mockup

- The old section-4 YES/NO modal remains useful as visual history, but
  it is no longer the target interaction.
- The intermediate scoped recovery row under the clicked preset is also
  superseded. Recovery now appears in the bottom Recovery auto-saves
  section.
- The section-5 four-state button cycle remains the pattern for true
  committing primary actions. `Save to…` and `Load from…` are utility
  navigation buttons that open the folder picker; they do not need the
  Save button's write-state cycle.

## Out of scope (deliberately, for this PR)

- Multi-library support
- Per-snapshot library binding
- Bulk migrate-presets-to-new-folder action
- Recent-libraries dropdown

The single-library design is documented intent ([curated-sidebar.md
§ library directory](../maintainers/curated-sidebar.md)). The redesign
only consolidates surface, it does not change the model.
