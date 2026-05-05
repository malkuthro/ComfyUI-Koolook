# Kforge Labs sidebar starter preset — quick reference

The "Kforge Labs" sidebar tab (id `koolook.curatedNodes`); implementation in
[`web/koolook_sidebar.js`](../../web/koolook_sidebar.js).

## What ships to fresh users

A single bundled snapshot file at [`web/starter_preset.json`](../../web/starter_preset.json) — same format the user-facing **Snapshot** library uses (picks + workflows + tags + archive). On first launch the seeder copies it into the user's snapshot library directory as `starter.json`. The user opens **Snapshot → Load**, sees one entry called *starter*, clicks → state populated.

Nothing seeds into `localStorage` directly anymore — replaced the legacy `curated_defaults.json` pick-only seed in 0.3.0.

## Update the shipped starter preset

1. In ComfyUI's **Kforge Labs** tab, customize the state you want fresh users to see — picks via `+` / `×` / right-click; workflows via the Workflows section; tags via right-click "Tags…".
2. In the **Tools** row (between Snapshot and the search field), click the **↓** button → full snapshot JSON copied to clipboard.
3. Open [`web/starter_preset.json`](../../web/starter_preset.json) in your editor, paste over the contents, save.
4. `git commit ... && git push`. Does **not** trigger publish — only `pyproject.toml` changes merged to `main` do.

## Files & storage keys

| Where | What |
|---|---|
| `web/starter_preset.json` (repo) | Bundled starter preset, snapshot format |
| `web/koolook_sidebar.js` (repo) | Sidebar entry point |
| `web/sidebar/snapshot.js` (repo) | `seedStarterPresetIfNeeded` + `exportStarterPreset` |
| `<library-dir>/starter.json` (per-install) | The seeded preset on the user's disk |
| `localStorage["koolook.starter.seeded.v1"]` | `"1"` once the seed has been attempted on this browser |
| `localStorage["koolook.curated.userPicks.v1"]` | User's actual picks list (per-browser) |
| `localStorage["koolook.snapshot.currentPresetName.v1"]` | Last preset the user loaded/saved |

The library directory is configured by (in priority order):
1. The `libraryPath` field in `<comfyui-userdata>/koolook-settings.json` (set via Snapshot → Settings dialog),
2. The `KFORGELABS_PRESETS` env var (deployment / facility config),
3. Default fallback `<comfyui-userdata>/koolook-presets/`.

See [`koolook_routes.py`](../../koolook_routes.py) for the resolution chain.

## Reset (force re-seed)

In the browser DevTools console with ComfyUI open:

```js
localStorage.removeItem("koolook.starter.seeded.v1");
localStorage.removeItem("koolook.curated.userPicks.v1");
// And delete <library-dir>/starter.json on disk (or via Snapshot → Load → × button)
location.reload();
```

The seeder bails when picks are non-empty (treats it as "existing user, don't re-seed"), so the picks deletion is necessary for the seed to fire.

## Things that surprised us — keep in mind

- **The seeder needs the snapshot library reachable.** It calls the same `/koolook/presets/file` endpoint the Save dialog uses. If the library path is read-only, points at a non-existent directory, or the routes module isn't loaded, the seed silently retries on the next page load — no error toast for fresh users. See the console for `[Koolook] starter seed: …` messages.
- **Paste is manual.** Browsers can't write to your filesystem. The ↓ button only copies to clipboard / falls back to a download. Nothing auto-updates `starter_preset.json`.
- **Other users can't reach the repo.** Their ↓ click only touches their own clipboard / library. Only you, with push access, can change the distributed starter.
- **Seeding runs exactly once per browser.** Existing users with picks already in localStorage are explicitly skipped (the `loadUserPicks().length > 0` early-return) so a new starter doesn't disturb their state. To preview the fresh-install UX yourself, use the reset block above.
- **The seed populates the library, not the canvas.** After seeding, the user still has empty picks until they explicitly Load the starter preset. This was the design choice for 0.3.0 — discoverable, opt-in, respects user agency. Pattern 1 (auto-apply on first run, identical to the old curated_defaults UX) was rejected because the source-of-state becomes invisible to the user.
- **No publish on this kind of push.** Edits under `web/` never trigger the registry publish workflow; only a `pyproject.toml` change merged to `main` does. See [`releasing.md`](releasing.md).
