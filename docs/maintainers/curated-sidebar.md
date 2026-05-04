# Curated sidebar — distributed defaults workflow

The "Curated Nodes" sidebar tab in ComfyUI shows a per-user favorites tree
of nodes drawn from any installed pack. Implementation: pure JS in
[`web/koolook_sidebar.js`](../../web/koolook_sidebar.js).

This document covers the **maintainer** workflow: capturing the current
favorites list and shipping it as the package's default.

## Where the data lives

| Location | Scope | Purpose |
|---|---|---|
| `web/curated_defaults.json` | Repo-tracked, ships with the package | The default favorites list |
| `localStorage["koolook.curated.userPicks.v1"]` | Per-browser, per-user | The user's actual list (additions/removes on top of defaults) |
| `localStorage["koolook.curated.seeded.v1"]` | Per-browser | `"1"` once defaults have been applied (or skipped because user already had picks) |

## Data flow

```
[web/curated_defaults.json]              ← committed in repo, distributed via Comfy Registry
        │
        │  (on first ComfyUI load with empty localStorage)
        ▼
[browser localStorage]                   ← per-user, durable across ComfyUI restarts
        │
        ▼
[Curated Nodes sidebar tab]              ← rendered tree with + / × / right-click
        │
        │  (click ↓ Export button)
        ▼
[clipboard / downloaded JSON file]       ← maintainer pastes back into curated_defaults.json
```

## Updating the distributed defaults

1. **Customize in ComfyUI.** In the Curated Nodes tab:
   - Select a node on the canvas, click **+** in the toolbar, OR
   - Right-click a canvas node → **Add to Curated Sidebar**, OR
   - Hover any entry → click **×** to remove.

2. **Click the ↓ Export button** in the toolbar.
   Toast confirms `Copied N picks to clipboard. Paste into web/curated_defaults.json.`
   (If clipboard write fails, it falls back to downloading a `curated_defaults.json` file.)

3. **Paste into [`web/curated_defaults.json`](../../web/curated_defaults.json)**, replacing
   the entire file contents. The exported JSON is sorted alphabetically for stable diffs.

4. **Test the seeded experience** before committing:
   - In your browser DevTools console: `localStorage.clear()`
   - Refresh the ComfyUI page (or run `python scripts/sync_to_dev.py` if you also tweaked the JS)
   - Curated Nodes tab should show the new defaults under "Nodes (favorites)"

5. **Commit and push.**
   This change does **not** trigger the publish workflow — only `pyproject.toml`
   changes merged to `main` do. See [`releasing.md`](releasing.md) for the
   release procedure that distributes the updated defaults to Comfy Registry users.

## Effect on users when defaults change

| User state | What happens on next ComfyUI load |
|---|---|
| Fresh install (empty localStorage) | New defaults seeded into localStorage |
| Existing user who already has picks | Their picks unchanged. Defaults file ignored. `seeded` flag set so we don't re-trigger |
| Already-seeded user (returning) | localStorage is sole source of truth. No re-seed |

The seed runs **exactly once per browser**. After that, the defaults file is
never re-read for that browser. This means: shipping a new default in v0.3.0
won't appear for users who already had ComfyUI-Koolook installed at v0.2.x —
they keep their personalized list. Only fresh installs and users who clear
their browser data get the new default.

This is intentional: never overwrite a user's personalized favorites
silently. If you ever need to force-reseed (e.g. recovering from a bad
default), see "Reset workflow" below.

## Reset workflow (testing or recovering from a bad default)

In the browser DevTools console with ComfyUI open:

```js
localStorage.removeItem("koolook.curated.userPicks.v1");
localStorage.removeItem("koolook.curated.seeded.v1");
location.reload();
```

On the next page load, the seeder will read `curated_defaults.json` afresh
and populate localStorage. Use this to verify a new default before shipping,
or to recover after a manual edit gone wrong.

## Why localStorage (not a server-side userdata file)

- No Python HTTP route or filesystem write needed; pure frontend
- Survives ComfyUI server restarts and browser tab reloads
- Only lost via deliberate user action (clear data, switch browser)
- Same persistence model as Chrome's bookmark bar — fine for a "favorites" UI

If we later need cross-machine sync, the migration path is to swap the
storage layer for ComfyUI's [`/userdata` API](https://docs.comfy.org/development/comfyui-server/comms_overview)
without changing the seeding semantics.
