# Curated Nodes sidebar — quick reference

The "Curated Nodes" sidebar tab in ComfyUI; implementation in
[`web/koolook_sidebar.js`](../../web/koolook_sidebar.js).

## Update the distributed defaults

1. In ComfyUI's **Curated Nodes** tab, customize picks via `+` / `×` / right-click.
2. Click the **↓** button → JSON copied to clipboard.
3. Open [`web/curated_defaults.json`](../../web/curated_defaults.json) in your editor, paste over the contents, save.
4. `git commit ... && git push`. Does **not** trigger publish — only `pyproject.toml` changes merged to `main` do.

## Files & storage keys

| Where | What |
|---|---|
| `web/curated_defaults.json` (repo) | Distributed default favorites |
| `web/koolook_sidebar.js` (repo) | Sidebar implementation |
| `localStorage["koolook.curated.userPicks.v1"]` | User's actual list (per-browser) |
| `localStorage["koolook.curated.seeded.v1"]` | `"1"` once defaults seeded for this browser |

## Reset (force re-seed from defaults)

In the browser DevTools console with ComfyUI open:

```js
localStorage.removeItem("koolook.curated.userPicks.v1");
localStorage.removeItem("koolook.curated.seeded.v1");
location.reload();
```

Use this to verify a new default before shipping, or to recover after a manual edit gone wrong.

## Things that surprised us — keep in mind

- **Paste is manual.** Browsers can't write to your filesystem. The ↓ button only copies to clipboard / falls back to a download. Nothing auto-updates `curated_defaults.json`.
- **Other users can't reach the repo.** Their ↓ click only touches their own clipboard. localStorage is browser-local; nothing uploads anywhere. Only you, with push access, can change the distributed default.
- **Seeding runs exactly once per browser.** Existing users keep their picks when you ship a new default — only fresh installs and users who clear browser data get the new list. To verify the seeded UX yourself, use the reset block above.
- **No publish on this kind of push.** Edits under `web/` never trigger the registry publish workflow; only a `pyproject.toml` change merged to `main` does. See [`releasing.md`](releasing.md).
