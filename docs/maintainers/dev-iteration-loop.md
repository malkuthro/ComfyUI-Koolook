# Iterative dev loop — sidebar work

How the Curated Nodes / Workflows sidebar was built, in case you come back
to it after a break. Optimized for fast iteration without the publish
round-trip.

## The loop

1. Edit `web/koolook_sidebar.js` or any module under `web/sidebar/` (or related JSON) in the worktree.
2. `python scripts/sync_to_dev.py` — copies runtime files into your live
   ComfyUI `custom_nodes/<koolook>/`. Reads `KOLOOK_COMFYUI_DEV_PATH` from `.env`.
3. Restart ComfyUI; verify visually.
4. Iterate.

End-to-end: ~10–30 seconds. No tag/publish round-trip.

## Trigger phrases when working with the agent

| You say | Agent does |
|---|---|
| `dev-sync` (or "sync to dev", "copy those files") | runs `scripts/sync_to_dev.py` |
| `go` | implements the most-recently proposed plan as code |
| paste exported JSON from the ↓ button | replaces the matching `web/<feature>_defaults.json` |

## What survives across re-syncs and ComfyUI restarts

| Data | Lives in | Persistence |
|---|---|---|
| Node favorites (per user) | `localStorage["koolook.curated.userPicks.v1"]` | survives JS updates, ComfyUI restarts, browser refreshes |
| Saved workflows incl. archive (per install) | ComfyUI `/userdata/koolook_workflows.json` | survives JS updates and ComfyUI restarts |
| Distributed defaults — nodes | `web/curated_defaults.json` (in repo) | seeded once into user `localStorage` on first ComfyUI load |
| Distributed defaults — workflows | `web/workflow_defaults.json` (in repo) | seeded once into `/userdata` on first ComfyUI load |

dev-sync + restart never wipes user state. Only `localStorage.clear()`,
deleting `/userdata/koolook_workflows.json`, or wiping browser data does.

## Commit / push / publish gates

- Commit after a stable batch of working changes — not after every edit.
- Pushing a feature/PR branch **never** triggers a Comfy Registry publish.
- Publish fires only when `pyproject.toml` changes and the push lands on
  `main`. Full procedure: [`releasing.md`](releasing.md).

## Tips that paid off in this build

- **Annotated screenshots** (a red arrow on the spot you're talking about)
  beat text descriptions for UI feedback every time.
- **Verify uncertain frontend APIs against `Comfy-Org/ComfyUI_frontend`
  source before guessing.** E.g. the `loadGraphData(graph, true, true, wfName, {})`
  4th-arg trick that names the workflow tab was pulled from upstream
  `src/scripts/app.ts`, not guessed.
- **Defensive fallbacks** when calling a newer ComfyUI API surface —
  multiple property paths inside try/catch so older frontends still work.
- **Persist folder expansion state across re-renders** so saving doesn't
  collapse the user's view (the `pathStates` Map in `web/sidebar/tree.js`).

## Files this loop touches most

- [`web/koolook_sidebar.js`](../../web/koolook_sidebar.js) — sidebar entry (boots the extension; module bodies live in [`web/sidebar/`](../../web/sidebar/))
- [`web/curated_defaults.json`](../../web/curated_defaults.json) — node defaults
- [`web/workflow_defaults.json`](../../web/workflow_defaults.json) — workflow defaults
- [`scripts/sync_to_dev.py`](../../scripts/sync_to_dev.py) — dev sync helper
- [`docs/maintainers/curated-sidebar.md`](curated-sidebar.md) — node-defaults round-trip detail
