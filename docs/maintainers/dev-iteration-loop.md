# Iterative dev loop — sidebar work

How the Curated Nodes / Workflows sidebar was built, in case you come back
to it after a break. Optimized for fast iteration without the publish
round-trip.

## First-time setup on a new machine

`KOLOOK_COMFYUI_DEV_PATH` should point at the eventual Koolook subdirectory
inside your live ComfyUI `custom_nodes/` folder — **not** the `custom_nodes/`
parent itself.

Target the **`koolook/`** leaf, not `ComfyUI-Koolook/`. The ComfyUI Manager
and the Comfy Registry both install at `custom_nodes/koolook/` (derived from
`[project].name = "koolook"` in our [`pyproject.toml`](../../pyproject.toml)).
Pointing dev-sync at the same path overwrites the Manager install in place,
which is what you want. Targeting `custom_nodes/ComfyUI-Koolook/` instead
creates a parallel install — both get loaded on every ComfyUI boot, both
register the same routes / sidebar tab, both write to the same `/userdata`
file, and the workflow store silently corrupts on every restart. The
`__init__.py` duplicate-install guard catches this and prints a critical
log naming both paths, but the cleanest path is to never create the dual
install. See issue [#162](https://github.com/malkuthro/ComfyUI-Koolook/issues/162)
for the full failure mode.

| Platform | Typical path |
|---|---|
| macOS | `/Volumes/Data/ComfyUI/custom_nodes/koolook` |
| Linux | `/home/<user>/ComfyUI/custom_nodes/koolook` |
| Windows | `C:/ComfyUI_portable/ComfyUI/custom_nodes/koolook` (forward slashes work fine) |

```bash
cp .env.example .env                            # if you don't have one yet
$EDITOR .env                                    # set KOLOOK_COMFYUI_DEV_PATH
python scripts/sync_to_dev.py --init            # creates the leaf folder + first sync
# restart ComfyUI to pick up the new install
```

After that first run, drop the `--init` — plain `python scripts/sync_to_dev.py`
is enough for every subsequent edit. The `--init` flag is a one-shot guard
against typos: it refuses to create a Koolook folder unless the parent looks
like a ComfyUI `custom_nodes/` directory (named `custom_nodes` or sitting
inside a `ComfyUI` parent).

## The loop

1. Edit `web/koolook_sidebar.js` or any module under `web/sidebar/` (or related JSON) in the worktree.
2. `python scripts/sync_to_dev.py` — copies runtime files into your live
   ComfyUI `custom_nodes/<koolook>/`. Reads `KOLOOK_COMFYUI_DEV_PATH` from `.env`.
3. Restart ComfyUI; verify visually.
4. **If a design mockup exists** for what you just implemented (look
   under [`docs/designs/`](../designs/)), screenshot the running result
   and diff against the mockup. Don't request review until visual parity
   is reached or each deviation is documented. See the [project
   CLAUDE.md](../../CLAUDE.md) *"Visual verification for design-driven
   implementation"* section for the full rule.
5. Iterate.

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
| Distributed defaults — starter preset | `web/starter_preset.json` (in repo) | copied once into the user's snapshot library as `starter.json` on first ComfyUI load (replaces the legacy `curated_defaults.json` localStorage seed) |
| Distributed defaults — workflows | `web/workflow_defaults.json` (in repo) | seeded once into `/userdata` on first ComfyUI load |

dev-sync and manual ComfyUI restarts never wipe user state. Only
`localStorage.clear()`, deleting `/userdata/koolook_workflows.json`, or
wiping browser data does.

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
- **For frontend preview buttons, inspect the virtual-node implementation
  before guessing.** ComfyUI custom nodes such as EasyUse GET/SET can be
  frontend-only tunnels: render-time Python receives the resolved value, but
  a JS preview button may only see the visible GET key unless it follows the
  tunnel itself. Read the installed custom node JS, then simulate the relevant
  LiteGraph shape locally before asking the maintainer to test.

## Debugging frontend preview buttons

Frontend preview buttons, such as Easy AI Pipeline's `Get output file path`,
are not the same execution path as the node's Python outputs. The button code
runs in browser JS and often peeks at connected widgets; the real render path
runs through ComfyUI execution and may resolve virtual or dynamic nodes that
the preview cannot see by default.

When a preview disagrees with render-time output:

1. Confirm whether the fix is Python, JS, or both. Python changes need a
   ComfyUI restart; JS changes need the browser tab to reload the extension
   module. Restarting ComfyUI alone does not guarantee an already-open tab has
   re-imported `web/*.js`.
2. Verify the served extension file directly, e.g.
   `http://127.0.0.1:8188/extensions/koolook/ai_pipeline.js`, and add a
   short temporary marker when needed so the loaded frontend version is
   provable.
3. Inspect the connected custom node's frontend source in the live
   `custom_nodes/` install. EasyUse GET/SET nodes live in
   `comfyui-easy-use/web_version/v1/js/getset.js`.
4. If the connected node is a virtual tunnel, follow its graph semantics
   manually. EasyUse `easy getNode` stores the visible key in its `Constant`
   widget; the actual value lives upstream of the matching `easy setNode`
   input link.
5. Build a small local JS simulation of the LiteGraph shape
   (`GET -> SET -> source`) before dev-syncing another guess.

Do not assume the in-browser automation can always read `app.graph` from the
modern ComfyUI frontend. If the graph object is not exposed, use source
inspection, served-file checks, and small simulations instead.

## Files this loop touches most

- [`web/koolook_sidebar.js`](../../web/koolook_sidebar.js) — sidebar entry (boots the extension; module bodies live in [`web/sidebar/`](../../web/sidebar/))
- [`web/starter_preset.json`](../../web/starter_preset.json) — bundled starter preset (replaces the legacy `curated_defaults.json` pick-only seed)
- [`web/workflow_defaults.json`](../../web/workflow_defaults.json) — workflow defaults
- [`scripts/sync_to_dev.py`](../../scripts/sync_to_dev.py) — dev sync helper
- [`docs/maintainers/curated-sidebar.md`](curated-sidebar.md) — starter-preset round-trip detail
