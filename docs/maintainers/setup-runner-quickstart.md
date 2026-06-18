# Setup Runner (Mockup Frontend) — Quick Reference

The "mockup frontend" for testing published setups is the **Setup Runner
Simulator**. It is served by ComfyUI itself — you do not launch it separately.

## Open it

Pattern — **always** the running Comfy address + a fixed path:

```
<your-comfy-address>/koolook/setup_runner_simulator.html
```

So whatever URL ComfyUI prints in its console on startup, e.g.

```
To see the GUI go to: http://127.0.0.1:8000
```

just tack on `/koolook/setup_runner_simulator.html`:

```
http://127.0.0.1:8000/koolook/setup_runner_simulator.html
```

Open it in a browser **on the same PC as ComfyUI**. Only the port changes
between machines — the `/koolook/...` path is constant. Bookmark it for next
time.

## Run a setup from the live registry (can actually Run)

1. Click **Load setups** — pulls every published setup from the running Comfy
   via the Koolook API.
2. Pick one in the **Published setup** dropdown.
3. Fill the app form, then click **Run setup**.

This path executes on the local Comfy nodes and polls to a terminal status
(`succeeded` / `failed` / `lost`).

## Load a published-setup `.json` file (inspect-only)

1. Use the **Setup file** picker and choose an exported published-setup JSON.
2. The form and contract render for review — but **Run is disabled** for
   file-loaded setups. Only the live registry can execute.
3. To actually run one, publish it into the running ComfyUI registry first,
   then use **Load setups** above.

## Notes

- The **ComfyUI API base** field can stay empty when opened from the koolook
  route (same origin). Only fill it if you opened the page from disk.
- `?demo=1` on the URL renders the UI with fake setups, no Comfy needed.
- A `failed` run with `video is not a valid path` (or similar) is a workflow /
  input-path problem, not a frontend problem — the frontend reaching that error
  means the round-trip works.

Source: [`../../web/setup_runner_simulator.html`](../../web/setup_runner_simulator.html),
route in [`../../koolook_routes.py`](../../koolook_routes.py).
