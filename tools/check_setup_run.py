#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
#
# ComfyUI-Koolook -- live run/poll smoke check for the published-setup API.
"""Exercise the published-setup run -> poll flow against a live ComfyUI.

This is a *live* smoke check (not a unit test): it queues a real render through
the same routes the runner simulator uses, then polls the run endpoint exactly
like the simulator's poller, printing the status timeline. Use it to self-verify
end-to-end behaviour (e.g. "does a long render reach succeeded?") without driving
the browser UI.

    python tools/check_setup_run.py                 # first valid setup, default inputs
    python tools/check_setup_run.py --match upscale # first setup whose id contains "upscale"
    python tools/check_setup_run.py --id pub_ltx-2.3_upscale_v09 --inputs '{"prompt":"a bear"}'
    python tools/check_setup_run.py --list          # just list setups and exit

Exits 0 on a succeeded run, 1 otherwise. Default base is http://127.0.0.1:8000.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

TERMINAL = {"succeeded", "failed", "lost"}
ACTIVE = {"queued", "running"}
ALLOWED_URL_SCHEMES = {"http", "https"}


def _checked_url(url: str) -> str:
    scheme = urllib.parse.urlsplit(url).scheme
    if scheme not in ALLOWED_URL_SCHEMES:
        raise ValueError(f"unsupported URL scheme: {scheme or '<empty>'}")
    return url


def _get(url: str, timeout: float = 30.0):
    with urllib.request.urlopen(_checked_url(url), timeout=timeout) as r:  # nosec B310
        return json.load(r)


def _post(url: str, payload: dict, timeout: float = 60.0):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        _checked_url(url),
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:  # nosec B310
        return json.load(r)


def _progress_signature(run: dict) -> str:
    outputs = []
    for output in run.get("outputs") or []:
        if not isinstance(output, dict):
            continue
        items = output.get("items") if isinstance(output.get("items"), list) else []
        outputs.append(
            {
                "key": output.get("key"),
                "itemCount": len(items),
                "values": [item.get("value", "") for item in items if isinstance(item, dict) and item.get("value")],
            }
        )
    return json.dumps(
        {
            "status": str(run.get("status", "")).lower(),
            "promptId": run.get("promptId", ""),
            "queuePosition": run.get("queuePosition", run.get("position")),
            "updatedAt": run.get("updatedAt") or run.get("timestamp") or run.get("lastUpdate") or "",
            "outputs": outputs,
        },
        sort_keys=True,
    )


def list_setups(base: str) -> list[dict]:
    body = _get(f"{base}/koolook/api/setups")
    rows = body.get("setups") if isinstance(body, dict) else body
    return rows if isinstance(rows, list) else []


def pick_setup(rows: list[dict], setup_id: str | None, match: str | None) -> dict | None:
    valid = [r for r in rows if (r.get("validation") or {}).get("status") == "valid"]
    if setup_id:
        return next((r for r in rows if r.get("id") == setup_id), None)
    if match:
        return next((r for r in valid if match.lower() in str(r.get("id", "")).lower()), None)
    return valid[0] if valid else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--id", default=None, help="exact setup id to run")
    ap.add_argument("--match", default=None, help="substring to pick a setup id")
    ap.add_argument("--inputs", default="{}", help="JSON object of run inputs")
    ap.add_argument("--list", action="store_true", help="list setups and exit")
    ap.add_argument("--interval", type=float, default=3.0, help="poll interval seconds")
    ap.add_argument("--stall", type=float, default=180.0,
                    help="give up after this many seconds with no active/terminal status")
    args = ap.parse_args()
    base = args.base.rstrip("/")

    try:
        rows = list_setups(base)
    except urllib.error.URLError as exc:
        print(f"FAIL: cannot reach ComfyUI at {base}: {exc}")
        return 1

    print(f"{len(rows)} published setup(s):")
    for r in rows:
        print(f"  {r.get('id'):<34} status={(r.get('validation') or {}).get('status')}")
    if args.list:
        return 0

    setup = pick_setup(rows, args.id, args.match)
    if setup is None:
        print("FAIL: no matching/valid setup to run.")
        return 1
    setup_id = setup["id"]

    try:
        inputs = json.loads(args.inputs)
        assert isinstance(inputs, dict)
    except Exception as exc:
        print(f"FAIL: --inputs must be a JSON object: {exc}")
        return 1

    print(f"\nRunning '{setup_id}' with inputs={inputs} ...")
    started = time.monotonic()
    try:
        queued = _post(f"{base}/koolook/api/setups/{setup_id}/run", {"inputs": inputs})
    except urllib.error.HTTPError as exc:
        print(f"FAIL: run request HTTP {exc.code}: {exc.read().decode()[:300]}")
        return 1
    run = queued.get("run") or {}
    run_id = run.get("runId")
    print(f"  queued runId={run_id} promptId={run.get('promptId')} status={run.get('status')}")
    if not run_id:
        print("FAIL: no runId returned.")
        return 1

    # Poll exactly like the simulator: treat --stall as a no-progress window.
    # The deadline resets only when the run payload changes in a meaningful way;
    # repeated queued/running responses with no changing marker are a stall.
    deadline = time.monotonic() + args.stall
    last_status = None
    last_progress = _progress_signature(run)
    while time.monotonic() <= deadline:
        time.sleep(args.interval)
        try:
            body = _get(f"{base}/koolook/api/runs/{run_id}")
        except urllib.error.HTTPError as exc:
            print(f"  poll HTTP {exc.code} -- retrying")
            continue
        run = body.get("run") or {}
        status = str(run.get("status", "")).lower()
        elapsed = int(time.monotonic() - started)
        if status != last_status:
            print(f"  [t+{elapsed}s] status={status}")
            last_status = status
        if status in TERMINAL:
            ok = status == "succeeded"
            print(f"\n{'PASS' if ok else 'FAIL'}: run {status} after {elapsed}s")
            for out in run.get("outputs") or []:
                for item in out.get("items") or []:
                    if item.get("value"):
                        print(f"  {out.get('label') or out.get('key')}: {item['value']}")
            return 0 if ok else 1
        if status in ACTIVE:
            progress = _progress_signature(run)
            if progress != last_progress:
                last_progress = progress
                deadline = time.monotonic() + args.stall

    print(f"\nFAIL: no terminal status within the {args.stall}s no-progress window.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
