"""Run the loop demo against a local ComfyUI server.

This is a repo-local validation harness for
`docs/automations/loop-demo/LOOP_demo_pipeline.json`. It submits an API-format version
of the queue-controller demo graph to a running ComfyUI server and verifies
that four EXR files are written across four prompt executions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request


def _validate_http_url(url: str) -> None:
    parsed = urllib.parse.urlsplit(str(url or ""))
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError(f"Only http(s) ComfyUI server URLs are allowed: {url!r}")


def _post_json(url: str, payload: dict) -> dict:
    _validate_http_url(url)
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:  # nosec B310
        return json.load(response)


def _get_json(url: str, timeout: float = 15) -> dict:
    _validate_http_url(url)
    with urllib.request.urlopen(url, timeout=timeout) as response:  # nosec B310
        return json.load(response)


def _assert_inside(parent: Path, child: Path) -> None:
    parent = parent.resolve()
    child = child.resolve()
    if parent != child and parent not in child.parents:
        raise RuntimeError(f"Refusing to touch path outside repo: {child}")


def _dump_server_debug(server: str) -> None:
    for endpoint in ("queue", "history?max_items=20"):
        try:
            payload = _get_json(f"{server}/{endpoint}", timeout=10)
        except Exception as exc:
            print(f"Could not read /{endpoint}: {exc}")
            continue
        print(f"/{endpoint}:")
        print(json.dumps(payload, indent=2)[:8000])


def build_prompt(output_base: Path) -> dict:
    """Build the API prompt equivalent of the current loop demo.

    The canvas workflow contains a ComfyUI subgraph node. The API prompt below
    flattens that demo subgraph to its current single inner node,
    `EasyResize_Koolook`, so the prompt can be submitted directly to `/prompt`.
    """
    return {
        "12": {"class_type": "easy int", "inputs": {"value": 4}},
        "10": {
            "class_type": "Text Multiline",
            "inputs": {"text": str(output_base) + "\\"},
        },
        "9": {
            "class_type": "Easy_Pattern",
            "inputs": {
                "batch_size": ["12", 0],
                "width": 512,
                "height": 512,
                "bg_color_mode": "Custom",
                "show_text": True,
                "text_color_mode": "White",
                "start_from": 1,
                "step": 1,
                "font_size": 256,
                "position": "center",
                "zero_pad": 0,
                "bg_color": "#C71585",
                "text_color": "#FFFFFF",
                "prefix": "",
                "suffix": "",
            },
        },
        "8": {
            "class_type": "EasyAIPipeline",
            "inputs": {
                "shot_duration": 81,
                "seed_value": 453453453,
                "instruction": "Place your base folder path in the FIELD below",
                "base_directory_path": ["10", 0],
                "extension": ".%04d.exr",
                "shot_name": "faceLoop",
                "ai_method": "",
                "version": "1",
                "disable_versioning": False,
                "enable_overwrite": False,
                "no_subfolders": True,
            },
        },
        "22": {"class_type": "easy int", "inputs": {"value": 0}},
        "3": {
            "class_type": "ImageFromBatch",
            "inputs": {"image": ["9", 0], "batch_index": ["22", 0], "length": 1},
        },
        "19": {
            "class_type": "EasyResize_Koolook",
            "inputs": {
                "image": ["3", 0],
                "base_on": "Width",
                "base_size": 512,
                "aspect_ratio": "16:9",
                "divisible_by": 32,
                "upscale_method": "nearest-exact",
                "keep_proportion": "stretch",
                "crop_position": "center",
                "pad_color_mode": "Black",
                "panel_color_mode": "Black",
                "device": "cpu",
                "pad_color": "0, 0, 0",
                "panel_color": "0, 0, 0",
                "invert_composed_MASK": False,
            },
        },
        "21": {
            "class_type": "Koolook_LoopStatus",
            "inputs": {
                "value": ["19", 0],
                "index": ["22", 0],
                "total": ["12", 0],
                "filepath": ["8", 0],
                "label": "EXR_SAFE",
                "auto_queue_next": True,
                "index_node_id": "22",
                "server_url": "http://127.0.0.1:8188",
                "max_auto_queue_depth": 100,
                "remaining_auto_queue_depth": -1,
            },
        },
        "4": {
            "class_type": "SaveEXRFrames",
            "inputs": {
                "images": ["21", 0],
                "filepath": ["8", 0],
                "tonemap": "linear",
                "start_frame": ["22", 0],
                "overwrite": True,
                "save_workflow": "none",
            },
        },
    }


def run(server: str, output_base: Path, repo_root: Path) -> int:
    _assert_inside(repo_root, output_base)
    run_output = output_base / f"run-{int(time.time() * 1000)}"
    _assert_inside(repo_root, run_output)
    run_output.mkdir(parents=True, exist_ok=True)

    system = _get_json(f"{server}/system_stats")
    print(f"ComfyUI {system['system']['comfyui_version']} at {server}")

    payload = {
        "prompt": build_prompt(run_output.resolve()),
        "client_id": "koolook-loop-test",
    }
    try:
        queued = _post_json(f"{server}/prompt", payload)
    except urllib.error.HTTPError as exc:
        print(exc.read().decode("utf-8", errors="replace"))
        return 1

    prompt_id = queued["prompt_id"]
    print(f"queued {prompt_id}")
    for _ in range(240):
        time.sleep(0.5)
        history = _get_json(f"{server}/history/{prompt_id}")
        if prompt_id in history:
            status = history[prompt_id].get("status", {})
            print(json.dumps(status, indent=2))
            if status.get("status_str") != "success":
                return 1
            break
    else:
        print("Timed out waiting for ComfyUI history.")
        return 1

    files = []
    for _ in range(240):
        files = sorted(run_output.rglob("*.exr"))
        if len(files) >= 4:
            break
        time.sleep(0.5)
    for file in files:
        print(file)
    if len(files) != 4:
        print(f"Expected 4 EXRs, found {len(files)}.")
        _dump_server_debug(server)
        return 1
    print("OK: wrote 4 EXR frames.")
    return 0


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument(
        "--output-base",
        type=Path,
        default=repo_root / ".tmp" / "comfy-loop-test-output",
    )
    args = parser.parse_args()
    return run(args.server.rstrip("/"), args.output_base, repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
