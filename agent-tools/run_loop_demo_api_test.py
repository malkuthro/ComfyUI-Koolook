"""Run the loop demo against a local ComfyUI server.

This is a repo-local validation harness for
`agent-tools/LOOP_demo_pipeline_Codex.json`. It submits an API-format version
of the demo graph to a running ComfyUI server and verifies that four EXR files
are written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import urllib.error
import urllib.request


def _post_json(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=15) as response:
        return json.load(response)


def _assert_inside(parent: Path, child: Path) -> None:
    parent = parent.resolve()
    child = child.resolve()
    if parent != child and parent not in child.parents:
        raise RuntimeError(f"Refusing to touch path outside repo: {child}")


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
        "2": {"class_type": "easy forLoopStart", "inputs": {"total": ["12", 0]}},
        "3": {
            "class_type": "ImageFromBatch",
            "inputs": {"image": ["9", 0], "batch_index": ["2", 1], "length": 1},
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
                "index": ["2", 1],
                "total": ["12", 0],
                "filepath": ["8", 0],
                "label": "EXR_SAFE",
            },
        },
        "5": {
            "class_type": "easy forLoopEnd",
            "inputs": {"flow": ["2", 0], "initial_value1": ["21", 0]},
        },
        "20": {
            "class_type": "easy showAnything",
            "inputs": {"anything": ["5", 0]},
        },
        "4": {
            "class_type": "SaveEXRFrames",
            "inputs": {
                "images": ["21", 0],
                "filepath": ["8", 0],
                "tonemap": "linear",
                "start_frame": ["2", 1],
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
        "client_id": "codex-loop-test",
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

    files = sorted(run_output.rglob("*.exr"))
    for file in files:
        print(file)
    if len(files) != 4:
        print(f"Expected 4 EXRs, found {len(files)}.")
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
        default=repo_root / "agent-tools" / "comfy-loop-test-output",
    )
    args = parser.parse_args()
    return run(args.server.rstrip("/"), args.output_base, repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
