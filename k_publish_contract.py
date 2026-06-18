# SPDX-License-Identifier: GPL-3.0-or-later
#
# ComfyUI-Koolook - publish contract nodes
# Copyright (C) 2026 ComfyUI-Koolook contributors (kforgelabs).
"""Controlled nodes for declaring externally published setup fields."""

from __future__ import annotations

try:
    from .koolook_versioning import next_version_token
except ImportError:  # pragma: no cover - direct import in tests
    from koolook_versioning import next_version_token


INPUT_MODES: tuple[str, ...] = ("EXR", "QT", "Img", "Prompt")
INPUT_MODE_TO_INDEX = {name: index for index, name in enumerate(INPUT_MODES)}


def _resolve_mode_index(mode) -> int:
    if isinstance(mode, bool):
        return INPUT_MODE_TO_INDEX["Img"]
    if isinstance(mode, int) and 0 <= mode < len(INPUT_MODES):
        return mode
    text = str(mode).strip()
    if text.isdigit():
        index = int(text)
        if 0 <= index < len(INPUT_MODES):
            return index
    return INPUT_MODE_TO_INDEX.get(text, INPUT_MODE_TO_INDEX["Img"])


class Koolook_PublishInput:
    """App-facing setup inputs with stable output names for maintainers."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (
                    list(INPUT_MODES),
                    {
                        "default": "Img",
                        "tooltip": "External app source mode. The node outputs the stable numeric switch index.",
                    },
                ),
                "sequence_folder": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Folder path for EXR/image sequence sources.",
                    },
                ),
                "qt_file": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Full path to a QuickTime/video file.",
                    },
                ),
                "single_file": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Full path to a single source image/file.",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Internal prompt placeholder for setups that need the shared input structure.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("sequence_folder", "qt_file", "single_file", "prompt", "switch")
    CATEGORY = "Koolook/Publish"
    FUNCTION = "run"

    def run(
        self,
        mode: str,
        sequence_folder: str,
        qt_file: str,
        single_file: str,
        prompt: str,
    ):
        return (
            sequence_folder,
            qt_file,
            single_file,
            prompt,
            _resolve_mode_index(mode),
        )


class Koolook_PublishOutput:
    """App-facing setup output controls."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Folder where the setup should write outputs.",
                    },
                ),
                "name": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Base output name.",
                    },
                ),
                "version": (
                    "STRING",
                    {
                        "default": "1",
                        "multiline": False,
                        "tooltip": "Output version token or number.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("folder", "name", "version")
    CATEGORY = "Koolook/Publish"
    FUNCTION = "run"

    def run(self, folder: str, name: str, version: str):
        return (folder, name, version)


class Koolook_PublishResult:
    """Resolved setup result surfaced back to the external app."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "result": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Final resolved output path, folder, or status from the workflow.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    CATEGORY = "Koolook/Publish"
    FUNCTION = "run"
    OUTPUT_NODE = True

    def run(self, result: str):
        return {"ui": {"text": [result]}, "result": (result,)}


class Koolook_PublishRouter:
    """Switch-aligned payload router for externally published setup outputs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "selector": (
                    "INT",
                    {
                        "default": INPUT_MODE_TO_INDEX["Img"],
                        "min": 0,
                        "max": len(INPUT_MODES) - 1,
                        "tooltip": "Switch index from Koolook Publish Input. External setup runs keep the matching output branch.",
                    },
                ),
                "payload": (
                    "*",
                    {
                        "tooltip": "Payload to route to the switch-aligned writer branches.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("*", "*", "*", "*")
    RETURN_NAMES = INPUT_MODES
    CATEGORY = "Koolook/Publish"
    FUNCTION = "route"

    def route(self, selector: int, payload):
        return (payload, payload, payload, payload)


class Koolook_NextVersion:
    """Auto-detect the next free output version token for an existing path.

    Scans ``folder`` for existing ``<name>_vNNN`` files/sequence-folders and
    outputs the next token (e.g. ``v003``). Wire ``version`` into the writers'
    version inputs so a re-run lands on a fresh version instead of failing on an
    existing file. One source feeds every branch (EXR + video) so they stay in
    sync. ``folder``/``name`` typically come from ``Koolook Publish Output``.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Output directory to scan for existing <name>_vNNN entries.",
                    },
                ),
                "name": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Base output name (the part before _vNNN).",
                    },
                ),
            },
            "optional": {
                "version_prefix": (
                    "STRING",
                    {"default": "v", "tooltip": "Version prefix; 'v' -> v001."},
                ),
                "padding": (
                    "INT",
                    {"default": 3, "min": 1, "max": 8, "tooltip": "Zero-pad width, e.g. 3 -> v001."},
                ),
                "start": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 99999,
                        "tooltip": "Token to use when no existing version is found.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("version",)
    CATEGORY = "Koolook/Publish"
    FUNCTION = "run"

    def run(self, folder, name, version_prefix="v", padding=3, start=1):
        return (next_version_token(folder, name, version_prefix, padding, start),)

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        # The next version depends on current directory contents, so re-scan on
        # every run rather than caching a stale token.
        return float("nan")


NODE_CLASS_MAPPINGS = {
    "Koolook_PublishInput": Koolook_PublishInput,
    "Koolook_PublishOutput": Koolook_PublishOutput,
    "Koolook_PublishResult": Koolook_PublishResult,
    "Koolook_PublishRouter": Koolook_PublishRouter,
    "Koolook_NextVersion": Koolook_NextVersion,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Koolook_PublishInput": "Koolook Publish Input",
    "Koolook_PublishOutput": "Koolook Publish Output",
    "Koolook_PublishResult": "Koolook Publish Result",
    "Koolook_PublishRouter": "Koolook Publish Router",
    "Koolook_NextVersion": "Koolook Next Version",
}
