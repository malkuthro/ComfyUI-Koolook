# SPDX-License-Identifier: GPL-3.0-or-later
#
# ComfyUI-Koolook - publish contract nodes
# Copyright (C) 2026 ComfyUI-Koolook contributors (kforgelabs).
"""Controlled nodes for declaring externally published setup fields."""

from __future__ import annotations


INPUT_MODES: tuple[str, ...] = ("EXR", "QT", "Img", "Prompt")
INPUT_MODE_TO_INDEX = {name: index for index, name in enumerate(INPUT_MODES)}

# Output type is a strict subset of the input modes: ``Prompt`` is a source /
# no-op branch, never a real output format. ``Same as input`` is the default —
# it passes the wired input switch straight through so an unmodified setup
# writes the same type it reads; picking a concrete type overrides that.
SAME_AS_INPUT = "Same as input"
OUTPUT_MODES: tuple[str, ...] = (SAME_AS_INPUT, "EXR", "QT", "Img")
OUTPUT_MODE_TO_INDEX = {"EXR": 0, "QT": 1, "Img": 2}
_DEFAULT_SWITCH_INDEX = INPUT_MODE_TO_INDEX["Img"]  # 2


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


def _coerce_switch_index(value, default: int = _DEFAULT_SWITCH_INDEX) -> int:
    """Coerce a wired/typed switch value to an int branch index.

    ``bool`` is rejected first (``True``/``False`` are ints in Python but never
    a real branch index); everything else falls back to ``default`` (Img).
    """
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    return default


def _resolve_output_switch(output_mode, input_switch) -> int:
    """Resolve the writer-branch index emitted by ``Koolook_PublishOutput``.

    ``Same as input`` (the default) passes the wired ``input_switch`` through,
    so a setup writes the same type it reads unless the author/app overrides it.
    A concrete ``EXR``/``QT``/``Img`` — as the combo label OR the numeric index
    the external runner injects into ``output_mode`` — overrides it. ``Prompt``
    is not an output format, so an index of ``3`` is treated as "no override"
    and falls back to the input switch too.
    """
    if not isinstance(output_mode, bool):
        if isinstance(output_mode, int):
            if output_mode in OUTPUT_MODE_TO_INDEX.values():
                return output_mode
        else:
            text = str(output_mode).strip()
            if text in OUTPUT_MODE_TO_INDEX:
                return OUTPUT_MODE_TO_INDEX[text]
            if text.isdigit() and int(text) in OUTPUT_MODE_TO_INDEX.values():
                return int(text)
    # "Same as input" / Prompt / anything unrecognized -> follow the input switch.
    return _coerce_switch_index(input_switch)


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
                        "tooltip": "Source mode exposed to the external app. The switch output keeps this mode's stable numeric index.",
                    },
                ),
                "sequence_folder": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Directory path for sequence sources such as EXR or numbered image frames.",
                    },
                ),
                "qt_file": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Full path to one QuickTime/video file, including directory and filename.",
                    },
                ),
                "single_file": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Full path to one still image or single source file.",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Prompt text field for setups that route a text prompt through the published app surface.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("sequence_folder", "qt_file", "single_file", "prompt", "switch")
    CATEGORY = "Koolook/Publish"
    FUNCTION = "run"
    DESCRIPTION = "Declare source inputs for a workflow published to an external app."

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
                        "tooltip": "Destination folder exposed to the external app and wired into downstream writer/path nodes.",
                    },
                ),
                "name": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Base output name exposed to the external app.",
                    },
                ),
                "version": (
                    "STRING",
                    {
                        "default": "1",
                        "multiline": False,
                        "tooltip": "Output version token or number exposed to the external app.",
                    },
                ),
                "output_mode": (
                    list(OUTPUT_MODES),
                    {
                        "default": SAME_AS_INPUT,
                        "tooltip": "Writer branch to route the payload into. 'Same as input' passes the wired input switch through (output type follows input type); pick EXR/QT/Img to write a different type than the source. The 'switch' output wires into Koolook Publish Router's selector.",
                    },
                ),
            },
            "optional": {
                "input_switch": (
                    "INT",
                    {
                        "default": _DEFAULT_SWITCH_INDEX,
                        "min": 0,
                        "max": len(INPUT_MODES) - 1,
                        "tooltip": "Wire Koolook Publish Input's 'switch' output here. Used only when output_mode is 'Same as input' — it makes the output type follow the chosen input type. Ignored when a concrete output_mode is selected.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("folder", "name", "version", "switch")
    CATEGORY = "Koolook/Publish"
    FUNCTION = "run"
    DESCRIPTION = "Declare output folder, name, version, and output-type switch for a published setup."

    def run(
        self,
        folder: str,
        name: str,
        version: str,
        output_mode: str = SAME_AS_INPUT,
        input_switch: int = _DEFAULT_SWITCH_INDEX,
    ):
        return (folder, name, version, _resolve_output_switch(output_mode, input_switch))


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
                        "tooltip": "Resolved result path, folder, or status string to show after the external app run finishes.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    CATEGORY = "Koolook/Publish"
    FUNCTION = "run"
    OUTPUT_NODE = True
    DESCRIPTION = "Report a resolved setup result string back to the published setup runner."

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
                        "tooltip": "Switch index from Koolook Publish Input. Published setup runs keep only the matching writer branch.",
                    },
                ),
                "payload": (
                    "*",
                    {
                        "tooltip": "Main workflow payload to send to the switch-aligned writer branches.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("*", "*", "*", "*")
    RETURN_NAMES = INPUT_MODES
    CATEGORY = "Koolook/Publish"
    FUNCTION = "route"
    DESCRIPTION = "Route one payload into switch-aligned writer branches for published setup runs."

    def route(self, selector: int, payload):
        return (payload, payload, payload, payload)


NODE_CLASS_MAPPINGS = {
    "Koolook_PublishInput": Koolook_PublishInput,
    "Koolook_PublishOutput": Koolook_PublishOutput,
    "Koolook_PublishResult": Koolook_PublishResult,
    "Koolook_PublishRouter": Koolook_PublishRouter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Koolook_PublishInput": "Koolook Publish Input",
    "Koolook_PublishOutput": "Koolook Publish Output",
    "Koolook_PublishResult": "Koolook Publish Result",
    "Koolook_PublishRouter": "Koolook Publish Router",
}
