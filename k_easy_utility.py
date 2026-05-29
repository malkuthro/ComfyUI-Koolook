# SPDX-License-Identifier: GPL-3.0-or-later
#
# ComfyUI-Koolook — Easy Utility node.
# Copyright (C) 2026 ComfyUI-Koolook contributors (kforgelabs).
"""Easy Utility (Koolook) — small dispatchable converter.

One node, multiple conversion modes selected by a ``mode`` widget. Start
small (``int_to_padded_string`` for driving the global version token into
``EasyAIPipeline`` / ``Easy_VideoCombine``) and add further modes by
extending ``_MODES`` and the dispatch in :meth:`Easy_Utility.run`. The
``mode`` widget shipped from day one so adding modes later does not
rewire saved workflows.

For the version-token use case, ``prefix`` defaults to empty: send the
bare padded digits (``001``) and the downstream consumer adds the ``v``.
``koolook_versioning.resolve_version_token`` converts a bare digit string
to ``v###`` automatically (see that module's docstring for the rule).
"""
from __future__ import annotations

# Order here drives the dropdown order in the node picker. New modes
# append to the end so the existing default stays index 0.
_MODES: tuple[str, ...] = ("int_to_padded_string",)


class Easy_Utility:
    """Multi-mode converter. v1 mode: INT -> zero-padded STRING (e.g. ``001``)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (list(_MODES), {"default": _MODES[0]}),
                "int_value": ("INT", {"default": 1, "min": 0, "max": 99999, "step": 1}),
                "prefix": ("STRING", {"default": "", "multiline": False}),
                "pad_width": ("INT", {"default": 3, "min": 0, "max": 10, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    CATEGORY = "Koolook/Utility"
    FUNCTION = "run"

    def run(self, mode: str, int_value: int, prefix: str, pad_width: int):
        if mode == "int_to_padded_string":
            n = int(int_value)
            w = max(0, int(pad_width))
            return (f"{prefix}{n:0{w}d}",)
        # Unknown mode -> stringify the int so the workflow doesn't crash.
        return (str(int(int_value)),)


NODE_CLASS_MAPPINGS = {"Easy_Utility": Easy_Utility}
NODE_DISPLAY_NAME_MAPPINGS = {"Easy_Utility": "Easy Utility (Koolook)"}
