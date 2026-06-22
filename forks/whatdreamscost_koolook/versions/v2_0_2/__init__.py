# SPDX-License-Identifier: GPL-3.0-or-later
"""
WhatDreamsCost Koolook v2.0.2 mappings.

A faithful replica of upstream WhatDreamsCost-ComfyUI 2.0.2 `LTXDirector`,
namespaced as `LTXDirector__koolook`, with exactly one additive change: the
`snap_keyframes_to_grid` latent-bucket keyframe snap (issue #258). No other
Koolook customizations — Prompt Relay, audio, and every 2.0.2 feature are
upstream-exact.

The package entrypoint loads this version last, so the canonical
`LTXDirector__koolook` resolves here. The legacy `LTXDirector__koolook_v1_3_2`
ID stays registered by the v1_3_9 package for workflows saved against it.
"""

from .ltx_director import LTXDirector


NODE_CLASS_MAPPINGS = {
    "LTXDirector__koolook": LTXDirector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXDirector__koolook": "LTX Director (Koolook)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
