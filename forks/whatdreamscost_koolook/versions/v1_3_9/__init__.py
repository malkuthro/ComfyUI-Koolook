# SPDX-License-Identifier: GPL-3.0-or-later
"""
WhatDreamsCost Koolook v1.3.9 mappings.

The canonical ComfyUI node ID is stable (`LTXDirector__koolook`) so
future upstream-version upgrades can fit in place. The previous
`LTXDirector__koolook_v1_3_2` ID remains registered as a compatibility
alias for saved workflows, backed by this v1.3.9 implementation.
"""

from .ltx_director import LTXDirector, LTXDirectorLegacyV132


NODE_CLASS_MAPPINGS = {
    "LTXDirector__koolook": LTXDirector,
    "LTXDirector__koolook_v1_3_2": LTXDirectorLegacyV132,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXDirector__koolook": "LTX Director (Koolook)",
    "LTXDirector__koolook_v1_3_2": "LTX Director (Koolook)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
