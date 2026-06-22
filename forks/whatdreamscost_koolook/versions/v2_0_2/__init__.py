# SPDX-License-Identifier: GPL-3.0-or-later
"""
WhatDreamsCost Koolook v2.0.2 mappings.

Carries the canonical stable node ID ``LTXDirector__koolook`` forward to the
upstream 2.0.2 implementation (the package entrypoint loads this version last,
so it wins the shared IDs over v1_3_9). The legacy ``LTXDirector__koolook_v1_3_2``
ID stays registered as a compatibility alias for saved workflows, now backed by
this v2.0.2 implementation.

Koolook deltas carried forward from v1_3_9 (see UPSTREAM_PIN.yaml / THIRD_PARTY.md):
  - Prompt Relay ``relay_overrides`` per-stream overrides
  - ``audio_transcript_json`` timed-transcript expansion
  - per-segment sigma in ``prompt_relay.build_segments``
New in v2.0.2:
  - latent-bucket keyframe grid-snap (``snap_keyframes_to_grid``, issue #258)
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
