"""Radiance Koolook package entrypoint for versioned node sets.

Each `versions/vX_Y_Z/` subpackage owns a namespaced slice of nodes.
This entrypoint just merges all version mappings into a single dict
for the root loader.

Adding a new version: see `.claude/skills/add-external-fork/SKILL.md`.

History:
- v1_0_1 (24 wrapped Radiance v1.0 classes + 2 modified) was removed in
  v0.1.5 of ComfyUI-Koolook — the wrappers were vestigial (never used by
  the Koolook authors), and users who want Radiance functionality should
  install upstream Radiance directly. See CHANGELOG.md [0.1.5].
"""

from .versions.v2_3_3 import NODE_CLASS_MAPPINGS as V2_3_3_NODE_CLASS_MAPPINGS
from .versions.v2_3_3 import NODE_DISPLAY_NAME_MAPPINGS as V2_3_3_NODE_DISPLAY_NAME_MAPPINGS

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

NODE_CLASS_MAPPINGS.update(V2_3_3_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(V2_3_3_NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
