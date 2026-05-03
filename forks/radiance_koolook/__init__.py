"""Radiance Koolook package entrypoint for versioned node sets.

Each `versions/vX_Y_Z/` subpackage owns a namespaced slice of nodes
(suffix `__koolook_vX_Y_Z`). This entrypoint just merges all version
mappings into a single dict for the root loader.

Adding a new version: see `.claude/skills/add-external-fork/SKILL.md`.
"""

from .versions.v1_0_1 import NODE_CLASS_MAPPINGS as V1_0_1_NODE_CLASS_MAPPINGS
from .versions.v1_0_1 import NODE_DISPLAY_NAME_MAPPINGS as V1_0_1_NODE_DISPLAY_NAME_MAPPINGS

from .versions.v2_3_3 import NODE_CLASS_MAPPINGS as V2_3_3_NODE_CLASS_MAPPINGS
from .versions.v2_3_3 import NODE_DISPLAY_NAME_MAPPINGS as V2_3_3_NODE_DISPLAY_NAME_MAPPINGS

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

NODE_CLASS_MAPPINGS.update(V1_0_1_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(V1_0_1_NODE_DISPLAY_NAME_MAPPINGS)

NODE_CLASS_MAPPINGS.update(V2_3_3_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(V2_3_3_NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
