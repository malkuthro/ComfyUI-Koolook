"""WhatDreamsCost Koolook package entrypoint for versioned node sets.

Each `versions/vX_Y_Z/` subpackage owns a namespaced slice of nodes from a
pinned upstream commit. This entrypoint merges all version mappings into a
single dict for the root loader.

Adding a new version: see `.claude/skills/add-external-fork/SKILL.md`.
"""

from .versions.v1_3_2 import NODE_CLASS_MAPPINGS as V1_3_2_NODE_CLASS_MAPPINGS
from .versions.v1_3_2 import NODE_DISPLAY_NAME_MAPPINGS as V1_3_2_NODE_DISPLAY_NAME_MAPPINGS

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

NODE_CLASS_MAPPINGS.update(V1_3_2_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(V1_3_2_NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
