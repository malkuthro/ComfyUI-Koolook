"""Koolook Matte — mask-guided one-step video matting nodes.

Part of the Koolook pack. Registered via the parent ``__init__``'s guarded
``_merge_node_group("Koolook Matte", ".matte")`` so a missing ``diffusers``
disables only these nodes, not the rest of Koolook.

The nodes LOAD (never bundle) the VideoMaMa UNet + SVD VAE weights at runtime;
those weights carry their own non-commercial licenses — see ``matte/NOTICE``.
"""
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
