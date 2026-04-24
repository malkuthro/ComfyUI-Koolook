"""
Radiance Koolook v1.0.1 mappings.

Loads locally tracked modified nodes and namespaces mapping IDs.
"""

from .nodes_hdr import (
    NODE_CLASS_MAPPINGS as HDR_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as HDR_NODE_DISPLAY_NAME_MAPPINGS,
)
from .nodes_color_management import (
    NODE_CLASS_MAPPINGS as COLOR_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as COLOR_NODE_DISPLAY_NAME_MAPPINGS,
)


NAMESPACE_SUFFIX = "__koolook_v1_0_1"
DISPLAY_SUFFIX = " (Koolook v1.0.1)"

# Optional compact IDs for frequently used nodes.
# If a node is listed here, this ID is used instead of the default
# "<original_id>__koolook_v1_0_1" format.
SHORT_ID_OVERRIDES = {
    "RadianceOCIOColorTransformV2": "k_easy_OCIO_v101",
}


def _namespace_mappings(class_mappings, display_mappings):
    namespaced_classes = {}
    namespaced_display = {}

    for node_id, node_cls in class_mappings.items():
        namespaced_id = SHORT_ID_OVERRIDES.get(node_id, f"{node_id}{NAMESPACE_SUFFIX}")
        namespaced_classes[namespaced_id] = node_cls
        base_display = display_mappings.get(node_id, node_id)
        namespaced_display[namespaced_id] = f"{base_display}{DISPLAY_SUFFIX}"

    return namespaced_classes, namespaced_display


_ALL_CLASSES = {}
_ALL_DISPLAY = {}
_ALL_CLASSES.update(HDR_NODE_CLASS_MAPPINGS)
_ALL_CLASSES.update(COLOR_NODE_CLASS_MAPPINGS)
_ALL_DISPLAY.update(HDR_NODE_DISPLAY_NAME_MAPPINGS)
_ALL_DISPLAY.update(COLOR_NODE_DISPLAY_NAME_MAPPINGS)

NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = _namespace_mappings(
    _ALL_CLASSES, _ALL_DISPLAY
)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
