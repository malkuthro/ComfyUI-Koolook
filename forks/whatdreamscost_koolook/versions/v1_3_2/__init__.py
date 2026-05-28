# SPDX-License-Identifier: GPL-3.0-or-later
"""
WhatDreamsCost Koolook v1.3.2 mappings.

Loads the modified `LTXDirector` node (with the added `relay_overrides`
multiline input + per-segment-sigma Prompt-Relay implementation) and
namespaces its ID with the suffix `__koolook_v1_3_2` to avoid collisions
with an installed copy of upstream WhatDreamsCost-ComfyUI v1.3.2.

The companion `LTXDirectorGuide` node is intentionally NOT included here:
it is upstream code we did not modify, and graphs are expected to wire
`LTXDirector__koolook_v1_3_2.guide_data` into the user's upstream
`LTXDirectorGuide` (or our future Koolook variant).

See `forks/THIRD_PARTY.md` and `UPSTREAM_PIN.yaml` for upstream
attribution and the full change log.
"""

from .ltx_director import LTXDirector


NAMESPACE_SUFFIX = "__koolook_v1_3_2"
DISPLAY_SUFFIX = " (Koolook v1.3.2)"

# IDs in this set are exposed verbatim (no suffix). Empty for this version —
# the only registered node is LTXDirector, whose plain ID collides with the
# upstream class of the same name. Future Koolook-original additions here
# (with no upstream counterpart) can opt out via this set, same as
# `forks/radiance_koolook/versions/v2_3_3/__init__.py`.
SKIP_VERSION_SUFFIX: set[str] = set()


_BASE_CLASS_MAPPINGS = {
    "LTXDirector": LTXDirector,
}

_BASE_DISPLAY_MAPPINGS = {
    "LTXDirector": "LTX Director",
}


def _namespace_mappings(class_mappings, display_mappings):
    namespaced_classes = {}
    namespaced_display = {}

    for node_id, node_cls in class_mappings.items():
        if node_id in SKIP_VERSION_SUFFIX:
            namespaced_id = node_id
            namespaced_display[namespaced_id] = display_mappings.get(node_id, node_id)
        else:
            namespaced_id = f"{node_id}{NAMESPACE_SUFFIX}"
            base_display = display_mappings.get(node_id, node_id)
            namespaced_display[namespaced_id] = f"{base_display}{DISPLAY_SUFFIX}"
        namespaced_classes[namespaced_id] = node_cls

    return namespaced_classes, namespaced_display


NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = _namespace_mappings(
    _BASE_CLASS_MAPPINGS,
    _BASE_DISPLAY_MAPPINGS,
)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
