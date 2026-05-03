# SPDX-License-Identifier: GPL-3.0-or-later
"""
Radiance Koolook v2.3.3 mappings.

Loads the slim, video-friendly RadianceVAEEncode / RadianceVAEDecode
re-implementation and namespaces the IDs with the suffix
`__koolook_v2_3_3` to avoid collisions with installed copies of
upstream Radiance v2.3.3 (which ships its own `RadianceVAEEncode`
alias for `RadianceVAE4KEncode`).

See `forks/THIRD_PARTY.md` and `UPSTREAM_PIN.yaml` for upstream
attribution and the full change log.
"""

from .nodes_vae import (
    NODE_CLASS_MAPPINGS as VAE_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as VAE_NODE_DISPLAY_NAME_MAPPINGS,
)


NAMESPACE_SUFFIX = "__koolook_v2_3_3"
DISPLAY_SUFFIX = " (Koolook v2.3.3)"

# IDs in this set are exposed verbatim (no `__koolook_v2_3_3` suffix on the
# ID, no ` (Koolook v2.3.3)` suffix on the display name). Use this for
# Koolook-original node names that don't collide with anything upstream —
# the version suffix is redundant noise in the ComfyUI search UI for those.
# The version is still recorded structurally (the file lives under
# `versions/v2_3_3/`) and textually (file header + UPSTREAM_PIN.yaml +
# forks/THIRD_PARTY.md).
SKIP_VERSION_SUFFIX = {
    "Easy_hdr_VAE_encode",
    "Easy_hdr_VAE_decode",
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
    VAE_NODE_CLASS_MAPPINGS,
    VAE_NODE_DISPLAY_NAME_MAPPINGS,
)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
