// SPDX-FileCopyrightText: 2025-2026 Kforge Labs <https://github.com/malkuthro/ComfyUI-Koolook>
// SPDX-License-Identifier: GPL-3.0-only

export function resolveFolderExpanded({
    forceExpanded = false,
    forceExpandedWhenFiltered = true,
    iconKind = null,
    isPinned = false,
    hasStoredState = false,
    storedState = false,
    startExpanded = true,
} = {}) {
    if (forceExpanded && forceExpandedWhenFiltered && iconKind !== "archive") return true;
    if (isPinned) return true;
    if (hasStoredState) return storedState === true;
    return startExpanded === true;
}
