// SPDX-FileCopyrightText: 2025-2026 Kforge Labs <https://github.com/malkuthro/ComfyUI-Koolook>
// SPDX-License-Identifier: GPL-3.0-only

// =============================================================================
// Right-click context menu on canvas nodes — adds to favorites.
//
// Patches `LGraphCanvas.prototype.getNodeMenuOptions` (wrap-not-replace) so the
// "Add to Kforge Labs" item appears on every node's right-click menu —
// including nodes from other packs. The patch is idempotent so a soft refresh
// doesn't stack duplicate entries.
// =============================================================================
import { toast } from "./constants.js";
import { addToMyPicks, notifyPicksChanged } from "./picks_store.js";
import { spotlightAddedPicks } from "./tree.js";

export function patchCanvasMenu() {
    const C = (typeof LGraphCanvas !== "undefined") ? LGraphCanvas : null;
    if (!C || !C.prototype || !C.prototype.getNodeMenuOptions) {
        console.warn("[Koolook] LGraphCanvas.getNodeMenuOptions not available; right-click menu skipped.");
        return;
    }
    // Idempotency: this is a global prototype mutation that affects every
    // pack's right-click menu. If the extension re-loads (HMR, soft refresh,
    // multiple registerExtension calls) we'd otherwise stack duplicate
    // "Add to Kforge Labs" entries on every node.
    if (C.prototype.__koolookCuratedPatched) return;
    C.prototype.__koolookCuratedPatched = true;
    const orig = C.prototype.getNodeMenuOptions;
    C.prototype.getNodeMenuOptions = function (node) {
        const options = orig.apply(this, arguments);
        options.push(null);
        options.push({
            content: "Add to Kforge Labs",
            callback: () => {
                if (!node || !node.type) return;
                const status = addToMyPicks(node.type);
                if (status === "added") {
                    toast(`Added "${node.title || node.type}" to favorites.`);
                    spotlightAddedPicks([node.type]);
                    notifyPicksChanged();
                } else if (status === "duplicate") {
                    toast("Already in favorites.");
                    // Spotlight on duplicate too — the user reached for this
                    // exact node; remind them where it lives.
                    spotlightAddedPicks([node.type]);
                    notifyPicksChanged();
                } else {
                    toast(`Failed to save "${node.title || node.type}". See console.`);
                }
            },
        });
        return options;
    };
}
