// SPDX-FileCopyrightText: 2025-2026 Kforge Labs <https://github.com/malkuthro/ComfyUI-Koolook>
// SPDX-License-Identifier: GPL-3.0-only

// =============================================================================
// Koolook curated-nodes sidebar — entry point.
//
// This file boots the sidebar; module bodies live in `web/sidebar/`. ComfyUI
// discovers the extension via `WEB_DIRECTORY = "./web"` (set in __init__.py)
// and calls `app.registerExtension(...).setup()` once during frontend startup.
// =============================================================================
import { app } from "../../../scripts/app.js";

import {
    TAB_ID,
    TAB_TITLE,
    TAB_TOOLTIP,
    TAB_ICON,
} from "./sidebar/constants.js";
import {
    seedStarterPresetIfNeeded,
    startPeriodicAutosave,
} from "./sidebar/snapshot.js";
import {
    loadWorkflowsStore,
    seedWorkflowDefaultsIfNeeded,
} from "./sidebar/workflows_store.js";
import { renderPanel } from "./sidebar/tree.js";
import { patchCanvasMenu } from "./sidebar/canvas_menu.js";

app.registerExtension({
    name: "koolook.curated_sidebar",
    async setup() {
        if (!app.extensionManager || !app.extensionManager.registerSidebarTab) {
            console.warn("[Koolook] extensionManager.registerSidebarTab not available; sidebar not registered.");
            return;
        }
        // Seeds the bundled starter preset into the user's snapshot library
        // on first run (replaces the legacy `curated_defaults.json` localStorage
        // pick-seed). Async-fires server I/O — kept off the critical path so
        // sidebar tab registration isn't blocked behind it on a slow server.
        seedStarterPresetIfNeeded();
        const loadResult = await loadWorkflowsStore();
        // Skip workflow seeding when /userdata file is corrupt — we don't want
        // to overwrite a recoverable-but-unparseable file with stock defaults.
        if (!loadResult.corrupt) {
            await seedWorkflowDefaultsIfNeeded();
        }
        app.extensionManager.registerSidebarTab({
            id: TAB_ID,
            title: TAB_TITLE,
            tooltip: TAB_TOOLTIP,
            icon: TAB_ICON,
            type: "custom",
            render: (el) => renderPanel(el),
        });
        patchCanvasMenu();
        // Periodic defensive auto-save — fires every 5 minutes if state has
        // changed since the last successful auto-save. Keeps last 5 in the
        // snapshot library as `_autosave_periodic_<iso>.json`. Started AFTER
        // sidebar registration so it doesn't compete with first-render work,
        // and uses an internal grace period before the first tick so the
        // load/seed flows above settle first.
        startPeriodicAutosave();
    },
});
