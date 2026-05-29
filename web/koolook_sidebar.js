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
    WORKFLOWS_FALLBACK_KEY,
    criticalToast,
    toast,
} from "./sidebar/constants.js";
import {
    seedStarterPresetIfNeeded,
    startPeriodicAutosave,
    markStateSaved,
    getCurrentPresetName,
    detectBootDrift,
    SNAPSHOT_KIND,
    SNAPSHOT_VERSION,
    writePreset,
} from "./sidebar/snapshot.js";
import {
    loadWorkflowsStore,
    seedWorkflowDefaultsIfNeeded,
    clearOfflineFallback,
} from "./sidebar/workflows_store.js";
import { loadUserPicks } from "./sidebar/picks_store.js";
import { showConfirmModal } from "./sidebar/modals.js";
import { renderSidebar } from "./sidebar/tree.js";
import { patchCanvasMenu } from "./sidebar/canvas_menu.js";
import { claimSidebarRegistration } from "./sidebar/extension_guard.js";

function wireOfflineFallbackRecovery(initialBlob) {
    criticalToast(
        "Offline workflow data found in browser localStorage from a " +
        "previous /userdata outage. If workflows you saved earlier are " +
        "missing now, click Restore to save them as a snapshot — or " +
        "Copy details for the raw JSON. Discard removes the offline " +
        "copy permanently.",
        {
            copyText: initialBlob,
            actions: [
                {
                    label: "Restore as snapshot",
                    primary: true,
                    busyLabel: "Restoring…",
                    onClick: async () => {
                        // Re-read at click time — the user may have already
                        // discarded the blob from another tab; `null` means
                        // there's nothing left to restore and we should
                        // just dismiss.
                        const blob = localStorage.getItem(WORKFLOWS_FALLBACK_KEY);
                        if (!blob) {
                            toast("Offline copy was already cleared.");
                            return { dismiss: true };
                        }
                        let fallbackStore;
                        try {
                            fallbackStore = JSON.parse(blob);
                        } catch (e) {
                            console.error("[Koolook] offline fallback parse failed:", e);
                            toast("Offline copy is unparseable — copy the raw JSON instead.");
                            return { dismiss: false };
                        }
                        const isoNow = new Date().toISOString();
                        const tsForFile = isoNow.replace(/[:.]/g, "-");
                        const fileName = `recovered-${tsForFile}`;
                        // Wrap as a full snapshot envelope so the recovered
                        // file is a first-class snapshot the user can
                        // inspect / Load via the regular Snapshot UI. Picks
                        // come from the live store (NOT the offline blob —
                        // the offline blob only holds workflows) so that
                        // loading the recovered snapshot later wouldn't
                        // wipe whatever picks the user has accumulated
                        // since the outage.
                        const envelope = {
                            kind: SNAPSHOT_KIND,
                            version: SNAPSHOT_VERSION,
                            name: `Offline recovery · ${isoNow}`,
                            exportedAt: isoNow,
                            picks: loadUserPicks(),
                            workflows: fallbackStore,
                        };
                        try {
                            await writePreset(fileName, envelope);
                        } catch (e) {
                            console.error("[Koolook] offline restore write failed:", e);
                            toast(`Could not save recovery snapshot: ${e.message}`);
                            return { dismiss: false };
                        }
                        clearOfflineFallback();
                        toast(
                            `Saved offline copy as snapshot "${fileName}". ` +
                            `Open Snapshot → Load to inspect or restore.`,
                            5500
                        );
                        return { dismiss: true };
                    },
                },
                {
                    label: "Discard offline copy",
                    busyLabel: "Discarding…",
                    onClick: () => new Promise((resolve) => {
                        showConfirmModal({
                            title: "Discard offline copy?",
                            message:
                                "This permanently removes the localStorage " +
                                "fallback. Make sure you've already recovered " +
                                "anything you needed — once cleared, this " +
                                "data cannot be retrieved.",
                            confirmLabel: "Discard",
                            danger: true,
                            onConfirm: () => {
                                clearOfflineFallback();
                                toast("Offline copy discarded.");
                                resolve({ dismiss: true });
                            },
                            // Cancel resolves with `{ dismiss: false }` so the
                            // toast button re-enables and the user can Dismiss
                            // or Restore from the still-open recovery banner
                            // instead of being stuck in a "Discarding…" state.
                            onCancel: () => resolve({ dismiss: false }),
                        });
                    }),
                },
            ],
        }
    );
}

// Client-side duplicate-install guard (#162). The backend `__init__.py`
// guard already prevents two Koolook installations from publishing
// competing `./web/` folders to ComfyUI (the non-winning install drops
// its `WEB_DIRECTORY`), so this path normally fires only once. But a
// downgraded sibling install on an older Koolook version without the
// backend guard, or a stale Manager-side cache that loads two
// `koolook_sidebar.js` files anyway, would land here twice and the
// second `app.registerExtension(...)` would silently shadow the first
// in some ComfyUI builds. The claim helper sets a window sentinel on
// first call so the second load skips registration entirely.
if (!claimSidebarRegistration(window)) {
    console.warn(
        "[Koolook] duplicate Kforge Labs install detected — this file was loaded twice. " +
        "Check ComfyUI's custom_nodes/ for both `koolook/` and `ComfyUI-Koolook/` folders " +
        "(or any other Koolook copy) and remove one. The first registration stays live."
    );
    // Defer the user-facing toast until the next microtask — `criticalToast`
    // appends to `document.body`, which exists at script-eval time in ComfyUI
    // but the surrounding style sheet (`ensureStyle()`) is installed inside
    // `renderPanel`. Queuing the toast lets the user see it as soon as the
    // sidebar tab opens, even on a duplicate install where the second
    // `setup()` never runs.
    queueMicrotask(() => {
        try {
            criticalToast(
                "Kforge Labs is installed twice in ComfyUI's custom_nodes/. " +
                "Two copies (typically `koolook/` from the ComfyUI Manager / Registry " +
                "and `ComfyUI-Koolook/` from a git clone) both register the same " +
                "sidebar tab and write to the same /userdata file — running both " +
                "silently corrupts your workflow store. Remove one of the folders " +
                "and restart ComfyUI. See the server log for the resolved paths."
            );
        } catch (e) {
            // No usable DOM yet — the console.warn above already carries
            // the essential signal. Don't escalate.
        }
    });
} else {
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
        // Offline-fallback recovery toast — fires when /userdata loaded
        // successfully but a stale localStorage blob from a past outage
        // still exists. Two recovery actions:
        //   • Restore as snapshot — wraps the offline workflows into a
        //     full snapshot envelope (with current picks so a later Load
        //     wouldn't wipe them) and writes it as `recovered-<iso>.json`
        //     into the snapshot library. Clears the fallback key on
        //     success so the banner self-extinguishes.
        //   • Discard offline copy — confirm-then-clear. Same end state
        //     for the banner, but the user explicitly accepts data loss.
        // Wired here (not in `loadWorkflowsStore`) because the action
        // handlers need `writePreset`, `loadUserPicks`, and
        // `showConfirmModal` — pulling those into workflows_store would
        // create a circular import via snapshot.js / modals.js.
        if (loadResult.fallbackBlob) {
            wireOfflineFallbackRecovery(loadResult.fallbackBlob);
        }
        // Baseline the saved-state fingerprint at session start IFF the
        // tracker says a preset is currently loaded AND we don't already
        // have a baseline persisted from a previous session. The "saved
        // fingerprint persists across reloads" path is more accurate, but
        // first-ever-session needs this seeding to avoid showing "unsaved"
        // immediately for users who closed the tab on a clean state.
        // Without this, the indicator would flicker "unsaved → saved" the
        // first time the user clicks Save (since markStateSaved baselines
        // there). With it, "saved" is shown from the start. Trade-off: if
        // the user closed mid-edit (state on /userdata diverges from the
        // tracked preset's content), we'll show "saved" briefly until they
        // mutate again — acceptable for an indicator.
        if (getCurrentPresetName() && !localStorage.getItem("koolook.snapshot.savedFingerprint.v1")) {
            markStateSaved();
        }
        // Boot-time tracked-snapshot drift check (#161). When a preset is
        // tracked, compare the named snapshot file on disk against the
        // live /userdata + picks state. On mismatch, flag drift — the
        // status pill flips to "drifted (reload?)" and periodic auto-
        // saves redirect to `_unsaved_autosave/` so the named preset's
        // recovery folder is never overwritten with the (potentially
        // corrupt) live state.
        //
        // Runs AFTER the localStorage fingerprint baseline above so the
        // drift flag wins precedence over the fingerprint match — see
        // `getSnapshotStatus()` for the precedence rationale.
        //
        // Failure here is silent (logged, not toasted): a missing file or
        // unreachable server are distinct failure modes the user already
        // has surface for elsewhere. Drift specifically means "both
        // present, both differ".
        const trackedNameForDrift = getCurrentPresetName();
        if (trackedNameForDrift) {
            try {
                await detectBootDrift(trackedNameForDrift);
            } catch (e) {
                console.warn("[Koolook] boot drift check threw:", e);
            }
        }
        app.extensionManager.registerSidebarTab({
            id: TAB_ID,
            title: TAB_TITLE,
            tooltip: TAB_TOOLTIP,
            icon: TAB_ICON,
            type: "custom",
            render: (el) => renderSidebar(el),
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
}
