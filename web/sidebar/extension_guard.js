// SPDX-FileCopyrightText: 2025-2026 Kforge Labs <https://github.com/malkuthro/ComfyUI-Koolook>
// SPDX-License-Identifier: GPL-3.0-only

// =============================================================================
// Client-side duplicate-install guard (#162). Pulled out of
// `koolook_sidebar.js` so the sentinel + decision can be unit-tested under
// Node — the sidebar entry imports `../../../scripts/app.js` (ComfyUI's
// frontend app), which doesn't exist in a headless context.
//
// The backend `__init__.py` guard normally prevents a duplicate's `./web/`
// folder from being published in the first place. This module is the
// fallback: if a stale build / older Koolook copy on a sibling install
// ships a `koolook_sidebar.js` to the browser anyway, the sentinel set
// here makes sure only the first-loaded copy calls `app.registerExtension`.
//
// `globalScope` defaults to `window` for production but is parameterised
// so tests can pass a synthetic object and verify the contract without
// touching real browser globals.
// =============================================================================

export const SIDEBAR_SENTINEL_KEY = "__koolookCuratedSidebarRegistered";

// Returns `true` if no prior Koolook sidebar has registered on this
// scope (and sets the sentinel as a side-effect), `false` otherwise.
//
// The set-on-success contract means the helper is single-shot per
// scope: subsequent calls return `false` even from the same caller.
// That's the duplicate-detected signal — `koolook_sidebar.js` reads
// the return value as "should I register?".
export function claimSidebarRegistration(globalScope) {
    const scope = globalScope || (typeof window !== "undefined" ? window : globalThis);
    if (scope[SIDEBAR_SENTINEL_KEY]) return false;
    scope[SIDEBAR_SENTINEL_KEY] = true;
    return true;
}

// Reset the sentinel on a scope. Production code never calls this; it
// exists so tests can exercise the claim flow more than once inside a
// single Node process without leaking state across cases.
export function resetSidebarRegistrationForTesting(globalScope) {
    const scope = globalScope || (typeof window !== "undefined" ? window : globalThis);
    delete scope[SIDEBAR_SENTINEL_KEY];
}
