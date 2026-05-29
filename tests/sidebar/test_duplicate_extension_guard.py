"""Tests for the client-side duplicate-install guard (#162) —
``web/sidebar/extension_guard.js``.

The full ``web/koolook_sidebar.js`` entry imports ComfyUI's frontend
``app.js``, which doesn't exist in a headless Node context, so the
guard logic was pulled into ``extension_guard.js`` and the sidebar
entry now calls ``claimSidebarRegistration`` to make its decision.
Testing the helper here covers the contract; the sidebar entry's
wiring (the ``console.warn`` + ``criticalToast`` branch on a `false`
return) is exercised end-to-end via the visual harness / dev-sync.
"""
from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_node(source: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["node", "--input-type=module"],
        input=source,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_first_claim_succeeds_and_sets_sentinel() -> None:
    """First call into a fresh scope wins the registration slot and
    flips the sentinel field to ``true``. The sidebar entry calls
    ``app.registerExtension`` only when this returns ``true``."""
    script = textwrap.dedent(
        """
        import assert from "node:assert/strict";
        import {
            claimSidebarRegistration,
            SIDEBAR_SENTINEL_KEY,
        } from "./web/sidebar/extension_guard.js";

        const scope = {};
        const won = claimSidebarRegistration(scope);
        assert.equal(won, true, "first claim should win");
        assert.equal(scope[SIDEBAR_SENTINEL_KEY], true, "sentinel should be set");
        """
    )
    result = _run_node(script)
    assert result.returncode == 0, result.stderr


def test_second_claim_returns_false() -> None:
    """Second call into the same scope returns ``false`` — the sidebar
    entry interprets this as "another Koolook install already booted on
    this window; skip registerExtension and surface the duplicate
    warning to the user instead.\""""
    script = textwrap.dedent(
        """
        import assert from "node:assert/strict";
        import { claimSidebarRegistration } from "./web/sidebar/extension_guard.js";

        const scope = {};
        const first = claimSidebarRegistration(scope);
        const second = claimSidebarRegistration(scope);
        assert.equal(first, true);
        assert.equal(second, false, "second claim should report duplicate");
        """
    )
    result = _run_node(script)
    assert result.returncode == 0, result.stderr


def test_claim_isolates_across_scopes() -> None:
    """Distinct scopes don't interfere — claiming on scope A doesn't
    set the sentinel on scope B. Production only ever has one
    ``window``; this is a safety check for the parameterisation
    contract (so reset-then-test patterns in other suites can reuse
    distinct scope objects)."""
    script = textwrap.dedent(
        """
        import assert from "node:assert/strict";
        import { claimSidebarRegistration } from "./web/sidebar/extension_guard.js";

        const scopeA = {};
        const scopeB = {};
        assert.equal(claimSidebarRegistration(scopeA), true);
        assert.equal(claimSidebarRegistration(scopeB), true);
        assert.equal(claimSidebarRegistration(scopeA), false);
        assert.equal(claimSidebarRegistration(scopeB), false);
        """
    )
    result = _run_node(script)
    assert result.returncode == 0, result.stderr


def test_reset_helper_clears_sentinel() -> None:
    """Production never calls this, but tests do — verify the reset
    works so a future test suite can safely call ``claim`` more than
    once on the same scope."""
    script = textwrap.dedent(
        """
        import assert from "node:assert/strict";
        import {
            claimSidebarRegistration,
            resetSidebarRegistrationForTesting,
            SIDEBAR_SENTINEL_KEY,
        } from "./web/sidebar/extension_guard.js";

        const scope = {};
        claimSidebarRegistration(scope);
        assert.equal(scope[SIDEBAR_SENTINEL_KEY], true);
        resetSidebarRegistrationForTesting(scope);
        assert.equal(scope[SIDEBAR_SENTINEL_KEY], undefined);
        assert.equal(claimSidebarRegistration(scope), true,
            "after reset, claim should win again");
        """
    )
    result = _run_node(script)
    assert result.returncode == 0, result.stderr
