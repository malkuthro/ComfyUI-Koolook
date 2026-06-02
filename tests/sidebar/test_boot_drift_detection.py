"""Tests for the boot-time tracked-snapshot drift guard (#161).

Covers the four observable contracts the rest of the codebase depends on:
  1. ``detectBootDrift()`` flags drift on file-vs-live mismatch.
  2. ``detectBootDrift()`` does NOT flag drift when the named file and
     live state agree.
  3. ``getSnapshotStatus()`` returns ``state: "drifted"`` while the
     flag is set — with precedence over the localStorage-baseline
     ``"saved"`` state.
  4. ``markStateSaved()`` clears the drift flag — Save and Load both
     route through it, so this is the single point that retires the
     warning.

The pre-load autosave routing (``_autosaveSubdir`` returning
``_unsaved_autosave/`` while drifted) is covered as a 5th case via
``writePreLoadAutosave`` — calling that exercises the same private
helper the periodic timer uses.

All tests stub ``fetch`` so the snapshot module operates against an
in-memory "library" without touching the real network.
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


# Common browser-globals + fetch-stub preamble used by every test. Kept
# as a python f-string so each test can append its own scenario body.
_PREAMBLE = """
import assert from "node:assert/strict";
import {
    detectBootDrift,
    getSnapshotStatus,
    markStateSaved,
    clearBootDrift,
    isBootDrifted,
    setCurrentPresetName,
    writePreLoadAutosave,
} from "./web/sidebar/snapshot.js";
import {
    loadWorkflowsStore,
    replaceAllWorkflows,
    getAllWorkflowsForExport,
} from "./web/sidebar/workflows_store.js";
import { setAllPicks } from "./web/sidebar/picks_store.js";

setupBrowserStubs();

// In-memory "server" the fetch stub reads from / writes to. Tests
// arrange their scenario by mutating these before calling the SUT.
const SERVER = {
    workflowsStore: { directories: {} },
    presets: new Map(),  // fileName (with .json) -> snapshot object
    writes: [],          // every POST captured for assertion
};

globalThis.fetch = async (url, init = {}) => {
    const u = new URL(String(url), "http://localhost");
    const pathOnly = u.pathname;
    if (pathOnly.startsWith("/userdata/koolook_workflows.json")) {
        if (init.method === "POST") {
            SERVER.workflowsStore = JSON.parse(init.body);
            return okJson("");
        }
        return okJson(JSON.stringify(SERVER.workflowsStore));
    }
    if (pathOnly === "/koolook/presets/file") {
        const name = u.searchParams.get("name");
        const dir = u.searchParams.get("dir") || "";
        const key = (dir ? dir + "/" : "") + name;
        if (init.method === "POST") {
            SERVER.presets.set(key, JSON.parse(init.body));
            SERVER.writes.push({ name, dir, body: JSON.parse(init.body) });
            return okJson("");
        }
        if (init.method === "HEAD") {
            return SERVER.presets.has(key)
                ? { ok: true, status: 200 }
                : { ok: false, status: 404 };
        }
        const stored = SERVER.presets.get(key);
        if (!stored) return { ok: false, status: 404, text: async () => "not found" };
        return okJson(JSON.stringify(stored));
    }
    if (pathOnly === "/koolook/presets/info") {
        return okJson(JSON.stringify({ path: "/tmp/koolook-presets", isDefault: true }));
    }
    return { ok: false, status: 404, text: async () => "", json: async () => ({}) };
};

function okJson(body) {
    return {
        ok: true,
        status: 200,
        text: async () => body,
        json: async () => JSON.parse(body || "{}"),
    };
}

function setupBrowserStubs() {
    const storage = new Map();
    globalThis.localStorage = {
        getItem: (key) => storage.has(key) ? storage.get(key) : null,
        setItem: (key, value) => storage.set(key, String(value)),
        removeItem: (key) => storage.delete(key),
    };
    globalThis.CustomEvent = class CustomEvent { constructor(type) { this.type = type; } };
    globalThis.window = { dispatchEvent() {}, addEventListener() {} };
    Object.defineProperty(globalThis, "navigator", {
        value: { clipboard: { writeText: async () => {} } },
        configurable: true,
    });
    const makeEl = () => ({
        className: "", textContent: "", style: {},
        classList: { add() {}, remove() {} },
        appendChild() {}, remove() {}, addEventListener() {},
        setAttribute() {},
    });
    globalThis.document = {
        createElement: makeEl,
        body: { appendChild() {} },
        head: { appendChild() {} },
        querySelectorAll: () => [],
        getElementById: () => null,
        hidden: false,
    };
}
"""


def _run(scenario_body: str) -> subprocess.CompletedProcess[str]:
    return _run_node(_PREAMBLE + textwrap.dedent(scenario_body))


# =============================================================================
# 1. Drift detection on file-vs-live mismatch
# =============================================================================


def test_detect_drift_flags_mismatch() -> None:
    """The named snapshot file holds 2 workflows; the live store holds
    a different 1-workflow shape. ``detectBootDrift()`` must flag this
    as drift and console.warn with the diagnostics."""
    result = _run(
        """
        // Arrange: named file with workflows A + B.
        SERVER.presets.set("Foo.json", {
            kind: "koolook-snapshot",
            version: 1,
            name: "Foo",
            exportedAt: "2026-05-29T10:00:00.000Z",
            picks: ["EasyResize_Koolook"],
            workflows: {
                directories: {
                    "Renders": {
                        workflows: {
                            "A": { graph: { nodes: [{ id: 1 }] }, archived: false, module: false, tags: [] },
                            "B": { graph: { nodes: [{ id: 2 }] }, archived: false, module: false, tags: [] },
                        },
                        directories: {},
                    },
                },
            },
        });
        // Arrange: live store carries a single C workflow — clearly different.
        SERVER.workflowsStore = {
            directories: {
                "Renders": {
                    workflows: {
                        "C": { graph: { nodes: [{ id: 3 }] }, archived: false, module: false, tags: [] },
                    },
                    directories: {},
                },
            },
        };
        await loadWorkflowsStore();
        setAllPicks([]);
        setCurrentPresetName("Foo");

        // Act
        const outcome = await detectBootDrift("Foo");

        // Assert
        assert.equal(outcome.drifted, true, "outcome should report drift");
        assert.equal(outcome.trackedName, "Foo");
        assert.equal(isBootDrifted(), true, "module flag should be set");
        assert.ok(outcome.diagnostics.fileFingerprintBytes > 0);
        assert.ok(outcome.diagnostics.liveFingerprintBytes > 0);
        """
    )
    assert result.returncode == 0, result.stderr


def test_detect_drift_skips_when_states_match() -> None:
    """Named file and live state are byte-equal in payload (modulo the
    snapshot envelope) — no drift flagged."""
    result = _run(
        """
        const sharedStore = {
            directories: {
                "Renders": {
                    workflows: {
                        "A": { graph: { nodes: [{ id: 1 }] }, archived: false, module: false, tags: [] },
                    },
                    directories: {},
                },
            },
        };
        SERVER.workflowsStore = sharedStore;
        SERVER.presets.set("Bar.json", {
            kind: "koolook-snapshot",
            version: 1,
            name: "Bar",
            exportedAt: "2026-05-29T11:00:00.000Z",
            picks: [],
            workflows: sharedStore,
        });
        await loadWorkflowsStore();
        setAllPicks([]);
        setCurrentPresetName("Bar");

        const outcome = await detectBootDrift("Bar");

        assert.equal(outcome.drifted, false);
        assert.equal(isBootDrifted(), false);
        """
    )
    assert result.returncode == 0, result.stderr


def test_detect_drift_skips_when_named_file_missing() -> None:
    """Tracked preset name but no file on disk — that's a distinct
    failure mode the regular Load dialog already surfaces. Don't flag
    drift; let the localStorage fingerprint baseline drive status."""
    result = _run(
        """
        SERVER.workflowsStore = { directories: {} };
        await loadWorkflowsStore();
        setAllPicks([]);
        setCurrentPresetName("Missing");

        const outcome = await detectBootDrift("Missing");

        // outcome is null on the "file not readable" path
        assert.equal(outcome, null);
        assert.equal(isBootDrifted(), false);
        """
    )
    assert result.returncode == 0, result.stderr


# =============================================================================
# 2. Status precedence — "drifted" wins over "saved"
# =============================================================================


def test_status_drifted_wins_over_saved_baseline() -> None:
    """Even when the persisted localStorage fingerprint matches the
    current live state (which would otherwise report "saved"), the
    drift flag promotes the status to "drifted". This is the core
    precedence rule that protects against a stale-but-matching
    fingerprint baseline."""
    result = _run(
        """
        SERVER.workflowsStore = { directories: {} };
        await loadWorkflowsStore();
        setAllPicks([]);
        setCurrentPresetName("Baz");

        // Baseline: live state matches itself, so markStateSaved persists
        // a fingerprint that satisfies the "saved" check.
        markStateSaved();
        let status = getSnapshotStatus();
        assert.equal(status.state, "saved", "baseline should report saved");

        // Now arrange a divergent named file and run the drift check.
        // markStateSaved was just called, so the fingerprint baseline
        // matches the live state — but the named file on disk doesn't,
        // and that's the authoritative comparison.
        SERVER.presets.set("Baz.json", {
            kind: "koolook-snapshot",
            version: 1,
            name: "Baz",
            exportedAt: "2026-05-29T12:00:00.000Z",
            picks: ["EasyResize_Koolook", "EasyAIPipeline"],
            workflows: { directories: { "X": { workflows: { "Y": { graph: { nodes: [] }, archived: false, module: false, tags: [] } }, directories: {} } } },
        });
        const outcome = await detectBootDrift("Baz");
        assert.equal(outcome.drifted, true);

        status = getSnapshotStatus();
        assert.equal(status.state, "drifted", "drift should win precedence over saved baseline");
        assert.equal(status.name, "Baz");
        """
    )
    assert result.returncode == 0, result.stderr


# =============================================================================
# 3. markStateSaved clears the drift flag (Save/Load realignment)
# =============================================================================


def test_mark_state_saved_clears_drift_flag() -> None:
    """A successful Save/Load means the named file and live state are
    realigned. ``markStateSaved()`` is the single point both paths call;
    it must retire the drift warning so the pill returns to "saved"
    without the user having to dismiss anything."""
    result = _run(
        """
        SERVER.workflowsStore = { directories: {} };
        SERVER.presets.set("Qux.json", {
            kind: "koolook-snapshot",
            version: 1,
            name: "Qux",
            exportedAt: "2026-05-29T13:00:00.000Z",
            picks: ["EasyResize_Koolook"],
            workflows: { directories: { "X": { workflows: { "Y": { graph: { nodes: [] }, archived: false, module: false, tags: [] } }, directories: {} } } },
        });
        await loadWorkflowsStore();
        setAllPicks([]);
        setCurrentPresetName("Qux");

        await detectBootDrift("Qux");
        assert.equal(isBootDrifted(), true);

        // User realigns by Saving — production path runs markStateSaved().
        markStateSaved();

        assert.equal(isBootDrifted(), false, "Save/Load must retire the drift warning");
        const status = getSnapshotStatus();
        assert.equal(status.state, "saved");
        """
    )
    assert result.returncode == 0, result.stderr


def test_clear_boot_drift_public_export_works() -> None:
    """The exported ``clearBootDrift()`` escape hatch (for direct
    test-side reset) must flip the flag and refresh status."""
    result = _run(
        """
        SERVER.workflowsStore = { directories: {} };
        SERVER.presets.set("Zap.json", {
            kind: "koolook-snapshot",
            version: 1,
            name: "Zap",
            exportedAt: "2026-05-29T14:00:00.000Z",
            picks: ["EasyResize_Koolook"],
            workflows: { directories: { "X": { workflows: { "Y": { graph: { nodes: [] }, archived: false, module: false, tags: [] } }, directories: {} } } },
        });
        await loadWorkflowsStore();
        setAllPicks([]);
        setCurrentPresetName("Zap");

        await detectBootDrift("Zap");
        assert.equal(isBootDrifted(), true);
        clearBootDrift();
        assert.equal(isBootDrifted(), false);
        """
    )
    assert result.returncode == 0, result.stderr


# =============================================================================
# 4. Pre-load autosave routes to _unsaved_autosave/ while drifted
# =============================================================================


def test_pre_load_autosave_redirects_to_unsaved_when_drifted() -> None:
    """The same ``_autosaveSubdir`` helper that controls the periodic
    timer also controls pre-load autosaves. While drifted, both must
    route to ``_unsaved_autosave/`` so the named preset's recovery
    folder stays clean."""
    result = _run(
        """
        SERVER.workflowsStore = { directories: {} };
        SERVER.presets.set("Drifty.json", {
            kind: "koolook-snapshot",
            version: 1,
            name: "Drifty",
            exportedAt: "2026-05-29T15:00:00.000Z",
            picks: ["EasyResize_Koolook"],
            workflows: { directories: { "X": { workflows: { "Y": { graph: { nodes: [] }, archived: false, module: false, tags: [] } }, directories: {} } } },
        });
        await loadWorkflowsStore();
        setAllPicks([]);
        setCurrentPresetName("Drifty");

        await detectBootDrift("Drifty");
        assert.equal(isBootDrifted(), true);

        SERVER.writes.length = 0;
        await writePreLoadAutosave("test-redirect");

        const writtenDirs = SERVER.writes.map((w) => w.dir).filter(Boolean);
        assert.ok(writtenDirs.length > 0, "pre-load autosave should write at least one file");
        for (const d of writtenDirs) {
            assert.equal(d, "_unsaved_autosave",
                `pre-load autosave routed to "${d}" while drifted; expected _unsaved_autosave`);
        }
        """
    )
    assert result.returncode == 0, result.stderr


def test_detect_drift_does_not_fire_on_normalize_only_shape_differences() -> None:
    """Cross-machine snapshot import + hand-edited files (the documented
    NFS/SMB/Dropbox sync workflow this module supports) save fields in
    shapes the live cache passes through ``normalizeWorkflowsStore`` —
    ``archived: "false"`` becomes ``archived: false``; missing
    ``directories: {}`` is filled in; non-string tag entries get dropped.
    Without normalising the file side too, the first boot after a fresh
    cross-machine import would false-positive as drift and route every
    autosave to ``_unsaved_autosave/`` until the user re-saved.

    This guards the symmetric-fingerprint contract: when the disk shape
    and the live shape are equivalent *after* normalisation, no drift
    is flagged.
    """
    result = _run(
        """
        // Live: post-normalize shape — strict bool, deduped tags, present
        // directories: {}. This is what workflowsCache looks like the
        // moment loadWorkflowsStore() returns.
        SERVER.workflowsStore = {
            directories: {
                "Renders": {
                    workflows: {
                        "A": {
                            graph: { nodes: [{ id: 1 }] },
                            archived: false,
                            module: false,
                            tags: ["lit", "exterior"],
                        },
                    },
                    directories: {},
                },
            },
        };
        // File: pre-normalize shape — `archived` as a string literal,
        // duplicate + whitespace-padded tag entries, missing
        // `directories: {}` on the leaf node. Hand-edited / cross-
        // machine import shape that should normalize to the live shape.
        SERVER.presets.set("CrossMachine.json", {
            kind: "koolook-snapshot",
            version: 1,
            name: "CrossMachine",
            exportedAt: "2026-05-29T17:00:00.000Z",
            picks: [],
            workflows: {
                directories: {
                    "Renders": {
                        workflows: {
                            "A": {
                                graph: { nodes: [{ id: 1 }] },
                                archived: "false",
                                tags: ["lit", " exterior", "exterior ", "lit"],
                            },
                        },
                        // directories field deliberately omitted — pre-v0.3
                        // shape that normalize fills with `{}`.
                    },
                },
            },
        });
        await loadWorkflowsStore();
        setAllPicks([]);
        setCurrentPresetName("CrossMachine");

        const outcome = await detectBootDrift("CrossMachine");
        assert.equal(outcome.drifted, false,
            "shape differences that normalize to the same canonical form should NOT trigger drift");
        assert.equal(isBootDrifted(), false);
        """
    )
    assert result.returncode == 0, result.stderr


def test_pre_load_autosave_uses_preset_subdir_when_not_drifted() -> None:
    """Sanity check the other branch: when drift is NOT set,
    pre-load autosaves go to ``<preset>_autosave/`` per the regular
    contract — the drift redirect is the only override."""
    result = _run(
        """
        const sharedStore = {
            directories: {
                "X": {
                    workflows: {
                        "Y": { graph: { nodes: [] }, archived: false, module: false, tags: [] },
                    },
                    directories: {},
                },
            },
        };
        SERVER.workflowsStore = sharedStore;
        SERVER.presets.set("Clean.json", {
            kind: "koolook-snapshot",
            version: 1,
            name: "Clean",
            exportedAt: "2026-05-29T16:00:00.000Z",
            picks: [],
            workflows: sharedStore,
        });
        await loadWorkflowsStore();
        setAllPicks([]);
        setCurrentPresetName("Clean");

        await detectBootDrift("Clean");
        assert.equal(isBootDrifted(), false);

        SERVER.writes.length = 0;
        await writePreLoadAutosave("baseline");

        const writtenDirs = SERVER.writes.map((w) => w.dir).filter(Boolean);
        assert.ok(writtenDirs.length > 0);
        for (const d of writtenDirs) {
            assert.equal(d, "Clean_autosave",
                `non-drifted autosave should route to Clean_autosave/, got "${d}"`);
        }
        """
    )
    assert result.returncode == 0, result.stderr
