from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _tree_src() -> str:
    return (REPO_ROOT / "web" / "sidebar" / "tree.js").read_text(encoding="utf-8")


def _compare_guard_block(src: str) -> str:
    # The read-only capture-guard block lives between the NAV_ALLOW declaration
    # and the "Forward declarations" marker that follows the read-only block.
    start = src.index("const NAV_ALLOW =")
    return src[start:src.index("// Forward declarations", start)]


def run_node_scenario(source: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["node", "--input-type=module"],
        input=source,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


# ---------------------------------------------------------------------------
# Pure logic (#197 Phase 2a) — the snapshot-entry lookup the pull-in reads, and
# the copy-into-live-kit round-trip that carries graph + tags.
# ---------------------------------------------------------------------------
def test_get_workflow_entry_from_store_lookup() -> None:
    proc = run_node_scenario(
        textwrap.dedent(
            """
            import assert from "node:assert/strict";
            import { getWorkflowEntryFromStore } from "./web/sidebar/snapshot_diff.js";

            const store = { directories: {
                Basics: {
                    workflows: { txt2img: { graph: { nodes: [1, 2] }, tags: ["fast", "fav"] } },
                    directories: {
                        Sub: { workflows: { nested: { graph: { n: 9 } } }, directories: {} },
                    },
                },
            } };

            const top = getWorkflowEntryFromStore(store, "Basics/txt2img");
            assert.ok(top, "top-level entry resolves by full path");
            assert.deepEqual(top.graph, { nodes: [1, 2] });
            assert.deepEqual(top.tags, ["fast", "fav"]);

            const nested = getWorkflowEntryFromStore(store, "Basics/Sub/nested");
            assert.ok(nested, "nested entry resolves");
            assert.deepEqual(nested.graph, { n: 9 });

            assert.equal(getWorkflowEntryFromStore(store, "Basics/missing"), null);
            assert.equal(getWorkflowEntryFromStore(null, "Basics/txt2img"), null);
            """
        )
    )
    assert proc.returncode == 0, proc.stderr


def test_pull_in_copy_round_trips_graph_and_tags_into_live_kit() -> None:
    # The B->A pull-in copies a snapshot workflow into the live store the way
    # the context-menu handler does: saveWorkflowEntry + re-apply the entry's
    # tags. The copied entry must carry the same graph and tags, the live
    # export must contain it (so Save round-trips it), and the SOURCE snapshot
    # must be left untouched (no rekey, no mutation) — the 2a read-only promise.
    proc = run_node_scenario(
        textwrap.dedent(
            """
            import assert from "node:assert/strict";
            import { getWorkflowEntryFromStore } from "./web/sidebar/snapshot_diff.js";
            import {
                saveWorkflowEntry,
                addTag,
                getAllWorkflowsForExport,
                getWorkflowTags,
                getWorkflowGraph,
            } from "./web/sidebar/workflows_store.js";

            const snapshot = { directories: {
                Basics: {
                    workflows: { txt2img: { graph: { nodes: [1, 2] }, tags: ["fast", "fav"] } },
                    directories: {},
                },
            } };
            const snapshotBefore = JSON.stringify(snapshot);

            // What the pull-in does on Save. The graph is DEEP-CLONED first so
            // the live entry never aliases the snapshot's graph object (the
            // aliasing guard at the end pins this).
            const entry = getWorkflowEntryFromStore(snapshot.directories ? snapshot : null, "Basics/txt2img");
            const graphForCopy = JSON.parse(JSON.stringify(entry.graph));
            const res = saveWorkflowEntry(["Basics"], "txt2img", graphForCopy, { module: false });
            assert.ok(res, "entry saved into the live kit");
            for (const tag of entry.tags) addTag(["Basics"], "txt2img", tag);

            const live = getAllWorkflowsForExport();
            const copied = live.directories.Basics.workflows.txt2img;
            assert.deepEqual(copied.graph, { nodes: [1, 2] }, "graph rides the copy");
            assert.deepEqual([...getWorkflowTags(["Basics"], "txt2img")].sort(), ["fast", "fav"], "tags ride the copy");

            // Same path/name preserved — never rekeyed.
            assert.ok(live.directories.Basics.workflows.txt2img, "path + name preserved");

            // Source snapshot object untouched by the copy itself.
            assert.equal(JSON.stringify(snapshot), snapshotBefore, "source snapshot is not mutated by the copy");

            // Aliasing guard (#197 read-only promise): the copy must DEEP-CLONE the
            // graph, not alias the snapshot's object into the live store. Otherwise a
            // later edit of the copied (live) workflow bleeds back into the read-only
            // snapshot. Probe the real failure: mutate the live graph in place, then
            // assert the snapshot's graph is still pristine.
            getWorkflowGraph(["Basics"], "txt2img").nodes.push(99);
            assert.deepEqual(
                snapshot.directories.Basics.workflows.txt2img.graph,
                { nodes: [1, 2] },
                "editing the live copy must not mutate the source snapshot's graph",
            );
            """
        )
    )
    assert proc.returncode == 0, proc.stderr


# ---------------------------------------------------------------------------
# Path-preserving copy engine (#197 extended) — copyWorkflowIntoStore replicates
# the source's folder path in the destination, auto-creating / merging dirs,
# with skip-identical / keep-both collision handling. Pure, store-agnostic
# (used for both the live store and a loaded snapshot).
# ---------------------------------------------------------------------------
def test_copy_engine_path_preserving_create_and_merge() -> None:
    proc = run_node_scenario(
        textwrap.dedent(
            """
            import assert from "node:assert/strict";
            import { copyWorkflowIntoStore } from "./web/sidebar/workflows_store.js";

            // Empty destination -> the whole path is auto-created.
            const dest = { directories: {} };
            let r = copyWorkflowIntoStore(dest, ["Basics", "Upscale"], "myWf", { nodes: [1] }, { tags: ["fav"], module: false });
            assert.equal(r.status, "added");
            assert.equal(r.finalName, "myWf");
            const leaf = dest.directories.Basics.directories.Upscale.workflows.myWf;
            assert.deepEqual(leaf.graph, { nodes: [1] }, "graph lands at the same path");
            assert.deepEqual(leaf.tags, ["fav"], "tags ride");

            // A second, different item into an EXISTING folder merges in place —
            // no duplicate Basics/Upscale folder is created.
            r = copyWorkflowIntoStore(dest, ["Basics", "Upscale"], "otherWf", { nodes: [2] }, {});
            assert.equal(r.status, "added");
            assert.equal(Object.keys(dest.directories).length, 1, "no duplicate top folder");
            assert.equal(Object.keys(dest.directories.Basics.directories.Upscale.workflows).length, 2, "merged into existing folder");
            """
        )
    )
    assert proc.returncode == 0, proc.stderr


def test_copy_engine_collision_skip_identical_and_keep_both() -> None:
    proc = run_node_scenario(
        textwrap.dedent(
            """
            import assert from "node:assert/strict";
            import { copyWorkflowIntoStore } from "./web/sidebar/workflows_store.js";

            const dest = { directories: { Basics: { workflows: { wf: { graph: { nodes: [1] }, tags: [] } }, directories: {} } } };

            // Identical graph at the same path -> skip (already there), no new key.
            let r = copyWorkflowIntoStore(dest, ["Basics"], "wf", { nodes: [1] }, {});
            assert.equal(r.status, "skipped");
            assert.equal(Object.keys(dest.directories.Basics.workflows).length, 1, "skip adds nothing");

            // Different graph, same name -> keep both with a suffix.
            r = copyWorkflowIntoStore(dest, ["Basics"], "wf", { nodes: [9] }, { sourceLabel: "snapA" });
            assert.equal(r.status, "kept-both");
            assert.equal(r.finalName, "wf (from snapA)");
            assert.ok(dest.directories.Basics.workflows["wf (from snapA)"], "kept-both entry exists");
            assert.deepEqual(dest.directories.Basics.workflows.wf.graph, { nodes: [1] }, "original untouched");
            """
        )
    )
    assert proc.returncode == 0, proc.stderr


def test_copy_folder_bulk_merge() -> None:
    # "Copy folder to the other side" copies every active workflow under the
    # folder (recursively, path-preserving), skips archived, and reports a merge
    # summary. dirSegs [] copies the whole tree.
    proc = run_node_scenario(
        textwrap.dedent(
            """
            import assert from "node:assert/strict";
            import { copyFolderIntoStore } from "./web/sidebar/workflows_store.js";

            const source = { directories: {
                Basics: {
                    workflows: { a: { graph: { n: 1 } }, old: { graph: { n: 9 }, archived: true } },
                    directories: { Sub: { workflows: { b: { graph: { n: 2 }, tags: ["x"] } }, directories: {} } },
                },
                Other: { workflows: { z: { graph: { n: 3 } } }, directories: {} },
            } };

            // Copy just Basics (recursively), path-preserving, archived skipped.
            const target = { directories: {} };
            let s = copyFolderIntoStore(target, source, ["Basics"], { sourceLabel: "snap" });
            assert.equal(s.total, 2, "2 active under Basics (archived skipped)");
            assert.equal(s.added, 2);
            assert.deepEqual(target.directories.Basics.workflows.a.graph, { n: 1 }, "a at Basics/a");
            assert.deepEqual(target.directories.Basics.directories.Sub.workflows.b.graph, { n: 2 }, "b at Basics/Sub/b");
            assert.deepEqual(target.directories.Basics.directories.Sub.workflows.b.tags, ["x"], "tags ride");
            assert.ok(!("old" in target.directories.Basics.workflows), "archived not copied");
            assert.ok(!("Other" in target.directories), "only the chosen folder copied");

            // Identical re-copy -> all skips.
            s = copyFolderIntoStore(target, source, ["Basics"], {});
            assert.equal(s.skipped, 2);
            assert.equal(s.added, 0);

            // dirSegs [] copies the entire tree.
            const all = { directories: {} };
            s = copyFolderIntoStore(all, source, [], {});
            assert.equal(s.total, 3, "whole-tree copy gets every active workflow");
            assert.ok(all.directories.Other.workflows.z, "Other/z copied");
            """
        )
    )
    assert proc.returncode == 0, proc.stderr


def test_copy_engine_rejects_prototype_polluting_names() -> None:
    # The copy engine writes user-named keys (from a possibly-crafted snapshot),
    # so it must reject __proto__/constructor/prototype like the other name-keyed
    # store mutators (#203) — no Object.prototype pollution.
    proc = run_node_scenario(
        textwrap.dedent(
            """
            import assert from "node:assert/strict";
            import { copyWorkflowIntoStore } from "./web/sidebar/workflows_store.js";

            const target = { directories: {} };
            assert.equal(copyWorkflowIntoStore(target, ["Basics"], "__proto__", { n: 1 }, {}).status, "failed");
            assert.equal(copyWorkflowIntoStore(target, ["__proto__"], "wf", { n: 1 }, {}).status, "failed");
            assert.equal(copyWorkflowIntoStore(target, ["Basics"], "constructor", { n: 1 }, {}).status, "failed");
            assert.equal(({}).n, undefined, "Object.prototype not polluted");
            """
        )
    )
    assert proc.returncode == 0, proc.stderr


def test_copy_engine_deep_clones_graph() -> None:
    # The engine must deep-clone so the destination never aliases the source
    # graph object — a later edit on either side must not bleed across.
    proc = run_node_scenario(
        textwrap.dedent(
            """
            import assert from "node:assert/strict";
            import { copyWorkflowIntoStore } from "./web/sidebar/workflows_store.js";

            const srcGraph = { nodes: [1, 2] };
            const dest = { directories: {} };
            copyWorkflowIntoStore(dest, ["D"], "wf", srcGraph, {});
            dest.directories.D.workflows.wf.graph.nodes.push(99);
            assert.deepEqual(srcGraph, { nodes: [1, 2] }, "source graph must be untouched by edits to the copy");
            """
        )
    )
    assert proc.returncode == 0, proc.stderr


def test_diff_excludes_archived_entries() -> None:
    # Archived workflow versions (surfaced under the synthetic "Archive" folder)
    # are old setups — noise, not copy candidates — so the Compare diff must NOT
    # classify them as new/modified. The copy-lookup path keeps them (tested via
    # getWorkflowEntryFromStore still resolving an archived path).
    proc = run_node_scenario(
        textwrap.dedent(
            """
            import assert from "node:assert/strict";
            import { diffWorkflows, getWorkflowEntryFromStore } from "./web/sidebar/snapshot_diff.js";

            const working = { directories: {} };
            const comparison = { directories: {
                Basics: { workflows: {
                    active: { graph: { n: 1 } },
                    "old (archived 2026)": { graph: { n: 2 }, archived: true },
                }, directories: {} },
            } };

            const status = diffWorkflows(working, comparison);
            assert.equal(status["Basics/active"], "new", "active entry is still classified");
            assert.ok(!("Basics/old (archived 2026)" in status), "archived entry excluded from the diff");

            // ...but it can still be looked up for an explicit copy.
            assert.ok(getWorkflowEntryFromStore(comparison, "Basics/old (archived 2026)"), "archived stays copyable");
            """
        )
    )
    assert proc.returncode == 0, proc.stderr


# ---------------------------------------------------------------------------
# Wiring invariants (#197) — pin the pull-in into the read-only guard, keep the
# copy path additive (never applySnapshot), and require the A<->B swap control.
# ---------------------------------------------------------------------------
def test_readonly_guard_routes_pullable_rows_to_copy_menu() -> None:
    guard = _compare_guard_block(_tree_src())
    # The contextmenu handler still blocks by default but routes node + workflow
    # rows to a pull-in copy menu when a pull-in is wired.
    assert "data-koolook-node-type" in guard, "node rows must be pull-able"
    assert "data-koolook-wf-path" in guard, "workflow rows must be pull-able"
    assert "showContextMenu" in guard, "the pull-in opens a context menu"
    assert "pullIn" in guard, "the guard consults the pull-in descriptor"


def test_readonly_guard_keeps_all_capture_handlers() -> None:
    # The #183 non-destructive guarantee must survive the 2a change: all five
    # capture-phase handlers stay, so drag/drop/click stay neutralized.
    guard = _compare_guard_block(_tree_src())
    for evt in ("click", "contextmenu", "dragstart", "dragover", "drop"):
        assert f'addEventListener("{evt}"' in guard, f"missing capture handler for '{evt}'"
    assert guard.count(", true)") >= 5, "all guard listeners must be capture-phase"


def _copy_layer(src: str) -> str:
    # The whole bidirectional copy layer: from makePullIn through the last copy
    # handler, up to the next major function (applyCompareTint).
    start = src.index("function makePullIn(")
    return src[start:src.index("function applyCompareTint(", start)]


def test_copy_layer_is_additive_and_bidirectional() -> None:
    layer = _copy_layer(_tree_src())
    # Both directions exist: into the live kit AND into the snapshot.
    assert "copyWorkflowIntoLiveStore(" in layer, "B->A copies into the live store"
    assert "copyWorkflowIntoStore(" in layer, "A->B copies into the snapshot store"
    assert "copyNodeToLive(" in layer and "copyNodeToSnapshot(" in layer, "node copy is bidirectional"
    assert "addToMyPicks(" in layer, "B->A node copy adds to favorites"
    # Never destructive: the copy path must not apply the snapshot.
    assert "applySnapshot" not in layer, "the copy layer must never apply the snapshot"


def test_copy_is_path_preserving_without_a_folder_picker() -> None:
    # The user-facing change: copy lands at the SOURCE's folder path (the engine
    # auto-creates / merges), so there is no manual folder-picker step anymore.
    layer = _copy_layer(_tree_src())
    assert "splitWfPath(" in layer, "the destination path is derived from the source path"
    assert "copyWorkflowIntoLiveStore(dirSegs" in layer, "live copy preserves the source dir path"
    assert "showSaveWorkflowModal" not in layer, "the copy no longer opens a folder-picker modal"


def test_snapshot_edits_are_in_memory_not_per_copy_disk_writes() -> None:
    # A->B copies edit the in-memory snapshot working copy and mark it dirty —
    # NO per-copy writePreset (autosaves stay a pure safety net, never an A->B
    # write target).
    src = _tree_src()
    start = src.index("function copyWorkflowToSnapshot(")
    body = src[start:src.index("\nfunction ", start + 10)]
    assert "compareSnapshot.workflows" in body, "A->B edits the in-memory snapshot"
    assert "compareDirty = true" in body, "A->B marks the snapshot dirty"
    assert "writePreset" not in body, "A->B must not write to disk per copy"


def test_snapshot_save_writes_named_file_and_clears_dirty() -> None:
    # The explicit Save writes a NAMED snapshot (default library dir — never an
    # autosave subdir), pre-fills the shadowed name, and clears the dirty flag.
    src = _tree_src()
    start = src.index("function saveCompareSnapshot(")
    body = src[start:src.index("\nfunction ", start + 10)]
    assert "writePreset(name, snapshotRef)" in body, "Save writes a named file to the default dir"
    assert "compareDirty = false" in body, "Save clears the dirty flag"
    assert "compareDefaultSaveName(" in body, "Save pre-fills the shadowed / loaded name"
    # The default-name helper strips the _autosave suffix -> save to the named
    # parent, never the rotating autosave shadow.
    helper_start = src.index("function compareDefaultSaveName(")
    helper = src[helper_start:src.index("\nfunction ", helper_start + 10)]
    assert "_autosave" in helper and 'slice(0, -"_autosave".length)' in helper, (
        "derive the named parent from the autosave dir"
    )


def test_snapshot_save_guards_overwrite_and_sanitizes() -> None:
    # writePreset ends in an atomic os.replace, and the name is pre-filled, so
    # Save must sanitize + confirm before overwriting an existing file (mirrors
    # the live Save flow) and must not fire overlapping writes.
    src = _tree_src()
    start = src.index("function saveCompareSnapshot(")
    body = src[start:src.index("\nfunction ", start + 10)]
    assert "sanitizeName(" in body, "Save sanitizes the name like other writePreset callers"
    assert "presetExists(" in body, "Save checks existence before writing"
    assert "exists === null" in body, "Save aborts if the library can't be reached"
    assert "showConfirmModal(" in body, "Save confirms before overwriting an existing snapshot"
    assert "compareSaving" in body, "Save guards against overlapping in-flight writes"


def test_exit_warns_on_unsaved_snapshot_edits() -> None:
    src = _tree_src()
    start = src.index("function exitCompareMode(")
    body = src[start:src.index("\nfunction ", start + 10)]
    assert "compareDirty" in body and "showConfirmModal" in body, (
        "exiting with unsaved snapshot edits must confirm before discarding them"
    )


def test_save_state_is_its_own_stripe_separate_from_orientation() -> None:
    # The SOURCE/TARGET footers carry orientation only; all save state /
    # instructions / dirty / Save button live in a SEPARATE stripe.
    src = _tree_src()
    assert "function buildSaveStripe(" in src, "a dedicated save stripe exists"
    assert "koolook-compare-savebar" in src, "the save stripe is its own element"
    assert "koolook-savebar-btn" in src, "the save stripe carries the Save button"
    assert "koolook-savebar-unsaved" in src, "the save stripe shows the unsaved state"
    # The orientation footer no longer carries save behavior.
    foot_start = src.index("function labelColumnFoot(")
    foot = src[foot_start:src.index("\nfunction ", foot_start + 10)]
    assert "savebtn" not in foot and "auto-save" not in foot, "footer is orientation-only now"


def test_autosave_target_shows_clean_name_not_long_path() -> None:
    # An autosave-loaded target must show a clean derived name (parent + marker),
    # never the long `Foo_autosave/pre_load_...` path.
    src = _tree_src()
    assert "function compareSnapName(" in src, "snapshot display name is cleaned"
    assert "· autosave" in src, "autosave targets are marked, named-parent shown"
    assert 'slice(0, -"_autosave".length)' in src, "derive the named parent from the autosave dir"


def test_source_target_footers_and_legend_present() -> None:
    src = _tree_src()
    assert "function labelColumnFoot(" in src, "each column gets a SOURCE/TARGET footer"
    assert '"TARGET"' in src and '"SOURCE"' in src, "footers label source vs target"
    assert "koolook-compare-colfoot" in src, "footer stripe is rendered"


def test_diff_filter_browser_present() -> None:
    src = _tree_src()
    assert "let compareFilter" in src, "compare host tracks the diff filter"
    assert "function applyCompareFilter(" in src, "a post-render pass applies the filter"
    assert "koolook-cmp-chip" in src, "the legend doubles as clickable filter chips"
    # Force-expand matching folders + hide the rest via the nested children DOM.
    flt = src[src.index("function applyCompareFilter("):src.index("function applyCompareFilter(") + 1400]
    assert "koolook-children" in flt, "filter walks the nested folder DOM"
    assert 'style.display = "none"' in flt, "filter hides non-matching rows/folders"


def test_diff_highlight_and_filter_follow_the_source_side() -> None:
    # The green/red highlight + the filter must land on the SOURCE side (the
    # read-only side you copy FROM), flipping with the swap — not hardwired to
    # the snapshot panel. And "new" must mean "in the source, not the target"
    # (the copy candidates), so the diff direction follows the swap too.
    src = _tree_src()
    assert "const sourceHost = liveIsActive ? rightCol.host : leftCol.host" in src, (
        "tint + filter apply to the source (inactive) panel, flipping with the swap"
    )
    assert "applyCompareTint(sourceHost," in src and "applyCompareFilter(sourceHost)" in src
    tint = src[src.index("function applyCompareTint("):src.index("function applyCompareTint(") + 900]
    assert "diffPicks(targetPicks, sourcePicks)" in tint, "new picks = in source, absent from target"
    assert "diffWorkflows(targetStore, sourceStore)" in tint, "new/diff workflows = source vs target"


def test_compare_panels_have_isolated_expansion_state() -> None:
    # The two compare panels must NOT share folder-expansion state. renderTree
    # prunes the active pathStates map to the rendering panel's paths, so a
    # shared map means one panel's re-render (e.g. after a copy) collapses the
    # other panel's open folders. The snapshot panel renders under its own map.
    src = _tree_src()
    assert "let pathStates = new Map()" in src, "the live expansion map must be swappable"
    assert "const comparePathStates = new Map()" in src, "snapshot panel needs its own expansion map"
    assert "function withComparePathStates(" in src
    assert "withSnapshotSource(snapshot, () => withComparePathStates(fn))" in src, (
        "compare tree renders must run under the compare expansion map"
    )
    # A->B copy opens its destination folder in the snapshot panel's own map.
    assert "function openCompareDestPath(" in src
    assert "openCompareDestPath(dirSegs)" in src, "A->B copy opens its dest folder"


def test_folder_pull_in_is_wired() -> None:
    # Right-click a workflow folder on the source side -> "Copy folder (with
    # contents)" bulk-copies it path-preservingly to the target.
    src = _tree_src()
    assert "row.dataset.koolookFolderPath = path" in src, "folder rows carry their tree path"
    guard = _compare_guard_block(src)
    assert "data-koolook-folder-path" in guard, "guard recognizes folder rows"
    assert "copyFolder(" in guard, "guard offers the folder copy"
    layer = _copy_layer(src)
    assert "copyFolderIntoLiveStore(" in layer, "B->A bulk copy into the live kit"
    assert "copyFolderIntoStore(" in layer, "A->B bulk copy into the snapshot"
    assert "function copyFolderToLive(" in src and "function copyFolderToSnapshot(" in src


def test_swap_toggle_is_wired_in_compare_host() -> None:
    src = _tree_src()
    assert "let activeSide" in src, "compare host tracks which side is active"
    assert "function swapCompareSides(" in src, "a swap handler flips the active side"
    assert "koolook-compare-swap" in src, "the swap control is rendered in the compare host"


def test_autosave_dir_threaded_for_writeback() -> None:
    # A->B write-back must target the exact file the snapshot was read from, so
    # the autosave subfolder dir rides the compare onChoose meta.
    src = (REPO_ROOT / "web" / "sidebar" / "modals.js").read_text(encoding="utf-8")
    assert "dir: item.dir" in src, "compare onChoose must pass the autosave dir for write-back"
