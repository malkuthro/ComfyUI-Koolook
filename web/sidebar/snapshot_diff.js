// SPDX-FileCopyrightText: 2025-2026 Kforge Labs <https://github.com/malkuthro/ComfyUI-Koolook>
// SPDX-License-Identifier: GPL-3.0-only

import { sortJsonValue } from "./workflows_store.js";

// Pure snapshot-comparison helpers for the Compare-mode sidebar (issue #181).
// No DOM, no globals, no I/O — every function takes plain data and returns
// plain data so it can be unit-tested by piping a scenario through Node. The
// Compare view renders the existing sidebar a second time and uses these
// results purely to tint comparison rows (green = only in comparison, red =
// same key, contents differ); it never mutates the working state.

// Partition two pick lists (arrays of string identifiers) against each other.
// Order follows the `working` list for `onlyWorking`/`shared` and the
// `comparison` list for `onlyComparison`. Membership is by set, so duplicates
// collapse to a single classification.
export function diffPicks(working, comparison) {
    const workingSet = new Set(working);
    const comparisonSet = new Set(comparison);
    return {
        onlyWorking: working.filter((id) => !comparisonSet.has(id)),
        onlyComparison: comparison.filter((id) => !workingSet.has(id)),
        shared: working.filter((id) => comparisonSet.has(id)),
    };
}

// Flatten a workflows store ({ directories: { [name]: DirNode } }, the shape
// returned by getAllWorkflowsForExport / stored under snapshot.workflows) into
// a Map of full path ("Basics/txt2img", "Basics/Sub/wf") -> WorkflowEntry.
function collectWorkflows(node, prefix, out, skipArchived = false) {
    const dirs = node && typeof node.directories === "object" && node.directories ? node.directories : {};
    for (const dirName of Object.keys(dirs)) {
        const child = dirs[dirName];
        if (!child || typeof child !== "object") continue;
        const childPrefix = `${prefix}${dirName}/`;
        const wfs = child.workflows && typeof child.workflows === "object" ? child.workflows : {};
        for (const wfName of Object.keys(wfs)) {
            // Archived entries are old versions surfaced under the synthetic
            // "Archive" folder. The diff skips them (skipArchived) so stale
            // setups don't read as new/modified noise (#197); the copy-lookup
            // path leaves them in so an archived version can still be copied.
            if (skipArchived && wfs[wfName] && wfs[wfName].archived === true) continue;
            out.set(`${childPrefix}${wfName}`, wfs[wfName]);
        }
        collectWorkflows(child, childPrefix, out, skipArchived);
    }
    return out;
}

// Two workflow graphs are equal when their canonical (key-order-normalized)
// JSON matches. Comparing the `graph` subtree alone ignores the sibling
// `savedAt` stamp, which is rewritten on every save and would otherwise make
// every re-saved workflow read as changed.
function graphsEqual(a, b) {
    return JSON.stringify(sortJsonValue(a)) === JSON.stringify(sortJsonValue(b));
}

// Classify every workflow in the comparison store against the working store,
// keyed by full path:
//   "new"  -> present only in the comparison (green tint)
//   "diff" -> present in both, graph differs ignoring savedAt (red tint)
//   "same" -> present in both, identical graph (plain)
export function diffWorkflows(workingStore, comparisonStore) {
    const working = collectWorkflows(workingStore, "", new Map(), true);
    const comparison = collectWorkflows(comparisonStore, "", new Map(), true);
    const status = {};
    for (const [path, entry] of comparison) {
        if (!working.has(path)) {
            status[path] = "new";
        } else {
            status[path] = graphsEqual(entry.graph, working.get(path).graph) ? "same" : "diff";
        }
    }
    return status;
}

// Resolve a single workflow entry in a raw store by full path
// ("Basics/txt2img", "Basics/Sub/wf"), or null if absent. Reuses the same
// flatten the diff uses, so lookups follow identical path semantics. Pure:
// reads only the passed store. The Compare-mode pull-in (#197) uses this to
// read a snapshot workflow's graph + tags so it can copy the entry into the
// live kit WITHOUT first applying the snapshot.
export function getWorkflowEntryFromStore(store, fullPath) {
    if (!store || typeof store !== "object" || typeof fullPath !== "string") return null;
    const all = collectWorkflows(store, "", new Map());
    return all.has(fullPath) ? all.get(fullPath) : null;
}
