// SPDX-FileCopyrightText: 2025-2026 Kforge Labs <https://github.com/malkuthro/ComfyUI-Koolook>
// SPDX-License-Identifier: GPL-3.0-only

// =============================================================================
// Sidebar tree — section-registry engine, data gathering, DOM row factories,
// folder builder, context menus for workflow/directory rows, and the panel
// renderer (search, action bars, tree mount, event subscriptions).
//
// `pathStates` is module-private — folder expansion state survives across
// re-renders triggered by saves, picks-changed events, or storage events.
// Stale path keys are pruned every render against the set of paths actually
// emitted, so renames and deletes don't leak entries.
// =============================================================================
import {
    REPOS,
    ROOT_GROUP_LABEL,
    WORKFLOWS_GROUP_LABEL,
    STORAGE_KEY,
    WORKFLOWS_FALLBACK_KEY,
    PICKS_CHANGED_EVENT,
    WORKFLOWS_CHANGED_EVENT,
    SNAPSHOT_STATUS_CHANGED_EVENT,
    GROUP_MODE_KEY,
    GROUP_MODE_DEFAULT,
    MODULE_TAG,
    ensureStyle,
    toast,
    compareNames,
} from "./constants.js";
import {
    loadUserPicks,
    removeFromMyPicks,
    notifyPicksChanged,
    addToMyPicks,
} from "./picks_store.js";
import {
    persistMutation,
    listDirectoryNames,
    listAllDirectoryPaths,
    dirOf,
    addDirectory,
    renameDirectory,
    deleteDirectory,
    saveWorkflowEntry,
    archiveWorkflow,
    unarchiveWorkflow,
    renameWorkflow,
    deleteWorkflow,
    moveWorkflow,
    moveDirectory,
    clearArchive,
    pathsEqual,
    getWorkflowGraph,
    getWorkflowTags,
    addTag,
    removeTag,
} from "./workflows_store.js";
import {
    serializeFullCanvas,
    serializeSelection,
    canvasIsNonEmpty,
    loadWorkflowOntoCanvas,
    insertWorkflowOntoCanvas,
    insertNode,
    getSelectedNodeTypes,
    dropPlaceholdersForPacks,
} from "./canvas_io.js";
import { discoverMissingPacks } from "./installer.js";
import {
    showInputModal,
    showConfirmModal,
    showSaveWorkflowModal,
    showContextMenu,
    showTagsModal,
    showInstallMissingModal,
    showSaveSnapshotDialog,
    showLoadSnapshotDialog,
    showSnapshotSettingsDialog,
} from "./modals.js";
import { attachHoverPreview, teardownPreview } from "./node_preview.js";
import {
    sanitizeName,
    gatherSnapshot,
    applySnapshot,
    listPresets,
    readPreset,
    writePreset,
    presetExists,
    deletePreset,
    getLibraryInfo,
    getSettings,
    saveSettings,
    getCurrentPresetName,
    setCurrentPresetName,
    exportStarterPreset,
    writePreLoadAutosave,
    markStateSaved,
    getSnapshotStatus,
    listAutosaves,
} from "./snapshot.js";

// =============================================================================
// Built-in section IDs. These are also pathState key prefixes — every
// `pathStates` entry begins with one of these strings followed by `/`. The
// engine handles the prefixing for paths emitted via `ctx.folder({path: …})`,
// but `renderPanel`'s save flow has to construct pin keys directly, so the
// constants are exported here (module-private) instead of duplicated as
// literals.
// =============================================================================
const SECTION_ID_NODES = "nodes";
const SECTION_ID_WORKFLOWS = "workflows";
const SECTION_ID_TAGS = "tags";

// =============================================================================
// Folder expansion state — Map<path, boolean>; truthy = expanded.
// Path keys are auto-prefixed by the engine with the owning section's id, so
// section authors only write segment-relative paths.
// =============================================================================
const pathStates = new Map();

// Paths flagged by `pinExpanded` to be force-opened on the next render. The
// set is consumed each render: kept entries are written to pathStates and the
// set is cleared. Used by save flows that mutate state and want the
// destination folder open after the resulting re-render.
const pinnedPaths = new Set();

// Section registry. Sections render in declaration order, separated by
// dividers. Built-in sections register at module load (bottom of this file).
const SECTIONS = [];

// =============================================================================
// Engine — public surface (3 functions)
// =============================================================================

/**
 * @typedef {Object} FolderOpts
 * @property {string} name
 * @property {number} [count]
 * @property {string} [iconKind]                      "favorites" | "workflows"
 *                                                    | "archive" | "folder".
 * @property {string} path                            Segment-relative; engine
 *                                                    prefixes with section.id.
 * @property {boolean} [startExpanded=false]
 * @property {(e: MouseEvent) => void} [onContextMenu]
 * @property {(sub: SectionCtx) => void} build        Populate the sub-folder.
 *
 * @typedef {Object} SectionCtx
 * @property {*} data                                 Result returned by `gather`.
 *                                                    Top-level ctx only; nested
 *                                                    builds close over their
 *                                                    own data via the outer
 *                                                    `build` scope.
 * @property {string} query                           Lowercased+trimmed search.
 * @property {boolean} isFiltered                     True when query is non-empty.
 * @property {(opts: FolderOpts) => void} folder      Append a sub-folder.
 * @property {(opts: { row: HTMLElement }) => void} leaf
 *                                                    Append a leaf row.
 *
 * @typedef {Object} SectionSpec
 * @property {string} id                              Stable; used as the
 *                                                    pathState root for
 *                                                    this section.
 * @property {string} label                           Group folder header.
 * @property {string} [iconKind]                      Same enum as FolderOpts.
 * @property {string} [emptyMessage]                  Shown when the section
 *                                                    gathers nothing AND no
 *                                                    search is active.
 * @property {(query: string) => { total: number }} gather
 *                                                    Returns a result with at
 *                                                    least `total`. Other
 *                                                    fields are opaque to
 *                                                    the engine; the section
 *                                                    reads them via
 *                                                    `ctx.data` in `build`.
 * @property {(ctx: SectionCtx) => void} build        Called when total > 0;
 *                                                    uses `ctx.folder` and
 *                                                    `ctx.leaf` to populate
 *                                                    the section tree.
 */

/**
 * Register a tree section.
 *
 * @param {SectionSpec} spec
 */
function addSection(spec) {
    SECTIONS.push(spec);
}

/**
 * Mark paths as expanded for the next render. The paths must include the
 * section-id prefix (e.g. "workflows", "workflows/MyDir"). Called from save
 * flows so the destination folder is open after the resulting re-render.
 *
 * @param {string[]} paths
 */
function pinExpanded(paths) {
    for (const p of paths) pinnedPaths.add(p);
}

/**
 * Spotlight effect for newly-saved picks: collapse every folder under the
 * Nodes section, then pin the just-added node's pack-folder + subcategory-
 * folder open. Designed as a pedagogical aid — when the user adds a node
 * via the toolbar `+` or the right-click "Add to Kforge Labs", the sidebar
 * snaps to focus on exactly where that node lives in the pack hierarchy,
 * helping newcomers internalize which pack/category each node belongs to.
 *
 * Caller is responsible for triggering a re-render afterwards (typically
 * via `notifyPicksChanged()`); this function only mutates `pathStates` +
 * `pinnedPaths`.
 *
 * Multi-add (caller passes several types): every pack/subcategory hit by
 * the additions is pinned open. So adding 5 nodes from 3 packs lights up
 * all 3 pack folders simultaneously rather than just the last.
 *
 * Idempotent / safe: types not in the LiteGraph registry are silently
 * skipped (no-op). Empty input → no-op (no collapse, no pin).
 *
 * @param {string[]} typeNames
 */
export function spotlightAddedPicks(typeNames) {
    if (!Array.isArray(typeNames) || typeNames.length === 0) return;
    const pathsToExpand = new Set();
    for (const t of typeNames) {
        const loc = findPackPathForType(t);
        if (!loc) continue;
        pathsToExpand.add(`${SECTION_ID_NODES}/${loc.packLabel}`);
        pathsToExpand.add(`${SECTION_ID_NODES}/${loc.packLabel}/${loc.sub}`);
    }
    if (pathsToExpand.size === 0) return;

    // Collapse every existing Nodes-section entry (the section root + every
    // pack/subcategory). The render-time fallback (`startExpanded`) keeps
    // the section header itself open since `addSection` registers it with
    // `startExpanded: true`; pack folders default to closed, which is what
    // we want for everything except the spotlighted location.
    for (const key of [...pathStates.keys()]) {
        if (key === SECTION_ID_NODES || key.startsWith(SECTION_ID_NODES + "/")) {
            pathStates.delete(key);
        }
    }
    pinExpanded([...pathsToExpand]);
}

/**
 * Single render entry point. Idempotent — safe to call from search input,
 * picks-changed, workflows-changed, or storage events.
 *
 * @param {{ treeEl: HTMLElement, query: string }} opts
 */
function renderTree({ treeEl, query }) {
    // Tear down any active hover preview before detaching rows. `innerHTML
    // = ""` removes leaf rows without firing `mouseleave`, so a card that
    // was up at the moment of re-render would otherwise leak — its anchor
    // gone, the card's `pointer-events: none` blocks user dismissal, and
    // it'd persist until the next successful hover. The preview module
    // owns a singleton card; one call clears whatever's live.
    teardownPreview();
    treeEl.innerHTML = "";
    const q = (query || "").trim().toLowerCase();
    const isFiltered = q.length > 0;
    const validPaths = new Set();
    // Track which sections actually rendered this pass. Phase 3 only prunes
    // keys whose section rendered — sections gated empty by a search filter
    // keep their pathStates entries so clearing the filter restores user
    // expansions.
    const renderedSectionIds = new Set();

    // Phase 1: gather every section.
    const gathered = SECTIONS.map(section => ({
        section,
        result: section.gather(q),
    }));

    // Phase 2: render. Cross-section empty under a non-trivial search emits a
    // single placeholder; otherwise iterate sections with dividers between
    // non-empty ones. Phase 3 ALWAYS runs after this, so pins always apply.
    const anyVisible = gathered.some(({ result }) => result.total > 0);
    if (!anyVisible && isFiltered) {
        const empty = document.createElement("div");
        empty.className = "koolook-empty";
        empty.textContent = "No nodes or workflows match your search.";
        treeEl.appendChild(empty);
    } else {
        let appended = 0;
        for (const { section, result } of gathered) {
            if (result.total === 0) {
                // Per-section empty placeholder only when not filtered. Under
                // filter, the cross-section "no matches" message handles it.
                if (!isFiltered && section.emptyMessage) {
                    if (appended > 0) appendDivider(treeEl);
                    const el = document.createElement("div");
                    el.className = "koolook-empty";
                    el.textContent = section.emptyMessage;
                    treeEl.appendChild(el);
                    appended += 1;
                }
                continue;
            }

            if (appended > 0) appendDivider(treeEl);

            renderedSectionIds.add(section.id);
            const sectionPath = section.id;
            validPaths.add(sectionPath);
            const sectionFolder = buildFolder({
                name: section.label,
                count: result.total,
                iconKind: section.iconKind,
                startExpanded: true,
                path: sectionPath,
                forceExpanded: isFiltered,
                childrenBuilder: (children) => {
                    const ctx = makeSectionCtx(children, sectionPath, isFiltered, validPaths);
                    ctx.data = result;
                    ctx.query = q;
                    ctx.isFiltered = isFiltered;
                    section.build(ctx);
                },
            });
            treeEl.appendChild(sectionFolder);
            appended += 1;
        }
    }

    // Phase 3: prune pathStates + apply pins. Always runs so pins never leak
    // across renders. Prune is scoped to keys whose owning section actually
    // rendered — keys belonging to a section that was gated empty by the
    // current filter survive untouched, so the user's expansions return when
    // they clear the filter.
    const pinned = new Set(pinnedPaths);
    pinnedPaths.clear();
    for (const key of [...pathStates.keys()]) {
        const slashIdx = key.indexOf("/");
        const sectionId = slashIdx === -1 ? key : key.slice(0, slashIdx);
        if (!renderedSectionIds.has(sectionId)) continue;
        if (!validPaths.has(key) && !pinned.has(key)) {
            pathStates.delete(key);
        }
    }
    for (const p of pinned) pathStates.set(p, true);
}

// Section ctx — handed to `section.build(ctx)`. Each `ctx.folder({...})` call
// recurses with a fresh sub-ctx whose path prefix is extended, so nested
// folders only need to provide segment-relative paths.
function makeSectionCtx(parentEl, prefix, isFiltered, validPaths) {
    return {
        folder({ name, count, iconKind, path, startExpanded = false, onContextMenu, build, draggablePayload, dropTarget }) {
            const fullPath = prefix ? `${prefix}/${path}` : path;
            validPaths.add(fullPath);
            const folder = buildFolder({
                name, count, iconKind, startExpanded,
                path: fullPath,
                forceExpanded: isFiltered,
                onContextMenu,
                draggablePayload,
                dropTarget,
                childrenBuilder: (children) => {
                    const sub = makeSectionCtx(children, fullPath, isFiltered, validPaths);
                    build(sub);
                },
            });
            parentEl.appendChild(folder);
        },
        leaf({ row }) {
            parentEl.appendChild(row);
        },
    };
}

function appendDivider(treeEl) {
    const d = document.createElement("div");
    d.className = "koolook-tree-divider";
    treeEl.appendChild(d);
}

// =============================================================================
// Drag-and-drop primitives. Workflow leaf rows + directory folder rows are
// draggable; directory folder rows + the synthetic Archive folder accept
// drops. The MIME type "application/x-koolook-row" ensures we only react to
// our own rows (not files / external drags).
//
// Drop semantics dispatched by handleDndDrop:
//   workflow → directory  → moveWorkflow (no-op if same dir)
//   workflow → Archive    → archive (move to dest dir first if cross-dir)
//   directory → directory → moveDirectory (cycle-checked in store)
// =============================================================================
const DND_MIME = "application/x-koolook-row";

function decorateDraggable(row, payload) {
    row.draggable = true;
    row.addEventListener("dragstart", (e) => {
        e.stopPropagation();
        e.dataTransfer.effectAllowed = "move";
        e.dataTransfer.setData(DND_MIME, JSON.stringify(payload));
    });
}

function decorateDropTarget(row, target) {
    row.addEventListener("dragover", (e) => {
        if (!Array.from(e.dataTransfer.types).includes(DND_MIME)) return;
        e.preventDefault();
        e.stopPropagation();
        e.dataTransfer.dropEffect = "move";
        row.classList.add("koolook-drop-target");
    });
    row.addEventListener("dragleave", () => {
        row.classList.remove("koolook-drop-target");
    });
    row.addEventListener("drop", (e) => {
        row.classList.remove("koolook-drop-target");
        if (!Array.from(e.dataTransfer.types).includes(DND_MIME)) return;
        e.preventDefault();
        e.stopPropagation();
        const raw = e.dataTransfer.getData(DND_MIME);
        if (!raw) return;
        let payload;
        try { payload = JSON.parse(raw); }
        catch { return; }
        handleDndDrop(payload, target);
    });
}

function handleDndDrop(payload, target) {
    if (!payload || !target) return;
    if (payload.type === "workflow") {
        const srcPath = payload.path;
        const wfName = payload.name;
        if (target.kind === "dir") {
            // No-op when dropped onto its own directory.
            if (pathsEqual(srcPath, target.path)) return;
            persistMutation({
                mutate: () => moveWorkflow(srcPath, wfName, target.path),
                onSuccess: () => toast(`Moved "${wfName}" to ${target.path.join(" / ")}.`),
                onNoOp: () => toast(`Could not move (name conflict in destination?).`),
            });
            return;
        }
        if (target.kind === "archive") {
            // Drop on Archive folder of dir target.path → archive in that
            // dir. If the workflow lives in a different dir, move it first.
            const sameDir = pathsEqual(srcPath, target.path);
            persistMutation({
                mutate: () => {
                    if (!sameDir) {
                        if (!moveWorkflow(srcPath, wfName, target.path)) return false;
                    }
                    return archiveWorkflow(target.path, wfName);
                },
                onSuccess: () => {
                    const where = sameDir ? "" : ` in ${target.path.join(" / ")}`;
                    toast(`Archived "${wfName}"${where}.`);
                },
                onNoOp: () => toast(`Could not archive (name conflict or workflow missing).`),
            });
            return;
        }
        return;
    }
    if (payload.type === "directory") {
        if (target.kind !== "dir") return; // dirs don't drop onto Archive
        const srcParentPath = payload.parentPath;
        const dirName = payload.name;
        // Same-parent drop is a no-op.
        if (pathsEqual(srcParentPath, target.path)) return;
        // Self-drop (drop a dir onto its own row) is also a no-op. The
        // same-parent guard above misses this for root-level dirs because
        // the dir's own row carries `target.path = [...srcParentPath, name]`,
        // which doesn't equal `srcParentPath`. Without this check the
        // cycle guard in `moveDirectory` would still reject the move, but
        // the user would see a misleading "cycle, name collision, or
        // invalid target" toast for what's just a no-op gesture.
        if (pathsEqual([...srcParentPath, dirName], target.path)) return;
        persistMutation({
            mutate: () => moveDirectory(srcParentPath, dirName, target.path),
            onSuccess: () => {
                const where = target.path.length === 0 ? "(root)" : target.path.join(" / ");
                toast(`Moved directory "${dirName}" into ${where}.`);
            },
            onNoOp: () => toast(`Could not move directory (cycle, name collision, or invalid target).`),
        });
    }
}

// =============================================================================
// Data gathering — nodes
// =============================================================================
function subcategoryFor(category, categoryRoot) {
    if (!category) return "(uncategorized)";
    if (category === categoryRoot) return "(root)";
    if (categoryRoot && category.startsWith(categoryRoot + "/")) {
        return category.slice(categoryRoot.length + 1);
    }
    // Fallback: the node's category neither equals the repo's `categoryRoot`
    // nor lives under it. Reachable only via a logic error — typically a
    // `REPOS` entry whose `categoryRoot` no longer matches the upstream
    // node's actual category prefix (upstream renamed; repo config is
    // stale). Surface the misconfig in the console so it's debuggable
    // instead of producing silently mis-grouped sidebar entries.
    console.warn(
        `[Koolook] subcategoryFor fallback fired: category="${category}" did not match ` +
        `categoryRoot="${categoryRoot}". The repo config may need updating to track an ` +
        `upstream category rename.`
    );
    const parts = category.split("/");
    return parts.length > 1 ? parts.slice(1).join("/") : category;
}

// Given a node ID, compute the (packLabel, sub) pair the gather code would
// place it under in the sidebar. Mirrors the precedence in
// `gatherNodesByRepo` + `gatherUserPickPacks`: registered REPOS first
// (`select="all"` matches by category prefix; `select=Array` matches by
// explicit ID), then the user-picks fallback derives the pack label from
// the first segment of the node's category.
//
// Returns `{packLabel, sub}` on success, `null` if the node isn't in the
// LiteGraph registry (caller should skip silently — the node won't render
// in the sidebar either way). Used by `spotlightAddedPicks` to know which
// folder to expand after a save.
function findPackPathForType(typeName) {
    const registry = (typeof LiteGraph !== "undefined" && LiteGraph.registered_node_types) || {};
    const nc = registry[typeName];
    if (!nc) return null;
    const cat = (nc && nc.category) || "";

    for (const repo of REPOS) {
        if (repo.select === "all") {
            const root = repo.categoryRoot || "";
            if (root && (cat === root || cat.startsWith(root + "/"))) {
                return { packLabel: repo.label, sub: subcategoryFor(cat, root) };
            }
        } else if (Array.isArray(repo.select)) {
            if (repo.select.includes(typeName)) {
                return { packLabel: repo.label, sub: subcategoryFor(cat, repo.categoryRoot || "") };
            }
        }
    }

    const packLabel = cat.split("/")[0] || "(uncategorized)";
    return { packLabel, sub: subcategoryFor(cat, packLabel) };
}

function matchesQuery(display, type, q) {
    if (!q) return true;
    return display.toLowerCase().includes(q) || type.toLowerCase().includes(q);
}

function gatherNodesByRepo(query) {
    const q = (query || "").trim().toLowerCase();
    const registry = (typeof LiteGraph !== "undefined" && LiteGraph.registered_node_types) || {};
    const out = [];

    for (const repo of REPOS) {
        let candidateIds;
        if (repo.select === "all") {
            const root = repo.categoryRoot || "";
            candidateIds = Object.entries(registry)
                .filter(([, nc]) => {
                    const cat = (nc && nc.category) || "";
                    return root && (cat === root || cat.startsWith(root + "/"));
                })
                .map(([type]) => type);
        } else if (Array.isArray(repo.select)) {
            candidateIds = repo.select.filter(t => registry[t] !== undefined);
        } else {
            candidateIds = [];
        }

        const subcats = new Map();
        let total = 0;
        for (const type of candidateIds) {
            const nc = registry[type];
            const display = (nc && nc.title) || type;
            if (repo.excludePatterns && repo.excludePatterns.some(re => re.test(display))) continue;
            if (!matchesQuery(display, type, q)) continue;

            const sub = subcategoryFor((nc && nc.category) || "", repo.categoryRoot);
            if (!subcats.has(sub)) subcats.set(sub, []);
            subcats.get(sub).push({ type, display });
            total += 1;
        }

        const categories = [...subcats.entries()]
            .map(([name, nodes]) => ({
                name,
                nodes: nodes.sort((a, b) => compareNames(a.display, b.display)),
            }))
            .sort((a, b) => compareNames(a.name, b.name));

        out.push({ label: repo.label, categories, total });
    }
    return out;
}

function gatherUserPickPacks(query) {
    const q = (query || "").trim().toLowerCase();
    const registry = (typeof LiteGraph !== "undefined" && LiteGraph.registered_node_types) || {};
    const picks = loadUserPicks();

    const autoCategoryRoots = new Set(
        REPOS.filter(r => r.select === "all" && r.categoryRoot).map(r => r.categoryRoot)
    );

    const byPack = new Map();
    for (const type of picks) {
        const nc = registry[type];
        if (!nc) continue;
        const cat = nc.category || "";
        const packLabel = cat.split("/")[0] || "(uncategorized)";
        if (autoCategoryRoots.has(packLabel)) continue;

        const display = nc.title || type;
        if (!matchesQuery(display, type, q)) continue;

        const sub = subcategoryFor(cat, packLabel);
        if (!byPack.has(packLabel)) byPack.set(packLabel, new Map());
        const subcatMap = byPack.get(packLabel);
        if (!subcatMap.has(sub)) subcatMap.set(sub, []);
        subcatMap.get(sub).push({ type, display });
    }

    const result = [];
    for (const [packLabel, subcatMap] of byPack.entries()) {
        const categories = [...subcatMap.entries()]
            .map(([name, nodes]) => ({
                name,
                nodes: nodes.sort((a, b) => compareNames(a.display, b.display)),
            }))
            .sort((a, b) => compareNames(a.name, b.name));
        const total = categories.reduce((acc, c) => acc + c.nodes.length, 0);
        result.push({ label: packLabel, categories, total, isUserPicks: true });
    }
    result.sort((a, b) => compareNames(a.label, b.label));
    return result;
}

// =============================================================================
// Data gathering — theme-first mode
//
// Groups every user pick by a single semantic theme — independent of which
// pack the node ships from. Picks-only (no REPOS-driven auto-pulled nodes)
// because the user's mental model is "show me my image-related favorites in
// one place," not "show me every image-related node ComfyUI knows about."
//
// Theme extraction (`extractTheme`):
//   1. Read the node's CATEGORY string from its registered class.
//   2. Look up the source pack via `findPackPathForType` (the same
//      precedence the toolbar `+` button uses for spotlight).
//   3. If the FIRST CATEGORY segment matches the pack label (after
//      canonical-key normalization — case + whitespace + underscore +
//      hyphen folded), strip it. The next segment is the theme.
//   4. Otherwise the first segment is already the theme.
//   5. Canonicalize the theme key for cross-pack merging so
//      `KJNodes/image/...`, `Koolook/Image/...`, and `image/...` all
//      land in the same bucket regardless of casing.
//
// Output shape (flat — no nested folders within a theme):
//   Map<canonicalThemeKey, {
//     displayLabels: Map<originalCasing, count>,   // for label resolution
//     nodes: [{ type, display, isUserPick, unresolved? }],
//   }>
// =============================================================================

// Canonical key for a single category-path segment. Lowercase + strip
// internal whitespace/underscores/hyphens so `style_model`, `StyleModel`,
// and `style model` all collapse to `stylemodel`. Segments are not
// stemmed (singular vs plural keep distinct) — see issue #73 for the
// deliberate "no Levenshtein" tradeoff.
function canonicalSegment(s) {
    return String(s).toLowerCase().replace(/[\s_\-]+/g, "");
}

// Resolves a folder's display label from the count map of original
// casings seen. Picks the most common; ties → first-seen. Map iteration
// is insertion order, which mirrors the gather walk order — stable
// across renders for a given registry state. Empty maps are only valid
// for synthetic buckets where the caller pre-seeds the label.
function pickDisplayLabel(displayCounts) {
    let best = null;
    let bestCount = -1;
    for (const [name, count] of displayCounts) {
        if (count > bestCount) {
            best = name;
            bestCount = count;
        }
    }
    return best;
}

// =============================================================================
// Theme classifier — broad semantic buckets via keyword substring match.
//
// The strict canonical-segment approach (`image` ≠ `images` ≠ `Image-Save`)
// produces too many tiny buckets — "image", "images", "HQ-Image-Save",
// "rgthree/Image Comparer", "upscaling/Upscale Image By" all fragment apart
// even though the user mentally lumps them all under "Image."
//
// This classifier folds anything that mentions an image-related word
// (image / img / pixel / color / upscale / resize / etc.) into a single
// "Image" bucket regardless of which pack or subcategory the node lives
// under. First classifier whose pattern matches the search corpus wins.
//
// Corpus = pack name + full category path + node display title + node ID,
// lower-cased. Searching all four fields catches:
//   • "HQ-Image-Save" (pack name has "image")
//   • "rgthree/Image Comparer" (display title has "image")
//   • "EasyAIPipeline" (type ID has "pipeline")
//
// Keyword choices favor breadth — adjust this list when a future pick
// lands in the wrong bucket. Patterns deeper in the list act as fallbacks
// for picks the broader earlier patterns missed.
// =============================================================================
const THEME_CLASSIFIERS = [
    {
        key: "image",
        label: "Image",
        patterns: [
            /\bimg\b/i, /image/i, /pixel/i,
            /color/i, /\bocio\b/i, /\bexr\b/i, /\bhdr\b/i,
            /upscal/i, /resize/i, /comparer/i, /\btile\b/i,
            /frame/i, /thumbnail/i,
        ],
    },
    {
        key: "video",
        label: "Video",
        patterns: [/video/i, /\bwan\b/i, /hunyuan/i, /cogvideo/i, /\bltx\b/i, /motion/i, /sequence/i],
    },
    {
        key: "mask",
        label: "Mask",
        patterns: [/\bmask/i, /\balpha\b/i, /segment/i, /matte/i, /rmbg/i],
    },
    {
        key: "audio",
        label: "Audio",
        patterns: [/audio/i, /\bwav\b/i, /\bsound\b/i],
    },
    {
        key: "model",
        label: "Model",
        patterns: [
            /model/i, /checkpoint/i, /\blora\b/i, /\bvae\b/i, /\bclip\b/i,
            /controlnet/i, /ipadapter/i, /diffusion/i, /\bunet\b/i,
        ],
    },
    {
        key: "sampler",
        label: "Sampler",
        patterns: [/sampler/i, /scheduler/i, /\bnoise\b/i, /\bsigma\b/i],
    },
    {
        key: "conditioning",
        label: "Conditioning",
        patterns: [/condition/i, /\bembed/i, /tokeniz/i],
    },
    {
        key: "text",
        label: "Text & Prompt",
        patterns: [/\btext\b/i, /prompt/i, /\bstring\b/i, /caption/i],
    },
    {
        key: "math",
        label: "Math & Logic",
        patterns: [
            /\bmath\b/i, /\blogic\b/i, /\bint\b/i, /\bfloat\b/i, /\bbool/i,
            /switch/i, /\bcompare\b/i, /numeric/i,
        ],
    },
    {
        key: "pipeline",
        label: "Pipeline & Batch",
        patterns: [/pipeline/i, /workflow/i, /\bbatch\b/i, /\bcache\b/i],
    },
    {
        key: "io",
        label: "I/O",
        patterns: [/\bload\b/i, /\bsave\b/i, /\bfile\b/i, /\bpath\b/i, /import/i, /export/i],
    },
    {
        key: "camera",
        label: "Camera",
        patterns: [/camera/i, /\blens\b/i, /\bpose\b/i, /\btrack/i],
    },
    {
        key: "utility",
        label: "Utility",
        // `\bget\b` / `\bset\b` are kept short on purpose — they're the
        // last-resort net for trivial getter/setter nodes that didn't hit
        // any earlier classifier. Do NOT generalize them to `/get/` or
        // `/set/` (no boundaries) without re-walking the upstream pattern
        // list — un-anchored variants would fish picks out of node names
        // like "AssetGet…" that should land in their own theme.
        patterns: [/\butil/i, /helper/i, /\btool\b/i, /\bget\b/i, /\bset\b/i, /reroute/i],
    },
];

// Returns `{key, label}` of the first classifier whose patterns match the
// corpus, or `null` if no classifier matched. Walked in order — broader
// upstream classifiers preempt downstream ones. "Image Comparer (rgthree)"
// lands in Image because the corpus contains the word "image", which the
// Image classifier's `/image/i` pattern matches; ordering would also have
// protected it (Math is below Image), but the substring hit comes first.
function classifyTheme(corpus) {
    for (const cls of THEME_CLASSIFIERS) {
        for (const pat of cls.patterns) {
            if (pat.test(corpus)) return { key: cls.key, label: cls.label };
        }
    }
    return null;
}

// Per-pick: classify into one of the broad themes above, falling back to
// the pack-prefix-stripped first CATEGORY segment when no classifier
// matched. Returns one of:
//   { kind: "unresolved" }           — pack not loaded, no registry entry
//   { kind: "uncategorized" }        — pack loaded but CATEGORY empty/missing
//   { kind: "theme", themeRaw, themeCanonical }
//
// The fallback path keeps long-tail picks (a niche pack with no recognizable
// keywords) in their original-segment bucket rather than collapsing every
// unrecognized thing into one "(other)" pile.
function extractTheme(typeName, registry) {
    const nc = registry[typeName];
    if (!nc) return { kind: "unresolved" };
    const cat = nc.category || "";
    const display = nc.title || typeName;
    const loc = findPackPathForType(typeName);
    const packLabel = loc?.packLabel || "";

    // Search corpus combines every signal we have about this node so a
    // keyword in any of them — pack name, category subtree, display title,
    // raw type ID — counts. No `.toLowerCase()` needed: every classifier
    // pattern carries the `/i` flag.
    const corpus = `${packLabel} ${cat} ${display} ${typeName}`;
    const classified = classifyTheme(corpus);
    if (classified) {
        return {
            kind: "theme",
            themeRaw: classified.label,
            themeCanonical: classified.key,
        };
    }

    // Fallback: original pack-prefix-stripped first-segment behavior, so a
    // niche pick from "MyPack/specialFoo/SomeNode" lands in "specialFoo"
    // rather than dumping everything unmatched into "(other)".
    const segs = cat.split("/").map(s => s.trim()).filter(Boolean);
    if (segs.length === 0) return { kind: "uncategorized" };
    let themeIdx = 0;
    if (loc && loc.packLabel && segs.length > 1) {
        if (canonicalSegment(loc.packLabel) === canonicalSegment(segs[0])) {
            themeIdx = 1;
        }
    }
    const themeRaw = segs[themeIdx];
    if (!themeRaw) return { kind: "uncategorized" };
    return {
        kind: "theme",
        themeRaw,
        themeCanonical: canonicalSegment(themeRaw),
    };
}

function gatherNodesByTheme(query) {
    const q = (query || "").trim().toLowerCase();
    const registry = (typeof LiteGraph !== "undefined" && LiteGraph.registered_node_types) || {};
    const userPicks = loadUserPicks();

    // Map<canonicalThemeKey, { displayLabels: Map<rawLabel, count>, nodes: [...] }>
    const themes = new Map();
    const ensureBucket = (key, seedLabel) => {
        if (!themes.has(key)) {
            themes.set(key, {
                displayLabels: seedLabel ? new Map([[seedLabel, 1]]) : new Map(),
                nodes: [],
            });
        }
        return themes.get(key);
    };

    for (const type of userPicks) {
        const result = extractTheme(type, registry);

        if (result.kind === "unresolved") {
            // Pack not loaded — surface the pick under (unresolved) so the
            // user can see what they picked even before installing the pack.
            // Display falls back to the bare type since `nc.title` is unreachable.
            if (!matchesQuery(type, type, q)) continue;
            const bucket = ensureBucket("(unresolved)", "(unresolved)");
            bucket.nodes.push({ type, display: type, isUserPick: true, unresolved: true });
            continue;
        }

        // `nc` is guaranteed non-null here: extractTheme() above already
        // bailed with `kind: "unresolved"` for any type missing from the
        // registry, and we caught that branch before reaching this line.
        const nc = registry[type];
        const display = nc.title || type;
        if (!matchesQuery(display, type, q)) continue;

        if (result.kind === "uncategorized") {
            const bucket = ensureBucket("(uncategorized)", "(uncategorized)");
            bucket.nodes.push({ type, display, isUserPick: true });
            continue;
        }

        const bucket = ensureBucket(result.themeCanonical);
        bucket.displayLabels.set(
            result.themeRaw,
            (bucket.displayLabels.get(result.themeRaw) || 0) + 1,
        );
        bucket.nodes.push({ type, display, isUserPick: true });
    }

    return themes;
}

// Total-leaf count across every theme bucket. Drives the section header's
// count badge so the user sees "N picks" regardless of theme distribution.
function countNodesInThemes(themes) {
    let c = 0;
    for (const bucket of themes.values()) c += bucket.nodes.length;
    return c;
}

// =============================================================================
// Group mode — read/write the user's chosen mode for the Nodes section.
// Two values: "repo" (default) and "category". Anything else falls back to
// the default so a stray localStorage value can't lock the panel into a
// broken state.
// =============================================================================
function loadGroupMode() {
    try {
        const v = localStorage.getItem(GROUP_MODE_KEY);
        return v === "category" || v === "repo" ? v : GROUP_MODE_DEFAULT;
    } catch {
        return GROUP_MODE_DEFAULT;
    }
}

function saveGroupMode(mode) {
    if (mode !== "repo" && mode !== "category") return;
    try { localStorage.setItem(GROUP_MODE_KEY, mode); } catch { /* quota / disabled */ }
}

// =============================================================================
// Data gathering — workflows (recursive)
//
// Each output entry has shape:
//   { name, path, active, archived, subdirs }
// where `path` is the full string[] address and `subdirs` is the recursive
// list of child entries with the same shape. Empty subtrees (no matching
// active, archived, or subdirs) are dropped, so a search-narrowed render
// only includes paths that produced at least one visible row.
// =============================================================================
function gatherWorkflows(query) {
    const q = (query || "").trim().toLowerCase();
    const stats = { total: 0 };
    const directories = gatherDirsAt([], q, stats);
    return { directories, total: stats.total };
}

function gatherDirsAt(parentPath, q, stats) {
    const out = [];
    for (const dirName of listDirectoryNames(parentPath)) {
        const dirPath = [...parentPath, dirName];
        const dir = dirOf(dirPath);
        if (!dir) continue;

        const matches = (n) => !q || n.toLowerCase().includes(q) || dirName.toLowerCase().includes(q);

        const allNames = Object.keys(dir.workflows || {});
        const active = [];
        const archived = [];
        for (const n of allNames) {
            if (!matches(n)) continue;
            if (dir.workflows[n] && dir.workflows[n].archived) archived.push(n);
            else active.push(n);
        }
        active.sort(compareNames);
        archived.sort(compareNames);

        const subdirs = gatherDirsAt(dirPath, q, stats);

        // Under an active search, drop entries that contributed nothing
        // (no matched workflows of their own, no matched descendants) so
        // the filtered tree only shows paths leading to a hit. With NO
        // search, keep every directory that exists — including freshly-
        // created empty subdirs — so the tree mirrors the actual store.
        if (q && active.length === 0 && archived.length === 0 && subdirs.length === 0) continue;

        out.push({ name: dirName, path: dirPath, active, archived, subdirs });
        stats.total += active.length + archived.length;
    }
    return out;
}

// Recursive count of every workflow (active + archived) under a gathered
// directory entry, including all descendants. Used for the dir's header
// count so a parent dir with subdirs but no direct workflows doesn't show
// "0" while clearly hosting things underneath.
function countWorkflowsInGatheredDir(dir) {
    let count = dir.active.length + dir.archived.length;
    for (const sub of dir.subdirs) count += countWorkflowsInGatheredDir(sub);
    return count;
}

// Recursive counts on a raw DirNode (not the gathered shape) for delete-
// confirmation messages. Returns { workflows, subdirs } totals.
function countDescendantsOfRawDir(dir) {
    if (!dir) return { workflows: 0, subdirs: 0 };
    let workflows = Object.keys(dir.workflows || {}).length;
    let subdirs = 0;
    for (const child of Object.values(dir.directories || {})) {
        const sub = countDescendantsOfRawDir(child);
        workflows += sub.workflows;
        subdirs += 1 + sub.subdirs;
    }
    return { workflows, subdirs };
}

// =============================================================================
// Data gathering — tags (DFS walk over the workflow tree).
//
// Each output entry has shape:
//   { name, entries: [{ wfName, path }, ...] }
// `path` is the full directory string[] address of the workflow's home dir.
// Archived workflows are skipped here so the Tags section reflects the
// active set only — their tags are still persisted on the entry, so a
// restore (via the Archive folder's right-click) brings them back into
// the section. Tag grouping is case-sensitive: "AI" and "ai" produce
// separate tag folders (the dedupe in `normalizeDirNode` is also
// case-sensitive). The search filter is case-insensitive substring match
// against tag name OR workflow name — typing a tag narrows to that group,
// typing a workflow name keeps every tag the workflow carries.
// =============================================================================
function gatherTags(query) {
    const q = (query || "").trim().toLowerCase();
    const tagMap = new Map();

    const walk = (parentPath) => {
        for (const dirName of listDirectoryNames(parentPath)) {
            const dirPath = [...parentPath, dirName];
            const dir = dirOf(dirPath);
            if (!dir) continue;
            for (const [wfName, wf] of Object.entries(dir.workflows || {})) {
                if (wf.archived === true) continue;
                const tags = Array.isArray(wf.tags) ? wf.tags : [];
                if (tags.length === 0) continue;
                for (const tag of tags) {
                    if (q && !tag.toLowerCase().includes(q) && !wfName.toLowerCase().includes(q)) continue;
                    if (!tagMap.has(tag)) tagMap.set(tag, []);
                    tagMap.get(tag).push({ wfName, path: dirPath });
                }
            }
            walk(dirPath);
        }
    };
    walk([]);

    const tags = [...tagMap.entries()]
        .map(([name, entries]) => ({
            name,
            entries: entries.sort((a, b) => compareNames(a.wfName, b.wfName)),
        }))
        .sort((a, b) => compareNames(a.name, b.name));
    const total = tags.reduce((acc, t) => acc + t.entries.length, 0);
    return { tags, total };
}

// =============================================================================
// DOM helpers
// =============================================================================

// Toolbar button factory — replaces the four near-identical button-construction
// blocks in `renderPanel` for icon-only buttons (export, new-dir, save-canvas,
// save-selection). The "+" Add-to-favorites button (`addBtn`) is text-content
// only and stays hand-rolled; this factory does not cover that case. Every
// covered button shares the `koolook-add-btn koolook-icon-btn` pair (both
// styling rules; the disabled state on `:disabled` lives on `koolook-icon-btn`).
// `iconClass` is the PrimeIcons class (e.g. `pi pi-download`).
//
// The icon span is built with `createElement` + `className` rather than
// `innerHTML = …${iconClass}…` so a future caller passing a non-static
// `iconClass` (e.g. data sourced from a workflow or registry) can't inject
// markup. All current callers pass static literals — this is helper-boundary
// hardening, not a fix for an active vulnerability.
function makeToolbarButton({ iconClass, title, onClick }) {
    const btn = document.createElement("button");
    btn.className = "koolook-add-btn koolook-icon-btn";
    const icon = document.createElement("span");
    icon.className = iconClass;
    btn.appendChild(icon);
    btn.title = title;
    btn.addEventListener("click", onClick);
    return btn;
}

function makeFolderRow({ name, count, iconKind, onToggle, onContextMenu, draggablePayload, dropTarget }) {
    const row = document.createElement("div");
    row.className = "koolook-row";

    const chevron = document.createElement("span");
    chevron.className = "koolook-chevron";
    chevron.textContent = "▾";
    row.appendChild(chevron);

    const icon = document.createElement("span");
    if (iconKind === "favorites") {
        icon.className = "pi pi-star koolook-pin-icon";
    } else if (iconKind === "workflows") {
        icon.className = "pi pi-th-large koolook-workflows-icon";
    } else if (iconKind === "archive") {
        icon.className = "pi pi-box koolook-archive-icon";
    } else {
        icon.className = "pi pi-folder koolook-folder-icon";
    }
    row.appendChild(icon);

    const nameEl = document.createElement("span");
    nameEl.className = "koolook-name";
    nameEl.textContent = name;
    row.appendChild(nameEl);

    if (count != null) {
        const cnt = document.createElement("span");
        cnt.className = "koolook-count";
        cnt.textContent = String(count);
        row.appendChild(cnt);
    }

    row.addEventListener("click", onToggle);
    if (onContextMenu) row.addEventListener("contextmenu", onContextMenu);
    if (draggablePayload) decorateDraggable(row, draggablePayload);
    if (dropTarget) decorateDropTarget(row, dropTarget);
    return { row, chevron };
}

function makeNodeLeafRow({ display, type, removable, onClick, unresolved, breadcrumb }) {
    const row = document.createElement("div");
    row.className = "koolook-row koolook-leaf";
    if (unresolved) row.classList.add("koolook-leaf-unresolved");
    // Breadcrumb-equipped rows under search-flatten mode also surface the
    // origin path in the row tooltip (`type` is still appended) so a hover
    // confirms exactly which folder the leaf would live under in the tree.
    row.title = breadcrumb ? `${breadcrumb} › ${type}` : type;

    const chevron = document.createElement("span");
    chevron.className = "koolook-chevron";
    row.appendChild(chevron);

    const dot = document.createElement("span");
    dot.className = "koolook-leaf-dot";
    row.appendChild(dot);

    // The name cell holds an optional dim-grey breadcrumb prefix (set under
    // search-flatten mode) followed by the node's display name. We avoid
    // innerHTML and build text nodes instead — display strings come from
    // `nc.title` and breadcrumbs from category labels, both of which can in
    // principle contain user-controlled HTML if a custom-node author
    // included markup in `CATEGORY` or class attributes.
    const nameEl = document.createElement("span");
    nameEl.className = "koolook-name";
    if (breadcrumb) {
        const crumb = document.createElement("span");
        crumb.className = "koolook-leaf-crumb";
        crumb.textContent = `${breadcrumb} › `;
        nameEl.appendChild(crumb);
        nameEl.appendChild(document.createTextNode(display));
    } else {
        nameEl.textContent = display;
    }
    row.appendChild(nameEl);

    if (removable) {
        const rm = document.createElement("span");
        rm.className = "koolook-remove";
        rm.textContent = "×";
        rm.title = "Remove from favorites";
        rm.addEventListener("click", (e) => {
            e.stopPropagation();
            removeFromMyPicks(type);
            notifyPicksChanged();
        });
        row.appendChild(rm);
    }

    row.addEventListener("click", onClick);
    // Hover preview — every leaf gets one. The card looks up the live
    // node class on hover (lazy), so leaves whose pack isn't loaded
    // still get a "Pack not loaded" stub instead of silently no-oping.
    attachHoverPreview(row, type);
    return row;
}

function makeWorkflowLeafRow({ name, dirName, onClick, onContextMenu, draggablePayload, isModule = false }) {
    const row = document.createElement("div");
    row.className = "koolook-row koolook-leaf";
    // Module-tagged entries get a distinct hover tooltip so the green icon
    // isn't the only signal that a left-click does something different. The
    // `pi pi-plus-circle` class swap is what actually makes them look like
    // "drop in" affordances rather than "open" affordances.
    row.title = isModule
        ? `${dirName} / ${name} — module (left-click to insert into canvas)`
        : `${dirName} / ${name}`;

    const chevron = document.createElement("span");
    chevron.className = "koolook-chevron";
    row.appendChild(chevron);

    const icon = document.createElement("span");
    icon.className = isModule
        ? "pi pi-plus-circle koolook-module-icon"
        : "pi pi-file koolook-leaf-icon";
    row.appendChild(icon);

    const nameEl = document.createElement("span");
    nameEl.className = "koolook-name";
    nameEl.textContent = name;
    row.appendChild(nameEl);

    row.addEventListener("click", onClick);
    if (onContextMenu) row.addEventListener("contextmenu", onContextMenu);
    if (draggablePayload) decorateDraggable(row, draggablePayload);
    return row;
}

// Resolve module-ness from the workflow's stored tags. Computed at render
// time (cheap — small array `.includes` per row) so we don't have to widen
// the gather pipeline's row shape just to carry one boolean. Returns false
// for archived entries: archived workflows are old versions; clicking them
// in the Archive folder should still mean "load to review", not "splice an
// outdated cluster into my live canvas". The right-click menu on archived
// rows still exposes both Insert and Load if the user really wants to
// insert an archived version.
function isModuleWorkflow(dirPath, wfName) {
    const tags = getWorkflowTags(dirPath, wfName);
    if (!Array.isArray(tags)) return false;
    return tags.includes(MODULE_TAG);
}

function buildFolder({ name, count, iconKind, childrenBuilder, onContextMenu, startExpanded = true, path, forceExpanded = false, draggablePayload, dropTarget }) {
    // `path` is required — every section/sub-section calling site routes
    // through `makeSectionCtx`, which always supplies a non-empty
    // section-prefixed string. The assertion below converts a "future
    // caller forgot to pass path" regression into a loud failure at the
    // boundary; without it, a stray `undefined` would silently land in
    // `pathStates.set(undefined, …)` and pollute one shared key across
    // every misconfigured folder. The pathStates branch further down
    // assumes a string key.
    if (typeof path !== "string" || !path) {
        throw new Error("buildFolder: `path` must be a non-empty string");
    }

    // Resolution order:
    //   1. forceExpanded (e.g. search active) — overrides everything
    //   2. pinnedPaths (a save/spotlight flow has flagged this for force-open
    //      on the current render) — Phase 3 writes pins into pathStates AFTER
    //      Phase 2 builds DOM, so without a direct pinnedPaths read here the
    //      pin would only take effect on the *next* render. The spotlight
    //      effect for newly-added picks depends on pinned paths expanding
    //      on the immediate render.
    //   3. pathStates (user has previously toggled this folder)
    //   4. startExpanded (the natural default for this folder)
    let initiallyExpanded;
    if (forceExpanded) {
        initiallyExpanded = true;
    } else if (pinnedPaths.has(path)) {
        initiallyExpanded = true;
    } else if (pathStates.has(path)) {
        initiallyExpanded = pathStates.get(path);
    } else {
        initiallyExpanded = startExpanded;
    }

    // Track the live expanded state in a closure variable instead of
    // round-tripping through `wrapper.dataset.expanded`. The DOM dataset
    // forces every value to a string, which led to the slightly awkward
    // `!== "false"` read; closure-locality keeps the toggle a plain boolean
    // and removes one source of truth that wasn't the canonical one
    // (`pathStates` is the canonical store; the DOM dataset just mirrored
    // it for reads).
    let isExpanded = initiallyExpanded;
    const wrapper = document.createElement("div");

    const children = document.createElement("div");
    children.className = "koolook-children";
    if (!isExpanded) children.style.display = "none";

    const { row, chevron } = makeFolderRow({
        name,
        count,
        iconKind,
        onContextMenu,
        draggablePayload,
        dropTarget,
        onToggle: () => {
            isExpanded = !isExpanded;
            chevron.textContent = isExpanded ? "▾" : "▸";
            children.style.display = isExpanded ? "" : "none";
            pathStates.set(path, isExpanded);
        },
    });
    if (!initiallyExpanded) chevron.textContent = "▸";

    wrapper.appendChild(row);
    wrapper.appendChild(children);
    childrenBuilder(children);
    return wrapper;
}

// =============================================================================
// Context-menu wiring for workflow rows
// =============================================================================
function workflowRowContextMenu(event, dirPath, wfName, isArchived = false) {
    // The "Move to…" submenu lists every other directory path in the tree
    // so a workflow can be relocated across nesting levels in one click.
    const moveItems = listAllDirectoryPaths()
        .filter(p => !pathsEqual(p, dirPath))
        .map(p => ({
            label: `→ ${p.join(" / ")}`,
            action: () => persistMutation({
                mutate: () => moveWorkflow(dirPath, wfName, p),
                onSuccess: () => toast(`Moved "${wfName}" to ${p.join(" / ")}.`),
                onNoOp: () => toast(`Could not move (name conflict?).`),
            }),
        }));

    // "+ New directory…" / "+ New subdirectory under …" both create a fresh
    // directory and move the workflow into it as one atomic mutation. If
    // the move couldn't land we manually undo the dir add so we don't leak
    // an empty orphan; in practice that branch is unreachable (the new dir
    // is empty so the move can't collide), but the explicit cleanup keeps a
    // future regression from silently leaving stray directories behind.
    function moveToNewDirAction(parentPath, parentLabel) {
        showInputModal({
            title: parentLabel
                ? `New subdirectory under "${parentLabel}"`
                : "New top-level directory",
            label: "Name",
            placeholder: "e.g. drafts",
            confirmLabel: "Create & Move",
            onSubmit: (name) => persistMutation({
                mutate: () => {
                    if (!addDirectory(parentPath, name)) return false;
                    const target = [...parentPath, name];
                    if (!moveWorkflow(dirPath, wfName, target)) {
                        deleteDirectory(parentPath, name);
                        return false;
                    }
                    return target;
                },
                onSuccess: (target) => toast(`Moved "${wfName}" to ${target.join(" / ")}.`),
                onNoOp: () => toast(`Could not create — name in use, empty, or "Archive" reserved.`),
            }),
        });
    }

    const newDirItem = {
        label: "+ New directory…",
        action: () => moveToNewDirAction([], null),
    };
    const newSubdirItem = {
        label: `+ New subdirectory under "${dirPath.join(" / ")}"…`,
        action: () => moveToNewDirAction(dirPath, dirPath.join(" / ")),
    };

    const archiveItem = isArchived
        ? {
            label: "Restore from archive",
            action: () => persistMutation({
                mutate: () => unarchiveWorkflow(dirPath, wfName),
                onSuccess: () => toast(`Restored "${wfName}".`),
            }),
        }
        : {
            label: "Move to archive",
            action: () => persistMutation({
                mutate: () => archiveWorkflow(dirPath, wfName),
                onSuccess: () => toast(`Archived "${wfName}".`),
            }),
        };

    const duplicateItem = {
        label: "Duplicate…",
        action: () => {
            const sourceGraph = getWorkflowGraph(dirPath, wfName);
            if (!sourceGraph) {
                toast("Could not duplicate — workflow not found.");
                return;
            }
            // Capture tags at modal-open time so a same-name duplicate (which
            // archives the original via `saveWorkflowEntry`) still inherits
            // the source's categorization onto the new active entry.
            const sourceTags = getWorkflowTags(dirPath, wfName) || [];
            showInputModal({
                title: "Duplicate workflow",
                label: "New name",
                defaultValue: `${wfName} (copy)`,
                confirmLabel: "Duplicate",
                onSubmit: (newName) => persistMutation({
                    mutate: () => {
                        // Re-validate at submit-time. A concurrent action
                        // (another tab, drag-and-drop) could have deleted
                        // the source between menu-open and submit. Without
                        // this guard, `saveWorkflowEntry` would call
                        // `ensureDirectoryAtPath` and silently re-create
                        // the parent directory with a clone of the
                        // captured graph — effectively undoing someone
                        // else's delete with no feedback.
                        if (getWorkflowGraph(dirPath, wfName) === null) return false;
                        const cloned = JSON.parse(JSON.stringify(sourceGraph));
                        const result = saveWorkflowEntry(dirPath, newName, cloned);
                        if (!result) return false;
                        // Route tag inheritance through the public `addTag`
                        // mutator instead of writing `wf.tags` directly, so
                        // the Duplicate path stays inside the documented
                        // store API. The whole sequence rides on the same
                        // persistMutation snapshot — commit failure rolls
                        // back the save AND the tag adds together.
                        for (const t of sourceTags) addTag(dirPath, newName, t);
                        return result;
                    },
                    onSuccess: (result) => {
                        if (result && result.archivedAs) {
                            toast(`Duplicated to "${newName}" — previous "${newName}" archived as "${result.archivedAs}".`);
                        } else {
                            toast(`Duplicated to "${newName}".`);
                        }
                    },
                    onNoOp: () => toast(`Could not duplicate — "${wfName}" no longer exists.`),
                }),
            });
        },
    };

    const tagsItem = {
        label: "Tags…",
        action: () => {
            showTagsModal({
                wfName,
                // Pass the raw `null` through so the modal can render an
                // explicit gone-state when the workflow is deleted/moved/
                // renamed by a concurrent action. Collapsing to `[]` would
                // silently morph "workflow gone" into "no tags yet" and
                // accept doomed Add inputs.
                getCurrentTags: () => getWorkflowTags(dirPath, wfName),
                onAddTag: (tag, onDone) => persistMutation({
                    mutate: () => addTag(dirPath, wfName, tag),
                    onSuccess: () => { onDone(); toast(`Tagged "${wfName}" with "${tag}".`); },
                    onNoOp: () => toast(`Could not add — empty or already tagged.`),
                }),
                onRemoveTag: (tag, onDone) => persistMutation({
                    mutate: () => removeTag(dirPath, wfName, tag),
                    onSuccess: () => { onDone(); toast(`Removed tag "${tag}" from "${wfName}".`); },
                    onNoOp: () => toast(`Tag "${tag}" was not present.`),
                }),
            });
        },
    };

    showContextMenu(event, [
        {
            label: "Load",
            action: () => loadWorkflowOntoCanvas(dirPath, wfName),
        },
        {
            // Non-destructive sibling of Load — splices the saved cluster into
            // whatever's already on canvas (placed at the viewport center,
            // inserted nodes selected so the user can immediately drag).
            // Aborts cleanly if any referenced node type isn't installed.
            label: "Insert into canvas",
            action: () => insertWorkflowOntoCanvas(dirPath, wfName),
        },
        {
            label: "Rename…",
            action: () => {
                showInputModal({
                    title: "Rename workflow",
                    label: "New name",
                    defaultValue: wfName,
                    confirmLabel: "Rename",
                    onSubmit: (newName) => persistMutation({
                        mutate: () => renameWorkflow(dirPath, wfName, newName),
                        onSuccess: () => toast(`Renamed to "${newName}".`),
                        onNoOp: () => toast(`Rename failed (name in use?).`),
                    }),
                });
            },
        },
        duplicateItem,
        tagsItem,
        archiveItem,
        // Move section — separator first; existing-path entries (when any),
        // then a separator, then the always-available new-dir / new-subdir
        // entries. Skipping the inner separator when there are no existing
        // paths keeps the menu visually clean for a fresh store with one
        // directory.
        null,
        ...moveItems,
        ...(moveItems.length > 0 ? [null] : []),
        newDirItem,
        newSubdirItem,
        null,
        {
            label: "Delete",
            danger: true,
            action: () => {
                showConfirmModal({
                    title: "Delete workflow?",
                    message: `"${wfName}" will be removed. This cannot be undone.`,
                    confirmLabel: "Delete",
                    danger: true,
                    onConfirm: () => persistMutation({
                        mutate: () => deleteWorkflow(dirPath, wfName),
                        onSuccess: () => toast(`Deleted "${wfName}".`),
                    }),
                });
            },
        },
    ]);
}

// Right-click on the synthetic Archive folder. The Archive folder isn't a
// real DirNode — it's rendered for any dir with `archived: true` workflows
// at this level. The only useful bulk op is "delete every archived entry
// here in one go"; per-entry restore/delete still works on each archived
// workflow's own row.
function archiveFolderContextMenu(event, dirPath, archivedCount) {
    const dirDisplay = dirPath.join(" / ");
    const noun = `${archivedCount} archived workflow${archivedCount === 1 ? "" : "s"}`;
    showContextMenu(event, [
        {
            label: `Delete archive (${archivedCount})`,
            danger: true,
            action: () => {
                showConfirmModal({
                    title: "Delete archived workflows?",
                    message: `${noun} in "${dirDisplay}" will be permanently deleted. Active workflows in this directory are not affected.`,
                    confirmLabel: "Delete archive",
                    danger: true,
                    onConfirm: () => persistMutation({
                        mutate: () => clearArchive(dirPath),
                        onSuccess: (result) => toast(
                            `Deleted ${result.count} archived workflow${result.count === 1 ? "" : "s"} in "${dirDisplay}".`
                        ),
                    }),
                });
            },
        },
    ]);
}

function directoryRowContextMenu(event, dirPath) {
    const dir = dirOf(dirPath);
    const dirName = dirPath[dirPath.length - 1];
    const parentPath = dirPath.slice(0, -1);
    const displayPath = dirPath.join(" / ");
    const counts = countDescendantsOfRawDir(dir);
    const isEmpty = counts.workflows === 0 && counts.subdirs === 0;

    showContextMenu(event, [
        {
            label: "Create subdirectory…",
            action: () => {
                showInputModal({
                    title: `Create subdirectory under "${displayPath}"`,
                    label: "Name",
                    placeholder: "e.g. Type-A",
                    confirmLabel: "Create",
                    onSubmit: (name) => persistMutation({
                        mutate: () => addDirectory(dirPath, name),
                        onSuccess: () => toast(`Created "${name}" in ${displayPath}.`),
                        onNoOp: () => toast(`Could not create — name in use, empty, or "Archive" reserved.`),
                    }),
                });
            },
        },
        {
            label: "Rename directory…",
            action: () => {
                showInputModal({
                    title: "Rename directory",
                    label: "New name",
                    defaultValue: dirName,
                    confirmLabel: "Rename",
                    onSubmit: (newName) => persistMutation({
                        mutate: () => renameDirectory(parentPath, dirName, newName),
                        onSuccess: () => toast(`Renamed to "${newName}".`),
                        onNoOp: () => toast(`Rename failed — name in use or "Archive" reserved.`),
                    }),
                });
            },
        },
        null,
        {
            label: "Delete directory",
            danger: true,
            action: () => {
                if (isEmpty) {
                    showConfirmModal({
                        title: "Delete empty directory?",
                        message: `"${displayPath}" will be removed.`,
                        confirmLabel: "Delete",
                        danger: true,
                        onConfirm: () => persistMutation({
                            mutate: () => deleteDirectory(parentPath, dirName),
                            onSuccess: () => toast(`Deleted directory "${displayPath}".`),
                        }),
                    });
                } else {
                    const parts = [];
                    if (counts.workflows > 0) parts.push(`${counts.workflows} workflow${counts.workflows === 1 ? "" : "s"}`);
                    if (counts.subdirs > 0) parts.push(`${counts.subdirs} subdirector${counts.subdirs === 1 ? "y" : "ies"}`);
                    showConfirmModal({
                        title: "Delete non-empty directory?",
                        message: `"${displayPath}" contains ${parts.join(" and ")}. They will be permanently deleted.`,
                        confirmLabel: "Delete all",
                        danger: true,
                        onConfirm: () => persistMutation({
                            mutate: () => deleteDirectory(parentPath, dirName),
                            onSuccess: () => toast(`Deleted "${displayPath}" and its contents.`),
                        }),
                    });
                }
            },
        },
    ]);
}

// =============================================================================
// Built-in section descriptors. Order here = render order in the panel.
// =============================================================================
addSection({
    id: SECTION_ID_NODES,
    label: ROOT_GROUP_LABEL,
    iconKind: "favorites",
    emptyMessage: "No curated nodes yet. Click + above (with a canvas node selected) or right-click a node on the canvas → Add to Kforge Labs.",
    gather(q) {
        const mode = loadGroupMode();
        if (mode === "category") {
            const themes = gatherNodesByTheme(q);
            return { mode, themes, total: countNodesInThemes(themes) };
        }
        const auto = gatherNodesByRepo(q);
        const user = gatherUserPickPacks(q);
        const packs = [...auto, ...user]
            .filter(p => p.total > 0)
            .sort((a, b) => compareNames(a.label, b.label));
        const total = packs.reduce((acc, p) => acc + p.total, 0);
        return { mode: "repo", packs, total };
    },
    build(ctx) {
        // Search-flatten branch: under any non-empty query the Nodes section
        // collapses to a flat list of matching leaves with breadcrumb
        // prefixes, regardless of group mode. The tree's spatial signal is
        // already mostly lost once a query has narrowed the set, so a flat
        // sorted list with breadcrumb prefixes scans faster than a forest
        // of forced-open folder spines. Workflows / Tags sections retain
        // their existing tree-under-filter behavior.
        if (ctx.isFiltered) {
            emitNodesFlatSearchResults(ctx);
            return;
        }
        if (ctx.data.mode === "category") {
            emitThemeChildren(ctx, ctx.data.themes);
            return;
        }
        for (const pack of ctx.data.packs) {
            ctx.folder({
                name: pack.label,
                count: pack.total,
                path: pack.label,
                build: (packCtx) => {
                    for (const cat of pack.categories) {
                        packCtx.folder({
                            name: cat.name,
                            count: cat.nodes.length,
                            path: cat.name,
                            build: (catCtx) => {
                                for (const n of cat.nodes) {
                                    catCtx.leaf({
                                        row: makeNodeLeafRow({
                                            display: n.display,
                                            type: n.type,
                                            removable: !!pack.isUserPicks,
                                            onClick: () => insertNode(n.type),
                                        }),
                                    });
                                }
                            },
                        });
                    }
                },
            });
        }
    },
});

// Flat-list emitters for the Nodes section under search. Both group modes
// produce the same `{ display, type, breadcrumb, removable, unresolved? }`
// shape so the leaf factory call is uniform downstream. Sorted by display
// name (case-insensitive); breadcrumb is a `›`-joined path of the folder
// labels the leaf would live under in the un-filtered tree, so the user
// retains the spatial-origin signal.

function emitNodesFlatSearchResults(ctx) {
    const items = [];
    if (ctx.data.mode === "category") {
        collectFlatFromThemes(ctx.data.themes, items);
    } else {
        collectFlatFromRepoMode(ctx.data.packs, items);
    }
    items.sort((a, b) => compareNames(a.display, b.display));
    for (const item of items) {
        ctx.leaf({
            row: makeNodeLeafRow({
                display: item.display,
                type: item.type,
                breadcrumb: item.breadcrumb,
                unresolved: !!item.unresolved,
                removable: !!item.removable,
                onClick: () => insertNode(item.type),
            }),
        });
    }
}

function collectFlatFromThemes(themes, out) {
    for (const bucket of themes.values()) {
        // Theme label = the most common original casing seen across the
        // picks that landed in this bucket. Falls back to the canonical key
        // for synthetic buckets where `displayLabels` was pre-seeded with
        // the bucket name itself.
        const label = pickDisplayLabel(bucket.displayLabels);
        for (const n of bucket.nodes) {
            out.push({
                display: n.display,
                type: n.type,
                unresolved: !!n.unresolved,
                removable: !!n.isUserPick,
                breadcrumb: label || null,
            });
        }
    }
}

function collectFlatFromRepoMode(packs, out) {
    for (const pack of packs) {
        for (const cat of pack.categories) {
            // Collapse the subcategory out of the breadcrumb when it adds no
            // info: either the synthetic "(root)" label (`subcategoryFor`'s
            // marker for nodes whose category equals the repo's
            // `categoryRoot`) or a duplicate of the pack label itself
            // (happens for user picks under "(uncategorized)" where both
            // labels collapse to the same synthetic token).
            const subRedundant = cat.name === "(root)" || cat.name === pack.label;
            const crumbs = subRedundant ? [pack.label] : [pack.label, cat.name];
            const breadcrumb = crumbs.join(" › ");
            for (const n of cat.nodes) {
                out.push({
                    display: n.display,
                    type: n.type,
                    removable: !!pack.isUserPicks,
                    breadcrumb,
                });
            }
        }
    }
}

// Emits one folder per theme bucket under `parentCtx`. Each theme folder
// holds a flat sorted list of its picks — no further nesting. Folder
// `path` uses the canonical key (not the resolved display label) so the
// user's expansion state in `pathStates` survives a future upstream
// casing change in the most-common label for that theme.
function emitThemeChildren(parentCtx, themes) {
    const themeEntries = [...themes.entries()]
        .map(([canonical, bucket]) => ({
            canonical,
            bucket,
            label: pickDisplayLabel(bucket.displayLabels) || canonical,
        }))
        .sort((a, b) => compareNames(a.label, b.label));

    for (const { canonical, bucket, label } of themeEntries) {
        parentCtx.folder({
            name: label,
            count: bucket.nodes.length,
            path: canonical,
            build: (sub) => {
                const sortedNodes = [...bucket.nodes].sort(
                    (a, b) => compareNames(a.display, b.display),
                );
                for (const n of sortedNodes) {
                    sub.leaf({
                        row: makeNodeLeafRow({
                            display: n.display,
                            type: n.type,
                            removable: !!n.isUserPick,
                            unresolved: !!n.unresolved,
                            onClick: () => insertNode(n.type),
                        }),
                    });
                }
            },
        });
    }
}

addSection({
    id: SECTION_ID_WORKFLOWS,
    label: WORKFLOWS_GROUP_LABEL,
    iconKind: "workflows",
    // No emptyMessage — when the workflow store is empty and no search is
    // active, we silently hide the section. The toolbar above still lets the
    // user create a directory or save a workflow.
    gather(q) {
        return gatherWorkflows(q);
    },
    build(ctx) {
        for (const dir of ctx.data.directories) renderGatheredDir(ctx, dir);
    },
});

// Tags section — flat list of tag folders, each containing the workflows
// carrying that tag. Workflows that share a tag appear under the tag folder
// regardless of which directory they live in. Click loads the workflow from
// its real path; the right-click menu operates on the workflow at its
// canonical location, so renames/moves originating here update the same
// underlying entry the Workflows section shows.
addSection({
    id: SECTION_ID_TAGS,
    label: "Tags",
    iconKind: "folder",
    // No emptyMessage — until any workflow carries a tag, the section stays
    // silent rather than nag the user about an empty feature.
    gather(q) {
        return gatherTags(q);
    },
    build(ctx) {
        for (const tag of ctx.data.tags) {
            ctx.folder({
                name: tag.name,
                count: tag.entries.length,
                path: tag.name,
                build: (tagCtx) => {
                    for (const entry of tag.entries) {
                        // Module-ness is per-workflow, not per-tag-folder —
                        // a workflow tagged `module` is still a module when
                        // it shows up under any other tag it carries.
                        const moduleEntry = isModuleWorkflow(entry.path, entry.wfName);
                        tagCtx.leaf({
                            row: makeWorkflowLeafRow({
                                name: entry.wfName,
                                dirName: entry.path.join(" / "),
                                onClick: () => moduleEntry
                                    ? insertWorkflowOntoCanvas(entry.path, entry.wfName)
                                    : loadWorkflowOntoCanvas(entry.path, entry.wfName),
                                onContextMenu: (e) => workflowRowContextMenu(e, entry.path, entry.wfName, false),
                                isModule: moduleEntry,
                            }),
                        });
                    }
                },
            });
        }
    },
});

// Render one gathered directory entry (and its descendants) under
// `parentCtx`. Layout per directory: subdirectories first (so their own
// trees nest under the parent), then the synthetic Archive folder if any
// archived workflows exist at this level, then active workflows last so
// the freshest version sits closest to the bottom of the directory.
function renderGatheredDir(parentCtx, dir) {
    const dirPath = dir.path;
    const parentPath = dirPath.slice(0, -1);
    const dirDisplay = dirPath.join(" / ");
    const totalInTree = countWorkflowsInGatheredDir(dir);
    parentCtx.folder({
        name: dir.name,
        count: totalInTree,
        path: dir.name,
        onContextMenu: (e) => directoryRowContextMenu(e, dirPath),
        // Directory rows are draggable (drag to nest under another dir) and
        // also drop targets (drop a workflow or another dir onto them).
        draggablePayload: { type: "directory", parentPath, name: dir.name },
        dropTarget: { kind: "dir", path: dirPath },
        build: (dirCtx) => {
            for (const sub of dir.subdirs) renderGatheredDir(dirCtx, sub);
            if (dir.archived.length > 0) {
                dirCtx.folder({
                    name: "Archive",
                    count: dir.archived.length,
                    iconKind: "archive",
                    path: "Archive",
                    onContextMenu: (e) => archiveFolderContextMenu(e, dirPath, dir.archived.length),
                    // Archive folders accept workflow drops only (drop = archive
                    // in this directory; cross-dir drops move + archive).
                    dropTarget: { kind: "archive", path: dirPath },
                    build: (archCtx) => {
                        for (const wfName of dir.archived) {
                            archCtx.leaf({
                                row: makeWorkflowLeafRow({
                                    name: wfName,
                                    dirName: dirDisplay,
                                    onClick: () => loadWorkflowOntoCanvas(dirPath, wfName),
                                    onContextMenu: (e) => workflowRowContextMenu(e, dirPath, wfName, true),
                                    draggablePayload: { type: "workflow", path: dirPath, name: wfName },
                                }),
                            });
                        }
                    },
                });
            }
            for (const wfName of dir.active) {
                const moduleEntry = isModuleWorkflow(dirPath, wfName);
                dirCtx.leaf({
                    row: makeWorkflowLeafRow({
                        name: wfName,
                        dirName: dirDisplay,
                        // Module convention: left-click inserts; non-module
                        // entries left-click loads (replaces canvas, with
                        // confirm modal as before). Both actions are always
                        // available via the right-click menu regardless.
                        onClick: () => moduleEntry
                            ? insertWorkflowOntoCanvas(dirPath, wfName)
                            : loadWorkflowOntoCanvas(dirPath, wfName),
                        onContextMenu: (e) => workflowRowContextMenu(e, dirPath, wfName, false),
                        draggablePayload: { type: "workflow", path: dirPath, name: wfName },
                        isModule: moduleEntry,
                    }),
                });
            }
        },
    });
}

// =============================================================================
// Panel rendering
// =============================================================================
export function renderPanel(container) {
    ensureStyle();
    container.innerHTML = "";
    container.classList.add("koolook-sidebar");

    // Forward declarations — `tree` and `search` are created several rows
    // below, but the Nodes-row mode toggle (built mid-renderPanel) needs to
    // re-render the tree from its click handler. JS closures bind by
    // reference, so the handler resolves both at click time, by which point
    // the assignments below have run. Without these `let` placeholders the
    // toggle would have to be appended out-of-order.
    let tree;
    let search;
    const rerenderTree = () => {
        if (tree) renderTree({ treeEl: tree, query: search ? search.value : "" });
    };

    // Three small wrappers over the snapshot dialogs. Each opens directly
    // — no tabbed parent modal — so the user goes from toolbar click to
    // the right form in one step. Capability functions are passed in
    // explicitly to keep `modals.js` UI-only (no `snapshot.js` import
    // crossing the layer).
    const openSaveDialog = () => showSaveSnapshotDialog({
        getCurrentPresetName,
        setCurrentPresetName,
        presetExists,
        writePreset,
        gatherSnapshot,
        sanitizeName,
        getLibraryInfo,
        markStateSaved,
        onToast: toast,
    });
    const openLoadDialog = () => showLoadSnapshotDialog({
        listPresets,
        readPreset,
        deletePreset,
        applySnapshot,
        setCurrentPresetName,
        getCurrentPresetName,
        getLibraryInfo,
        writePreLoadAutosave,
        markStateSaved,
        listAutosaves,
        onToast: toast,
    });
    const openSettingsDialog = () => showSnapshotSettingsDialog({
        getSettings,
        saveSettings,
        onToast: toast,
    });

    // ---- Row 0: Snapshot library action bar (top-level kit ops) ----
    // Mirrors the Nodes/Workflows action-row pattern but lives above the
    // search field — these are infrequent meta-actions (save/load the
    // whole Kforge Labs state as a named preset) and shouldn't be
    // mixed into the per-section toolbars.
    const snapshotRow = document.createElement("div");
    snapshotRow.className = "koolook-actions-row";

    // Snapshot status indicator (replaces the static "Snapshot" label).
    // Layout:  [● <preset name>  · <state>]   where:
    //   • the dot's color encodes status (saved=green, autosaved=blue,
    //     unsaved=orange, none=grey)
    //   • the name is the user's last-loaded/saved preset, ellipsised if
    //     long; "(no snapshot)" italic when nothing is tracked
    //   • the state is a small textual qualifier ("saved" / "auto-saved" /
    //     "unsaved")
    // Tooltip on the whole row carries the long-form explanation including
    // the latest auto-save timestamp when relevant.
    const snapshotStatus = document.createElement("div");
    snapshotStatus.className = "koolook-snap-status";
    const snapshotDot = document.createElement("span");
    snapshotDot.className = "koolook-snap-status-dot koolook-snap-status-none";
    snapshotStatus.appendChild(snapshotDot);
    const snapshotName = document.createElement("span");
    snapshotName.className = "koolook-snap-status-name";
    snapshotStatus.appendChild(snapshotName);
    const snapshotState = document.createElement("span");
    snapshotState.className = "koolook-snap-status-state";
    snapshotStatus.appendChild(snapshotState);
    snapshotRow.appendChild(snapshotStatus);

    function refreshSnapshotStatus() {
        const status = getSnapshotStatus();
        snapshotDot.className = "koolook-snap-status-dot koolook-snap-status-" + status.state;
        if (status.name) {
            snapshotName.classList.remove("koolook-snap-status-name-empty");
            snapshotName.textContent = status.name;
        } else {
            snapshotName.classList.add("koolook-snap-status-name-empty");
            snapshotName.textContent = "(no snapshot)";
        }
        let stateText = "";
        let title = "";
        const since = status.lastAutosaveAt
            ? ` (latest auto-save: ${status.lastAutosaveAt})`
            : "";
        switch (status.state) {
            case "saved":
                stateText = "· saved";
                title = `Live state matches the last named save "${status.name}"${since}.`;
                break;
            case "autosaved":
                stateText = "· auto-saved";
                title = status.name
                    ? `Live state matches the latest periodic auto-save (named save "${status.name}" has diverged)${since}.`
                    : `Live state matches the latest periodic auto-save${since}.`;
                break;
            case "unsaved":
                stateText = "· unsaved";
                title = `Live state has changes since the last save / auto-save of "${status.name}"${since}. Use Save to persist.`;
                break;
            case "none":
            default:
                stateText = "";
                title = `No named snapshot tracked${since}. Use Save to create one.`;
                break;
        }
        snapshotState.textContent = stateText;
        snapshotStatus.title = title;
    }
    refreshSnapshotStatus();
    window.addEventListener(PICKS_CHANGED_EVENT, refreshSnapshotStatus);
    window.addEventListener(WORKFLOWS_CHANGED_EVENT, refreshSnapshotStatus);
    window.addEventListener(SNAPSHOT_STATUS_CHANGED_EVENT, refreshSnapshotStatus);

    snapshotRow.appendChild(makeToolbarButton({
        iconClass: "pi pi-cloud-download",
        title: "Load a saved snapshot (replaces current state)",
        onClick: openLoadDialog,
    }));
    // Quick Save — single-click overwrite of the currently-tracked preset
    // with no dialog. Disabled when no preset is tracked (then the regular
    // Save button is the entry point because it prompts for a new name).
    // Distinct icon from Save so the two are scannable: Save = cloud-upload,
    // Quick Save = floppy-disk save icon.
    const quickSaveBtn = makeToolbarButton({
        iconClass: "pi pi-save",
        title: "Quick Save — overwrite the currently loaded preset (no dialog)",
        onClick: async () => {
            const current = getCurrentPresetName();
            if (!current) {
                toast("No preset loaded — use Save to name a new one.");
                return;
            }
            try {
                const snap = gatherSnapshot(current);
                await writePreset(current, snap);
                markStateSaved();
                toast(`Quick-saved "${current}".`);
            } catch (e) {
                console.error("[Koolook] quick save failed:", e);
                toast(`Could not Quick Save: ${e.message}`);
            }
        },
    });
    function refreshQuickSaveDisabled() {
        quickSaveBtn.disabled = !getCurrentPresetName();
    }
    refreshQuickSaveDisabled();
    window.addEventListener(SNAPSHOT_STATUS_CHANGED_EVENT, refreshQuickSaveDisabled);
    snapshotRow.appendChild(quickSaveBtn);
    snapshotRow.appendChild(makeToolbarButton({
        iconClass: "pi pi-cloud-upload",
        title: "Save current state — overwrites the last-loaded preset, or prompts for a name",
        onClick: openSaveDialog,
    }));
    snapshotRow.appendChild(makeToolbarButton({
        iconClass: "pi pi-cog",
        title: "Snapshot library settings (where presets are saved on disk)",
        onClick: openSettingsDialog,
    }));

    container.appendChild(snapshotRow);

    // ---- Row 0b: Tools (admin/advanced — not for daily use) ----
    // Sits between the Snapshot row and the search field. Holds power-user
    // utilities a typical user shouldn't trigger by accident: exporting the
    // curated_defaults.json (maintainer flow), the Manager-driven install of
    // missing packs (only useful on a fresh install or after wiping nodes),
    // and the canvas-handoff fallback for `security_level=normal` users
    // (drops placeholder nodes so Manager's own "Install Missing Custom
    // Nodes" UI takes over). Keeping these out of the per-section toolbars
    // declutters the everyday Add/Save flow.
    const dividerAfterSnapshot = document.createElement("div");
    dividerAfterSnapshot.className = "koolook-tree-divider";
    container.appendChild(dividerAfterSnapshot);

    const toolsRow = document.createElement("div");
    toolsRow.className = "koolook-actions-row";

    const toolsLabel = document.createElement("span");
    toolsLabel.className = "koolook-actions-label";
    toolsLabel.textContent = "Tools";
    toolsRow.appendChild(toolsLabel);

    toolsRow.appendChild(makeToolbarButton({
        iconClass: "pi pi-download",
        title: "Export current state as starter_preset.json (copies snapshot JSON to clipboard — paste into web/starter_preset.json to ship as the next release's starter)",
        onClick: exportStarterPreset,
    }));

    // "Install missing for picks" — always enabled. The modal itself decides
    // whether anything actually needs installing (after probing Manager and
    // walking picks against the live LiteGraph registry), so leaving the
    // button enabled lets the user trigger the discovery flow whenever they
    // want a status check, including the "everything's already installed"
    // confirmation. Click handlers don't read picks here — the modal pulls
    // a fresh `loadUserPicks()` snapshot at open-time.
    toolsRow.appendChild(makeToolbarButton({
        iconClass: "pi pi-cloud-download",
        title: "Install missing custom nodes for current picks (via ComfyUI-Manager)",
        onClick: () => showInstallMissingModal({ picks: loadUserPicks() }),
    }));

    // "Drop missing onto canvas" — the security_level=normal escape hatch.
    // Discovery runs the same detect+map+resolve pipeline as the install
    // modal, then for each missing pack we instantiate one placeholder node
    // (the pack's first mapped node ID) on a fresh canvas. ComfyUI renders
    // unknown types as red error nodes, which Manager's "Install Missing
    // Custom Nodes" feature scans for — so the user gets a one-click
    // handoff to Manager's UI-driven install path that doesn't go through
    // /customnode/install/git_url's security gate.
    toolsRow.appendChild(makeToolbarButton({
        iconClass: "pi pi-th-large",
        title: "Drop placeholders for missing packs onto canvas — then use Manager's \"Install Missing Custom Nodes\" (works at security_level=normal)",
        onClick: async () => {
            const picks = loadUserPicks();
            if (!picks || picks.length === 0) {
                toast("No picks to check. Add some favorites first.");
                return;
            }
            const discovery = await discoverMissingPacks(picks);
            if (!discovery.ok) {
                if (discovery.reason === "manager-unreachable") {
                    toast("ComfyUI-Manager isn't reachable — can't resolve picks to packs.");
                } else {
                    toast(`Could not load Manager's mapping database: ${discovery.error?.message || discovery.reason}.`);
                }
                return;
            }
            const byUrl = discovery.result.willInstall.byUrl;
            if (byUrl.size === 0) {
                toast("Nothing missing — every pick is installed (or unmapped).");
                return;
            }
            await dropPlaceholdersForPacks(byUrl);
        },
    }));

    container.appendChild(toolsRow);

    const dividerBeforeSearch = document.createElement("div");
    dividerBeforeSearch.className = "koolook-tree-divider";
    container.appendChild(dividerBeforeSearch);

    // ---- Row 1: Search ----
    const searchRow = document.createElement("div");
    searchRow.className = "koolook-search-row";

    const searchWrap = document.createElement("div");
    searchWrap.className = "koolook-search-wrap";

    const searchIcon = document.createElement("span");
    searchIcon.className = "pi pi-search koolook-search-icon";
    searchWrap.appendChild(searchIcon);

    search = document.createElement("input");
    search.type = "search";
    search.className = "koolook-search";
    search.placeholder = "Search nodes & workflows...";
    searchWrap.appendChild(search);

    searchRow.appendChild(searchWrap);
    container.appendChild(searchRow);

    // Divider between the search field and the Nodes action bar — gives the
    // search field its own visual zone instead of jamming it against the
    // first action row beneath it.
    const dividerAfterSearch = document.createElement("div");
    dividerAfterSearch.className = "koolook-tree-divider";
    container.appendChild(dividerAfterSearch);

    // ---- Row 2: Nodes action bar ----
    const nodesRow = document.createElement("div");
    nodesRow.className = "koolook-actions-row";

    const nodesLabel = document.createElement("span");
    nodesLabel.className = "koolook-actions-label";
    nodesLabel.textContent = "Nodes";
    nodesRow.appendChild(nodesLabel);

    // Segmented mode toggle: [📦 by-pack] [🌐 by-category]. The active mode
    // is highlighted; click switches and re-renders. The choice persists in
    // localStorage (`GROUP_MODE_KEY`) and survives reloads. We listen for
    // storage events further down so a second tab updates in lockstep.
    const modeToggle = document.createElement("div");
    modeToggle.className = "koolook-mode-toggle";

    function makeModeBtn(modeId, iconClass, title) {
        const btn = document.createElement("button");
        btn.className = "koolook-mode-toggle-btn";
        btn.title = title;
        btn.dataset.mode = modeId;
        const icon = document.createElement("span");
        icon.className = iconClass;
        btn.appendChild(icon);
        btn.addEventListener("click", () => {
            if (loadGroupMode() === modeId) return;
            saveGroupMode(modeId);
            refreshModeToggle();
            rerenderTree();
        });
        return btn;
    }

    const repoModeBtn = makeModeBtn(
        "repo",
        "pi pi-sitemap",
        "Group by pack — each repo's nodes under their original category subtree.",
    );
    const categoryModeBtn = makeModeBtn(
        "category",
        "pi pi-bars",
        "Group by theme — picks regrouped by category theme (e.g. all image nodes together) regardless of source pack.",
    );

    function refreshModeToggle() {
        const m = loadGroupMode();
        repoModeBtn.classList.toggle("koolook-mode-active", m === "repo");
        categoryModeBtn.classList.toggle("koolook-mode-active", m === "category");
    }
    refreshModeToggle();

    modeToggle.appendChild(repoModeBtn);
    modeToggle.appendChild(categoryModeBtn);
    nodesRow.appendChild(modeToggle);

    const addBtn = document.createElement("button");
    // Same height as the icon-bar buttons (toolbar pi-* glyphs) — uses an
    // explicit `pi pi-plus` icon at the same font-size rather than a 16px "+"
    // text glyph that sat taller than the rest of the row. The dusty-green
    // accent class flags it as the primary action in the Nodes row.
    addBtn.className = "koolook-add-btn koolook-icon-btn koolook-add-btn-green";
    const addBtnIcon = document.createElement("i");
    addBtnIcon.className = "pi pi-plus";
    addBtn.appendChild(addBtnIcon);
    addBtn.title = "Add the selected canvas node(s) to favorites";
    addBtn.addEventListener("click", () => {
        const types = getSelectedNodeTypes();
        if (types.length === 0) {
            toast("Select a node on the canvas first.");
            return;
        }
        let added = 0;
        let duplicates = 0;
        let failed = 0;
        const successfulTypes = [];
        for (const t of types) {
            const status = addToMyPicks(t);
            if (status === "added") added += 1;
            else if (status === "duplicate") duplicates += 1;
            else failed += 1;
            // Spotlight even on duplicates — the user clicked + with this node
            // selected; reminding them where it lives is the whole feature.
            if (status !== "failed") successfulTypes.push(t);
        }
        spotlightAddedPicks(successfulTypes);
        if (added > 0) {
            const noun = added === 1 ? "node" : "nodes";
            toast(`Added ${added} ${noun} to favorites.`);
            notifyPicksChanged();
        } else if (duplicates > 0 && failed === 0) {
            toast("Already in favorites.");
            // Re-render anyway so the spotlight effect lands even on pure-
            // duplicate clicks. notifyPicksChanged is the agreed-upon trigger
            // for any sidebar refresh, even when picks didn't actually change.
            notifyPicksChanged();
        }
        if (failed > 0) {
            toast(`Failed to save ${failed} pick${failed === 1 ? "" : "s"}. See console.`);
        }
    });
    nodesRow.appendChild(addBtn);

    // Daily-use Nodes row keeps only the Add (+) button — Export and
    // Install-missing have been promoted up to the dedicated Tools row above
    // the search field (they're admin/advanced operations the typical user
    // shouldn't trigger from the everyday toolbar).

    container.appendChild(nodesRow);

    // Divider between the two action bars
    const dividerBetweenBars = document.createElement("div");
    dividerBetweenBars.className = "koolook-tree-divider";
    container.appendChild(dividerBetweenBars);

    // ---- Row 3: Workflows action bar ----
    const wfRow = document.createElement("div");
    wfRow.className = "koolook-actions-row";

    const wfLabel = document.createElement("span");
    wfLabel.className = "koolook-actions-label";
    wfLabel.textContent = "Workflows";
    wfRow.appendChild(wfLabel);

    wfRow.appendChild(makeToolbarButton({
        iconClass: "pi pi-folder-open",
        title: "Create new workflow directory",
        onClick: () => {
            showInputModal({
                title: "Create directory",
                label: "Directory name",
                placeholder: "e.g. Inpainting",
                confirmLabel: "Create",
                onSubmit: (name) => persistMutation({
                    // Top-level directory creation. Subdirectories are created
                    // via right-click → "Create subdirectory…" on an existing
                    // folder.
                    mutate: () => addDirectory([], name),
                    onSuccess: () => toast(`Directory "${name}" created.`),
                    onNoOp: () => toast(`Directory "${name}" already exists.`),
                }),
            });
        },
    }));

    // Shared save handler for both buttons. Uses persistMutation so the cache
    // mutation rolls back automatically if both /userdata and the localStorage
    // fallback rejected the write — the user isn't told a save succeeded when
    // nothing was persisted, and the in-memory cache stays consistent with disk.
    //
    // `asModule` rides the same persistMutation so a commit failure rolls
    // back BOTH the entry write and the tag add. Doing the addTag in a
    // separate persistMutation would race: the save could land while the
    // tag rejected, leaving a "module" entry the user thinks is tagged
    // but isn't.
    const saveAndToast = async (graph, name, dirPath, asModule = false) => {
        // Pin the workflows section + every ancestor of the destination dir
        // so the resulting re-render leaves the whole path open. Pins are
        // consumed on the next render.
        const pinKeys = [SECTION_ID_WORKFLOWS];
        let cur = SECTION_ID_WORKFLOWS;
        for (const seg of dirPath) {
            cur = `${cur}/${seg}`;
            pinKeys.push(cur);
        }
        pinExpanded(pinKeys);
        const dirDisplay = dirPath.join(" / ");
        await persistMutation({
            mutate: () => {
                const result = saveWorkflowEntry(dirPath, name, graph);
                if (!result) return false;
                if (asModule) addTag(dirPath, name, MODULE_TAG);
                return result;
            },
            onSuccess: (result) => {
                const moduleSuffix = asModule ? " as module" : "";
                if (result.archivedAs) {
                    toast(`Saved "${name}" in ${dirDisplay}${moduleSuffix}. Previous version moved to Archive.`);
                } else {
                    toast(`Saved "${name}" in ${dirDisplay}${moduleSuffix}.`);
                }
            },
            persistFailedMessage: `Save failed — could not write "${name}". See console.`,
        });
    };

    wfRow.appendChild(makeToolbarButton({
        iconClass: "pi pi-save",
        title: "Save entire canvas as a workflow",
        onClick: () => {
            // Check emptiness first so a serialize() throw doesn't get
            // misreported as "Canvas is empty" — distinct toasts for distinct
            // failure modes.
            if (!canvasIsNonEmpty()) {
                toast("Canvas is empty.");
                return;
            }
            const graph = serializeFullCanvas();
            if (!graph || !graph.nodes || graph.nodes.length === 0) {
                toast("Failed to serialize canvas. See console.");
                return;
            }
            showSaveWorkflowModal({
                titleSuffix: "entire canvas",
                // Whole-canvas saves default to NOT being modules — the
                // typical "save my whole workflow" intent is to keep the
                // load-replaces-canvas semantics. The checkbox is still
                // visible so a power user can opt in.
                defaultModule: false,
                onSave: ({ name, dirPath, asModule }) => saveAndToast(graph, name, dirPath, asModule),
            });
        },
    }));

    wfRow.appendChild(makeToolbarButton({
        iconClass: "pi pi-objects-column",
        title: "Save current selection as a workflow",
        onClick: () => {
            const result = serializeSelection();
            if (result.kind === "empty") {
                toast("Select one or more nodes on the canvas first.");
                return;
            }
            if (result.kind === "stale") {
                // Selection set is non-empty but every id in it points at a
                // deleted node — typically after an undo or a delete-then-
                // change-tab sequence. Generic "select one or more" was
                // misleading because the user *had* selected something; the
                // canvas just doesn't have it anymore.
                toast("Selected node(s) no longer exist. Click a node on the canvas to re-select.");
                return;
            }
            const { graph } = result;
            showSaveWorkflowModal({
                titleSuffix: `${graph.nodes.length} selected node${graph.nodes.length === 1 ? "" : "s"}`,
                // Selection saves are the canonical "module" candidate —
                // small clusters the user wants to splice into other
                // workflows. Pre-check the box; the user can untick if
                // they'd rather have replace-on-load semantics.
                defaultModule: true,
                onSave: ({ name, dirPath, asModule }) => saveAndToast(graph, name, dirPath, asModule),
            });
        },
    }));

    container.appendChild(wfRow);

    // Divider between the workflows action bar and the list below
    const dividerBeforeTree = document.createElement("div");
    dividerBeforeTree.className = "koolook-tree-divider";
    container.appendChild(dividerBeforeTree);

    // ---- Tree (scrollable) ----
    tree = document.createElement("div");
    tree.className = "koolook-tree";
    container.appendChild(tree);

    renderTree({ treeEl: tree, query: "" });

    // ---- Search wiring ----
    let debounce = null;
    search.addEventListener("input", (e) => {
        const q = e.target.value;
        if (debounce) clearTimeout(debounce);
        debounce = setTimeout(() => renderTree({ treeEl: tree, query: q }), 60);
    });

    // ---- Event subscriptions for live re-rendering ----
    window.addEventListener(PICKS_CHANGED_EVENT, () => renderTree({ treeEl: tree, query: search.value }));
    window.addEventListener(WORKFLOWS_CHANGED_EVENT, () => renderTree({ treeEl: tree, query: search.value }));
    window.addEventListener("storage", (e) => {
        if (e.key === STORAGE_KEY || e.key === WORKFLOWS_FALLBACK_KEY || e.key === GROUP_MODE_KEY) {
            renderTree({ treeEl: tree, query: search.value });
        }
    });
}
