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
    ensureStyle,
    toast,
    compareNames,
} from "./constants.js";
import {
    loadUserPicks,
    removeFromMyPicks,
    notifyPicksChanged,
    addToMyPicks,
    exportPicks,
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
    insertNode,
    getSelectedNodeTypes,
} from "./canvas_io.js";
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
 * Single render entry point. Idempotent — safe to call from search input,
 * picks-changed, workflows-changed, or storage events.
 *
 * @param {{ treeEl: HTMLElement, query: string }} opts
 */
function renderTree({ treeEl, query }) {
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

function makeNodeLeafRow({ display, type, removable, onClick }) {
    const row = document.createElement("div");
    row.className = "koolook-row koolook-leaf";
    row.title = type;

    const chevron = document.createElement("span");
    chevron.className = "koolook-chevron";
    row.appendChild(chevron);

    const dot = document.createElement("span");
    dot.className = "koolook-leaf-dot";
    row.appendChild(dot);

    const nameEl = document.createElement("span");
    nameEl.className = "koolook-name";
    nameEl.textContent = display;
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
    return row;
}

function makeWorkflowLeafRow({ name, dirName, onClick, onContextMenu, draggablePayload }) {
    const row = document.createElement("div");
    row.className = "koolook-row koolook-leaf";
    row.title = `${dirName} / ${name}`;

    const chevron = document.createElement("span");
    chevron.className = "koolook-chevron";
    row.appendChild(chevron);

    const icon = document.createElement("span");
    icon.className = "pi pi-file koolook-leaf-icon";
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
    //   2. pathStates (user has previously toggled this folder)
    //   3. startExpanded (the natural default for this folder)
    let initiallyExpanded;
    if (forceExpanded) {
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
        const auto = gatherNodesByRepo(q);
        const user = gatherUserPickPacks(q);
        const packs = [...auto, ...user]
            .filter(p => p.total > 0)
            .sort((a, b) => compareNames(a.label, b.label));
        const total = packs.reduce((acc, p) => acc + p.total, 0);
        return { packs, total };
    },
    build(ctx) {
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
                        tagCtx.leaf({
                            row: makeWorkflowLeafRow({
                                name: entry.wfName,
                                dirName: entry.path.join(" / "),
                                onClick: () => loadWorkflowOntoCanvas(entry.path, entry.wfName),
                                onContextMenu: (e) => workflowRowContextMenu(e, entry.path, entry.wfName, false),
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
                dirCtx.leaf({
                    row: makeWorkflowLeafRow({
                        name: wfName,
                        dirName: dirDisplay,
                        onClick: () => loadWorkflowOntoCanvas(dirPath, wfName),
                        onContextMenu: (e) => workflowRowContextMenu(e, dirPath, wfName, false),
                        draggablePayload: { type: "workflow", path: dirPath, name: wfName },
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
        onToast: toast,
    });
    const openLoadDialog = () => showLoadSnapshotDialog({
        listPresets,
        readPreset,
        deletePreset,
        applySnapshot,
        setCurrentPresetName,
        getLibraryInfo,
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

    const snapshotLabel = document.createElement("span");
    snapshotLabel.className = "koolook-actions-label";
    snapshotLabel.textContent = "Snapshot";
    snapshotRow.appendChild(snapshotLabel);

    snapshotRow.appendChild(makeToolbarButton({
        iconClass: "pi pi-cloud-download",
        title: "Load a saved snapshot (replaces current state)",
        onClick: openLoadDialog,
    }));
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

    const search = document.createElement("input");
    search.type = "search";
    search.className = "koolook-search";
    search.placeholder = "Search nodes & workflows...";
    searchWrap.appendChild(search);

    searchRow.appendChild(searchWrap);
    container.appendChild(searchRow);

    // ---- Row 2: Nodes action bar ----
    const nodesRow = document.createElement("div");
    nodesRow.className = "koolook-actions-row";

    const nodesLabel = document.createElement("span");
    nodesLabel.className = "koolook-actions-label";
    nodesLabel.textContent = "Nodes";
    nodesRow.appendChild(nodesLabel);

    const addBtn = document.createElement("button");
    addBtn.className = "koolook-add-btn";
    addBtn.textContent = "+";
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
        for (const t of types) {
            const status = addToMyPicks(t);
            if (status === "added") added += 1;
            else if (status === "duplicate") duplicates += 1;
            else failed += 1;
        }
        if (added > 0) {
            const noun = added === 1 ? "node" : "nodes";
            toast(`Added ${added} ${noun} to favorites.`);
            notifyPicksChanged();
        }
        if (failed > 0) {
            toast(`Failed to save ${failed} pick${failed === 1 ? "" : "s"}. See console.`);
        } else if (added === 0 && duplicates > 0) {
            toast("Already in favorites.");
        }
    });
    nodesRow.appendChild(addBtn);

    nodesRow.appendChild(makeToolbarButton({
        iconClass: "pi pi-download",
        title: "Export current picks as curated_defaults.json (copies JSON to clipboard)",
        onClick: exportPicks,
    }));

    // "Install missing for picks" — always enabled. The modal itself decides
    // whether anything actually needs installing (after probing Manager and
    // walking picks against the live LiteGraph registry), so leaving the
    // button enabled lets the user trigger the discovery flow whenever they
    // want a status check, including the "everything's already installed"
    // confirmation. Click handlers don't read picks here — the modal pulls
    // a fresh `loadUserPicks()` snapshot at open-time.
    nodesRow.appendChild(makeToolbarButton({
        iconClass: "pi pi-cloud-download",
        title: "Install missing custom nodes for current picks (via ComfyUI-Manager)",
        onClick: () => showInstallMissingModal({ picks: loadUserPicks() }),
    }));

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
    const saveAndToast = async (graph, name, dirPath) => {
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
            mutate: () => saveWorkflowEntry(dirPath, name, graph),
            onSuccess: (result) => {
                if (result.archivedAs) {
                    toast(`Saved "${name}" in ${dirDisplay}. Previous version moved to Archive.`);
                } else {
                    toast(`Saved "${name}" in ${dirDisplay}.`);
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
                onSave: ({ name, dirPath }) => saveAndToast(graph, name, dirPath),
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
                onSave: ({ name, dirPath }) => saveAndToast(graph, name, dirPath),
            });
        },
    }));

    container.appendChild(wfRow);

    // Divider between the workflows action bar and the list below
    const dividerBeforeTree = document.createElement("div");
    dividerBeforeTree.className = "koolook-tree-divider";
    container.appendChild(dividerBeforeTree);

    // ---- Tree (scrollable) ----
    const tree = document.createElement("div");
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
        if (e.key === STORAGE_KEY || e.key === WORKFLOWS_FALLBACK_KEY) {
            renderTree({ treeEl: tree, query: search.value });
        }
    });
}
