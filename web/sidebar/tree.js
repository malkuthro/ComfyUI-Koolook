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
} from "./modals.js";

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
// DOM helpers
// =============================================================================
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

function buildFolder({ name, count, iconKind, childrenBuilder, onContextMenu, startExpanded = true, path = null, forceExpanded = false, draggablePayload, dropTarget }) {
    // Resolution order:
    //   1. forceExpanded (e.g. search active) — overrides everything
    //   2. pathStates (user has previously toggled this folder)
    //   3. startExpanded (the natural default for this folder)
    let initiallyExpanded;
    if (forceExpanded) {
        initiallyExpanded = true;
    } else if (path && pathStates.has(path)) {
        initiallyExpanded = pathStates.get(path);
    } else {
        initiallyExpanded = startExpanded;
    }

    const wrapper = document.createElement("div");
    wrapper.dataset.expanded = initiallyExpanded ? "true" : "false";

    const children = document.createElement("div");
    children.className = "koolook-children";
    if (!initiallyExpanded) children.style.display = "none";

    const { row, chevron } = makeFolderRow({
        name,
        count,
        iconKind,
        onContextMenu,
        draggablePayload,
        dropTarget,
        onToggle: () => {
            const expanded = wrapper.dataset.expanded !== "false";
            const next = !expanded;
            wrapper.dataset.expanded = next ? "true" : "false";
            chevron.textContent = next ? "▾" : "▸";
            children.style.display = next ? "" : "none";
            if (path) pathStates.set(path, next);
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
        archiveItem,
        ...(moveItems.length > 0 ? [null, ...moveItems] : []),
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
    id: "nodes",
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
    id: "workflows",
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

    const exportBtn = document.createElement("button");
    exportBtn.className = "koolook-add-btn koolook-export-btn";
    exportBtn.innerHTML = '<span class="pi pi-download"></span>';
    exportBtn.title = "Export current picks as curated_defaults.json (copies JSON to clipboard)";
    exportBtn.addEventListener("click", exportPicks);
    nodesRow.appendChild(exportBtn);

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

    const newDirBtn = document.createElement("button");
    newDirBtn.className = "koolook-add-btn koolook-icon-btn";
    newDirBtn.innerHTML = '<span class="pi pi-folder-open"></span>';
    newDirBtn.title = "Create new workflow directory";
    newDirBtn.addEventListener("click", () => {
        showInputModal({
            title: "Create directory",
            label: "Directory name",
            placeholder: "e.g. Inpainting",
            confirmLabel: "Create",
            onSubmit: (name) => persistMutation({
                // Top-level directory creation. Subdirectories are created via
                // right-click → "Create subdirectory…" on an existing folder.
                mutate: () => addDirectory([], name),
                onSuccess: () => toast(`Directory "${name}" created.`),
                onNoOp: () => toast(`Directory "${name}" already exists.`),
            }),
        });
    });
    wfRow.appendChild(newDirBtn);

    // Shared save handler for both buttons. Uses persistMutation so the cache
    // mutation rolls back automatically if both /userdata and the localStorage
    // fallback rejected the write — the user isn't told a save succeeded when
    // nothing was persisted, and the in-memory cache stays consistent with disk.
    const saveAndToast = async (graph, name, dirPath) => {
        // Pin the workflows section + every ancestor of the destination dir
        // so the resulting re-render leaves the whole path open. Pins are
        // consumed on the next render.
        const pinKeys = ["workflows"];
        let cur = "workflows";
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

    const saveCanvasBtn = document.createElement("button");
    saveCanvasBtn.className = "koolook-add-btn koolook-icon-btn";
    saveCanvasBtn.innerHTML = '<span class="pi pi-save"></span>';
    saveCanvasBtn.title = "Save entire canvas as a workflow";
    saveCanvasBtn.addEventListener("click", () => {
        // Check emptiness first so a serialize() throw doesn't get misreported
        // as "Canvas is empty" — distinct toasts for distinct failure modes.
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
            onSave: ({ name, dir }) => saveAndToast(graph, name, dir),
        });
    });
    wfRow.appendChild(saveCanvasBtn);

    const saveSelectionBtn = document.createElement("button");
    saveSelectionBtn.className = "koolook-add-btn koolook-icon-btn";
    saveSelectionBtn.innerHTML = '<span class="pi pi-objects-column"></span>';
    saveSelectionBtn.title = "Save current selection as a workflow";
    saveSelectionBtn.addEventListener("click", () => {
        const graph = serializeSelection();
        if (!graph) {
            toast("Select one or more nodes on the canvas first.");
            return;
        }
        showSaveWorkflowModal({
            titleSuffix: `${graph.nodes.length} selected node${graph.nodes.length === 1 ? "" : "s"}`,
            onSave: ({ name, dir }) => saveAndToast(graph, name, dir),
        });
    });
    wfRow.appendChild(saveSelectionBtn);

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
