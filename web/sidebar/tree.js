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
        folder({ name, count, iconKind, path, startExpanded = false, onContextMenu, build }) {
            const fullPath = prefix ? `${prefix}/${path}` : path;
            validPaths.add(fullPath);
            const folder = buildFolder({
                name, count, iconKind, startExpanded,
                path: fullPath,
                forceExpanded: isFiltered,
                onContextMenu,
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
// Data gathering — workflows
// =============================================================================
function gatherWorkflows(query) {
    const q = (query || "").trim().toLowerCase();
    const out = [];
    let total = 0;

    for (const dirName of listDirectoryNames()) {
        const dir = dirOf(dirName);
        const allNames = Object.keys(dir.workflows || {});
        const matches = (n) => !q || n.toLowerCase().includes(q) || dirName.toLowerCase().includes(q);

        const active = [];
        const archived = [];
        for (const n of allNames) {
            if (!matches(n)) continue;
            if (dir.workflows[n] && dir.workflows[n].archived) archived.push(n);
            else active.push(n);
        }
        active.sort(compareNames);
        archived.sort(compareNames);
        if (active.length === 0 && archived.length === 0) continue;
        out.push({ name: dirName, active, archived });
        total += active.length + archived.length;
    }

    return { directories: out, total };
}

// =============================================================================
// DOM helpers
// =============================================================================
function makeFolderRow({ name, count, iconKind, onToggle, onContextMenu }) {
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

function makeWorkflowLeafRow({ name, dirName, onClick, onContextMenu }) {
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
    return row;
}

function buildFolder({ name, count, iconKind, childrenBuilder, onContextMenu, startExpanded = true, path = null, forceExpanded = false }) {
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
function workflowRowContextMenu(event, dirName, wfName, isArchived = false) {
    const dirNames = listDirectoryNames();
    const moveItems = dirNames
        .filter(d => d !== dirName)
        .map(d => ({
            label: `→ ${d}`,
            action: () => persistMutation({
                mutate: () => moveWorkflow(dirName, wfName, d),
                onSuccess: () => toast(`Moved "${wfName}" to ${d}.`),
                onNoOp: () => toast(`Could not move (name conflict?).`),
            }),
        }));

    const archiveItem = isArchived
        ? {
            label: "Restore from archive",
            action: () => persistMutation({
                mutate: () => unarchiveWorkflow(dirName, wfName),
                onSuccess: () => toast(`Restored "${wfName}".`),
            }),
        }
        : {
            label: "Move to archive",
            action: () => persistMutation({
                mutate: () => archiveWorkflow(dirName, wfName),
                onSuccess: () => toast(`Archived "${wfName}".`),
            }),
        };

    showContextMenu(event, [
        {
            label: "Load",
            action: () => loadWorkflowOntoCanvas(dirName, wfName),
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
                        mutate: () => renameWorkflow(dirName, wfName, newName),
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
                        mutate: () => deleteWorkflow(dirName, wfName),
                        onSuccess: () => toast(`Deleted "${wfName}".`),
                    }),
                });
            },
        },
    ]);
}

function directoryRowContextMenu(event, dirName) {
    const dir = dirOf(dirName);
    const isEmpty = !dir || Object.keys(dir.workflows || {}).length === 0;

    showContextMenu(event, [
        {
            label: "Rename directory…",
            action: () => {
                showInputModal({
                    title: "Rename directory",
                    label: "New name",
                    defaultValue: dirName,
                    confirmLabel: "Rename",
                    onSubmit: (newName) => persistMutation({
                        mutate: () => renameDirectory(dirName, newName),
                        onSuccess: () => toast(`Renamed to "${newName}".`),
                        onNoOp: () => toast(`Rename failed (name in use?).`),
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
                        message: `"${dirName}" will be removed.`,
                        confirmLabel: "Delete",
                        danger: true,
                        onConfirm: () => persistMutation({
                            mutate: () => deleteDirectory(dirName),
                            onSuccess: () => toast(`Deleted directory "${dirName}".`),
                        }),
                    });
                } else {
                    const count = Object.keys(dir.workflows).length;
                    showConfirmModal({
                        title: "Delete non-empty directory?",
                        message: `"${dirName}" contains ${count} workflow${count === 1 ? "" : "s"}. They will be permanently deleted.`,
                        confirmLabel: "Delete all",
                        danger: true,
                        onConfirm: () => persistMutation({
                            mutate: () => deleteDirectory(dirName),
                            onSuccess: () => toast(`Deleted "${dirName}" and its ${count} workflow${count === 1 ? "" : "s"}.`),
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
    emptyMessage: "No curated nodes yet. Click + above (with a canvas node selected) or right-click a node on the canvas → Add to Curated Sidebar.",
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
        for (const dir of ctx.data.directories) {
            const totalInDir = dir.active.length + dir.archived.length;
            ctx.folder({
                name: dir.name,
                count: totalInDir,
                path: dir.name,
                onContextMenu: (e) => directoryRowContextMenu(e, dir.name),
                build: (dirCtx) => {
                    // Archive first, then active — so the latest (active)
                    // workflow is closest to the bottom of the directory.
                    if (dir.archived.length > 0) {
                        dirCtx.folder({
                            name: "Archive",
                            count: dir.archived.length,
                            iconKind: "archive",
                            path: "Archive",
                            build: (archCtx) => {
                                for (const wfName of dir.archived) {
                                    archCtx.leaf({
                                        row: makeWorkflowLeafRow({
                                            name: wfName,
                                            dirName: dir.name,
                                            onClick: () => loadWorkflowOntoCanvas(dir.name, wfName),
                                            onContextMenu: (e) => workflowRowContextMenu(e, dir.name, wfName, true),
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
                                dirName: dir.name,
                                onClick: () => loadWorkflowOntoCanvas(dir.name, wfName),
                                onContextMenu: (e) => workflowRowContextMenu(e, dir.name, wfName, false),
                            }),
                        });
                    }
                },
            });
        }
    },
});

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
                mutate: () => addDirectory(name),
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
    const saveAndToast = async (graph, name, dir) => {
        // Pin the workflows section + destination dir so the resulting
        // re-render leaves them open. Pins are consumed on next render.
        pinExpanded(["workflows", `workflows/${dir}`]);
        await persistMutation({
            mutate: () => saveWorkflowEntry(dir, name, graph),
            onSuccess: (result) => {
                if (result.archivedAs) {
                    toast(`Saved "${name}" in ${dir}. Previous version moved to Archive.`);
                } else {
                    toast(`Saved "${name}" in ${dir}.`);
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
