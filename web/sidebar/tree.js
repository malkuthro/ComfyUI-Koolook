// =============================================================================
// Sidebar tree — data gathering, DOM row factories, folder builder, context
// menus for workflow/directory rows, the tree-rebuild dispatcher, and the
// panel renderer (search, action bars, tree mount, event subscriptions).
//
// `pathStates` is module-private — folder expansion state survives across
// re-renders triggered by saves, picks-changed events, or storage events.
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
// Folder expansion state — persisted across re-renders so saving (or any
// other action that triggers a tree rebuild) doesn't collapse what the user
// was looking at. Map<path, boolean>; truthy = expanded.
// =============================================================================
const pathStates = new Map();

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
        if (!dir) continue;
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
// Tree rendering
// =============================================================================
function rebuildTree(treeEl, query) {
    treeEl.innerHTML = "";
    const isFiltered = (query || "").trim().length > 0;

    // ------ Gather both sections first so we know whether to render a divider ------
    const autoPacks = gatherNodesByRepo(query);
    const userPickPacks = gatherUserPickPacks(query);
    const allPacks = [...autoPacks, ...userPickPacks]
        .filter(p => p.total > 0)
        .sort((a, b) => compareNames(a.label, b.label));
    const nodesTotal = allPacks.reduce((acc, p) => acc + p.total, 0);
    const showNodes = nodesTotal > 0;

    const wf = gatherWorkflows(query);
    const showWorkflows = wf.directories.length > 0;

    // Search returned nothing across both sections.
    if (!showNodes && !showWorkflows && isFiltered) {
        const empty = document.createElement("div");
        empty.className = "koolook-empty";
        empty.textContent = "No nodes or workflows match your search.";
        treeEl.appendChild(empty);
        return;
    }

    // ------ Nodes section ------
    if (showNodes) {
        const nodesGroup = buildFolder({
            name: ROOT_GROUP_LABEL,
            count: nodesTotal,
            iconKind: "favorites",
            startExpanded: true,
            path: "nodes",
            forceExpanded: isFiltered,
            childrenBuilder: (rootChildren) => {
                for (const pack of allPacks) {
                    const packFolder = buildFolder({
                        name: pack.label,
                        count: pack.total,
                        startExpanded: false,
                        path: `nodes/${pack.label}`,
                        forceExpanded: isFiltered,
                        childrenBuilder: (packChildren) => {
                            for (const cat of pack.categories) {
                                const catFolder = buildFolder({
                                    name: cat.name,
                                    count: cat.nodes.length,
                                    startExpanded: false,
                                    path: `nodes/${pack.label}/${cat.name}`,
                                    forceExpanded: isFiltered,
                                    childrenBuilder: (catChildren) => {
                                        for (const n of cat.nodes) {
                                            catChildren.appendChild(makeNodeLeafRow({
                                                display: n.display,
                                                type: n.type,
                                                removable: !!pack.isUserPicks,
                                                onClick: () => insertNode(n.type),
                                            }));
                                        }
                                    },
                                });
                                packChildren.appendChild(catFolder);
                            }
                        },
                    });
                    rootChildren.appendChild(packFolder);
                }
            },
        });
        treeEl.appendChild(nodesGroup);
    } else if (!isFiltered) {
        const empty = document.createElement("div");
        empty.className = "koolook-empty";
        empty.textContent = "No curated nodes yet. Click + above (with a canvas node selected) or right-click a node on the canvas → Add to Curated Sidebar.";
        treeEl.appendChild(empty);
    }

    // ------ Divider between sections ------
    if (showNodes && showWorkflows) {
        const divider = document.createElement("div");
        divider.className = "koolook-tree-divider";
        treeEl.appendChild(divider);
    }

    // ------ Workflows section ------
    if (showWorkflows) {
        const workflowsGroup = buildFolder({
            name: WORKFLOWS_GROUP_LABEL,
            count: wf.total,
            iconKind: "workflows",
            startExpanded: true,
            path: "workflows",
            forceExpanded: isFiltered,
            childrenBuilder: (wfChildren) => {
                for (const dir of wf.directories) {
                    const totalInDir = dir.active.length + dir.archived.length;
                    const dirFolder = buildFolder({
                        name: dir.name,
                        count: totalInDir,
                        startExpanded: false,
                        path: `workflows/${dir.name}`,
                        forceExpanded: isFiltered,
                        onContextMenu: (e) => directoryRowContextMenu(e, dir.name),
                        childrenBuilder: (dirChildren) => {
                            // Archive first, then active — so the latest (active)
                            // workflow is closest to the bottom of the directory.
                            if (dir.archived.length > 0) {
                                const archiveFolder = buildFolder({
                                    name: "Archive",
                                    count: dir.archived.length,
                                    iconKind: "archive",
                                    startExpanded: false,
                                    path: `workflows/${dir.name}/Archive`,
                                    forceExpanded: isFiltered,
                                    childrenBuilder: (archiveChildren) => {
                                        for (const wfName of dir.archived) {
                                            archiveChildren.appendChild(makeWorkflowLeafRow({
                                                name: wfName,
                                                dirName: dir.name,
                                                onClick: () => loadWorkflowOntoCanvas(dir.name, wfName),
                                                onContextMenu: (e) => workflowRowContextMenu(e, dir.name, wfName, true),
                                            }));
                                        }
                                    },
                                });
                                dirChildren.appendChild(archiveFolder);
                            }
                            for (const wfName of dir.active) {
                                dirChildren.appendChild(makeWorkflowLeafRow({
                                    name: wfName,
                                    dirName: dir.name,
                                    onClick: () => loadWorkflowOntoCanvas(dir.name, wfName),
                                    onContextMenu: (e) => workflowRowContextMenu(e, dir.name, wfName, false),
                                }));
                            }
                        },
                    });
                    wfChildren.appendChild(dirFolder);
                }
            },
        });
        treeEl.appendChild(workflowsGroup);
    }
    // (When workflows is empty and not filtered, we silently hide the section;
    // the toolbar above still lets the user create a directory or save a workflow.)
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
        pathStates.set("workflows", true);
        pathStates.set(`workflows/${dir}`, true);
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

    rebuildTree(tree, "");

    // ---- Search wiring ----
    let debounce = null;
    search.addEventListener("input", (e) => {
        const q = e.target.value;
        if (debounce) clearTimeout(debounce);
        debounce = setTimeout(() => rebuildTree(tree, q), 60);
    });

    // ---- Event subscriptions for live re-rendering ----
    window.addEventListener(PICKS_CHANGED_EVENT, () => rebuildTree(tree, search.value));
    window.addEventListener(WORKFLOWS_CHANGED_EVENT, () => rebuildTree(tree, search.value));
    window.addEventListener("storage", (e) => {
        if (e.key === STORAGE_KEY || e.key === WORKFLOWS_FALLBACK_KEY) {
            rebuildTree(tree, search.value);
        }
    });
}
