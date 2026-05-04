import { app } from "../../../scripts/app.js";

// =============================================================================
// Configuration — static "always shown" entries.
//
// You normally don't need to edit this. To grow the panel from inside
// ComfyUI, use either:
//   • the "+" button at the top of the sidebar (with a canvas node selected), or
//   • right-click a canvas node → "Add to Curated Sidebar".
//
// Those additions are stored in the browser's localStorage and surface as
// the "My Picks" group at the top of the tree.
// =============================================================================
const REPOS = [
    {
        label: "Koolook",
        categoryRoot: "Koolook",
        select: "all",
        excludePatterns: [/\bdeprecated\b/i],
    },
];

const TAB_ID = "koolook.curatedNodes";
const TAB_TITLE = "Curated Nodes";
const TAB_TOOLTIP = "Curated selection of ComfyUI nodes";
const TAB_ICON = "pi pi-star";
const ROOT_GROUP_LABEL = "Nodes (favorites)";

const STORAGE_KEY = "koolook.curated.userPicks.v1";
const SEEDED_KEY = "koolook.curated.seeded.v1";
const PICKS_CHANGED_EVENT = "koolook-picks-changed";
const DEFAULTS_URL = new URL("./curated_defaults.json", import.meta.url).href;

// =============================================================================
// Styles
// =============================================================================
const STYLE_ID = "koolook-sidebar-style";
function ensureStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const s = document.createElement("style");
    s.id = STYLE_ID;
    s.textContent = `
.koolook-sidebar { display: flex; flex-direction: column; height: 100%; font-size: 13px; user-select: none; }
.koolook-toolbar { display: flex; gap: 4px; margin: 6px; align-items: stretch; flex-shrink: 0; }
.koolook-search-wrap { position: relative; flex: 1; }
.koolook-search-icon { position: absolute; left: 8px; top: 50%; transform: translateY(-50%); opacity: 0.55; font-size: 11px; pointer-events: none; }
.koolook-search { width: 100%; padding: 5px 8px 5px 26px; box-sizing: border-box; background: var(--comfy-input-bg, rgba(0,0,0,0.25)); border: 1px solid var(--border-color, rgba(255,255,255,0.1)); border-radius: 4px; color: var(--input-text, inherit); font-size: 12px; outline: none; }
.koolook-search:focus { border-color: var(--p-primary-color, rgba(100,150,255,0.5)); }
.koolook-add-btn { padding: 0 12px; background: var(--comfy-input-bg, rgba(0,0,0,0.25)); border: 1px solid var(--border-color, rgba(255,255,255,0.1)); border-radius: 4px; cursor: pointer; color: var(--input-text, inherit); font-size: 16px; line-height: 1; flex-shrink: 0; }
.koolook-add-btn:hover { background: rgba(255,255,255,0.1); }
.koolook-add-btn:active { background: rgba(255,255,255,0.15); }
.koolook-export-btn { padding: 0 10px; font-size: 12px; }
.koolook-tree { flex: 1; overflow-y: auto; padding: 0 4px 8px; }
.koolook-row { display: flex; align-items: center; padding: 3px 6px; cursor: pointer; gap: 6px; border-radius: 3px; line-height: 1.3; }
.koolook-row:hover { background: var(--comfy-input-bg, rgba(255,255,255,0.06)); }
.koolook-chevron { width: 10px; display: inline-block; opacity: 0.7; text-align: center; font-size: 10px; flex-shrink: 0; }
.koolook-folder-icon { opacity: 0.85; flex-shrink: 0; }
.koolook-pin-icon { color: #ffb84d; opacity: 0.95; flex-shrink: 0; }
.koolook-leaf-dot { width: 6px; height: 6px; margin: 0 2px; border-radius: 50%; background: rgba(255,255,255,0.45); flex-shrink: 0; }
.koolook-name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.koolook-count { background: rgba(255,255,255,0.08); padding: 1px 7px; border-radius: 8px; font-size: 11px; opacity: 0.75; }
.koolook-children { padding-left: 14px; }
.koolook-empty { padding: 14px 8px; opacity: 0.65; font-size: 12px; line-height: 1.4; }
.koolook-leaf .koolook-chevron { visibility: hidden; }
.koolook-remove { padding: 0 4px; margin-left: 2px; opacity: 0; font-size: 14px; line-height: 1; cursor: pointer; flex-shrink: 0; }
.koolook-row:hover .koolook-remove { opacity: 0.5; }
.koolook-remove:hover { opacity: 1 !important; color: #ff7777; }
.koolook-toast { position: fixed; bottom: 30px; right: 30px; background: rgba(40,40,40,0.95); color: #fff; padding: 8px 14px; border-radius: 4px; font-size: 12px; z-index: 9999; transition: opacity 0.3s; box-shadow: 0 2px 8px rgba(0,0,0,0.4); pointer-events: none; }
`;
    document.head.appendChild(s);
}

// =============================================================================
// User picks (localStorage, browser-local persistence)
// =============================================================================
function loadUserPicks() {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) return [];
        const parsed = JSON.parse(raw);
        return Array.isArray(parsed) ? parsed.filter(x => typeof x === "string") : [];
    } catch (e) {
        return [];
    }
}

function saveUserPicks(picks) {
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(picks));
    } catch (e) {
        console.warn("[Koolook] failed to save picks:", e);
    }
}

function addToMyPicks(typeName) {
    if (!typeName) return false;
    const picks = loadUserPicks();
    if (picks.includes(typeName)) return false;
    picks.push(typeName);
    saveUserPicks(picks);
    return true;
}

function removeFromMyPicks(typeName) {
    saveUserPicks(loadUserPicks().filter(p => p !== typeName));
}

function notifyPicksChanged() {
    window.dispatchEvent(new CustomEvent(PICKS_CHANGED_EVENT));
}

// Seed defaults from `web/curated_defaults.json` exactly once per browser.
// Existing local picks always win — defaults are only applied to a truly empty
// localStorage. After seeding (or skipping), SEEDED_KEY is set so this never
// runs again for that browser, regardless of what the defaults file changes to.
async function seedDefaultsIfNeeded() {
    if (localStorage.getItem(SEEDED_KEY)) return;
    if (loadUserPicks().length > 0) {
        localStorage.setItem(SEEDED_KEY, "1");
        return;
    }
    try {
        const resp = await fetch(DEFAULTS_URL);
        if (!resp.ok) {
            localStorage.setItem(SEEDED_KEY, "1");
            return;
        }
        const data = await resp.json();
        const picks = (data && Array.isArray(data.picks))
            ? data.picks.filter(p => typeof p === "string")
            : [];
        if (picks.length > 0) saveUserPicks(picks);
        localStorage.setItem(SEEDED_KEY, "1");
        console.log(`[Koolook] seeded ${picks.length} default pick(s)`);
    } catch (e) {
        console.warn("[Koolook] failed to load curated_defaults.json:", e);
        localStorage.setItem(SEEDED_KEY, "1");
    }
}

// Build a deterministic JSON snapshot of the user's current picks suitable
// for committing to the package as `web/curated_defaults.json`.
function picksAsDistributionJSON() {
    const picks = [...loadUserPicks()].sort();
    return JSON.stringify({ picks }, null, 2) + "\n";
}

async function exportPicks() {
    const picks = loadUserPicks();
    if (picks.length === 0) {
        toast("No picks to export yet.");
        return;
    }
    const json = picksAsDistributionJSON();
    const noun = picks.length === 1 ? "pick" : "picks";
    try {
        await navigator.clipboard.writeText(json);
        toast(`Copied ${picks.length} ${noun} to clipboard. Paste into web/curated_defaults.json.`);
        return;
    } catch (e) {
        console.warn("[Koolook] clipboard write failed, falling back to download:", e);
    }
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "curated_defaults.json";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    toast(`Downloaded ${picks.length} ${noun} as curated_defaults.json. Replace the file in web/ with it.`);
}

// =============================================================================
// Toast helper
// =============================================================================
function toast(msg, duration = 2200) {
    const t = document.createElement("div");
    t.className = "koolook-toast";
    t.textContent = msg;
    document.body.appendChild(t);
    setTimeout(() => {
        t.style.opacity = "0";
        setTimeout(() => t.remove(), 300);
    }, duration);
}

// =============================================================================
// Data gathering
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
                nodes: nodes.sort((a, b) => a.display.localeCompare(b.display)),
            }))
            .sort((a, b) => a.name.localeCompare(b.name));

        out.push({ label: repo.label, categories, total });
    }
    return out;
}

function gatherUserPickPacks(query) {
    const q = (query || "").trim().toLowerCase();
    const registry = (typeof LiteGraph !== "undefined" && LiteGraph.registered_node_types) || {};
    const picks = loadUserPicks();

    // Don't duplicate nodes already shown via an auto-included REPO entry.
    const autoCategoryRoots = new Set(
        REPOS.filter(r => r.select === "all" && r.categoryRoot).map(r => r.categoryRoot)
    );

    // Group picks by source pack (first segment of the node's category).
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
                nodes: nodes.sort((a, b) => a.display.localeCompare(b.display)),
            }))
            .sort((a, b) => a.name.localeCompare(b.name));
        const total = categories.reduce((acc, c) => acc + c.nodes.length, 0);
        result.push({ label: packLabel, categories, total, isUserPicks: true });
    }
    result.sort((a, b) => a.label.localeCompare(b.label));
    return result;
}

// =============================================================================
// Node insertion
// =============================================================================
function placeAtCanvasCenter(node) {
    try {
        const canvas = app.canvas;
        const ds = canvas.ds;
        const cx = -ds.offset[0] + canvas.canvas.width / (2 * ds.scale);
        const cy = -ds.offset[1] + canvas.canvas.height / (2 * ds.scale);
        node.pos = [cx - node.size[0] / 2, cy - node.size[1] / 2];
    } catch (e) {
        // Default position is fine if the canvas isn't ready yet.
    }
}

function insertNode(typeName) {
    if (typeof LiteGraph === "undefined") return;
    const node = LiteGraph.createNode(typeName);
    if (!node) {
        console.warn(`[Koolook] could not create node: ${typeName}`);
        return;
    }
    app.graph.add(node);
    placeAtCanvasCenter(node);
    app.canvas.setDirty(true, true);
}

function getSelectedNodeTypes() {
    try {
        const sel = (app.canvas && app.canvas.selected_nodes) || {};
        return Object.values(sel)
            .filter(n => n && n.type)
            .map(n => n.type);
    } catch (e) {
        return [];
    }
}

// =============================================================================
// DOM helpers
// =============================================================================
function makeFolderRow({ name, count, isRootGroup, onToggle }) {
    const row = document.createElement("div");
    row.className = "koolook-row";

    const chevron = document.createElement("span");
    chevron.className = "koolook-chevron";
    chevron.textContent = "▾";
    row.appendChild(chevron);

    const icon = document.createElement("span");
    icon.className = isRootGroup
        ? "pi pi-star koolook-pin-icon"
        : "pi pi-folder koolook-folder-icon";
    row.appendChild(icon);

    const nameEl = document.createElement("span");
    nameEl.className = "koolook-name";
    nameEl.textContent = name;
    row.appendChild(nameEl);

    const cnt = document.createElement("span");
    cnt.className = "koolook-count";
    cnt.textContent = String(count);
    row.appendChild(cnt);

    row.addEventListener("click", onToggle);
    return { row, chevron };
}

function makeLeafRow({ display, type, removable, onClick }) {
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
        rm.title = "Remove from My Picks";
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

function buildFolder(name, count, isRootGroup, childrenBuilder, startExpanded = true) {
    const wrapper = document.createElement("div");
    wrapper.dataset.expanded = startExpanded ? "true" : "false";

    const children = document.createElement("div");
    children.className = "koolook-children";
    if (!startExpanded) children.style.display = "none";

    const { row, chevron } = makeFolderRow({
        name,
        count,
        isRootGroup,
        onToggle: () => {
            const expanded = wrapper.dataset.expanded !== "false";
            wrapper.dataset.expanded = expanded ? "false" : "true";
            chevron.textContent = expanded ? "▸" : "▾";
            children.style.display = expanded ? "none" : "";
        },
    });
    if (!startExpanded) chevron.textContent = "▸";

    wrapper.appendChild(row);
    wrapper.appendChild(children);
    childrenBuilder(children);
    return wrapper;
}

// =============================================================================
// Tree rendering
// =============================================================================
function rebuildTree(treeEl, query) {
    treeEl.innerHTML = "";

    const autoPacks = gatherNodesByRepo(query);
    const userPickPacks = gatherUserPickPacks(query);
    const allPacks = [...autoPacks, ...userPickPacks].filter(p => p.total > 0);
    const grandTotal = allPacks.reduce((acc, p) => acc + p.total, 0);

    if (grandTotal === 0) {
        const empty = document.createElement("div");
        empty.className = "koolook-empty";
        empty.textContent = (query || "").trim()
            ? "No nodes match your search."
            : "No curated nodes yet. Click + above (with a canvas node selected) or right-click a node on the canvas → Add to Curated Sidebar.";
        treeEl.appendChild(empty);
        return;
    }

    // Pack and category folders default to collapsed so the panel is compact
    // when first opened. When a search query is active, expand everything so
    // matches don't hide inside collapsed folders.
    const isFiltered = (query || "").trim().length > 0;

    // One top-level group containing every pack — auto-included and user-picked alike.
    const rootGroup = buildFolder(ROOT_GROUP_LABEL, grandTotal, true, (rootChildren) => {
        for (const pack of allPacks) {
            const packFolder = buildFolder(pack.label, pack.total, false, (packChildren) => {
                for (const cat of pack.categories) {
                    const catFolder = buildFolder(cat.name, cat.nodes.length, false, (catChildren) => {
                        for (const n of cat.nodes) {
                            catChildren.appendChild(makeLeafRow({
                                display: n.display,
                                type: n.type,
                                removable: !!pack.isUserPicks,
                                onClick: () => insertNode(n.type),
                            }));
                        }
                    }, isFiltered);
                    packChildren.appendChild(catFolder);
                }
            }, isFiltered);
            rootChildren.appendChild(packFolder);
        }
    }, true);

    treeEl.appendChild(rootGroup);
}

// =============================================================================
// Panel rendering (toolbar + tree)
// =============================================================================
function renderPanel(container) {
    ensureStyle();
    container.innerHTML = "";
    container.classList.add("koolook-sidebar");

    const toolbar = document.createElement("div");
    toolbar.className = "koolook-toolbar";

    const searchWrap = document.createElement("div");
    searchWrap.className = "koolook-search-wrap";

    const searchIcon = document.createElement("span");
    searchIcon.className = "pi pi-search koolook-search-icon";
    searchWrap.appendChild(searchIcon);

    const search = document.createElement("input");
    search.type = "search";
    search.className = "koolook-search";
    search.placeholder = "Search nodes...";
    searchWrap.appendChild(search);
    toolbar.appendChild(searchWrap);

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
        for (const t of types) {
            if (addToMyPicks(t)) added += 1;
        }
        if (added > 0) {
            const noun = added === 1 ? "node" : "nodes";
            toast(`Added ${added} ${noun} to favorites.`);
            notifyPicksChanged();
        } else {
            toast("Already in favorites.");
        }
    });
    toolbar.appendChild(addBtn);

    const exportBtn = document.createElement("button");
    exportBtn.className = "koolook-add-btn koolook-export-btn";
    exportBtn.innerHTML = '<span class="pi pi-download"></span>';
    exportBtn.title = "Export current picks as curated_defaults.json (copies JSON to clipboard for committing to the package)";
    exportBtn.addEventListener("click", exportPicks);
    toolbar.appendChild(exportBtn);

    container.appendChild(toolbar);

    const tree = document.createElement("div");
    tree.className = "koolook-tree";
    container.appendChild(tree);

    rebuildTree(tree, "");

    let debounce = null;
    search.addEventListener("input", (e) => {
        const q = e.target.value;
        if (debounce) clearTimeout(debounce);
        debounce = setTimeout(() => rebuildTree(tree, q), 60);
    });

    // Re-render when picks change (from + button, ×, right-click menu, or another tab)
    window.addEventListener(PICKS_CHANGED_EVENT, () => rebuildTree(tree, search.value));
    window.addEventListener("storage", (e) => {
        if (e.key === STORAGE_KEY) rebuildTree(tree, search.value);
    });
}

// =============================================================================
// Right-click context menu patch
// =============================================================================
function patchCanvasMenu() {
    const C = (typeof LGraphCanvas !== "undefined") ? LGraphCanvas : null;
    if (!C || !C.prototype || !C.prototype.getNodeMenuOptions) {
        console.warn("[Koolook] LGraphCanvas.getNodeMenuOptions not available; right-click menu skipped.");
        return;
    }
    const orig = C.prototype.getNodeMenuOptions;
    C.prototype.getNodeMenuOptions = function (node) {
        const options = orig.apply(this, arguments);
        options.push(null); // separator
        options.push({
            content: "Add to Curated Sidebar",
            callback: () => {
                if (!node || !node.type) return;
                if (addToMyPicks(node.type)) {
                    toast(`Added "${node.title || node.type}" to My Picks.`);
                    notifyPicksChanged();
                } else {
                    toast("Already in My Picks.");
                }
            },
        });
        return options;
    };
}

// =============================================================================
// Tab registration
// =============================================================================
app.registerExtension({
    name: "koolook.curated_sidebar",
    async setup() {
        if (!app.extensionManager || !app.extensionManager.registerSidebarTab) {
            console.warn("[Koolook] extensionManager.registerSidebarTab not available; sidebar not registered.");
            return;
        }
        await seedDefaultsIfNeeded();
        app.extensionManager.registerSidebarTab({
            id: TAB_ID,
            title: TAB_TITLE,
            tooltip: TAB_TOOLTIP,
            icon: TAB_ICON,
            type: "custom",
            render: (el) => renderPanel(el),
        });
        patchCanvasMenu();
    },
});
