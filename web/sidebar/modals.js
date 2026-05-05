// =============================================================================
// Modal + context-menu DOM. The shell takes care of the Escape-key / overlay-
// click / Cancel / OK dismissal paths and ties the document-level keydown
// listener to the overlay's lifetime so opening many modals doesn't leak.
// =============================================================================
import { compareNames } from "./constants.js";
import { listDirectoryNames, dirOf } from "./workflows_store.js";

function makeModalShell({ title, body, actions }) {
    const overlay = document.createElement("div");
    overlay.className = "koolook-modal-overlay";

    const modal = document.createElement("div");
    modal.className = "koolook-modal";

    const titleEl = document.createElement("div");
    titleEl.className = "koolook-modal-title";
    titleEl.textContent = title;
    modal.appendChild(titleEl);

    if (body) modal.appendChild(body);

    const actionsEl = document.createElement("div");
    actionsEl.className = "koolook-modal-actions";
    for (const action of actions) actionsEl.appendChild(action);
    modal.appendChild(actionsEl);

    // Centralized close so every dismissal path (Escape, overlay click, Cancel
    // button, OK button calling overlay.remove()) tears down the document-level
    // keydown listener too. Without this, every modal opened leaks one listener.
    let escHandler = null;
    const close = () => {
        if (escHandler) {
            document.removeEventListener("keydown", escHandler);
            escHandler = null;
        }
        if (overlay.parentNode) overlay.remove();
    };
    // Make overlay.remove() route through close so existing call sites that
    // do `overlay.remove()` continue to work (and clean up the listener too).
    const origRemove = overlay.remove.bind(overlay);
    overlay.remove = () => {
        if (escHandler) {
            document.removeEventListener("keydown", escHandler);
            escHandler = null;
        }
        origRemove();
    };

    overlay.appendChild(modal);
    overlay.addEventListener("click", (e) => {
        if (e.target === overlay) close();
    });
    escHandler = (e) => { if (e.key === "Escape") close(); };
    document.addEventListener("keydown", escHandler);
    document.body.appendChild(overlay);
    return { overlay, modal, close };
}

function makeModalButton({ label, primary, danger, onClick }) {
    const btn = document.createElement("button");
    btn.className = "koolook-modal-btn";
    if (primary) btn.classList.add("koolook-modal-btn-primary");
    if (danger) btn.classList.add("koolook-modal-btn-danger");
    btn.textContent = label;
    btn.addEventListener("click", onClick);
    return btn;
}

export function showInputModal({ title, label, defaultValue, placeholder, confirmLabel, onSubmit }) {
    const body = document.createElement("div");

    const lbl = document.createElement("label");
    lbl.className = "koolook-modal-label";
    lbl.textContent = label || "Name";
    body.appendChild(lbl);

    const input = document.createElement("input");
    input.className = "koolook-modal-input";
    input.value = defaultValue || "";
    input.placeholder = placeholder || "";
    body.appendChild(input);

    let overlay;
    const submit = () => {
        const v = input.value.trim();
        if (!v) {
            input.focus();
            return;
        }
        overlay.remove();
        onSubmit(v);
    };

    input.addEventListener("keydown", (e) => {
        if (e.key === "Enter") submit();
    });

    const cancel = makeModalButton({ label: "Cancel", onClick: () => overlay.remove() });
    const ok = makeModalButton({ label: confirmLabel || "OK", primary: true, onClick: submit });

    ({ overlay } = makeModalShell({ title, body, actions: [cancel, ok] }));
    setTimeout(() => { input.focus(); input.select(); }, 0);
}

export function showConfirmModal({ title, message, confirmLabel, cancelLabel, danger, onConfirm }) {
    const body = document.createElement("div");
    const msg = document.createElement("div");
    msg.className = "koolook-modal-message";
    msg.textContent = message;
    body.appendChild(msg);

    let overlay;
    const cancel = makeModalButton({ label: cancelLabel || "Cancel", onClick: () => overlay.remove() });
    const ok = makeModalButton({
        label: confirmLabel || "OK",
        primary: !danger,
        danger,
        onClick: () => { overlay.remove(); onConfirm(); },
    });
    ({ overlay } = makeModalShell({ title, body, actions: [cancel, ok] }));
}

export function showSaveWorkflowModal({ titleSuffix, defaultName, defaultDir, onSave }) {
    const body = document.createElement("div");

    // ---- Directory (cascading picker) ----
    // Each cascade level is a <select> showing the immediate children at
    // that depth. Sub-level selects start with a "(save in <path>)"
    // sentinel so the user can stop drilling at any level. Subdirectories
    // are created via right-click; "+ New directory…" at the top level
    // creates a NEW top-level directory only.
    const NEW_TOP = "__new__";
    const SAVE_HERE = "__here__";
    const topNames = listDirectoryNames([]);

    const dirLbl = document.createElement("label");
    dirLbl.className = "koolook-modal-label";
    dirLbl.textContent = "Directory";
    body.appendChild(dirLbl);

    const cascadeContainer = document.createElement("div");
    body.appendChild(cascadeContainer);

    const newDirInput = document.createElement("input");
    newDirInput.className = "koolook-modal-input";
    newDirInput.placeholder = "New directory name";
    newDirInput.style.marginTop = "6px";
    newDirInput.style.display = "none";
    body.appendChild(newDirInput);

    // The cascade: cascadeSelects[0] is the top-level <select>, [1] is the
    // immediate children of [0]'s value, etc. Each child select includes a
    // "(save in <path>)" option so drilling can stop at any depth.
    const cascadeSelects = [];

    function buildTopSelect() {
        const sel = document.createElement("select");
        sel.className = "koolook-modal-select";
        for (const name of topNames) {
            const opt = document.createElement("option");
            opt.value = name;
            opt.textContent = name;
            if (Array.isArray(defaultDir) && defaultDir[0] === name) opt.selected = true;
            sel.appendChild(opt);
        }
        const newOpt = document.createElement("option");
        newOpt.value = NEW_TOP;
        newOpt.textContent = "+ New directory…";
        if (topNames.length === 0) newOpt.selected = true;
        sel.appendChild(newOpt);
        sel.addEventListener("change", () => onCascadeChange(0));
        return sel;
    }

    function buildChildSelect(level, parentPath) {
        // Returns a <select> for the children of `parentPath`, or null if
        // that directory has no children (in which case no further select
        // is shown — save lands at parentPath).
        const children = listDirectoryNames(parentPath);
        if (children.length === 0) return null;

        const sel = document.createElement("select");
        sel.className = "koolook-modal-select";
        sel.style.marginTop = "6px";

        const hereOpt = document.createElement("option");
        hereOpt.value = SAVE_HERE;
        hereOpt.textContent = `(save in "${parentPath.join(" / ")}")`;
        sel.appendChild(hereOpt);

        let preselected = false;
        for (const name of children) {
            const opt = document.createElement("option");
            opt.value = name;
            opt.textContent = name;
            if (Array.isArray(defaultDir) && defaultDir[level] === name) {
                opt.selected = true;
                preselected = true;
            }
            sel.appendChild(opt);
        }
        // If no descendant of defaultDir matched at this level, default to
        // "(save here)" so the user doesn't fall accidentally deeper than
        // they intended.
        if (!preselected) sel.value = SAVE_HERE;
        sel.addEventListener("change", () => onCascadeChange(level));
        return sel;
    }

    function onCascadeChange(changedLevel) {
        // Tear down every cascade level deeper than the one that changed.
        while (cascadeSelects.length > changedLevel + 1) {
            const old = cascadeSelects.pop();
            old.remove();
        }

        const top = cascadeSelects[0];
        if (top.value === NEW_TOP) {
            newDirInput.style.display = "";
            // Tear down child selects too — they don't apply to a brand-new
            // top-level directory.
            while (cascadeSelects.length > 1) {
                const old = cascadeSelects.pop();
                old.remove();
            }
            applyState();
            return;
        }
        newDirInput.style.display = "none";

        // Walk the kept selects to compute the current path. Stop at
        // SAVE_HERE — anything below it has been torn down already.
        const path = [top.value];
        for (let i = 1; i < cascadeSelects.length; i += 1) {
            const v = cascadeSelects[i].value;
            if (v === SAVE_HERE) {
                applyState();
                return;
            }
            path.push(v);
        }

        // Try to extend the cascade one level deeper if children exist.
        const deeper = buildChildSelect(path.length, path);
        if (deeper) {
            cascadeContainer.appendChild(deeper);
            cascadeSelects.push(deeper);
            // If the new level pre-selected a real child (not "save here"),
            // recurse to build the level after it. This lets `defaultDir`
            // navigate the cascade without user clicks.
            if (deeper.value !== SAVE_HERE) {
                onCascadeChange(cascadeSelects.length - 1);
                return;
            }
        }
        applyState();
    }

    function getSelectedPath() {
        if (cascadeSelects.length === 0) return null;
        const top = cascadeSelects[0].value;
        if (top === NEW_TOP) {
            const t = newDirInput.value.trim();
            return t ? [t] : null;
        }
        const path = [top];
        for (let i = 1; i < cascadeSelects.length; i += 1) {
            const v = cascadeSelects[i].value;
            if (v === SAVE_HERE) break;
            path.push(v);
        }
        return path;
    }

    // Mount the top-level select and let onCascadeChange drive the rest.
    const topSelect = buildTopSelect();
    cascadeContainer.appendChild(topSelect);
    cascadeSelects.push(topSelect);
    // If no top-level dirs exist, the only option is "+ New directory…"
    // and the new-dir input should already be visible.
    if (topNames.length === 0) newDirInput.style.display = "";

    // ---- Base on existing (only shown when the chosen directory has workflows) ----
    const baseLbl = document.createElement("label");
    baseLbl.className = "koolook-modal-label";
    baseLbl.textContent = "Base on existing";
    body.appendChild(baseLbl);

    const baseSelect = document.createElement("select");
    baseSelect.className = "koolook-modal-select";
    body.appendChild(baseSelect);

    // ---- Action ----
    const actionLbl = document.createElement("label");
    actionLbl.className = "koolook-modal-label";
    actionLbl.textContent = "Action";
    body.appendChild(actionLbl);

    const actionSelect = document.createElement("select");
    actionSelect.className = "koolook-modal-select";
    [
        { value: "new", label: "New name" },
        { value: "use_existing", label: "Use existing name (archive previous)" },
        { value: "modify_existing", label: "Modify existing name" },
    ].forEach(a => {
        const opt = document.createElement("option");
        opt.value = a.value;
        opt.textContent = a.label;
        actionSelect.appendChild(opt);
    });
    body.appendChild(actionSelect);

    // ---- Workflow name (only shown when needed) ----
    const nameLbl = document.createElement("label");
    nameLbl.className = "koolook-modal-label";
    nameLbl.textContent = "Workflow name";
    body.appendChild(nameLbl);

    const nameInput = document.createElement("input");
    nameInput.className = "koolook-modal-input";
    nameInput.value = defaultName || "";
    nameInput.placeholder = "My workflow";
    body.appendChild(nameInput);

    // ---- Wiring ----
    // Candidates for "Base on existing": collect every active workflow at
    // the destination dir AND walk up its ancestors. So if the user picks
    // an empty subdir like "UP-scale / seedvr2", the parent's workflows
    // remain available to base on (instead of "Use existing name" /
    // "Modify existing name" being disabled because the leaf dir is
    // empty). De-duplication is by name, deepest-first wins (closest to
    // the destination is the most relevant if names collide).
    //
    // The `value` of each option stays the bare name (the existing submit
    // path uses `baseSelect.value` directly to populate the workflow
    // name). `textContent` adds a "  ·  in <path>" suffix when the
    // candidate is from an ancestor, so the user knows which directory
    // they're basing on.
    function getCandidatesForBase() {
        const path = getSelectedPath();
        if (!path || path.length === 0) return [];
        // For "+ New directory…" the destination doesn't exist yet — no
        // base candidates. (Ancestors don't apply: the new dir is at root.)
        if (cascadeSelects[0].value === NEW_TOP) return [];
        const seen = new Set();
        const out = [];
        for (let i = path.length; i >= 1; i -= 1) {
            const ancestorPath = path.slice(0, i);
            const dir = dirOf(ancestorPath);
            if (!dir || !dir.workflows) continue;
            const names = Object.keys(dir.workflows)
                .filter(n => !dir.workflows[n].archived)
                .sort(compareNames);
            for (const name of names) {
                if (seen.has(name)) continue;
                seen.add(name);
                out.push({ name, fromPath: ancestorPath, isCurrent: i === path.length });
            }
        }
        return out;
    }

    function rebuildBaseOptions(candidates) {
        const previous = baseSelect.value;
        baseSelect.innerHTML = "";
        for (const c of candidates) {
            const opt = document.createElement("option");
            opt.value = c.name;
            opt.textContent = c.isCurrent
                ? c.name
                : `${c.name}  ·  in ${c.fromPath.join(" / ")}`;
            baseSelect.appendChild(opt);
        }
        if (candidates.some(c => c.name === previous)) baseSelect.value = previous;
    }

    function applyState({ refocusName = false } = {}) {
        // newDirInput visibility is managed by onCascadeChange — it's tied
        // to whether the top select is on "+ New directory…", not to
        // anything applyState recomputes here.
        const dirIsNew = cascadeSelects[0]?.value === NEW_TOP;
        const candidates = dirIsNew ? [] : getCandidatesForBase();
        const hasBase = candidates.length > 0;

        // Base on existing — visible whenever the destination dir OR any of
        // its ancestors has active workflows. (Empty leaf dirs no longer
        // disable the existing-name actions when the parent has things to
        // base on.)
        if (hasBase) {
            baseLbl.style.display = "";
            baseSelect.style.display = "";
            rebuildBaseOptions(candidates);
        } else {
            baseLbl.style.display = "none";
            baseSelect.style.display = "none";
        }

        // Action — hide the whole dropdown when there's nothing to base on.
        // "Use existing" and "Modify existing" are meaningless without a
        // base, and disabled <option> elements render subtly across browsers
        // (the user just sees a "New name" they can't change). With no base,
        // the only useful action is a fresh save by name — the Workflow Name
        // field below already covers that case, so the dropdown is just
        // noise. With a base, restore the full action picker.
        if (hasBase) {
            actionLbl.style.display = "";
            actionSelect.style.display = "";
            for (const opt of actionSelect.options) {
                if (opt.value === "use_existing" || opt.value === "modify_existing") {
                    opt.disabled = false;
                }
            }
        } else {
            actionLbl.style.display = "none";
            actionSelect.style.display = "none";
            // Pin the underlying value to "new" so submit takes the Workflow
            // Name path.
            if (actionSelect.value !== "new") actionSelect.value = "new";
        }

        // Workflow name — visible only when needed.
        const action = actionSelect.value;
        if (action === "use_existing") {
            // Name comes from the base; field is irrelevant.
            nameLbl.style.display = "none";
            nameInput.style.display = "none";
        } else {
            nameLbl.style.display = "";
            nameInput.style.display = "";
            if (action === "modify_existing" && baseSelect.value) {
                nameInput.value = baseSelect.value;
                nameInput.readOnly = false;
                if (refocusName) {
                    setTimeout(() => {
                        nameInput.focus();
                        const len = nameInput.value.length;
                        nameInput.setSelectionRange(len, len);
                    }, 0);
                }
            } else {
                // "new" — leave the value alone so the user's typing isn't clobbered.
                nameInput.readOnly = false;
            }
        }
    }

    actionSelect.addEventListener("change", () => applyState({ refocusName: true }));
    baseSelect.addEventListener("change", () => {
        if (actionSelect.value === "modify_existing") {
            nameInput.value = baseSelect.value;
        }
    });

    // Drive the initial cascade — onCascadeChange recurses to honor
    // `defaultDir` if any deeper levels need pre-selection. Then
    // applyState() runs to compute the rest of the modal.
    onCascadeChange(0);
    applyState();
    if (cascadeSelects[0].value === NEW_TOP) newDirInput.focus();

    let overlay;
    const submit = async () => {
        // `dir` is a path array drawn from the cascade. "+ New directory…"
        // yields a new top-level (single-segment) path once the user types
        // a name; existing-dir selections yield the deepest path they
        // drilled to before stopping at "(save in <path>)".
        const dir = getSelectedPath();
        if (!dir || dir.length === 0) {
            // Top is "__new__" with empty input — prompt for a name.
            if (cascadeSelects[0].value === NEW_TOP) newDirInput.focus();
            return;
        }
        const action = actionSelect.value;
        let name;
        if (action === "use_existing") {
            name = (baseSelect.value || "").trim();
            if (!name) { actionSelect.value = "new"; applyState(); nameInput.focus(); return; }
        } else {
            name = nameInput.value.trim();
            if (!name) { nameInput.focus(); return; }
        }
        overlay.remove();
        await onSave({ name, dir });
    };

    nameInput.addEventListener("keydown", (e) => { if (e.key === "Enter") submit(); });
    newDirInput.addEventListener("keydown", (e) => { if (e.key === "Enter") submit(); });

    const cancel = makeModalButton({ label: "Cancel", onClick: () => overlay.remove() });
    const ok = makeModalButton({ label: "Save", primary: true, onClick: submit });
    ({ overlay } = makeModalShell({
        title: titleSuffix ? `Save workflow — ${titleSuffix}` : "Save workflow",
        body,
        actions: [cancel, ok],
    }));
    setTimeout(() => {
        if (nameInput.style.display !== "none") {
            nameInput.focus();
            nameInput.select();
        }
    }, 0);
}

// =============================================================================
// Tags modal — adds/removes tags one at a time, each via its own
// persistMutation (passed in by the caller). The modal itself stays open until
// the user clicks Close, so several edits can be made in a row. Chip rendering
// reads from `getCurrentTags` after every successful mutation, so the live
// state matches what the caller persisted.
//
// `getCurrentTags()` returns `null` when the underlying workflow no longer
// exists (deleted/moved/renamed by a concurrent action). We surface this as
// an explicit "gone" state — empty placeholder + disabled add input — so the
// user doesn't keep typing tags into a modal that can't persist them.
// =============================================================================
export function showTagsModal({ wfName, getCurrentTags, onAddTag, onRemoveTag }) {
    const body = document.createElement("div");

    const lbl = document.createElement("label");
    lbl.className = "koolook-modal-label";
    lbl.textContent = "Current tags";
    body.appendChild(lbl);

    const chipsContainer = document.createElement("div");
    chipsContainer.className = "koolook-tags-chips";
    body.appendChild(chipsContainer);

    let input;
    let addBtn;
    function renderChips() {
        chipsContainer.innerHTML = "";
        const tags = getCurrentTags();
        if (tags === null) {
            // Workflow disappeared between menu-open and now. Stay open so
            // the user sees the explicit reason instead of a generic "no
            // tags" placeholder, and disable the add path so a second click
            // can't fire a doomed persistMutation.
            const empty = document.createElement("span");
            empty.className = "koolook-tags-empty";
            empty.textContent = "Workflow no longer exists.";
            chipsContainer.appendChild(empty);
            if (input) {
                input.disabled = true;
                input.placeholder = "(workflow gone)";
            }
            if (addBtn) addBtn.disabled = true;
            return;
        }
        if (tags.length === 0) {
            const empty = document.createElement("span");
            empty.className = "koolook-tags-empty";
            empty.textContent = "No tags yet.";
            chipsContainer.appendChild(empty);
            return;
        }
        for (const tag of tags) {
            const chip = document.createElement("span");
            chip.className = "koolook-tag-chip";
            const name = document.createElement("span");
            name.textContent = tag;
            chip.appendChild(name);
            const x = document.createElement("span");
            x.className = "koolook-tag-chip-x";
            x.textContent = "×";
            x.title = `Remove "${tag}"`;
            x.addEventListener("click", () => {
                onRemoveTag(tag, () => renderChips());
            });
            chip.appendChild(x);
            chipsContainer.appendChild(chip);
        }
    }

    const addLbl = document.createElement("label");
    addLbl.className = "koolook-modal-label";
    addLbl.textContent = "Add tag";
    body.appendChild(addLbl);

    const addRow = document.createElement("div");
    addRow.className = "koolook-tag-add-row";
    body.appendChild(addRow);

    input = document.createElement("input");
    input.className = "koolook-modal-input";
    input.placeholder = "tag name";
    addRow.appendChild(input);

    const submit = () => {
        if (input.disabled) return;
        const v = input.value.trim();
        if (!v) { input.focus(); return; }
        onAddTag(v, () => {
            input.value = "";
            renderChips();
            input.focus();
        });
    };
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") submit(); });

    addBtn = makeModalButton({ label: "Add", primary: true, onClick: submit });
    addRow.appendChild(addBtn);

    // Now that input + addBtn exist, run the initial render so the
    // gone-state branch can disable them if the workflow is already missing.
    renderChips();

    let overlay;
    const close = makeModalButton({ label: "Close", onClick: () => overlay.remove() });
    ({ overlay } = makeModalShell({
        title: `Tags for "${wfName}"`,
        body,
        actions: [close],
    }));
    setTimeout(() => { if (!input.disabled) input.focus(); }, 0);
}

// =============================================================================
// Context menu helper
// =============================================================================
export function showContextMenu(event, items) {
    event.preventDefault();
    event.stopPropagation();

    const menu = document.createElement("div");
    menu.className = "koolook-context-menu";

    for (const item of items) {
        if (!item) {
            const sep = document.createElement("div");
            sep.className = "koolook-context-sep";
            menu.appendChild(sep);
            continue;
        }
        const m = document.createElement("div");
        m.className = "koolook-context-item";
        if (item.danger) m.classList.add("koolook-context-danger");
        m.textContent = item.label;
        if (item.disabled) {
            m.style.opacity = "0.4";
            m.style.cursor = "not-allowed";
        } else {
            m.addEventListener("click", () => {
                menu.remove();
                item.action();
            });
        }
        menu.appendChild(m);
    }

    menu.style.left = `${event.clientX}px`;
    menu.style.top = `${event.clientY}px`;
    document.body.appendChild(menu);

    const rect = menu.getBoundingClientRect();
    if (rect.right > window.innerWidth) menu.style.left = `${event.clientX - rect.width}px`;
    if (rect.bottom > window.innerHeight) menu.style.top = `${event.clientY - rect.height}px`;

    setTimeout(() => {
        const closeOnClick = (ev) => {
            if (!menu.contains(ev.target)) {
                menu.remove();
                document.removeEventListener("click", closeOnClick);
                document.removeEventListener("contextmenu", closeOnClick);
            }
        };
        document.addEventListener("click", closeOnClick);
        document.addEventListener("contextmenu", closeOnClick);
    }, 0);
}
