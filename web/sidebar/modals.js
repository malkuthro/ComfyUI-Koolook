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

    // ---- Directory ----
    const dirLbl = document.createElement("label");
    dirLbl.className = "koolook-modal-label";
    dirLbl.textContent = "Directory";
    body.appendChild(dirLbl);

    const dirNames = listDirectoryNames();
    const dirSelect = document.createElement("select");
    dirSelect.className = "koolook-modal-select";
    if (dirNames.length === 0) {
        const opt = document.createElement("option");
        opt.value = "__new__";
        opt.textContent = "+ New directory…";
        dirSelect.appendChild(opt);
    } else {
        for (const d of dirNames) {
            const opt = document.createElement("option");
            opt.value = d;
            opt.textContent = d;
            if (d === defaultDir) opt.selected = true;
            dirSelect.appendChild(opt);
        }
        const newOpt = document.createElement("option");
        newOpt.value = "__new__";
        newOpt.textContent = "+ New directory…";
        dirSelect.appendChild(newOpt);
    }
    body.appendChild(dirSelect);

    const newDirInput = document.createElement("input");
    newDirInput.className = "koolook-modal-input";
    newDirInput.placeholder = "New directory name";
    newDirInput.style.marginTop = "6px";
    newDirInput.style.display = dirNames.length === 0 ? "" : "none";
    body.appendChild(newDirInput);

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
    function getActiveWorkflowsInCurrentDir() {
        const v = dirSelect.value;
        if (v === "__new__") return [];
        const dir = dirOf(v);
        if (!dir) return [];
        return Object.keys(dir.workflows)
            .filter(n => !dir.workflows[n].archived)
            .sort(compareNames);
    }

    function rebuildBaseOptions(names) {
        const previous = baseSelect.value;
        baseSelect.innerHTML = "";
        for (const n of names) {
            const opt = document.createElement("option");
            opt.value = n;
            opt.textContent = n;
            baseSelect.appendChild(opt);
        }
        if (names.includes(previous)) baseSelect.value = previous;
    }

    function applyState({ refocusName = false } = {}) {
        const dirIsNew = dirSelect.value === "__new__";
        newDirInput.style.display = dirIsNew ? "" : "none";

        const activeNames = dirIsNew ? [] : getActiveWorkflowsInCurrentDir();
        const hasBase = activeNames.length > 0;

        // Base on existing — visible only when the chosen directory has active workflows.
        if (hasBase) {
            baseLbl.style.display = "";
            baseSelect.style.display = "";
            rebuildBaseOptions(activeNames);
        } else {
            baseLbl.style.display = "none";
            baseSelect.style.display = "none";
        }

        // Action options — disable "Use existing" / "Modify existing" when no base is available.
        for (const opt of actionSelect.options) {
            if (opt.value === "use_existing" || opt.value === "modify_existing") {
                opt.disabled = !hasBase;
            }
        }
        if (!hasBase && actionSelect.value !== "new") {
            actionSelect.value = "new";
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

    dirSelect.addEventListener("change", () => {
        if (dirSelect.value === "__new__") newDirInput.focus();
        applyState();
    });
    actionSelect.addEventListener("change", () => applyState({ refocusName: true }));
    baseSelect.addEventListener("change", () => {
        if (actionSelect.value === "modify_existing") {
            nameInput.value = baseSelect.value;
        }
    });

    applyState();

    let overlay;
    const submit = async () => {
        let dir = dirSelect.value;
        if (dir === "__new__") {
            dir = newDirInput.value.trim();
            if (!dir) { newDirInput.focus(); return; }
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
