// =============================================================================
// Modal + context-menu DOM. The shell takes care of the Escape-key / overlay-
// click / Cancel / OK dismissal paths and ties the document-level keydown
// listener to the overlay's lifetime so opening many modals doesn't leak.
// =============================================================================
import { compareNames, toast } from "./constants.js";
import { listDirectoryNames, dirOf } from "./workflows_store.js";
import {
    detectManager,
    loadMappings,
    resolvePicksToInstall,
    queueInstallGitUrl,
    startQueue,
    pollUntilDone,
    reboot,
} from "./installer.js";

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

// Tiny factory for the small-caps label rows above modal inputs/selects.
// Replaces the four-line `createElement` + `className` + `textContent` +
// `appendChild` block that appears 7× across the modal helpers below.
function modalLabel(text) {
    const lbl = document.createElement("label");
    lbl.className = "koolook-modal-label";
    lbl.textContent = text;
    return lbl;
}

export function showInputModal({ title, label, defaultValue, placeholder, confirmLabel, onSubmit }) {
    const body = document.createElement("div");

    body.appendChild(modalLabel(label || "Name"));

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
    //
    // The leading double-underscores on cascade sentinels (`__new__`,
    // `__here__`) are deliberate: real directory names cannot start with
    // double-underscore in any current store, so the sentinel can never
    // collide with a user-created directory's name when stored as a
    // <select>'s `value`. The action values are bare lowercase strings
    // because they're only ever compared against the action <select>'s
    // own `value`, which we control.
    const NEW_TOP = "__new__";
    const SAVE_HERE = "__here__";
    const ACTION_NEW = "new";
    const ACTION_USE_EXISTING = "use_existing";
    const ACTION_MODIFY_EXISTING = "modify_existing";
    const topNames = listDirectoryNames([]);

    body.appendChild(modalLabel("Directory"));

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
    const baseLbl = modalLabel("Base on existing");
    body.appendChild(baseLbl);

    const baseSelect = document.createElement("select");
    baseSelect.className = "koolook-modal-select";
    body.appendChild(baseSelect);

    // ---- Action ----
    const actionLbl = modalLabel("Action");
    body.appendChild(actionLbl);

    const actionSelect = document.createElement("select");
    actionSelect.className = "koolook-modal-select";
    [
        { value: ACTION_NEW, label: "New name" },
        { value: ACTION_USE_EXISTING, label: "Use existing name (archive previous)" },
        { value: ACTION_MODIFY_EXISTING, label: "Modify existing name" },
    ].forEach(a => {
        const opt = document.createElement("option");
        opt.value = a.value;
        opt.textContent = a.label;
        actionSelect.appendChild(opt);
    });
    body.appendChild(actionSelect);

    // ---- Workflow name (only shown when needed) ----
    const nameLbl = modalLabel("Workflow name");
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
                if (opt.value === ACTION_USE_EXISTING || opt.value === ACTION_MODIFY_EXISTING) {
                    opt.disabled = false;
                }
            }
        } else {
            actionLbl.style.display = "none";
            actionSelect.style.display = "none";
            // Pin the underlying value to "new" so submit takes the Workflow
            // Name path.
            if (actionSelect.value !== ACTION_NEW) actionSelect.value = ACTION_NEW;
        }

        // Workflow name — visible only when needed.
        const action = actionSelect.value;
        if (action === ACTION_USE_EXISTING) {
            // Name comes from the base; field is irrelevant.
            nameLbl.style.display = "none";
            nameInput.style.display = "none";
        } else {
            nameLbl.style.display = "";
            nameInput.style.display = "";
            if (action === ACTION_MODIFY_EXISTING && baseSelect.value) {
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
        if (actionSelect.value === ACTION_MODIFY_EXISTING) {
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
        const dirPath = getSelectedPath();
        if (!dirPath || dirPath.length === 0) {
            // Top is "__new__" with empty input — prompt for a name.
            if (cascadeSelects[0].value === NEW_TOP) newDirInput.focus();
            return;
        }
        const action = actionSelect.value;
        let name;
        if (action === ACTION_USE_EXISTING) {
            name = (baseSelect.value || "").trim();
            if (!name) { actionSelect.value = ACTION_NEW; applyState(); nameInput.focus(); return; }
        } else {
            name = nameInput.value.trim();
            if (!name) { nameInput.focus(); return; }
        }
        overlay.remove();
        // `dirPath` is a string[] of segment names (`["UP-scale", "Type-A"]`).
        // Renamed from the historical `dir` field on this contract so callers
        // match the codebase-wide convention (`dirPath` for arrays of
        // segments, `dirName` for single-segment strings, `dir` for resolved
        // DirNode objects).
        await onSave({ name, dirPath });
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

    body.appendChild(modalLabel("Current tags"));

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

    body.appendChild(modalLabel("Add tag"));

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
// Install missing nodes modal — three phases driven by ComfyUI-Manager's HTTP
// API.
//
//   1. Discovery   — probe Manager, fetch the node-id → git-url mapping,
//                    bucket picks into already-installed / will-install /
//                    unresolved.
//   2. Confirm     — show the counts + per-bucket detail; user clicks Install
//                    or Copy URL list.
//   3. Progress    — POST /customnode/install/git_url for each unique URL,
//                    POST /manager/queue/start, then poll /manager/queue/status
//                    while updating a progress bar.
//   4. Result      — summarize successes/failures and offer Reboot.
//
// We don't render the unresolved list as a hard-fail block — installs can
// still proceed for the resolved subset, and the unresolved IDs surface
// inside a collapsible <details> so the user can act on them separately.
// =============================================================================
export function showInstallMissingModal({ picks }) {
    const body = document.createElement("div");

    const message = document.createElement("div");
    message.className = "koolook-modal-message";
    message.textContent = "Checking ComfyUI-Manager…";
    body.appendChild(message);

    const stats = document.createElement("div");
    body.appendChild(stats);

    const progress = document.createElement("div");
    progress.className = "koolook-install-progress";
    progress.style.display = "none";
    const progressBar = document.createElement("div");
    progressBar.className = "koolook-install-progress-bar";
    progress.appendChild(progressBar);
    body.appendChild(progress);

    // Track the polling abort so the Cancel button (or modal dismiss) stops
    // pumping `pollUntilDone` even if installs are still chugging on the
    // server side. We can't *cancel* an install — Manager doesn't expose a
    // per-task cancel — so this just disengages our UI loop, not the work.
    const pollAbort = { aborted: false };
    let overlay;

    const closeBtn = makeModalButton({
        label: "Cancel",
        onClick: () => { pollAbort.aborted = true; overlay.remove(); },
    });
    const copyBtn = makeModalButton({ label: "Copy URL list", onClick: () => {} });
    copyBtn.style.display = "none";
    const installBtn = makeModalButton({ label: "Install via Manager", primary: true, onClick: () => {} });
    installBtn.style.display = "none";
    const rebootBtn = makeModalButton({ label: "Reboot now", primary: true, onClick: () => {} });
    rebootBtn.style.display = "none";

    ({ overlay } = makeModalShell({
        title: "Install missing nodes",
        body,
        actions: [closeBtn, copyBtn, installBtn, rebootBtn],
    }));

    function setStatLines(lines) {
        stats.innerHTML = "";
        for (const line of lines) {
            const div = document.createElement("div");
            div.className = "koolook-install-stat-line";
            if (line.fail) div.classList.add("koolook-install-stat-fail");
            div.textContent = line.text;
            stats.appendChild(div);
        }
    }

    function appendUnresolvedDetail(unresolvedIds) {
        if (unresolvedIds.length === 0) return;
        const details = document.createElement("details");
        const summary = document.createElement("summary");
        summary.className = "koolook-install-unresolved-summary";
        summary.textContent = `Show unresolved (${unresolvedIds.length})`;
        details.appendChild(summary);
        const list = document.createElement("div");
        list.className = "koolook-install-unresolved";
        list.textContent = unresolvedIds.join(", ");
        details.appendChild(list);
        stats.appendChild(details);
    }

    // ---- Phase 1: Discovery ----
    (async () => {
        const managerOk = await detectManager();
        if (!managerOk) {
            message.textContent = "ComfyUI-Manager isn't reachable. Install it via the official ComfyUI installer, or run `comfy node install <pack>` from the CLI for each missing pack.";
            return;
        }
        let mappings;
        try {
            mappings = await loadMappings();
        } catch (e) {
            console.warn("[Koolook] loadMappings failed:", e);
            message.textContent = `Could not load Manager's mapping database (${e.message}). In Manager → click "Update DB" or restart ComfyUI, then retry.`;
            return;
        }
        const result = resolvePicksToInstall(picks, mappings.urlByNodeId);
        const urlCount = result.willInstall.byUrl.size;
        const willInstallNodeCount = [...result.willInstall.byUrl.values()]
            .reduce((acc, ids) => acc + ids.length, 0);

        message.textContent = "";
        const lines = [
            { text: `Already installed: ${result.alreadyInstalled.length} pick${result.alreadyInstalled.length === 1 ? "" : "s"}` },
            { text: `Will install: ${urlCount} pack${urlCount === 1 ? "" : "s"} (${willInstallNodeCount} pick${willInstallNodeCount === 1 ? "" : "s"})` },
        ];
        if (result.unresolved.length > 0) {
            lines.push({ text: `Unresolved (no mapping found): ${result.unresolved.length}` });
        }
        setStatLines(lines);
        appendUnresolvedDetail(result.unresolved);

        if (urlCount === 0) {
            message.textContent = result.alreadyInstalled.length === picks.length && picks.length > 0
                ? "Every pick is already installed."
                : "Nothing to install.";
            closeBtn.textContent = "Close";
            return;
        }

        // ---- Phase 2: Confirm ----
        const urls = [...result.willInstall.byUrl.keys()];
        copyBtn.style.display = "";
        installBtn.style.display = "";

        copyBtn.addEventListener("click", async () => {
            try {
                await navigator.clipboard.writeText(urls.join("\n") + "\n");
                toast(`Copied ${urls.length} git URL${urls.length === 1 ? "" : "s"} to clipboard.`);
            } catch (e) {
                console.warn("[Koolook] clipboard write failed:", e);
                toast("Clipboard write failed — see console for the URL list.");
                console.log("[Koolook] git URLs:\n" + urls.join("\n"));
            }
        });

        installBtn.addEventListener("click", async () => {
            // Lock buttons so a double-click can't double-queue.
            installBtn.disabled = true;
            copyBtn.disabled = true;

            // ---- Phase 3: Progress ----
            progress.style.display = "";
            stats.style.display = "none";
            message.textContent = "Queueing installs…";

            const queueResults = [];
            for (const url of urls) {
                if (pollAbort.aborted) break;
                const r = await queueInstallGitUrl(url);
                queueResults.push({ url, ...r });
            }

            const queuedOk = queueResults.filter(r => r.ok).length;
            if (queuedOk === 0) {
                progress.style.display = "none";
                stats.style.display = "";
                message.textContent = "No installs were accepted.";
                setStatLines(queueResults.map(r => ({
                    text: `${r.url} — ${r.message}`,
                    fail: true,
                })));
                closeBtn.textContent = "Close";
                installBtn.style.display = "none";
                copyBtn.style.display = "none";
                return;
            }

            await startQueue();
            const finalStatus = await pollUntilDone({
                onTick: (s) => {
                    const pct = s.total_count > 0 ? (s.done_count / s.total_count) * 100 : 0;
                    progressBar.style.width = `${pct}%`;
                    message.textContent = `Installing… ${s.done_count} of ${s.total_count}`;
                },
                signal: pollAbort,
            });

            // ---- Phase 4: Result ----
            progress.style.display = "none";
            stats.style.display = "";
            installBtn.style.display = "none";
            copyBtn.style.display = "none";

            const resultLines = [];
            if (finalStatus) {
                resultLines.push({
                    text: `Installed ${finalStatus.done_count} of ${finalStatus.total_count} pack${finalStatus.total_count === 1 ? "" : "s"}.`,
                });
            } else {
                resultLines.push({ text: "Stopped polling — installs may still complete in the background." });
            }
            for (const r of queueResults.filter(r => !r.ok)) {
                resultLines.push({ text: `Failed to queue: ${r.url} — ${r.message}`, fail: true });
            }
            if (queuedOk > 0) {
                resultLines.push({ text: "Restart ComfyUI to load the newly installed nodes." });
            }
            message.textContent = "";
            setStatLines(resultLines);
            appendUnresolvedDetail(result.unresolved);

            closeBtn.textContent = "Close";
            if (queuedOk > 0) {
                rebootBtn.style.display = "";
                rebootBtn.addEventListener("click", async () => {
                    rebootBtn.disabled = true;
                    await reboot();
                    overlay.remove();
                    toast("Reboot requested. Reload the page in a few seconds.");
                });
            }
        });
    })();
}

// =============================================================================
// Snapshot / preset library — three separate dialogs (Save / Load / Settings),
// one per toolbar button. No tabs, no shared state across them.
//
// Capability functions are passed in by the caller instead of imported here,
// so this file stays UI-only and the snapshot logic stays in its own module
// (`./snapshot.js`).
// =============================================================================

// Local helper: today's date in YYYY-MM-DD form for the default name.
function todayStamp() {
    return new Date().toISOString().slice(0, 10);
}

// Local helper: render a preset's metadata line (used by the Load dialog).
function formatPreviewMeta(p) {
    const parts = [];
    parts.push(`${p.workflowCount} workflow${p.workflowCount === 1 ? "" : "s"}`);
    parts.push(`${p.pickCount} pick${p.pickCount === 1 ? "" : "s"}`);
    if (p.exportedAt) {
        try {
            const d = new Date(p.exportedAt);
            if (!isNaN(d.getTime())) parts.push(`exported ${d.toLocaleDateString()}`);
        } catch (e) { /* noop */ }
    }
    return parts.join(" · ");
}

// Local helper: a 3-button confirm modal (Save / Save as new / Cancel) for
// the Save flow when there's a current preset to overwrite. `showConfirmModal`
// is 2-button only; we'd need to bend it. Cleaner to assemble directly with
// `makeModalShell`.
function showThreeWayDialog({ title, message, primaryLabel, secondaryLabel, cancelLabel, onPrimary, onSecondary }) {
    const body = document.createElement("div");
    const msg = document.createElement("div");
    msg.className = "koolook-modal-message";
    msg.textContent = message;
    body.appendChild(msg);

    let overlay;
    const close = () => overlay.remove();

    const cancel = makeModalButton({
        label: cancelLabel || "Cancel",
        onClick: close,
    });
    const secondary = makeModalButton({
        label: secondaryLabel,
        onClick: () => { close(); onSecondary(); },
    });
    const primary = makeModalButton({
        label: primaryLabel,
        primary: true,
        onClick: () => { close(); onPrimary(); },
    });
    ({ overlay } = makeModalShell({
        title,
        body,
        actions: [cancel, secondary, primary],
    }));
}

// =============================================================================
// Save flow.
//
// One click semantics:
//   - If a current preset is tracked → confirm "Save over <name>?" with
//     three options (overwrite / rename / cancel).
//   - If not → straight to a name prompt with `preset YYYY-MM-DD` default.
//
// Save success → updates the tracker so subsequent Save clicks default
// to overwrite-the-same-one.
// =============================================================================
export function showSaveSnapshotDialog({
    getCurrentPresetName,
    setCurrentPresetName,
    presetExists,
    writePreset,
    gatherSnapshot,
    sanitizeName,
    onToast,
}) {
    const toast = onToast || (() => {});

    async function doSave(name, { confirmOverwrite }) {
        const sanitized = sanitizeName(name);
        if (!sanitized) {
            toast("Snapshot name is empty after stripping unsafe characters.");
            return;
        }
        // For Save-as-new only: check existence + prompt to overwrite.
        // The "overwrite current" path skipped this prompt by design.
        if (confirmOverwrite) {
            let exists = false;
            try { exists = await presetExists(sanitized); }
            catch (e) { /* assume not, fall through */ }
            if (exists) {
                showConfirmModal({
                    title: "Overwrite existing preset?",
                    message: `A snapshot named "${sanitized}" already exists. Overwrite it?`,
                    confirmLabel: "Overwrite",
                    danger: true,
                    onConfirm: () => writeAndToast(sanitized),
                });
                return;
            }
        }
        writeAndToast(sanitized);
    }

    async function writeAndToast(name) {
        try {
            const snap = gatherSnapshot(name);
            await writePreset(name, snap);
            setCurrentPresetName(name);
            toast(`Saved "${name}".`);
        } catch (e) {
            console.error("[Koolook] preset save failed:", e);
            toast(`Could not save preset: ${e.message}`);
        }
    }

    function promptForName(defaultName) {
        showInputModal({
            title: "Save snapshot",
            label: "Snapshot name",
            defaultValue: defaultName,
            placeholder: "e.g. Wan video kit",
            confirmLabel: "Save",
            onSubmit: (typed) => doSave(typed, { confirmOverwrite: true }),
        });
    }

    const current = getCurrentPresetName();
    if (current) {
        showThreeWayDialog({
            title: "Save snapshot",
            message: `Save current state over "${current}"?`,
            primaryLabel: "Save",
            secondaryLabel: "Save as new…",
            cancelLabel: "Cancel",
            onPrimary: () => doSave(current, { confirmOverwrite: false }),
            onSecondary: () => promptForName(`${current} (copy)`),
        });
    } else {
        promptForName(`preset ${todayStamp()}`);
    }
}

// =============================================================================
// Load flow.
//
// Single dialog with the preset list. Each row click loads (with confirm),
// each row's × button deletes (with confirm). Top of the dialog shows the
// current library path so the user can see where they're loading from.
//
// Load success → updates the tracker so a subsequent Save defaults to
// overwriting that preset.
// =============================================================================
export function showLoadSnapshotDialog({
    listPresets,
    readPreset,
    deletePreset,
    applySnapshot,
    setCurrentPresetName,
    getLibraryInfo,
    onToast,
}) {
    const toast = onToast || (() => {});
    const body = document.createElement("div");

    const pathLine = document.createElement("div");
    pathLine.style.cssText = "font-size: 11px; opacity: 0.55; margin-bottom: 10px; word-break: break-all;";
    pathLine.textContent = "Library: (loading…)";
    body.appendChild(pathLine);

    const listWrap = document.createElement("div");
    body.appendChild(listWrap);

    let overlay;
    const close = () => overlay.remove();

    function renderEmpty(text) {
        const el = document.createElement("div");
        el.className = "koolook-snapshot-empty";
        el.textContent = text;
        return el;
    }

    async function refresh() {
        listWrap.innerHTML = "";
        listWrap.appendChild(renderEmpty("Loading…"));
        let previews;
        try { previews = await listPresets(); }
        catch (e) { previews = []; }
        listWrap.innerHTML = "";
        if (previews.length === 0) {
            listWrap.appendChild(renderEmpty("No presets in this library yet. Use Save to create one."));
            return;
        }
        const list = document.createElement("div");
        list.className = "koolook-snapshot-list";
        for (const p of previews) {
            const row = document.createElement("div");
            row.className = "koolook-snapshot-row";

            const info = document.createElement("div");
            info.className = "koolook-snapshot-row-info";
            info.title = `Click to load "${p.displayName}"`;
            const name = document.createElement("div");
            name.className = "koolook-snapshot-row-name";
            name.textContent = p.displayName;
            info.appendChild(name);
            const meta = document.createElement("div");
            meta.className = "koolook-snapshot-row-meta";
            meta.textContent = formatPreviewMeta(p);
            info.appendChild(meta);
            info.addEventListener("click", () => promptApply(p));
            row.appendChild(info);

            const actions = document.createElement("div");
            actions.className = "koolook-snapshot-row-actions";
            const delBtn = document.createElement("button");
            delBtn.className = "koolook-snapshot-row-btn koolook-snapshot-row-btn-danger";
            delBtn.textContent = "×";
            delBtn.title = `Delete "${p.displayName}"`;
            delBtn.addEventListener("click", (e) => { e.stopPropagation(); promptDelete(p); });
            actions.appendChild(delBtn);
            row.appendChild(actions);

            list.appendChild(row);
        }
        listWrap.appendChild(list);
    }

    function promptApply(preview) {
        showConfirmModal({
            title: "Replace current state?",
            message:
                `Loading "${preview.displayName}" will replace your current ` +
                `picks and workflows with the snapshot's contents (` +
                `${formatPreviewMeta(preview)}). This cannot be undone.`,
            confirmLabel: "Load preset",
            danger: true,
            onConfirm: async () => {
                try {
                    const snap = await readPreset(preview.fileName);
                    const { picksOk, workflowsOk } = await applySnapshot(snap);
                    setCurrentPresetName(preview.fileName);
                    close();
                    if (picksOk && workflowsOk) {
                        toast(`Loaded preset "${preview.displayName}".`);
                    } else if (picksOk || workflowsOk) {
                        toast(
                            `Loaded "${preview.displayName}" — partial: ` +
                            `picks ${picksOk ? "OK" : "FAIL"}, ` +
                            `workflows ${workflowsOk ? "OK" : "FAIL"}. See console.`
                        );
                    } else {
                        toast(`Loaded "${preview.displayName}" in memory but persist failed. See console.`);
                    }
                } catch (e) {
                    console.error("[Koolook] preset load failed:", e);
                    toast(`Could not load "${preview.displayName}": ${e.message}`);
                }
            },
        });
    }

    function promptDelete(preview) {
        showConfirmModal({
            title: "Delete preset?",
            message: `"${preview.displayName}" will be removed from the library. This cannot be undone.`,
            confirmLabel: "Delete",
            danger: true,
            onConfirm: async () => {
                try {
                    await deletePreset(preview.fileName);
                    toast(`Deleted "${preview.displayName}".`);
                    await refresh();
                } catch (e) {
                    console.error("[Koolook] preset delete failed:", e);
                    toast(`Could not delete "${preview.displayName}": ${e.message}`);
                }
            },
        });
    }

    const closeBtn = makeModalButton({ label: "Close", onClick: close });
    ({ overlay } = makeModalShell({
        title: "Load snapshot",
        body,
        actions: [closeBtn],
    }));
    refresh();
    if (typeof getLibraryInfo === "function") {
        getLibraryInfo().then((info) => {
            if (info) pathLine.textContent = `Library: ${info.path}`;
            else pathLine.textContent = "Library path unavailable.";
        }).catch(() => { pathLine.textContent = "Library path unavailable."; });
    } else {
        pathLine.textContent = "";
    }
}

// =============================================================================
// Settings flow.
//
// Single dialog with one editable field (library path). Two action buttons:
// Save (writes the path to the server-side settings file) and Reset to
// default (clears the saved path; server falls back to env var or built-in
// default). A read-only line shows the currently-resolved path + source so
// the user can see what's actually in effect right now.
// =============================================================================
export function showSnapshotSettingsDialog({
    getSettings,
    saveSettings,
    onToast,
}) {
    const toast = onToast || (() => {});
    const body = document.createElement("div");

    body.appendChild(modalLabel("Library path"));

    const input = document.createElement("input");
    input.className = "koolook-modal-input";
    input.placeholder = "/absolute/path/to/library  (leave empty to use env-var or default)";
    input.disabled = true;
    body.appendChild(input);

    const hint = document.createElement("div");
    hint.style.cssText = "font-size: 11px; opacity: 0.55; margin: 8px 0 12px; line-height: 1.5; word-break: break-all;";
    hint.textContent = "Loading current settings…";
    body.appendChild(hint);

    let overlay;
    const close = () => overlay.remove();

    async function refresh() {
        let settings;
        try { settings = await getSettings(); }
        catch (e) { settings = null; }
        if (!settings) {
            hint.textContent =
                "Settings endpoint unavailable — server-side routes not registered. " +
                "Restart ComfyUI to fix.";
            return;
        }
        input.value = settings.savedLibraryPath || "";
        input.disabled = false;
        let sourceLabel;
        if (settings.source === "settings") sourceLabel = "from this Settings panel";
        else if (settings.source === "env") sourceLabel = `from ${settings.envVar} env var`;
        else sourceLabel = "built-in default";
        hint.textContent =
            `Currently in effect: ${settings.resolvedPath}\n(source: ${sourceLabel})`;
        hint.style.whiteSpace = "pre-line";
    }

    async function doSave(rawValue) {
        try {
            const result = await saveSettings(rawValue);
            const sourceLabel = result.source === "settings"
                ? "from this Settings panel"
                : (result.source === "env" ? "from env var" : "built-in default");
            hint.textContent =
                `Currently in effect: ${result.resolvedPath}\n(source: ${sourceLabel})`;
            input.value = result.savedLibraryPath || "";
            toast(rawValue
                ? "Library path saved."
                : "Override cleared — using env var / default."
            );
        } catch (e) {
            console.error("[Koolook] settings save failed:", e);
            toast(`Could not save settings: ${e.message}`);
        }
    }

    refresh();

    const cancel = makeModalButton({ label: "Close", onClick: close });
    const reset = makeModalButton({
        label: "Reset to default",
        onClick: () => doSave(""),
    });
    const save = makeModalButton({
        label: "Save",
        primary: true,
        onClick: () => doSave(input.value.trim()),
    });
    input.addEventListener("keydown", (e) => {
        if (e.key === "Enter") doSave(input.value.trim());
    });

    ({ overlay } = makeModalShell({
        title: "Snapshot library settings",
        body,
        actions: [cancel, reset, save],
    }));
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
