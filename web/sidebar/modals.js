// SPDX-FileCopyrightText: 2025-2026 Kforge Labs <https://github.com/malkuthro/ComfyUI-Koolook>
// SPDX-License-Identifier: GPL-3.0-only

// =============================================================================
// Modal + context-menu DOM. The shell takes care of the Escape-key / overlay-
// click / Cancel / OK dismissal paths and ties the document-level keydown
// listener to the overlay's lifetime so opening many modals doesn't leak.
// =============================================================================
import { compareNames, criticalToast, formatLibraryPathBreadcrumb, toast } from "./constants.js";
import { formatLocalStamp } from "./format_time.js";
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

function makeModalShell({ title, titleTooltip, body, actions }) {
    const overlay = document.createElement("div");
    overlay.className = "koolook-modal-overlay";

    const modal = document.createElement("div");
    modal.className = "koolook-modal";

    const titleEl = document.createElement("div");
    titleEl.className = "koolook-modal-title";
    titleEl.textContent = title;
    // Optional hover-tooltip for explanatory context. Per
    // ``docs/maintainers/conventions.md``: header sections stay clean
    // (title only, no descriptive subtitle); use this tooltip for the
    // one-sentence "what does this dialog do" gloss instead. Functional
    // info rows (library-path indicator, etc.) live in the body, not
    // the header — they are not the same thing as a description.
    if (titleTooltip) titleEl.title = titleTooltip;
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
    // Click-to-dismiss with drag-out-of-input protection. A single click
    // fires `mousedown` then `mouseup` then `click` — and `click.target`
    // is the deepest common ancestor of mousedown's and mouseup's targets.
    // So if you drag-select text inside an input and release in the
    // overlay's dark area, click fires with target=overlay and the modal
    // would otherwise close mid-edit. Track whether the gesture STARTED
    // on the overlay too; only dismiss if both ends did.
    let mouseDownOnOverlay = false;
    overlay.addEventListener("mousedown", (e) => {
        mouseDownOnOverlay = (e.target === overlay);
    });
    overlay.addEventListener("click", (e) => {
        if (e.target === overlay && mouseDownOnOverlay) close();
        mouseDownOnOverlay = false;
    });
    escHandler = (e) => { if (e.key === "Escape") close(); };
    document.addEventListener("keydown", escHandler);
    document.body.appendChild(overlay);
    return { overlay, modal, titleEl, close };
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

export function showInputModal({ title, label, defaultValue, placeholder, confirmLabel, onSubmit, subtitle }) {
    const body = document.createElement("div");

    // Optional subtitle — string (rendered in the standard pathline style) or
    // a pre-built HTMLElement (caller controls styling AND can mutate after
    // the modal is open, e.g. to fill in an asynchronously-fetched library
    // path). Sits between title and label so the user sees context first.
    if (subtitle) {
        if (typeof subtitle === "string") {
            const sub = document.createElement("div");
            sub.className = "koolook-modal-pathline";
            sub.textContent = subtitle;
            body.appendChild(sub);
        } else {
            body.appendChild(subtitle);
        }
    }

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

export function showConfirmModal({ title, message, confirmLabel, cancelLabel, danger, onConfirm, onCancel, subtitle }) {
    const body = document.createElement("div");

    // Optional subtitle — same contract as showInputModal. Renders above
    // the message so callers can surface context (e.g. the library path
    // breadcrumb on the Save-as-new overwrite-confirm path) before the
    // user scans the action question. The redesigned Save / Load dialogs
    // surface that breadcrumb inline in their own body, but other callers
    // still benefit from the slot.
    if (subtitle) {
        if (typeof subtitle === "string") {
            const sub = document.createElement("div");
            sub.className = "koolook-modal-pathline";
            sub.textContent = subtitle;
            body.appendChild(sub);
        } else {
            body.appendChild(subtitle);
        }
    }

    const msg = document.createElement("div");
    msg.className = "koolook-modal-message";
    if (message && typeof message === "object" && typeof message.nodeType === "number") {
        msg.appendChild(message);
    } else {
        msg.textContent = message;
    }
    body.appendChild(msg);

    let overlay;
    // `onCancel` fires once for EVERY non-OK dismissal: Cancel button,
    // Escape, AND overlay-click. Promise-wrapping callers (recovery
    // toast's "Discard offline copy", `dropPlaceholdersForPacks` from
    // issue #84) need every dismissal path to settle the Promise — an
    // earlier cut wired the Cancel button only, so Esc / click-outside
    // still leaked. The `settled` flag keeps `onCancel` idempotent so
    // the Cancel-button path doesn't double-fire when overlay teardown
    // re-enters via the wrapped `overlay.remove`. Optional; existing
    // callers that don't pass `onCancel` keep working unchanged.
    let settled = false;
    const settleCancel = () => {
        if (settled) return;
        settled = true;
        if (typeof onCancel !== "function") return;
        try { onCancel(); }
        catch (e) { console.error("[Koolook] confirm modal onCancel failed:", e); }
    };

    const cancel = makeModalButton({
        label: cancelLabel || "Cancel",
        onClick: () => { settleCancel(); overlay.remove(); },
    });
    const ok = makeModalButton({
        label: confirmLabel || "OK",
        primary: !danger,
        danger,
        // Mark settled BEFORE removing the overlay so the wrapped
        // `overlay.remove` below doesn't fire `onCancel` on the OK path.
        onClick: () => { settled = true; overlay.remove(); onConfirm(); },
    });
    ({ overlay } = makeModalShell({ title, body, actions: [cancel, ok] }));

    // Escape and overlay-click route through `makeModalShell.close()`,
    // which calls `overlay.remove()` (the listener-cleanup override) but
    // is unaware of `onCancel`. Re-wrap to settle the cancel callback
    // before the underlying teardown runs.
    const baseRemove = overlay.remove.bind(overlay);
    overlay.remove = () => {
        settleCancel();
        baseRemove();
    };
}

export function showSaveWorkflowModal({ titleSuffix, defaultName, defaultDir, defaultModule = false, onSave }) {
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

    // ---- "Save as module" checkbox ----
    // Tagging the saved entry with the literal `module` tag flips the
    // sidebar's left-click default from Load to Insert, and re-icons the
    // row green. The actual `addTag` call lives in the caller's `onSave`
    // handler so it can ride the same persistMutation snapshot as the
    // entry write itself — a commit failure rolls back BOTH the save and
    // the tag in one go. Visually, the checkbox sits below the name field
    // (the last thing the user touches before clicking Save) so the
    // intent decision is co-located with the destination decision.
    const moduleRow = document.createElement("label");
    moduleRow.className = "koolook-modal-checkbox-row";
    const moduleCheckbox = document.createElement("input");
    moduleCheckbox.type = "checkbox";
    moduleCheckbox.checked = !!defaultModule;
    moduleRow.appendChild(moduleCheckbox);
    const moduleText = document.createElement("span");
    moduleText.textContent = "Save as module (left-click inserts into canvas instead of replacing)";
    moduleRow.appendChild(moduleText);
    body.appendChild(moduleRow);

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
        // `asModule` is the "Save as module" checkbox state. Caller is
        // expected to fold an `addTag(..., MODULE_TAG)` into the same
        // persistMutation as the save when truthy, so a commit failure
        // rolls back both the entry write and the tag together.
        await onSave({ name, dirPath, asModule: moduleCheckbox.checked });
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

// Local helper: render a preset's metadata line (used by the Load dialog
// for both regular snapshot rows AND recovery auto-save rows). Prefers
// `mtime` (Unix epoch seconds from the server-side stat) over the
// snapshot's self-reported `exportedAt` because (1) mtime survives
// out-of-band file copies in Finder while exportedAt is frozen at
// gather-time, and (2) "saved <local time>" is what the user actually
// wants to see for "which row is newest". Falls back to `exportedAt`
// for legacy rows whose listing endpoint didn't carry mtime.
function formatPreviewMeta(p) {
    const parts = [];
    parts.push(`${p.pickCount} pick${p.pickCount === 1 ? "" : "s"}`);
    parts.push(`${p.workflowCount} workflow${p.workflowCount === 1 ? "" : "s"}`);
    if (typeof p.mtime === "number" && isFinite(p.mtime)) {
        const stamp = formatLocalStamp(new Date(p.mtime * 1000));
        if (stamp) parts.push(p.kind ? stamp : `saved ${stamp}`);
    } else if (p.exportedAt) {
        const stamp = formatLocalStamp(new Date(p.exportedAt));
        if (stamp) parts.push(p.kind ? stamp : `exported ${stamp}`);
    }
    return parts.join(" · ");
}

function pathLeaf(path) {
    const parts = String(path || "").split(/[\\/]+/).filter(Boolean);
    return parts.length ? parts[parts.length - 1] : String(path || "");
}

// =============================================================================
// Save flow — redesigned per issue #137, mockup section 2.
//
// One unified dialog (was three: three-way overwrite / input prompt / confirm).
// Layout:
//
//   [ Save snapshot                                            ]
//   [ Saved to                              Open folder ↗      ]
//   [ <leaf folder>                                            ]
//   [ <full path>                                              ]
//   [                                                          ]
//   [ Save current state over "<name>"?                        ]
//   [                                                          ]
//   [ [ Save to… ]            [ Cancel ] [ Save as new… ] [Save]]
//
// The Save button cycles through four states per mockup section 5:
// Default ("Save") → In progress ("Saving…", disabled) → Done ("Saved",
// subtle disabled). When the user changes the library via Save to…, the
// dialog enters Dirty state — Save label flips to "Save to new folder"
// so the click reads as "you're about to write into a different place."
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
    getLibraryInfo,
    saveSettings,
    browseDirectories,
    createBrowseDirectory,
    revealPresetFolder,
    markStateSaved,
    onToast,
}) {
    const toast = onToast || (() => {});
    const current = getCurrentPresetName();

    // Internal state: original library path (set after the first
    // getLibraryInfo() resolution) and current library path (mutated when
    // the user picks a different folder via Save to…). The two diverge
    // → Dirty state → primary label becomes "Save to new folder".
    let originalLibraryPath = "";
    let currentLibraryPath = "";
    let libraryRevealInFlight = false;

    const body = document.createElement("div");

    // -----------------------------------------------------------------
    // Library-row info block (top of body). Pure information — the
    // mockup explicitly puts every action in the bottom command bar.
    // Async-populated so the dialog opens immediately without waiting
    // on a server round-trip; the path appears in place when ready.
    // -----------------------------------------------------------------
    const libRow = document.createElement("div");
    libRow.className = "koolook-snap-lib-row";

    const libRowInfo = document.createElement("div");
    libRowInfo.className = "koolook-snap-lib-row-info";

    const libRowTop = document.createElement("div");
    libRowTop.className = "koolook-snap-lib-row-top";
    const libLabel = document.createElement("span");
    libLabel.className = "koolook-snap-lib-label";
    libLabel.textContent = "Saved to";
    libRowTop.appendChild(libLabel);

    const openFolderLink = document.createElement("a");
    openFolderLink.className = "koolook-snap-open-folder-link";
    openFolderLink.href = "#";
    openFolderLink.textContent = "Open folder ↗";
    openFolderLink.title = "Open the snapshot library folder in your file manager";
    openFolderLink.addEventListener("click", async (e) => {
        e.preventDefault();
        if (typeof revealPresetFolder !== "function") return;
        if (libraryRevealInFlight) return;
        libraryRevealInFlight = true;
        try {
            const r = await revealPresetFolder();
            toast(`Opened: ${r.path}`);
        } catch (err) {
            console.error("[Koolook] reveal failed:", err);
            toast(`Could not open library folder: ${err.message}`);
        } finally {
            libraryRevealInFlight = false;
        }
    });
    libRowTop.appendChild(openFolderLink);
    libRowInfo.appendChild(libRowTop);

    const libName = document.createElement("div");
    libName.className = "koolook-settings-folder-name";
    libName.textContent = "Library folder: (loading…)";
    libRowInfo.appendChild(libName);

    const libPath = document.createElement("div");
    libPath.className = "koolook-settings-folder-path";
    libRowInfo.appendChild(libPath);
    libRow.appendChild(libRowInfo);

    body.appendChild(libRow);

    function renderLibRow(path) {
        currentLibraryPath = path || "";
        const leaf = currentLibraryPath ? pathLeaf(currentLibraryPath) : "(unavailable)";
        libName.textContent = leaf;
        libPath.textContent = currentLibraryPath || "Path unavailable";
        libRow.title = currentLibraryPath || "";
        updatePrimaryLabel();
    }

    if (typeof getLibraryInfo === "function") {
        getLibraryInfo().then((info) => {
            const path = (info && typeof info.path === "string") ? info.path : "";
            originalLibraryPath = path;
            renderLibRow(path);
        }).catch(() => {
            libName.textContent = "Library path unavailable";
            libPath.textContent = "";
        });
    }

    // -----------------------------------------------------------------
    // Message line — "Save current state over '<name>'?" if there's a
    // tracked preset, or "Save current state as a new preset" if not.
    // -----------------------------------------------------------------
    const msg = document.createElement("p");
    msg.className = "koolook-modal-message koolook-snap-save-message";
    msg.textContent = current
        ? `Save current state over "${current}"?`
        : "Save current state as a new preset.";
    body.appendChild(msg);

    // -----------------------------------------------------------------
    // Primary "Save" button — four-state per mockup section 5. The
    // helper centralises label / class / disabled so the dialog only
    // toggles state names, never DOM attributes directly.
    // -----------------------------------------------------------------
    let primaryState = current ? "default" : "default-new";
    const primaryBtn = document.createElement("button");
    primaryBtn.className = "koolook-modal-btn";

    function isDirty() {
        return Boolean(
            originalLibraryPath &&
            currentLibraryPath &&
            originalLibraryPath !== currentLibraryPath
        );
    }

    function updatePrimaryLabel() {
        // ``dirty`` only meaningfully changes behaviour when there's a tracked
        // preset to overwrite — otherwise the primary action is "ask for a
        // name and write" regardless of the destination folder. Without this
        // guard, opening Save with no tracked preset, then changing the
        // folder via Save to…, would enter the "dirty" branch with
        // ``current === null`` and the primary click would call
        // ``doOverwrite(null)`` → ``sanitizeName(null)`` instead of routing
        // to the Save-as-new name prompt.
        if (!current) {
            primaryState = "default-new";
        } else {
            primaryState = isDirty() ? "dirty" : "default";
        }
        renderPrimary();
    }

    function renderPrimary() {
        primaryBtn.classList.remove(
            "koolook-modal-btn-primary",
            "koolook-snap-save-in-progress",
            "koolook-snap-save-done",
        );
        switch (primaryState) {
            case "default":
                primaryBtn.textContent = "Save";
                primaryBtn.classList.add("koolook-modal-btn-primary");
                primaryBtn.disabled = false;
                primaryBtn.title = `Overwrite "${current}" with the current state.`;
                break;
            case "default-new":
                // No tracked preset → Save needs a name → routes to Save as new.
                primaryBtn.textContent = "Save";
                primaryBtn.classList.add("koolook-modal-btn-primary");
                primaryBtn.disabled = false;
                primaryBtn.title = "Choose a name and write the snapshot.";
                break;
            case "dirty":
                primaryBtn.textContent = "Save to new folder";
                primaryBtn.classList.add("koolook-modal-btn-primary");
                primaryBtn.disabled = false;
                primaryBtn.title = `Write into ${currentLibraryPath}.`;
                break;
            case "in-progress":
                primaryBtn.textContent = "Saving…";
                primaryBtn.classList.add(
                    "koolook-modal-btn-primary",
                    "koolook-snap-save-in-progress",
                );
                primaryBtn.disabled = true;
                primaryBtn.title = "";
                break;
            case "done":
                primaryBtn.textContent = "Saved";
                primaryBtn.classList.add("koolook-snap-save-done");
                primaryBtn.disabled = true;
                primaryBtn.title = "";
                break;
        }
    }
    renderPrimary();

    // -----------------------------------------------------------------
    // Footer buttons.
    // -----------------------------------------------------------------
    const saveToBtn = makeModalButton({
        label: "Save to…",
        onClick: () => openSaveTo(),
    });

    const cancelBtn = makeModalButton({
        label: "Cancel",
        onClick: () => close(),
    });

    const saveAsNewBtn = makeModalButton({
        label: "Save as new…",
        onClick: () => {
            promptForName(current ? `${current} (copy)` : `preset ${todayStamp()}`);
        },
    });

    primaryBtn.addEventListener("click", async () => {
        if (primaryBtn.disabled) return;
        if (primaryState === "default-new") {
            promptForName(`preset ${todayStamp()}`);
            return;
        }
        // "default" or "dirty" — write over the tracked preset (dirty
        // means the library path changed; the preset *name* is the same).
        await doOverwrite(current);
    });

    const spacer = document.createElement("span");
    spacer.className = "koolook-folder-picker-spacer";

    let overlay;
    const close = () => overlay.remove();

    // -----------------------------------------------------------------
    // Save to… → folder picker. On Use this folder, persist via
    // saveSettings() and update the in-dialog state. The original path
    // is kept (in `originalLibraryPath`) so the dirty indicator stays
    // accurate even after navigating multiple times.
    // -----------------------------------------------------------------
    function openSaveTo() {
        if (typeof browseDirectories !== "function") {
            toast("Folder picker unavailable in this session.");
            return;
        }
        if (saveToBtn.disabled) return;
        saveToBtn.disabled = true;
        setTimeout(() => { saveToBtn.disabled = false; }, 0);
        showFolderPicker({
            title: "Save snapshots to",
            titleTooltip: "Pick the library folder this Save will write into.",
            initialPath: currentLibraryPath,
            browseDirectories,
            createBrowseDirectory,
            onUseFolder: async (chosen) => {
                if (typeof saveSettings !== "function") {
                    renderLibRow(chosen);
                    return;
                }
                try {
                    const r = await saveSettings(chosen);
                    renderLibRow(r.savedLibraryPath || chosen);
                    toast(`Snapshot library set to: ${currentLibraryPath}.`);
                } catch (err) {
                    console.error("[Koolook] saveSettings failed:", err);
                    toast(`Could not save library path: ${err.message}`);
                }
            },
        });
    }

    // -----------------------------------------------------------------
    // Overwrite the tracked preset directly. The mockup intentionally
    // skips a second "are you sure?" — the dialog already says the
    // destination preset name and (if dirty) the folder change, so the
    // click is informed.
    // -----------------------------------------------------------------
    async function doOverwrite(name) {
        primaryState = "in-progress";
        renderPrimary();
        try {
            const sanitized = sanitizeName(name);
            const snap = gatherSnapshot(sanitized);
            await writePreset(sanitized, snap);
            setCurrentPresetName(sanitized);
            if (typeof markStateSaved === "function") markStateSaved();
            primaryState = "done";
            renderPrimary();
            toast(`Saved "${sanitized}".`);
            // Brief "Saved" pause for legibility, then close — per the
            // four-state pattern, the user sees the confirmation tag
            // before the dialog disappears.
            setTimeout(close, 600);
        } catch (e) {
            console.error("[Koolook] preset save failed:", e);
            toast(`Could not save preset: ${e.message}`);
            updatePrimaryLabel();
        }
    }

    // -----------------------------------------------------------------
    // Save as new… → name prompt → existence check → write. The
    // existence-check guard lives here (not in `doOverwrite`) because
    // overwrite intent is explicit; rename intent is not.
    // -----------------------------------------------------------------
    function promptForName(defaultName) {
        showInputModal({
            title: "Save snapshot",
            label: "Snapshot name",
            defaultValue: defaultName,
            placeholder: "e.g. Wan video kit",
            confirmLabel: "Save",
            onSubmit: async (typed) => {
                const sanitized = sanitizeName(typed);
                if (!sanitized) {
                    toast("Snapshot name is empty after stripping unsafe characters.");
                    return;
                }
                const exists = await presetExists(sanitized);
                if (exists === null) {
                    toast(
                        "Cannot reach the preset library to verify name. " +
                        "Save canceled — check the library path or your connection."
                    );
                    return;
                }
                if (exists === true) {
                    showConfirmModal({
                        title: "Overwrite existing preset?",
                        message: `A snapshot named "${sanitized}" already exists. Overwrite it?`,
                        confirmLabel: "Overwrite",
                        danger: true,
                        onConfirm: () => writeAsNew(sanitized),
                    });
                    return;
                }
                writeAsNew(sanitized);
            },
        });
    }

    async function writeAsNew(name) {
        primaryState = "in-progress";
        renderPrimary();
        try {
            const snap = gatherSnapshot(name);
            await writePreset(name, snap);
            setCurrentPresetName(name);
            if (typeof markStateSaved === "function") markStateSaved();
            primaryState = "done";
            renderPrimary();
            toast(`Saved "${name}".`);
            setTimeout(close, 600);
        } catch (e) {
            console.error("[Koolook] preset save failed:", e);
            toast(`Could not save preset: ${e.message}`);
            updatePrimaryLabel();
        }
    }

    ({ overlay } = makeModalShell({
        title: "Save snapshot",
        titleTooltip: "Write the current sidebar state to a preset file.",
        body,
        actions: [saveToBtn, spacer, cancelBtn, saveAsNewBtn, primaryBtn],
    }));
}

// =============================================================================
// Load flow — redesigned per issue #137, mockup section 3.
//
// Single dialog. Layout:
//
//   [ Load snapshot                                            ]
//   [ Loaded from                            Open folder ↗     ]
//   [ <leaf folder>                                            ]
//   [ <full path>                                              ]
//   [                                                          ]
//   [ <preset row>             [×]                             ]
//   [ <preset row>             [×]                             ]
//   [ ↓ when a preset has a newer autosave, clicking it expands ]
//   [   exactly ONE scoped recovery row beneath that preset:   ]
//   [   [Pre-load badge] May 12 14:47 · 11 picks · 6 wf        ]
//   [                                                          ]
//   [ [ Load from… ]                              [ Close ]    ]
//
// Click flow:
//   * Preset row, no newer autosave → load directly (with pre-load backup).
//   * Preset row WITH a newer autosave → scope the recovery section to that
//     preset alone, render one row (newest of periodic / pre_load_*),
//     insert it directly below the clicked preset. The user then chooses:
//     click the named row again to load named, or click the recovery row
//     to load the autosave.
//   * Recovery row → restore that autosave (with pre-load backup).
//
// Delete (×) is inline: clicking × outlines the target row red and the
// command-bar Close button transforms to "Yes (delete)". A second click on
// Yes commits; Escape cancels and reverts.
//
// Load from… opens the folder picker (commit 2). On Use this folder, the
// dialog persists the path via saveSettings() and refreshes the preset
// listing.
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
    getCurrentPresetName,
    getLibraryInfo,
    saveSettings,
    browseDirectories,
    createBrowseDirectory,
    writePreLoadAutosave,
    markStateSaved,
    markStateAutosaved,
    listAutosaves,
    revealPresetFolder,
    onToast,
    // Compare mode (#181): when provided, selecting a preset (or an autosave)
    // calls onChoose(snapshot, meta) and closes — instead of the destructive
    // applySnapshot load. Default undefined keeps the normal Load flow intact.
    onChoose,
}) {
    const toast = onToast || (() => {});
    const body = document.createElement("div");
    let libraryPath = "";

    // ---- Library row (mockup section 3, "Loaded from") ----
    // Pure information row at the top. Label + Open folder link sit on the
    // top edge so a long absolute path can never cover the link area.
    const libRow = document.createElement("div");
    libRow.className = "koolook-snap-lib-row";

    const libRowInfo = document.createElement("div");
    libRowInfo.className = "koolook-snap-lib-row-info";

    const libRowTop = document.createElement("div");
    libRowTop.className = "koolook-snap-lib-row-top";
    const libLabel = document.createElement("span");
    libLabel.className = "koolook-snap-lib-label";
    libLabel.textContent = "Loaded from";
    libRowTop.appendChild(libLabel);

    const openFolderLink = document.createElement("a");
    openFolderLink.className = "koolook-snap-open-folder-link";
    openFolderLink.href = "#";
    openFolderLink.textContent = "Open folder ↗";
    openFolderLink.title = "Open the snapshot library folder in your file manager";
    openFolderLink.addEventListener("click", async (e) => {
        e.preventDefault();
        if (typeof revealPresetFolder !== "function") return;
        if (libraryRevealInFlight) return;
        libraryRevealInFlight = true;
        try {
            const r = await revealPresetFolder();
            toast(`Opened: ${r.path}`);
        } catch (err) {
            console.error("[Koolook] reveal failed:", err);
            toast(`Could not open library folder: ${err.message}`);
        } finally {
            libraryRevealInFlight = false;
        }
    });
    libRowTop.appendChild(openFolderLink);
    libRowInfo.appendChild(libRowTop);

    const libName = document.createElement("div");
    libName.className = "koolook-settings-folder-name";
    libName.textContent = "Library folder: (loading…)";
    libRowInfo.appendChild(libName);

    const libPath = document.createElement("div");
    libPath.className = "koolook-settings-folder-path";
    libRowInfo.appendChild(libPath);
    libRow.appendChild(libRowInfo);

    body.appendChild(libRow);

    const listWrap = document.createElement("div");
    body.appendChild(listWrap);

    const recoverySection = document.createElement("div");
    recoverySection.className = "koolook-recovery-section";
    const recoverySummary = document.createElement("div");
    recoverySummary.className = "koolook-recovery-summary";
    recoverySummary.textContent = "▸ Recovery auto-saves";
    recoverySummary.classList.add("koolook-recovery-summary-passive");
    recoverySection.appendChild(recoverySummary);
    const recoveryContent = document.createElement("div");
    recoveryContent.className = "koolook-recovery-list";
    recoveryContent.hidden = true;
    recoverySection.appendChild(recoveryContent);
    recoverySummary.addEventListener("click", () => {
        if (scopedRecovery) {
            clearScopedRecovery();
            setDialogTitle("Load snapshot");
            applyCloseButtonState();
        }
    });
    body.appendChild(recoverySection);

    let overlay;
    let titleEl;
    const close = () => overlay.remove();

    // Per-dialog state. ``pendingDelete`` holds the target whose × was
    // clicked and the row element currently outlined red; the command
    // bar swaps to a "Confirm delete?" prompt until the user confirms or
    // Escape cancels. ``scopedRecovery`` holds the bottom recovery
    // section state for one preset at a time (mockup section 3).
    let pendingDelete = null;
    let scopedRecovery = null;
    let selectedNamed = null;
    let selectedRecovery = null;
    let recoveryRequestId = 0;
    let libraryRevealInFlight = false;

    function renderEmpty(text) {
        const el = document.createElement("div");
        el.className = "koolook-snapshot-empty";
        el.textContent = text;
        return el;
    }

    async function refresh() {
        cancelDelete();
        listWrap.innerHTML = "";
        listWrap.appendChild(renderEmpty("Loading…"));
        let previews;
        try { previews = await listPresets(); }
        catch (e) { previews = []; }
        listWrap.innerHTML = "";
        clearScopedRecovery();
        clearLoadSelection();
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
            info.addEventListener("click", () => onPresetClick(p, row));
            row.appendChild(info);

            const actions = document.createElement("div");
            actions.className = "koolook-snapshot-row-actions";
            const delBtn = document.createElement("button");
            delBtn.className = "koolook-snapshot-row-btn koolook-snapshot-row-btn-danger";
            delBtn.textContent = "×";
            delBtn.title = `Delete "${p.displayName}"`;
            delBtn.addEventListener("click", (e) => { e.stopPropagation(); armDelete(p, row); });
            actions.appendChild(delBtn);
            row.appendChild(actions);

            list.appendChild(row);
        }
        listWrap.appendChild(list);
    }

    // Action helpers. Backup-chain logic is identical to the pre-#137
    // version — what changes is the *routing* into these helpers (no
    // more "Replace current state?" confirm modal for the unambiguous
    // case, and no YES/NO choice modal at all). ``close()`` here refers
    // to the OUTER Load Snapshot dialog's overlay.
    async function doNamedLoad(preview) {
        // Compare mode: read-only — hand the parsed snapshot back and close.
        // No pre-load auto-save, no applySnapshot, no tracker change.
        if (onChoose) {
            try {
                const snap = await readPreset(preview.fileName);
                close();
                onChoose(snap, { fileName: preview.fileName, displayName: preview.displayName });
            } catch (e) {
                console.error("[Koolook] preset read (compare) failed:", e);
                toast(`Could not read "${preview.displayName}": ${e.message}`);
            }
            return;
        }
        try {
            // Defensive auto-save BEFORE any destructive write — see
            // `writePreLoadAutosave` rationale in snapshot.js. If the
            // backup fails, abort the Load entirely; landing in a
            // "Load destroyed my state with no backup" world is
            // exactly what this whole code path exists to prevent.
            let backupName = null;
            if (typeof writePreLoadAutosave === "function") {
                try {
                    backupName = await writePreLoadAutosave(preview.displayName);
                } catch (e) {
                    console.error("[Koolook] pre-load auto-save failed:", e);
                    close();
                    criticalToast(
                        `Could not write pre-load auto-save: ${e.message}. ` +
                        `Load aborted to protect your current state. Fix ` +
                        `the library access issue (Settings → Library path) ` +
                        `and retry.`
                    );
                    return;
                }
            }
            const snap = await readPreset(preview.fileName);
            const { picksOk, workflowsOk } = await applySnapshot(snap);
            close();
            // Gate the tracker on FULL success only. A partial-apply
            // (one of picks / workflows persisted, the other didn't)
            // leaves on-disk and in-memory state inconsistent — if
            // we tracked the preset name anyway, the next Save would
            // default to "Save over '<name>'?" and overwrite the
            // saved preset with the corrupted half-state. Clear the
            // tracker on partial failure so the next Save forces a
            // fresh name prompt.
            if (picksOk && workflowsOk) {
                setCurrentPresetName(preview.fileName);
                // Baseline the saved-state fingerprint — current
                // state IS the loaded preset, so the indicator
                // should read "saved" until the next mutation.
                if (typeof markStateSaved === "function") markStateSaved();
                const backupSuffix = backupName ? ` (backup: ${backupName})` : "";
                toast(`Loaded preset "${preview.displayName}"${backupSuffix}.`);
            } else if (picksOk || workflowsOk) {
                setCurrentPresetName(null);
                toast(
                    `Loaded "${preview.displayName}" — PARTIAL: ` +
                    `picks ${picksOk ? "OK" : "FAIL"}, ` +
                    `workflows ${workflowsOk ? "OK" : "FAIL"}. ` +
                    `Reload to recover prior state. Tracker cleared.`
                );
            } else {
                setCurrentPresetName(null);
                toast(`Loaded "${preview.displayName}" in memory but persist failed. Reload to recover.`);
            }
        } catch (e) {
            console.error("[Koolook] preset load failed:", e);
            toast(`Could not load "${preview.displayName}": ${e.message}`);
        }
    }

    async function doAutosaveRestore(item) {
        const tooltipName = `${item.dir}/${item.fileName}`;
        // Compare mode: read-only — hand the parsed snapshot back and close.
        if (onChoose) {
            try {
                const snap = await readPreset(item.fileName, { dir: item.dir });
                close();
                onChoose(snap, { fileName: item.fileName, displayName: tooltipName });
            } catch (e) {
                console.error("[Koolook] autosave read (compare) failed:", e);
                toast(`Could not read "${tooltipName}": ${e.message}`);
            }
            return;
        }
        try {
            let backupName = null;
            if (typeof writePreLoadAutosave === "function") {
                try {
                    backupName = await writePreLoadAutosave(`Pre-recovery (${tooltipName})`);
                } catch (e) {
                    console.error("[Koolook] pre-recovery autosave failed:", e);
                    close();
                    criticalToast(
                        `Could not write pre-recovery auto-save: ${e.message}. ` +
                        `Restore aborted to protect your current state.`
                    );
                    return;
                }
            }
            const snap = await readPreset(item.fileName, { dir: item.dir });
            const { picksOk, workflowsOk } = await applySnapshot(snap);
            close();
            // Derive the original named preset from the subfolder so
            // the post-restore tracker reads naturally — `Foo_autosave`
            // → "Foo", `_unsaved_autosave` → null (no tracked preset).
            let restoredName = null;
            if (item.dir !== "_unsaved_autosave" && item.dir.endsWith("_autosave")) {
                restoredName = item.dir.slice(0, -"_autosave".length);
            }
            if (picksOk && workflowsOk) {
                setCurrentPresetName(restoredName);
                // Baseline as "auto-saved", NOT "saved". The named file
                // on disk is still the OLDER deliberate save — only
                // periodic.json matches what's now in memory. Calling
                // `markStateSaved()` here would mis-claim the named save
                // is up to date and blur the deliberate-save model
                // (named saves only change on Save / Quick Save). The
                // dot turns blue, prompting Quick Save when the user
                // wants to commit the restored state to the named file.
                if (typeof markStateAutosaved === "function") markStateAutosaved();
                const backupSuffix = backupName ? ` (backup: ${backupName})` : "";
                toast(
                    `Restored auto-save${restoredName ? ` of "${restoredName}"` : ""}` +
                    ` — Quick Save to commit to the named file${backupSuffix}.`
                );
            } else if (picksOk || workflowsOk) {
                setCurrentPresetName(null);
                toast(
                    `Restored "${tooltipName}" — PARTIAL: ` +
                    `picks ${picksOk ? "OK" : "FAIL"}, ` +
                    `workflows ${workflowsOk ? "OK" : "FAIL"}.`
                );
            } else {
                setCurrentPresetName(null);
                toast(`Restored "${tooltipName}" in memory but persist failed.`);
            }
        } catch (e) {
            console.error("[Koolook] autosave restore failed:", e);
            toast(`Could not restore "${tooltipName}": ${e.message}`);
        }
    }

    // ---- Click routing ----
    // Row clicks select; footer buttons commit. A newer recovery opens
    // the bottom section and changes the dialog title/button pair, but
    // there is no double-click-to-load convention.
    function onPresetClick(preview, rowEl) {
        // Cancel any pending delete-confirm on a different row.
        if (pendingDelete && pendingDelete.preview.fileName !== preview.fileName) {
            cancelDelete();
        }
        if (selectedNamed?.preview.fileName === preview.fileName && !pendingDelete) return;
        clearLoadSelection();
        selectedNamed = { preview, rowEl };
        rowEl.classList.add("koolook-snapshot-row-selected");

        clearScopedRecovery();
        if (typeof preview.latestAutosaveMtime === "number" &&
            typeof preview.mtime === "number" &&
            preview.latestAutosaveMtime > preview.mtime) {
            openScopedRecovery(preview);
            setDialogTitle("Auto-save is newer than the saved version");
            applyCloseButtonState();
            return;
        }
        setDialogTitle("Load snapshot");
        applyCloseButtonState();
    }

    async function openScopedRecovery(preview) {
        const requestId = ++recoveryRequestId;
        const expectedFileName = preview.fileName;
        scopedRecovery = { parentPreview: preview, item: null, rowEl: null, loading: true };
        recoverySummary.textContent = "▾ Recovery auto-saves";
        recoverySummary.classList.remove("koolook-recovery-summary-passive");
        recoveryContent.hidden = false;
        recoveryContent.innerHTML = "";
        recoveryContent.appendChild(renderEmpty("Loading recovery…"));

        let items = [];
        if (typeof listAutosaves === "function") {
            try {
                items = await listAutosaves();
            } catch (e) {
                items = [];
                toast("Could not load recovery list; saved snapshot is still available.");
            }
        }
        if (requestId !== recoveryRequestId ||
            selectedNamed?.preview.fileName !== expectedFileName) {
            return;
        }
        // Filter to this preset's autosave subdir AND pick the single
        // freshest entry by mtime. ``periodic.json`` and the rotated
        // ``pre_load_*.json`` files all live in
        // ``<preset>_autosave/`` — pick whichever was written most
        // recently regardless of kind.
        const subdir = `${preview.fileName.replace(/\.json$/i, "")}_autosave`;
        const scoped = items
            .filter((item) => item.dir === subdir)
            .sort((a, b) => (b.mtime || 0) - (a.mtime || 0));
        recoveryContent.innerHTML = "";
        if (scoped.length === 0) {
            clearScopedRecovery();
            setDialogTitle("Load snapshot");
            applyCloseButtonState();
            return;
        }
        const newest = scoped[0];

        const group = document.createElement("div");
        group.className = "koolook-recovery-group";

        const groupHead = document.createElement("div");
        groupHead.className = "koolook-recovery-group-head";
        const groupTitle = document.createElement("div");
        groupTitle.className = "koolook-recovery-group-title";
        groupTitle.textContent = subdir;
        groupHead.appendChild(groupTitle);

        const groupOpen = document.createElement("a");
        groupOpen.className = "koolook-snap-open-folder-link";
        groupOpen.href = "#";
        groupOpen.textContent = "Open folder ↗";
        groupOpen.title = "Open this recovery folder in your file manager";
        let groupRevealInFlight = false;
        groupOpen.addEventListener("click", async (e) => {
            e.preventDefault();
            if (typeof revealPresetFolder !== "function") return;
            if (groupRevealInFlight) return;
            groupRevealInFlight = true;
            try {
                const r = await revealPresetFolder({ dir: subdir });
                toast(`Opened: ${r.path}`);
            } catch (err) {
                console.error("[Koolook] reveal recovery folder failed:", err);
                toast(`Could not open recovery folder: ${err.message}`);
            } finally {
                groupRevealInFlight = false;
            }
        });
        groupHead.appendChild(groupOpen);
        group.appendChild(groupHead);

        const groupPath = document.createElement("div");
        groupPath.className = "koolook-recovery-group-path";
        const fullGroupPath = libraryPath ? `${libraryPath}/${subdir}/` : subdir;
        groupPath.textContent = formatLibraryPathBreadcrumb(fullGroupPath, 56);
        groupPath.title = fullGroupPath;
        group.appendChild(groupPath);

        const row = document.createElement("div");
        row.className = "koolook-recovery-row";

        const info = document.createElement("div");
        info.className = "koolook-recovery-row-info";
        info.title = "Click to load this auto-save instead of the named preset.";

        const kindBadge = document.createElement("span");
        kindBadge.className = "koolook-recovery-kind koolook-recovery-kind-" + newest.kind;
        kindBadge.textContent = newest.kind === "pre_load" ? "Pre-load" :
                                newest.kind === "periodic" ? "Periodic" : "Other";
        info.appendChild(kindBadge);

        const meta = document.createElement("div");
        meta.className = "koolook-recovery-row-meta";
        meta.textContent = formatPreviewMeta(newest);
        info.appendChild(meta);

        info.addEventListener("click", () => {
            if (pendingDelete) {
                cancelDelete();
            }
            if (selectedNamed) {
                selectedNamed.rowEl.classList.remove("koolook-snapshot-row-selected");
                selectedNamed = null;
            }
            if (selectedRecovery) selectedRecovery.rowEl.classList.remove("koolook-recovery-row-selected");
            selectedRecovery = { item: newest, rowEl: row };
            row.classList.add("koolook-recovery-row-selected");
            applyCloseButtonState();
        });
        row.appendChild(info);

        const actions = document.createElement("div");
        actions.className = "koolook-snapshot-row-actions";
        const delBtn = document.createElement("button");
        delBtn.className = "koolook-snapshot-row-btn koolook-snapshot-row-btn-danger";
        delBtn.textContent = "×";
        delBtn.title = "Delete this recovery auto-save";
        delBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            armDelete(newest, row, { dir: newest.dir });
        });
        actions.appendChild(delBtn);
        row.appendChild(actions);

        group.appendChild(row);
        recoveryContent.appendChild(group);
        scopedRecovery = { parentPreview: preview, item: newest, rowEl: row };
        applyCloseButtonState();
    }

    function clearScopedRecovery() {
        recoveryRequestId += 1;
        recoverySummary.textContent = "▸ Recovery auto-saves";
        recoverySummary.classList.add("koolook-recovery-summary-passive");
        recoveryContent.hidden = true;
        recoveryContent.innerHTML = "";
        scopedRecovery = null;
        if (selectedRecovery) {
            selectedRecovery.rowEl.classList.remove("koolook-recovery-row-selected");
            selectedRecovery = null;
        }
    }

    function clearLoadSelection() {
        if (selectedNamed) selectedNamed.rowEl.classList.remove("koolook-snapshot-row-selected");
        if (selectedRecovery) selectedRecovery.rowEl.classList.remove("koolook-recovery-row-selected");
        selectedNamed = null;
        selectedRecovery = null;
    }

    function setDialogTitle(text) {
        if (titleEl) titleEl.textContent = text;
    }

    // ---- Inline delete confirm ----
    // Mockup-locked: × outlines the row red, Close transforms to
    // "Yes (delete)". A second confirm click commits; Escape cancels.
    function armDelete(preview, rowEl, { dir = "" } = {}) {
        if (pendingDelete) cancelDelete();
        clearLoadSelection();
        if (!dir) clearScopedRecovery();
        rowEl.classList.add("koolook-snapshot-row-pending-delete");
        pendingDelete = { preview, rowEl, dir };
        setDialogTitle("Load snapshot");
        applyCloseButtonState();
    }

    function cancelDelete() {
        if (!pendingDelete) return;
        pendingDelete.rowEl.classList.remove("koolook-snapshot-row-pending-delete");
        pendingDelete = null;
        applyCloseButtonState();
    }

    async function commitDelete() {
        const target = pendingDelete;
        if (!target) return;
        try {
            await deletePreset(target.preview.fileName, { dir: target.dir });
            if (typeof getCurrentPresetName === "function" &&
                !target.dir &&
                getCurrentPresetName() === target.preview.fileName) {
                setCurrentPresetName(null);
            }
            toast(`Deleted "${target.preview.displayName}".`);
            pendingDelete = null;
            applyCloseButtonState();
            await refresh();
        } catch (e) {
            console.error("[Koolook] preset delete failed:", e);
            toast(`Could not delete "${target.preview.displayName}": ${e.message}`);
            cancelDelete();
        }
    }

    // Footer buttons.
    const loadFromBtn = makeModalButton({
        label: "Load from…",
        onClick: () => openLoadFrom(),
    });

    const confirmText = document.createElement("span");
    confirmText.className = "koolook-delete-confirm-text";
    confirmText.hidden = true;

    const loadSavedBtn = makeModalButton({
        label: "NO - load saved",
        onClick: () => {
            if (!scopedRecovery) return;
            doNamedLoad(scopedRecovery.parentPreview);
        },
    });
    loadSavedBtn.hidden = true;

    const loadLatestBtn = makeModalButton({
        label: "YES - load latest",
        primary: true,
        onClick: () => {
            if (!scopedRecovery || !scopedRecovery.item) return;
            doAutosaveRestore(selectedRecovery?.item || scopedRecovery.item);
        },
    });
    loadLatestBtn.hidden = true;

    const closeBtn = makeModalButton({
        label: "Close",
        onClick: () => {
            if (pendingDelete) {
                commitDelete();
                return;
            }
            if (selectedNamed) {
                doNamedLoad(selectedNamed.preview);
                return;
            }
            close();
        },
    });

    function applyCloseButtonState() {
        if (pendingDelete) {
            loadFromBtn.hidden = true;
            loadSavedBtn.hidden = true;
            loadLatestBtn.hidden = true;
            confirmText.hidden = false;
            confirmText.textContent = `Confirm delete "${pendingDelete.preview.displayName}"?`;
            closeBtn.hidden = false;
            closeBtn.textContent = "Yes";
            closeBtn.classList.add("koolook-modal-btn-danger");
            closeBtn.classList.remove("koolook-modal-btn-primary");
            closeBtn.title = "Confirm deletion. Press Esc to cancel.";
        } else if (scopedRecovery && scopedRecovery.item) {
            loadFromBtn.hidden = true;
            confirmText.hidden = true;
            confirmText.textContent = "";
            loadSavedBtn.hidden = false;
            loadLatestBtn.hidden = false;
            closeBtn.hidden = true;
            closeBtn.classList.remove("koolook-modal-btn-danger", "koolook-modal-btn-primary");
        } else if (scopedRecovery && scopedRecovery.loading) {
            loadFromBtn.hidden = true;
            confirmText.hidden = true;
            confirmText.textContent = "";
            loadSavedBtn.hidden = true;
            loadLatestBtn.hidden = true;
            closeBtn.hidden = false;
            closeBtn.textContent = "Close";
            closeBtn.classList.remove("koolook-modal-btn-danger", "koolook-modal-btn-primary");
            closeBtn.title = "";
        } else if (selectedNamed) {
            loadFromBtn.hidden = false;
            confirmText.hidden = true;
            confirmText.textContent = "";
            loadSavedBtn.hidden = true;
            loadLatestBtn.hidden = true;
            closeBtn.hidden = false;
            closeBtn.textContent = "Load";
            closeBtn.classList.add("koolook-modal-btn-primary");
            closeBtn.classList.remove("koolook-modal-btn-danger");
            closeBtn.title = `Load "${selectedNamed.preview.displayName}".`;
        } else {
            loadFromBtn.hidden = false;
            confirmText.hidden = true;
            confirmText.textContent = "";
            loadSavedBtn.hidden = true;
            loadLatestBtn.hidden = true;
            closeBtn.hidden = false;
            closeBtn.textContent = "Close";
            closeBtn.classList.remove("koolook-modal-btn-danger", "koolook-modal-btn-primary");
            closeBtn.title = "";
        }
    }

    function openLoadFrom() {
        if (typeof browseDirectories !== "function") {
            toast("Folder picker unavailable in this session.");
            return;
        }
        if (loadFromBtn.disabled) return;
        loadFromBtn.disabled = true;
        setTimeout(() => { loadFromBtn.disabled = false; }, 0);
        showFolderPicker({
            title: "Load snapshots from",
            titleTooltip: "Switch the snapshot library this Load is reading from.",
            initialPath: libraryPath,
            browseDirectories,
            createBrowseDirectory,
            onUseFolder: async (chosen) => {
                if (typeof saveSettings === "function") {
                    try { await saveSettings(chosen); }
                    catch (err) {
                        console.error("[Koolook] saveSettings failed:", err);
                        toast(`Could not save library path: ${err.message}`);
                        return;
                    }
                }
                libraryPath = chosen;
                renderLibRow(chosen);
                refresh();
                toast(`Loading snapshots from: ${chosen}.`);
            },
        });
    }

    function renderLibRow(path) {
        libraryPath = path || "";
        const leaf = libraryPath ? pathLeaf(libraryPath) : "(unavailable)";
        libName.textContent = leaf;
        libPath.textContent = libraryPath || "Path unavailable";
        libRow.title = libraryPath || "";
    }

    const spacer = document.createElement("span");
    spacer.className = "koolook-folder-picker-spacer";

    ({ overlay, titleEl } = makeModalShell({
        title: "Load snapshot",
        titleTooltip: "Restore the sidebar state from a preset file.",
        body,
        actions: [loadFromBtn, confirmText, spacer, loadSavedBtn, loadLatestBtn, closeBtn],
    }));

    // Escape-to-cancel for inline delete. The modal shell already wires
    // Escape to overlay close; intercept BEFORE that fires so a pending
    // delete just unwinds the row state without dismissing the whole
    // dialog. ``capture: true`` runs this listener before the shell's.
    // Listener is captured by name so we can tear it down on close —
    // ``document.addEventListener`` does not auto-clean up when the
    // dialog overlay is removed, and repeated open/close cycles would
    // otherwise stack one stale handler per session. Tie the cleanup to
    // ``overlay.remove`` (the canonical close path that every dismissal
    // route already goes through, thanks to the shell's wrapping).
    const escapeForDelete = (e) => {
        if (e.key !== "Escape") return;
        if (!pendingDelete) return;
        e.stopPropagation();
        cancelDelete();
    };
    document.addEventListener("keydown", escapeForDelete, { capture: true });
    const overlayRemoveBeforeCleanup = overlay.remove.bind(overlay);
    overlay.remove = () => {
        document.removeEventListener("keydown", escapeForDelete, { capture: true });
        overlayRemoveBeforeCleanup();
    };

    refresh();
    if (typeof getLibraryInfo === "function") {
        getLibraryInfo().then((info) => {
            if (info && typeof info.path === "string") renderLibRow(info.path);
            else { libName.textContent = "Library path unavailable"; libPath.textContent = ""; }
        }).catch(() => {
            libName.textContent = "Library path unavailable";
            libPath.textContent = "";
        });
    } else {
        libName.textContent = "";
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
// =============================================================================
// Folder-browse picker — issue #137, mockup §6 (navigate-into model).
//
// Replaces the legacy ``openBrowseDialog`` (a select-driven "pick a sibling"
// browser scoped to the Settings dialog) with a top-level reusable picker
// driven by a path input. The path input is the source of truth: typing a
// path + Enter navigates, clicking a folder row drills in (and updates the
// input), ↑ Up climbs one level. Files are shown greyed below the
// directories as a "yes, this is the folder I expected" affordance — they
// do NOT count toward the selection and clicking them is inert.
//
// Visual spec: docs/designs/snapshot-dialogs.html §6.
// Visual harness: docs/designs/_harness/folder-picker.html.
// =============================================================================
export function showFolderPicker({
    title,
    titleTooltip,
    initialPath,
    browseDirectories,
    createBrowseDirectory,
    onUseFolder,
    onCancel,
}) {
    const body = document.createElement("div");
    body.className = "koolook-folder-picker";

    // Toolbar row: ↑ Up + path input. The toolbar is rebuilt in place when
    // the New folder… flow opens (a transient name-input replaces the
    // path-input UI), and restored on Create / Cancel.
    const toolbar = document.createElement("div");
    toolbar.className = "koolook-folder-picker-toolbar";

    const upBtn = makeModalButton({
        label: "↑ Up",
        onClick: () => { if (currentParent) navigate(currentParent); },
    });
    upBtn.classList.add("koolook-folder-picker-up");

    const pathInput = document.createElement("input");
    pathInput.className = "koolook-modal-input koolook-folder-picker-path";
    pathInput.value = initialPath || "";
    pathInput.spellcheck = false;
    pathInput.title = "Type or paste a path and press Enter, or click a subfolder below.";
    pathInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            e.preventDefault();
            const target = pathInput.value.trim();
            if (target) navigate(target);
        }
    });

    toolbar.appendChild(upBtn);
    toolbar.appendChild(pathInput);
    body.appendChild(toolbar);

    // List body.
    const list = document.createElement("div");
    list.className = "koolook-folder-picker-list";
    body.appendChild(list);

    let currentPath = initialPath || "";
    let currentParent = "";
    let overlay;
    const close = () => overlay.remove();

    function renderState(text, isError) {
        list.innerHTML = "";
        const el = document.createElement("div");
        el.className = isError
            ? "koolook-folder-picker-empty koolook-folder-picker-error"
            : "koolook-folder-picker-empty";
        el.textContent = text;
        list.appendChild(el);
    }

    function revealPathEnd() {
        // Even with `direction: rtl` the caret/scroll position needs a nudge
        // after programmatic value changes so the END of the path is on-
        // screen. Defer to the next tick so layout has settled.
        setTimeout(() => { pathInput.scrollLeft = pathInput.scrollWidth; }, 0);
    }

    function renderListing(data) {
        list.innerHTML = "";
        const dirs = data.dirs || [];
        const files = data.files || [];
        if (!dirs.length && !files.length) {
            renderState("Folder is empty.");
            return;
        }
        for (const d of dirs) {
            const row = document.createElement("button");
            row.type = "button";
            row.className = "koolook-folder-picker-row";
            row.title = `Open ${d.name}`;
            const icon = document.createElement("span");
            icon.className = "koolook-folder-picker-icon";
            icon.textContent = "📁";
            row.appendChild(icon);
            const name = document.createElement("span");
            name.className = "koolook-folder-picker-name";
            name.textContent = d.name;
            row.appendChild(name);
            row.addEventListener("click", () => navigate(d.path));
            list.appendChild(row);
        }
        for (const f of files) {
            const row = document.createElement("div");
            row.className = "koolook-folder-picker-row koolook-folder-picker-row-file";
            row.title = "Files are shown for context only — Use this folder commits the folder, not a file.";
            const icon = document.createElement("span");
            icon.className = "koolook-folder-picker-icon";
            icon.textContent = "📄";
            row.appendChild(icon);
            const name = document.createElement("span");
            name.className = "koolook-folder-picker-name";
            name.textContent = f.name;
            row.appendChild(name);
            list.appendChild(row);
        }
    }

    async function navigate(target) {
        renderState("Loading…");
        useBtn.disabled = true;
        upBtn.disabled = true;
        newFolderBtn.disabled = true;
        let data;
        try {
            // ``{files: true}`` is the opt-in for the files-for-context
            // affordance — the listing endpoint returns greyed file
            // entries below the directory rows so the user can verify
            // "yes, this is the folder I expected" before committing.
            data = await browseDirectories(target, { files: true });
        } catch (e) {
            renderState(e.message || String(e), true);
            // Keep the input value as-typed so the user can fix it and retry.
            useBtn.disabled = true;
            upBtn.disabled = !currentParent;
            newFolderBtn.disabled = false;
            return;
        }
        currentPath = data.path;
        currentParent = data.parentPath || "";
        pathInput.value = data.path;
        revealPathEnd();
        useBtn.disabled = false;
        upBtn.disabled = !currentParent;
        newFolderBtn.disabled = typeof createBrowseDirectory !== "function";
        renderListing(data);
    }

    // Footer buttons.
    const newFolderBtn = makeModalButton({
        label: "New folder…",
        onClick: () => openNewFolderInput(),
    });

    const cancelBtn = makeModalButton({
        label: "Cancel",
        onClick: () => {
            close();
            if (onCancel) onCancel();
        },
    });

    const useBtn = makeModalButton({
        label: "Use this folder",
        primary: true,
        onClick: () => {
            const chosen = pathInput.value.trim();
            if (!chosen) return;
            close();
            if (onUseFolder) onUseFolder(chosen);
        },
    });

    // Inline new-folder input — swaps toolbar contents transiently. Cleaner
    // than another modal-on-modal: the picker stays open, the body still
    // shows the destination's existing children, and the user has visual
    // continuity about *where* the new folder is being created.
    function openNewFolderInput() {
        if (typeof createBrowseDirectory !== "function") return;
        // Snapshot the parent at the moment New folder… is clicked, so a
        // late navigation doesn't redirect the create call. (Defensive —
        // we disable nav while the input is open, but the assignment makes
        // the contract explicit.)
        const parentForCreate = currentPath;

        toolbar.innerHTML = "";
        toolbar.classList.add("koolook-folder-picker-toolbar-newfolder");

        const label = document.createElement("span");
        label.className = "koolook-folder-picker-newfolder-label";
        // Show only the leaf folder name to keep the toolbar height stable;
        // the full path is visible on hover via the ``title`` attribute and
        // was on screen in the path input the user just dismissed.
        label.textContent = `New folder in ${pathLeaf(parentForCreate)}:`;
        label.title = parentForCreate;
        toolbar.appendChild(label);

        const nameInput = document.createElement("input");
        nameInput.className = "koolook-modal-input koolook-folder-picker-newfolder-input";
        nameInput.placeholder = "untitled";
        nameInput.spellcheck = false;
        toolbar.appendChild(nameInput);

        function restoreToolbar() {
            toolbar.classList.remove("koolook-folder-picker-toolbar-newfolder");
            toolbar.innerHTML = "";
            toolbar.appendChild(upBtn);
            toolbar.appendChild(pathInput);
            revealPathEnd();
        }

        const createBtn = makeModalButton({
            label: "Create",
            primary: true,
            onClick: async () => {
                const name = nameInput.value.trim();
                if (!name) { nameInput.focus(); return; }
                createBtn.disabled = true;
                cancelInline.disabled = true;
                try {
                    const r = await createBrowseDirectory(parentForCreate, name);
                    restoreToolbar();
                    // Navigate INTO the newly created folder so the user can
                    // immediately commit or drill further.
                    await navigate(r.path || `${parentForCreate}/${name}`);
                } catch (e) {
                    renderState(e.message || String(e), true);
                    createBtn.disabled = false;
                    cancelInline.disabled = false;
                }
            },
        });

        const cancelInline = makeModalButton({
            label: "Cancel",
            onClick: () => restoreToolbar(),
        });

        toolbar.appendChild(createBtn);
        toolbar.appendChild(cancelInline);

        nameInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter") { e.preventDefault(); createBtn.click(); }
            if (e.key === "Escape") { e.preventDefault(); restoreToolbar(); }
        });
        setTimeout(() => nameInput.focus(), 0);
    }

    const spacer = document.createElement("span");
    spacer.className = "koolook-folder-picker-spacer";

    ({ overlay } = makeModalShell({
        title,
        titleTooltip,
        body,
        actions: [newFolderBtn, spacer, cancelBtn, useBtn],
    }));

    // Initial load — kick off with whatever path the caller supplied (or
    // the empty string, which the server resolves to the configured
    // library dir).
    navigate(initialPath || "");
}

// =============================================================================
// Context menu helper
// =============================================================================
let activeContextMenuCleanup = null;

function closeActiveContextMenu() {
    if (activeContextMenuCleanup) {
        activeContextMenuCleanup();
        activeContextMenuCleanup = null;
    }
}

export function showContextMenu(event, items) {
    event.preventDefault();
    event.stopPropagation();
    closeActiveContextMenu();

    const menu = document.createElement("div");
    menu.className = "koolook-context-menu";
    let closeOnClick = null;

    const cleanup = () => {
        menu.remove();
        if (closeOnClick) {
            document.removeEventListener("click", closeOnClick);
            document.removeEventListener("contextmenu", closeOnClick);
            closeOnClick = null;
        }
        if (activeContextMenuCleanup === cleanup) activeContextMenuCleanup = null;
    };
    activeContextMenuCleanup = cleanup;

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
                cleanup();
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
        closeOnClick = (ev) => {
            if (!menu.contains(ev.target)) {
                cleanup();
            }
        };
        document.addEventListener("click", closeOnClick);
        document.addEventListener("contextmenu", closeOnClick);
    }, 0);
}
