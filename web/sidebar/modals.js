// SPDX-FileCopyrightText: 2025-2026 Kforge Labs <https://github.com/malkuthro/ComfyUI-Koolook>
// SPDX-License-Identifier: GPL-3.0-only

// =============================================================================
// Modal + context-menu DOM. The shell takes care of the Escape-key / overlay-
// click / Cancel / OK dismissal paths and ties the document-level keydown
// listener to the overlay's lifetime so opening many modals doesn't leak.
// =============================================================================
import { compareNames, criticalToast, toast } from "./constants.js";
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

    // Optional subtitle — same contract as showInputModal / showThreeWayDialog.
    // Used by the Save-as-new overwrite-confirm path to surface the library
    // breadcrumb so the user keeps "where am I writing to" context across
    // the multi-modal Save flow (input modal → existence check → confirm).
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
    msg.textContent = message;
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
    parts.push(`${p.workflowCount} workflow${p.workflowCount === 1 ? "" : "s"}`);
    parts.push(`${p.pickCount} pick${p.pickCount === 1 ? "" : "s"}`);
    if (typeof p.mtime === "number" && isFinite(p.mtime)) {
        const stamp = formatLocalStamp(new Date(p.mtime * 1000));
        if (stamp) parts.push(`saved ${stamp}`);
    } else if (p.exportedAt) {
        const stamp = formatLocalStamp(new Date(p.exportedAt));
        if (stamp) parts.push(`exported ${stamp}`);
    }
    return parts.join(" · ");
}

function pathLeaf(path) {
    const parts = String(path || "").split(/[\\/]+/).filter(Boolean);
    return parts.length ? parts[parts.length - 1] : String(path || "");
}

function renderLibraryLocation(el, path, { label = "Library folder", title } = {}) {
    el.innerHTML = "";
    const folder = document.createElement("div");
    folder.className = "koolook-settings-folder-name";
    // `title` override skips the `<label>: <leaf>` formatting so callers
    // (autosave group headers) can render just the subdir name without
    // the "Library folder:" prefix while still reusing the same CSS
    // pair (`folder-name` + `folder-path`) for visual consistency.
    folder.textContent = title ?? `${label}: ${path ? pathLeaf(path) : "(unavailable)"}`;
    el.appendChild(folder);

    const full = document.createElement("div");
    full.className = "koolook-settings-folder-path";
    full.textContent = path || "Path unavailable";
    el.appendChild(full);
    el.title = path || "";
}

// Local helper: a 3-button confirm modal (Save / Save as new / Cancel) for
// the Save flow when there's a current preset to overwrite. `showConfirmModal`
// is 2-button only; we'd need to bend it. Cleaner to assemble directly with
// `makeModalShell`.
function showThreeWayDialog({ title, message, primaryLabel, secondaryLabel, cancelLabel, onPrimary, onSecondary, subtitle }) {
    const body = document.createElement("div");

    // Optional subtitle — same contract as showInputModal. Sits above the
    // message so library path / context is visible before the user scans
    // the action question.
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
    getLibraryInfo,
    markStateSaved,
    onToast,
}) {
    const toast = onToast || (() => {});

    // Build a reactive library-path subtitle element. Both the input modal
    // (Save as new…) and the three-way overwrite dialog reuse the same node
    // so the user sees consistent context regardless of which path they
    // entered the Save flow through. Async-populated to avoid blocking the
    // modal open on a server round-trip — the user sees "(loading…)" briefly,
    // then the resolved path.
    function buildPathSubtitle() {
        const el = document.createElement("div");
        el.className = "koolook-modal-pathline";
        el.textContent = "Library: (loading…)";
        if (typeof getLibraryInfo === "function") {
            getLibraryInfo().then((info) => {
                if (info && typeof info.path === "string" && info.path) {
                    el.className = "koolook-load-library-location";
                    renderLibraryLocation(el, info.path, { label: "Library folder" });
                } else {
                    el.textContent = "Library path unavailable.";
                }
            }).catch(() => {
                el.textContent = "Library path unavailable.";
            });
        } else {
            el.textContent = "";
        }
        return el;
    }

    async function doSave(name, { confirmOverwrite }) {
        const sanitized = sanitizeName(name);
        if (!sanitized) {
            toast("Snapshot name is empty after stripping unsafe characters.");
            return;
        }
        // For Save-as-new only: check existence + prompt to overwrite.
        // The "overwrite current" path skipped this prompt by design
        // (the user already confirmed via the three-way dialog).
        //
        // `presetExists` is now tri-state: true / false / null. The
        // null case means the existence check itself failed (network
        // down, server error) — refuse to silently write, since we
        // can't tell whether we'd be clobbering an existing preset.
        if (confirmOverwrite) {
            const exists = await presetExists(sanitized);
            if (exists === null) {
                toast(
                    "Cannot reach the preset library to verify name. " +
                    "Save canceled — check Settings or your connection."
                );
                return;
            }
            if (exists === true) {
                showConfirmModal({
                    title: "Overwrite existing preset?",
                    message: `A snapshot named "${sanitized}" already exists. Overwrite it?`,
                    confirmLabel: "Overwrite",
                    danger: true,
                    // Carry the library breadcrumb across the input → exists →
                    // confirm chain so the user knows which library they're
                    // overwriting in (matters when Settings → libraryPath has
                    // been changed mid-session, or facility users juggle
                    // multiple shared libraries).
                    subtitle: buildPathSubtitle(),
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
            // Baseline the saved-state fingerprint — flips the sidebar
            // status indicator from "unsaved" to "saved" without waiting
            // for an unrelated mutation event to refresh it.
            if (typeof markStateSaved === "function") markStateSaved();
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
            subtitle: buildPathSubtitle(),
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
            subtitle: buildPathSubtitle(),
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
    getCurrentPresetName,
    getLibraryInfo,
    writePreLoadAutosave,
    markStateSaved,
    markStateAutosaved,
    listAutosaves,
    revealPresetFolder,
    onToast,
}) {
    const toast = onToast || (() => {});
    const body = document.createElement("div");

    // Captured from `getLibraryInfo()` resolution below — the recovery
    // section uses it to render full subdir paths (`<library>/<subdir>`)
    // under each group header, matching the library section's title-plus-
    // path layout. Empty string is the not-yet-resolved sentinel; group
    // headers fall back to no subtitle in that case.
    let libraryPath = "";

    // Header row: library title + path (rendered by `renderLibraryLocation`
    // for visual parity with the recovery group headers below) on the
    // left, `📂 Open` button on the right. Same flex layout (`flex-start`
    // alignment + `space-between` justification) as group headers so both
    // sections feel like one consistent UI.
    const pathRow = document.createElement("div");
    pathRow.className = "koolook-load-library-row";
    pathRow.style.display = "flex";
    pathRow.style.alignItems = "flex-start";
    pathRow.style.justifyContent = "space-between";
    pathRow.style.gap = "8px";
    const pathLine = document.createElement("div");
    pathLine.className = "koolook-load-library-location";
    pathLine.style.flex = "1";
    pathLine.style.minWidth = "0";
    pathLine.textContent = "Library: (loading…)";
    pathRow.appendChild(pathLine);
    if (typeof revealPresetFolder === "function") {
        const openLibraryBtn = document.createElement("button");
        // `koolook-snapshot-row-btn` matches the size + styling of the
        // group-header Open button in the recovery section, so both
        // surfaces look like the same control instead of one large
        // (`koolook-modal-button`) and one small (row-btn) variant.
        openLibraryBtn.className = "koolook-snapshot-row-btn";
        openLibraryBtn.textContent = "📂 Open";
        openLibraryBtn.title = "Open the snapshot library folder in your file manager";
        openLibraryBtn.style.flex = "0 0 auto";
        openLibraryBtn.addEventListener("click", async () => {
            try {
                const r = await revealPresetFolder();
                toast(`Opened: ${r.path}`);
            } catch (e) {
                console.error("[Koolook] reveal failed:", e);
                toast(`Could not open library folder: ${e.message}`);
            }
        });
        pathRow.appendChild(openLibraryBtn);
    }
    body.appendChild(pathRow);

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

    // Action helpers. Extracted from the in-line `onConfirm` bodies so
    // both the standard load flow AND the autosave-newer YES/NO choice
    // modal can dispatch the same pre-load-autosave + applySnapshot +
    // tracker-rebase + partial-failure logic without duplicating ~30
    // lines per call site. `close()` here refers to the OUTER Load
    // Snapshot dialog's overlay — closures over `close` from the
    // enclosing `showLoadSnapshotDialog` scope.
    async function doNamedLoad(preview) {
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

    function promptApply(preview) {
        // When the corresponding `<base>_autosave/periodic.json` is
        // strictly newer than the named save, route to the YES/NO
        // choice modal so the user can pick which version to load
        // without navigating the Recovery disclosure or having to
        // mentally reason about which timestamp is newer. The standard
        // single-button confirm is reserved for rows where there's no
        // newer auto-save (the unambiguous case).
        if (typeof preview.latestAutosaveMtime === "number" &&
            typeof preview.mtime === "number" &&
            preview.latestAutosaveMtime > preview.mtime) {
            promptApplyChoice(preview);
            return;
        }
        showConfirmModal({
            title: "Replace current state?",
            message:
                `Loading "${preview.displayName}" will replace your current ` +
                `picks and workflows with the snapshot's contents (` +
                `${formatPreviewMeta(preview)}). A pre-load auto-save of your ` +
                `current state will be written first as a recovery point.`,
            confirmLabel: "Load preset",
            danger: true,
            onConfirm: () => doNamedLoad(preview),
        });
    }

    // YES/NO choice modal — appears when the row's auto-save is newer
    // than its named save. Built directly via `makeModalShell` (not
    // `showConfirmModal`) because we need ESC / overlay click to mean
    // "cancel both" rather than "fire one of the actions": showConfirm
    // would bind one of the two paths to its cancel handler, which
    // also runs on Escape. Two affirmative buttons, one neutral
    // dismissal — this layout requires the direct shell.
    function promptApplyChoice(preview) {
        const namedStamp = formatLocalStamp(new Date(preview.mtime * 1000));
        const autoStamp = formatLocalStamp(new Date(preview.latestAutosaveMtime * 1000));

        const body = document.createElement("div");

        const intro = document.createElement("div");
        intro.className = "koolook-modal-message";
        intro.textContent =
            `Loading "${preview.displayName}" will replace your current ` +
            `picks and workflows. A pre-load auto-save of your current ` +
            `state will be written first as a recovery point.`;
        body.appendChild(intro);

        const compareHead = document.createElement("div");
        compareHead.className = "koolook-modal-message";
        compareHead.style.marginTop = "10px";
        compareHead.textContent = "Auto-save is NEWER than the saved version:";
        body.appendChild(compareHead);

        const compareList = document.createElement("ul");
        compareList.style.margin = "4px 0 0 18px";
        compareList.style.padding = "0";
        compareList.style.fontSize = "12px";
        const liNamed = document.createElement("li");
        liNamed.textContent = `Saved version — ${namedStamp}`;
        const liAuto = document.createElement("li");
        liAuto.textContent = `Auto-save — ${autoStamp}`;
        compareList.appendChild(liNamed);
        compareList.appendChild(liAuto);
        body.appendChild(compareList);

        const question = document.createElement("div");
        question.className = "koolook-modal-message";
        question.style.marginTop = "10px";
        question.textContent =
            "Do you want to load the auto-saved version? " +
            "(NO loads the saved version. Press Esc to cancel.)";
        body.appendChild(question);

        let shell;
        const noBtn = makeModalButton({
            label: "NO",
            onClick: () => { shell.close(); doNamedLoad(preview); },
        });
        const yesBtn = makeModalButton({
            label: "YES — load auto-save",
            primary: true,
            onClick: () => {
                shell.close();
                doAutosaveRestore({
                    dir: `${preview.fileName}_autosave`,
                    fileName: "periodic",
                });
            },
        });
        shell = makeModalShell({
            title: "Replace current state?",
            body,
            actions: [noBtn, yesBtn],
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
                    // If the deleted preset was the one tracked as
                    // "currently loaded," clear the tracker. Otherwise
                    // the next Save would offer to "Save over '<deleted>'"
                    // and silently re-create the file.
                    if (typeof getCurrentPresetName === "function" &&
                        getCurrentPresetName() === preview.fileName) {
                        setCurrentPresetName(null);
                    }
                    toast(`Deleted "${preview.displayName}".`);
                    await refresh();
                } catch (e) {
                    console.error("[Koolook] preset delete failed:", e);
                    toast(`Could not delete "${preview.displayName}": ${e.message}`);
                }
            },
        });
    }

    // ---- Recovery auto-saves section (collapsible) ----
    // Closes the gap from the v2 layout: pre-load and periodic autosaves
    // live in `<preset>_autosave/` subfolders, deliberately hidden from the
    // main Load list to keep that list clean. But "hidden" can't mean
    // "unreachable" — the whole point of pre-load autosave is recovery.
    // This section is the explicit recovery path: lazy-loaded on first
    // expand, grouped by subdir, click-to-restore (with the same pre-load
    // protection the regular Load flow uses).
    const recoverySection = document.createElement("details");
    recoverySection.className = "koolook-recovery-section";
    const recoverySummary = document.createElement("summary");
    recoverySummary.className = "koolook-recovery-summary";
    // Native `<details>` already renders a disclosure triangle on
    // `<summary>`. Don't add a second arrow character here — the previous
    // `▸ / ▾` text duplicated the native marker and the user saw two
    // arrows side-by-side.
    recoverySummary.textContent = "Recovery auto-saves";
    recoverySection.appendChild(recoverySummary);
    const recoveryList = document.createElement("div");
    recoveryList.className = "koolook-recovery-list";
    recoverySection.appendChild(recoveryList);

    function promptDeleteAutosave(item) {
        showConfirmModal({
            title: "Delete recovery auto-save?",
            message: `"${item.dir}/${item.fileName}" will be removed. ` +
                     `If this is the only remaining recovery point for that ` +
                     `preset, you won't be able to restore from it later.`,
            confirmLabel: "Delete",
            danger: true,
            onConfirm: async () => {
                try {
                    await deletePreset(item.fileName, { dir: item.dir });
                    toast(`Deleted "${item.dir}/${item.fileName}".`);
                    await refreshRecovery();
                } catch (e) {
                    console.error("[Koolook] autosave delete failed:", e);
                    toast(`Could not delete: ${e.message}`);
                }
            },
        });
    }

    function promptApplyAutosave(item) {
        const tooltipName = `${item.dir}/${item.fileName}`;
        showConfirmModal({
            title: "Restore from auto-save?",
            message:
                `Restoring "${tooltipName}" will replace your current ` +
                `picks and workflows with the auto-save's contents (` +
                `${formatPreviewMeta(item)}). A pre-load auto-save of ` +
                `your current state will be written first as a recovery ` +
                `point — this restore is itself reversible.`,
            confirmLabel: "Restore",
            danger: true,
            onConfirm: () => doAutosaveRestore(item),
        });
    }

    let recoveryLoaded = false;
    async function refreshRecovery() {
        recoveryList.innerHTML = "";
        recoveryList.appendChild(renderEmpty("Loading…"));
        let entries = [];
        if (typeof listAutosaves === "function") {
            try { entries = await listAutosaves(); } catch (e) { entries = []; }
        }
        recoveryList.innerHTML = "";
        if (entries.length === 0) {
            recoveryList.appendChild(renderEmpty("No recovery auto-saves yet."));
            return;
        }
        // Group by subdir for readability. Map iteration preserves first-
        // seen order, which from the server is alphabetical-by-subdir —
        // matches expectations.
        const groups = new Map();
        for (const p of entries) {
            if (!groups.has(p.dir)) groups.set(p.dir, []);
            groups.get(p.dir).push(p);
        }
        for (const [subdir, items] of groups) {
            const groupEl = document.createElement("div");
            groupEl.className = "koolook-recovery-group";
            // Subfolder header: title (subdir name) + path subtitle on the
            // left, `📂 Open` button on the right. Mirrors the library
            // section's layout exactly — same flex options, same
            // `renderLibraryLocation` rendering pair (`folder-name` +
            // `folder-path` CSS), same Open button class. Visually the
            // two surfaces should be indistinguishable apart from the
            // contents.
            const groupHeader = document.createElement("div");
            groupHeader.className = "koolook-recovery-group-header";
            groupHeader.style.display = "flex";
            groupHeader.style.alignItems = "flex-start";
            groupHeader.style.justifyContent = "space-between";
            groupHeader.style.gap = "8px";
            const headerInfo = document.createElement("div");
            headerInfo.style.flex = "1";
            headerInfo.style.minWidth = "0";
            const fullSubdirPath = libraryPath ? `${libraryPath}/${subdir}` : "";
            renderLibraryLocation(headerInfo, fullSubdirPath, { title: subdir });
            groupHeader.appendChild(headerInfo);
            if (typeof revealPresetFolder === "function") {
                const openBtn = document.createElement("button");
                openBtn.className = "koolook-snapshot-row-btn";
                openBtn.textContent = "📂 Open";
                openBtn.title = `Open "${subdir}/" in your file manager`;
                openBtn.style.flex = "0 0 auto";
                openBtn.addEventListener("click", async (e) => {
                    e.stopPropagation();
                    try {
                        const r = await revealPresetFolder({ dir: subdir });
                        toast(`Opened: ${r.path}`);
                    } catch (err) {
                        console.error("[Koolook] reveal failed:", err);
                        toast(`Could not open "${subdir}/": ${err.message}`);
                    }
                });
                groupHeader.appendChild(openBtn);
            }
            groupEl.appendChild(groupHeader);
            // Wrap rows in the same `koolook-snapshot-list` container the
            // library section uses so each group gets the same bordered-
            // box treatment around its rows. Without this, the recovery
            // rows render flat (no border, no rounded corners) and
            // visually diverge from the Koolook_v03 row even though the
            // row contents are identical.
            const rowsList = document.createElement("div");
            rowsList.className = "koolook-snapshot-list";
            for (const item of items) {
                const row = document.createElement("div");
                row.className = "koolook-snapshot-row";

                const info = document.createElement("div");
                info.className = "koolook-snapshot-row-info";
                info.title = `Click to restore "${item.dir}/${item.fileName}"`;

                const name = document.createElement("div");
                name.className = "koolook-snapshot-row-name";
                const kindBadge = document.createElement("span");
                kindBadge.className = "koolook-recovery-kind koolook-recovery-kind-" + item.kind;
                kindBadge.textContent = item.kind === "pre_load" ? "Pre-load" :
                                        item.kind === "periodic" ? "Periodic" : "Other";
                name.appendChild(kindBadge);
                // Show the bare filename (`pre_load_2026-05-07T14-32-…`,
                // `periodic`) instead of the snapshot's self-reported
                // displayName, which for autosaves is a redundant
                // `Periodic auto-save · <subdir> · <iso>` string that
                // pushes the timestamp off-screen on narrow modals.
                name.appendChild(document.createTextNode(item.fileName));
                info.appendChild(name);

                const meta = document.createElement("div");
                meta.className = "koolook-snapshot-row-meta";
                // mtime is when the file was actually written — strictly
                // more relevant for "which autosave is freshest" than the
                // snapshot's self-reported `exportedAt` (the two match for
                // autosaves anyway, but mtime is what survives an out-of-
                // band file copy in Finder).
                meta.textContent = formatPreviewMeta(item);
                info.appendChild(meta);

                info.addEventListener("click", () => promptApplyAutosave(item));
                row.appendChild(info);

                const actions = document.createElement("div");
                actions.className = "koolook-snapshot-row-actions";
                const delBtn = document.createElement("button");
                delBtn.className = "koolook-snapshot-row-btn koolook-snapshot-row-btn-danger";
                delBtn.textContent = "×";
                delBtn.title = `Delete "${item.dir}/${item.fileName}"`;
                delBtn.addEventListener("click", (e) => { e.stopPropagation(); promptDeleteAutosave(item); });
                actions.appendChild(delBtn);
                row.appendChild(actions);

                rowsList.appendChild(row);
            }
            groupEl.appendChild(rowsList);
            recoveryList.appendChild(groupEl);
        }
    }

    // Lazy-load on first expand; the autosave listing is N+1 reads (one
    // per file to validate it's a snapshot and extract the displayName)
    // and we don't want to pay that cost when the user is just looking
    // for a regular preset.
    recoverySection.addEventListener("toggle", () => {
        if (!recoverySection.open) return;
        if (recoveryLoaded) return;
        recoveryLoaded = true;
        refreshRecovery();
    });
    body.appendChild(recoverySection);

    const closeBtn = makeModalButton({ label: "Close", onClick: close });
    ({ overlay } = makeModalShell({
        title: "Load snapshot",
        body,
        actions: [closeBtn],
    }));
    refresh();
    if (typeof getLibraryInfo === "function") {
        getLibraryInfo().then((info) => {
            if (info) {
                renderLibraryLocation(pathLine, info.path, { label: "Library folder" });
                // Stash for the recovery section's group-header path
                // subtitle. If recovery is already loaded (user expanded
                // before this resolved), refresh so the path appears.
                libraryPath = info.path || "";
                if (recoveryLoaded) refreshRecovery();
            } else {
                pathLine.textContent = "Library path unavailable.";
            }
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


export function showSnapshotSettingsDialog({
    getSettings,
    saveSettings,
    browseDirectories,
    createBrowseDirectory,
    onToast,
}) {
    const toast = onToast || (() => {});
    const body = document.createElement("div");

    body.appendChild(modalLabel("Library path"));

    const inputRow = document.createElement("div");
    inputRow.className = "koolook-path-input-row";

    const input = document.createElement("input");
    input.className = "koolook-modal-input";
    input.placeholder = "/absolute/path/to/library  (leave empty to use env-var or default)";
    input.disabled = true;
    inputRow.appendChild(input);

    const browseBtn = makeModalButton({
        label: "Browse...",
        onClick: () => openBrowseDialog(input.value.trim() || ""),
    });
    browseBtn.disabled = true;
    inputRow.appendChild(browseBtn);
    body.appendChild(inputRow);

    const hint = document.createElement("div");
    hint.style.cssText = "font-size: 11px; opacity: 0.55; margin: 8px 0 12px; line-height: 1.5; word-break: break-all;";
    hint.textContent = "Loading current settings…";
    body.appendChild(hint);

    let overlay;
    let savedLibraryPath = "";
    const close = () => overlay.remove();

    function renderSelectedFolder(el, path) {
        el.innerHTML = "";
        if (!path) {
            el.textContent = "No folder selected";
            return;
        }
        const label = document.createElement("span");
        label.textContent = "Selected folder: ";
        const name = document.createElement("strong");
        name.className = "koolook-browse-selected-name";
        name.textContent = pathLeaf(path);
        const full = document.createElement("div");
        full.className = "koolook-browse-selected-path";
        full.textContent = path;
        el.appendChild(label);
        el.appendChild(name);
        el.appendChild(full);
    }

    function renderPathSummary({ saved }) {
        const rawPath = input.value.trim();
        const folder = rawPath ? pathLeaf(rawPath) : "(default)";
        const label = saved ? "Saved folder" : "Selected folder";
        hint.innerHTML = "";

        const name = document.createElement("div");
        name.className = "koolook-settings-folder-name";
        name.textContent = `${label}: ${folder}`;
        hint.appendChild(name);

        const path = document.createElement("div");
        path.className = "koolook-settings-folder-path";
        path.textContent = rawPath || "Using env var / default preset folder";
        hint.appendChild(path);

        if (!saved) {
            const note = document.createElement("div");
            note.className = "koolook-settings-save-note";
            note.textContent = "Saves folder location only. Use Save/Quick Save to write a snapshot.";
            hint.appendChild(note);
        }
    }

    function revealPathEnd() {
        setTimeout(() => {
            input.scrollLeft = input.scrollWidth;
        }, 0);
    }

    function updateSaveState() {
        if (!save) return;
        const dirty = input.value.trim() !== savedLibraryPath;
        save.disabled = !dirty;
        save.textContent = dirty ? "Save folder" : "Saved";
        save.classList.toggle("koolook-modal-btn-primary", dirty);
        save.title = dirty
            ? "Save snapshot folder location"
            : "Snapshot folder is saved";
        renderPathSummary({ saved: !dirty });
    }

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
        savedLibraryPath = settings.savedLibraryPath || "";
        input.value = savedLibraryPath;
        revealPathEnd();
        // Native tooltip surfaces the full path on hover — the input is only
        // ~440px wide so long paths get truncated by horizontal scroll.
        // The user can still drag-select to view, but the tooltip avoids
        // making them do so.
        input.title = settings.savedLibraryPath || "";
        input.disabled = false;
        let sourceLabel;
        if (settings.source === "settings") sourceLabel = "from this Settings panel";
        else if (settings.source === "env") sourceLabel = `from ${settings.envVar} env var`;
        else sourceLabel = "built-in default";
        browseBtn.disabled = typeof browseDirectories !== "function";
        updateSaveState();
        hint.title = `Currently in effect: ${settings.resolvedPath} (${sourceLabel})`;
    }

    async function doSave(rawValue) {
        if (rawValue === savedLibraryPath) {
            updateSaveState();
            return;
        }
        try {
            const result = await saveSettings(rawValue);
            const sourceLabel = result.source === "settings"
                ? "from this Settings panel"
                : (result.source === "env" ? "from env var" : "built-in default");
            savedLibraryPath = result.savedLibraryPath || "";
            input.value = savedLibraryPath;
            input.title = savedLibraryPath;
            revealPathEnd();
            hint.title = `Currently in effect: ${result.resolvedPath} (${sourceLabel})`;
            toast(rawValue
                ? "Library folder saved. Use Save/Quick Save to write snapshots here."
                : "Override cleared — using env var / default."
            );
            updateSaveState();
        } catch (e) {
            console.error("[Koolook] settings save failed:", e);
            toast(`Could not save settings: ${e.message}`);
        }
    }

    function openBrowseDialog(startPath) {
        if (typeof browseDirectories !== "function") return;
        const pickerBody = document.createElement("div");

        const current = document.createElement("div");
        current.className = "koolook-browse-current";
        current.textContent = "Loading...";
        pickerBody.appendChild(current);

        const navRow = document.createElement("div");
        navRow.className = "koolook-browse-nav";
        const upBtn = makeModalButton({ label: "Up", onClick: () => {} });
        const driveLabel = document.createElement("span");
        driveLabel.className = "koolook-browse-drive-label";
        driveLabel.textContent = "Location";
        const rootsSelect = document.createElement("select");
        rootsSelect.className = "koolook-modal-select";
        navRow.appendChild(upBtn);
        navRow.appendChild(driveLabel);
        navRow.appendChild(rootsSelect);
        pickerBody.appendChild(navRow);

        const list = document.createElement("div");
        list.className = "koolook-browse-list";
        pickerBody.appendChild(list);

        let selectedPath = startPath;
        let overlay;
        const closePicker = () => overlay.remove();

        async function render(path) {
            choose.disabled = true;
            list.innerHTML = "";
            const loading = document.createElement("div");
            loading.className = "koolook-snapshot-empty";
            loading.textContent = "Loading...";
            list.appendChild(loading);
            let data;
            try {
                data = await browseDirectories(path);
            } catch (e) {
                list.innerHTML = "";
                const err = document.createElement("div");
                err.className = "koolook-snapshot-empty";
                err.textContent = e.message;
                list.appendChild(err);
                return;
            }

            selectedPath = data.path || "";
            choose.disabled = !selectedPath;
            renderSelectedFolder(current, selectedPath);
            current.title = selectedPath;
            upBtn.disabled = !data.parentPath;
            upBtn.onclick = () => { if (data.parentPath) render(data.parentPath); };

            rootsSelect.innerHTML = "";
            for (const root of data.roots || []) {
                const opt = document.createElement("option");
                opt.value = root.path;
                const isCurrentRoot = selectedPath &&
                    selectedPath.toLowerCase().startsWith(root.path.toLowerCase());
                opt.textContent = isCurrentRoot ? selectedPath : root.name;
                if (isCurrentRoot) {
                    opt.selected = true;
                }
                rootsSelect.appendChild(opt);
            }
            rootsSelect.onchange = () => render(rootsSelect.value);

            list.innerHTML = "";
            const dirs = Array.isArray(data.dirs) ? data.dirs : [];
            if (dirs.length === 0) {
                const empty = document.createElement("div");
                empty.className = "koolook-snapshot-empty";
                empty.textContent = "No folders inside selected folder.";
                list.appendChild(empty);
                return;
            }
            const childLabel = document.createElement("div");
            childLabel.className = "koolook-browse-list-label";
            childLabel.textContent = "Folders inside selected folder";
            list.appendChild(childLabel);
            for (const dir of dirs) {
                const row = document.createElement("button");
                row.type = "button";
                row.className = "koolook-browse-row";
                row.textContent = dir.name;
                row.title = dir.path;
                row.addEventListener("click", () => render(dir.path));
                list.appendChild(row);
            }
        }

        const cancel = makeModalButton({ label: "Cancel", onClick: closePicker });
        const newFolder = makeModalButton({
            label: "New folder...",
            onClick: () => {
                if (typeof createBrowseDirectory !== "function") return;
                showInputModal({
                    title: "New snapshot folder",
                    label: "Folder name",
                    defaultValue: "",
                    placeholder: "e.g. koolook-presets3",
                    confirmLabel: "Create",
                    subtitle: `Inside: ${selectedPath}`,
                    onSubmit: async (name) => {
                        const trimmed = name.trim();
                        if (!trimmed) {
                            toast("Folder name is empty.");
                            return;
                        }
                        try {
                            const result = await createBrowseDirectory(selectedPath, trimmed);
                            await render(result.path);
                            toast(`Created folder "${pathLeaf(result.path)}".`);
                        } catch (e) {
                            console.error("[Koolook] create folder failed:", e);
                            toast(e.message);
                        }
                    },
                });
            },
        });
        const choose = makeModalButton({
            label: "Use this folder",
            primary: true,
            onClick: () => {
                input.value = selectedPath || "";
                input.title = selectedPath || "";
                revealPathEnd();
                updateSaveState();
                closePicker();
            },
        });

        ({ overlay } = makeModalShell({
            title: "Browse snapshot folder",
            body: pickerBody,
            actions: [cancel, newFolder, choose],
        }));
        render(startPath);
    }

    const cancel = makeModalButton({ label: "Close", onClick: close });
    const reset = makeModalButton({
        label: "Reset to default",
        onClick: () => doSave(""),
    });
    const save = makeModalButton({
        label: "Save",
        onClick: () => doSave(input.value.trim()),
    });
    input.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && input.value.trim() !== savedLibraryPath) {
            doSave(input.value.trim());
        }
    });
    input.addEventListener("input", updateSaveState);

    ({ overlay } = makeModalShell({
        title: "Snapshot library settings",
        body,
        actions: [cancel, reset, save],
    }));
    updateSaveState();
    refresh();
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
