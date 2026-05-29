// SPDX-License-Identifier: GPL-3.0-or-later
//
// Shared frontend widget helpers for ComfyUI-Koolook nodes.

// Pysssss's presetText.js wraps every STRING widget's serializeValue with a
// callback chain that does `value.replace(...)`. If widget.value is undefined
// or null (widget-to-input conversion voiding the value, or a workflow saved
// before this widget existed), `.replace` throws and graphToPrompt aborts.
// Trap reads/writes via a property descriptor so widget.value is always a
// string regardless of caller.
export function bulletproofStringWidget(widget, fallback = "") {
    if (!widget) return;
    let stored;
    const initial = widget.value;
    if (typeof initial === "string") {
        stored = initial;
    } else if (initial == null) {
        stored = fallback;
    } else {
        stored = String(initial);
    }
    Object.defineProperty(widget, "value", {
        configurable: true,
        enumerable: true,
        get() { return stored; },
        set(v) {
            if (typeof v === "string") stored = v;
            else if (v == null) stored = fallback;
            else stored = String(v);
        },
    });
}
