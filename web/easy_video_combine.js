// SPDX-License-Identifier: GPL-3.0-or-later
//
// ComfyUI-Koolook — Easy Video Combine frontend extension
// Copyright (C) 2026 ComfyUI-Koolook contributors (kforgelabs).
//
// VHS's frontend extension (web/js/VHS.core.js:2030) hard-codes
// `nodeData?.name == "VHS_VideoCombine"` to gate its format-widget
// machinery — picking a format like ProRes or h264 dynamically adds
// format-specific widgets (`profile`, `crf`, `pix_fmt`, …). Because
// our subclass is registered under a different node ID
// (`Easy_VideoCombine`), none of that code runs for our node, and
// the format-specific widgets never render. The backend then warns
// "Missing input for <widget> has been set to <default>" on every
// execution and the user has no way to choose anything other than
// the default.
//
// This extension replicates the relevant bit of VHS's logic
// (`addFormatWidgets`) for our node, so the format dropdown's
// callback adds/removes per-format widgets the same way as upstream.
// Self-contained — does NOT import from VHS's module (their helpers
// aren't exported). Only ~80 lines including comments.

import { app } from "../../scripts/app.js";
import { setWidgetConfig } from "../../extensions/core/widgetInputs.js";

// Pysssss's presetText.js wraps every STRING widget's serializeValue with a
// callback chain that does `value.replace(...)`. If widget.value is undefined
// or null (widget-to-input conversion voiding the value, or a workflow saved
// before this widget existed), `.replace` throws and graphToPrompt aborts —
// Run becomes a no-op for the whole graph. Trap reads/writes via a property
// descriptor so widget.value is always a string regardless of caller.
function bulletproofStringWidget(widget, fallback = "") {
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

function chainCallback(object, property, callback) {
    if (object == undefined) {
        console.error("[Koolook EasyVideoCombine] chainCallback on undefined object");
        return;
    }
    if (property in object && object[property]) {
        const original = object[property];
        object[property] = function () {
            const r = original.apply(this, arguments);
            return callback.apply(this, arguments) ?? r;
        };
    } else {
        object[property] = callback;
    }
}

function fitHeight(node) {
    node.setSize([node.size[0], node.computeSize([node.size[0], node.size[1]])[1]]);
    node?.graph?.setDirtyCanvas(true);
}

function setWidgetValue(widget, value) {
    widget.value = value;
    widget.callback?.(widget.value);
}

function coerceBoolValue(value, fallback) {
    if (typeof value === "boolean") return value;
    if (typeof value === "number") return value !== 0;
    if (value == null) return fallback;
    const s = String(value).trim().toLowerCase();
    if (["true", "1", "yes", "on"].includes(s)) return true;
    if (["false", "0", "no", "off", ""].includes(s)) return false;
    return fallback;
}

const BOOLEAN_WIDGETS = new Set([
    "pingpong",
    "create_path_if_missing",
    "save_metadata_png",
    "save_metadata_json",
    "keep_silent_intermediate",
]);

function setSavedWidgetValue(node, name, value) {
    const widget = node.widgets?.find((w) => w.name === name);
    if (!widget || widget.type === "button") return;
    if (BOOLEAN_WIDGETS.has(name)) {
        setWidgetValue(widget, coerceBoolValue(value, widget.value));
    } else {
        setWidgetValue(widget, value);
    }
    const input = node.inputs?.find((i) => i.name === name);
    if (input && widget.config) setWidgetConfig(input, widget.config);
}

function useNamedWidgetState(nodeType) {
    chainCallback(nodeType.prototype, "onNodeCreated", function () {
        chainCallback(this, "onConfigure", function (info) {
            if (!this.widgets || info?.widgets_values == null) return;

            const saved = info.widgets_values;
            if (!Array.isArray(saved) && typeof saved === "object") {
                for (const name of ["frame_rate", "loop_count", "filename_prefix"]) {
                    if (Object.hasOwn(saved, name)) setSavedWidgetValue(this, name, saved[name]);
                }
                // Restore format before the rest; its callback creates the
                // format-specific widgets that also need named restoration.
                if (Object.hasOwn(saved, "format")) {
                    setSavedWidgetValue(this, "format", saved.format);
                }
                for (const [name, value] of Object.entries(saved)) {
                    setSavedWidgetValue(this, name, value);
                }
            }
        });

        chainCallback(this, "onSerialize", function (info) {
            if (!this.widgets) return;
            info.widgets_values = {};
            for (const widget of this.widgets) {
                if (widget.type === "button") continue;
                info.widgets_values[widget.name] = widget.value;
            }
        });
    });
}

function addFormatWidgets(nodeType) {
    chainCallback(nodeType.prototype, "onNodeCreated", function () {
        let formatWidget = null;
        let formatWidgetIndex = -1;
        for (let i = 0; i < this.widgets.length; i++) {
            if (this.widgets[i].name === "format") {
                formatWidget = this.widgets[i];
                formatWidgetIndex = i + 1;
                break;
            }
        }
        if (!formatWidget) return;
        let formatWidgetsCount = 0;
        chainCallback(formatWidget, "callback", (value) => {
            const formats = LiteGraph.registered_node_types[this.type]
                ?.nodeData?.input?.required?.format?.[1]?.formats;
            const newWidgets = [];
            if (formats?.[value]) {
                for (const wDef of formats[value]) {
                    let type = wDef[2]?.widgetType ?? wDef[1];
                    if (Array.isArray(type)) type = "COMBO";
                    app.widgets[type](this, wDef[0], wDef.slice(1), app);
                    const w = this.widgets.pop();
                    w.config = wDef.slice(1);
                    newWidgets.push(w);
                }
            }
            const removed = this.widgets.splice(formatWidgetIndex, formatWidgetsCount, ...newWidgets);
            const newNames = new Set(newWidgets.map((w) => w.name));
            for (const w of removed) {
                w?.onRemove?.();
                if (newNames.has(w.name)) continue;
                const slot = this.inputs.findIndex((i) => i.name == w.name);
                if (slot >= 0) this.removeInput(slot);
            }
            for (const w of newWidgets) {
                const existing = this.inputs.find((i) => i.name == w.name);
                if (existing) {
                    setWidgetConfig(existing, w.config);
                } else {
                    this.addInput(w.name, w.config[0], { widget: { name: w.name } });
                }
            }
            fitHeight(this);
            formatWidgetsCount = newWidgets.length;
        });
    });
}

function bulletproofVersionWidget(nodeType) {
    chainCallback(nodeType.prototype, "onNodeCreated", function () {
        // Run after onConfigure replays widgets_values. Linked-input version
        // widgets get value=undefined from LiteGraph; pre-PR#180 workflows
        // have no entry for version in the dict at all. Either trips pysssss.
        chainCallback(this, "onConfigure", function () {
            bulletproofStringWidget(this.widgets?.find((w) => w.name === "version"), "");
        });
        // Also bulletproof on plain creation (new node, no onConfigure).
        bulletproofStringWidget(this.widgets?.find((w) => w.name === "version"), "");
    });
}

app.registerExtension({
    name: "Koolook.EasyVideoCombine",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData?.name === "Easy_VideoCombine") {
            useNamedWidgetState(nodeType);
            addFormatWidgets(nodeType);
            bulletproofVersionWidget(nodeType);
        }
    },
});
