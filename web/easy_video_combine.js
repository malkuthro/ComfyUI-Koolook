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

app.registerExtension({
    name: "Koolook.EasyVideoCombine",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData?.name === "Easy_VideoCombine") {
            addFormatWidgets(nodeType);
        }
    },
});
