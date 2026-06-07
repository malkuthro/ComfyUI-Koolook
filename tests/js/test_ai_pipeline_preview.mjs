import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";

const repoRoot = path.resolve(import.meta.dirname, "../..");
const sourcePath = path.join(repoRoot, "web", "ai_pipeline.js");
const source = fs.readFileSync(sourcePath, "utf8")
  .replace(/^import .*;\r?\n/gm, "");

const registeredExtensions = [];
const context = {
  bulletproofStringWidget(widget, fallback = "") {
    if (!widget) return;
    if (widget.value == null) widget.value = fallback;
    else if (typeof widget.value !== "string") widget.value = String(widget.value);
  },
  app: {
    graph: null,
    registerExtension(extension) {
      registeredExtensions.push(extension);
    },
  },
  ComfyWidgets: {
    STRING(node) {
      const previewWidget = {
        value: "",
        inputEl: { readOnly: false, style: {} },
      };
      node.widgets.push(previewWidget);
      return {
        widget: previewWidget,
      };
    },
  },
  globalThis: {},
};
context.globalThis = context;

vm.runInNewContext(source, context, { filename: sourcePath });

const extension = registeredExtensions.find((entry) => entry.name === "koolook.ai_pipeline");
assert.ok(extension, "AI pipeline extension should register");

function widget(name, value) {
  return { name, value, inputEl: { readOnly: false, style: {} } };
}

function makeGraph(nodes, links) {
  return {
    _nodes: nodes,
    _links: Object.fromEntries(links.map((link) => [link.id, link])),
    getLink(id) {
      return this._links[id] ?? null;
    },
    getNodeById(id) {
      return this._nodes.find((node) => node.id === id) ?? null;
    },
  };
}

function makePipelineNode(graph) {
  const buttons = [];
  return {
    graph,
    widgets: [
      widget("shot_duration", 81),
      widget("seed_value", 453453453),
      widget("instruction", "Place your base folder path in the FIELD below"),
      widget("base_directory_path", "e:/G-Drive-BaconX/Jobs/Jeep_Animals/ComfyUI_LTX23"),
      widget("extension", ".%04d.exr"),
      widget("shot_name", "stale_widget_name"),
      widget("ai_method", ""),
      widget("version", "001"),
      widget("disable_versioning", false),
      widget("enable_overwrite", false),
      widget("no_subfolders", false),
    ],
    inputs: [
      { name: "shot_duration", link: null, widget: { name: "shot_duration" } },
      { name: "seed_value", link: null, widget: { name: "seed_value" } },
      { name: "instruction", link: null, widget: { name: "instruction" } },
      { name: "base_directory_path", link: null, widget: { name: "base_directory_path" } },
      { name: "extension", link: null, widget: { name: "extension" } },
      { name: "shot_name", link: 13, widget: { name: "shot_name" } },
      { name: "ai_method", link: null, widget: { name: "ai_method" } },
      { name: "version", link: 14, widget: { name: "version" } },
      { name: "disable_versioning", link: null, widget: { name: "disable_versioning" } },
      { name: "enable_overwrite", link: null, widget: { name: "enable_overwrite" } },
      { name: "no_subfolders", link: null, widget: { name: "no_subfolders" } },
    ],
    outputs: [],
    addWidget(type, name, value, callback) {
      const button = { type, name, value, callback };
      buttons.push(button);
      return button;
    },
    setDirtyCanvas() {},
    buttons,
  };
}

const namePart = {
  id: 1,
  type: "Text Multiline",
  widgets_values: ["Bear_3x-FR_w.Audio"],
  outputs: [{ name: "STRING", type: "STRING", links: [11] }],
};
const suffixPart = {
  id: 2,
  type: "Text Multiline",
  widgets_values: ["EXR"],
  outputs: [{ name: "STRING", type: "STRING", links: [12] }],
};
const concat = {
  id: 3,
  type: "Text Concatenate",
  widgets_values: ["_", "true"],
  inputs: [
    { name: "text_a", type: "STRING", link: 11 },
    { name: "text_b", type: "STRING", link: 12 },
    { name: "text_c", type: "STRING", link: null },
    { name: "text_d", type: "STRING", link: null },
  ],
  outputs: [{ name: "STRING", type: "STRING", links: [13] }],
};
const versionSource = {
  id: 4,
  type: "Text Multiline",
  widgets_values: ["003"],
  outputs: [{ name: "STRING", type: "STRING", links: [14] }],
};

const graph = makeGraph(
  [namePart, suffixPart, concat, versionSource],
  [
    { id: 11, origin_id: 1, origin_slot: 0, target_id: 3, target_slot: 0, type: "STRING" },
    { id: 12, origin_id: 2, origin_slot: 0, target_id: 3, target_slot: 1, type: "STRING" },
    { id: 13, origin_id: 3, origin_slot: 0, target_id: 99, target_slot: 5, type: "STRING" },
    { id: 14, origin_id: 4, origin_slot: 0, target_id: 99, target_slot: 7, type: "STRING" },
  ],
);
context.app.graph = graph;

function previewFilePath() {
  const nodeType = function EasyAIPipelineNode() {};
  extension.beforeRegisterNodeDef(nodeType, { name: "EasyAIPipeline" });
  const node = makePipelineNode(graph);
  nodeType.prototype.onNodeCreated.call(node);
  const button = node.buttons.find((entry) => entry.name === "Get output file path");
  assert.ok(button, "preview file-path button should exist");
  button.callback();
  return node.widgets.find((entry) => entry.inputEl?.readOnly && entry.inputEl?.style?.height === "100px").value;
}

const expected = "e:/G-Drive-BaconX/Jobs/Jeep_Animals/ComfyUI_LTX23/Bear_3x-FR_w.Audio_EXR/v003/Bear_3x-FR_w.Audio_EXR_v003.%04d.exr";

const first = previewFilePath();
const second = previewFilePath();

assert.equal(first, expected);
assert.equal(second, expected);
assert.ok(!first.includes("/_/"), "Text Concatenate delimiter must not replace the shot name");
assert.ok(!first.includes("__"), "empty fields must not multiply underscores");
assert.ok(!first.includes("vv003"), "already-normalized version tokens must not get an extra v");

{
  const publishOutput = {
    id: 10,
    type: "Koolook_PublishOutput",
    widgets: [
      widget("folder", "/Volumes/Data/G-Drive-BaconX/Jobs/OndtBlod/001_0035/ai/publish-OUT"),
      widget("name", "publish-OUT"),
      widget("version", "1"),
    ],
    outputs: [
      { name: "folder", type: "STRING", links: [30] },
      { name: "name", type: "STRING", links: [] },
      { name: "version", type: "STRING", links: [] },
    ],
  };
  const reroute = {
    id: 11,
    type: "Reroute",
    inputs: [{ name: "", type: "STRING", link: 30 }],
    outputs: [{ name: "", type: "STRING", links: [31] }],
  };
  const publishGraph = makeGraph(
    [publishOutput, reroute],
    [
      { id: 30, origin_id: 10, origin_slot: 0, target_id: 11, target_slot: 0, type: "STRING" },
      { id: 31, origin_id: 11, origin_slot: 0, target_id: 99, target_slot: 3, type: "STRING" },
    ],
  );
  context.app.graph = publishGraph;

  const nodeType = function EasyAIPipelineNode() {};
  extension.beforeRegisterNodeDef(nodeType, { name: "EasyAIPipeline" });
  const node = makePipelineNode(publishGraph);
  node.inputs.find((input) => input.name === "base_directory_path").link = 31;
  node.inputs.find((input) => input.name === "shot_name").link = null;
  node.inputs.find((input) => input.name === "version").link = null;
  node.widgets.find((entry) => entry.name === "shot_name").value = "mask";
  node.widgets.find((entry) => entry.name === "version").value = "1";
  node.widgets.find((entry) => entry.name === "no_subfolders").value = true;
  nodeType.prototype.onNodeCreated.call(node);
  const button = node.buttons.find((entry) => entry.name === "Get output directory path");
  assert.ok(button, "preview directory button should exist");
  button.callback();

  const preview = node.widgets.find((entry) => entry.inputEl?.readOnly && entry.inputEl?.style?.height === "100px").value;
  assert.equal(
    preview,
    "/Volumes/Data/G-Drive-BaconX/Jobs/OndtBlod/001_0035/ai/publish-OUT/v001",
  );
}
