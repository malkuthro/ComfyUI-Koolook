// SPDX-FileCopyrightText: 2025-2026 Kforge Labs <https://github.com/malkuthro/ComfyUI-Koolook>
// SPDX-License-Identifier: GPL-3.0-only

import { getWorkflowGraph as defaultGetWorkflowGraph } from "./workflows_store.js";

const PUBLISH_ROUTE = "/koolook/api/setups";

function cloneJson(value) {
    return JSON.parse(JSON.stringify(value));
}

function normalizeTags(value) {
    if (Array.isArray(value)) {
        return value
            .filter(item => typeof item === "string")
            .map(item => item.trim())
            .filter(Boolean);
    }
    if (typeof value !== "string") return [];
    return value
        .split(",")
        .map(item => item.trim())
        .filter(Boolean);
}

function sourcePath(dirPath, wfName) {
    return [...dirPath, wfName].join("/");
}

function normalizeMetadata(metadata) {
    return {
        id: typeof metadata?.id === "string" ? metadata.id.trim() : "",
        title: typeof metadata?.title === "string" ? metadata.title.trim() : "",
        description: typeof metadata?.description === "string" ? metadata.description.trim() : "",
        category: typeof metadata?.category === "string" ? metadata.category.trim() : "",
        tags: normalizeTags(metadata?.tags),
        previewImage: typeof metadata?.previewImage === "string" ? metadata.previewImage.trim() : "",
    };
}

export async function publishSavedWorkflowSetup({
    dirPath,
    wfName,
    visualGraph,
    apiPrompt,
    captureApiPrompt,
    metadata,
    inputContract,
    outputContract,
    getWorkflowGraph = defaultGetWorkflowGraph,
    fetchImpl = fetch,
}) {
    const path = Array.isArray(dirPath) ? dirPath : [];
    const graph = (visualGraph && typeof visualGraph === "object")
        ? visualGraph
        : getWorkflowGraph(path, wfName);
    if (!graph || typeof graph !== "object") {
        throw new Error("Saved workflow not found.");
    }
    let resolvedApiPrompt = (apiPrompt && typeof apiPrompt === "object") ? apiPrompt : null;
    if (!resolvedApiPrompt && typeof captureApiPrompt === "function") {
        resolvedApiPrompt = await captureApiPrompt(graph);
        if (!resolvedApiPrompt || typeof resolvedApiPrompt !== "object") {
            throw new Error("API prompt capture failed. Publish needs ComfyUI's API-format prompt for this workflow.");
        }
    }

    const payload = {
        visualGraph: cloneJson(graph),
        ...(resolvedApiPrompt && typeof resolvedApiPrompt === "object" ? { apiPrompt: cloneJson(resolvedApiPrompt) } : {}),
        metadata: normalizeMetadata(metadata || {}),
        inputContract: cloneJson(inputContract || { inputs: [] }),
        outputContract: cloneJson(outputContract || { outputs: [] }),
        source: {
            kind: "sidebar-workflow",
            path: sourcePath(path, wfName),
            inventoryPath: cloneJson(path),
            name: typeof wfName === "string" ? wfName : "",
        },
    };

    const response = await fetchImpl(PUBLISH_ROUTE, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok || body.ok === false) {
        const errors = Array.isArray(body.errors) ? body.errors : [`Publish failed (${response.status}).`];
        throw new Error(errors.join("; "));
    }
    return body;
}
