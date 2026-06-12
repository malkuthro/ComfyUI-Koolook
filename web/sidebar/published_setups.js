// SPDX-FileCopyrightText: 2025-2026 Kforge Labs <https://github.com/malkuthro/ComfyUI-Koolook>
// SPDX-License-Identifier: GPL-3.0-only

import { getWorkflowGraph as defaultGetWorkflowGraph } from "./workflows_store.js";

const PUBLISH_ROUTE = "/koolook/api/setups";
const REVEAL_ROUTE = "/koolook/api/setups/reveal";

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

function asObject(value) {
    return (value && typeof value === "object") ? value : {};
}

function asString(value) {
    return typeof value === "string" ? value : "";
}

// Reduce a publish API response into the flat view-model the success card
// renders. Pure (no DOM) so the mapping is unit-testable; the modal renders
// it and wires actions from the boolean flags. Open folder / Copy path are
// only offered when the server told us where the setup landed.
export function buildPublishSuccessView(resultBody) {
    const body = asObject(resultBody);
    const setup = asObject(body.setup);
    const storagePath = asString(body.storagePath);
    return {
        setupId: asString(setup.id),
        title: asString(asObject(setup.metadata).title),
        sourcePath: asString(asObject(setup.source).path),
        validationStatus: asString(asObject(setup.validation).status),
        storagePath,
        canCopyPath: Boolean(storagePath),
        canOpenFolder: Boolean(storagePath),
    };
}

export async function revealPublishedSetupFolder({ fetchImpl = fetch } = {}) {
    const response = await fetchImpl(REVEAL_ROUTE, { method: "POST" });
    if (!response.ok) {
        let reason = "";
        try {
            reason = (await response.text() || "").trim();
        } catch (e) {
            /* fall through to status text */
        }
        throw new Error(reason || response.statusText || `Reveal failed (${response.status}).`);
    }
    return response.json();
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
    if (!resolvedApiPrompt) {
        throw new Error("API prompt capture is required before publishing this workflow.");
    }

    const payload = {
        visualGraph: cloneJson(graph),
        apiPrompt: cloneJson(resolvedApiPrompt),
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
