// SPDX-FileCopyrightText: 2025-2026 Kforge Labs <https://github.com/malkuthro/ComfyUI-Koolook>
// SPDX-License-Identifier: GPL-3.0-only

const TERMINAL_RUN_STATUSES = new Set(["succeeded", "failed", "lost"]);
const DEFAULT_LOCAL_COMFY_BASE = "http://127.0.0.1:8188";

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function normalizeBaseUrl(value) {
    return String(value || "").trim().replace(/\/+$/, "");
}

function apiUrl(baseUrl, path) {
    return `${normalizeBaseUrl(baseUrl)}${path}`;
}

function isFilePage() {
    return globalThis.location?.protocol === "file:";
}

function defaultBaseUrl() {
    return isFilePage() ? DEFAULT_LOCAL_COMFY_BASE : "";
}

async function readJsonResponse(response, fallbackMessage) {
    let body;
    try {
        body = await response.json();
    } catch (error) {
        const message = fallbackMessage || `Request failed (${response.status}).`;
        const responseError = new Error(`${message} Response was not valid JSON.`);
        responseError.status = response.status;
        throw responseError;
    }
    if (!response.ok || body.ok === false) {
        const errors = Array.isArray(body.errors) && body.errors.length
            ? body.errors
            : [fallbackMessage || `Request failed (${response.status}).`];
        const error = new Error(errors.join("; "));
        error.payload = body;
        error.status = response.status;
        throw error;
    }
    return body;
}

function responseShapeError(message, payload) {
    const error = new Error(message);
    error.payload = payload;
    return error;
}

function requireObject(value, message) {
    if (value && typeof value === "object" && !Array.isArray(value)) return value;
    throw responseShapeError(message, value);
}

export async function listPublishedSetups({ baseUrl = "", fetchImpl = fetch } = {}) {
    const response = await fetchImpl(apiUrl(baseUrl, "/koolook/api/setups"));
    const body = await readJsonResponse(response, `Setup list request failed (${response.status}).`);
    if (Array.isArray(body)) return body;
    if (body && typeof body === "object" && Array.isArray(body.setups)) return body.setups;
    throw responseShapeError("Setup list response must be an array or an object with a setups array.", body);
}

export async function getPublishedSetup({ setupId, baseUrl = "", fetchImpl = fetch }) {
    const id = typeof setupId === "string" ? setupId.trim() : "";
    if (!id) throw new Error("Published setup id is required.");
    const response = await fetchImpl(apiUrl(baseUrl, `/koolook/api/setups/${encodeURIComponent(id)}`));
    const body = await readJsonResponse(response, `Setup detail request failed (${response.status}).`);
    return requireObject(body, "Setup detail response must be an object.");
}

export async function runPublishedSetup({
    setupId,
    inputs = {},
    baseUrl = "",
    fetchImpl = fetch,
}) {
    const id = typeof setupId === "string" ? setupId.trim() : "";
    if (!id) throw new Error("Published setup id is required.");
    const response = await fetchImpl(apiUrl(baseUrl, `/koolook/api/setups/${encodeURIComponent(id)}/run`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ inputs: inputs && typeof inputs === "object" ? inputs : {} }),
    });
    const body = await readJsonResponse(response, `Run request failed (${response.status}).`);
    const payload = requireObject(body, "Run response must be an object.");
    return requireObject(payload.run, "Run response must include a run object.");
}

export async function getPublishedSetupRun({ runId, baseUrl = "", fetchImpl = fetch }) {
    const id = typeof runId === "string" ? runId.trim() : "";
    if (!id) throw new Error("Published setup run id is required.");
    const response = await fetchImpl(apiUrl(baseUrl, `/koolook/api/runs/${encodeURIComponent(id)}`));
    const body = await readJsonResponse(response, `Run status request failed (${response.status}).`);
    const payload = requireObject(body, "Run status response must be an object.");
    return requireObject(payload.run, "Run status response must include a run object.");
}

export async function runAndPollPublishedSetup({
    setupId,
    inputs = {},
    baseUrl = "",
    fetchImpl = fetch,
    intervalMs = 1000,
    timeoutMs = 30000,
    sleepImpl = sleep,
    onUpdate,
}) {
    const queued = await runPublishedSetup({ setupId, inputs, baseUrl, fetchImpl });
    if (typeof onUpdate === "function") onUpdate(queued);
    const runId = queued?.runId;
    if (!runId || TERMINAL_RUN_STATUSES.has(String(queued.status || "").toLowerCase())) {
        return queued;
    }
    const startedAt = Date.now();
    let lastRun = queued;
    while (Date.now() - startedAt <= timeoutMs) {
        await sleepImpl(intervalMs);
        lastRun = await getPublishedSetupRun({ runId, baseUrl, fetchImpl });
        if (typeof onUpdate === "function") onUpdate(lastRun);
        if (TERMINAL_RUN_STATUSES.has(String(lastRun?.status || "").toLowerCase())) {
            return lastRun;
        }
    }
    const error = new Error(`Timed out waiting for run ${runId}.`);
    error.run = lastRun;
    throw error;
}

export function formatRun(run) {
    if (!run || typeof run !== "object") return "No run payload.";
    const lines = [];
    if (run.runId) lines.push(`Run: ${run.runId}`);
    if (run.setupId) lines.push(`Setup: ${run.setupId}`);
    if (run.promptId) lines.push(`Prompt: ${run.promptId}`);
    if (run.status) lines.push(`Status: ${run.status}`);
    const resultLines = summarizeRunResultLines(run);
    if (resultLines.length) {
        lines.push("");
        lines.push("Result:");
        lines.push(...resultLines);
    }
    if (Array.isArray(run.outputs) && run.outputs.length) {
        lines.push("");
        lines.push("Outputs:");
        lines.push(JSON.stringify(run.outputs, null, 2));
    }
    const details = {};
    for (const [key, value] of Object.entries(run)) {
        if (!["runId", "setupId", "promptId", "status", "outputs"].includes(key)) {
            details[key] = value;
        }
    }
    if (Object.keys(details).length) {
        lines.push("");
        lines.push("Details:");
        lines.push(JSON.stringify(details, null, 2));
    }
    return lines.join("\n") || JSON.stringify(run, null, 2);
}

export function formatError(error) {
    if (error?.payload && typeof error.payload === "object") {
        return JSON.stringify(error.payload, null, 2);
    }
    if (error?.run && typeof error.run === "object") {
        return `${error.message}\n\nLast status:\n${formatRun(error.run)}`;
    }
    return error?.message || "Request failed.";
}

export function summarizeRunResultLines(run) {
    if (!Array.isArray(run?.outputs)) return [];
    const lines = [];
    for (const output of run.outputs) {
        const label = output?.label || output?.key || "Output";
        const items = Array.isArray(output?.items) ? output.items : [];
        for (const item of items) {
            if (item?.value == null || item.value === "") continue;
            lines.push(`${label}: ${String(item.value)}`);
        }
    }
    return lines;
}

export function setupDisplayTitle(setup) {
    return setup?.metadata?.title || setup?.title || setup?.id || "(untitled)";
}

function appSurface(setup) {
    const app = setup?.setupSurface?.app;
    return app && typeof app === "object" ? app : {};
}

export function setupHasAppSurface(setup) {
    const app = appSurface(setup);
    return Boolean(
        app.switch
        || (Array.isArray(app.inputs) && app.inputs.length)
        || (Array.isArray(app.outputs) && app.outputs.length)
    );
}

function fieldsByKey(fields) {
    const out = {};
    if (!Array.isArray(fields)) return out;
    for (const field of fields) {
        if (field && typeof field === "object" && typeof field.key === "string" && field.key) {
            out[field.key] = field;
        }
    }
    return out;
}

function visibleSwitchOptions(app) {
    const options = Array.isArray(app?.switch?.options) ? app.switch.options : [];
    return options.filter(option => option && typeof option === "object" && option.visible !== false);
}

function defaultSwitchValue(app) {
    const visibleOptions = visibleSwitchOptions(app);
    const defaultValue = app?.switch?.default;
    if (visibleOptions.some(option => option.value === defaultValue)) return defaultValue;
    return visibleOptions.length ? visibleOptions[0].value : defaultValue;
}

function fieldDefault(field) {
    return field?.default == null ? "" : field.default;
}

export function defaultRunInputsFromSetup(setup) {
    const app = appSurface(setup);
    const inputs = {};
    const switchField = app.switch && typeof app.switch === "object" ? app.switch : null;
    const selectedSwitch = switchField ? defaultSwitchValue(app) : undefined;
    if (switchField?.key && selectedSwitch !== undefined) {
        inputs[switchField.key] = selectedSwitch;
    }

    const inputFields = fieldsByKey(app.inputs);
    const selectedOption = visibleSwitchOptions(app).find(option => option.value === selectedSwitch);
    if (selectedOption?.input && inputFields[selectedOption.input]) {
        inputs[selectedOption.input] = fieldDefault(inputFields[selectedOption.input]);
    } else if (Array.isArray(app.inputs)) {
        for (const field of app.inputs) {
            if (field?.visible !== false && field?.key) inputs[field.key] = fieldDefault(field);
        }
    }
    if (Array.isArray(app.outputs)) {
        for (const field of app.outputs) {
            if (field?.visible !== false && field?.key) inputs[field.key] = fieldDefault(field);
        }
    }
    return inputs;
}

export function runInputsFromAppValues(setup, values = {}) {
    const app = appSurface(setup);
    const inputs = {};
    const switchField = app.switch && typeof app.switch === "object" ? app.switch : null;
    const selectedSwitch = switchField?.key && switchField.key in values
        ? values[switchField.key]
        : defaultSwitchValue(app);
    if (switchField?.key && selectedSwitch !== undefined) {
        inputs[switchField.key] = selectedSwitch;
    }
    const inputFields = fieldsByKey(app.inputs);
    const selectedOption = visibleSwitchOptions(app).find(option => String(option.value) === String(selectedSwitch));
    if (selectedOption?.input && inputFields[selectedOption.input]) {
        inputs[selectedOption.input] = values[selectedOption.input] ?? fieldDefault(inputFields[selectedOption.input]);
    } else if (Array.isArray(app.inputs)) {
        for (const field of app.inputs) {
            if (field?.visible !== false && field?.key) {
                inputs[field.key] = values[field.key] ?? fieldDefault(field);
            }
        }
    }
    if (Array.isArray(app.outputs)) {
        for (const field of app.outputs) {
            if (field?.visible !== false && field?.key) {
                inputs[field.key] = values[field.key] ?? fieldDefault(field);
            }
        }
    }
    return inputs;
}

function formatFetchFailure(error, baseUrl) {
    const message = formatError(error);
    if (!/failed to fetch|load failed|networkerror/i.test(message)) return message;
    const base = baseUrl || DEFAULT_LOCAL_COMFY_BASE;
    return [
        message,
        "",
        "The simulator could not reach the Koolook API.",
        `Check that ComfyUI is running at ${base}, or open this page from the Koolook route:`,
        `${base}/koolook/setup_runner_simulator.html`,
    ].join("\n");
}

let demoRunInputs = {};

function demoFetch(url, options = {}) {
    const path = String(url).replace(/^https?:\/\/[^/]+/, "");
    const ok = body => ({ ok: true, status: 200, async json() { return body; } });
    if (path.endsWith("/koolook/api/setups") && !options.method) {
        return ok({
            ok: true,
            setups: [
                {
                    id: "director-demo",
                    title: "Director Demo",
                    description: "Sample published setup",
                    category: "Video",
                    tags: ["demo"],
                    validation: { status: "valid" },
                },
            ],
        });
    }
    if (path.endsWith("/koolook/api/setups/director-demo")) {
        return ok({
            id: "director-demo",
            metadata: { title: "Director Demo", description: "Sample published setup" },
            inputContract: { inputs: [] },
            outputContract: { outputs: [{ key: "video", label: "Video", type: "video" }] },
            setupSurface: {
                app: {
                    switch: {
                        key: "switch",
                        label: "Input type",
                        default: 2,
                        options: [
                            { value: 0, label: "EXR", input: "sequence_folder", visible: true },
                            { value: 1, label: "QT", input: "qt_file", visible: true },
                            { value: 2, label: "Img", input: "single_file", visible: true },
                            { value: 3, label: "Prompt", input: "prompt", visible: false },
                        ],
                    },
                    inputs: [
                        { key: "sequence_folder", label: "Sequence folder", visible: true, default: "/shots/demo/plates" },
                        { key: "qt_file", label: "QT file", visible: true, default: "/shots/demo/source.mov" },
                        { key: "single_file", label: "Single file", visible: true, default: "/shots/demo/source.png" },
                        { key: "prompt", label: "Prompt", visible: false, default: "" },
                    ],
                    outputs: [
                        { key: "folder", label: "Output folder", visible: true, default: "/shots/demo/output" },
                        { key: "name", label: "Output name", visible: true, default: "demo" },
                        { key: "version", label: "Version", visible: true, default: "1" },
                    ],
                    results: [{ key: "result", label: "Result", visible: true, default: "" }],
                },
            },
            validation: { status: "valid" },
        });
    }
    if (path.endsWith("/koolook/api/setups/director-demo/run")) {
        const request = JSON.parse(options.body || "{}");
        demoRunInputs = request.inputs && typeof request.inputs === "object" ? request.inputs : {};
        return ok({ ok: true, run: { runId: "run-000001", promptId: "prompt-123", status: "queued" } });
    }
    if (path.endsWith("/koolook/api/runs/run-000001")) {
        const folder = String(demoRunInputs.folder || "/shots/demo/output").replace(/\/+$/, "");
        const name = String(demoRunInputs.name || "demo");
        const rawVersion = String(demoRunInputs.version || "1");
        const version = rawVersion.padStart(3, "0");
        const extension = Number(demoRunInputs.switch) === 1 ? "mov" : "png";
        return ok({
            ok: true,
            run: {
                runId: "run-000001",
                setupId: "director-demo",
                promptId: "prompt-123",
                status: "succeeded",
                comfyStatus: { completed: true, status_str: "success" },
                outputs: [
                    {
                        key: "result",
                        label: "Result",
                        type: "result",
                        items: [{ nodeId: "300", kind: "text", value: `${folder}/${name}_v${version}.${extension}` }],
                    },
                ],
            },
        });
    }
    return { ok: false, status: 404, async json() { return { ok: false, errors: ["demo route not found"] }; } };
}

function bindSimulator(documentRef, fetchImpl) {
    const root = documentRef.querySelector("[data-simulator]");
    if (!root) return;
    const baseInput = root.querySelector("[data-base-url]");
    const loadBtn = root.querySelector("[data-load-setups]");
    const setupSelect = root.querySelector("[data-setup-select]");
    const inputsText = root.querySelector("[data-inputs-json]");
    const appForm = root.querySelector("[data-app-form]");
    const rawInputs = root.querySelector("[data-raw-inputs]");
    const runBtn = root.querySelector("[data-run-setup]");
    const statusText = root.querySelector("[data-status]");
    const detailText = root.querySelector("[data-detail]");
    const routeLink = root.querySelector("[data-route-link]");

    const state = { setups: [], selectedSetup: null, formActive: false };
    const baseUrl = () => baseInput.value.trim();
    const setStatus = value => { statusText.textContent = value; };
    const setDetail = value => { detailText.textContent = value; };

    function updatePayloadPreview() {
        if (!state.selectedSetup) {
            inputsText.value = "{}";
            return;
        }
        if (state.formActive) {
            inputsText.value = JSON.stringify(collectFormInputs(), null, 2);
        }
    }

    function fieldLabelText(field) {
        return field?.label || field?.key || "Field";
    }

    function createInputField(field, { name, value } = {}) {
        const label = documentRef.createElement("label");
        label.dataset.fieldKey = name || field.key;
        label.textContent = fieldLabelText(field);
        const input = documentRef.createElement("input");
        input.dataset.appField = name || field.key;
        input.value = value == null ? "" : String(value);
        label.appendChild(input);
        input.addEventListener("input", updatePayloadPreview);
        return label;
    }

    function renderAppForm(setup) {
        if (!appForm) return;
        appForm.innerHTML = "";
        state.selectedSetup = setup;
        state.formActive = setupHasAppSurface(setup);
        if (inputsText) inputsText.readOnly = state.formActive;
        if (rawInputs) rawInputs.open = !state.formActive;
        const app = appSurface(setup);
        const inputFields = fieldsByKey(app.inputs);
        const options = visibleSwitchOptions(app);
        if (!state.formActive) {
            appForm.textContent = "This setup does not expose an app form yet. Use raw inputs JSON below.";
            if (inputsText) inputsText.value = "{}";
            return;
        }

        const heading = documentRef.createElement("div");
        heading.className = "form-heading";
        heading.textContent = "External app form";
        appForm.appendChild(heading);

        if (app.switch && options.length) {
            const switchLabel = documentRef.createElement("label");
            switchLabel.textContent = app.switch.label || "Input type";
            const select = documentRef.createElement("select");
            select.dataset.appSwitch = app.switch.key || "switch";
            const selectedDefault = defaultSwitchValue(app);
            for (const option of options) {
                const opt = documentRef.createElement("option");
                opt.value = String(option.value);
                opt.textContent = option.label || String(option.value);
                if (String(option.value) === String(selectedDefault)) opt.selected = true;
                select.appendChild(opt);
            }
            switchLabel.appendChild(select);
            appForm.appendChild(switchLabel);
            const sourceWrap = documentRef.createElement("div");
            sourceWrap.dataset.sourceFieldWrap = "1";
            appForm.appendChild(sourceWrap);
            const renderSourceField = () => {
                sourceWrap.innerHTML = "";
                const selected = options.find(option => String(option.value) === String(select.value));
                const field = selected?.input ? inputFields[selected.input] : null;
                if (field) sourceWrap.appendChild(createInputField(field, { name: selected.input, value: fieldDefault(field) }));
                updatePayloadPreview();
            };
            select.addEventListener("change", renderSourceField);
            renderSourceField();
        } else if (Array.isArray(app.inputs)) {
            for (const field of app.inputs) {
                if (field?.visible !== false && field?.key) {
                    appForm.appendChild(createInputField(field, { value: fieldDefault(field) }));
                }
            }
        }

        const visibleOutputs = Array.isArray(app.outputs)
            ? app.outputs.filter(field => field?.visible !== false && field?.key)
            : [];
        if (visibleOutputs.length) {
            const outputHeading = documentRef.createElement("div");
            outputHeading.className = "form-heading";
            outputHeading.textContent = "Output";
            appForm.appendChild(outputHeading);
            for (const field of visibleOutputs) {
                appForm.appendChild(createInputField(field, { value: fieldDefault(field) }));
            }
        }
        updatePayloadPreview();
    }

    function collectFormInputs() {
        if (!state.selectedSetup) return {};
        const values = {};
        const switchEl = appForm?.querySelector("[data-app-switch]");
        if (switchEl) {
            const numeric = Number(switchEl.value);
            values[switchEl.dataset.appSwitch] = Number.isNaN(numeric) ? switchEl.value : numeric;
        }
        for (const input of appForm?.querySelectorAll("[data-app-field]") || []) {
            values[input.dataset.appField] = input.value;
        }
        return runInputsFromAppValues(state.selectedSetup, values);
    }

    async function loadSetups() {
        setStatus("Loading setups...");
        setDetail("");
        runBtn.disabled = true;
        try {
            state.setups = await listPublishedSetups({ baseUrl: baseUrl(), fetchImpl });
            setupSelect.innerHTML = "";
            for (const setup of state.setups) {
                const opt = documentRef.createElement("option");
                opt.value = setup.id || "";
                opt.textContent = setupDisplayTitle(setup);
                setupSelect.appendChild(opt);
            }
            runBtn.disabled = state.setups.length === 0;
            setStatus(state.setups.length ? `${state.setups.length} setup(s) loaded.` : "No published setups found.");
            if (state.setups.length) await showSelectedSetup();
        } catch (error) {
            setStatus("Setup list failed.");
            setDetail(formatFetchFailure(error, baseUrl()));
        }
    }

    async function showSelectedSetup() {
        if (!setupSelect.value) return;
        try {
            const setup = await getPublishedSetup({ setupId: setupSelect.value, baseUrl: baseUrl(), fetchImpl });
            renderAppForm(setup);
            setDetail(JSON.stringify(setup, null, 2));
        } catch (error) {
            setDetail(formatFetchFailure(error, baseUrl()));
        }
    }

    async function runSelectedSetup() {
        if (!setupSelect.value) return;
        let inputs;
        try {
            inputs = state.formActive ? collectFormInputs() : JSON.parse(inputsText.value || "{}");
        } catch (error) {
            setStatus("Inputs JSON is invalid.");
            setDetail(error.message);
            return;
        }
        runBtn.disabled = true;
        setStatus("Queueing setup...");
        try {
            const finalRun = await runAndPollPublishedSetup({
                setupId: setupSelect.value,
                inputs,
                baseUrl: baseUrl(),
                fetchImpl,
                onUpdate: run => {
                    setStatus(`Run ${run.status || "unknown"}`);
                    setDetail(formatRun(run));
                },
            });
            setStatus(`Run ${finalRun?.status || "finished"}`);
            setDetail(formatRun(finalRun));
        } catch (error) {
            setStatus("Run failed.");
            setDetail(formatFetchFailure(error, baseUrl()));
        } finally {
            runBtn.disabled = state.setups.length === 0;
        }
    }

    loadBtn.addEventListener("click", loadSetups);
    setupSelect.addEventListener("change", showSelectedSetup);
    runBtn.addEventListener("click", runSelectedSetup);
    const routeBase = isFilePage() ? DEFAULT_LOCAL_COMFY_BASE : globalThis.location.origin;
    if (routeLink) routeLink.href = `${routeBase}/koolook/setup_runner_simulator.html`;
    baseInput.value = defaultBaseUrl();
    if (isFilePage()) {
        setStatus("Enter or confirm the ComfyUI API base, then load setups.");
        setDetail([
            "You opened this page from disk.",
            "",
            "For the live API path, run ComfyUI and either:",
            `- keep ${DEFAULT_LOCAL_COMFY_BASE} as the API base and click Load setups`,
            `- open ${DEFAULT_LOCAL_COMFY_BASE}/koolook/setup_runner_simulator.html`,
            "",
            "If you only want to verify the simulator UI, open demo mode:",
            "setup_runner_simulator.html?demo=1",
        ].join("\n"));
    }
    if (new URLSearchParams(globalThis.location?.search || "").get("demo") === "1") {
        loadSetups();
    }
}

if (typeof document !== "undefined") {
    const fetchImpl = new URLSearchParams(globalThis.location?.search || "").get("demo") === "1"
        ? demoFetch
        : fetch;
    bindSimulator(document, fetchImpl);
}
