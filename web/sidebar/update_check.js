// SPDX-License-Identifier: GPL-3.0-only

import { noStoreUrl } from "./constants.js";

const LOCAL_VERSION_ROUTE = "/koolook/api/version";
const FALLBACK_LATEST_RELEASE_API =
    "https://api.github.com/repos/malkuthro/ComfyUI-Koolook/releases/latest";
const RELEASES_URL = "https://github.com/malkuthro/ComfyUI-Koolook/releases";

function releaseNotesUrl(version, rawUrl) {
    const expected = `${RELEASES_URL}/tag/v${version}`;
    if (typeof rawUrl !== "string") return expected;
    try {
        const parsed = new URL(rawUrl);
        if (
            parsed.protocol === "https:" &&
            parsed.hostname === "github.com" &&
            parsed.pathname === `/malkuthro/ComfyUI-Koolook/releases/tag/v${version}`
        ) {
            return parsed.href;
        }
    } catch (_err) {
        // Fall through to the canonical release URL.
    }
    return expected;
}

function parseVersion(value) {
    if (typeof value !== "string") return null;
    const match = value.trim().match(/^v?(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?(?:\+.*)?$/);
    if (!match) return null;
    return {
        parts: match.slice(1, 4).map((part) => Number(part)),
        prerelease: match[4] || "",
    };
}

export function parseSemver(value) {
    const parsed = parseVersion(value);
    return parsed ? parsed.parts : null;
}

export function compareSemver(a, b) {
    const av = parseVersion(a);
    const bv = parseVersion(b);
    if (!av || !bv) return 0;
    for (let i = 0; i < 3; i += 1) {
        if (av.parts[i] > bv.parts[i]) return 1;
        if (av.parts[i] < bv.parts[i]) return -1;
    }
    if (av.prerelease && !bv.prerelease) return -1;
    if (!av.prerelease && bv.prerelease) return 1;
    return 0;
}

function normalizeLatestRelease(payload) {
    if (!payload || typeof payload !== "object") return null;
    const version = typeof payload.tag_name === "string"
        ? payload.tag_name.replace(/^v/i, "")
        : "";
    if (!parseSemver(version)) return null;
    return {
        version,
        url: releaseNotesUrl(version, payload.html_url),
    };
}

export async function checkForUpdate(fetchImpl = fetch) {
    try {
        const localResponse = await fetchImpl(noStoreUrl(LOCAL_VERSION_ROUTE), {
            cache: "no-store",
        });
        if (!localResponse || !localResponse.ok) return null;
        const local = await localResponse.json();
        const current = typeof local.version === "string" ? local.version : "";
        if (!parseSemver(current)) return null;

        const latestUrl = typeof local.latestReleaseApiUrl === "string" && local.latestReleaseApiUrl
            ? local.latestReleaseApiUrl
            : FALLBACK_LATEST_RELEASE_API;
        const latestResponse = await fetchImpl(latestUrl, { cache: "no-store" });
        if (!latestResponse || !latestResponse.ok) return null;
        const latest = normalizeLatestRelease(await latestResponse.json());
        if (!latest || compareSemver(latest.version, current) <= 0) return null;

        return {
            current,
            latest: latest.version,
            url: latest.url,
        };
    } catch (_err) {
        return null;
    }
}

export function renderUpdateFooter(footerEl, update) {
    footerEl.textContent = "";
    footerEl.classList.remove("koolook-update-footer-visible");
    if (!update) return;

    const label = document.createElement("span");
    label.textContent = "Update Available";
    const link = document.createElement("a");
    link.href = update.url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = "Release notes";
    link.title = `Kforge Labs ${update.latest} release notes`;
    footerEl.append(label, link);
    footerEl.classList.add("koolook-update-footer-visible");
}
