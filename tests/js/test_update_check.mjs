// Behavior tests for the sidebar update-available check.

import assert from "node:assert/strict";
import path from "node:path";

const repoRoot = path.resolve(import.meta.dirname, "../..");
const updateCheck = await import(
  `file:///${path.join(repoRoot, "web", "sidebar", "update_check.js").replaceAll("\\", "/")}`
);

function jsonResponse(body, ok = true) {
  return {
    ok,
    async json() {
      return body;
    },
  };
}

assert.equal(updateCheck.compareSemver("0.4.1", "0.4.0"), 1);
assert.equal(updateCheck.compareSemver("v0.4.0", "0.4.0"), 0);
assert.equal(updateCheck.compareSemver("0.3.9", "0.4.0"), -1);
assert.equal(updateCheck.compareSemver("0.4.0", "0.4.0-rc.1"), 1);
assert.equal(updateCheck.compareSemver("not-a-version", "0.4.0"), 0);

{
  const calls = [];
  const update = await updateCheck.checkForUpdate(async (url) => {
    calls.push(String(url));
    if (calls.length === 1) {
      assert.ok(calls[0].startsWith("/koolook/api/version?_="));
      return jsonResponse({
        version: "0.4.0",
        latestReleaseApiUrl: "https://api.example.test/latest",
      });
    }
    assert.equal(url, "https://api.example.test/latest");
    return jsonResponse({
      tag_name: "v0.4.1",
      html_url: "https://github.com/malkuthro/ComfyUI-Koolook/releases/tag/v0.4.1",
    });
  });
  assert.deepEqual(update, {
    current: "0.4.0",
    latest: "0.4.1",
    url: "https://github.com/malkuthro/ComfyUI-Koolook/releases/tag/v0.4.1",
  });
}

{
  let calls = 0;
  const update = await updateCheck.checkForUpdate(async () => {
    calls += 1;
    if (calls === 1) return jsonResponse({ version: "0.4.0" });
    return jsonResponse({
      tag_name: "v0.4.1",
      html_url: "https://github.example.test/releases/v0.4.1",
    });
  });
  assert.deepEqual(update, {
    current: "0.4.0",
    latest: "0.4.1",
    url: "https://github.com/malkuthro/ComfyUI-Koolook/releases/tag/v0.4.1",
  });
}

{
  const update = await updateCheck.checkForUpdate(async () => jsonResponse({ version: "0.4.0" }));
  assert.equal(update, null);
}

{
  let calls = 0;
  const update = await updateCheck.checkForUpdate(async () => {
    calls += 1;
    if (calls === 1) return jsonResponse({ version: "0.4.0" });
    return jsonResponse({ tag_name: "v0.4.0", html_url: "https://example.test/current" });
  });
  assert.equal(update, null);
}

{
  const update = await updateCheck.checkForUpdate(async () => {
    throw new Error("offline");
  });
  assert.equal(update, null);
}

console.log("test_update_check.mjs: all scenarios passed");
