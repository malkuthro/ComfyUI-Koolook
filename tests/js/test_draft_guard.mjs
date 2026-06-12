// Behavior tests for web/koolook_draft_guard.js — the generation-agnostic
// guard around ComfyUI's browser-side workflow draft storage.
//
// Black-box: each scenario builds a fake quota-enforcing localStorage,
// seeds it with draft keys from one or more storage generations, loads the
// guard source in an isolated vm context (which runs the boot prune and
// installs the setItem wrapper, exactly like a real page load), then
// asserts on the resulting storage state / thrown errors / toasts.
//
// Storage generations modelled here (see the guard's header comment):
//   V1  unsuffixed   Comfy.Workflow.Drafts / Comfy.Workflow.DraftOrder
//   V1  per-workspace ...Drafts:<ws> / ...DraftOrder:<ws>
//   V2  (1.44+)      Comfy.Workflow.DraftIndex.v2:<ws> + Comfy.Workflow.Draft.v2:<ws>:<hash>
//
// Run: node tests/js/test_draft_guard.mjs

import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";

const repoRoot = path.resolve(import.meta.dirname, "../..");
const sourcePath = path.join(repoRoot, "web", "koolook_draft_guard.js");
const source = fs.readFileSync(sourcePath, "utf8");

const V1_DRAFTS = "Comfy.Workflow.Drafts";
const V1_ORDER = "Comfy.Workflow.DraftOrder";
const v2Index = (ws) => `Comfy.Workflow.DraftIndex.v2:${ws}`;
const v2Payload = (ws, hash) => `Comfy.Workflow.Draft.v2:${ws}:${hash}`;

function makeLocalStorage(initialQuota = Infinity) {
  const map = new Map();
  let quota = initialQuota;
  const used = () => {
    let total = 0;
    for (const [k, v] of map) total += k.length + v.length;
    return total;
  };
  return {
    getItem(k) {
      k = String(k);
      return map.has(k) ? map.get(k) : null;
    },
    setItem(k, v) {
      k = String(k);
      v = String(v);
      const current = map.has(k) ? k.length + map.get(k).length : 0;
      if (used() - current + k.length + v.length > quota) {
        throw new DOMException("Simulated quota exceeded", "QuotaExceededError");
      }
      map.set(k, v);
    },
    removeItem(k) {
      map.delete(String(k));
    },
    key(i) {
      return [...map.keys()][i] ?? null;
    },
    get length() {
      return map.size;
    },
    __map: map,
    __used: used,
    __setQuota(n) {
      quota = n;
    },
  };
}

function loadGuard(store, { toasts = [], warns = [], context = null } = {}) {
  const ctx = context ?? {
    localStorage: store,
    DOMException,
    setTimeout: () => 0,
    document: {
      createElement: () => ({ textContent: "", style: {}, remove() {} }),
      body: { appendChild: (el) => toasts.push(el.textContent) },
    },
    console: {
      warn: (...args) => warns.push(args.map(String).join(" ")),
      log() {},
      error() {},
    },
  };
  ctx.globalThis = ctx;
  vm.runInNewContext(source, ctx, { filename: sourcePath });
  return ctx;
}

function seedV1(store, suffix, entries) {
  const drafts = {};
  const order = [];
  for (const [p, data, updatedAt] of entries) {
    drafts[p] = { data, updatedAt, name: p, isTemporary: false };
    order.push(p);
  }
  store.setItem(V1_DRAFTS + suffix, JSON.stringify(drafts));
  store.setItem(V1_ORDER + suffix, JSON.stringify(order));
}

function seedV2(store, ws, entries) {
  const index = { v: 2, updatedAt: 1, order: [], entries: {} };
  for (const [hash, data, updatedAt] of entries) {
    index.order.push(hash);
    index.entries[hash] = { name: hash, isTemporary: false, updatedAt, path: `workflows/${hash}.json` };
    store.setItem(v2Payload(ws, hash), JSON.stringify({ data, updatedAt }));
  }
  store.setItem(v2Index(ws), JSON.stringify(index));
}

function v1Paths(store, suffix = "") {
  const raw = store.getItem(V1_DRAFTS + suffix);
  return raw === null ? null : Object.keys(JSON.parse(raw));
}

// ---------------------------------------------------------------------------
// S1: wrapper installs once; a second load (stale copy scenario) is a no-op.
{
  const store = makeLocalStorage();
  const nativeSetItem = store.setItem;
  const ctx = loadGuard(store);
  assert.notEqual(store.setItem, nativeSetItem, "setItem should be wrapped after load");
  const firstWrapper = store.setItem;
  loadGuard(store, { context: ctx });
  assert.equal(store.setItem, firstWrapper, "second install must not re-wrap (sentinel)");
}

// ---------------------------------------------------------------------------
// S2: quota error on a NON-draft key is rethrown untouched; no draft evicted.
{
  const store = makeLocalStorage();
  seedV1(store, "", [["wf-a", "x".repeat(200), 10]]);
  const toasts = [];
  loadGuard(store, { toasts });
  store.__setQuota(store.__used() + 50);
  assert.throws(
    () => store.setItem("koolook.workflows.fallback.v1", "y".repeat(5000)),
    (err) => err instanceof DOMException && err.name === "QuotaExceededError",
    "non-draft quota error must propagate",
  );
  assert.deepEqual(v1Paths(store), ["wf-a"], "draft entries must not be touched");
  assert.equal(toasts.length, 0, "no toast for non-draft failures");
}

// ---------------------------------------------------------------------------
// S3: the live bug — a V2 payload write is rescued by evicting old V1 drafts.
{
  const store = makeLocalStorage();
  seedV1(store, "", [
    ["wf-old", "x".repeat(300), 10],
    ["wf-new", "y".repeat(300), 20],
  ]);
  seedV2(store, "personal", [["aaaa1111", "z".repeat(100), 30]]);
  const toasts = [];
  loadGuard(store, { toasts });

  const key = v2Payload("personal", "bbbb2222");
  const value = JSON.stringify({ data: "w".repeat(220), updatedAt: 40 });
  store.__setQuota(store.__used() + key.length + value.length - 1);

  store.setItem(key, value); // must not throw
  assert.equal(store.getItem(key), value, "new V2 payload must be written");
  assert.deepEqual(v1Paths(store), ["wf-new"], "oldest V1 draft must be evicted first");
  assert.ok(
    store.getItem(v2Payload("personal", "aaaa1111")) !== null,
    "newer V2 payload must survive",
  );
  assert.equal(toasts.length, 1, "one rescue toast per page session");
  assert.match(toasts[0], /removed the oldest/i);
}

// ---------------------------------------------------------------------------
// S4: eviction picks the OLDEST draft across generations (V2 older than V1).
{
  const store = makeLocalStorage();
  seedV1(store, "", [["wf-v1", "x".repeat(150), 100]]);
  seedV2(store, "personal", [
    ["old00001", "a".repeat(150), 50],
    ["new00001", "b".repeat(150), 200],
  ]);
  loadGuard(store);

  const key = v2Payload("personal", "cccc3333");
  const value = JSON.stringify({ data: "c".repeat(60), updatedAt: 300 });
  store.__setQuota(store.__used() + key.length + value.length - 1);

  store.setItem(key, value);
  assert.equal(store.getItem(v2Payload("personal", "old00001")), null, "oldest (V2, age 50) evicted");
  assert.deepEqual(v1Paths(store), ["wf-v1"], "younger V1 draft survives");
  assert.ok(store.getItem(v2Payload("personal", "new00001")) !== null, "younger V2 payload survives");
  const index = JSON.parse(store.getItem(v2Index("personal")));
  assert.ok(!index.order.includes("old00001"), "V2 index order drops the evicted hash");
  assert.ok(!("old00001" in index.entries), "V2 index entries drop the evicted hash");
}

// ---------------------------------------------------------------------------
// S5: boot prune deletes suffixed-V1 families superseded by a V2 index,
//     keeps families that are still that workspace's only draft store.
{
  const store = makeLocalStorage();
  seedV1(store, "", [["wf-root", "r".repeat(50), 5]]);
  seedV1(store, ":wsA", [["wf-a", "a".repeat(50), 5]]);
  seedV2(store, "wsA", [["aaaa0001", "p".repeat(50), 6]]);
  seedV1(store, ":wsB", [["wf-b", "b".repeat(50), 5]]);
  loadGuard(store);

  assert.equal(store.getItem(V1_DRAFTS + ":wsA"), null, "migrated V1 family is dead weight");
  assert.equal(store.getItem(V1_ORDER + ":wsA"), null, "migrated V1 order key removed too");
  assert.notEqual(store.getItem(V1_DRAFTS + ":wsB"), null, "unmigrated workspace keeps its V1 family");
  assert.deepEqual(v1Paths(store), ["wf-root"], "unsuffixed V1 family untouched");
  assert.notEqual(store.getItem(v2Index("wsA")), null, "V2 index untouched");
}

// ---------------------------------------------------------------------------
// S6: boot prune drops oversized entries (per-entry cap) in both generations.
{
  const store = makeLocalStorage();
  seedV1(store, "", [
    ["wf-huge", "h".repeat(800_000), 10],
    ["wf-ok", "o".repeat(100), 20],
  ]);
  seedV2(store, "personal", [
    ["huge0001", "H".repeat(800_000), 10],
    ["fine0001", "f".repeat(100), 20],
  ]);
  loadGuard(store);

  assert.deepEqual(v1Paths(store), ["wf-ok"], "oversized V1 entry dropped, sibling kept");
  assert.equal(store.getItem(v2Payload("personal", "huge0001")), null, "oversized V2 payload dropped");
  assert.notEqual(store.getItem(v2Payload("personal", "fine0001")), null, "small V2 payload kept");
}

// ---------------------------------------------------------------------------
// S7: boot prune enforces the total draft budget, oldest first.
{
  const store = makeLocalStorage();
  seedV2(store, "personal", [
    ["aged0001", "a".repeat(700_000), 1],
    ["aged0002", "b".repeat(700_000), 2],
    ["aged0003", "c".repeat(700_000), 3],
  ]);
  const warns = [];
  loadGuard(store, { warns });

  assert.equal(store.getItem(v2Payload("personal", "aged0001")), null, "oldest payload evicted");
  assert.notEqual(store.getItem(v2Payload("personal", "aged0002")), null, "second-oldest survives");
  assert.notEqual(store.getItem(v2Payload("personal", "aged0003")), null, "newest survives");
  assert.ok(
    warns.some((w) => w.includes("[Koolook draft-guard]")),
    "prune logs what it did",
  );
}

// ---------------------------------------------------------------------------
// S8: a corrupt draft key is removed without touching other generations.
{
  const store = makeLocalStorage();
  store.setItem(V1_DRAFTS, "{not json");
  store.setItem(V1_ORDER, "[]");
  seedV2(store, "personal", [["good0001", "g".repeat(100), 10]]);
  loadGuard(store);

  assert.equal(store.getItem(V1_DRAFTS), null, "corrupt V1 blob removed");
  assert.equal(store.getItem(V1_ORDER), null, "its order twin removed");
  assert.notEqual(store.getItem(v2Payload("personal", "good0001")), null, "V2 untouched");
}

// ---------------------------------------------------------------------------
// S9a: unknown future draft keys are NOT evicted while structured candidates
//      remain...
{
  const store = makeLocalStorage();
  seedV1(store, "", [
    ["wf-old", "x".repeat(300), 10],
    ["wf-new", "y".repeat(300), 20],
  ]);
  store.setItem("Comfy.Workflow.DraftCache.v3:future", "F".repeat(120));
  loadGuard(store);

  const key = v2Payload("personal", "dddd4444");
  const value = JSON.stringify({ data: "d".repeat(80), updatedAt: 99 });
  store.__setQuota(store.__used() + key.length + value.length - 1);
  store.setItem(key, value);

  assert.notEqual(store.getItem("Comfy.Workflow.DraftCache.v3:future"), null, "future key kept");
  assert.deepEqual(v1Paths(store), ["wf-new"], "structured candidate evicted instead");
}

// S9b: ...but ARE evicted as a last resort, so a future frontend rename
//      cannot brick draft saves again.
{
  const store = makeLocalStorage();
  store.setItem("Comfy.Workflow.DraftCache.v3:future", "F".repeat(400));
  loadGuard(store);

  const key = v2Payload("personal", "eeee5555");
  const value = JSON.stringify({ data: "e".repeat(200), updatedAt: 99 });
  store.__setQuota(store.__used() + key.length + value.length - 1);
  store.setItem(key, value);

  assert.equal(store.getItem("Comfy.Workflow.DraftCache.v3:future"), null, "future key evicted last-resort");
  assert.equal(store.getItem(key), value, "write rescued");
}

// ---------------------------------------------------------------------------
// S10: when nothing can be evicted, the original quota error propagates and
//      the user gets the "still full" toast.
{
  const store = makeLocalStorage();
  store.setItem("someone.elses.data", "Z".repeat(500));
  const toasts = [];
  loadGuard(store, { toasts });
  store.__setQuota(store.__used() + 20);

  const key = v2Payload("personal", "ffff6666");
  assert.throws(
    () => store.setItem(key, "f".repeat(100)),
    (err) => err instanceof DOMException && err.name === "QuotaExceededError",
    "unrescuable write rethrows the quota error",
  );
  // A second failing write in the same session still throws but must NOT
  // re-toast — otherwise a genuinely-full origin gets per-edit toast spam.
  assert.throws(
    () => store.setItem(key, "f".repeat(100)),
    (err) => err instanceof DOMException && err.name === "QuotaExceededError",
    "repeat unrescuable write still rethrows",
  );
  assert.equal(
    toasts.filter((t) => /still full/i.test(t)).length,
    1,
    "still-full toast is latched once per session",
  );
  assert.notEqual(store.getItem("someone.elses.data"), null, "non-draft data never touched");
}

// ---------------------------------------------------------------------------
// S11: a pre-set sentinel (older guard copy already installed) skips the
//      wrapper but the boot prune still runs.
{
  const store = makeLocalStorage();
  store.__koolookDraftQuotaGuardInstalled = true;
  const nativeSetItem = store.setItem;
  seedV1(store, ":wsA", [["wf-a", "a".repeat(50), 5]]);
  seedV2(store, "wsA", [["aaaa0001", "p".repeat(50), 6]]);
  loadGuard(store);

  assert.equal(store.setItem, nativeSetItem, "wrapper must not double-install");
  assert.equal(store.getItem(V1_DRAFTS + ":wsA"), null, "boot prune still runs");
}

// ---------------------------------------------------------------------------
// S12: when the failing write IS a draft container (V1 blob growing), prefer
//      freeing OTHER keys — rewriting the target key smaller cannot help,
//      because the retry overwrites it with the same big value anyway.
{
  const store = makeLocalStorage();
  seedV1(store, "", [["wf-keep", "k".repeat(200), 5]]); // older than the V2 entry
  seedV2(store, "personal", [["gggg7777", "g".repeat(200), 50]]);
  loadGuard(store);

  const grown = JSON.stringify({
    "wf-keep": { data: "k".repeat(200), updatedAt: 5, name: "wf-keep", isTemporary: false },
    "wf-added": { data: "n".repeat(120), updatedAt: 60, name: "wf-added", isTemporary: false },
  });
  const delta = grown.length - store.getItem(V1_DRAFTS).length;
  store.__setQuota(store.__used() + delta - 1);

  store.setItem(V1_DRAFTS, grown);
  assert.equal(store.getItem(V1_DRAFTS), grown, "grown V1 blob written intact");
  assert.equal(store.getItem(v2Payload("personal", "gggg7777")), null, "space freed from another key");
}

console.log("test_draft_guard.mjs: all scenarios passed");
