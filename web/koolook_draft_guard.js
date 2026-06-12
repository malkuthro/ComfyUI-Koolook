// Koolook guard for ComfyUI's browser-side workflow draft storage.
//
// ComfyUI's frontend autosaves the active graph as a "draft" in
// localStorage. When the origin's storage quota is hit, every draft save
// fails and the frontend toasts "Failed to save workflow draft" — and since
// frontend 1.44 the V2 persistence layer additionally latches a session-wide
// "storage unavailable" flag after one failed save+evict cycle, so the toast
// then repeats on every edit until the page is reloaded.
//
// The key scheme has changed across frontend versions:
//
//   V1 (legacy, still written on tab switches as of 1.44):
//       Comfy.Workflow.Drafts            one JSON blob of all drafts
//       Comfy.Workflow.DraftOrder        LRU order, oldest first
//   V1 per-workspace (interim generation; the 1.44 migration reads these
//   once without deleting them — later frontends added cleanup, and an
//   interrupted migration can leave them behind on any version):
//       Comfy.Workflow.Drafts:<ws>
//       Comfy.Workflow.DraftOrder:<ws>
//   V2 (1.44+, written ~512ms after every graph edit):
//       Comfy.Workflow.DraftIndex.v2:<ws>        index {v, updatedAt, order[], entries{}}
//       Comfy.Workflow.Draft.v2:<ws>:<hash>      one payload {data, updatedAt} per draft
//
// All generations share the "Comfy.Workflow.Draft" key prefix, so this
// guard matches by prefix instead of exact names — a future upstream rename
// inside that prefix degrades gracefully (last-resort eviction) instead of
// silently missing, which is how the previous exact-name guard went stale.
//
// Two mechanisms, both installed at page load from this file:
//
//   1. Boot prune — deletes draft generations the frontend can no longer
//      read (suffixed V1 families already migrated to V2), corrupt keys,
//      oversized entries, and enforces a total draft budget so the rest of
//      the origin's storage keeps headroom.
//   2. setItem quota guard — wraps localStorage.setItem; when a write to a
//      draft key throws QuotaExceededError, evicts the oldest draft across
//      ALL generations and retries. Because the wrapper sits inside
//      setItem, the rescue happens before the V2 layer's own catch sees the
//      error — the save succeeds, no toast, no storage-unavailable latch.
//
// Only keys under the draft prefix are ever touched. Koolook /userdata
// stores, snapshots, sidebar state, and other extensions' keys are not.
//
// Maintainer docs: docs/maintainers/workflows-sidebar.md
// ("Comfy workflow draft quota gotcha"). History: the exact-name version of
// this guard lived in web/whatdreamscost_koolook/ltx_director.js until the
// frontend 1.44 key-scheme change made it dead code.

(() => {
  "use strict";

  const DRAFT_KEY_PREFIX = "Comfy.Workflow.Draft";
  const V1_DRAFTS_KEY = "Comfy.Workflow.Drafts";
  const V1_ORDER_KEY = "Comfy.Workflow.DraftOrder";
  const V2_INDEX_PREFIX = "Comfy.Workflow.DraftIndex.v2:";
  const V2_PAYLOAD_PREFIX = "Comfy.Workflow.Draft.v2:";

  // Budgets in UTF-16 code units (what localStorage quotas count). Browsers
  // commonly allow ~5M units per origin; drafts must not crowd out the rest.
  // The old embedded guard capped the unsuffixed V1 blob alone at 1.5M; the
  // 2M total here covers every generation combined — a scope widening more
  // than a raise.
  const MAX_TOTAL_DRAFT_CHARS = 2_000_000;
  const MAX_DRAFT_ENTRY_CHARS = 750_000;
  const MAX_EVICTIONS_PER_WRITE = 25;
  const MAX_PRUNE_EVICTIONS = 200;
  const LOG_PREFIX = "[Koolook draft-guard]";

  const originalSetItem = localStorage.setItem.bind(localStorage);
  let rescueToastShown = false;
  let stillFullToastShown = false;

  function isDraftKey(key) {
    return typeof key === "string" && key.startsWith(DRAFT_KEY_PREFIX);
  }

  function isQuotaError(err) {
    return (
      err instanceof DOMException &&
      (err.name === "QuotaExceededError" ||
        err.name === "NS_ERROR_DOM_QUOTA_REACHED" ||
        err.code === 22 ||
        err.code === 1014)
    );
  }

  function listDraftKeys() {
    const keys = [];
    for (let i = 0; i < localStorage.length; i += 1) {
      const key = localStorage.key(i);
      if (isDraftKey(key)) keys.push(key);
    }
    return keys;
  }

  function parseJson(raw) {
    try {
      return JSON.parse(raw);
    } catch (_err) {
      return undefined;
    }
  }

  function totalDraftChars() {
    let total = 0;
    for (const key of listDraftKeys()) {
      total += key.length + (localStorage.getItem(key) || "").length;
    }
    return total;
  }

  function removeV1Family(suffix) {
    localStorage.removeItem(V1_DRAFTS_KEY + suffix);
    localStorage.removeItem(V1_ORDER_KEY + suffix);
  }

  function v1FamilySuffixes(keys) {
    const suffixes = new Set();
    for (const key of keys) {
      if (key === V1_DRAFTS_KEY || key === V1_ORDER_KEY) suffixes.add("");
      else if (key.startsWith(V1_DRAFTS_KEY + ":")) suffixes.add(key.slice(V1_DRAFTS_KEY.length));
      else if (key.startsWith(V1_ORDER_KEY + ":")) suffixes.add(key.slice(V1_ORDER_KEY.length));
    }
    return suffixes;
  }

  // A V1 family is the {drafts blob, order} key pair for one suffix.
  // Returns null when absent, {corrupt: true} when unreadable.
  function readV1Family(suffix) {
    const rawDrafts = localStorage.getItem(V1_DRAFTS_KEY + suffix);
    if (rawDrafts === null) {
      // Stray order key without a drafts blob is still dead weight.
      return localStorage.getItem(V1_ORDER_KEY + suffix) === null
        ? null
        : { suffix, corrupt: true };
    }
    const drafts = parseJson(rawDrafts);
    if (!drafts || typeof drafts !== "object" || Array.isArray(drafts)) {
      return { suffix, corrupt: true };
    }
    let order = parseJson(localStorage.getItem(V1_ORDER_KEY + suffix) || "[]");
    if (!Array.isArray(order)) order = [];
    order = order.filter((path) => typeof path === "string");
    const known = Object.keys(drafts);
    const orderedKeys = [
      ...order.filter((path) => Object.prototype.hasOwnProperty.call(drafts, path)),
      ...known.filter((path) => !order.includes(path)),
    ];
    return { suffix, drafts, orderedKeys };
  }

  function writeV1Family(family) {
    originalSetItem(V1_DRAFTS_KEY + family.suffix, JSON.stringify(family.drafts));
    originalSetItem(
      V1_ORDER_KEY + family.suffix,
      JSON.stringify(
        family.orderedKeys.filter((path) =>
          Object.prototype.hasOwnProperty.call(family.drafts, path),
        ),
      ),
    );
  }

  function readV2Index(ws) {
    const raw = localStorage.getItem(V2_INDEX_PREFIX + ws);
    if (raw === null) return null;
    const index = parseJson(raw);
    if (
      !index ||
      typeof index !== "object" ||
      !Array.isArray(index.order) ||
      typeof index.entries !== "object" ||
      index.entries === null
    ) {
      return { ws, corrupt: true };
    }
    return { ws, index };
  }

  function isKnownDraftKey(key) {
    return (
      key === V1_DRAFTS_KEY ||
      key === V1_ORDER_KEY ||
      key.startsWith(V1_DRAFTS_KEY + ":") ||
      key.startsWith(V1_ORDER_KEY + ":") ||
      key.startsWith(V2_INDEX_PREFIX) ||
      key.startsWith(V2_PAYLOAD_PREFIX)
    );
  }

  // One eviction candidate = the single cheapest-to-lose draft unit in one
  // storage generation. Sorted by (penalty, age):
  //   penalty 0 — structured eviction that frees space in a key OTHER than
  //               the one currently being written (a real rescue),
  //   penalty 1 — unknown future-generation draft key (works, but we can't
  //               pick "oldest", so structured candidates go first),
  //   penalty 2 — rewriting the target key itself smaller; useless for the
  //               pending write (the retry overwrites it with the same
  //               value) but kept as a final attempt for browsers that need
  //               transient headroom while replacing a key.
  function collectEvictionCandidates(targetKey) {
    const keys = listDraftKeys();
    const candidates = [];

    for (const suffix of v1FamilySuffixes(keys)) {
      const draftsKey = V1_DRAFTS_KEY + suffix;
      const family = readV1Family(suffix);
      if (!family) continue;
      if (family.corrupt || family.orderedKeys.length === 0) {
        candidates.push({
          age: -1,
          penalty: 0,
          label: `unreadable/empty v1 family "${suffix || "(root)"}"`,
          evict: () => removeV1Family(suffix),
        });
        continue;
      }
      const front = family.orderedKeys[0];
      const entry = family.drafts[front];
      candidates.push({
        age: typeof entry?.updatedAt === "number" ? entry.updatedAt : 0,
        penalty: draftsKey === targetKey ? 2 : 0,
        label: `v1 draft "${front}"`,
        evict: () => {
          delete family.drafts[front];
          family.orderedKeys.shift();
          try {
            writeV1Family(family);
          } catch (_err) {
            // Even the smaller rewrite is over quota: drop the family.
            removeV1Family(suffix);
          }
        },
      });
    }

    const v2Workspaces = new Set();
    const v2PayloadsByWs = new Map();
    for (const key of keys) {
      if (key.startsWith(V2_INDEX_PREFIX)) {
        v2Workspaces.add(key.slice(V2_INDEX_PREFIX.length));
      } else if (key.startsWith(V2_PAYLOAD_PREFIX)) {
        const rest = key.slice(V2_PAYLOAD_PREFIX.length);
        const sep = rest.lastIndexOf(":");
        const ws = sep >= 0 ? rest.slice(0, sep) : rest;
        const hash = sep >= 0 ? rest.slice(sep + 1) : "";
        if (!v2PayloadsByWs.has(ws)) v2PayloadsByWs.set(ws, []);
        v2PayloadsByWs.get(ws).push({ key, hash });
        v2Workspaces.add(ws);
      }
    }

    for (const ws of v2Workspaces) {
      const indexKey = V2_INDEX_PREFIX + ws;
      const record = readV2Index(ws);
      const payloads = v2PayloadsByWs.get(ws) || [];

      if (record?.corrupt) {
        candidates.push({
          age: -1,
          penalty: indexKey === targetKey ? 2 : 0,
          label: `corrupt v2 index "${ws}"`,
          evict: () => localStorage.removeItem(indexKey),
        });
      }

      const order = record && !record.corrupt ? record.index.order : [];
      const entries = record && !record.corrupt ? record.index.entries : {};
      const present = new Set(payloads.map((p) => p.hash));

      for (const payload of payloads) {
        if (order.includes(payload.hash)) continue;
        // Orphan payload: unreachable through the index, pure dead weight.
        candidates.push({
          age: -1,
          penalty: payload.key === targetKey ? 2 : 0,
          label: `orphan v2 payload "${ws}:${payload.hash}"`,
          evict: () => localStorage.removeItem(payload.key),
        });
      }

      const front = order.find((hash) => present.has(hash));
      if (front !== undefined) {
        const payloadKey = V2_PAYLOAD_PREFIX + ws + ":" + front;
        const entryAge = entries[front]?.updatedAt;
        candidates.push({
          age: typeof entryAge === "number" ? entryAge : 0,
          penalty: payloadKey === targetKey ? 2 : 0,
          label: `v2 draft "${ws}:${front}"`,
          evict: () => {
            localStorage.removeItem(payloadKey);
            if (indexKey === targetKey) return; // retry overwrites it anyway
            try {
              const next = {
                ...record.index,
                updatedAt: Date.now(),
                order: order.filter((hash) => hash !== front),
                entries: { ...entries },
              };
              delete next.entries[front];
              originalSetItem(indexKey, JSON.stringify(next));
            } catch (_err) {
              // Stale index entry is fine — the frontend self-heals it on read.
            }
          },
        });
      }
    }

    for (const key of keys) {
      if (isKnownDraftKey(key)) continue;
      candidates.push({
        age: 0,
        penalty: key === targetKey ? 2 : 1,
        label: `unknown draft key "${key}"`,
        evict: () => localStorage.removeItem(key),
      });
    }

    candidates.sort((a, b) => a.penalty - b.penalty || a.age - b.age);
    return candidates;
  }

  function evictOneDraftUnit(targetKey) {
    for (const candidate of collectEvictionCandidates(targetKey)) {
      try {
        candidate.evict();
      } catch (err) {
        console.warn(`${LOG_PREFIX} eviction step failed for ${candidate.label}`, err);
        continue;
      }
      console.warn(`${LOG_PREFIX} evicted ${candidate.label} to free draft storage.`);
      return true;
    }
    return false;
  }

  function showDraftQuotaToast(message) {
    try {
      const toast = document.createElement("div");
      toast.textContent = message;
      toast.style.cssText = [
        "position:fixed",
        "right:30px",
        "bottom:30px",
        "z-index:9999",
        "max-width:420px",
        "padding:10px 14px",
        "border-radius:4px",
        "background:rgba(180,60,60,0.95)",
        "color:#fff",
        "font:12px/1.4 ui-sans-serif,system-ui,sans-serif",
        "box-shadow:0 2px 8px rgba(0,0,0,0.4)",
      ].join(";");
      document.body.appendChild(toast);
      setTimeout(() => toast.remove(), 6500);
    } catch (_err) {
      // Best-effort UI: a toast must never break the storage path
      // (document can be unavailable in exotic embeddings).
    }
  }

  function notifyRescueOnce() {
    if (rescueToastShown) return;
    rescueToastShown = true;
    showDraftQuotaToast(
      "Browser draft storage was full. Koolook removed the oldest workflow draft(s) only, so autosave keeps working.",
    );
  }

  function installComfyDraftQuotaGuard() {
    try {
      if (localStorage.__koolookDraftQuotaGuardInstalled) return;
      Object.defineProperty(localStorage, "__koolookDraftQuotaGuardInstalled", {
        value: true,
        configurable: true,
      });
      localStorage.setItem = (key, value) => {
        try {
          return originalSetItem(key, value);
        } catch (err) {
          if (!isQuotaError(err) || !isDraftKey(String(key))) throw err;
          // evictOneDraftUnit reports "removed one draft unit", not "freed
          // enough bytes" — the retry below decides that. Units are finite
          // and never recreated here, so the loop terminates even without
          // the attempt cap.
          for (let attempt = 0; attempt < MAX_EVICTIONS_PER_WRITE; attempt += 1) {
            if (!evictOneDraftUnit(String(key))) break;
            try {
              const result = originalSetItem(key, value);
              notifyRescueOnce();
              return result;
            } catch (retryErr) {
              if (!isQuotaError(retryErr)) throw retryErr;
            }
          }
          // Out of evictable drafts. Rethrow so Comfy handles the failure;
          // on 1.44+ its V2 layer then latches storage-unavailable and
          // stops calling setItem for the session, so this wrapper cannot
          // help again until reload — latch our hint toast to match.
          if (!stillFullToastShown) {
            stillFullToastShown = true;
            showDraftQuotaToast(
              "Comfy draft cache is still full. Koolook could not free enough space; export or delete old drafts manually.",
            );
          }
          throw err;
        }
      };
    } catch (err) {
      console.warn(`${LOG_PREFIX} could not install Comfy draft quota guard.`, err);
    }
  }

  function pruneComfyDraftCache() {
    try {
      if (listDraftKeys().length === 0) return;
      let pruned = false;

      // 1. Suffixed V1 families already migrated to V2 are dead weight: the
      //    frontend's migration reads them once and never deletes them.
      for (const suffix of v1FamilySuffixes(listDraftKeys())) {
        if (!suffix) continue;
        const ws = suffix.slice(1);
        if (localStorage.getItem(V2_INDEX_PREFIX + ws) !== null) {
          removeV1Family(suffix);
          pruned = true;
        }
      }

      // 2. Corrupt families and oversized entries, scoped to the offending
      //    key — never the whole draft store.
      for (const suffix of v1FamilySuffixes(listDraftKeys())) {
        const family = readV1Family(suffix);
        if (!family) continue;
        if (family.corrupt) {
          removeV1Family(suffix);
          pruned = true;
          continue;
        }
        let changed = false;
        for (const path of [...family.orderedKeys]) {
          const data = family.drafts[path]?.data;
          if (typeof data === "string" && data.length > MAX_DRAFT_ENTRY_CHARS) {
            delete family.drafts[path];
            family.orderedKeys.splice(family.orderedKeys.indexOf(path), 1);
            changed = true;
          }
        }
        if (changed) {
          try {
            writeV1Family(family);
          } catch (_err) {
            removeV1Family(suffix);
          }
          pruned = true;
        }
      }
      for (const key of listDraftKeys()) {
        if (!key.startsWith(V2_PAYLOAD_PREFIX) && !key.startsWith(V2_INDEX_PREFIX)) continue;
        const raw = localStorage.getItem(key) || "";
        const oversized = key.startsWith(V2_PAYLOAD_PREFIX) && raw.length > MAX_DRAFT_ENTRY_CHARS;
        if (oversized || parseJson(raw) === undefined) {
          localStorage.removeItem(key);
          pruned = true;
        }
      }

      // 3. Total budget across every generation, oldest drafts first.
      let attempts = 0;
      while (totalDraftChars() > MAX_TOTAL_DRAFT_CHARS && attempts < MAX_PRUNE_EVICTIONS) {
        if (!evictOneDraftUnit(null)) break;
        attempts += 1;
        pruned = true;
      }

      if (pruned) {
        console.warn(
          `${LOG_PREFIX} pruned Comfy workflow draft storage to prevent save-draft failures.`,
        );
      }
    } catch (err) {
      console.warn(`${LOG_PREFIX} draft prune skipped:`, err);
    }
  }

  installComfyDraftQuotaGuard();
  pruneComfyDraftCache();
})();
