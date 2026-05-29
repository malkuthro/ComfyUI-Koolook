# WhatDreamsCost Koolook

Versioned local fork of WhatDreamsCost-ComfyUI nodes (LTX Director + supporting
Prompt-Relay implementation).

- Package entrypoint: `forks/whatdreamscost_koolook/__init__.py`
- Version folders: `forks/whatdreamscost_koolook/versions/<version>/`
- Each version folder keeps:
  - modified node files tracked in MAIN
  - vendored unmodified upstream deps the modified files import
  - `UPSTREAM_PIN.yaml` for parity with pinned external raw checkout

Current active version:

- `v1_3_9` — current audio-lipsync loop target. Based on upstream
  WhatDreamsCost-ComfyUI commit `3b65410` (pyproject/README version
  `1.3.9`; no `v1.3.9` git tag at fork time), with the Koolook
  `relay_overrides` and per-segment-sigma patches carried forward. New
  workflows use stable node ID `LTXDirector__koolook`; older
  `LTXDirector__koolook_v1_3_2` workflows load through an alias.
- `v1_3_2` — see [`forks/THIRD_PARTY.md`](../THIRD_PARTY.md) for upstream
  attribution and what changed locally. Kept registered for saved-workflow
  compatibility. Tracks
  [WhatDreamsCost/WhatDreamsCost-ComfyUI](https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI)
  at commit `e81223a`. License: GPL-3.0 (compatible with this pack).

The maintainer-facing iteration loop that drives changes here lives at
[`docs/automations/LTX-2.3/audio-lipsync/`](../../docs/automations/LTX-2.3/audio-lipsync/).
