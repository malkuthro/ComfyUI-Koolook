# User Guide

End-user documentation for nodes shipped by `ComfyUI-Koolook`. Pages
mirror the ComfyUI category each node lives under, so the structure here
matches what you see when you type "koolook" in the node-add menu.

Each per-node page includes a **Source** block at the top that records
whether the node is Koolook-native or imported from a fork — so fork
attribution is visible without needing a fork-shaped directory tree.

## Browse by category

| ComfyUI menu | Status | Nodes |
|---|---|---|
| `Koolook/Pipeline` | 🚧 page TBD | `EasyAIPipeline` |
| `Koolook/Image` | 🚧 `EasyResize_Koolook` page TBD · ✅ [`easy_ImageBatch`](nodes/koolook_image/easy_image_batch.md) | `EasyResize_Koolook`, `easy_ImageBatch` |
| `Koolook/Camera` | 🚧 page TBD | `KoolookLoadCameraPosesAbsolute` |
| `Koolook/Wan_Video` | 🚧 page TBD | `EasyWan22Prompt` |
| `Koolook/VAE` | ✅ [`nodes/radiance_koolook_v2_3_3/`](nodes/radiance_koolook_v2_3_3/) | `Easy_hdr_VAE_encode`, `Easy_hdr_VAE_decode` |
| `Koolook/Testing` | 🚧 page TBD | `Easy_Pattern` |

## Cross-cutting concepts

Pages here cover ideas that span more than one node:

- [Kforge Labs sidebar guide](../../web/guide/index.html) — visual onboarding for
  Snapshots, favorite nodes, saved workflows, reusable modules, tags, and recovery.
- [VAE encode/decode pairing](nodes/radiance_koolook_v2_3_3/encode_decode_pairing.md) — which
  encoder fields must match the decoder for a clean roundtrip, with
  worked examples for SDR / HDR / cinema-log / video workflows.
- [HDR modes deep dive](nodes/radiance_koolook_v2_3_3/hdr_modes.md) — what `Clip (SDR)`,
  `Soft Clip`, `Compress (Log)`, and `Passthrough` actually do to your
  pixels before they hit the VAE, with per-pixel worked examples.

## When to add a guide

A node deserves a page if it has any of:

- Non-obvious input semantics (e.g. aspect-ratio strings, frame indexing
  conventions).
- A workflow recipe more useful than the per-input tooltips.
- A known limitation or compatibility caveat.
- An encode/decode or input/output pairing relationship with other nodes.

Convention:

- One markdown file per canonical node ID, named after the ID
  (`easy_hdr_vae_encode.md`, `easy_image_batch.md`).
- Place it in the folder that matches the node's ComfyUI category.
- Screenshots and helper images go in [`img/`](img/), named to match the
  page they're embedded in (e.g. `easy_hdr_vae_encode_pairing.png`).

## Where to find the canonical node ID list

The repo [`README.md`](../../README.md) lists all currently-registered
node IDs and display names. The historical inventory (including removed
nodes) is in the [`CHANGELOG.md`](../../CHANGELOG.md).
