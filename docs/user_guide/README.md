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
| `Koolook/Pipeline` | ✅ [`EasyAIPipeline`](nodes/koolook_pipeline/easy_ai_pipeline.md) | `EasyAIPipeline` |
| `Koolook/Image` | ✅ [`EasyResize_Koolook`](nodes/koolook_image/easy_resize_koolook.md) · ✅ [`easy_ImageBatch`](nodes/koolook_image/easy_image_batch.md) | `EasyResize_Koolook`, `easy_ImageBatch` |
| `Koolook/Wan_Video` | ✅ [`EasyWan22Prompt`](nodes/koolook_wan_video/easy_wan22_prompt.md) | `EasyWan22Prompt` |
| `Koolook/VAE` | ✅ [`nodes/radiance_koolook_v2_3_3/`](nodes/radiance_koolook_v2_3_3/) | `Easy_hdr_VAE_encode`, `Easy_hdr_VAE_decode` |
| `Koolook/Video` | ✅ [`Easy_LoadVideo`](nodes/koolook_video/easy_load_video.md) · ✅ [`Easy_VideoCombine`](nodes/koolook_video/easy_video_combine.md) | `Easy_LoadVideo`, `Easy_VideoCombine` |
| `Koolook/Utility` | ✅ [`Easy_Utility`](nodes/koolook_utility/easy_utility.md) | `Easy_Utility` |
| `Koolook/Loop` | ✅ [`Koolook_LoopStatus`](nodes/koolook_loop/koolook_loop_status.md) | `Koolook_LoopStatus` |
| `Koolook/Publish` | ✅ [`Publish contract nodes`](nodes/koolook_publish/publish_contract_nodes.md) | `Koolook_PublishInput`, `Koolook_PublishOutput`, `Koolook_PublishRouter`, `Koolook_PublishResult` |
| `Koolook/Camera` | ✅ [`KoolookLoadCameraPosesAbsolute`](nodes/koolook_camera/koolook_load_camera_poses_absolute.md) | `KoolookLoadCameraPosesAbsolute` |
| `Koolook/Testing` | ✅ [`Easy_Pattern`](nodes/koolook_testing/easy_pattern.md) | `Easy_Pattern` |
| `Koolook/PromptRelay` | ✅ [`LTXDirector__koolook`](nodes/whatdreamscost_koolook/ltx_director_koolook.md) | `LTXDirector__koolook` |

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

The repo [`README.md`](../../README.md) lists the canonical public node IDs
and display names. Some compatibility aliases remain registered so older saved
workflows load, but new workflows should pick the canonical IDs documented
here. The historical inventory (including removed nodes) is in the
[`CHANGELOG.md`](../../CHANGELOG.md).
