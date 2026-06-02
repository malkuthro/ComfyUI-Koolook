# ComfyUI-Koolook

Kforge Labs for ComfyUI: a project-kit sidebar plus Koolook nodes for VFX,
image, and video workflows.

**Start here:** [KForge Labs ComfyUI-Koolook overview](https://www.kforgelabs.com/comfyui-projects/comfyui-koolook/)

That overview explains the sidebar, snapshots, saved workflows, modules,
recovery tools, and included nodes for new users.

The main experience is the **Kforge Labs sidebar**. It helps you keep a
project setup together: favorite nodes, saved workflows, reusable modules,
tags, snapshots, and recovery tools.

## Kforge Labs Sidebar

Use the sidebar as a reusable project kit.

| Feature | What it does |
|---|---|
| **Snapshots** | Save and load the whole Kforge Labs state: favorite nodes, workflows, modules, tags, archive, and recovery data. |
| **Search** | Find nodes, workflows, directories, and tags from one field. Tags keep exact capitalization, so `Depth` and `depth` can be separate folders. |
| **Nodes** | Keep favorite nodes close, grouped by theme or repository name. |
| **Workflows** | Save the whole script, save a selection of nodes, load workflows, or insert saved modules into the current canvas. |
| **Recovery** | Use autosaves, starter presets, snapshot library settings, and missing-pack install helpers. |

New here? Start with the
[KForge overview](https://www.kforgelabs.com/comfyui-projects/comfyui-koolook/).
Using it inside ComfyUI? Open the operational guide:

- **Browse on the web:** [malkuthro.github.io/ComfyUI-Koolook/web/guide/](https://malkuthro.github.io/ComfyUI-Koolook/web/guide/)
- **Inside ComfyUI:** Kforge Labs sidebar → **Tools** row → **Help** (`H` icon).
  The in-app button opens the bundled offline copy at [`web/guide/index.html`](web/guide/index.html).

## Included Nodes

| Node ID | Display name | Category |
|---|---|---|
| `EasyWan22Prompt` | Wan 2.2 Easy Prompt (Koolook) | `Koolook/Wan_Video` |
| `EasyResize_Koolook` | Easy Resize (Koolook) | `Koolook/Image` |
| `EasyAIPipeline` | Easy AI Pipeline (Koolook) | `Koolook/Pipeline` |
| `easy_ImageBatch` | Easy Image Batch (Koolook) | `Koolook/Image` |
| `KoolookLoadCameraPosesAbsolute` | Koolook Load Camera Poses (Absolute Path) | `Koolook/Camera` |
| `Easy_hdr_VAE_encode` | Easy HDR VAE Encode (Koolook) | `Koolook/VAE` |
| `Easy_hdr_VAE_decode` | Easy HDR VAE Decode (Koolook) | `Koolook/VAE` |
| `Easy_Pattern` | Easy Pattern (Koolook) | `Koolook/Testing` |
| `Easy_LoadVideo` | Easy Load Video (Koolook) | `Koolook/Video` |
| `Easy_VideoCombine` | Easy Video Combine (Koolook) | `Koolook/Video` |
| `Easy_Utility` | Easy Utility (Koolook) | `Koolook/Utility` |
| `LTXDirector__koolook` | LTX Director (Koolook) | `Koolook/PromptRelay` |

The root Koolook nodes and the slim Radiance Koolook VAE wrappers have no
extra Python dependencies beyond `torch`, which ComfyUI already requires.
The video wrappers require Video Helper Suite, and the LTX Director fork is
for workflows that already use the LTX / Prompt Relay node ecosystem.

## Install

### ComfyUI-Manager

1. Open ComfyUI Manager.
2. Choose **Install Custom Nodes**.
3. Install by Git URL:
   `https://github.com/malkuthro/ComfyUI-Koolook.git`
4. Restart ComfyUI.
5. Open the **Kforge Labs** sidebar tab.

### Manual

Clone this repository into `ComfyUI/custom_nodes/`:

```bash
git clone https://github.com/malkuthro/ComfyUI-Koolook.git
```

Restart ComfyUI after installation.

> **One install at a time.** If you already installed Kforge Labs via
> ComfyUI Manager, it lives at `custom_nodes/koolook/` (the Manager /
> Registry-derived folder name). A second `custom_nodes/ComfyUI-Koolook/`
> created by `git clone` is a parallel install — ComfyUI loads BOTH on
> every boot, both register the same routes and sidebar tab, and the
> workflow store silently corrupts on every restart. From v0.3.8 the
> plugin detects this and prints a critical log naming both paths,
> disabling the non-winning copy — but the cleanest fix is to remove
> one folder. Pick the install method you want and stick with it.

## Learn More

- **Public overview:** [kforgelabs.com/comfyui-projects/comfyui-koolook/](https://www.kforgelabs.com/comfyui-projects/comfyui-koolook/)
- **In-app visual guide:** [malkuthro.github.io/ComfyUI-Koolook/web/guide/](https://malkuthro.github.io/ComfyUI-Koolook/web/guide/)
  (offline copy: [`web/guide/index.html`](web/guide/index.html))
- **Support / bug reports:** [`SUPPORT.md`](SUPPORT.md)
- **User guide:** [`docs/user_guide/`](docs/user_guide/)
- **Change history:** [`CHANGELOG.md`](CHANGELOG.md)
- **Maintainer docs:** [`docs/maintainers/`](docs/maintainers/)
- **Fork attribution and provenance:** [`forks/THIRD_PARTY.md`](forks/THIRD_PARTY.md)
- **Versioning reference:** [`docs/reference/versioning.md`](docs/reference/versioning.md)

## Troubleshooting

| What you see | What it usually means | What to do |
|---|---|---|
| The **Kforge Labs** tab does not appear | ComfyUI needs a restart, or the extension did not load | Restart ComfyUI and check the terminal for `[Koolook]` messages. |
| The Help guide does not open | The browser blocked the popup | Allow popups for the ComfyUI page, or open [`web/guide/index.html`](web/guide/index.html) directly. |
| Missing custom nodes after loading a workflow | One or more node packs used by that workflow are not installed | In Kforge Labs, use **Tools → Install missing packs**, or use ComfyUI Manager's **Install Missing Custom Nodes** flow. |
| Snapshot save/load fails | The snapshot library path may be missing, read-only, or unavailable | Use **Snapshot → Save to...** or **Load from...** and choose a writable folder. |
| A loaded snapshot looks older than expected | Named saves and auto-saves are intentionally separate | In **Snapshot → Load**, select the snapshot and check **Recovery auto-saves** for a newer periodic/pre-load recovery. |

## Support

For reproducible bugs, open a
[GitHub issue](https://github.com/malkuthro/ComfyUI-Koolook/issues/new/choose).
For the fastest answer, include your ComfyUI version, Koolook version,
operating system, install method, terminal log, browser console errors, and
steps to reproduce. See [`SUPPORT.md`](SUPPORT.md) for the exact checklist.

## Stability

For production work, install a pinned release tag or commit instead of following
a moving branch head. Node IDs are treated carefully so saved ComfyUI workflows
continue to load across compatible updates.

## License

**GPL-3.0**. See [`LICENSE`](LICENSE).

This package incorporates and adapts GPL-3.0 code from
[fxtdstudios/radiance](https://github.com/fxtdstudios/radiance) under
`forks/radiance_koolook/`. Attribution and per-fork modification notes are
tracked in [`forks/THIRD_PARTY.md`](forks/THIRD_PARTY.md) and
[`forks/forks_manifest.yaml`](forks/forks_manifest.yaml).
