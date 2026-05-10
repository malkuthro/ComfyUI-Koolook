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

The root Koolook nodes and the slim Radiance Koolook VAE wrappers have no
extra Python dependencies beyond `torch`, which ComfyUI already requires.

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

## Learn More

- **Public overview:** [kforgelabs.com/comfyui-projects/comfyui-koolook/](https://www.kforgelabs.com/comfyui-projects/comfyui-koolook/)
- **In-app visual guide:** [malkuthro.github.io/ComfyUI-Koolook/web/guide/](https://malkuthro.github.io/ComfyUI-Koolook/web/guide/)
  (offline copy: [`web/guide/index.html`](web/guide/index.html))
- **User guide:** [`docs/user_guide/`](docs/user_guide/)
- **Change history:** [`CHANGELOG.md`](CHANGELOG.md)
- **Maintainer docs:** [`docs/maintainers/`](docs/maintainers/)
- **Fork attribution and provenance:** [`forks/THIRD_PARTY.md`](forks/THIRD_PARTY.md)
- **Versioning reference:** [`docs/reference/versioning.md`](docs/reference/versioning.md)

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
