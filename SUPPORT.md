# Support

Kforge Labs is easiest to support when reports include the exact context
needed to reproduce the problem.

## Before Opening An Issue

Try the quick checks first:

| Problem | First thing to try |
|---|---|
| The **Kforge Labs** tab is missing | Restart ComfyUI and check the terminal for `[Koolook]` messages. |
| The Help guide does not open | Allow popups for the ComfyUI page, or open `web/guide/index.html` directly. |
| A workflow loads with missing nodes | Use **Tools -> Install missing packs** in Kforge Labs, or ComfyUI Manager's **Install Missing Custom Nodes** flow. |
| Snapshot save/load fails | Use **Snapshot -> Save to...** or **Load from...** and choose a writable snapshot folder. |
| A loaded snapshot looks older than expected | In **Snapshot -> Load**, select the snapshot and check **Recovery auto-saves** for a newer recovery copy. |

## Bug Reports

Open a GitHub issue:
<https://github.com/malkuthro/ComfyUI-Koolook/issues/new/choose>

Please include:

- What you were trying to do.
- What happened instead.
- Steps to reproduce the problem.
- Your operating system.
- Your ComfyUI version or commit.
- Your Koolook version or commit.
- How you installed Koolook: ComfyUI Manager, Comfy Registry, or manual git clone.
- Relevant terminal output with `[Koolook]` lines.
- Browser console errors, if the issue involves the sidebar.
- Screenshots or a small workflow JSON when visual state or workflow loading matters.

## What Not To Share Publicly

Before attaching logs or workflow JSON, remove:

- API keys and tokens.
- Private file paths that reveal client or project names.
- Private prompts, references, images, or generated media.
- Personal contact details.

## Questions And Feature Requests

Use GitHub Issues for now. A reproducible bug report is best as a bug issue;
workflow questions and feature ideas can use a regular issue with enough
context to understand the use case.
