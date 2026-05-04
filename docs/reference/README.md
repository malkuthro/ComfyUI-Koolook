# Reference

Lookup material — definitions, lists, spec sheets. Anything someone would
search for to answer a "what is X?" or "which nodes do Y?" question.

## Contents

| File | What it is |
|---|---|
| [`glossary.md`](glossary.md) | Plain-English definitions for the terms used in this repo's workflow (MAIN, fork, namespace suffix, sibling project, etc.) |
| [`versioning.md`](versioning.md) | The three independent version axes in this codebase (pack version, fork wrapper version, upstream pinned commit) — what each one means, when each one bumps, and which skill/doc owns it |
| [`comfyUI_nodes-inventory_by_NAME.png`](comfyUI_nodes-inventory_by_NAME.png) | Screenshot inventory of all currently-registered nodes by display name (helpful for matching what you see in ComfyUI back to the source files) |

## When to add a reference doc here

- It's a *fact*, not a *procedure* (procedures go in [`../maintainers/`](../maintainers/))
- It's something someone would *look up*, not *read top-to-bottom* (narrative content goes in [`../user_guide/`](../user_guide/))
- It's worth keeping in sync as the codebase evolves
