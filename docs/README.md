# ComfyUI-Koolook documentation

Three audience buckets — pick where to look based on what you're trying to do.

| If you want to … | Go to | One-line description |
|---|---|---|
| Use a node in your ComfyUI workflow | [`user_guide/`](user_guide/) | Per-node guides, screenshots, recipes |
| Look something up — a term, a node ID, a list | [`reference/`](reference/) | [Glossary](reference/glossary.md), node inventory, spec sheets |
| Cut a release / fork a node / debug CI / hit the Comfy Registry API | [`maintainers/`](maintainers/) | [Releasing](maintainers/releasing.md), [Registry API](maintainers/registry-api.md), [Node versioning rules](maintainers/node-versioning.md) |
| Compare UI design proposals before implementation | [`designs/`](designs/) | Focused mockups and screenshots for in-progress UX decisions |

## Files that intentionally stay at the repo root

| File | Why it's not in `docs/` |
|---|---|
| [`../README.md`](../README.md) | GitHub repo home + Comfy Registry pulls description from here |
| [`../LICENSE`](../LICENSE) | `pyproject.toml` references it; GitHub recognizes the license badge from this path only |
| [`../CHANGELOG.md`](../CHANGELOG.md) | Changelog parsers and release tooling expect it at root |
| [`../CLAUDE.md`](../CLAUDE.md) | Claude Code reads project-level agent instructions from `<repo>/CLAUDE.md` |
