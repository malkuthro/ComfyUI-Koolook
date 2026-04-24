# Third-Party Code & Attribution

Use this file to track external repositories, licenses, and local modifications.

## Entry Template

- Name:
- Upstream repo URL:
- Upstream commit/tag used:
- License:
- Local path(s):
- What changed locally:
- Why changed:
- Last reviewed:

## Notes

- Keep this file updated when syncing upstream changes.
- If a node is heavily modified, document behavior differences clearly for users.
- For external fork variants, track exact upstream tag and pinned commit in `forks_manifest.yaml`.
- Freeze baseline first, then namespace node IDs before introducing newer upstream variants.
- Keep paths relative for portability (`../ComfyUI-Forks` by default).
- Keep only wrappers/manifests in MAIN; do not vendor large third-party trees here.
- Keep modified node source files in MAIN when you want full GitHub tracking and one-repo commits.
