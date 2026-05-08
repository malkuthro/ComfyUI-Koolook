# ComfyUI-Koolook Codex Instructions

## Required Project Context

Read `CLAUDE.md` before making substantive changes. It contains the
project-specific workflow, dev-sync rules, release rules, and sidebar
maintenance notes.

## Visual QA Is Mandatory For UI Work

For any change that affects rendered UI, guide pages, screenshots,
CSS/layout, onboarding docs, or browser-visible behavior:

1. Do not rely on CSS coordinates, source inspection, or user screenshots alone.
2. Open the changed page/app in a browser-rendered environment that Codex can
   screenshot.
3. If a `file://` guide page is blocked by browser tooling, serve the repo over
   local HTTP instead, for example:
   `python3 -m http.server <port>` from the repo root, then open
   `http://127.0.0.1:<port>/...`.
4. Take a rendered screenshot after the change and inspect the actual visual
   result before committing or pushing.
5. If visual verification is blocked, say that clearly and do not present the UI
   change as ready unless the user explicitly waives visual QA.

This is especially important for overlays, icon highlights, responsive layout,
text fitting, and visual guide pages.
