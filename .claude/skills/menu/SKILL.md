---
name: menu
description: List all project-level skills available under .claude/skills/ with their one-line descriptions, so the maintainer can browse what tools are available without remembering each one. Use when the user types /menu, "list skills", "what skills do we have", or "show me the skill index". This is the project-specific inventory; user-level skills under ~/.claude/skills/ are listed separately when relevant.
---

# menu

Quick inventory of the project-level skills under `.claude/skills/`.
Run when the user wants a one-look overview of what's available without
having to remember each skill by name.

## Trigger phrases

- `/menu`
- "list skills"
- "what skills do we have"
- "show me the skill index"
- "skill menu"

## Procedure

1. **List skills under `.claude/skills/`.** Use the agent's Glob or Bash
   tool to enumerate `.claude/skills/*/SKILL.md`.

2. **Parse each SKILL.md frontmatter.** YAML frontmatter is delimited by
   `---` lines at the top of each file. Extract:
   - `name` — the skill's canonical name.
   - `description` — the one-liner used for trigger matching. Often
     long; truncate to the first sentence (text up to the first period
     or 200 characters, whichever comes first) for the menu display.

3. **Render as a table.** Markdown table with columns:
   - **Name** — the skill name (also functions as a link if rendered in
     a viewer that supports relative links).
   - **One-liner** — the truncated description.
   - **How to invoke** — the trigger phrase. Default form is `/<name>`
     (e.g. `/docs-sync`) or the first phrase from the description if
     the skill has more specific triggers.

4. **(Optional) List user-level skills.** If the user asks for "all
   skills" or the project list is short (< 3 skills), also enumerate
   `~/.claude/skills/*/SKILL.md` (or the equivalent on Windows) under
   a separate "User-level skills" heading. Skip this section by default
   to keep the output tight.

5. **End with a short reminder** of how to add new skills:
   *"To add a new project skill: create `.claude/skills/<name>/SKILL.md`
   with YAML frontmatter (`name:`, `description:`) and a markdown body
   describing the procedure. Re-run `/menu` to see it appear here."*

## Output format

Match this layout (substitute real values):

```
## Project skills

| Name | One-liner | Invoke |
|---|---|---|
| docs-sync | Find-and-replace a version tag across docs, code, and manifests in one controlled pass. | `/docs-sync <old> <new>` |
| add-external-fork | Set up an external upstream repository as a pinned reference checkout under ../ComfyUI-Forks. | `/add-external-fork` |
| license-pre-check | Run a license-compatibility audit before incorporating any third-party code. | `/license-pre-check` |
| menu | List all project-level skills with their one-line descriptions. | `/menu` |

To add a new project skill: create `.claude/skills/<name>/SKILL.md`
with YAML frontmatter and a markdown body. Re-run `/menu` to see it.
```

## Safety guardrails

- **Read-only.** This skill never edits files.
- **No external calls.** Reads only the local filesystem.

## Related skills

- (none — this skill is purely an indexer)
