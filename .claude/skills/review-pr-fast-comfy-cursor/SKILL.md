---
name: review-pr-fast-koolook-cursor
description: Quick first-pass ComfyUI-Koolook PR review for Cursor — one Grok 4.5 Task reads the diff and changed files, identifies obvious issues, and posts ONE comment. Use for /review-pr-fast-koolook-cursor (~1-2 min). Escalate to /review-pr-koolook-cursor before merge.
version: 1
disable-model-invocation: true
---

# ComfyUI-Koolook Fast PR Review (Cursor / Grok 4.5)

**Skill version:** v1 — Cursor/Grok first pass for this repo.

**Fast vs deep:** this skill posts **one** PR comment only (never approve /
request-changes). Before merge, run `/review-pr-koolook-cursor` for the full
v2 batched-verify team.

Claude Code sessions can keep using global `/review-pr-fast`.

## Step 0: Parse arguments

- Argument passed (e.g. `/review-pr-fast-koolook-cursor 278`) → `PR_NUM`
- Else ask: "Which PR number should I review?"

## Step 1: Refresh PR head (parent)

Same freshness goal as `/review-pr-koolook-cursor` Step 1 — fetch
`refs/pull/$PR_NUM/head` and ensure file reads target that SHA. Prefer the
detached review worktree path when the current branch is not the PR head.
If a full worktree refresh is too heavy for a fast pass, at minimum:

```bash
gh pr view "$PR_NUM" --json title,body,additions,deletions,changedFiles,files,url,headRefName,baseRefName,headRefOid
git fetch origin "pull/$PR_NUM/head:refs/remotes/origin/pr-$PR_NUM"
```

Store `PR_HEAD_SHA` / `REVIEW_ROOT` (worktree or current tree only if it
already matches `headRefOid`). Do not grep a stale checkout.

## Step 2: Get PR context (parent)

```bash
gh pr view "$PR_NUM" --json title,body,additions,deletions,changedFiles,files,url,headRefName,baseRefName
gh pr diff "$PR_NUM"
```

## Step 3: Read project conventions (parent)

Read root `CLAUDE.md` and `AGENTS.md`. Skim
`docs/maintainers/node-versioning.md` if the PR touches `forks/` or node IDs.

## Step 4: Launch one Grok Task for the review pass

One `Task`, `subagent_type: generalPurpose`, `model: cursor-grok-4.5-high-fast`,
`readonly: true` when supported, `run_in_background: false`:

```markdown
You are doing a quick ComfyUI-Koolook first-pass PR review (review-pr-fast-koolook-cursor).

## PR Context
- PR #<N>: <title>
- URL: <url>
- Branch: <head> -> <base>
- GitHub PR head: <PR_HEAD_SHA>
- Review root: <REVIEW_ROOT>
- Size: +<additions>/-<deletions> across <changedFiles> files

## PR Description
<body>

## Diff
<gh pr diff output, or note if truncated + instruct to fetch via gh / git -C REVIEW_ROOT>

## Conventions already loaded by orchestrator
- CLAUDE.md / AGENTS.md present
- Fork back-compat rules apply under forks/; root Koolook nodes break cleanly unless asked

## Instructions
1. For files with substantial diffs, Read the full file under REVIEW_ROOT (not just hunks).
2. Identify issues focusing on:
   - Correctness: bugs, off-by-one, type mismatches, unhandled errors
   - Silent failures: caught-and-ignored errors, soft defaults that mask bugs,
     snapshot/draft-guard/publish paths that look successful on failure
   - Scope drift: changes outside the PR's stated intent
   - Koolook invariants:
     - Vendoring third-party trees into MAIN / editing outside the repo
     - Fork node ID or RETURN_TYPES breakage without versioning/alias
     - Hardcoded username absolute paths instead of KOLOOK_* / .env
     - Silent constraints-test.txt / [test] lock churn
     - UI changes claiming readiness without visual/harness evidence when mockups apply
   - Tests: missing coverage for new behavior that similar areas already test
3. Return ONLY the markdown body to post as a PR comment (no approve/request-changes).
   - If clean: a short LGTM line that mentions review-pr-fast-koolook-cursor and tells
     the operator to run /review-pr-koolook-cursor before merge.
   - If findings: `## First-pass review (review-pr-fast-koolook-cursor)` then each item as
     `**[BLOCKING|IMPORTANT|NIT]** path:line — one-sentence description`, then
     escalate line pointing at `/review-pr-koolook-cursor`.
4. End the body with: `Model: cursor-grok-4.5-high-fast (review-pr-fast-koolook-cursor)`.
```

## Step 5: Post ONE comment (parent)

Do **NOT** use `gh pr review --approve` or `gh pr review --request-changes`.

```bash
gh pr comment "$PR_NUM" --body-file "$COMMENT_BODY"
```

If the Task failed/timed out, stop and report — do not invent a clean LGTM.

Clean up any temporary review worktree created in Step 1.

## Tone

Terse. One sentence per finding. No filler.

## When to escalate

After this pass, run `/review-pr-koolook-cursor` before merge for the full
invariants / scope / quality / silent-failure team with Medium+ verification.
