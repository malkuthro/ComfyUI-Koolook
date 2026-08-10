---
name: review-pr-comfy-cursor
description: ComfyUI-Koolook PR review for Cursor — v2 batched-verify team (invariants, scope, code quality, silent failure → synthesizer → Medium+ verifier batches) with every subagent on cursor-grok-4.5-high-fast via Task. Use for /review-pr-comfy-cursor. Prefer /review-pr-fast-comfy-cursor for a quick first pass.
version: 2
verifier_pipeline: batched-verify
disable-model-invocation: true
---

# ComfyUI-Koolook PR Review (Cursor / Grok 4.5)

**Skill version:** v2 (`batched-verify`) — Medium+ findings verified in
file-grouped batches of ≤8 (sequential), not one verifier per finding.

Repo-local Cursor skill. **Does not** shadow global `/review-pr`. Claude Code
sessions should keep using global `/review-pr` unless a Claude twin
(`review-pr-comfy`) is added later.

**Canonical copy:** `.claude/skills/review-pr-comfy-cursor/SKILL.md`.
The `.cursor/skills/review-pr-comfy-cursor/SKILL.md` discovery mirror must
stay **byte-identical** (enforced by `tests/test_skill_mirrors.py`).

**Fast vs deep:** use `/review-pr-fast-comfy-cursor` for a ~1–2 min first
pass (one comment, no approve/request-changes). Use **this** skill before
merge when you want the full team + verification gate.

## Runtime mapping

| Stage | Cursor `/review-pr-comfy-cursor` |
|-------|-------------------------------------|
| 4 reviewers | `Task` × 4, `subagent_type: generalPurpose`, `run_in_background: true`, `model: cursor-grok-4.5-high-fast` |
| Synthesizer | `Task`, `generalPurpose`, foreground, same model |
| Verifiers | `Task` × 1 per ≤8-finding batch, sequential, foreground, same model |

| Role | Brief file |
|------|------------|
| Invariants | `.claude/skills/review-pr-comfy-cursor/agents/invariants.md` |
| Scope & Spec | `.claude/skills/review-pr-comfy-cursor/agents/scope-reviewer.md` |
| Code Quality | `.claude/skills/review-pr-comfy-cursor/agents/code-quality.md` |
| Silent Failure | `.claude/skills/review-pr-comfy-cursor/agents/silent-failure-hunter.md` |
| Synthesizer | `.claude/skills/review-pr-comfy-cursor/agents/synthesizer.md` |
| Verifier (batched Medium+) | `.claude/skills/review-pr-comfy-cursor/agents/verifier.md` |

**Important:** always use `subagent_type: generalPurpose` and inject the brief
into the Task `prompt`. Custom role names are **not** in the Cursor Task enum.

For every `Task`: set `model: cursor-grok-4.5-high-fast`. Set `readonly: true`
when the Task schema accepts it (best-effort). Never substitute Sonnet / Opus /
Composer / `inherit` for these subagents.

Launch the 4 reviewers **in parallel** (one message, multiple Task calls).
Synthesizer runs **foreground** after they complete. Verifier batches run
**sequentially**.

## Step 0: Parse arguments

- Argument passed (e.g. `/review-pr-comfy-cursor 278`) → `PR_NUM`
- Else ask: "Which PR number should I review?"

## Step 1: Refresh PR head and base from GitHub

Mandatory. Never launch reviewers until local reads/greps can target the GitHub
PR head. Port of the global `/review-pr` freshness logic.

Run via bash (Git Bash / WSL on Windows):

```bash
set -e

PR_META=$(gh pr view "$PR_NUM" --json title,body,headRefName,baseRefName,additions,deletions,changedFiles,state,url,comments,reviews,reviewDecision)
HEAD_REF=$(gh pr view "$PR_NUM" --json headRefName --jq '.headRefName')
BASE_REF=$(gh pr view "$PR_NUM" --json baseRefName --jq '.baseRefName')
PR_FETCH_REF="refs/remotes/origin/pr-$PR_NUM"

git fetch origin "+refs/heads/${BASE_REF}:refs/remotes/origin/${BASE_REF}" "+refs/pull/${PR_NUM}/head:${PR_FETCH_REF}"
git fetch origin "+refs/heads/${HEAD_REF}:refs/remotes/origin/${HEAD_REF}" 2>/dev/null || true

DIRTY=$(git status --porcelain | head -1)
CURRENT_BRANCH=$(git branch --show-current)
PR_HEAD_SHA=$(git rev-parse "$PR_FETCH_REF")
BASE_SHA=$(git rev-parse "origin/$BASE_REF")
REVIEW_ROOT=$(pwd)
REVIEW_WORKTREE_CREATED="no"
LOCAL_UPSTREAM=$(git rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || true)
TRACKS_PR_HEAD="no"

if [ "$CURRENT_BRANCH" = "$HEAD_REF" ] &&
   [ "$LOCAL_UPSTREAM" = "origin/$HEAD_REF" ] &&
   git show-ref --verify --quiet "refs/remotes/origin/$HEAD_REF"; then
  ORIGIN_HEAD_SHA=$(git rev-parse "origin/$HEAD_REF")
  if [ "$ORIGIN_HEAD_SHA" = "$PR_HEAD_SHA" ]; then
    TRACKS_PR_HEAD="yes"
  fi
fi

if [ "$TRACKS_PR_HEAD" = "yes" ]; then
  BEHIND_PR=$(git rev-list --count "HEAD..$PR_FETCH_REF" 2>/dev/null || echo 0)
  AHEAD_PR=$(git rev-list --count "$PR_FETCH_REF..HEAD" 2>/dev/null || echo 0)
else
  BEHIND_PR="n/a"
  AHEAD_PR="n/a"
fi

create_review_worktree() {
  # Keep the detached checkout inside the repo so Cursor Task Read/Grep
  # (often workspace-scoped) can reach it. /tmp is outside the workspace and
  # can silently fall back to the wrong tree.
  REPO_ROOT=$(git rev-parse --show-toplevel)
  REVIEW_ROOT="$REPO_ROOT/.review-worktrees/pr-$PR_NUM"
  mkdir -p "$REPO_ROOT/.review-worktrees"
  if [ -e "$REVIEW_ROOT" ]; then
    git worktree remove --force "$REVIEW_ROOT" 2>/dev/null || rm -rf "$REVIEW_ROOT"
  fi
  git worktree add --detach "$REVIEW_ROOT" "$PR_FETCH_REF"
  REVIEW_WORKTREE_CREATED="yes"
}

verify_review_root() {
  REVIEW_SHA=$(git -C "$REVIEW_ROOT" rev-parse HEAD)
  if [ "$REVIEW_SHA" != "$PR_HEAD_SHA" ]; then
    echo "ERROR: review root $REVIEW_ROOT is at $REVIEW_SHA, expected PR head $PR_HEAD_SHA"
    exit 1
  fi
}

if [ "$TRACKS_PR_HEAD" = "yes" ] && [ -n "$DIRTY" ]; then
  SYNC_STATUS="warn-dirty"
  SYNC_NOTE="Sync: WARN -- local $HEAD_REF has uncommitted changes. Creating detached review worktree at GitHub PR head $PR_HEAD_SHA."
  create_review_worktree
elif [ "$TRACKS_PR_HEAD" = "yes" ] && [ "$BEHIND_PR" = "0" ] && [ "$AHEAD_PR" = "0" ]; then
  SYNC_STATUS="clean"
  SYNC_NOTE="Sync: local $HEAD_REF is current at $PR_HEAD_SHA"
elif [ "$TRACKS_PR_HEAD" = "yes" ] && [ "$AHEAD_PR" != "0" ]; then
  SYNC_STATUS="warn-ahead"
  SYNC_NOTE="Sync: WARN -- local $HEAD_REF has $AHEAD_PR unpushed commit(s). Creating detached review worktree at GitHub PR head $PR_HEAD_SHA."
  create_review_worktree
elif [ "$TRACKS_PR_HEAD" = "yes" ]; then
  SYNC_STATUS="fast-forward"
  SYNC_NOTE="Sync: fast-forwarding local $HEAD_REF by $BEHIND_PR commit(s) to GitHub PR head $PR_HEAD_SHA"
  git merge --ff-only "$PR_FETCH_REF"
  REVIEW_ROOT=$(pwd)
else
  SYNC_STATUS="fetched"
  SYNC_NOTE="Sync: fetched PR head $PR_HEAD_SHA and base origin/$BASE_REF at $BASE_SHA. Current branch is $CURRENT_BRANCH, so creating detached review worktree."
  create_review_worktree
fi

verify_review_root

echo "$SYNC_NOTE"
echo "SYNC_STATUS=$SYNC_STATUS"
echo "PR_HEAD_SHA=$PR_HEAD_SHA"
echo "REVIEW_ROOT=$REVIEW_ROOT"
echo "REVIEW_DIFF=git -C \"$REVIEW_ROOT\" diff \"$BASE_SHA...$PR_HEAD_SHA\""
```

Store: `PR_META`, `HEAD_REF`, `BASE_REF`, `PR_HEAD_SHA`, `BASE_SHA`,
`SYNC_STATUS`, `SYNC_NOTE`, `REVIEW_ROOT`, `REVIEW_WORKTREE_CREATED`,
`REVIEW_DIFF`.

If `gh pr view` or `git fetch` fails, **stop**. Never fast-forward a dirty tree
or a branch with unpushed commits — use the detached worktree path instead.

## Step 2: Fetch PR discussion and file list

```bash
git -C "$REVIEW_ROOT" diff --name-only "$BASE_SHA...$PR_HEAD_SHA"
gh api repos/$(gh repo view --json nameWithOwner --jq '.nameWithOwner')/pulls/$PR_NUM/comments --paginate
```

Flatten comments/reviews into markdown (leads to verify, not truth):

```markdown
- Review decision: <reviewDecision or "none">
- Comments: …
- Reviews: …
- Inline review comments: …
```

Search PR body for issue links and any referenced docs under `docs/`.

Store: `PR_FILES`, `PR_DISCUSSION`, `SPEC_DOC` (path or "none").

## Step 3: Diff size check

From `additions + deletions`:

- **Under 3000 lines:** Diff Access may use
  `git -C "<REVIEW_ROOT>" diff "<BASE_SHA>...<PR_HEAD_SHA>"` and/or
  `gh pr diff $PR_NUM`.
- **Over 3000 lines:** instruct agents to use
  `gh api …/pulls/$PR_NUM/files --paginate` and per-file `patch` only.
  Do not use `gh pr diff` with a pathspec.

Always tell agents: **Read/Grep files only under `REVIEW_ROOT`.**

## Step 4: Launch 4 reviewers in parallel

Read each brief. For each role, one Task (`generalPurpose`, background, Grok):

```markdown
<brief content after frontmatter>

## PR Context
- PR #<N>: <title>
- URL: <url>
- Branch: <head> -> <base>
- GitHub PR head: <PR_HEAD_SHA>
- Review root: <REVIEW_ROOT>
- Size: +<additions>/-<deletions> across <changedFiles> files
- Spec doc: <path or "none">
- Sync: <SYNC_NOTE>

## PR Description
<body>

## Existing PR Discussion
<PR_DISCUSSION>

## Changed Files
<file list>

## Diff Access
<from Step 3>
Read and grep files only under <REVIEW_ROOT>.

## Instructions
1. Read existing PR discussion and decide which claims need verification.
2. Fetch diff context per Diff Access above.
3. Review per your brief.
4. Produce your structured report; note confirmed, refuted, or superseded prior comments.
```

## Step 5: Collect results

Wait for all 4 Tasks. On failure/timeout:

```text
<agent name>: FAILED — <reason>
```

## Step 6: Launch synthesizer (foreground)

Read `agents/synthesizer.md`. One Task, `run_in_background: false`:

```markdown
<synthesizer brief>

---

Review PR #<N> — <title>

Here are the 4 review agent reports. Deduplicate, prioritize, and produce the consolidated review.

## Invariants Report
<output>

## Scope & Spec Report
<output>

## Code Quality Report
<output>

## Silent Failure Report
<output>
```

## Step 7: Verify findings (conditional, batched)

### 7a. Gate

- Collect Medium / High / Blocking from synthesizer (`Highest severity:` + findings).
- **Fail closed** if `Highest severity:` is missing or contradicts the findings/table —
  treat as having verifiable findings (do not skip).
- Low findings = `unverified (nit)`.
- Clean or Low-only → **skip verification**; go to Step 8 with synthesized review.
  Verdict APPROVE (or NEEDS_DISCUSSION for an open product question).
  **Never REQUEST_CHANGES on the skip path.**
- Medium+ → continue to 7b.

### 7b. Batched Tasks (group by file, ≤8, sequential, preserve partials)

Read `agents/verifier.md` (`maxTurns: 60` — do not lower it).

1. Group verifiable findings by cited file (path, then line).
2. Chunk into batches of ≤8; keep same-file clusters together when ≤8.
3. Run batches **sequentially**.
4. Each batch: one Task, `generalPurpose`, Grok, foreground.

```markdown
<verifier brief>

---

Verify the Medium+ findings listed below from PR #<N> — <title>.
Do not invent findings. Output one classification block per finding, in list order.
Findings are grouped by cited file — read each file once under <REVIEW_ROOT>, then
classify all of its findings before moving on.

## Findings to verify

### File: `<path/a.py>`
#### Finding 1 — Title, Severity, File, Rule, Issue, Proposed fix
#### Finding 2 — …

### File: `<path/b.js>`
#### Finding 3 — …

## PR Context / Description / Discussion / Changed Files / Diff Access
<same as Step 4; include REVIEW_ROOT and PR_HEAD_SHA>

## Instructions
1. Process file groups in order; one read per file under REVIEW_ROOT, then classify.
2. Classify each: CONFIRMED, FALSE POSITIVE, DOWNGRADED, or UNVERIFIABLE.
3. Cross-reference each fix against issue/spec.
4. Emit one verified block per listed finding.
5. If you must stop early, still emit every completed block.
```

**On failure / incomplete batch:** keep every complete
`### Verified finding:` block; mark only unreached findings
`UNVERIFIABLE — verifier did not complete`; retry once with only unreached
findings. Never wipe completed classifications.

### 7c. Assemble final review

```markdown
## Verified PR Review

**PR:** #<N> — <title>
**Findings:** <verifiable> verified — <confirmed> confirmed, <downgraded> downgraded, <dropped> dropped, <unverified> unverified
**Verdict:** APPROVE / REQUEST_CHANGES / NEEDS_DISCUSSION

### Blocking
…

### High
…

### Medium / Low
… (Low labeled `unverified (nit)`)

### Dropped / Downgraded / Unverified (audit)
…
```

Verdict rules (fail closed):

- **REQUEST_CHANGES** — ≥1 confirmed Blocking
- **NEEDS_DISCUSSION** — no Blocking, but surviving High, open product question,
  or Blocking/High left `UNVERIFIABLE — verifier did not complete`
- **APPROVE** — no surviving Blocking/High, no Blocking/High stuck unverified
  from verifier non-completion, and no other actionable confirmed concerns
  (per `AGENTS.md` approve-by-default)

Do not present raw synthesizer output when verification ran.

## Step 8: Present and post

If `REVIEW_WORKTREE_CREATED` is `yes`, remove it after verification:

```bash
git worktree remove --force "$REVIEW_ROOT"
```

If `SYNC_STATUS` is `warn-dirty`, `warn-ahead`, or `fetched`, prepend:

```markdown
Note: <SYNC_NOTE> Review was anchored to GitHub PR head `<PR_HEAD_SHA>`.
```

Print the final review, then:

```text
---
Review complete. Say "post it" to submit as a GitHub PR review, or "edit" to modify first.
```

On "post it": write body to a temp file; then:

```bash
# APPROVE
gh pr review $PR_NUM --approve --body-file "$REVIEW_BODY"
# REQUEST_CHANGES
gh pr review $PR_NUM --request-changes --body-file "$REVIEW_BODY"
# NEEDS_DISCUSSION
gh pr review $PR_NUM --comment --body-file "$REVIEW_BODY"
```

Self-approval rejected by GitHub → post as `--comment` and report that.

## Error handling

- Refresh / `gh pr view` / fetch failure → stop
- Empty diff → "PR has no diff — nothing to review."
- Reviewer timeout → note in synthesis
- Verifier non-completion on Blocking/High → blocks APPROVE (7c)
- Worktree create failure → stop; do not review a stale checkout
