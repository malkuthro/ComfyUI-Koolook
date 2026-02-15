# Workflow (Cristian + kflame)

This is the operating flow for this repo.

## 1) Where planning happens
- Primary workspace: OpenClaw web dashboard session.
- Telegram: quick pings only.

## 2) Source of truth
- Tasks: `TODO.md`
- Progress/handoff: `WORKLOG.md`
- Decisions: `DECISIONS.md`
- Code/config truth: repo files + commits

## 3) Start of each work block
1. Read `TODO.md`
2. Read latest entry in `WORKLOG.md`
3. Read `DECISIONS.md`
4. Pick one task and execute

## 4) End of each work block
1. Update `WORKLOG.md` (done/next/blockers)
2. Update `TODO.md` statuses
3. If a meaningful choice was made, update `DECISIONS.md`
4. Commit changes with clear message

## 5) Task status model
- `NOW`: actively working
- `NEXT`: queued next
- `BLOCKED`: waiting on external dependency
- `DONE`: complete

## 6) Resume from any device
Use this order:
1. `WORKLOG.md` latest entry
2. `TODO.md` NOW/NEXT items
3. `DECISIONS.md`
4. Git history (`git log --oneline -n 10`)

## 7) Rule of thumb
If it matters later, write it to file now.
