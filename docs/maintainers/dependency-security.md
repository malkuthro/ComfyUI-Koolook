# Dependency security

This project treats the locked test dependency set as the dependency-review
surface. Local bootstrap, pull-request CI, and the weekly scheduled audit all
check `constraints-test.txt` with `pip-audit`.

## What gets audited

- `constraints-test.txt` pins the `[test]` extras from `pyproject.toml` plus
  their transitive dependencies.
- The bootstrap scripts install the test extras against that lock, then run
  `pip-audit` on the installed environment.
- CI audits the committed lock with `pip-audit -r constraints-test.txt` on each
  pull request and on a weekly schedule, so newly disclosed CVEs in old pins are
  caught even when nobody has touched the dependency files.
- Bandit remains a separate first-party-code scan. A green Bandit job does not
  mean the dependency lock is clean.

## Regenerating the lock

Regenerate the lock only as an intentional dependency-change PR:

```powershell
scripts\bootstrap_test_env.ps1 -Force -Relock
```

```bash
bash scripts/bootstrap_test_env.sh --force --relock
```

Review and commit the resulting `constraints-test.txt` diff. Do not hand-edit
individual pins unless you are immediately re-running the relock command to
prove the full resolved set.

## Responding to a CVE

1. Open or update a dependency-security issue with the vulnerable package,
   affected version, fixed version, and audit output.
2. Relock with the commands above.
3. Confirm `pip-audit` reports no known vulnerabilities.
4. Run the focused bootstrap/CI contract tests:

```powershell
.\.venv\Scripts\python -m pytest tests\scripts\test_bootstrap_constraints.py -q
```

5. If the relock changes more than the vulnerable package and normal
   transitive bumps, call that out in the PR description so reviewers know what
   to inspect.

Use `-NoAudit` / `--no-audit` only for explicit offline troubleshooting. Do not
merge a dependency lock that has not passed the audit gate.
