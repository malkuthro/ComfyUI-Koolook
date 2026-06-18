# Bootstrap a repo-local .venv with the LOCKED, AUDITED test dependency set.
#
#   * Reproducible -- installs against constraints-test.txt when present (a
#     pinned resolve of `.[test]` + its full transitive closure). Pass
#     -Relock (with -Force) to re-resolve and rewrite that lock; commit the
#     diff as the dependency-change review surface.
#   * Verified -- runs pip-audit after install; a known CVE fails the
#     bootstrap (exit 1). Pass -NoAudit to skip (e.g. offline).
#   * Idempotent -- no-op if .venv already exists. Pass -Force to recreate.
#
# Usage: scripts\bootstrap_test_env.ps1 [-Force] [-Relock] [-NoAudit]

param(
    [switch]$Force,
    [switch]$Relock,
    [switch]$NoAudit
)
$ErrorActionPreference = "Stop"

$Constraints = "constraints-test.txt"

if (Test-Path .venv) {
    if (-not $Force) {
        if ($Relock) {
            Write-Host "-Relock requires -Force (the lock is rewritten from a fresh resolve); nothing was changed."
        }
        Write-Host ".venv already exists. Pass -Force to recreate."
        exit 0
    }
    Write-Host "Removing existing .venv ..."
    Remove-Item -Recurse -Force .venv
}

Write-Host "Creating .venv ..."
python -m venv .venv

Write-Host "Upgrading pip + setuptools ..."
.\.venv\Scripts\python -m pip install --quiet --upgrade pip setuptools

if ((Test-Path $Constraints) -and (-not $Relock)) {
    Write-Host "Installing project + test extras (locked via $Constraints) ..."
    .\.venv\Scripts\python -m pip install --quiet -e ".[test]" -c $Constraints
} else {
    Write-Host "Resolving + installing project + test extras ..."
    .\.venv\Scripts\python -m pip install --quiet -e ".[test]"
    Write-Host "Writing locked set to $Constraints ..."
    @'
# Locked test dependency set for ComfyUI-Koolook.
#
# Pinned resolve of the `[test]` extras in pyproject.toml plus their full
# transitive closure. The bootstrap scripts install with `-c
# constraints-test.txt`, so every fresh .venv is reproducible and
# pip-audit-verifiable.
#
# DO NOT hand-edit the version pins. To change the set: edit the `[test]`
# extras in pyproject.toml, then regenerate this file with
#   bash scripts/bootstrap_test_env.sh --force --relock   (POSIX)
#   scripts\bootstrap_test_env.ps1 -Force -Relock         (Windows)
# and commit the diff -- that diff is the dependency-change review surface.
#
'@ | Out-File -Encoding ascii $Constraints
    .\.venv\Scripts\python -m pip list --format=freeze --exclude pip --exclude setuptools --exclude wheel --exclude koolook | Out-File -Encoding ascii -Append $Constraints
}

if (-not $NoAudit) {
    Write-Host "Auditing installed set (pip-audit) ..."
    .\.venv\Scripts\pip-audit --skip-editable
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "BLOCKER: pip-audit did not pass -- a known vulnerability was found,"
        Write-Host "or the audit could not complete. Review the output above."
        Write-Host "To bootstrap anyway (e.g. offline), re-run with -NoAudit."
        exit 1
    }
    Write-Host "pip-audit: no known vulnerabilities."
}

Write-Host ""
Write-Host "Test env ready. Run tests with:"
Write-Host "  .\.venv\Scripts\python -m pytest"
