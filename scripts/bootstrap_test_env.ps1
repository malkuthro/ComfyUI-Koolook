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

if ((Test-Path .venv) -and (-not $Force)) {
    if ($Relock) {
        Write-Host "-Relock requires -Force (the lock is rewritten from a fresh resolve); nothing was changed."
    }
    Write-Host ".venv already exists. Pass -Force to recreate."
    exit 0
}

if ($Relock) {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "-Relock requires uv to generate the universal Python 3.11 lock; nothing was changed."
    }
    Write-Host "Resolving upgraded universal Python 3.11 test lock ..."
    uv pip compile pyproject.toml --extra test --universal --python-version 3.11 --upgrade --no-annotate --no-emit-package pip --output-file $Constraints
    if ($LASTEXITCODE -ne 0) {
        throw "uv could not generate $Constraints."
    }
}

if (Test-Path .venv) {
    Write-Host "Removing existing .venv ..."
    Remove-Item -Recurse -Force .venv
}

Write-Host "Creating .venv ..."
python -m venv .venv

Write-Host "Upgrading pip + setuptools ..."
.\.venv\Scripts\python -m pip install --quiet --upgrade pip setuptools

if (-not (Test-Path $Constraints)) {
    throw "$Constraints is missing. Run: scripts\\bootstrap_test_env.ps1 -Force -Relock"
}

Write-Host "Installing project + test extras (locked via $Constraints) ..."
.\.venv\Scripts\python -m pip install --quiet -e ".[test]" -c $Constraints

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
