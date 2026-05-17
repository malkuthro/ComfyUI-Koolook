# Bootstrap a repo-local .venv and install test dependencies.
# Idempotent — no-op if .venv already exists. Pass -Force to recreate.
#
# Usage: scripts\bootstrap_test_env.ps1 [-Force]

param([switch]$Force)
$ErrorActionPreference = "Stop"

if (Test-Path .venv) {
    if (-not $Force) {
        Write-Host ".venv already exists. Pass -Force to recreate."
        exit 0
    }
    Write-Host "Removing existing .venv ..."
    Remove-Item -Recurse -Force .venv
}

Write-Host "Creating .venv ..."
python -m venv .venv

Write-Host "Upgrading pip ..."
.\.venv\Scripts\python -m pip install --quiet --upgrade pip

Write-Host "Installing project + test extras ..."
.\.venv\Scripts\python -m pip install --quiet -e ".[test]"

Write-Host ""
Write-Host "Test env ready. Run tests with:"
Write-Host "  .\.venv\Scripts\python -m pytest"
