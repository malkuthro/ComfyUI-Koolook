#!/usr/bin/env bash
# Bootstrap a repo-local .venv and install test dependencies.
# Idempotent — no-op if .venv already exists. Pass --force to recreate.
#
# Usage: bash scripts/bootstrap_test_env.sh [--force]

set -euo pipefail

FORCE=""
if [ "${1:-}" = "--force" ]; then
    FORCE="1"
fi

PYTHON_BIN="${PYTHON:-}"
if [ -z "$PYTHON_BIN" ]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    else
        echo "Could not find python3 or python on PATH." >&2
        exit 1
    fi
fi

if [ -d .venv ]; then
    if [ -z "$FORCE" ]; then
        echo ".venv already exists. Pass --force to recreate."
        exit 0
    fi
    echo "Removing existing .venv ..."
    rm -rf .venv
fi

echo "Creating .venv ..."
"$PYTHON_BIN" -m venv .venv

echo "Upgrading pip ..."
.venv/bin/python -m pip install --quiet --upgrade pip

echo "Installing project + test extras ..."
.venv/bin/python -m pip install --quiet -e '.[test]'

echo ""
echo "Test env ready. Run tests with:"
echo "  .venv/bin/python -m pytest"
