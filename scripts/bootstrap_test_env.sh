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

if [ -d .venv ]; then
    if [ -z "$FORCE" ]; then
        echo ".venv already exists. Pass --force to recreate."
        exit 0
    fi
    echo "Removing existing .venv ..."
    rm -rf .venv
fi

echo "Creating .venv ..."
python -m venv .venv

echo "Upgrading pip ..."
.venv/bin/python -m pip install --quiet --upgrade pip

echo "Installing project + test extras ..."
.venv/bin/python -m pip install --quiet -e '.[test]'

echo ""
echo "Test env ready. Run tests with:"
echo "  .venv/bin/python -m pytest"
