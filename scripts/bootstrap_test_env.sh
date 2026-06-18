#!/usr/bin/env bash
# Bootstrap a repo-local .venv with the LOCKED, AUDITED test dependency set.
#
#   * Reproducible -- installs against constraints-test.txt when present (a
#     pinned resolve of `.[test]` + its full transitive closure). Pass
#     --relock (with --force) to re-resolve and rewrite that lock; commit
#     the diff as the dependency-change review surface.
#   * Verified -- runs pip-audit after install; a known CVE fails the
#     bootstrap (exit 1). Pass --no-audit to skip (e.g. offline).
#   * Idempotent -- no-op if .venv already exists. Pass --force to recreate.
#
# Usage: bash scripts/bootstrap_test_env.sh [--force] [--relock] [--no-audit]

set -euo pipefail

FORCE=""
RELOCK=""
AUDIT="1"
for arg in "$@"; do
    case "$arg" in
        --force)    FORCE="1" ;;
        --relock)   RELOCK="1" ;;
        --no-audit) AUDIT="" ;;
        *) echo "Unknown argument: $arg" >&2
           echo "Usage: bash scripts/bootstrap_test_env.sh [--force] [--relock] [--no-audit]" >&2
           exit 2 ;;
    esac
done

CONSTRAINTS="constraints-test.txt"

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
        if [ -n "$RELOCK" ]; then
            echo "--relock requires --force (the lock is rewritten from a fresh resolve); nothing was changed." >&2
        fi
        echo ".venv already exists. Pass --force to recreate."
        exit 0
    fi
    echo "Removing existing .venv ..."
    rm -rf .venv
fi

echo "Creating .venv ..."
"$PYTHON_BIN" -m venv .venv

echo "Upgrading pip + setuptools ..."
.venv/bin/python -m pip install --quiet --upgrade pip setuptools

if [ -f "$CONSTRAINTS" ] && [ -z "$RELOCK" ]; then
    echo "Installing project + test extras (locked via $CONSTRAINTS) ..."
    .venv/bin/python -m pip install --quiet -e '.[test]' -c "$CONSTRAINTS"
else
    echo "Resolving + installing project + test extras ..."
    .venv/bin/python -m pip install --quiet -e '.[test]'
    echo "Writing locked set to $CONSTRAINTS ..."
    cat > "$CONSTRAINTS" <<'EOF'
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
EOF
    .venv/bin/python -m pip list --format=freeze --exclude pip --exclude setuptools --exclude wheel --exclude koolook >> "$CONSTRAINTS"
fi

if [ -n "$AUDIT" ]; then
    echo "Auditing installed set (pip-audit) ..."
    if .venv/bin/pip-audit --skip-editable; then
        echo "pip-audit: no known vulnerabilities."
    else
        echo "" >&2
        echo "BLOCKER: pip-audit did not pass -- a known vulnerability was found," >&2
        echo "or the audit could not complete. Review the output above." >&2
        echo "To bootstrap anyway (e.g. offline), re-run with --no-audit." >&2
        exit 1
    fi
fi

echo ""
echo "Test env ready. Run tests with:"
echo "  .venv/bin/python -m pytest"
