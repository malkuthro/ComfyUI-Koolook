"""Guards on the locked test dependency set (``constraints-test.txt``).

The bootstrap scripts install the test extras against this pinned lock, so
every fresh ``.venv`` is reproducible and ``pip-audit``-verifiable. These
tests fail loudly if the lock drifts out of sync with the ``[test]`` extras
declared in ``pyproject.toml`` (e.g. an extra was added without re-locking).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[2]
CONSTRAINTS = REPO_ROOT / "constraints-test.txt"
PYPROJECT = REPO_ROOT / "pyproject.toml"


def _canonical(name: str) -> str:
    """PEP 503 normalised distribution name."""
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def _pinned_names() -> dict[str, str]:
    """Map canonical distribution name -> exact version from the lock."""
    pins: dict[str, str] = {}
    for raw in CONSTRAINTS.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        assert "==" in line, f"constraint not pinned with '==': {line!r}"
        name, version = line.split("==", 1)
        pins[_canonical(name)] = version.strip()
    return pins


def _test_extra_names() -> list[str]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]["test"]
    return [_canonical(re.split(r"[<>=!~;\[ ]", spec, maxsplit=1)[0]) for spec in extras]


def test_constraints_file_exists_and_nonempty():
    assert CONSTRAINTS.is_file(), "constraints-test.txt is missing"
    assert _pinned_names(), "constraints-test.txt has no pinned entries"


def test_no_editable_or_self_package_leaked():
    text = CONSTRAINTS.read_text(encoding="utf-8")
    assert "-e " not in text, "an editable install leaked into the lock"
    assert "koolook" not in _pinned_names(), "the self package leaked into the lock"


def test_every_top_level_test_extra_is_pinned():
    pins = _pinned_names()
    missing = [name for name in _test_extra_names() if name not in pins]
    assert not missing, (
        f"these [test] extras are not pinned in constraints-test.txt: {missing}. "
        "Regenerate the lock: bash scripts/bootstrap_test_env.sh --force --relock"
    )


def test_no_duplicate_pins():
    names: list[str] = []
    for raw in CONSTRAINTS.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            names.append(_canonical(line.split("==", 1)[0]))
    dupes = sorted({name for name in names if names.count(name) > 1})
    assert not dupes, f"duplicate pins in constraints-test.txt: {dupes}"


def test_bootstrap_upgrades_setuptools_before_audit():
    """Fresh venvs should not fail audit on the ensurepip setuptools seed."""
    ps1 = (REPO_ROOT / "scripts" / "bootstrap_test_env.ps1").read_text(encoding="utf-8")
    sh = (REPO_ROOT / "scripts" / "bootstrap_test_env.sh").read_text(encoding="utf-8")

    assert "--upgrade pip setuptools" in ps1
    assert "--upgrade pip setuptools" in sh


def test_bootstrap_relock_does_not_inspect_editable_git_metadata():
    """Relock should work from cross-drive git worktrees on Windows."""
    ps1 = (REPO_ROOT / "scripts" / "bootstrap_test_env.ps1").read_text(encoding="utf-8")
    sh = (REPO_ROOT / "scripts" / "bootstrap_test_env.sh").read_text(encoding="utf-8")

    assert "pip list --format=freeze" in ps1
    assert "pip list --format=freeze" in sh
    assert "pip freeze --exclude-editable" not in ps1
    assert "pip freeze --exclude-editable" not in sh


def test_ci_audits_committed_lock_on_prs_and_schedule():
    ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "schedule:" in ci
    assert "pip-audit -r constraints-test.txt" in ci


def test_ci_pytest_installs_from_committed_lock():
    ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert '-e ".[test]" -c constraints-test.txt' in ci
