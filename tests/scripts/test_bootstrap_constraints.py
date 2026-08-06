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
        pins[_canonical(name)] = version.split(";", 1)[0].strip()
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
    pins: list[str] = []
    for raw in CONSTRAINTS.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            pins.append(re.sub(r"\s+", " ", line))
    dupes = sorted({pin for pin in pins if pins.count(pin) > 1})
    assert not dupes, f"duplicate pins in constraints-test.txt: {dupes}"


def test_bootstrap_upgrades_setuptools_before_audit():
    """Fresh venvs should not fail audit on the ensurepip setuptools seed."""
    ps1 = (REPO_ROOT / "scripts" / "bootstrap_test_env.ps1").read_text(encoding="utf-8")
    sh = (REPO_ROOT / "scripts" / "bootstrap_test_env.sh").read_text(encoding="utf-8")

    assert "--upgrade pip setuptools" in ps1
    assert "--upgrade pip setuptools" in sh


def test_bootstrap_relock_checks_uv_before_replacing_venv():
    """A missing relock prerequisite must not discard a usable test env."""
    ps1 = (REPO_ROOT / "scripts" / "bootstrap_test_env.ps1").read_text(encoding="utf-8")
    sh = (REPO_ROOT / "scripts" / "bootstrap_test_env.sh").read_text(encoding="utf-8")

    assert ps1.index("Get-Command uv") < ps1.index("Remove-Item -Recurse -Force .venv")
    assert sh.index("command -v uv") < sh.index("rm -rf .venv")
    assert "pip list --format=freeze" not in ps1
    assert "pip list --format=freeze" not in sh


def test_bootstrap_relock_generates_a_universal_ci_lock():
    """Relocks include conditional dependencies for both CI platforms."""
    ps1 = (REPO_ROOT / "scripts" / "bootstrap_test_env.ps1").read_text(encoding="utf-8")
    sh = (REPO_ROOT / "scripts" / "bootstrap_test_env.sh").read_text(encoding="utf-8")

    for script in (ps1, sh):
        assert "uv pip compile pyproject.toml --extra test --universal" in script
        assert "--python-version 3.11" in script
        assert "--upgrade" in script
        assert "--no-emit-package pip" in script


def test_constraints_cover_conditional_ci_dependencies():
    pins = _pinned_names()

    assert "typing-extensions" in pins
    assert "colorama" in pins


def test_marker_versions_are_parsed_without_environment_markers():
    pins = _pinned_names()

    assert pins["typing-extensions"] == "4.16.0"
    assert pins["colorama"] == "0.4.6"


def test_ci_audits_committed_lock_on_prs_and_schedule():
    ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "schedule:" in ci
    assert "pip-audit -r constraints-test.txt" in ci
    assert "pip-audit --no-deps -r /tmp/constraints-all-platforms.txt" in ci
    assert "sed -E 's/[[:space:]]*;.*$//' constraints-test.txt" in ci


def test_ci_pytest_installs_from_committed_lock():
    ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert '-e ".[test]" -c constraints-test.txt' in ci
