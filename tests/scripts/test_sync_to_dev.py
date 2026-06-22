"""Guards on which modules ``dev-sync`` ships, so the loader can never point
at a file the install lacks.

The #183 incident (root cause behind #198): a module that ``__init__.py``
imported but ``scripts/sync_to_dev.py``'s ``RUNTIME_PATHS`` never copied raised
``ImportError`` at startup. After the #198 refactor the blast radius of such a
gap depends on *where* the import sits — a per-group node import degrades to
"that node absent", but the context probe (``koolook_versioning``) and the
install guard gate *all* registration. So the real invariant is: every module
``__init__.py`` imports at load must be shipped by ``RUNTIME_PATHS``. This test
derives those imports from ``__init__.py`` itself (rather than a ``k_*.py``
glob, which misses the non-``k_`` gates) so it can't drift.
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = REPO_ROOT / "scripts" / "sync_to_dev.py"
_spec = importlib.util.spec_from_file_location("koolook_sync_to_dev_under_test", _SCRIPT)
assert _spec is not None and _spec.loader is not None
sync_to_dev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sync_to_dev)


def _init_load_time_module_deps() -> set[str]:
    """Top-level package modules/packages ``__init__.py`` imports at load.

    Covers relative ``from .X import ...`` / ``from . import X`` statements and
    the ``_merge_node_group(label, ".X")`` node-group imports. Returns the
    first path segment of each (e.g. ``forks.radiance_koolook`` -> ``forks``),
    the granularity ``RUNTIME_PATHS`` ships at. Absolute imports (the install
    guard's out-of-package fallback) are level 0 and intentionally ignored.
    """
    tree = ast.parse((REPO_ROOT / "__init__.py").read_text(encoding="utf-8"))
    deps: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.level or 0) >= 1:
            if node.module:  # from .pkg import ...
                deps.add(node.module.split(".")[0])
            else:  # from . import a, b
                deps.update(alias.name.split(".")[0] for alias in node.names)
        elif (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "_merge_node_group"
        ):
            for arg in node.args:
                if (
                    isinstance(arg, ast.Constant)
                    and isinstance(arg.value, str)
                    and arg.value.startswith(".")
                ):
                    deps.add(arg.value.lstrip(".").split(".")[0])
    return deps


def _route_load_time_module_deps() -> set[str]:
    tree = ast.parse((REPO_ROOT / "koolook_routes.py").read_text(encoding="utf-8"))
    deps: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if (node.level or 0) >= 1 and node.module:
                deps.add(node.module.split(".")[0])
            elif node.module and node.module.startswith("koolook_"):
                deps.add(node.module.split(".")[0])
    return deps


def test_init_load_deps_are_shipped_by_runtime_paths():
    runtime_paths = set(sync_to_dev.RUNTIME_PATHS)
    deps = _init_load_time_module_deps() | _route_load_time_module_deps()
    # Sanity: the parse actually found the imports (guards against a refactor
    # that renames _merge_node_group and silently makes this test vacuous).
    assert "koolook_versioning" in deps and "k_ai_pipeline" in deps, (
        f"__init__.py import parse looks broken — only found: {sorted(deps)}"
    )
    # A single-file module ships as "<dep>.py"; a package (e.g. forks) ships as
    # its top-level dir entry.
    missing = sorted(
        dep
        for dep in deps
        if f"{dep}.py" not in runtime_paths and dep not in runtime_paths
    )
    assert not missing, (
        "__init__.py imports these at load but dev-sync RUNTIME_PATHS does not "
        f"ship them: {missing}. Add them to RUNTIME_PATHS in "
        "scripts/sync_to_dev.py so a plain dev-sync can't 404 the loader."
    )


def test_dev_sync_ships_manager_visible_package_metadata():
    """The live dev install should report the same pack version as the source.

    ComfyUI Manager reads package metadata from ``pyproject.toml``. If dev-sync
    skips it, the code can be current while the UI still reports an older
    installed version.
    """
    assert "pyproject.toml" in sync_to_dev.RUNTIME_PATHS


# --- find_dotenv: worktree -> main-repo .env fallback (dev-sync from a worktree) ---

def test_find_dotenv_prefers_worktree_env(tmp_path, monkeypatch):
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / ".env").write_text("X=1\n", encoding="utf-8")
    monkeypatch.setattr(sync_to_dev, "REPO_ROOT", wt)
    assert sync_to_dev.find_dotenv() == wt / ".env"


def test_find_dotenv_falls_back_to_main_repo_from_worktree(tmp_path, monkeypatch):
    main = tmp_path / "ComfyUI-Koolook"
    (main / ".git" / "worktrees" / "wt").mkdir(parents=True)
    (main / ".env").write_text("KOLOOK_COMFYUI_DEV_PATH=/somewhere\n", encoding="utf-8")
    wt = main / ".claude" / "worktrees" / "wt"
    wt.mkdir(parents=True)
    gitdir = main / ".git" / "worktrees" / "wt"
    (wt / ".git").write_text(f"gitdir: {gitdir.as_posix()}\n", encoding="utf-8")
    monkeypatch.setattr(sync_to_dev, "REPO_ROOT", wt)
    assert sync_to_dev.find_dotenv() == main / ".env"


def test_find_dotenv_none_when_no_env_and_not_a_worktree(tmp_path, monkeypatch):
    plain = tmp_path / "plain"
    plain.mkdir()
    monkeypatch.setattr(sync_to_dev, "REPO_ROOT", plain)
    assert sync_to_dev.find_dotenv() is None
