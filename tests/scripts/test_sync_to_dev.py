"""Guard: every root node module is shipped by dev-sync ``RUNTIME_PATHS``.

The #183 incident (root cause behind #198) was a node module
(``k_audio_timeline.py``) that ``__init__.py`` imported but
``scripts/sync_to_dev.py``'s ``RUNTIME_PATHS`` never copied, so dev installs
raised ``ImportError`` at startup. This test fails loudly if a new root
``k_*.py`` node module is ever added without listing it in ``RUNTIME_PATHS``.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = REPO_ROOT / "scripts" / "sync_to_dev.py"
_spec = importlib.util.spec_from_file_location("koolook_sync_to_dev_under_test", _SCRIPT)
assert _spec is not None and _spec.loader is not None
sync_to_dev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sync_to_dev)


def test_every_root_node_module_is_in_runtime_paths():
    root_node_modules = {p.name for p in REPO_ROOT.glob("k_*.py")}
    runtime_paths = set(sync_to_dev.RUNTIME_PATHS)
    missing = sorted(root_node_modules - runtime_paths)
    assert not missing, (
        "root node module(s) imported by __init__.py but not shipped by "
        f"dev-sync RUNTIME_PATHS: {missing}. Add them to RUNTIME_PATHS in "
        "scripts/sync_to_dev.py so a plain dev-sync can't 404 the node again."
    )
