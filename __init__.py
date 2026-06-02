"""ComfyUI-Koolook node registry.

Two layers of resilience guard the import:

1. Duplicate-install guard (#162) — if two Koolook folders exist in
   ``custom_nodes/``, only the alphabetical winner registers; the loser
   short-circuits before any heavy imports or route install.
2. Per-group registration (#198 / #183) — each node group is imported
   independently so a single missing or broken module degrades to "that
   node absent" instead of taking down the whole pack, and the Kforge Labs
   snapshot/preset routes install regardless of node-import results. The
   failure this guards against: a module that ``__init__`` imported but
   ``dev-sync`` never copied raised ``ImportError`` at startup, which zeroed
   every node *and* 404'd every snapshot route.
"""
from __future__ import annotations

import importlib
from pathlib import Path

# Duplicate-install guard (#162). Detection + winner selection live in the
# sibling ``koolook_install_guard`` module so the logic can be unit-tested
# from ``tests/test_install_guard.py`` without needing the relative-import
# parent-package context that ComfyUI's custom-node loader supplies (and
# pytest does not). See that module's docstring for the failure mode this
# guards against; the short version is: two Koolook folders in
# ``custom_nodes/`` race over ``/userdata/koolook_workflows.json`` and
# silently corrupt it.
#
# Import shape mirrors ``koolook_routes`` below — relative inside the
# package, absolute as a fallback so an out-of-package test harness can
# still load this file as a script.
try:
    from .koolook_install_guard import (  # noqa: E402
        build_duplicate_report,
        detect_duplicate_koolook_installs,
    )
except ImportError:  # pragma: no cover - exercised only outside package context
    from koolook_install_guard import (  # type: ignore[no-redef]
        build_duplicate_report,
        detect_duplicate_koolook_installs,
    )

NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}


def _merge_node_group(label: str, module_name: str) -> None:
    """Import one node group's mappings, isolating any import failure.

    A missing module, a broken import inside it, or an absent
    ``NODE_CLASS_MAPPINGS`` / ``NODE_DISPLAY_NAME_MAPPINGS`` attribute is
    caught and logged; the rest of the registry (and the route install
    below) is unaffected.
    """
    try:
        module = importlib.import_module(module_name, __package__)
        class_mappings = module.NODE_CLASS_MAPPINGS
        display_mappings = module.NODE_DISPLAY_NAME_MAPPINGS
    except (ImportError, AttributeError) as exc:
        print(f"[Koolook] {label} registry skipped: {exc}")
        return
    NODE_CLASS_MAPPINGS.update(class_mappings)
    NODE_DISPLAY_NAME_MAPPINGS.update(display_mappings)


_here = Path(__file__).resolve().parent
# The guard must never take down the plugin: any failure to scan or parse
# siblings degrades to "register normally" (fail-safe) instead of aborting
# the import (fail-silent — the plugin would vanish entirely, taking the
# alphabetical winner with it). detect_/read_ already swallow per-entry
# OSError/ValueError; this try is the final backstop for anything unforeseen.
_is_winning_install = True
try:
    _siblings = detect_duplicate_koolook_installs(_here)
    if _siblings:
        _is_winning_install, _report = build_duplicate_report(_here, _siblings)
        print(_report)
except Exception as _guard_err:  # pragma: no cover - defensive backstop
    print(
        f"[Koolook] install-guard check failed ({_guard_err!r}); "
        "registering normally."
    )
    _is_winning_install = True


if not _is_winning_install:
    # Non-winning duplicate (#162). Short-circuit BEFORE any of the heavy
    # node imports (torch, PIL, diffusers via the radiance/whatdreamscost
    # forks) or the aiohttp route install — the winner owns those for this
    # session. No WEB_DIRECTORY either, so the loser doesn't double-register
    # the sidebar web assets. Critical log already printed above.
    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
else:
    # Probe for a real package context. ComfyUI's custom-node loader provides
    # one; pytest's package collector and static-analysis tools import this
    # file as a standalone module, where relative imports raise. In that case
    # we register nothing (a single message) and let tooling import the
    # package directory without crashing. ``koolook_versioning`` is a
    # pure-stdlib helper, so the probe can only fail on the context, never on
    # its body.
    try:
        from . import koolook_versioning as _context_probe  # noqa: F401
    except ImportError as _context_exc:
        print(
            f"[Koolook] node registry skipped (non-Comfy import context): "
            f"{_context_exc}"
        )
        __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
    else:
        # Order is cosmetic (menu grouping); each group is independent.
        _merge_node_group("Wan 2.2 Easy Prompt", ".k_easy_wan22_prompt")
        _merge_node_group("Easy Resize", ".k_easy_resize")
        _merge_node_group("Easy AI Pipeline", ".k_ai_pipeline")
        _merge_node_group("Easy Image Batch", ".k_easy_image_batch")
        _merge_node_group("Koolook Camera Loader", ".k_easy_track")
        _merge_node_group("Easy Pattern", ".k_easy_pattern")
        _merge_node_group("Easy Utility", ".k_easy_utility")
        # The VHS-dependent video wrappers degrade to empty mappings when
        # VideoHelperSuite is absent (they catch the import internally); the
        # guard also covers a hard ImportError should one ever slip through.
        _merge_node_group("Easy Load Video", ".k_video_load")
        _merge_node_group("Easy Video Combine", ".k_video_combine")
        _merge_node_group("Radiance Koolook VAE", ".forks.radiance_koolook")
        _merge_node_group("WhatDreamsCost LTX Director", ".forks.whatdreamscost_koolook")

        # Register the Kforge Labs snapshot/preset endpoints. Installed
        # regardless of the node-import results above, so a broken node
        # module can no longer take the /koolook/presets/* routes down with
        # it. Both the import and the install() call are guarded: a missing
        # aiohttp (a ComfyUI hard-dep) or any other load error degrades to
        # "snapshot feature unavailable", never a crash.
        try:
            from . import koolook_routes  # noqa: E402

            if not koolook_routes.install():
                print(
                    "[Koolook] PromptServer unavailable at import time; preset routes "
                    "(/koolook/presets/*) not registered for this session."
                )
        except Exception as _routes_exc:  # pragma: no cover
            print(
                f"[Koolook] preset routes import failed; node mappings unaffected: "
                f"{_routes_exc}"
            )

        WEB_DIRECTORY = "./web"
        __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
