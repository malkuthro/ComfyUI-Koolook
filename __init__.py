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
    # Non-winning duplicate (#162). Short-circuit BEFORE any of the
    # heavy node imports (torch, PIL, diffusers via the radiance/
    # whatdreamscost forks) or the aiohttp route install — the winner
    # owns those for this session and there's no point paying their
    # import cost just to discard the mappings afterward. Critical log
    # was already printed by ``build_duplicate_report`` above.
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
else:
    try:
        from .k_easy_wan22_prompt import NODE_CLASS_MAPPINGS as wan_mappings
        from .k_easy_wan22_prompt import NODE_DISPLAY_NAME_MAPPINGS as wan_display

        from .k_easy_resize import NODE_CLASS_MAPPINGS as resize_mappings
        from .k_easy_resize import NODE_DISPLAY_NAME_MAPPINGS as resize_display

        from .k_ai_pipeline import NODE_CLASS_MAPPINGS as pipeline_mappings
        from .k_ai_pipeline import NODE_DISPLAY_NAME_MAPPINGS as pipeline_display

        from .k_easy_image_batch import NODE_CLASS_MAPPINGS as batch_mappings  # Added for new VFX batch node
        from .k_easy_image_batch import NODE_DISPLAY_NAME_MAPPINGS as batch_display  # Added for new VFX batch node

        from .k_easy_track import NODE_CLASS_MAPPINGS as cam_loader_mappings
        from .k_easy_track import NODE_DISPLAY_NAME_MAPPINGS as cam_loader_display

        from .k_easy_pattern import NODE_CLASS_MAPPINGS as pattern_mappings
        from .k_easy_pattern import NODE_DISPLAY_NAME_MAPPINGS as pattern_display

        from .k_easy_utility import NODE_CLASS_MAPPINGS as utility_mappings
        from .k_easy_utility import NODE_DISPLAY_NAME_MAPPINGS as utility_display

        from .k_audio_timeline import NODE_CLASS_MAPPINGS as audio_timeline_mappings
        from .k_audio_timeline import NODE_DISPLAY_NAME_MAPPINGS as audio_timeline_display

        from .k_timeline_editor import NODE_CLASS_MAPPINGS as timeline_editor_mappings
        from .k_timeline_editor import NODE_DISPLAY_NAME_MAPPINGS as timeline_editor_display

        from .forks.radiance_koolook import NODE_CLASS_MAPPINGS as radiance_koolook_mappings
        from .forks.radiance_koolook import NODE_DISPLAY_NAME_MAPPINGS as radiance_koolook_display

        from .forks.whatdreamscost_koolook import NODE_CLASS_MAPPINGS as whatdreamscost_koolook_mappings
        from .forks.whatdreamscost_koolook import NODE_DISPLAY_NAME_MAPPINGS as whatdreamscost_koolook_display
    except ImportError as _node_import_exc:
        # The relative imports above require a parent-package context that
        # only ComfyUI's custom-node loader provides. Pytest's package
        # collector (which walks up from test files looking for ``__init__.py``)
        # and static-analysis tools load this file as a standalone module,
        # where the imports raise. Falling back to empty registries lets
        # tooling import the package directory without crashing — tests
        # access sibling modules (e.g. ``koolook_routes``) via absolute
        # imports that don't need the package context.
        print(
            f"[Koolook] node registry skipped (non-Comfy import context): "
            f"{_node_import_exc}"
        )
        NODE_CLASS_MAPPINGS = {}
        NODE_DISPLAY_NAME_MAPPINGS = {}
        __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
    else:
        # VHS-dependent wrappers self-skip when VideoHelperSuite is absent, and
        # these guards keep a local import surprise from hiding the core nodes.
        video_load_mappings = {}
        video_load_display = {}
        try:
            from .k_video_load import NODE_CLASS_MAPPINGS as video_load_mappings
            from .k_video_load import NODE_DISPLAY_NAME_MAPPINGS as video_load_display
        except ImportError as _exc:
            print(f"[Koolook] Easy_LoadVideo registry skipped: {_exc}")

        video_combine_mappings = {}
        video_combine_display = {}
        try:
            from .k_video_combine import NODE_CLASS_MAPPINGS as video_combine_mappings
            from .k_video_combine import NODE_DISPLAY_NAME_MAPPINGS as video_combine_display
        except ImportError as _exc:
            print(f"[Koolook] Easy_VideoCombine registry skipped: {_exc}")

        # Register the Kforge Labs snapshot/preset endpoints. Failure here is
        # non-fatal — the node mappings still load and the rest of the plugin
        # works; only the snapshot feature in the sidebar is unavailable. Both
        # the import itself AND the `install()` call are guarded: a missing
        # `aiohttp` (extremely unusual since it's a ComfyUI hard-dep) or any
        # other module-load exception falls through to the print without
        # crashing the node registry.
        try:
            from . import koolook_routes  # noqa: E402

            if not koolook_routes.install():
                print(
                    "[Koolook] PromptServer unavailable at import time; preset routes "
                    "(/koolook/presets/*) not registered for this session."
                )
        except Exception as _exc:  # pragma: no cover
            print(f"[Koolook] preset routes import failed; node mappings unaffected: {_exc}")

        NODE_CLASS_MAPPINGS = {
            **wan_mappings,
            **resize_mappings,
            **pipeline_mappings,
            **batch_mappings,
            **cam_loader_mappings,
            **pattern_mappings,
            **utility_mappings,
            **audio_timeline_mappings,
            **timeline_editor_mappings,
            **video_load_mappings,
            **video_combine_mappings,
            **radiance_koolook_mappings,
            **whatdreamscost_koolook_mappings,
        }

        NODE_DISPLAY_NAME_MAPPINGS = {
            **wan_display,
            **resize_display,
            **pipeline_display,
            **batch_display,
            **cam_loader_display,
            **pattern_display,
            **utility_display,
            **audio_timeline_display,
            **timeline_editor_display,
            **video_load_display,
            **video_combine_display,
            **radiance_koolook_display,
            **whatdreamscost_koolook_display,
        }

        WEB_DIRECTORY = "./web"
        __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
