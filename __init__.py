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

from .forks.radiance_koolook import NODE_CLASS_MAPPINGS as radiance_koolook_mappings
from .forks.radiance_koolook import NODE_DISPLAY_NAME_MAPPINGS as radiance_koolook_display

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
    **radiance_koolook_mappings,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **wan_display,
    **resize_display,
    **pipeline_display,
    **batch_display,
    **cam_loader_display,
    **radiance_koolook_display,
}

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]