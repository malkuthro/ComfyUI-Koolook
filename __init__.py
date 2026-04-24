from .k_easy_version import NODE_CLASS_MAPPINGS as version_mappings
from .k_easy_version import NODE_DISPLAY_NAME_MAPPINGS as version_display

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

NODE_CLASS_MAPPINGS = {
    **version_mappings,
    **wan_mappings,
    **resize_mappings,
    **pipeline_mappings,
    **batch_mappings,
    **cam_loader_mappings,
    **radiance_koolook_mappings,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **version_display,
    **wan_display,
    **resize_display,
    **pipeline_display,
    **batch_display,
    **cam_loader_display,
    **radiance_koolook_display,
}

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]