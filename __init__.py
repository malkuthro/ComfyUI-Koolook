# Import node classes from their respective files
from .k_easy_version import NODE_CLASS_MAPPINGS as version_mappings
from .k_easy_version import NODE_DISPLAY_NAME_MAPPINGS as version_display

from .k_easy_wan22_prompt import NODE_CLASS_MAPPINGS as wan_mappings
from .k_easy_wan22_prompt import NODE_DISPLAY_NAME_MAPPINGS as wan_display

from .k_easy_resize import NODE_CLASS_MAPPINGS as image_mappings
from .k_easy_resize import NODE_DISPLAY_NAME_MAPPINGS as image_display

from .k_ai_pipeline import NODE_CLASS_MAPPINGS as pipeline_mappings
from .k_ai_pipeline import NODE_DISPLAY_NAME_MAPPINGS as pipeline_display

# Merge all mappings
NODE_CLASS_MAPPINGS = {
    **version_mappings,
    **wan_mappings,
    **image_mappings,
    **pipeline_mappings
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **version_display,
    **wan_display,
    **image_display,
    **pipeline_display
}