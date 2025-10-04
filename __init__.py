# Import node classes from their respective files
from .k_easy_version import Easy_Version
from .k_easy_wan22_prompt import EasyWan22Prompt
from .k_easy_resize import EasyResize
from .k_ai_pipeline import EasyAIPipeline

# Aggregate mappings from each node file
# Assuming each node file defines its own NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS

from .k_easy_version import NODE_CLASS_MAPPINGS as version_mappings
from .k_easy_version import NODE_DISPLAY_NAME_MAPPINGS as version_display

from .k_easy_wan22_prompt import NODE_CLASS_MAPPINGS as wan_mappings
from .k_easy_wan22_prompt import NODE_DISPLAY_NAME_MAPPINGS as wan_display

from .k_easy_resize import NODE_CLASS_MAPPINGS as resize_mappings
from .k_easy_resize import NODE_DISPLAY_NAME_MAPPINGS as resize_display

from .k_ai_pipeline import NODE_CLASS_MAPPINGS as pipeline_mappings
from .k_ai_pipeline import NODE_DISPLAY_NAME_MAPPINGS as pipeline_display

# Merge all mappings
NODE_CLASS_MAPPINGS = {
    **version_mappings,
    **wan_mappings,
    **resize_mappings,
    **pipeline_mappings
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **version_display,
    **wan_display,
    **resize_display,
    **pipeline_display
}