import os

class EasyAIPipeline:
    """
    A custom ComfyUI node for VFX pipelines that aggregates parameters like shot duration, seed, job path,
    shot name, AI method, and version. It generates a formatted version string and a complete output path
    for output saving, useful in structured VFX workflows (e.g., rendering or compositing sequences).
    Inspired by typical VFX directory structures for organized asset management.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Settings section
                "shot_duration": ("INT", {
                    "default": 81,
                    "min": 1,
                    "max": 10000,
                    "step": 1,
                    "display": "number"
                }),
                "seed": ("INT", {
                    "default": 453453453,
                    "min": 0,
                    "max": 999999999999,
                    "step": 1,
                    "display": "number"
                }),
                "job_path": ("STRING", {
                    "default": "N:/TRK_sync_BIG_NJOBS/baconx/Menneskehavn/outputs/VIDEO/ComfyUI/Wan2.2/Vortex/pComps",
                    "multiline": True
                }),
                # Output section
                "shot_name": ("STRING", {
                    "default": "oceanTop_writeTest",
                    "multiline": False
                }),
                "ai_method": ("STRING", {
                    "default": "v2v",
                    "multiline": False
                }),
                "version": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 999,
                    "step": 1,
                    "display": "number"
                }),
            }
        }

    # To customize output names, edit the strings in this tuple (e.g., "output_directory" to "output_xxxxx").
    # To change the order from top to bottom in the UI, reorder the elements here.
    # Ensure RETURN_TYPES and the return statement in generate_pipeline match the order.

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("version", "output_directory", "shot_duration", "seed", "shot_name", "ai_method")
    FUNCTION = "generate_pipeline"
    CATEGORY = "Koolook/VFX"
    OUTPUT_NODE = True  # Marks it as an output node for workflow integration

    def generate_pipeline(self, shot_duration, seed, job_path, shot_name, ai_method, version):
        # Generate version string like 'v001'
        version_str = f"v{version:03d}"

        # Construct output path based on typical VFX structure: job_path/shot_name/ai_method/version_str
        # Adjust as needed for specific pipeline conventions
        output_directory = os.path.join(job_path, shot_name, ai_method, version_str).replace("\\", "/")  # Normalize to forward slashes

        # Return all for chaining in workflows (e.g., connect to savers or prompts)
        return ( version_str, output_directory, shot_duration, seed, shot_name, ai_method)

# Individual node mappings
NODE_CLASS_MAPPINGS = {
    "EasyAIPipeline": EasyAIPipeline
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EasyAIPipeline": "Easy AI Pipeline (VFX)"
}