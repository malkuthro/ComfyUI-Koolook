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
                "seed_value": ("INT", {
                    "default": 453453453,
                    "min": 0,
                    "max": 999999999999,
                    "step": 1,
                    "display": "number"
                }),
                "job_path": ("STRING", {
                    "default": "n:/TRK_sync_BIG_NJOBS/baconx/Menneskehavn/outputs/VIDEO/ComfyUI/Wan2.1/Vortex/pComps",
                    "multiline": True
                }),
                "extension": ("STRING", {
                    "default": ".%04d.exr",
                    "multiline": False
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

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("file_path", "name", "version_string", "output_directory", "shot_duration", "seed_value", "shot_name", "ai_method")
    FUNCTION = "generate_pipeline"
    CATEGORY = "Koolook/VFX"
    OUTPUT_NODE = True

    def generate_pipeline(self, shot_duration, seed_value, job_path, extension, shot_name, ai_method, version):
        version_str = f"v{version:03d}"
        output_directory = os.path.join(job_path, shot_name, ai_method, version_str).replace("\\", "/")
        name = f"{shot_name}_{ai_method}_{version_str}{extension}"
        file_path = os.path.join(output_directory, name).replace("\\", "/")
        return (file_path, name, version_str, output_directory, shot_duration, seed_value, shot_name, ai_method)

NODE_CLASS_MAPPINGS = {
    "EasyAIPipeline": EasyAIPipeline
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EasyAIPipeline": "Easy AI Pipeline (VFX)"
}