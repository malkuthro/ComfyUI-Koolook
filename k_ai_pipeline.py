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
                "instruction": ("STRING", {
                    "default": "Place your base folder path in the FIELD below",
                    "multiline": False
                }),
                "base_directory_path": ("STRING", {
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
                "disable_versioning": ("BOOLEAN", {"default": False}),
                "enable_overwrite": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("WRITE_file_path", "output_name", "version_string", "output_directory", "shot_duration", "seed_value", "shot_name")
    FUNCTION = "generate_pipeline"
    CATEGORY = "Koolook/VFX"
    OUTPUT_NODE = True  # Marks it as an output node for workflow integration

    def generate_pipeline(self, shot_duration, seed_value, instruction, base_directory_path, extension, shot_name, ai_method, version, disable_versioning, enable_overwrite):
        # Generate version string like 'v001'
        version_str = f"v{version:03d}" if not disable_versioning else ""

        # Construct output path based on typical VFX structure: base_directory_path/shot_name/ai_method[/version_str]
        # Clean any double slashes for robustness
        parts = [base_directory_path.rstrip('/'), shot_name.lstrip('/'), ai_method.lstrip('/')]
        if not disable_versioning:
            parts.append(version_str.lstrip('/'))
        output_directory = os.path.join(*parts).replace('\\', '/')
        while '//' in output_directory:
            output_directory = output_directory.replace('//', '/')

        # Ensure output directory exists (create if missing) but don't treat an existing directory
        # as a reason to error — only an existing file with the same final path should block writing.
        if not os.path.exists(output_directory):
            try:
                os.makedirs(output_directory, exist_ok=True)
            except Exception:
                # If directory creation fails, leave the original path and let downstream handle it.
                pass

        # Construct full name: shot_name_ai_method[_version_str].extension
        name = f"{shot_name}_{ai_method}_{version_str}{extension}" if not disable_versioning else f"{shot_name}_{ai_method}{extension}"

        # Final file path: output_directory/name (with cleaning)
        file_path = os.path.join(output_directory, name).replace('\\', '/')
        while '//' in file_path:
            file_path = file_path.replace('//', '/')

        # Check for overwrite protection on the final file path only. This avoids raising an error
        # simply because the directory exists (useful when this node's output is connected to a loader),
        # but still prevents accidentally overwriting an existing file unless enable_overwrite is True.
        if not enable_overwrite and os.path.exists(file_path):
            raise ValueError(f"Output file already exists and overwrite is disabled. Enable 'enable_overwrite' or adjust parameters: {file_path}")

        # Return all for chaining in workflows (e.g., connect to savers or prompts)
        return (file_path, name, version_str, output_directory, shot_duration, seed_value, shot_name)

# Individual node mappings
NODE_CLASS_MAPPINGS = {
    "EasyAIPipeline": EasyAIPipeline
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EasyAIPipeline": "Easy AI Pipeline"
}