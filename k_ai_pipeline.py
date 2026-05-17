import os


def _normalize_base_path(raw: str) -> str:
    """Sanitize a user-pasted base directory string.

    Handles the common copy-paste variations that show up in the
    ``base_directory_path`` widget:

    - Surrounding whitespace (sloppy paste).
    - Surrounding quotes (Windows Explorer's ``Shift+Right-click → Copy as
      path`` wraps the result in double quotes; users occasionally wrap
      single quotes manually).
    - Trailing path separators (Explorer's address bar ends paths in ``\\``;
      Linux/Mac users habitually append ``/``).

    The trailing-separator case is the load-bearing one: without this, a
    base path like ``n:\\foo\\bar\\`` leaks an empty trailing component
    through ``os.path.join``, and the resulting trailing ``/`` on
    ``output_directory`` causes some downstream save nodes to split on
    ``/``, stringify the empty tail as ``"undefined"``, and write into
    ``…/undefined/…`` on disk.

    Drive roots (``C:\\``, ``n:/``) are preserved — stripping the trailing
    separator there would yield an invalid bare-drive reference like ``C:``
    that Windows interprets as "current directory on C: drive".
    """
    s = raw.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        s = s[1:-1].strip()
    while len(s) > 3 and s[-1] in ('/', '\\'):
        s = s[:-1]
    return s


def _sanitize_segment(s: str) -> str:
    """Strip drive prefix and any leading path separators from a path segment.

    Path segments like ``shot_name`` and ``ai_method`` are joined ONTO
    ``base_directory_path`` to build the output location — never meant to
    replace it. But ``os.path.join`` has a "last absolute path wins" rule:
    ``os.path.join('n:/safe', '/oops')`` returns ``'/oops'`` (or
    ``'C:/oops'`` on Windows), letting a user with a typo escape their
    own intended base. This helper strips:

    - Drive prefix (``os.path.splitdrive`` handles ``C:/foo`` → ``/foo``
      on Windows; on POSIX the drive is empty and this is a no-op).
    - Leading path separators (any mix of ``/`` and ``\\``).

    Examples:
    - ``"shot_v1"``         → ``"shot_v1"`` (unchanged)
    - ``"/oops_typo"``      → ``"oops_typo"``
    - ``"\\oops"``          → ``"oops"``
    - ``"C:/Windows/junk"`` → ``"Windows/junk"`` (drive + leading sep stripped)
    - ``"///oops"``         → ``"oops"`` (leading seps stripped before splitdrive
      so multi-slash input doesn't get swallowed as a UNC prefix on Windows)
    """
    s = s.lstrip("/\\")
    _, s = os.path.splitdrive(s)
    return s.lstrip("/\\")


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
                    "display": "number",
                    "tooltip": "Number of frames in the shot. Pass-through INT output — wire into samplers, video savers, or anything that needs a frame count.",
                }),
                "seed_value": ("INT", {
                    "default": 453453453,
                    "min": 0,
                    "max": 999999999999,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Master seed shared across the workflow. Pass-through INT output — wire into samplers so a single value controls everything downstream.",
                }),
                "instruction": ("STRING", {
                    "default": "Place your base folder path in the FIELD below",
                    "multiline": False,
                    "tooltip": "Read-only hint. The actual base folder goes in the multi-line text field below.",
                }),
                "base_directory_path": ("STRING", {
                    "default": "place your desired output dir here",
                    "multiline": True,
                    "tooltip": "Root folder for outputs. By default the node appends shot_name/ai_method[/vNNN] underneath; toggle no_subfolders to write directly into this folder instead. Pasted paths are auto-cleaned: surrounding whitespace, surrounding quotes (\"...\" from Explorer's Copy as path), and trailing slashes / backslashes are all stripped — n:\\foo\\bar\\, \"n:/foo/bar\", and n:\\foo\\bar all resolve identically.",
                }),
                "extension": ("STRING", {
                    "default": ".%04d.exr",
                    "multiline": False,
                    "tooltip": "Filename extension, appended verbatim. Use %04d for sequence numbering (e.g. .%04d.exr writes frame_0001.exr, frame_0002.exr, ...).",
                }),
                # Output section
                "shot_name": ("STRING", {
                    "default": "oceanTop_writeTest",
                    "multiline": False,
                    "tooltip": "Logical shot identifier. Used as the first segment of the filename and (unless no_subfolders is on) as a subfolder under the base path.",
                }),
                "ai_method": ("STRING", {
                    "default": "v2v",
                    "multiline": False,
                    "tooltip": "Free-form method tag (e.g. v2v, upscale, denoise). When set, joined into the filename with underscores and added as a subfolder. Leave blank to skip — no dangling _ in the filename, no empty subfolder.",
                }),
                "version": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 999,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Integer version, formatted as vNNN in the path/filename. Ignored when disable_versioning is on.",
                }),
                "disable_versioning": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Drop the vNNN segment from both the output directory and the filename.",
                }),
                "enable_overwrite": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Allow writing over an existing file at the final path. When off, the node raises an error if the target file already exists (the directory existing is fine).",
                }),
                "no_subfolders": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Write directly into base_directory_path — do NOT append shot_name/ai_method[/vNNN] subfolders. The base folder itself is still created on the fly if missing.",
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("WRITE_file_path", "output_name", "version_string", "output_directory", "shot_duration", "seed_value", "shot_name")
    FUNCTION = "generate_pipeline"
    CATEGORY = "Koolook/Pipeline"
    OUTPUT_NODE = True  # Marks it as an output node for workflow integration

    def generate_pipeline(self, shot_duration, seed_value, instruction, base_directory_path, extension, shot_name, ai_method, version, disable_versioning, enable_overwrite, no_subfolders):
        base_directory_path = _normalize_base_path(base_directory_path)
        version_str = "" if disable_versioning else f"v{version:03d}"

        # Sanitize the segments that get joined onto base_directory_path. Without this an
        # absolute-looking shot_name like "/oops" or "C:/Windows" would escape the base via
        # os.path.join's 'last absolute path wins' rule. The raw values are preserved for
        # the return tuple (downstream nodes that use shot_name as a label, not a path).
        shot_name_seg = _sanitize_segment(shot_name)
        ai_method_seg = _sanitize_segment(ai_method)

        # Build output directory. When no_subfolders is on, write straight into base_directory_path;
        # otherwise append shot_name/ai_method[/vNNN]. os.path.join drops empty segments, so a blank
        # ai_method or disabled versioning doesn't leave a phantom slash in the path.
        if no_subfolders:
            output_directory = base_directory_path
        else:
            output_directory = os.path.join(base_directory_path, shot_name_seg, ai_method_seg, version_str)
        output_directory = output_directory.replace('\\', '/')
        while '//' in output_directory:
            output_directory = output_directory.replace('//', '/')
        # Strip trailing slash. Without this, a base path ending in '\' (e.g. 'n:\foo\bar\')
        # propagates through os.path.join as an empty trailing component → a trailing '/' on
        # the final output_directory. Downstream save nodes that split on '/' then see an
        # empty tail and stringify it as 'undefined', creating a phantom folder on disk.
        # Guarded against drive roots like 'n:/' which need the slash to stay valid.
        if len(output_directory) > 3:
            output_directory = output_directory.rstrip('/')

        # Create the output directory if missing — applies to both modes, so a user can point
        # no_subfolders at a not-yet-existing base folder and have it created on the fly.
        if not os.path.exists(output_directory):
            try:
                os.makedirs(output_directory, exist_ok=True)
            except Exception:
                pass

        # Filename: join only the non-empty pieces with '_', then append extension. Empty ai_method
        # or disabled versioning naturally drop out — no trailing/dangling underscores. Use the
        # sanitized segments so a stray leading separator can't leak into the filename either.
        name_parts = [p for p in (shot_name_seg, ai_method_seg, version_str) if p]
        name = "_".join(name_parts) + extension

        file_path = os.path.join(output_directory, name).replace('\\', '/')
        while '//' in file_path:
            file_path = file_path.replace('//', '/')

        # Overwrite protection applies to the final file path only — an existing output directory
        # is fine (common when the node is wired into a recurring save loop).
        if not enable_overwrite and os.path.exists(file_path):
            raise ValueError(f"Output file already exists and overwrite is disabled. Enable 'enable_overwrite' or adjust parameters: {file_path}")

        return (file_path, name, version_str, output_directory, shot_duration, seed_value, shot_name)

# Individual node mappings
NODE_CLASS_MAPPINGS = {
    "EasyAIPipeline": EasyAIPipeline
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EasyAIPipeline": "Easy AI Pipeline (Koolook)"
}