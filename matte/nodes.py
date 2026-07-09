"""
Koolook Matte — ComfyUI nodes for mask-guided one-step video matting.

Part of the Koolook pack (see matte/NOTICE for the model-weight licenses). It pairs an independent
implementation of the published VideoMaMa method (pipeline.py) with:
  * temporal windowing (peak VRAM scales with the window, not the clip length),
  * subject-focused ROI cropping + paste-back (process only the subject, native res),
  * shortest-side / longest-axis resolution control, /64-aligned, with auto OOM
    resolution-backoff that preserves the temporal window (no new flicker).

It loads the VideoMaMa UNet + SVD VAE *weights* at runtime; those weights carry
their own (non-commercial) licenses and are the user's responsibility.
"""

import os
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from comfy.utils import ProgressBar

from .pipeline import KoolookMattePipeline


def _ensure_local_model(path: str, repo_id: str, label: str) -> str:
    """Return a usable local model dir. If `path` is missing/empty, try to fetch
    `repo_id` from the Hugging Face Hub (optional dep). Weights are never bundled."""
    if os.path.isdir(path) and os.listdir(path):
        return path
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        raise RuntimeError(
            f"{label} not found at {path} and huggingface_hub isn't installed. "
            f"Place the model there, or `pip install huggingface_hub`."
        )
    print(f"[KoolookMatte] downloading {label} ({repo_id}) -> {path}")
    os.makedirs(path, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=path)
    return path


class KoolookMatteLoader:
    """
    Loads the VideoMaMa pipeline with base model and fine-tuned UNet checkpoint.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_model_path": ("STRING", {
                    "default": "checkpoints/stabilityai/stable-video-diffusion-img2vid-xt",
                    "multiline": False
                }),
                "unet_checkpoint_path": ("STRING", {
                    "default": "checkpoints/VideoMaMa",
                    "multiline": False
                }),
                "precision": (["fp16", "bf16"], {
                    "default": "fp16"
                }),
                "enable_model_cpu_offload": ("BOOLEAN", {
                    "default": True
                }),
                "vae_encode_chunk_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 25,
                    "step": 1,
                    "tooltip": "Frames the VAE encodes/decodes at once. 1 = desktop (Sammie) default and lowest VRAM; raising it makes the VAE decode spike proportionally bigger and is the usual cause of OOM at high resolution / 4K. Leave at 1 unless you have VRAM to spare and want a small speedup."
                }),
                "attention_mode": (["auto", "xformers", "sdpa", "none"], {
                    "default": "auto"
                }),
                "enable_vae_tiling": ("BOOLEAN", {
                    "default": False
                }),
                "enable_vae_slicing": ("BOOLEAN", {
                    "default": True
                }),
            }
        }

    RETURN_TYPES = ("KOOLOOK_MATTE_PIPELINE",)
    FUNCTION = "load_pipeline"
    CATEGORY = "Koolook Matte"

    def load_pipeline(
        self,
        base_model_path: str,
        unet_checkpoint_path: str,
        precision: str,
        enable_model_cpu_offload: bool,
        vae_encode_chunk_size: int,
        attention_mode: str,
        enable_vae_tiling: bool,
        enable_vae_slicing: bool
    ):
        weight_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        node_dir = Path(__file__).parent
        # relative paths resolve next to this pack; absolute paths pass through
        base = base_model_path if os.path.isabs(base_model_path) else str(node_dir / base_model_path)
        unet = unet_checkpoint_path if os.path.isabs(unet_checkpoint_path) else str(node_dir / unet_checkpoint_path)

        base = _ensure_local_model(base, "stabilityai/stable-video-diffusion-img2vid-xt", "SVD base model")
        unet = _ensure_local_model(unet, "SammyLim/VideoMaMa", "VideoMaMa UNet")

        print(f"[KoolookMatte] loading pipeline | base={base} unet={unet} | {precision} "
              f"offload={enable_model_cpu_offload} vae_chunk={vae_encode_chunk_size} attn={attention_mode}")
        try:
            pipeline = KoolookMattePipeline(
                base_model_path=base,
                unet_checkpoint_path=unet,
                weight_dtype=weight_dtype,
                device="cuda" if torch.cuda.is_available() else "cpu",
                enable_model_cpu_offload=enable_model_cpu_offload,
                vae_encode_chunk_size=vae_encode_chunk_size,
                attention_mode=attention_mode,
                enable_vae_tiling=enable_vae_tiling,
                enable_vae_slicing=enable_vae_slicing,
            )
            print(f"[KoolookMatte] pipeline loaded ({precision})")
            return (pipeline,)
        except Exception as e:
            raise RuntimeError(f"Failed to load Koolook Matte pipeline: {e}")


def _ensure_2d_mask(mask_np: np.ndarray) -> np.ndarray:
    """Ensure mask array is 2D (H, W) for PIL grayscale mode.

    Some mask sources (e.g. MatAnyone) output masks with extra dimensions
    like [C, H, W] or [H, W, C] instead of the standard ComfyUI [H, W].
    This helper squeezes/removes those extra dimensions.
    """
    if mask_np.ndim == 2:
        return mask_np
    if mask_np.ndim == 3:
        # [1, H, W] -> [H, W]  or  [H, W, 1] -> [H, W]
        if mask_np.shape[0] == 1:
            return mask_np[0]
        if mask_np.shape[-1] == 1:
            return mask_np[:, :, 0]
    # General fallback: squeeze all singleton dimensions
    mask_np = np.squeeze(mask_np)
    if mask_np.ndim == 2:
        return mask_np
    # If still not 2D, take first slice along leading dims until 2D
    while mask_np.ndim > 2:
        mask_np = mask_np[0]
    return mask_np


class KoolookMatteSampler:
    """
    Runs one-step mask-guided matting on video frames with mask conditioning.
    Expects ComfyUI IMAGE format: [B, H, W, C] tensors with values in [0, 1].
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("KOOLOOK_MATTE_PIPELINE",),
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
                "max_resolution": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 2048,
                    "step": 8
                }),
                "fps": ("INT", {
                    "default": 7,
                    "min": 1,
                    "max": 60
                }),
                "motion_bucket_id": ("INT", {
                    "default": 127,
                    "min": 1,
                    "max": 255
                }),
                "noise_aug_strength": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "window_size": ("INT", {
                    "default": 16,
                    "min": 0,
                    "max": 128,
                    "step": 1,
                    "tooltip": "Frames sent through the UNet per pass. Lower = far less VRAM, so you can raise max_resolution (e.g. 1440) on 24GB. 0 = process the whole clip in one pass (legacy behaviour)."
                }),
                "window_overlap": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 32,
                    "step": 1,
                    "tooltip": "Frames re-processed and blended at each window boundary for temporal continuity. 0 disables overlap feedback/blending. Must be < window_size."
                }),
                "crop_to_mask": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Process only the union bounding box of the masks (Sammie-style ROI). Far fewer pixels through the UNet => much faster and lets max_resolution apply to just the subject. The output matte is pasted back into the full frame (zero alpha outside)."
                }),
                "crop_padding": ("INT", {
                    "default": 16,
                    "min": 0,
                    "max": 512,
                    "step": 8,
                    "tooltip": "Pixels of margin added around the mask bounding box before cropping, so soft edges/motion aren't clipped."
                }),
                "resize_mode": (["shortest_side", "longest_axis"], {
                    "default": "shortest_side",
                    "tooltip": "shortest_side = desktop/Sammie parity: cap the shorter side at max_resolution, never upscale, keep the long axis at full detail (sharpest matte, more VRAM). longest_axis = force the longer side to max_resolution (lower VRAM, softer edges)."
                }),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "run_inference"
    CATEGORY = "Koolook Matte"

    # SVD's UNet downsamples the latent 8x, so pixel dims should divide by 64.
    # Anything only /8 gets padded/cropped inside the UNet bottleneck, which softens
    # edges — so we align the model working size to 64.
    ALIGN = 64

    @classmethod
    def _compute_target_size(cls, width: int, height: int, max_resolution: int, mode: str = "shortest_side"):
        """Compute the model working size, aligned to a multiple of 64 (SVD-friendly).

        mode="shortest_side" (desktop / Sammie parity): cap the SHORTER side at
        max_resolution and NEVER upscale. A crop smaller than max_resolution is
        processed at native pixels, so the long axis keeps full detail — this is
        what makes the desktop app's edges sharp. Uses more VRAM than longest_axis
        for the same number (the long side is unbounded).

        mode="longest_axis": force the LONGER side to exactly max_resolution
        (upscaling small crops). Lower VRAM, but downscales the detail axis => softer.
        """
        a = cls.ALIGN
        if max_resolution <= 0:
            return max((width // a) * a, a), max((height // a) * a, a)
        if mode == "shortest_side":
            min_side = min(width, height)
            scale = (max_resolution / min_side) if min_side > max_resolution else 1.0
            new_width = int(width * scale)
            new_height = int(height * scale)
        else:  # longest_axis
            if width >= height:
                new_width = max_resolution
                new_height = int(height * max_resolution / width)
            else:
                new_height = max_resolution
                new_width = int(width * max_resolution / height)
        # Align to multiple of 64 (SVD UNet requirement — avoids internal padding softness)
        new_width = max((new_width // a) * a, a)
        new_height = max((new_height // a) * a, a)
        return new_width, new_height

    @staticmethod
    def _generate_windows(total_frames: int, window_size: int, overlap: int):
        """Generate sliding-window (start, end) pairs covering total_frames (end exclusive).

        window_size <= 0 or >= total_frames processes the whole clip in one pass.
        overlap that is >= window_size is clamped so the window always advances.
        Standard sliding-window scheme (independent implementation).
        """
        if window_size <= 0 or window_size >= total_frames:
            return [(0, total_frames)]

        step = window_size - overlap
        if step <= 0:
            print(f"[KoolookMatte] window_overlap ({overlap}) >= window_size ({window_size}); "
                  f"clamping so windows advance by 1 frame")
            step = 1

        windows = []
        pos = 0
        while pos < total_frames:
            end = min(pos + window_size, total_frames)
            windows.append((pos, end))
            if end >= total_frames:
                break
            pos += step
        return windows

    @staticmethod
    def _to_gray01(frame_pil) -> np.ndarray:
        """Convert a pipeline output frame (PIL, RGB or L) to float32 grayscale in [0, 1]."""
        arr = np.array(frame_pil).astype(np.float32) / 255.0
        if arr.ndim == 3:
            arr = arr.mean(axis=-1)
        return arr

    @staticmethod
    def _compute_union_bbox(masks_full, padding: int, orig_w: int, orig_h: int, threshold: int = 127):
        """Union bounding box (x1, y1, x2, y2; x2/y2 exclusive) over every mask in the
        clip, padded and clamped to the frame. `masks_full` is a list of 2D uint8
        arrays already resampled to the image resolution (orig_h, orig_w), so bbox,
        image crop and mask crop all share one coordinate space. Returns None if no
        mask is present (caller then processes the full frame).

        Mirrors Sammie-Roto-2's compute_mask_bounding_box: one rect for the entire
        range, so every window/frame uses an identical spatial region.
        """
        union = None
        for a in masks_full:
            b = a > threshold
            union = b if union is None else (union | b)
        if union is None or not union.any():
            return None
        rows = np.where(union.any(axis=1))[0]
        cols = np.where(union.any(axis=0))[0]
        y1, y2 = int(rows[0]), int(rows[-1]) + 1
        x1, x2 = int(cols[0]), int(cols[-1]) + 1
        # Pad and clamp to frame
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(orig_w, x2 + padding)
        y2 = min(orig_h, y2 + padding)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None            # degenerate region; process full frame
        return (x1, y1, x2, y2)

    def run_inference(
        self,
        pipeline,
        images,
        masks,
        seed: int,
        max_resolution: int,
        fps: int,
        motion_bucket_id: int,
        noise_aug_strength: float,
        window_size: int = 0,
        window_overlap: int = 0,
        crop_to_mask: bool = False,
        crop_padding: int = 16,
        resize_mode: str = "shortest_side"
    ):
        """Run VideoMaMa inference, optionally split into temporal windows to cap VRAM.

        With window_size > 0 and < num_frames the clip is processed in sliding
        windows of `window_size` frames. The trailing `window_overlap` soft-alpha
        outputs of each window are fed back as the leading masks of the next window
        (temporal continuity), and the overlap region is linearly blended across the
        seam. Peak VRAM then scales with window_size instead of the full clip length,
        which is what lets max_resolution reach 1440 on a 24GB card. window_size=0
        (or >= num_frames) reproduces the original single-pass behaviour exactly.
        """
        import time
        num_frames = images.shape[0]

        if masks.shape[0] != num_frames:
            raise ValueError(
                f"Number of image frames ({num_frames}) must match "
                f"number of mask frames ({masks.shape[0]})"
            )

        orig_h, orig_w = images.shape[1], images.shape[2]

        # --- Evaluation instrumentation: wall-clock + peak VRAM ---
        t_start = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Bring every mask up to the IMAGE resolution first (nearest — it's only a
        # coarse guide). This lets you feed the node the FULL-RES plate while the mask
        # keeps coming from the cheap downscaled SAM2/RMBG path: image, bbox and mask
        # crop then all share one coordinate space, so no detail is thrown away before
        # the model sees it (this is how the desktop app stays crisp).
        masks_full = []
        for i in range(num_frames):
            a = _ensure_2d_mask((masks[i].cpu().numpy() * 255).astype(np.uint8))
            if a.shape[:2] != (orig_h, orig_w):
                a = np.array(Image.fromarray(a, mode='L').resize((orig_w, orig_h), Image.NEAREST))
            masks_full.append(a)

        # --- Optional ROI crop: process only the subject's union bounding box ---
        crop_rect = self._compute_union_bbox(masks_full, crop_padding, orig_w, orig_h) if crop_to_mask else None
        if crop_rect is not None:
            cx1, cy1, cx2, cy2 = crop_rect
            src_w, src_h = cx2 - cx1, cy2 - cy1
        else:
            cx1, cy1, cx2, cy2 = 0, 0, orig_w, orig_h
            src_w, src_h = orig_w, orig_h

        # Model working size for the processed region (see _compute_target_size).
        target_w, target_h = self._compute_target_size(src_w, src_h, max_resolution, resize_mode)

        # Build the model input frames at a given working size. Called once up front,
        # and again smaller if we have to back off on a CUDA OOM. Crops to the ROI here;
        # masks are binarized (crisp guide) and resized nearest.
        def _build_frames(tw, th):
            cond, msk = [], []
            for i in range(num_frames):
                img_np = (images[i].cpu().numpy() * 255).astype(np.uint8)
                if crop_rect is not None:
                    img_np = img_np[cy1:cy2, cx1:cx2]
                cond.append(Image.fromarray(img_np, mode='RGB').resize((tw, th), Image.LANCZOS))

                m = masks_full[i]
                if crop_rect is not None:
                    m = m[cy1:cy2, cx1:cx2]
                m = np.where(m > 127, np.uint8(255), np.uint8(0))
                msk.append(Image.fromarray(m, mode='L').resize((tw, th), Image.NEAREST))
            return cond, msk

        try:
            # Run the windowed inference. On a CUDA OOM we back off on RESOLUTION, NOT
            # window_size — the temporal window stays full so mattes don't start
            # flickering. A sharp lower-res matte beats a temporally-inconsistent one.
            oom_types = (torch.cuda.OutOfMemoryError,) if hasattr(torch.cuda, "OutOfMemoryError") else (RuntimeError,)
            min_side_floor = 256
            while True:
                cond_frames, mask_frames = _build_frames(target_w, target_h)
                print(f"Input {orig_w}x{orig_h} | region {src_w}x{src_h}"
                      f"{' (cropped to mask)' if crop_rect else ''} -> target {target_w}x{target_h} "
                      f"(max_resolution={max_resolution}, {resize_mode})")
                try:
                    output_accum = self._run_windows(
                        pipeline, cond_frames, mask_frames, num_frames,
                        window_size, window_overlap, seed, fps,
                        motion_bucket_id, noise_aug_strength
                    )
                    break
                except oom_types as oom:
                    if "out of memory" not in str(oom).lower():
                        raise  # a genuine (non-OOM) error — don't mask or retry it
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if min(target_w, target_h) <= min_side_floor:
                        raise
                    new_w = max((int(target_w * 0.85) // self.ALIGN) * self.ALIGN, self.ALIGN)
                    new_h = max((int(target_h * 0.85) // self.ALIGN) * self.ALIGN, self.ALIGN)
                    print(f"[KoolookMatte] CUDA OOM at {target_w}x{target_h}; cleared cache, "
                          f"backing off resolution to {new_w}x{new_h} (window_size kept at "
                          f"{window_size} for temporal stability) and retrying.")
                    target_w, target_h = new_w, new_h

            # Resize each matte back to the region size and paste into the full
            # frame canvas (zero alpha outside the crop), then stack -> ComfyUI MASK
            output_masks = []
            for idx in range(num_frames):
                g = output_accum[idx]
                if g is None:
                    g = np.zeros((target_h, target_w), dtype=np.float32)
                g_pil = Image.fromarray((g * 255.0).clip(0, 255).astype(np.uint8), mode='L')
                g_pil = g_pil.resize((src_w, src_h), Image.LANCZOS)
                region = np.array(g_pil).astype(np.float32) / 255.0
                if crop_rect is not None:
                    full = np.zeros((orig_h, orig_w), dtype=np.float32)
                    full[cy1:cy2, cx1:cx2] = region
                    output_masks.append(full)
                else:
                    output_masks.append(region)

            output_tensor = torch.from_numpy(np.stack(output_masks, axis=0))

            elapsed = time.time() - t_start
            per_frame = elapsed / max(num_frames, 1)
            vram_str = ""
            if torch.cuda.is_available():
                vram_str = f", peak VRAM {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB"
            print(f"KoolookMatte done: {output_tensor.shape} in {elapsed:.1f}s "
                  f"({per_frame:.2f}s/frame){vram_str}")
            return (output_tensor,)

        except Exception as e:
            raise RuntimeError(f"Koolook Matte inference failed: {e}")

    def _run_windows(self, pipeline, cond_frames, mask_frames, num_frames,
                     window_size, window_overlap, seed, fps, motion_bucket_id,
                     noise_aug_strength):
        """Run the model over the clip in temporal windows and return a per-frame list
        of target-resolution grayscale float32 [0,1] mattes.

        Raises the CUDA OutOfMemoryError from pipeline.run unchanged so the caller can
        catch it and retry with a smaller window_size.
        """
        windows = self._generate_windows(num_frames, window_size, window_overlap)
        single_pass = (len(windows) == 1)
        # Effective overlap (0 for a single pass, clamped so windows always advance)
        overlap = 0 if single_pass else min(window_overlap, window_size - 1)

        if single_pass:
            print(f"Running Koolook Matte inference on {num_frames} frames (single pass)...")
        else:
            print(f"Running Koolook Matte inference on {num_frames} frames in "
                  f"{len(windows)} windows of up to {window_size} frames (overlap {overlap})...")

        # pipeline.run issues 4 progress updates per pass
        pbar = ProgressBar(4 * len(windows))

        # output_accum[i] holds a target-resolution grayscale float32 [0,1] array
        output_accum = [None] * num_frames
        prev_overlap_soft = None  # previous window's last `overlap` outputs, as PIL 'L'

        for w_idx, (w_start, w_end) in enumerate(windows):
            is_first = (w_idx == 0)
            win_cond = cond_frames[w_start:w_end]
            win_mask = list(mask_frames[w_start:w_end])

            # Feedback injection: replace the leading `overlap` conditioning masks
            # with the previous window's soft alpha output. Kept soft (not
            # binarized) so the model sees graduated edge values across the seam.
            if not is_first and overlap > 0 and prev_overlap_soft is not None:
                for i in range(min(overlap, len(win_mask), len(prev_overlap_soft))):
                    win_mask[i] = prev_overlap_soft[i]

            out_pil = pipeline.run(
                cond_frames=win_cond,
                mask_frames=win_mask,
                seed=seed,
                fps=fps,
                motion_bucket_id=motion_bucket_id,
                noise_aug_strength=noise_aug_strength,
                pbar=pbar
            )

            out_gray = [self._to_gray01(f) for f in out_pil]

            # Capture soft tail for the next window's feedback loop
            if overlap > 0 and len(out_gray) >= overlap:
                prev_overlap_soft = [
                    Image.fromarray((g * 255.0).clip(0, 255).astype(np.uint8), mode='L')
                    for g in out_gray[-overlap:]
                ]
            else:
                prev_overlap_soft = None

            # Commit outputs. On non-first windows the leading `overlap` frames
            # already have a committed prediction from the previous window, so we
            # linearly blend the two across the seam (weight ramps 0 -> 1).
            for i, g in enumerate(out_gray):
                abs_idx = w_start + i
                if is_first or i >= overlap:
                    output_accum[abs_idx] = g
                elif output_accum[abs_idx] is not None:
                    blend_w = (i + 1) / (overlap + 1)
                    output_accum[abs_idx] = (1.0 - blend_w) * output_accum[abs_idx] + blend_w * g
                else:
                    output_accum[abs_idx] = g

        return output_accum


class KoolookMatteFocusCrop:
    """Crop a video batch to a single static 'hero' box around the subject.

    The box is the union of `coarse_mask` across every frame (plus padding), so the
    crop is identical for the whole clip — a subject moving inside a locked frame,
    NOT a per-frame match-move (which would break SAM3 / VideoMaMa temporal tracking).
    Run SAM3 and VideoMaMa on the crop at the subject's native pixels, then paste the
    result back with KoolookMatteFocusStitch. This is the "focused comp" pattern: crop
    once, do all the work in that high-res space, stitch back at the end.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "coarse_mask": ("MASK",),
                "padding": ("INT", {
                    "default": 24, "min": 0, "max": 1024, "step": 8,
                    "tooltip": "Pixels of margin around the subject's union bounding box. Give thin appendages (claws, tails) room so they fall INSIDE the crop, otherwise they can't be recovered downstream."
                }),
                "max_long_side": ("INT", {
                    "default": 2560, "min": 0, "max": 8192, "step": 64,
                    "tooltip": "VRAM safety cap: if the native crop's LONGEST side exceeds this, downscale the crop to fit (aspect kept). Never upscales. This same cropped image feeds BOTH the segmentation and the matte model, so it bounds the area both process. 0 = no cap (native crop). 2560 comfortably fits a 24GB card."
                }),
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Master toggle for the focused two-pass. ON = crop to the subject, process at native res, stitch back. OFF = pass the FULL frame straight through unchanged (identity focus), so the downstream seg/matte run single-pass full-frame. Promote this to a subgraph to get a one-click two-pass switch."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "FOCUS")
    RETURN_NAMES = ("cropped_images", "cropped_mask", "focus")
    FUNCTION = "crop"
    CATEGORY = "Koolook Matte"

    def crop(self, images, coarse_mask, enabled, padding, max_long_side):
        num = images.shape[0]
        orig_h, orig_w = images.shape[1], images.shape[2]

        # --- Bypass: pass the full frame through unchanged with an identity focus, so
        # FocusStitch (which just pastes into the bbox) becomes a no-op passthrough.
        if not enabled:
            cm = []
            for i in range(coarse_mask.shape[0]):
                a = _ensure_2d_mask((coarse_mask[i].cpu().numpy() * 255).astype(np.uint8))
                if a.shape[:2] != (orig_h, orig_w):
                    a = np.array(Image.fromarray(a, mode='L').resize((orig_w, orig_h), Image.NEAREST))
                cm.append(a.astype(np.float32) / 255.0)
            cropped_mask = torch.from_numpy(np.stack(cm, axis=0)) if cm else torch.zeros((num, orig_h, orig_w))
            focus = {"bbox": (0, 0, orig_w, orig_h), "orig_w": orig_w, "orig_h": orig_h,
                     "crop_w": orig_w, "crop_h": orig_h}
            print(f"[FocusCrop] disabled -> full-frame passthrough ({orig_w}x{orig_h})")
            return (images, cropped_mask, focus)

        # Bring the coarse mask to image resolution so the bbox is in image space.
        masks_full = []
        for i in range(coarse_mask.shape[0]):
            a = _ensure_2d_mask((coarse_mask[i].cpu().numpy() * 255).astype(np.uint8))
            if a.shape[:2] != (orig_h, orig_w):
                a = np.array(Image.fromarray(a, mode='L').resize((orig_w, orig_h), Image.NEAREST))
            masks_full.append(a)

        bbox = KoolookMatteSampler._compute_union_bbox(masks_full, padding, orig_w, orig_h)
        if bbox is None:
            bbox = (0, 0, orig_w, orig_h)   # no mask found -> pass the whole frame
        x1, y1, x2, y2 = bbox
        crop_w0, crop_h0 = x2 - x1, y2 - y1

        # VRAM cap: downscale the crop only if its longest side exceeds max_long_side
        # (never upscales). This one cropped image feeds BOTH the segmentation and the
        # matte model, so it bounds the area both process.
        long_side = max(crop_w0, crop_h0)
        if max_long_side and max_long_side > 0 and long_side > max_long_side:
            scale = max_long_side / long_side
            out_w = int(crop_w0 * scale)
            out_h = int(crop_h0 * scale)
        else:
            out_w, out_h = crop_w0, crop_h0
        out_w = max((out_w // 8) * 8, 8)
        out_h = max((out_h // 8) * 8, 8)

        img_out = []
        for i in range(num):
            im = (images[i].cpu().numpy() * 255).astype(np.uint8)[y1:y2, x1:x2]
            im = np.array(Image.fromarray(im, mode='RGB').resize((out_w, out_h), Image.LANCZOS))
            img_out.append(im.astype(np.float32) / 255.0)

        msk_out = []
        for m in masks_full:
            mm = m[y1:y2, x1:x2]
            mm = np.array(Image.fromarray(mm, mode='L').resize((out_w, out_h), Image.NEAREST))
            msk_out.append(mm.astype(np.float32) / 255.0)

        cropped_images = torch.from_numpy(np.stack(img_out, axis=0))
        cropped_mask = torch.from_numpy(np.stack(msk_out, axis=0)) if msk_out else torch.zeros((num, out_h, out_w))
        focus = {"bbox": (x1, y1, x2, y2), "orig_w": orig_w, "orig_h": orig_h,
                 "crop_w": crop_w0, "crop_h": crop_h0}
        print(f"[FocusCrop] {orig_w}x{orig_h} -> hero box {crop_w0}x{crop_h0} at "
              f"({x1},{y1}) -> output {out_w}x{out_h}")
        return (cropped_images, cropped_mask, focus)


class KoolookMatteFocusStitch:
    """Paste a crop-space matte (produced inside KoolookMatteFocusCrop's region) back
    into the full-frame plate. Outside the hero box the matte is 0 (background)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "focus": ("FOCUS",),
                "cropped_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "stitch"
    CATEGORY = "Koolook Matte"

    def stitch(self, focus, cropped_mask):
        x1, y1, x2, y2 = focus["bbox"]
        orig_w, orig_h = focus["orig_w"], focus["orig_h"]
        crop_w, crop_h = focus["crop_w"], focus["crop_h"]

        out = []
        for i in range(cropped_mask.shape[0]):
            m = _ensure_2d_mask((cropped_mask[i].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8))
            # Resize the matte back to the hero box's native pixels, then paste.
            if m.shape[:2] != (crop_h, crop_w):
                m = np.array(Image.fromarray(m, mode='L').resize((crop_w, crop_h), Image.LANCZOS))
            canvas = np.zeros((orig_h, orig_w), dtype=np.float32)
            canvas[y1:y2, x1:x2] = m.astype(np.float32) / 255.0
            out.append(canvas)
        full = torch.from_numpy(np.stack(out, axis=0))
        print(f"[FocusStitch] pasted {crop_w}x{crop_h} matte back into {orig_w}x{orig_h}")
        return (full,)


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "KoolookMatteLoader": KoolookMatteLoader,
    "KoolookMatteSampler": KoolookMatteSampler,
    "KoolookMatteFocusCrop": KoolookMatteFocusCrop,
    "KoolookMatteFocusStitch": KoolookMatteFocusStitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KoolookMatteLoader": "Koolook Matte — Loader",
    "KoolookMatteSampler": "Koolook Matte — Sampler",
    "KoolookMatteFocusCrop": "Koolook Matte — Focus Crop",
    "KoolookMatteFocusStitch": "Koolook Matte — Focus Stitch",
}
