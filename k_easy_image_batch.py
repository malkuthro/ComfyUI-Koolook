import torch

class easy_ImageBatch:
    """
    A custom ComfyUI node for VFX workflows that takes 1-4 keyframe images with specified frame positions
    and creates an IMAGE batch sequence of a given total length, filling undefined frames with selected placeholder color (black or 50% gray).
    Also outputs a MASK batch where occupied frames are white (1.0) and empty are black (0.0), directly compatible with standard mask nodes.
    Supports VFX-style frame numbering with a start_frame offset (internal indices are 0-based).
    Useful for preparing sparse control sequences for video models like Wan 2.2, where placeholder frames indicate no latent update.
    Assumes all input images have identical dimensions.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "total_frames": ("INT", {
                    "default": 16,
                    "min": 1,
                    "max": 1024,
                    "step": 1,
                    "display": "number"
                }),
                "start_frame": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 999999,
                    "step": 1,
                    "display": "number"
                }),
                "placeholder_color": (["Black", "Gray"], {
                    "default": "Black"
                }),
                "image1": ("IMAGE", ),
                "image1_frame": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 999999,
                    "step": 1,
                    "display": "number"
                }),
            },
            "optional": {
                "image2": ("IMAGE", ),
                "image2_frame": ("INT", {
                    "default": 4,
                    "min": 0,
                    "max": 999999,
                    "step": 1,
                    "display": "number"
                }),
                "image3": ("IMAGE", ),
                "image3_frame": ("INT", {
                    "default": 8,
                    "min": 0,
                    "max": 999999,
                    "step": 1,
                    "display": "number"
                }),
                "image4": ("IMAGE", ),
                "image4_frame": ("INT", {
                    "default": 12,
                    "min": 0,
                    "max": 999999,
                    "step": 1,
                    "display": "number"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", )
    RETURN_NAMES = ("image_batch", "mask_batch", )
    FUNCTION = "create_batch"
    CATEGORY = "Koolook/VFX"
    OUTPUT_NODE = False  # Not necessarily an output node, but can be chained

    def create_batch(self, total_frames, start_frame, placeholder_color, image1, image1_frame, image2=None, image2_frame=None, image3=None, image3_frame=None, image4=None, image4_frame=None):
        if image1.shape[0] != 1:
            raise ValueError("Each input should be a single IMAGE (batch size 1), not a pre-batched tensor.")

        h, w, c = image1.shape[1:]
        if placeholder_color == "Black":
            placeholder = torch.zeros(h, w, c)  # [H, W, C] zero tensor for black
        else:  # Gray
            placeholder = torch.full((h, w, c), 0.5)  # 50% gray in 0-1 float space

        # For masks: single-channel [H, W]
        black_mask = torch.zeros((h, w), dtype=torch.float32)
        white_mask = torch.ones((h, w), dtype=torch.float32)

        # Initialize lists for frames and masks
        frame_images = [placeholder.clone() for _ in range(total_frames)]
        frame_masks = [black_mask.clone() for _ in range(total_frames)]

        # Collect connected keyframes
        keyframes = [(image1[0], image1_frame)]
        if image2 is not None and image2_frame is not None:
            if image2.shape[1:] != (h, w, c):
                raise ValueError("All images must have the same height, width, and channels.")
            keyframes.append((image2[0], image2_frame))
        if image3 is not None and image3_frame is not None:
            if image3.shape[1:] != (h, w, c):
                raise ValueError("All images must have the same height, width, and channels.")
            keyframes.append((image3[0], image3_frame))
        if image4 is not None and image4_frame is not None:
            if image4.shape[1:] != (h, w, c):
                raise ValueError("All images must have the same height, width, and channels.")
            keyframes.append((image4[0], image4_frame))

        # Place keyframes in the sequence
        for img, f in keyframes:
            index = f - start_frame
            if 0 <= index < total_frames:
                frame_images[index] = img
                frame_masks[index] = white_mask.clone()
            else:
                print(f"Warning: Frame {f} maps to index {index} which is out of range (0 to {total_frames-1}); ignoring this keyframe.")

        # Stack into batch tensors
        image_batch = torch.stack(frame_images, dim=0)  # [total_frames, H, W, C]
        mask_batch = torch.stack(frame_masks, dim=0)    # [total_frames, H, W]
        return (image_batch, mask_batch, )

# Node mappings for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "easy_ImageBatch": easy_ImageBatch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "easy_ImageBatch": "Easy Image Batch"
}