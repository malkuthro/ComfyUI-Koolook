import torch

class MultiKeyframeImageBatch:
    """
    A custom ComfyUI node for VFX workflows that takes 1-4 keyframe images with specified frame positions
    and creates an IMAGE batch sequence of a given total length, filling undefined frames with black images.
    Useful for preparing sparse control sequences for video models like Wan 2.2, where black frames indicate
    no latent update. Assumes all input images have identical dimensions.
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
                "image1": ("IMAGE", ),
                "frame1": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1023,
                    "step": 1,
                    "display": "number"
                }),
            },
            "optional": {
                "image2": ("IMAGE", ),
                "frame2": ("INT", {
                    "default": 4,
                    "min": 0,
                    "max": 1023,
                    "step": 1,
                    "display": "number"
                }),
                "image3": ("IMAGE", ),
                "frame3": ("INT", {
                    "default": 8,
                    "min": 0,
                    "max": 1023,
                    "step": 1,
                    "display": "number"
                }),
                "image4": ("IMAGE", ),
                "frame4": ("INT", {
                    "default": 12,
                    "min": 0,
                    "max": 1023,
                    "step": 1,
                    "display": "number"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image_batch", )
    FUNCTION = "create_batch"
    CATEGORY = "Koolook/VFX"
    OUTPUT_NODE = False  # Not necessarily an output node, but can be chained

    def create_batch(self, total_frames, image1, frame1, image2=None, frame2=None, image3=None, frame3=None, image4=None, frame4=None):
        if image1.shape[0] != 1:
            raise ValueError("Each input should be a single IMAGE (batch size 1), not a pre-batched tensor.")

        h, w, c = image1.shape[1:]
        black = torch.zeros(h, w, c)  # [H, W, C] zero tensor for black frame

        # Initialize list of frames with blacks
        frame_images = [black.clone() for _ in range(total_frames)]

        # Collect connected keyframes
        keyframes = [(image1[0], frame1)]
        if image2 is not None and frame2 is not None:
            if image2.shape[1:] != (h, w, c):
                raise ValueError("All images must have the same height, width, and channels.")
            keyframes.append((image2[0], frame2))
        if image3 is not None and frame3 is not None:
            if image3.shape[1:] != (h, w, c):
                raise ValueError("All images must have the same height, width, and channels.")
            keyframes.append((image3[0], frame3))
        if image4 is not None and frame4 is not None:
            if image4.shape[1:] != (h, w, c):
                raise ValueError("All images must have the same height, width, and channels.")
            keyframes.append((image4[0], frame4))

        # Place keyframes in the sequence
        for img, f in keyframes:
            if 0 <= f < total_frames:
                frame_images[f] = img
            else:
                print(f"Warning: Frame {f} is out of range (0 to {total_frames-1}); ignoring this keyframe.")

        # Stack into batch tensor [total_frames, H, W, C]
        batch = torch.stack(frame_images, dim=0)
        return (batch, )

# Node mappings for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "MultiKeyframeImageBatch": MultiKeyframeImageBatch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MultiKeyframeImageBatch": "Multi Keyframe Image Batch (VFX)"
}