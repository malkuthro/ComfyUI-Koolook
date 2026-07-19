"""
Koolook Matte — independent inference pipeline for mask-guided one-step video matting.

This is an ORIGINAL implementation of the *published* VideoMaMa method
("VideoMaMa: Mask-Guided Video Matting via Generative Prior", CVPR 2026), written
from the method description + the standard `diffusers` API — it does not incorporate
any third-party wrapper code. The method itself:

  * a Stable-Video-Diffusion UNet fine-tuned to take **12 latent channels**
    = 4 (noise) + 4 (VAE-encoded RGB frame) + 4 (VAE-encoded guide mask),
  * a **single** denoising step at timestep 1 with the CLIP cross-attention
    context zeroed (the fine-tune ignores it),
  * decode the predicted latents with SVD's temporal VAE and take luminance
    as the alpha matte.

The model WEIGHTS it loads (the VideoMaMa UNet, and SVD's VAE) carry their own
licenses (CC BY-NC 4.0 and the Stability Community License respectively); this
code does not redistribute them and the end user is responsible for compliance.

Licensed under GPL-3.0 — see the repository LICENSE.
"""

import numpy as np
import torch

from diffusers.models import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel


class KoolookMattePipeline:
    """One-step mask-guided SVD matting. Reusable across many clips/windows."""

    def __init__(self, base_model_path: str, unet_checkpoint_path: str, device: str = "cuda",
                 weight_dtype: torch.dtype = torch.float16, enable_model_cpu_offload: bool = False,
                 vae_encode_chunk_size: int = 1, attention_mode: str = "auto",
                 enable_vae_tiling: bool = False, enable_vae_slicing: bool = True):
        self.device = torch.device(device)
        self.weight_dtype = weight_dtype
        self.enable_model_cpu_offload = enable_model_cpu_offload
        self.vae_encode_chunk_size = max(1, int(vae_encode_chunk_size))

        # SVD's temporal VAE (from the base model) + the VideoMaMa-fine-tuned UNet.
        try:
            self.vae = AutoencoderKLTemporalDecoder.from_pretrained(base_model_path, subfolder="vae", variant="fp16")
        except Exception:
            # some snapshots ship the vae without an fp16 variant
            self.vae = AutoencoderKLTemporalDecoder.from_pretrained(base_model_path, subfolder="vae")
        self.unet = UNetSpatioTemporalConditionModel.from_pretrained(unet_checkpoint_path, subfolder="unet")

        self.vae.eval()
        self.unet.eval()
        self._apply_attention_optimization(attention_mode)

        if enable_vae_slicing:
            try:
                self.vae.enable_slicing()
            except (AttributeError, NotImplementedError):
                pass
        if enable_vae_tiling:
            try:
                self.vae.enable_tiling()
            except (AttributeError, NotImplementedError):
                pass

        if self.enable_model_cpu_offload:
            self.vae.to("cpu", dtype=self.weight_dtype)
            self.unet.to("cpu", dtype=self.weight_dtype)
        else:
            self.vae.to(self.device, dtype=self.weight_dtype)
            self.unet.to(self.device, dtype=self.weight_dtype)

    # -- setup helpers ------------------------------------------------------

    def _apply_attention_optimization(self, attention_mode: str):
        if attention_mode == "none":
            return
        if attention_mode in ("auto", "xformers"):
            try:
                import xformers  # noqa: F401
                self.unet.enable_xformers_memory_efficient_attention()
                return
            except Exception:
                if attention_mode == "xformers":
                    print("KoolookMatte: xformers unavailable, falling back")
        if attention_mode in ("auto", "sdpa"):
            if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                try:
                    from diffusers.models.attention_processor import AttnProcessor2_0
                    self.unet.set_attn_processor(AttnProcessor2_0())
                    return
                except Exception:
                    pass

    @staticmethod
    def _clear_cache(device):
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif device.type == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif device.type == "xpu":
            torch.xpu.empty_cache()

    # -- core inference -----------------------------------------------------

    @torch.no_grad()
    def run(self, cond_frames, mask_frames, seed=42, fps=7, motion_bucket_id=127,
            noise_aug_strength=0.0, pbar=None):
        """Run one-step matting on a list of PIL frames + PIL guide masks.

        Returns a list of grayscale PIL alpha frames (mode 'L').
        `pbar` is an optional ComfyUI ProgressBar (updated 4x per pass).
        """
        def tick():
            if pbar is not None:
                pbar.update(1)

        cond = self._pil_to_tensor(cond_frames).to(self.device)          # [1,F,3,H,W] in [-1,1]
        mask = self._pil_to_tensor(mask_frames, grayscale_ok=True).to(self.device)
        if mask.shape[2] != 3:                                            # replicate L -> 3ch for the VAE
            mask = mask.repeat(1, 1, 3, 1, 1)

        hidden = self.unet.config.cross_attention_dim
        encoder_hidden_states = torch.zeros((1, 1, hidden), dtype=self.weight_dtype, device=self.device)

        # --- encode RGB frame + guide mask to latents ---
        if self.enable_model_cpu_offload:
            self.vae.to(self.device)
        cond_latents = self._encode_latents(cond.to(self.weight_dtype))
        mask_latents = self._encode_latents(mask.to(self.weight_dtype))
        tick()
        del cond, mask
        self._clear_cache(self.device)
        if self.enable_model_cpu_offload:
            self.vae.to("cpu"); self._clear_cache(self.device)

        # --- single UNet step over all frames at once ---
        if self.enable_model_cpu_offload:
            self.unet.to(self.device)
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        noise = torch.randn(cond_latents.shape, generator=gen, device="cpu",
                            dtype=self.weight_dtype).to(self.device)
        timesteps = torch.ones((1,), device=self.device, dtype=torch.long)
        added_time_ids = self._add_time_ids(fps, motion_bucket_id, noise_aug_strength)

        unet_in = torch.cat([noise, cond_latents, mask_latents], dim=2)   # 12 channels
        del noise, cond_latents, mask_latents
        self._clear_cache(self.device)
        tick()

        pred = self.unet(unet_in, timesteps, encoder_hidden_states, added_time_ids=added_time_ids).sample
        del unet_in
        if self.enable_model_cpu_offload:
            self.unet.to("cpu")
        self._clear_cache(self.device)
        tick()

        # --- decode latents -> luminance alpha ---
        if self.enable_model_cpu_offload:
            self.vae.to(self.device)
        pred = pred.squeeze(0) / self.vae.config.scaling_factor          # [F,4,h,w]
        chunk = min(self.vae_encode_chunk_size, pred.shape[0])
        decoded = []
        for i in range(0, pred.shape[0], chunk):
            part = pred[i:i + chunk]
            decoded.append(self.vae.decode(part, num_frames=part.shape[0]).sample.cpu())
        del pred
        if self.enable_model_cpu_offload:
            self.vae.to("cpu")
        self._clear_cache(self.device)
        tick()

        video = torch.cat(decoded, dim=0)                                # [F,3,H,W]
        del decoded
        video = (video / 2.0 + 0.5).clamp(0, 1).mean(dim=1)              # [F,H,W] luminance
        out = []
        for frame in video.float().cpu().numpy():
            out.append(_to_pil_L(frame))
        return out

    # -- tensor plumbing ----------------------------------------------------

    def _pil_to_tensor(self, frames, grayscale_ok=False):
        arrs = []
        for f in frames:
            a = np.asarray(f)
            if a.ndim == 2:
                a = np.stack([a] * 3, axis=-1)
            t = torch.from_numpy(a.copy()).permute(2, 0, 1).float() / 255.0
            arrs.append(t)
        video = torch.stack(arrs).unsqueeze(0)                           # [1,F,3,H,W]
        return video * 2.0 - 1.0

    def _encode_latents(self, t):
        """Encode [1,F,3,H,W] -> raw VAE latents [1,F,4,h,w] (chunked over frames).

        The UNet is fed conditioning + mask latents at the RAW VAE scale (no
        ``scaling_factor`` applied), matching okdalto's ``VideoInferencePipeline``
        (which encodes ``* scaling_factor`` then divides it back out, netting to
        raw). Dividing here would over-scale the conditioning ~5.5x (1/0.18215),
        pushing the fine-tuned UNet out of distribution and softening the matte.
        """
        b, f = t.shape[0], t.shape[1]
        t = t.reshape(b * f, *t.shape[2:])
        parts = []
        for i in range(0, t.shape[0], self.vae_encode_chunk_size):
            parts.append(self.vae.encode(t[i:i + self.vae_encode_chunk_size]).latent_dist.sample())
        lat = torch.cat(parts, dim=0)
        return lat.reshape(b, f, *lat.shape[1:])

    def _add_time_ids(self, fps, motion_bucket_id, noise_aug_strength):
        vals = [fps, motion_bucket_id, noise_aug_strength]
        want = self.unet.config.addition_time_embed_dim * len(vals)
        have = self.unet.add_embedding.linear_1.in_features
        if want != have:
            raise ValueError(f"added_time_ids dim mismatch: model expects {have}, built {want}")
        return torch.tensor([vals], dtype=self.weight_dtype, device=self.device)


def _to_pil_L(arr01):
    from PIL import Image
    return Image.fromarray((arr01 * 255).clip(0, 255).astype(np.uint8), mode="L")
