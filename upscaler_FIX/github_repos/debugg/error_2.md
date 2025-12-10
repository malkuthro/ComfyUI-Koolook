got prompt
loading images: 100%|██████████| 241/241 [00:20<00:00, 11.71it/s]
Using pytorch attention in VAE
Using pytorch attention in VAE
VAE load device: cuda:0, offload device: cpu, dtype: torch.bfloat16
Found quantization metadata version 1
[MultiGPU Core Patching] text_encoder_device_patched returning device: cuda:0 (current_text_encoder_device=cuda:0)
Using MixedPrecisionOps for text encoder
CLIP/text encoder model load device: cuda:0, offload device: cpu, current: cpu, dtype: torch.float16
Requested to load WanTEModel
loaded completely; 21474.80 MB usable, 6419.49 MB loaded, full load: True
[MultiGPU Core Patching] Successfully patched ModelPatcher.partially_load
gguf qtypes: F32 (836), Q4_K (345), Q6_K (144), F16 (6)
model weight dtype torch.float16, manual cast: None
model_type FLOW
Requested to load WanVAE
loaded completely; 13567.55 MB usable, 242.03 MB loaded, full load: True
Requested to load WAN21_Vace
===============================================
    DisTorch2 Model Virtual VRAM Analysis
===============================================
Object   Role   Original(GB) Total(GB)  Virt(GB)
-----------------------------------------------
cuda:0   recip      23.99GB   27.99GB   +4.00GB
cpu      donor     127.12GB  123.12GB   -4.00GB
-----------------------------------------------
model    model      10.83GB    6.83GB   -4.00GB
==================================================
[MultiGPU DisTorch V2] Final Allocation String:
cuda:0,0.2848;cpu,0.0315
==================================================
    DisTorch2 Model Device Allocations
==================================================
Device    VRAM GB    Dev %   Model GB    Dist %
--------------------------------------------------
cuda:0      23.99    28.5%       6.83     63.0%
cpu        127.12     3.1%       4.00     37.0%
--------------------------------------------------
!!! Exception during processing !!! too many values to unpack (expected 4)
Traceback (most recent call last):
  File "C:\Users\ai.machine\AppData\Local\Programs\@comfyorgcomfyui-electron\resources\ComfyUI\execution.py", line 515, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\ai.machine\AppData\Local\Programs\@comfyorgcomfyui-electron\resources\ComfyUI\execution.py", line 329, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\ai.machine\AppData\Local\Programs\@comfyorgcomfyui-electron\resources\ComfyUI\execution.py", line 303, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "C:\Users\ai.machine\AppData\Local\Programs\@comfyorgcomfyui-electron\resources\ComfyUI\execution.py", line 291, in process_inputs
    result = f(**inputs)
             ^^^^^^^^^^^
  File "C:\Users\ai.machine\Documents\ComfyUI\custom_nodes\ComfyUI-SuperUltimateVaceTools\nodes.py", line 468, in upscale_video
    sample = nodes.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, conditions['positive'], conditions['negative'], conditions['out_latent'],
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\ai.machine\AppData\Local\Programs\@comfyorgcomfyui-electron\resources\ComfyUI\nodes.py", line 1505, in common_ksampler
    samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\ai.machine\AppData\Local\Programs\@comfyorgcomfyui-electron\resources\ComfyUI\comfy\sample.py", line 60, in sample
    samples = sampler.sample(noise, positive, negative, cfg=cfg, latent_image=latent_image, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, denoise_mask=noise_mask, sigmas=sigmas, callback=callback, disable_pbar=disable_pbar, seed=seed)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\ai.machine\Documents\ComfyUI\custom_nodes\ComfyUI-TiledDiffusion\utils.py", line 51, in KSampler_sample
    return orig_fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\ai.machine\AppData\Local\Programs\@comfyorgcomfyui-electron\resources\ComfyUI\comfy\samplers.py", line 1163, in sample
    return sample(self.model, noise, positive, negative, cfg, self.device, sampler, sigmas, self.model_options, latent_image=latent_image, denoise_mask=denoise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\ai.machine\AppData\Local\Programs\@comfyorgcomfyui-electron\resources\ComfyUI\comfy\samplers.py", line 1053, in sample
    return cfg_guider.sample(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\ai.machine\AppData\Local\Programs\@comfyorgcomfyui-electron\resources\ComfyUI\comfy\samplers.py", line 1035, in sample
    output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\ai.machine\AppData\Local\Programs\@comfyorgcomfyui-electron\resources\ComfyUI\comfy\patcher_extension.py", line 112, in execute
    return self.original(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\ai.machine\AppData\Local\Programs\@comfyorgcomfyui-electron\resources\ComfyUI\comfy\samplers.py", line 984, in outer_sample
    self.inner_model, self.conds, self.loaded_models = comfy.sampler_helpers.prepare_sampling(self.model_patcher, noise.shape, self.conds, self.model_options)
                                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\ai.machine\AppData\Local\Programs\@comfyorgcomfyui-electron\resources\ComfyUI\comfy\sampler_helpers.py", line 130, in prepare_sampling
    return executor.execute(model, noise_shape, conds, model_options=model_options)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\ai.machine\AppData\Local\Programs\@comfyorgcomfyui-electron\resources\ComfyUI\comfy\patcher_extension.py", line 112, in execute
    return self.original(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\ai.machine\AppData\Local\Programs\@comfyorgcomfyui-electron\resources\ComfyUI\comfy\sampler_helpers.py", line 138, in _prepare_sampling
    comfy.model_management.load_models_gpu([model] + models, memory_required=memory_required + inference_memory, minimum_memory_required=minimum_memory_required + inference_memory)
  File "C:\Users\ai.machine\Documents\ComfyUI\custom_nodes\comfyui-multigpu\distorch_2.py", line 199, in patched_load_models_gpu
    loaded_model.model_load(lowvram_model_memory, force_patch_weights=force_patch_weights)
  File "C:\Users\ai.machine\AppData\Local\Programs\@comfyorgcomfyui-electron\resources\ComfyUI\comfy\model_management.py", line 506, in model_load
    self.model_use_more_vram(use_more_vram, force_patch_weights=force_patch_weights)
  File "C:\Users\ai.machine\AppData\Local\Programs\@comfyorgcomfyui-electron\resources\ComfyUI\comfy\model_management.py", line 536, in model_use_more_vram
    return self.model.partially_load(self.device, extra_memory, force_patch_weights=force_patch_weights)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\ai.machine\Documents\ComfyUI\custom_nodes\comfyui-multigpu\distorch_2.py", line 240, in new_partially_load
    device_assignments = analyze_safetensor_loading(self, allocations, is_clip=is_clip_model)
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\ai.machine\Documents\ComfyUI\custom_nodes\comfyui-multigpu\distorch_2.py", line 422, in analyze_safetensor_loading
    total_memory = sum(module_size for module_size, _, _, _ in raw_block_list)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\ai.machine\Documents\ComfyUI\custom_nodes\comfyui-multigpu\distorch_2.py", line 422, in <genexpr>
    total_memory = sum(module_size for module_size, _, _, _ in raw_block_list)
                                       ^^^^^^^^^^^^^^^^^^^^
ValueError: too many values to unpack (expected 4)

Prompt executed in 61.79 seconds