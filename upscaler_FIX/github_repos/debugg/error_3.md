got prompt
loading images:  17%|█▋        | 14/81 [00:01<00:08,  8.29it/s]FETCH ComfyRegistry Data: 40/111
loading images:  52%|█████▏    | 42/81 [00:05<00:04,  7.97it/s]FETCH ComfyRegistry Data: 45/111
loading images:  85%|████████▌ | 69/81 [00:08<00:01,  8.14it/s]FETCH ComfyRegistry Data: 50/111
loading images: 100%|██████████| 81/81 [00:10<00:00,  7.94it/s]
loading images:  22%|██▏       | 18/81 [00:01<00:04, 14.08it/s]FETCH ComfyRegistry Data: 55/111
loading images:  81%|████████▏ | 66/81 [00:04<00:01, 13.81it/s]FETCH ComfyRegistry Data: 60/111
loading images: 100%|██████████| 81/81 [00:05<00:00, 14.04it/s]
FETCH ComfyRegistry Data: 65/111
Using pytorch attention in VAE
Using pytorch attention in VAE
VAE load device: cuda:0, offload device: cpu, dtype: torch.bfloat16
Found quantization metadata version 1
[MultiGPU Core Patching] text_encoder_device_patched returning device: cuda:0 (current_text_encoder_device=cuda:0)
Using MixedPrecisionOps for text encoder
CLIP/text encoder model load device: cuda:0, offload device: cpu, current: cpu, dtype: torch.float16
Requested to load WanTEModel
FETCH ComfyRegistry Data: 70/111
loaded completely; 21474.80 MB usable, 6419.49 MB loaded, full load: True
gguf qtypes: F32 (836), Q4_K (345), Q6_K (144), F16 (6)
model weight dtype torch.float16, manual cast: None
model_type FLOW
FETCH ComfyRegistry Data: 75/111
FETCH ComfyRegistry Data: 80/111
FETCH ComfyRegistry Data: 85/111
Requested to load WanVAE
loaded completely; 7532.40 MB usable, 242.03 MB loaded, full load: True
FETCH ComfyRegistry Data: 90/111
FETCH ComfyRegistry Data: 95/111
FETCH ComfyRegistry Data: 100/111
FETCH ComfyRegistry Data: 105/111
FETCH ComfyRegistry Data: 110/111
FETCH ComfyRegistry Data [DONE]
[ComfyUI-Manager] default cache updated: https://api.comfy.org/nodes
FETCH DATA from: C:\Users\ai.machine\Documents\ComfyUI\user\__manager\cache\1514988643_custom-node-list.json [DONE]
[ComfyUI-Manager] All startup tasks have been completed.
Requested to load WAN21_Vace
loaded partially; 9114.63 MB usable, 9072.38 MB loaded, 2154.32 MB offloaded, 42.25 MB buffer reserved, lowvram patches: 0
Attempting to release mmap (381)
100%|██████████| 4/4 [02:31<00:00, 37.91s/it]
Requested to load WanVAE
loaded completely; 3625.50 MB usable, 242.03 MB loaded, full load: True
第 1 部分视频第 1 块生成完成；整体完成 25.0 %
!!! Exception during processing !!! CUDA error: invalid argument
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

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
  File "C:\Users\ai.machine\Documents\ComfyUI\custom_nodes\ComfyUI-SuperUltimateVaceTools\nodes.py", line 461, in upscale_video
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
  File "C:\Users\ai.machine\AppData\Local\Programs\@comfyorgcomfyui-electron\resources\ComfyUI\comfy\model_management.py", line 671, in load_models_gpu
    free_memory(total_memory_required[device] * 1.1 + extra_mem, device)
  File "C:\Users\ai.machine\AppData\Local\Programs\@comfyorgcomfyui-electron\resources\ComfyUI\comfy\model_management.py", line 603, in free_memory
    if current_loaded_models[i].model_unload(memory_to_free):
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\ai.machine\AppData\Local\Programs\@comfyorgcomfyui-electron\resources\ComfyUI\comfy\model_management.py", line 529, in model_unload
    self.model.detach(unpatch_weights)
  File "C:\Users\ai.machine\AppData\Local\Programs\@comfyorgcomfyui-electron\resources\ComfyUI\comfy\model_patcher.py", line 986, in detach
    self.unpatch_model(self.offload_device, unpatch_weights=unpatch_all)
  File "C:\Users\ai.machine\AppData\Local\Programs\@comfyorgcomfyui-electron\resources\ComfyUI\comfy\model_patcher.py", line 854, in unpatch_model
    self.model.to(device_to)
  File "C:\Users\ai.machine\Documents\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1355, in to
    return self._apply(convert)
           ^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\ai.machine\Documents\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 915, in _apply
    module._apply(fn)
  File "C:\Users\ai.machine\Documents\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 915, in _apply
    module._apply(fn)
  File "C:\Users\ai.machine\Documents\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 915, in _apply
    module._apply(fn)
  [Previous line repeated 2 more times]
  File "C:\Users\ai.machine\Documents\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 942, in _apply
    param_applied = fn(param)
                    ^^^^^^^^^
  File "C:\Users\ai.machine\Documents\ComfyUI\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1341, in convert
    return t.to(
           ^^^^^
RuntimeError: CUDA error: invalid argument
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.


Prompt executed in 236.19 seconds