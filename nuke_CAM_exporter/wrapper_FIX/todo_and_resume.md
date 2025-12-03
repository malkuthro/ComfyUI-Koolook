# Wan FunCamera Wrapper Fix – Status & TODOs

## Summary of Current State
- **ASCI Export/Converter**
- Confirmed Nuke export columns `[Tx, Rx, Ty, Ry, Tz, Rz]` (degrees, ZXY order).
  - `nuke_ASCI-2-Pose_converter.py` converts to RealEstate-style `[R|t]` rows and works for arbitrary frame counts.
  - Latest file `outputs/camTrack_v01_converted_v5.txt` trimmed to 81 frames to match Wan’s multiple-of-4 plus one requirement.
- **ComfyUI/Wan Errors**
  - Resolved the earlier reshape error (frame count mismatch).
  - Still hitting lightning/transformer error: `tensor a (21)` vs `tensor b (41)` when adding video tokens to camera tokens.
  - Lightning LoRA disabled but Wan 14B still emits 21 tokens while FunCamera still outputs 41 tokens—needs alignment fix in the wrapper.
- **Next Step Identified**
  - Modify `wanvideo/fun_camera` node (or wrapper) so `control_camera_latents` downsample to the same token count as `image_cond` (21) before the transformer addition.

## TODOs
1. **Patch FunCamera Node**
   - In `ComfyUI-WanVideoWrapper/fun_camera/nodes.py`, after producing `control_camera_latents`, reshape/average tokens in groups of 2 to reduce from 41 → 21.
   - Ensure it works for both lightning (21 tokens) and full (41 tokens) by detecting the target token count based on `image_cond.shape[1]`.
2. **Verify Pipeline After Patch**
   - Re-run the 81-frame test prompt with `camTrack_v01_converted_v5.txt` and confirm sampling completes without the tensor-size error.
   - Document the wrapper change and any node parameters that need to be exposed for future adjustments.
3. **Optional**: expose camera stride/chunk parameters in ComfyUI UI so we can adjust without code edits.

## Notes for Later
- All pose TXT files live under `nuke_CAM_exporter/outputs/`. `v5` is the latest 81-frame version with width/height = 1248×704.
- Debug logs are under `nuke_CAM_exporter/debugging_issues/` (`bugs_reports_e.txt` contains the 21 vs 41 stack trace).
- Keep `wrapper_FIX/todo_and_resume.md` updated as we iterate on the FunCamera wrapper.
