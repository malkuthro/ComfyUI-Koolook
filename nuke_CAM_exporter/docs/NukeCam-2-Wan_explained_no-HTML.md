# Camera → Wan Pipeline Cheatsheet

This is the quick reference for translating DCC camera exports (Nuke / Houdini) into something Wan understands.

### Mental model

**Nuke / Houdini (Scanline, Karma, etc.)**
- Camera input: native camera node from the scene graph  
- Coordinate frame: whatever the scene uses  
- Intrinsics: stored on the camera node (filmback + focal)  
- Translation: camera center `C` in scene units  

**Wan video pipeline**
- Camera input: RealEstate10k TXT (`timestamp fx fy cx cy width height R\|t`)  
- Coordinate frame: fixed OpenCV axes (+X right, +Y down, +Z forward)  
- Intrinsics: requires normalized values (`fx = f_mm / sensor_w_mm`)  
- Translation: needs **world→camera** translation `t = –R · C` in meters 

Wan was trained on RealEstate10k, so its camera encoder only understands that schema. Anything else (raw ASCI, FBX, etc.) must be converted first.

## Why the converter is mandatory

1. **Different math objects**  
   - **Nuke’s ASCI** stores the camera center `(Tx, Ty, Tz)` plus ZXY Euler angles `(Rx, Ry, Rz)` in degrees. It tells you where the camera sits in world space and which way it’s pointing.  
   - **Wan / RealEstate** wants the inverse: a 3×4 matrix `[R|t]` that transforms any world point into the camera’s local space. In other words, “rotate + shift the world so the camera becomes the origin.”  
   - To bridge the two, we:  
     1. Convert the Euler angles to a rotation matrix `R` (standard Z→X→Y order).  
     2. Turn the camera center `C` into the translation part of the inverse matrix via `t = -R · C`.  
     The path doesn’t change—we simply express the same motion from the camera’s frame of reference so Wan’s math matches Nuke’s.
2. **Coordinate transforms**  
   - Nuke: +X right, +Y up, +Z forward.  
   - RealEstate: +X right, +Y down, +Z forward (OpenCV).  
   - Wan’s visualizer applies another transform for Matplotlib’s +Y-up space.  
   The converter reorients the axes so Wan’s `ray_condition()` can produce sensible rays.
3. **Units**  
   - Wan assumes translations are in meters.  
   - Nuke scenes often live in centimeters or abstract units, so we apply `unit_scale = real_distance / asci_delta` before writing the TXT.

## Practical workflow

1. **Export Nuke camera as ASCI** (columns `[Tx Rx Ty Ry Rz Tz]`).
2. **Measure real-world scale** so you can compute `unit_scale`.  
   Example: camera dolly moves 3.5 m, ASCI `Tz` delta is `0.11` ⇒ `unit_scale = 3.5 / 0.11 = 31.82`.
3. **Fill `configs/camera_config_*.json`** with:
   - Input/output paths
   - Lens data (focal + filmback) → converter derives `fx/fy/cx/cy`
   - Render resolution placeholders
   - `unit_scale` and optional frame caps
4. **Run** `python nuke_ASCI-2-Pose_converter.py --config configs/<shot>.json`
5. **Load TXT in ComfyUI** via `Koolook Load Camera Poses` → `WanVideoFunCameraEmbeds`.

Following this flow keeps the rendered motion aligned with the original plate, even though the raw numbers look different after conversion.***

## Where things happen inside the ComfyUI graph

1. **`Text Multiline` → `Koolook Load Camera Poses`**  
   - You paste the RealEstate10k TXT path.  
   - Koolook parses the header plus each row and returns `CAMERACTRL_POSES` (the same intrinsics + `[R|t]` data we exported).

2. **`WanVideo FunCamera Embeds`**  
   This node performs all the heavy lifting described in the converter:
   - `process_poses()` rescales intrinsics to match the render width/height.
   - `get_relative_pose()` turns each `[R|t]` row into camera-to-world matrices.  
   - `ray_condition()` converts those matrices into Plücker rays (origin + direction per pixel), the representation the Wan motion model was trained on.  
   - The tensor is reshaped and averaged to match Wan’s token budget (21 camera tokens per chunk) and the VAE stride.  
   - The result is a dictionary:
     ```python
     {
       "target_shape": (channels, t, h/8, w/8),
       "num_frames": len(poses),
       "control_embeds": {
         "control_camera_latents": …,
         "control_camera_start_percent": …,
         "control_camera_end_percent": …,
         ...
       }
     }
     ```
   - This payload is what the node outputs on `image_embeds`.

3. **`WanVideo ImageToVideo Encode`**  
   - Takes VAE/CLIP/start/end latents plus the `image_embeds` dict.  
   - Reads `control_embeds["control_camera_latents"]` and injects those camera rays into the diffusion network so every generated frame follows the exact path from Nuke.  
   - `target_shape` and `num_frames` keep the camera tokens aligned with the render resolution and animation length.

So all the conversions (intrinsics, `t = –R·C`, Plücker rays, token packing) happen before the encoder ever runs; by the time `image_embeds` hits Wan, it’s a set of “camera control instructions” telling the model where to point each ray for every frame.***

## Why Wan still drifts relative to the original plate

Even with perfect pose conversion, Wan isn’t a deterministic renderer—it’s still a diffusion model inventing geometry on the fly. Keep these limitations in mind:

- **Diffusion ≠ projection.** Wan reconstructs a scene that *statistically* fits the camera rays; it doesn’t ray-trace your tracked mesh, so pixels can land a little off compared to a Nuke render.
- **No shared 3D points.** Your tracked points in Nuke are actual world markers. Wan fabricates its own depth, so there’s nothing forcing it to align to those anchors.
- **Token downsampling.** Camera embeddings get compressed (Plücker rays → 21 tokens → VAE stride). That’s enough to guide global motion but not sub-pixel parallax.
- **Scale sensitivity.** Any tiny error in focal length, sensor size, or `unit_scale` shows up as sliding when you composite back with live-action elements.

### Tips when you need zero sliding

1. **Lock scale & intrinsics.** Double-check the measurement and lens metadata; millimeter accuracy matters.
2. **Treat Wan as a background pass.** Keep your foreground in Nuke and use Wan only for the new environment, then composite the two with projections/cards to fix residual offsets.
3. **Use camera projections for critical geometry.** Project the plate (or a painted matte) onto simple geo driven by your tracked camera, and blend that with Wan’s output.
4. **Post-warp if needed.** If the drift is small, track Wan’s render and apply a 2D/3D match-move or morph to align with the original plate.

Wan’s camera control ensures the *motion* matches; matching the *exact* pixel positions still requires traditional VFX techniques when you need perfectly locked foreground/background interaction.***

## Getting `unit_scale` and lens data dialed in

The most common reason for visible drift is a scale mismatch. Because Wan assumes the translations in `[R|t]` are in meters, a small error gets amplified—especially on wide lenses where foreground points travel further per degree of rotation.

### Why it matters

- **Focal length & filmback set the field of view.** If the RealEstate TXT says `fx = 33 mm / 36 mm = 0.916` but the real lens was 31 mm, Wan thinks the camera is *slightly tighter*. To match the same screen-space travel, it pushes the scene farther away, and your foreground doesn’t line up.
- **`unit_scale` scales the entire translation vector.** Doubling `unit_scale` doubles the apparent dolly move. On a 24 mm lens a 5 cm error is huge; on a 135 mm it’s barely noticeable.

### How to measure accurately

1. **Grab real production data whenever possible.** Use the on-set focal length and documented dolly track distance. Enter those directly into the config.
2. **Use Nuke’s reference tools.** In CameraTracker you can “Set Reference Distance” between two tracked points. If you tell it “this wall is exactly 4.2 m apart,” the entire camera solves into meters and `unit_scale` becomes 1.0.
3. **Manual measurement workflow:**
   - Pick two frames (start/end) and read the raw ASCI translation delta (e.g., Tz changes by 0.11842).  
     Formula:  
     ```
     unit_scale = real_world_distance_meters / abs(asci_delta)
     ```
     Example: camera travels 6.30 m, ASCI delta is 0.1891 ⇒ `unit_scale = 6.30 / 0.1891 = 33.31`.
   - Measure the real-world distance for that move (tape measure on set, LiDAR scan, or deduced from perspective grids).
   - Compute the ratio using the formula above. Keep at least 3 significant decimals.
   - Verify by projecting a checkerboard onto set geometry: if it still drifts, adjust by the measured percent difference.
4. **Wide lenses need tighter tolerances.** For anything 35 mm or wider, aim for centimeter-level accuracy. Every 1% error in scale equals ~1% sliding on screen.

Document the measurement in the config file (e.g. `"reference_distance_m": 6.53`). That way anyone regenerating the poses knows where the number came from and can double-check if the set changes.***