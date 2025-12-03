# Camera → Wan Pipeline Cheatsheet

This is the quick reference for translating DCC camera exports (Nuke / Houdini) into something Wan understands.

## Mental model

<table>
  <thead>
    <tr>
      <th>Renderer</th>
      <th>Camera input</th>
      <th>Coordinate frame</th>
      <th>Intrinsics</th>
      <th>Translation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Nuke / Houdini (Scanline, Karma, etc.)</td>
      <td>Native camera node from the scene graph</td>
      <td>Whatever the scene uses</td>
      <td>Stored on the camera node (filmback + focal)</td>
      <td>Camera center <code>C</code> in scene units</td>
    </tr>
    <tr>
      <td>Wan video pipeline</td>
      <td>RealEstate10k-style TXT (<code>timestamp fx fy cx cy width height R&#124;t</code>)</td>
      <td>Fixed OpenCV-style axes (+X right, +Y down, +Z forward)</td>
      <td>Requires normalized values (<code>fx = f_mm / sensor_w_mm</code>)</td>
      <td>Needs <strong>world→camera</strong> translation <code>t = –R · C</code> in meters</td>
    </tr>
  </tbody>
</table>

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

### Quick visual QA (top-view plot)

Whenever you need to confirm that the scaled translations and rotations still match what you see in Nuke, use the helper plot:

```
cd nuke_CAM_exporter
py -3 testing_tools/plot_top_view.py \
    --config configs/camera_config_v01.json \
    --output testing_tools/top_view.png
```

What it shows:

- Green = scaled Nuke camera centers (after `unit_scale` and frame slicing).
- Purple = world→camera translations from the RealEstate TXT (mostly for debugging; the green curve is what matches Nuke).
- Blue cones/arrows = camera forward vectors from the ASCI rotation channels (first/last frames), rendered in the same “view from –Y” orientation you see in Nuke’s top view.
- A 1×1 m square centered on the origin with the +Z / –X quadrant shaded so you know at a glance which way the camera is pointing relative to the plate.

As long as the green curve and blue cones match your Nuke viewport, the exporter is doing the right thing—any remaining mismatch in a ComfyUI render is down to Wan’s hallucinated geometry, not the pose math.

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

One way to sanity-check where things diverge is the top-view plot described above. The green curve (Nuke camera center after scaling) is the only trajectory you can compare directly to the plate; the purple `t` values are world→camera translations that Wan later inverts, so they don’t need to “look” like the Nuke path. If the green curve aligns with the tracked move, the export is correct. Any remaining X drift—like the character sliding toward the right wall in the render—is just Wan prioritising a pleasing composition over strict lateral translation, especially on wide dolly moves.

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

### Can Houdini export the “right” data directly?

Even if your Houdini scene already uses meters, the RealEstate schema still applies. Houdini’s `optransform("/obj/cam1", "/obj/world")` gives you a 4×4 world→camera matrix, but you still need to:

1. Normalize intrinsics (`fx = focal_len / aperture`, etc.).
2. Convert the matrix into `[R|t]` flattened rows with `t = –R·C` (or just invert the world transform you already have).
3. Emit timestamps + metadata (fps, width, height) in the RealEstate order.

If you want to script it inside Houdini, the following Python SOP snippet mirrors the converter logic:

```python
import math, json
from pathlib import Path

node = hou.node("/obj/cam1")
fps = hou.fps()
f_length = node.parm("focal").eval()
aperture = node.parm("aperture").eval()  # horizontal mm
sensor_h = node.parm("apertureh").eval() or aperture * (9/16)
fx = f_length / aperture
fy = f_length / sensor_h
cx, cy = 0.5, 0.5

out_lines = []
for frame in range(int(start), int(end) + 1):
    hou.setFrame(frame)
    m = node.worldTransform().inverted()  # world→camera
    R = m.extractRotationMatrix3()
    t = m.extractTranslates("trs")
    row = [
        (frame - start) * (1e6 / fps),
        fx, fy, cx, cy,
        width, height,
        R.at(0,0), R.at(0,1), R.at(0,2), t[0],
        R.at(1,0), R.at(1,1), R.at(1,2), t[1],
        R.at(2,0), R.at(2,1), R.at(2,2), t[2],
    ]
    out_lines.append(" ".join(f"{v:.12f}" for v in row))

Path("$HIP/poses/cam_track.txt").write_text(
    f"source:houdini fps:{fps} fx:{fx} fy:{fy} cx:{cx} cy:{cy} width:{width} height:{height}\n" +
    "\n".join(out_lines)
)
```

This is effectively re-implementing `nuke_ASCI-2-Pose_converter.py`. Unless you need it embedded in Houdini, it’s simpler to export a standard ASCI/FBX and run the converter with `unit_scale=1.0`.***

## Where CameraCtrl fits vs. UniCtrl / Uni3C

<table>
  <thead>
    <tr>
      <th></th>
      <th>CameraCtrl (AnimateDiff / Wan camera encoder)</th>
      <th>UniCtrl / Uni3C (Xuweiyi Chen et al., 2024)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Primary job</strong></td>
      <td>Consume explicit camera trajectories (<code>fx/fy/cx/cy</code>, <code>[R|t]</code>) and tell the motion model where the camera is each frame.</td>
      <td>Modify the diffusion UNet’s attention during sampling so adjacent frames share semantics and motion stays coherent.</td>
    </tr>
    <tr>
      <td><strong>Input requirements</strong></td>
      <td>RealEstate10K-style TXT + calibrated lens metadata from Nuke/Houdini/RealityCapture.</td>
      <td>None beyond the base prompt/image conditioning; purely inference-time.</td>
    </tr>
    <tr>
      <td><strong>Training</strong></td>
      <td>Needs a camera encoder / motion LoRA trained on RealEstate10K (AnimateDiff, Wan 14B, etc.).</td>
      <td>Training-free; plug-and-play on top of AnimateDiff / VideoCrafter / ModelScope weights.</td>
    </tr>
    <tr>
      <td><strong>What it guarantees</strong></td>
      <td>Camera <em>path</em> matches the supplied track (dolly, pan, roll, etc.); scene content still hallucinated.</td>
      <td>Better spatiotemporal consistency for the hallucinated content; does not enforce a specific path.</td>
    </tr>
    <tr>
      <td><strong>Failure mode</strong></td>
      <td>Wrong poses / scale ⇒ camera mismatch, even if UniCtrl is perfect.</td>
      <td>Still a generative scene; cannot force FG objects to stick to plate geometry.</td>
    </tr>
    <tr>
      <td><strong>Best use</strong></td>
      <td>Matching VFX plates, I2V backgrounds, virtual production where the camera move must match live action.</td>
      <td>Any T2V/I2V run where you want less flicker without retraining; works even with pure text prompts.</td>
    </tr>
  </tbody>
</table>

### How they’re interconnected

1. **Pipeline order:** Nuke/Houdini → exporter → RealEstate TXT → CameraCtrl nodes (`Koolook Load Camera Poses`, `WanVideo FunCamera Embeds`) → Wan encoder.  
2. **Optional stabilization:** While sampling, UniCtrl hooks into the UNet attention blocks and reuses the first frame’s keys/values, injects motion queries, and syncs latents so subsequent frames stay semantically aligned.  
3. **Outcome:** CameraCtrl keeps the *geometry* aligned with your tracked move. UniCtrl keeps the *diffusion process* from drifting frame-to-frame. Together you get the closest thing to a “virtual camera” that respects both the live-action motion and temporal coherence.

In short: CameraCtrl is the equivalent of importing your tracked Steadicam into a renderer; UniCtrl is more like an attention-based stabilizer that reduces flicker. Use CameraCtrl whenever you need Wan to obey a real camera path, and layer UniCtrl on top when the diffusion output itself needs extra temporal polish.