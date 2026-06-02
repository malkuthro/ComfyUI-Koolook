# LTX Director (Koolook)

| Aspect | Value |
|---|---|
| Display name | `LTX Director (Koolook)` |
| Node ID | `LTXDirector__koolook` |
| Compatibility ID | `LTXDirector__koolook_v1_3_2` |
| ComfyUI category | `Koolook/PromptRelay` |
| Source | Koolook fork of WhatDreamsCost/WhatDreamsCost-ComfyUI |
| Source file | [`forks/whatdreamscost_koolook/versions/v1_3_9/ltx_director.py`](../../../../forks/whatdreamscost_koolook/versions/v1_3_9/ltx_director.py) |
| Requires | LTX/Prompt Relay dependencies used by the WhatDreamsCost workflow family |

## What it does

Provides a visual timeline editor for LTX Director workflows. Local prompts,
image guide segments, audio layout, segment lengths, guide strengths, and
Prompt Relay overrides are managed as timeline data and emitted in the shape
expected by downstream LTX/Prompt Relay nodes.

## When to use it

Use it for LTX 2.x workflows where a single global prompt is not enough and the
clip needs per-segment local prompts, guide images, or audio/timeline structure.

## Important inputs

| Input | Description |
|---|---|
| `global_prompt` | Conditions the whole video and anchors persistent characters, objects, and scene context. |
| `duration_frames` | Timeline visual scale in pixel-space frames. The actual generated frame count still comes from the latent path. |
| `duration_seconds` | Display duration in seconds, synced with the frame count. |
| `timeline_data` | JSON state owned by the timeline editor. Usually auto-managed; do not hand-edit unless recovering/debugging. |
| `use_custom_audio` | Uses timeline audio clips when on, or generated/default audio behavior when off. |
| `local_prompts` | Auto-populated segment prompt string from the editor. |
| `segment_lengths` | Auto-populated segment lengths in pixel-space frames. |
| `epsilon` | Prompt Relay penalty decay. Very low values produce sharp boundaries; higher values soften transitions. |
| `frame_rate` | Timeline display rate. Used for seconds display and output metadata. |
| `display_mode` | Shows the editor in frames or seconds. Internal storage remains frame-based. |
| `guide_strength` | Auto-populated guide strengths for image segments. |
| `custom_width` / `custom_height` | Optional target dimensions for image guide segments. `0` uses original dimensions. |
| `resize_method` | How image guide segments fit the target dimensions. |
| `divisible_by` | Snaps guide dimensions to a model-friendly multiple such as `32`. |
| `img_compression` | H.264 CRF applied to guide images. `0` is no compression; higher values add more artifacts. |
| `relay_overrides` | Optional JSON dict for per-stream Prompt Relay overrides. |
| `audio_transcript_json` | Optional timed speech transcript JSON used to build local prompts and segment timing. |

## Outputs

| Output | Description |
|---|---|
| `model` | Model pass-through/patch output for downstream LTX nodes. |
| `positive` | Prompt Relay conditioning. |
| `video_latent` | Auto-generated LTXV empty latent when no optional latent is connected. |
| `audio_latent` | Auto-generated audio latent when audio VAE/custom audio is active. |
| `guide_data` | Image/audio guide data for downstream `LTXDirectorGuide`. |
| `frame_rate` | Frame rate used by the timeline. |
| `combined_audio` | Combined timeline audio layout. |

## Recipes

### Segment a clip by action

Use `global_prompt` for stable scene context, then add timeline segments such as:

```text
0-30: character enters the room
31-70: character notices the camera
71-120: character turns and walks away
```

The editor writes `local_prompts` and `segment_lengths`; downstream Prompt Relay
uses those segments to condition each region of the video.

### Add a transcript-driven timing pass

Feed a transcript JSON through `audio_transcript_json` when speech timing should
drive the local prompt segments. Keep `timeline_data` auto-managed unless you are
recovering a saved state.

This is a built-in input on `LTXDirector__koolook`. The earlier standalone
`KoolookAudioTranscriptTimeline` / `KoolookTimelineEditor` helper nodes were
experimental and have been removed.

## Caveats

- `LTXDirector__koolook_v1_3_2` is a saved-workflow compatibility alias backed
  by the current v1.3.9 implementation.
- This node is tied to the LTX/Prompt Relay ecosystem. If those external nodes
  are missing, workflows using it will not be complete.
- The timeline editor owns several string fields. Treat them as generated state
  unless you are intentionally debugging or restoring a workflow.
