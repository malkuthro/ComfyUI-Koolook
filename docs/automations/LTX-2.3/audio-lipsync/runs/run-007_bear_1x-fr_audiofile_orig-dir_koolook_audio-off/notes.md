# Run notes

## Maintainer feedback (OVERLAY - FEEDBACK, verbatim)

Quest to find a BASE

**Scores:** motion: 4/5 · sync: 5/5 · sharp: 4/5

## OVERLAY - INFO (verbatim)

    Phase 1

    Format: 540p

    Mult: 0.5

    Sheduler:
    Linear quadratic

    Steps: 16

    Denoise: 1

    ---

    BASE: Start video = OK

## SETUP variables (captured)

- Capture run number: `007 (from runs folder/log)`
- Setup name: `LTX-23-audio_tests_03`
- Image segments: `1 video / 2 audio`
- Prompts / similarity: `single`
- Commit No.: `3294e6e`
- LTX-Director (flavour): `Original upstream (LTXDirector)`
- Director pin tag: `v1.3.2`
- Audio src: `custom audio ON`
- epsilon: `0.001`
- Duration: `144 frames @ 24 fps (6 sec)`
- RELAY_OVERRIDES: `"video_strength": 10.0` (inert: active Director is upstream LTXDirector)
- GLOBAL [ path ] - working folder: `e:\G-Drive-BaconX\Jobs\Jeep_Animals\ComfyUI_LTX23\Runs-Audio-File`
- GLOBAL [ base name ]: `Bear_1x-FR_AudioFile_ORIG-Dir`
- INPUT Path [ EXR ]: `W:/projects/samsung_goat_3188/vfx/assets/AI_renders/001/1130/messi/video/v003/`
- GLOBAL [ version ]: `1`
- GLOBAL [ run offset ]: `6`
- Output folder: `e:\G-Drive-BaconX\Jobs\Jeep_Animals\ComfyUI_LTX23\Runs-Audio-File`
- Output name: `Bear_1x-FR_AudioFile_ORIG-Dir_h264_v001`
- OVERLAY - INFO: `(captured verbatim above)`
- OVERLAY - FEEDBACK: `(captured verbatim above)`

## Director node — structural state

- Variant: **upstream** `LTXDirector` (Koolook variant NOT wired - relay_overrides + per-segment sigma are INERT this render)
- Audio src: custom
- ε (epsilon): 0.001
- Duration: 144 frames @ 24 fps (6 sec)
- Segments: 1 video / 2 audio
- Segment prompt mode: single
