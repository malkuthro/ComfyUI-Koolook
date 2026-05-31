# Audio-file-sync tests — run log

One row per render. Newest at the bottom.

Schema: per the audio-lipsync card design, columns are drawn from the
five tracked `Text Multiline` nodes + the `LTX Director (Koolook)`
node's own widgets and input wiring. No `BasicScheduler` /
`KSamplerSelect` / `RandomNoise` / `CFGGuider` scrapes.

`Audio src` reflects the five structural states derived from
Director presence + `audio_vae` link + `use_custom_audio` widget +
`timeline_data.audioSegments`: `(no director)` · `off (no VAE)` ·
`model-gen` · `custom` · `custom (empty)`.

`Segments` is logged as `Nv/Na`: video segment count and audio segment
count from `timeline_data.segments` / `timeline_data.audioSegments`.

| Run | Date | Director | `relay_overrides` | Audio src | Segments | Scores | Notes |
|---|---|---|---|---|---|---|---|
| 001 | 2026-05-28 | `LTXDirector` | `{"video_strength": 10.0}` | model-gen | 1v/0a | M4·S5·Sh4 | Base animation - reference (upstream LTXDirector — relay_overrides INERT) |
| 002 | 2026-05-29 | `LTXDirector__koolook_v1_3_2` | `{"video_strength": 10.0}` | model-gen | 1v/0a | M4·S5·Sh4 | Base animation - reference |
| 003 | 2026-05-29 | `LTXDirector__koolook_v1_3_2` | `{"video_strength": 10.0}` | custom | 1v/0a | M1·S0·Sh4 | Base animation - reference |
| 004 | 2026-05-30 | `LTXDirector__koolook_v1_3_2` | `{}` | custom | 2v/2a | M4·S1·Sh4 | Base animation - reference |
| 005 | 2026-05-31 | `LTXDirector__koolook_v1_3_2` | `{}` | model-gen | 1v/2a | M4·S3·Sh4 | Quest to find a BASE |
| 006 | 2026-05-31 | `LTXDirector__koolook_v1_3_2` | `{}` | model-gen | 1v/2a | M4·S3·Sh4 | Quest to find a BASE |
| 007 | 2026-05-31 | `LTXDirector` | `"video_strength": 10.0` | custom | 1v/2a | M4·S5·Sh4 | Quest to find a BASE (upstream Director; relay_overrides INERT) |
| 008 | 2026-05-31 | `LTXDirector__koolook_v1_3_2` | (empty → defaults) | custom | 3v/1a | M4·S5·Sh4 | 1x Audio + 3x Keyframes |
| 009 | 2026-05-31 | `LTXDirector__koolook_v1_3_2` | `"video_strength": 1.0` | model-gen | 3v/1a | M4·S5·Sh4 | 1x Audio + 3x Keyframes |
