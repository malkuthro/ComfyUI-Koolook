# Audio-file-sync tests â€” run log

One row per render. Newest at the bottom.

Schema: per the audio-lipsync card design, columns are drawn from the
five tracked `Text Multiline` nodes + the `LTX Director (Koolook)`
node's own widgets and input wiring. No `BasicScheduler` /
`KSamplerSelect` / `RandomNoise` / `CFGGuider` scrapes.

`Audio src` reflects the five structural states derived from
Director presence + `audio_vae` link + `use_custom_audio` widget +
`timeline_data.audioSegments`: `(no director)` Â· `off (no VAE)` Â·
`model-gen` Â· `custom` Â· `custom (empty)`.

`Segments` is logged as `Nv/Na`: video segment count and audio segment
count from `timeline_data.segments` / `timeline_data.audioSegments`.

| Run | Date | Director | `relay_overrides` | Audio src | Segments | Scores | Notes |
|---|---|---|---|---|---|---|---|
| 001 | 2026-05-28 | `LTXDirector` | `{"video_strength": 10.0}` | model-gen | 1v/0a | M4·S5·Sh4 | Base animation - reference (upstream LTXDirector — relay_overrides INERT) |
| 002 | 2026-05-29 | `LTXDirector__koolook_v1_3_2` | `{"video_strength": 10.0}` | model-gen | 1v/0a | M4·S5·Sh4 | Base animation - reference |
| 003 | 2026-05-29 | `LTXDirector__koolook_v1_3_2` | `{"video_strength": 10.0}` | custom | 1v/0a | M1·S0·Sh4 | Base animation - reference |
| 004 | 2026-05-30 | `LTXDirector__koolook_v1_3_2` | `{}` | custom | 2v/2a | M4·S1·Sh4 | Base animation - reference |
