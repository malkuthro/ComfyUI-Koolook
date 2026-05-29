# Audio-file-sync tests — run log

One row per render. Newest at the bottom.

Schema: per the audio-lipsync card design, columns are drawn from the
five tracked `Text Multiline` nodes + the `LTX Director (Koolook v1.3.2)`
node's own widgets and input wiring. No `BasicScheduler` /
`KSamplerSelect` / `RandomNoise` / `CFGGuider` scrapes.

`Audio src` reflects the four structural states derived from
`audio_vae` link + `use_custom_audio` widget + `timeline_data.audioSegments`:
`model-gen` · `custom` · `custom (empty)` · `off (no VAE)`.

| Run | Date | Director | `relay_overrides` | Audio src | Segments | Scores | Notes |
|---|---|---|---|---|---|---|---|
| 001 | 2026-05-28 | `LTXDirector` | `{"video_strength": 10.0}` | model-gen | 1 | M4·S5·Sh4 | Base animation - reference (upstream LTXDirector — relay_overrides INERT) |
| 002 | 2026-05-29 | `LTXDirector__koolook_v1_3_2` | `{"video_strength": 10.0}` | model-gen | 1 | M4·S5·Sh4 | Base animation - reference |
