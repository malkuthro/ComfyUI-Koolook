# Run notes

## Maintainer feedback (OVERLAY - FEEDBACK, verbatim)

Base animation - reference

**Scores:** motion: 4/5 · sync: 5/5 · sharp: 4/5

## OVERLAY - INFO (verbatim)

    Phase 1 / Phase 2
    
    Format: HD / 2K
    
    Mult: 0.5 / 1
    
    Sheduler:
    Linear quadratic
    
    Steps: 16 / 8
    
    Denoise: 1 / 0.8
    
    BASE (notes):
    
    Changed Audio for each segment
    to match the duration

## Mechanical interp

- Director node: **upstream** `LTXDirector` (Koolook variant NOT wired — relay_overrides + per-segment σ are INERT this render)
- `use_custom_audio`: off — model-generated audio from `[Audio]:` prompt tag
- ε (epsilon): 0.001
- Duration: 120 frames @ 24 fps (120 sec)
- Director custom WxH: 3840x2160 (may be overridden by wired inputs upstream)
- Scheduler chain (BasicScheduler widgets): [['linear_quadratic', 8, 0.8], ['linear_quadratic', 16, 1]]
- Sampler picks: ['euler', 'euler']
- RandomNoise widgets: [[12, 'fixed']]
- CFGGuider widgets: [1, 1]
