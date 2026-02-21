# Heatmap Arc Experiments Log

## Scope
This log records the recent debugging and iteration history for:
- `MICCAI_FINAL/heatmap/train_arc.py`
- `MICCAI_FINAL/heatmap/infer.py`

Goal: improve left/right arc consistency and realism while keeping stable overall predictions.

## Problem Observed
- Arc lines (`left_arc`, `right_arc`) were frequently:
  - swapped left/right
  - crossing each other
  - endpoints coming from opposite arcs
  - collapsing to one side of image
- Horizontal flip augmentation likely introduced side-label ambiguity for arcs.

## Infer-Time Changes (Post-processing)
Implemented in `MICCAI_FINAL/heatmap/infer.py`:

1. Arc correction pipeline:
- enforce left/right ordering by x-center
- minimum arc separation constraint
- move arc to opposite side if both arcs on same side
- side anchoring (incisor-guided for mandible)
- top endpoint separation guard
- per-arc verticality constraint (`|dx|` bounded vs `|dy|`)
- endpoint re-pairing across arcs (top-left/bottom-left, top-right/bottom-right)

2. Scale rule:
- enforce `abs(scale_dx) <= 15 px`
- enforce minimum scale vertical separation

3. Jaw-aware indexing:
- maxilla arc lines: `1,2`
- mandible arc lines: `5,6`

4. Path alignment for arc run outputs:
- input: `heatmap_results_arc/test_predictions.json`
- outputs:
  - `hrnet_results_arc/visualizations`
  - `hrnet_results_arc/test_predictions_corrected.json`

## Train-Time Changes (`train_arc.py`)

### A. Initial arc-focused version
- weighted heatmap loss (higher weights for arc points)
- geometry regularization via differentiable soft-argmax:
  - scale horizontal spread penalty
  - arc ordering/separation penalties
  - arc verticality penalties
- export-time correction aligned with infer logic
- separate save dir: `heatmap_results_arc`

### B. Bug fix
- fixed validation normalization mismatch:
  - val mask now expanded to heatmap resolution before weighted normalization
  - made train/val loss scales consistent

### C. Stability rollback (after catastrophic errors)
- set `MAX_LAMBDA_GEOM = 0.0` (disable geometry term)
- reduced `ARC_POINT_WEIGHT` to gentler value
- added warm-start from baseline model:
  - `INIT_MODEL_PATH = heatmap_results_1/heatmap_best.pth`

Result: stable training restored.

## Key Metric Snapshots

### Earlier baseline reference
```
Overall Mean Line Pixel Error: 14.128
```

### After infer-only correction stage
```
Overall Mean Line Pixel Error: 10.940
```

### Failed unstable arc-training attempt
```
Overall Mean Line Pixel Error: 106.402
```

### After stability fixes (warm-start + geometry off)
```
Overall Mean Line Pixel Error: 11.322
```

### Best recent run reported
```
scale: 9.558
incisor1: 11.606
incisor2: 3.971
incisor3: 3.562
incisor4: 11.618
left_arc: 9.809
right_arc: 19.082
Overall Mean Line Pixel Error: 10.590
```

## Current Practical Status
- Visual outputs are reported as clinically/qualitatively good.
- Metric still shows right-side arc asymmetry (`right_arc` > `left_arc`).
- Recommended next low-risk step (if continuing):
  - set `MAX_LAMBDA_GEOM = 0.001`
  - keep warm-start and current stability settings
  - compare right-arc error + visual regressions only

## Notes
- If horizontal flip augmentation is used upstream, arc labels must be swapped in mirrored samples (`left_arc <-> right_arc`) to avoid persistent side confusion during training.
