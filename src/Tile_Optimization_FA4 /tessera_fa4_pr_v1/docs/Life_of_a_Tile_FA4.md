# Life of a Tile in Tessera (FA‑4 Edition)

This tutorial walks a single attention tile through the warp-specialized pipeline:
1. Load → 2. MMA → 3. Softmax (online) → 4. Correction (thresholded) → 5. Epilogue.

## IR Snippet
```mlir
tessera.schedule @fa4_pipeline {
  %w_load  = tessera.schedule.warp "load", 1
  %w_mma   = tessera.schedule.warp "mma", 1
  %w_smx   = tessera.schedule.warp "softmax", 8
  %w_corr  = tessera.schedule.warp "correction", 4
  %w_epi   = tessera.schedule.warp "epilogue", 2
  tessera.schedule.pipe %w_load, %w_mma, %w_smx, %w_corr, %w_epi { buffering = {K=3, V=3, S=2, O=2} }
  tessera.schedule.policy "persistent", 1, "static"
}
```

## Numerics policy
```mlir
tessera.numerics.softmax "poly3", 2.0e-3
```

## TMEM & MMA
```mlir
%acc = "tessera.tile.mma.tcgen05"(%q_tmem, %k_tmem -> %acc_tmem, 2)
```

## Autotuning
Use `tools/autotune/schema_v2.json` to sweep pipeline shape and thresholds.
