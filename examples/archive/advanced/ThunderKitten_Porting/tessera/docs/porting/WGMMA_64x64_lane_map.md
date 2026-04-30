
# WGMMA 64×64×16 Lane→Fragment Map (Scaffold)

This map assigns each thread (lane) in a 128-thread warpgroup to **2×2** FP32 accumulators within a 64×64 output tile.
We partition the 64 columns into 4 **warp** slices of width 16: warp `w` handles columns `[16w, 16w+16)`.

Within each warp:
- Let `lane` be `0..31`.
- Define:
  - `row_block = (lane / 8) * 8`  (0,8,16,24 for lanes grouped by 8)
  - `col_block = (lane % 8) * 2`  (0,2,4,...,14)
- The thread contributes to a **2×2** micro-tile at:
  - rows: `row_block + {0, 1}`
  - cols: `16*w + col_block + {0, 1}`

Across a full 64×64 tile, we iterate `tm` and `tn` in steps of 64 and accumulate sub-tiles into the output register tile.

> **Verification**: `tools/porting/tests/wgmma_map_check.cu` runs the WGMMA path and a WMMA reference, then compares results.
> If mismatched, adjust `lane_fragment_map()` (in `wgmma_lane_map.cuh`) or replace with your empirically derived table.
