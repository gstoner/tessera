
# Tessera Porting Patch v11.1 — WGMMA 64×64 Lane Map, Real TMA, Epilogues, Autotune
**Timestamp:** 2025-09-02 19:37

This patch delivers:
- A lane→fragment map scaffold for **WGMMA m64n64k16 (BF16→F32)** with a built-in **verification harness**.
- **Epilogues** (bias, SiLU, GELU) for BF16/FP8 accumulation paths.
- A **real TMA descriptor** flow (Hopper/Blackwell) via `tensormap.create` + `cp.async.bulk.tensor.2d.shared::cluster.global` + `mbarrier.try_wait.parity` (guarded).
- **Autotune** parameters for **swizzle choice** and **TMA row chunking (`cols_per_copy`)**, plumbed end-to-end.

> The WGMMA lane map used here is a **documented scaffold**: it partitions the 64×64 tile into 4 warp slices (16 columns each).
> Within each warp, lanes write 2×2 micro-tiles distributed across rows, with formulas that align to common layouts seen in practice.
> The included **verification harness** compares against a reference WMMA path to validate correctness on your target driver/GPU.
> If the check fails, you can adjust the `lane_fragment_map()` formulas or drop in a table discovered from your own micro-probing.

## New/Modified Flags
- Build-time:
  - `-DTESSERA_USE_WGMMA` / `-DTESSERA_REAL_WGMMA`
  - `-DTESSERA_USE_TMA` / `-DTESSERA_REAL_TMA`
  - `-DTESSERA_WGMMA_USE_IDENTITY_SWIZZLE=ON|OFF`
- Runtime (CLI / Makefile):
  - `--compute_path wmma|wgmma`
  - `--copy_path cp.async|tma`
  - `--epilogue none|bias|bias_silu|bias_gelu`
  - `--swizzle identity|xor128b`
  - `--tma_cols_per_copy <int>`

## Files touched
- `tessera/include/wgmma_lane_map.cuh` (NEW): lane→fragment scatter for 64×64.
- `tessera/include/epilogue_ops.cuh` (NEW): bias/SiLU/GELU epilogues.
- `tessera/include/tma_real.cuh` (NEW): `tensormap.create` + `cp.async.bulk.tensor.2d.*` + `mbarrier` helpers.
- `tessera/tools/porting/microbench/microbench_main.cu`: WGMMA 64×64 scatter, epilogues, CLI, TMA cols_per_copy.
- `tessera/tools/porting/autotune/hyperband.py`: adds `swizzle` and `tma_cols_per_copy` knobs.
- `tessera/docs/porting/WGMMA_64x64_lane_map.md` (NEW): explains mapping with formula and examples.
- `tessera/tools/porting/tests/wgmma_map_check.cu` (NEW): verification harness.

See `WGMMA_64x64_lane_map.md` for details and the exact mapping equations.
