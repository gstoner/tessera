// Tessera Tile IR → NVIDIA Tile IR (experimental) lowering skeleton.
// This file deliberately avoids MLIR headers; it documents intended patterns.

/*
Conceptual mapping (informative):

Tessera Tile IR ops:
  - tile.load/store (global/shared), tile.zero
  - mma.* (fp16/bf16/int8 accumulators)
  - barrier.tile, reduce.*
  - config.tma(...), async.copy

NVIDIA Tile IR (hypothetical public surface for NVIDIA's tile-level ops):
  - nvtile.tma.create_descriptor / nvtile.tma.async_copy.2d/.3d
  - nvtile.wgmma.mma_async (warp-group MMA on Hopper+)
  - nvtile.mbarrier.arrive/try_wait
  - nvtile.smem.alloc / nvtile.reg.alloc
  - nvtile.cp.async.bulk.tensor.* (PTX lowering hooks)

Lowering outline:
  1) For tensor-core GEMMs on sm_90+:
     - Prefer nvtile.wgmma.mma_async with m64n128k16 (bf16), or m64n64k16 variants as available.
     - Use nvtile.tma.async_copy to stage tiles into shared memory (or direct wgmma from TMA).
     - Synchronize with mbarrier for cp.async and wgmma streams.

  2) For sm_80..sm_89:
     - Map to nvgpu.mma.sync (WMMA) and device_async_copy (cp.async) with barriers.

  3) Fallback:
     - Call prebuilt WMMA kernels (see src/kernels/wmma_gemm_*.cu).

Notes:
  - This file is a design scaffold. Implement as MLIR patterns in your pass pipeline.
  - Select tile shapes via attributes on tessera.mma ops (e.g., m/n/k, dtype) to choose wgmma variant.
*/
