// Placeholder for Tessera Tile IR → NVGPU/NVVM lowering.
// Register these patterns in your compiler; this file is freestanding (no MLIR includes) on purpose.

/*
Pipeline sketch:
  - Hoist shared mem allocs; create gpu.shared memory buffers
  - Tile M/N/K loops to 16x16x16 fragments; map warps to tiles
  - Convert tile.mma ops to `nvgpu.mma.sync` builder calls (fragment types)
  - Replace async copies with `nvgpu.device_async_copy` / Hopper `nvgpu.tma.async_load`
  - Convert gpu/nvgpu -> nvvm, then llvm nvptx

Pseudo‑patterns:

  Pattern: tile.mma.fp16.acc.fp32(A,B,C)
    if (sm >= 70):
      build nvgpu.mma.sync with m16n16k16 f16->f32 fragments
    else:
      fallback to wmma kernel call or vectorized fma loop

  Pattern: tile.copy.global_to_shared(tile, gptr, sptr, shape, stride)
    -> nvgpu.device_async_copy(sptr, gptr, shapeBytes), then barrier

  Pattern: barrier.tile
    -> gpu.barrier (bar.sync 0)

  Pattern: reduce.add
    -> nvgpu.warpgroup.reduce / nvvm.shfl.sync.down loop

  Config:
    module attr sm_XX controls selection BF16 vs FP16 vs INT8 paths
*/
