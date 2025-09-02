<!-- MERGE_BEGIN -->
# Tessera Tile IR → NVGPU/NVVM/PTX Mapping

This cheat-sheet outlines how Tessera **Tile IR** ops lower to MLIR NVGPU/NVVM and, ultimately, PTX.

| Tessera Tile IR | MLIR NVGPU | MLIR NVVM / LLVM | PTX / SASS | Notes |
|---|---|---|---|---|
| `tile.load(tile, ptr, stride)` | `nvgpu.device_async_copy` (shared) / vector `memref.load` | `llvm.load` + addrspaces | `ld.global`, `ld.shared` | Use addrspaces 1 (global), 3 (shared). |
| `tile.store(tile, ptr, stride)` | `nvgpu.device_async_wait` + `memref.store` | `llvm.store` | `st.global`, `st.shared` | Hopper uses TMA via `nvgpu.tma.async_load` for 2D tiles. |
| `tile.zero` | vector.fill | `llvm.memset` | `memset`/`st` loop | Zero shared/regs. |
| `mma.fp16.acc.fp32` | `nvgpu.mma.sync` | `nvvm.mma.sync` | `mma.sync.aligned.m16n16k16.row.col.f16.f16.f32` | Tensor Core HMMA FP16. |
| `mma.bf16.acc.fp32` | `nvgpu.mma.sync` | `nvvm.mma.sync` | `mma.sync.aligned.m16n16k16.row.col.bf16.bf16.f32` | Ampere+ BF16 HMMA. |
| `mma.s8.acc.s32` | `nvgpu.mma.sync` (IMMA) | `nvvm.mma.sync` | `mma.sync.aligned.m16n16k16.row.col.s8.s8.s32` | INT8 Tensor Cores. |
| `reduce.add` | `nvgpu.warpgroup.reduce` | `nvvm.shfl.sync`/`llvm.intrinsics` | `shfl.sync` | Warp/wargroup reductions. |
| `barrier.tile` | `gpu.barrier` / `nvgpu.mbarrier.arrive/try_wait` | `nvvm.barrier0` | `bar.sync` | For shared memory tile sync. |
| `config.tma(...)` | `nvgpu.tma.create_descriptor` | `nvvm.cp.async.bulk.tensor.*` | `cp.async.bulk.tensor.*` | Hopper TMA copies (2D/3D). |

**Lowering pipeline (typical):**

```
Tessera Tile IR
  → (tiling, vectorization, smem alloc)
  → MLIR gpu + nvgpu (mma/tma/shmem)
  → NVVM dialect (nvvm.* ops)
  → LLVM NVPTX
  → PTX (via llc) → SASS (via `ptxas`) → cubin
```

**Tile shapes** (defaults):
- HMMA FP16/BF16: 16×16×16
- IMMA S8: 16×16×64 (macro-k tiles)

<!-- MERGE_END -->


## NVIDIA Tile IR (experimental)

| Tessera Tile IR | NVIDIA Tile IR | PTX | Notes |
|---|---|---|---|
| `mma.bf16.acc.fp32` | `nvtile.wgmma.mma_async` | `wgmma.mma_async.*.bf16.bf16.f32` | Hopper+ warp-group MMA |
| `async.copy.2d` | `nvtile.tma.async_copy.2d` | `cp.async.bulk.tensor.2d` | Uses TMA descriptor |
| `barrier.tile` | `nvtile.mbarrier.*` | `mbarrier.*` / `bar.sync` | Cooperative sync |
