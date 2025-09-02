# GPU Backend Overview

This backend targets NVIDIA GPUs via MLIR NVGPU/NVVM or plain PTX/NVRTC. It includes a runtime with CUDA Driver API to load PTX or cubins and launch kernels.

## Runtime feature detection
- Queries device SM version, Tensor Core availability, bf16 support (sm80+), etc.
- Provides helpers to JIT PTX with NVRTC (optional) and to load modules via `cuModuleLoadDataEx`.

## Kernels
- WMMA FP16/BF16 GEMM examples with bounds checks (work for non-multiple sizes, albeit with overhead).

## Integration
- Register `TesseraGpuBackend` as a target in your compiler; route Tile IR `mma` ops through the NVGPU `mma.sync` builder or to pre-built WMMA kernels when appropriate.


## NVIDIA Tile IR (experimental)

When targeting Hopper+ (SM90+), prefer **NVIDIA Tile IR**: TMA + WGMMA + mbarrier for bulk tensor copies and warp‑group MMA.
This backend ships a **lowering skeleton** (`src/lowering_nvtile_mlir.cpp`) and a **guarded PTX kernel** placeholder
(`src/kernels/ptx_wgmma_bf16.cu`) to help wire up a real path once the MLIR dialect and PTX details are integrated.
