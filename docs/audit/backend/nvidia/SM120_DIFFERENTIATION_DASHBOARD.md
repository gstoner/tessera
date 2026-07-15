---
last_updated: 2026-07-13
audit_role: reference
---

# NVIDIA sm_120 differentiation closeout

Updated 2026-07-13 from the RTX 5070 Ti WSL CUDA host. “Promoted” requires all
six evidence columns; an assembling instruction is not runtime evidence.

| Lane | Typed IR + verifier | Direct device comparison | Provenance / ABI | Smoke benchmark | Matrix/dashboard | Status |
|---|---|---|---|---|---|---|
| f16 `mma.sync` fused GEMM + epilogue | `sm120_differentiation_target_ir.mlir` | `test_live_nvidia_mma_fused_tensor_core` | `nvidia_cuda`; production candidate C ABI | `mma_sync_fused`, 0.917 ms median | baseline + this dashboard | **promoted** |
| f16 `mma.sync` attention | `sm120_differentiation_target_ir.mlir` | `test_live_nvidia_mma_attn_tensor_core` | `nvidia_cuda`; production candidate C ABI | `mma_sync_attention`, 0.642 ms median | baseline + this dashboard | **promoted** |
| FP8 E4M3/E5M2 storage conversion, f32 compute | `tessera_nvidia.fpquant` | `test_quantize_grid_matches_reference` | `nvidia_fpquant_compiled`; `native_gpu` | `cuda_fpquant`, 1.051 ms median | execution matrix + baseline | **promoted (storage)** |
| FP6 E2M3/E3M2 storage conversion, f32 compute | `tessera_nvidia.fpquant` | `test_quantize_grid_matches_reference` | `nvidia_fpquant_compiled`; `native_gpu` | `cuda_fpquant`, 1.024 ms median | execution matrix + baseline | **promoted (storage)** |
| NVFP4 E2M1 + UE4M3 block-scale MMA | `tessera_nvidia.nvfp4_block_scale_mma`; PTX assembles | **passes** fixed-tile unit and non-uniform scale oracle on sm_120a (`test_nvidia_nvfp4_compiled.py`) | no general-shape runtime ABI | none | execution proof is complete; runtime productization remains | **blocked at runtime-dispatch gate** |

The FP8/FP6 rows do not claim tensor-core MMA. Dense FP8/FP6 `mma.sync` is a
separate future promotion and must independently satisfy this same gate.

Measured baseline: `benchmarks/baselines/nvidia_sm120_hot_paths.json`, 10 timed
iterations after three warmups; timings include the existing host-pointer ABI's
allocation and copies. Thresholds are 2× median to tolerate CI noise.
