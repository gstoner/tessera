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

**Root cause of the NVFP4 "runtime-dispatch gate" (2026-07-17, external
corroboration).** The accurate framing is **not** "native FP4 unavailable" —
**sm_120 has native block-scaled FP4 Tensor Core execution** via warp-level
`mma.sync.aligned.kind::mxf4nvf4.block_scale`. The gate is a **software
kernel-selection gap**: sm_120 has no `tcgen05`/`TMEM`, so an SM100 CUTLASS
grouped-GEMM tactic **cannot be retargeted** and fails init; frameworks (vLLM /
FlashInfer / TensorRT-LLM) then fall back to **Marlin W4A16/W4A8-FP8** (dequant —
correct, not native-FP4 throughput). **TMA *does* exist on sm_120** and feeds a
sm_120-specific mainloop — `TMA → 99-KB SMEM → ldmatrix/regs → mma.sync.block_scale`
— under the CUTLASS SM120 schedules `KernelTmaWarpSpecialized{Cooperative,Pingpong}`.
SM120 constraints (NVIDIA [CUTLASS 4.4.1 SM120 doc](https://docs.nvidia.com/cutlass/4.4.1/media/docs/cpp/blackwell_functionality.html)):
**TN layout only** (A row-major, B col-major), **cluster fixed 1×1×1** (no
multicast), `EpilogueScheduleAuto`, tile `128×128×128`; `kind::` variants
`mxf8f6f4` / `mxf4` / `mxf4nvf4`. This is exactly `tessera_nvidia.nvfp4_block_scale_mma`,
and why the directly-emitted route (Decision #28: the generic library path does
not even run) is correct. **Selector rule:** the native sm_120 route and the
Marlin fallback are **separate candidates**; prove the native instruction route
and report Marlin explicitly — **a successful W4A16 must never be recorded as
native NVFP4** (the FP4 analog of "`reference_cpu` cannot earn `native_gpu`").
Reference kernels + hazards: `lna-lab/blackwell-geforce-nvfp4-gemm` (reports
*working native NVFP4*, not Marlin fallback), `VincentKaufmann/fp4-cuda-kernel`
(**the interleaved UE4M3 scale layout, CuTe atom `((32,4),(16,4))`, must be exact
— a wrong layout corrupts ~10% of outputs**, the key general-shape-ABI risk; ~1.2%
mean rel. error; native FP4 wins M≤2048, bf16 cuBLAS wins M=4096). Consumed by the
Sequence Mixer plan's W6 ([`../../compiler/SEQUENCE_MIXER_ENGINEERING_PLAN.md`](../../compiler/SEQUENCE_MIXER_ENGINEERING_PLAN.md)).

Measured baseline: `benchmarks/baselines/nvidia_sm120_hot_paths.json`, 10 timed
iterations after three warmups; timings include the existing host-pointer ABI's
allocation and copies. Thresholds are 2× median to tolerate CI noise.
