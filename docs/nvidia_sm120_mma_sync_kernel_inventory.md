---
status: Informative
classification: Reference / Kernel Inventory
authority: Companion to docs/nvidia_cuda13_kernel_inventory.md
last_updated: 2026-07-07
---

# NVIDIA sm_120 (consumer Blackwell) mma.sync Kernel Inventory

> Hardware-verified companion to
> [`docs/nvidia_cuda13_kernel_inventory.md`](nvidia_cuda13_kernel_inventory.md)
> (which is the SM_90+ WGMMA *planning* inventory). This doc enumerates the
> kernels Tessera actually **synthesizes, compiles, and runs on sm_120** — proven
> on an RTX 5070 Ti (compute cap 12.0, CUDA 13.3, driver ≥610.43.02). Statuses
> here are execution truth, not roofline targets; the generated dashboards
> (`docs/audit/generated/`) remain the drift-gated count/status surface.

Three kernel families feed the NVIDIA lane, each a distinct compile path:

1. **Compiler-emitted CUDA** — `emit/nvidia_cuda.py` synthesizes CUDA C, compiles
   it with `nvcc -arch=sm_120a`, and launches via `ctypes` (the three-seam generic
   plugin). Correctness-first, arch-agnostic; F4-gated on-device.
2. **Emit-path PTX** — `ptx_emit.py` emits PTX text, the shipped
   `tessera_nvidia_ptx_launch` bridge driver-JITs it (`cuModuleLoadDataEx`, cached
   by kernel name) and launches it (`cuLaunchKernel`).
3. **Shipped C-ABI** — `libtessera_nvidia_gemm.so` NVRTC-compiles a hand-written
   CUDA-C `mma.sync` GEMM at first call (the hand-tuned lane).

Execution status legend: **✅ proven** (execute-and-compare on sm_120) ·
**🟡 assemble-only** (ptxas accepts; not launched/numerically gated) ·
**⬜ skeleton** (instruction-encoding only, not assemblable).

---

## 1. Compiler-emitted CUDA lanes (`emit/nvidia_cuda.py`)

| Entry symbol | Op / region | Shape model | dtype | Kernel shape | Status |
|---|---|---|---|---|---|
| `tessera_nvidia_fused` | `FusedRegion` — matmul + prologue/epilogue/residual/reduction | runtime M/N/K | f32 | one thread per output row | ✅ |
| `tessera_nvidia_attn` | `AttentionRegion` — `O = softmax(scale·Q·Kᵀ)·V` | runtime M/Nk/D/Dv (Dv ≤ 256) | f32 | flash: one query/thread, online softmax, streaming KV, causal + transpose flags | ✅ (C4) |
| `tessera_nvidia_gated` | `GatedMatmulRegion` — SwiGLU gate `gate_act(A·Wg) ⊙ (A·Wu)` | runtime M/K/H | f32 | one output row/thread, shared A load | ✅ (C5) |
| `tessera_nvidia_pointwise` | `PointwiseGraphRegion` — same-shape pointwise DAG | runtime numel | f32 | one element/thread; DAG from `POINTWISE_OPS` C-expr table (+ NaN-safe `sign`/`clamp` shims) | ✅ (C5) |

All four are Tier-1 **synthesized** D1 arbiter candidates
(`Nvidia{Generic,FlashAttn,Gated,Pointwise}…Candidate`). f32 is the
correctness-first floor; 16-bit storage is served by the GEMM lanes below.

## 2. Emit-path PTX lanes (`ptx_emit.py` → `tessera_nvidia_ptx_launch`)

| Entry symbol | Op | Shape | dtype | Instruction | Status |
|---|---|---|---|---|---|
| `tessera_mma_m16n8k16_bf16` | matmul (single tile) | 16×8×16 | bf16→f32 | `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` | ✅ |
| `tessera_mma_gemm_bf16` | matmul (general) | aligned M%16/N%8/K%16, index < 2³¹ | bf16→f32 | m16n8k16 tile, K-loop + grid-tiled | ✅ (C2 breadth) |
| `tessera_mma_gemm_f16` | matmul (general) | aligned M%16/N%8/K%16, index < 2³¹ | f16→f32 | m16n8k16 tile, K-loop + grid-tiled | ✅ |
| `tessera_nvfp4_mma_m16n8k64` | matmul (block-scale) | 16×8×64 | fp4 e2m1 + ue4m3 scales → f32 | `mma.sync…m16n8k64…kind::mxf4nvf4.block_scale.scale_vec::4X` | 🟡 assemble-only (numerics gated on PTX-ISA scale spec) |
| `tessera_wgmma_matmul_bf16` | matmul (Hopper) | m64n{64,128,256}k16 | bf16→f32 | `wgmma.mma_async…` | ⬜ skeleton (needs smem descriptors + TMA; sm_90a, no Hopper box) |

The emitted general GEMM (`tessera_mma_gemm_{bf16,f16}`) is the **Tier-2 emitted**
D1 candidate (`NvidiaMmaGemmEmittedCandidate`); the launch bridge rejects
i32-index-overflow shapes (element count > 2³¹) honestly.

## 3. Shipped C-ABI lane (`libtessera_nvidia_gemm.so`)

| Entry symbol | Op | Shape | dtype | Status |
|---|---|---|---|---|
| `tessera_nvidia_mma_gemm_f16` | matmul (general tiled) | any M/N/K | f16→f32 | ✅ |
| `tessera_nvidia_mma_gemm_bf16` | matmul (general tiled) | any M/N/K | bf16→f32 | ✅ |
| `tessera_nvidia_mma_gemm_tf32` | matmul (general tiled) | any M/N/K | fp32/tf32-math→f32 | ✅ |

The shipped GEMM is the **Tier-3 hand-tuned** D1 candidate
(`NvidiaMmaGemmShippedCandidate`) — the arbiter default (lead-safe, Decision #28),
displaced only when D2's measured loop proves the emitted lane faster + in budget.

---

## Arbiter mapping (D1/D2/D3)

| op | Tier-3 hand-tuned | Tier-2 emitted | Tier-1 synthesized |
|---|---|---|---|
| `matmul` | `tessera_nvidia_mma_gemm_*` (shipped) | `tessera_mma_gemm_*` (PTX bridge) | — |
| `fused_region` | — | — | `tessera_nvidia_fused` |
| `attention` | — | — | `tessera_nvidia_attn` |
| `gated_matmul` | — | — | `tessera_nvidia_gated` |
| `pointwise` | — | — | `tessera_nvidia_pointwise` |

Selection: tier-priority by default; `emit/autotune.py` measures on-device and
caches the fastest per `(device, op, shape-bucket, dtype)` (D2); every dispatch is
recorded in the arbiter fallback log (D3, `arbiter_dispatch_histogram`).

## Toolchain pin

CUDA Toolkit **13.3** (PTX ISA 9.3); target `sm_120a` (FP4
`mma.sync.block_scale`); driver ≥610.43.02; smem 100 KB/SM. The emit-path and
shipped lanes need only the host compiler + CUDA driver (libcuda) + NVRTC at load
time; the compiler-emitted CUDA lane needs `nvcc`.

## Still open

NVFP4 execution + non-unit-scale numerics; mma.sync tensor-core versions of the
attention + fused lanes (perf); dtypes beyond f32 for the fused/attention/gated
lanes; the Hopper `wgmma` completion (sm_90a) and sm_100 `tcgen05` (their own
silicon). See [`docs/audit/backend/nvidia/NVIDIA_AUDIT.md`](audit/backend/nvidia/NVIDIA_AUDIT.md).
