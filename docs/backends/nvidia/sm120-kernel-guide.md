---
status: Informative
classification: Reference / Kernel Inventory
authority: Companion to docs/backends/nvidia/kernel-inventory.md
last_updated: 2026-07-18
---

# NVIDIA sm_120 (consumer Blackwell) mma.sync Kernel Inventory

> Hardware-verified companion to
> [`kernel-inventory.md`](kernel-inventory.md)
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
| `tessera_nvidia_mma_fused` | fused matmul + bias/activation | runtime M/N/K | f16/bf16/TF32/FP8 → f32 | one warp per 16×8; K16/K8/K32 | ✅ |
| `tessera_nvidia_mma_attn` | fused QK-softmax-PV | runtime M/Nk/D/Dv | f16/bf16/TF32/FP8 → f32 | one warp per 16 queries; dynamic score smem | ✅ |
| `tessera_nvidia_mma_gated` | paired gated projections | runtime M/H/K | f16/bf16/TF32/FP8 → f32 | one warp per 16×8; shared A fragment | ✅ |

All four are Tier-1 **synthesized** D1 arbiter candidates
(`Nvidia{Generic,FlashAttn,Gated,Pointwise}…Candidate`). f32 is the
correctness-first floor; 16-bit storage is served by the GEMM lanes below.

## 2. Emit-path PTX lanes (`ptx_emit.py` → `tessera_nvidia_ptx_launch`)

| Entry symbol | Op | Shape | dtype | Instruction | Status |
|---|---|---|---|---|---|
| `tessera_mma_m16n8k16_bf16` | matmul (single tile) | 16×8×16 | bf16→f32 | `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` | ✅ |
| `tessera_mma_gemm_bf16` | matmul (general) | aligned M%16/N%8/K%16, index < 2³¹ | bf16→f32 | m16n8k16 tile, K-loop + grid-tiled | ✅ (C2 breadth) |
| `tessera_mma_gemm_f16` | matmul (general) | aligned M%16/N%8/K%16, index < 2³¹ | f16→f32 | m16n8k16 tile, K-loop + grid-tiled | ✅ |
| `tessera_nvfp4_mma_m16n8k64` | matmul (block-scale) | 16×8×64 | fp4 e2m1 + ue4m3 scales → f32 | `mma.sync…m16n8k64…kind::mxf4nvf4.block_scale.scale_vec::4X` | ✅ sm_120a execute/compare, non-uniform block scales + SASS pinned |
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
| `tessera_nvidia_mma_gemm_e4m3` | matmul (general tiled) | any M/N/K | FP8 E4M3→f32 | ✅ |
| `tessera_nvidia_mma_gemm_e5m2` | matmul (general tiled) | any M/N/K | FP8 E5M2→f32 | ✅ |
| `tessera_nvidia_mma_gemm_nvfp4` | block-scaled matmul | any positive M/N/K, ragged zero-fill | packed E2M1 + UE4M3 scales→f32 | ✅ |

### FP8 representation versus execution

These are separate contracts. CUDA Math's `__nv_fp8_e4m3` and
`__nv_fp8_e5m2` provide byte representation, saturating construction, and
conversion; Tessera uses them in the resident f32→FP8 conversion kernels.
CUTLASS names the corresponding numeric types `float_e4m3_t` and
`float_e5m2_t` and also defines unsigned scale formats such as UE4M3/E8M0.
Tensor-core execution in Tessera does not claim the narrow `nvcuda::wmma` C++
surface: the shipped kernel and Tile lowering pack four FP8 bytes per 32-bit
fragment register and issue PTX `mma.sync.aligned.m16n8k32` E4M3/E5M2 forms.
E4M3 has finite range through ±448 and no infinity encoding; E5M2 reaches
±57344 and supports infinities. Accumulation/output here are f32.

The shipped GEMM is the **Tier-3 hand-tuned** D1 candidate
(`NvidiaMmaGemmShippedCandidate`) — the arbiter default (lead-safe, Decision #28),
displaced only when D2's measured loop proves the emitted lane faster + in budget.

---

## Arbiter mapping (D1/D2/D3)

| op | Tier-3 hand-tuned | Tier-2 emitted | Tier-1 synthesized |
|---|---|---|---|
| `matmul` | `tessera_nvidia_mma_gemm_*` (shipped) | `tessera_mma_gemm_*` (PTX bridge) | — |
| `fused_region` | — | native f16/bf16/TF32/FP8 `tessera_nvidia_mma_fused`; composed candidates retained | `tessera_nvidia_fused` |
| `attention` | — | native f16/bf16/TF32/FP8 `tessera_nvidia_mma_attn`; composed candidates retained | `tessera_nvidia_attn` |
| `gated_matmul` | — | native f16/bf16/TF32/FP8 `tessera_nvidia_mma_gated`; composed candidates retained | `tessera_nvidia_gated` |
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

General-shape NVFP4 dispatch and native TF32/FP8 transformer routes are now
landed. The retained two-run evidence promotes only cross-domain stable rows;
long-attention and several small-shape rows intentionally keep composed or
fallback selection. Remaining architecture work is Hopper `wgmma` (sm_90a) and
sm_100 `tcgen05` on their own silicon, plus future shape retuning when new stable
evidence justifies it. See
[`docs/audit/backend/nvidia/NVIDIA_AUDIT.md`](../../audit/backend/nvidia/NVIDIA_AUDIT.md).
