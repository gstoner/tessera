---
status: Informative
classification: Reference / Kernel Inventory
authority: Companion to Phase H ROCm backend pre-work
last_updated: 2026-05-11
---

# ROCm 7.2.4 MFMA Kernel Inventory

> Hardware-free reference enumerating every fused kernel Tessera plans
> to ship on AMD CDNA 3 / CDNA 4 / RDNA 3 / RDNA 4 under ROCm 7.2.4 + HIP 7.2.4.
> Companion to `docs/nvidia_cuda13_kernel_inventory.md` (parallel
> coverage tracking) and `docs/apple_gpu_kernel_inventory.md`.

This document is the **authoritative kernel inventory** for the
ROCm backend. It captures:

1. The **toolchain pin** — ROCm 7.2.4 + HIP 7.2.4 (RCCL 2.22,
   rocBLAS 5.0, MIOpen 3.5).
2. The **shipped + planned fused kernel surface** across CDNA 2
   (gfx90a / MI250), CDNA 3 (gfx940 / MI300A, gfx942 / MI300X),
   CDNA 4 (gfx950 / MI325X), RDNA 3 (gfx1100 / RX 7900-series),
   and RDNA 4 / GFX12 (gfx1200).
3. The **MFMA instruction shape contract** per kernel
   ((M, N, K, K_blocks)), LDS layout, dtype variant, and expected MFU.
4. The **AMDGCN intrinsic patterns** lit fixtures (H-4) validate against.
5. The **execution gates** — `artifact_only` → `compileable` →
   `executable` → `fused`.

---

## 1. Toolchain pin

| Pin | Value |
|---|---|
| ROCm | **7.2.4** |
| HIP | **7.2.4** |
| RCCL | **2.22** (bundled with ROCm 7.2.4) |
| rocBLAS | **≥ 5.0.0** |
| MIOpen | **≥ 3.5.0** |
| hipcc arch strings | `gfx90a`, `gfx940`, `gfx942`, `gfx950`, `gfx1100`, `gfx1200` |

Pinned in `python/tessera/compiler/rocm_target.py` as
`TESSERA_TARGET_ROCM`, `TESSERA_TARGET_HIP`,
`TESSERA_TARGET_RCCL_MIN`, `TESSERA_TARGET_ROCBLAS_MIN`,
`TESSERA_TARGET_MIOPEN_MIN`.

---

## 2. Per-arch feature matrix

The full matrix lives in `_ROCM_7_2_FEATURES` (`rocm_target.py`).
Summary:

| Feature | gfx90a | gfx940 | gfx942 | gfx950 | gfx1100 | gfx1200 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| `mfma` (baseline) | ✅ | ✅ | ✅ | ✅ | — | — |
| `mfma_f8` | — | ✅ | ✅ | ✅ | — | — |
| `mfma_xf32` | — | ✅ | ✅ | ✅ | — | — |
| `mfma_f4` | — | — | — | ✅ | — | — |
| `mfma_f6` | — | — | — | ✅ | — | — |
| `wmma_f16` | — | — | — | — | ✅ | ✅ |
| `wmma_bf16` | — | — | — | — | ✅ | ✅ |
| `wmma_f8` | — | — | — | — | 🟡 | ✅ |
| `wmma_i4` | — | — | — | — | — | ✅ |
| `scalar_load_u8_u16_i8_i16` | — | — | — | — | — | ✅ |
| `lds_async_copy` | — | ✅ | ✅ | ✅ | — | — |
| `buffer_load_lds` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `global_load_lds` | — | ✅ | ✅ | ✅ | — | — |
| `cluster_mode` | — | — | — | ✅ | — | — |
| `xnack` | ✅ | ✅ | ✅ | ✅ | — | — |
| `sram_ecc` | ✅ | ✅ | ✅ | ✅ | — | — |

**Wavefront width:** CDNA = 64 lanes; RDNA = 32 lanes.  `gfx1200`
is tracked as an RDNA4 / GFX12 WMMA-class artifact-planning target,
not a CDNA MFMA target.  AMD instruction spelling maps as follows:
`FP8`/`F8` → Tessera `fp8_e4m3`; `BF8` → Tessera `fp8_e5m2`;
`IU4` → planned-gated Tessera `int4` until a distinct unsigned
packed-4 storage policy exists.

---

## 3. MFMA instruction shape table

Per-arch frozen-set of `(M, N, K, K_blocks)` shapes the backend can
lower to (`_MFMA_VARIANTS` in `rocm_target.py`):

| Arch | Shapes |
|---|---|
| **gfx90a** (CDNA 2) | (32, 32, 8, 1), (16, 16, 16, 1) |
| **gfx940 / gfx942** (CDNA 3) | + (32, 32, 16, 1) [FP8], (16, 16, 32, 1) [FP8], (32, 32, 4, 1) [XF32], (16, 16, 8, 1) [XF32] |
| **gfx950** (CDNA 4) | + (32, 32, 32, 1) [FP4], (16, 16, 64, 1) [FP4] |
| **gfx1100** (RDNA 3) | ∅ — WMMA only, no MFMA |
| **gfx1200** (RDNA 4 / GFX12) | ∅ — WMMA/rocWMMA only, no MFMA |

---

## 4. Per-arch dtype matrix

| dtype | gfx90a | gfx940 / gfx942 | gfx950 | gfx1100 | gfx1200 |
|---|:-:|:-:|:-:|:-:|:-:|
| `fp64` | ✅ | ✅ | ✅ | — | — |
| `fp32` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `bf16` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `fp16` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `fp8_e4m3` | — | ✅ | ✅ | — | ✅ |
| `fp8_e5m2` | — | ✅ | ✅ | — | ✅ |
| `fp6_e2m3` | — | — | ✅ | — | — |
| `fp6_e3m2` | — | — | ✅ | — | — |
| `fp4_e2m1` | — | — | ✅ | — | — |
| `int8` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `int32` | — | — | — | — | ✅ |
| `int4` | — | — | — | — | 🟡 |

`gfx1200` also exposes scalar load instructions for unsigned/signed
8-bit and 16-bit values (`s_load_u8`, `s_load_u16`, `s_load_i8`,
`s_load_i16`).  Tessera documents those as scalar/storage support, not
as accelerated matrix dtypes.

---

## 5. Planned fused kernel inventory

Per-kernel MFMA shapes live in `_ROCM_KERNEL_MFMA_SHAPES`
(`backend_manifest.py`). MFU targets in `_ROCM_KERNEL_MFU`.

### 5.1 Matmul / contraction family

| Kernel | MFMA (M, N, K, K_blocks) | Dtypes | MFU (gfx942 / gfx950) | Notes |
|---|---|---|---|---|
| `matmul` / `gemm` | (32, 32, 8, 1) | bf16, fp16, fp8_e4m3, fp32, fp64 | 75% / 78% | canonical CDNA bf16 shape |
| `batched_gemm` | (32, 32, 8, 1) | bf16, fp16 | 72% / 75% | batch dim parallel |
| `einsum` | (32, 32, 8, 1) | bf16, fp16, fp32 | 75% / 78% | via matmul decomposition |
| `linear_general` | (32, 32, 8, 1) | bf16, fp16 | 75% / 78% | matmul + bias + activation |
| `qkv_projection` | (32, 32, 8, 1) | bf16, fp16 | 72% / 75% | three matmuls fused |
| `fused_epilogue` | (32, 32, 8, 1) | bf16, fp16 | 72% / 75% | matmul + bias + activation |
| `factorized_matmul` | (16, 16, 16, 1) | bf16, fp16 | 65% / 68% | low-rank split |

### 5.2 Attention family

Smaller MFMA tile (16 × 16) for score matrices since head_dim is
typically narrow (≤ 128). FA-2-style online softmax + 2-stage SMEM
double buffering through LDS.

| Kernel | MFMA Shape | Dtypes | MFU (gfx942 / gfx950) | Notes |
|---|---|---|---|---|
| `flash_attn` fwd | (16, 16, 16, 1) | bf16, fp16, fp8_e4m3 | 65% / 70% | rocm-FA2 idiom |
| `flash_attn` bwd | (16, 16, 16, 1) | bf16, fp16 | 55% / 60% | recompute fwd in LDS |
| `multi_head_attention` | (16, 16, 16, 1) | bf16, fp16 | 65% / 70% | wraps flash_attn |
| `gqa_attention` | (16, 16, 16, 1) | bf16, fp16 | 65% / 70% | KV-head broadcast |
| `mqa_attention` | (16, 16, 16, 1) | bf16, fp16 | 65% / 70% | 1-head KV |
| `mla_decode` | (16, 16, 16, 1) | bf16, fp16 | 50% / 55% | KV-memory-bound |
| `mla_decode_fused` | (16, 16, 16, 1) | bf16, fp16 | 55% / 60% | latent expand + flash_attn |
| `deepseek_sparse_attention` | (16, 16, 16, 1) | bf16, fp16 | 45% / 50% | top-k gather-bound |
| `attn_top_k_blocks` | (16, 16, 16, 1) | bf16, fp16 | 45% / 50% | top-k block selection |
| `attn_compressed_blocks` | (16, 16, 16, 1) | bf16, fp16 | 50% / 55% | block-pooled KV |
| `attn_sliding_window` | (16, 16, 16, 1) | bf16, fp16 | 55% / 60% | local-window |
| `lightning_attention` | (16, 16, 16, 1) | bf16, fp16 | 35% / 40% | recurrence-serial |
| `linear_attn` | (16, 16, 16, 1) | bf16, fp16 | 35% / 40% | linear-attention |
| `gated_deltanet` | (16, 16, 16, 1) | bf16, fp16 | 30% / 35% | delta-rule + gating |
| `kimi_delta_attention` | (16, 16, 16, 1) | bf16, fp16 | 35% / 40% | Kimi delta-rule |
| `modified_delta_attention` | (16, 16, 16, 1) | bf16, fp16 | 35% / 40% | modified delta-rule |
| `gated_attention` | (16, 16, 16, 1) | bf16, fp16 | 60% / 65% | gated softmax |
| `hybrid_attention` | (16, 16, 16, 1) | bf16, fp16 | 55% / 60% | MLA + Lightning hybrid |

### 5.3 Fused chains (multi-op kernels)

| Kernel | Composition | Dtypes | Notes |
|---|---|---|---|
| `matmul_softmax` | matmul → softmax in VGPRs | bf16, fp16 | saves score-matrix LDS round-trip |
| `matmul_softmax_matmul` | full attention block | bf16, fp16 | full attention via 2× MFMA + softmax |
| `matmul_gelu` | matmul → gelu | bf16, fp16, fp32 | bias + activation epilogue |
| `matmul_rmsnorm` | matmul → rmsnorm | bf16, fp16, fp32 | feed-forward post-norm |
| `matmul_silu_mul` | matmul → silu·mul | bf16, fp16 | half SwiGLU |
| `swiglu_mlp` | matmul → silu·mul → matmul | bf16, fp16 | full SwiGLU |

### 5.4 Normalization, activation, position encoding

No MFMA — LDS-based cooperative reductions.

| Kernel | Dtypes | MFU | Notes |
|---|---|---|---|
| `softmax` / `softmax_safe` | bf16, fp16, fp32 | 45-55% | LDS row reduce |
| `online_softmax` | bf16, fp16, fp32 | 45-55% | streaming chunked |
| `layer_norm` | bf16, fp16 | 50-60% | 2× row reductions |
| `rmsnorm` / `rmsnorm_safe` | bf16, fp16 | 55-65% | 1× row reduction |
| `gelu` | bf16, fp16, fp32 | 40-50% | elementwise |
| `silu` / `silu_mul` | bf16, fp16, fp32 | 40-50% | elementwise |
| `rope` | bf16, fp16, fp32 | 50-60% | rotary embedding |
| `alibi` | bf16, fp16 | 50-60% | ALiBi bias |

### 5.5 Optimizer / training-step fused kernels

| Kernel | Dtypes (param / accum) | Notes |
|---|---|---|
| `adamw_step` | bf16/fp16 param, fp32 accum + m + v | parameter update fused |
| `lion_step` | bf16/fp16 param, fp32 accum | sign-based |
| `muon_step` | bf16/fp16 param, fp32 accum | Newton-Schulz |
| `lamb_step` | bf16/fp16 param, fp32 accum + m + v | layerwise adaptive |
| `grad_clip_norm` | fp32 | global norm + scale; RCCL all-reduce |

### 5.6 KV-cache / paged-attention kernels

| Kernel | Dtypes | Notes |
|---|---|---|
| `kv_cache_append` | bf16, fp16, fp8 | paged write |
| `kv_cache_read` | bf16, fp16, fp8 | paged read |
| `kv_cache_prune` | bf16, fp16 | sliding-window |
| `quantize_kv` | bf16/fp16 → int8/int4 | block-quantized |
| `dequantize_kv` | int8/int4 → bf16/fp16 | inverse |

### 5.7 RNG + sampling kernels

| Kernel | Dtypes | Notes |
|---|---|---|
| `rng_uniform` / `rng_normal` | fp32, bf16, fp16 | Philox counter-based |
| `dropout` | bf16, fp16, fp32 | fused mask + scale |
| `multinomial` / `categorical` | fp32 | top-k / nucleus sampling |
| `permutation` | int32, int64 | Philox-shuffled |

### 5.8 Spectral family

| Kernel | Dtypes | Notes |
|---|---|---|
| `fft` / `ifft` | fp32, fp64 (complex) | radix-4 Stockham |
| `rfft` / `irfft` | fp32 | half-spectrum |
| `stft` / `istft` | fp32 | windowed FFT |
| `dct` | fp32 | type-II DCT |
| `spectral_conv` | fp32 | FFT-based conv |
| `spectral_filter` | fp32 | FFT-mask-IFFT |

### 5.9 Recurrent / SSM kernels

| Kernel | Dtypes | Notes |
|---|---|---|
| `lstm_cell` | bf16, fp16 | single-step LSTM |
| `gru_cell` | bf16, fp16 | single-step GRU |
| `simple_rnn_cell` | bf16, fp16 | tanh recurrence |
| `selective_ssm` | bf16, fp16 | Mamba2 chunked scan |
| `depthwise_conv1d` | bf16, fp16 | causal/streaming conv1d |

---

## 6. AMDGCN intrinsic patterns (Sprint H-4 lit fixtures validate these)

Each lit fixture under
`tests/tessera-ir/phase8/rocm_7_2/` asserts on the
following AMDGCN intrinsic text patterns:

### CDNA 2 / CDNA 3 / CDNA 4 MFMA (gfx90a / gfx94x / gfx950)

```
llvm.amdgcn.mfma.f32.32x32x8bf16.1k          # CDNA 2/3 bf16
llvm.amdgcn.mfma.f32.16x16x16bf16.1k         # CDNA 2/3 bf16
llvm.amdgcn.mfma.f32.32x32x16f8f8.fp8.fp8    # CDNA 3 FP8
llvm.amdgcn.mfma.f32.16x16x32f8f8.fp8.fp8    # CDNA 3 FP8
llvm.amdgcn.mfma.f32.32x32x4xf32             # CDNA 3 XF32
llvm.amdgcn.mfma.f32.16x16x8xf32             # CDNA 3 XF32
llvm.amdgcn.mfma.f32.32x32x32f4f4            # CDNA 4 FP4
llvm.amdgcn.mfma.f32.16x16x64f4f4            # CDNA 4 FP4
```

### LDS async copy (gfx940+)

```
llvm.amdgcn.global.load.lds                  # global → LDS
llvm.amdgcn.buffer.load.lds                  # buffer → LDS (gfx9x baseline)
llvm.amdgcn.s.barrier                        # wave-front barrier
```

### RDNA 3 / RDNA 4 WMMA (gfx1100 / gfx1200)

```
llvm.amdgcn.wmma.f32.16x16x16.f16            # WMMA bf16/fp16
llvm.amdgcn.wmma.f32.16x16x16.bf16
llvm.amdgcn.wmma.f32.16x16x16.f8             # GFX12 planning target
llvm.amdgcn.wmma.f32.16x16x16.bf8            # GFX12 planning target
llvm.amdgcn.wmma.i32.16x16x32.iu4            # GFX12 IU4 -> int32
llvm.amdgcn.swmmac.f32.16x16x32.f16          # GFX12 SWMMAC FP16 -> FP32
llvm.amdgcn.swmmac.f32.16x16x32.bf16         # GFX12 SWMMAC BF16 -> FP32
llvm.amdgcn.swmmac.i32.16x16x32.iu8          # GFX12 SWMMAC INT8 -> INT32
llvm.amdgcn.swmmac.i32.16x16x32.iu4          # GFX12 SWMMAC INT4 -> INT32
llvm.amdgcn.swmmac.i32.16x16x64.iu4          # GFX12 SWMMAC INT4 -> INT32
```

GFX12 scalar prefetch / load notes tracked for future scheduler work:
`s_prefetch_inst`, `s_prefetch_inst_pc_rel`, `s_prefetch_data`,
`s_prefetch_data_pc_rel`, `s_buffer_prefetch_data`, plus scalar
`s_load_u16`, `s_load_u8`, `s_load_i16`, and `s_load_i8`.

### XNACK / SRAM-ECC control (CDNA)

```
.amdhsa_user_sgpr_xnack
.amdhsa_user_sgpr_sram_ecc
```

---

## 7. Execution gates

| Gate | What it means | When it lifts |
|---|---|---|
| `artifact_only` (current state) | Target IR + AMDGCN intrinsic text are well-formed; lit fixtures pass FileCheck; no execution | now |
| `compileable` | `hipcc -S --offload-arch=gfx942` accepts the kernel; produces a valid hsaco; **without execution** | once ROCm 7.2.4 is installed on the dev box (no GPU needed) |
| `executable` | The hsaco loads on a real MI300X / MI325X and produces correct output vs CPU reference | requires MI300A/X or MI325X hardware |
| `fused` | Performance characterized; hits the MFU targets in §5 | requires hardware + perf tuning sprint |

Today the ROCm backend sits at `artifact_only` across every entry.
Sprint H-4 lit fixtures validate the IR + MFMA patterns hardware-free.
Sprint H-6/H-7/H-8 will promote entries to `compileable` once
`hipcc 7.2.4` runs the compile-only validation.

---

## 8. Source map

| Component | Path |
|---|---|
| Toolchain pin + feature matrix | `python/tessera/compiler/rocm_target.py` |
| Per-target capability registry | `python/tessera/compiler/capabilities.py` (`rocm`, `rocm_gfx90a`..`rocm_gfx1200`) |
| Per-kernel MFMA shape + MFU tables | `python/tessera/compiler/backend_manifest.py` (`_ROCM_KERNEL_MFMA_SHAPES`, `_ROCM_KERNEL_MFU`) |
| BackendKernelEntry schema (G-3) | `python/tessera/compiler/backend_manifest.py` |
| MLIR pass library | `src/compiler/codegen/Tessera_ROCM_Backend/` (MFMA full coverage, ROCm lowering) |
| MFMA shape lookup table (C++) | `src/compiler/codegen/Tessera_ROCM_Backend/.../mfma_table.inc` |
| Lit fixtures (Sprint H-4) | `tests/tessera-ir/phase8/rocm_7_2/` |
| Capability tests | `tests/unit/test_target_toolchain_pins.py` |

---

## 9. Roadmap — what's hardware-free vs. blocked

### Hardware-free (this batch lands now)
- ✅ Capability matrix (Sprint H-1)
- ✅ Kernel inventory (this doc — Sprint H-3)
- ✅ Schema extension (Sprint G-3, shared with NVIDIA)
- ✅ Lit fixtures with MFMA patterns (Sprint H-4)
- 🔜 `mfma_table.inc` C++ refresh to match `_MFMA_VARIANTS` (Sprint H-2)
- 🔜 `hipcc -S --offload-arch=gfx942` compile-only validation (Sprint H-6/H-7)
- 🔜 RCCL 2.22 bindings compile + symbol resolution (Sprint H-8)

### Blocked on hardware
- End-to-end execution on MI300A / MI300X / MI325X
- Numerical correctness vs CPU reference
- TFLOPS / latency / MFU measurement against §5 targets
- RCCL all-reduce numerical verification across 8x ranks
- Profiler timeline capture (rocprof)
- Multi-rank scaling tests

---

## 10. M7 Visual Complex Analysis (E3 follow-up)

Parallel to the NVIDIA M7 plan (see
`docs/nvidia_cuda13_kernel_inventory.md` §9). The backend manifest
reserves `status="planned"` slots for every M7 op on the ROCm target
with the matrix below; all 20 M7 ops run today **only** via the
Python reference path on CPU (`status="reference"`,
`dtypes=("fp32",)`).

| Op family | Lowering target | Planned dtype matrix |
|---|---|---|
| `complex_mul/div/exp/log/sqrt/pow` | Pointwise AMDGCN (packed `(re, im)`) | **fp32 / fp16 / bf16** |
| `complex_conjugate/abs/arg` | Pointwise AMDGCN | **fp32 / fp16 / bf16** |
| `mobius` / `stereographic` / `mobius_from_three_points` | Pointwise AMDGCN with broadcast scalar coefficients | **fp32 / fp16 / bf16** |
| `cross_ratio` / `is_concyclic` / `check_cauchy_riemann` | Small fixed-size projective math + reduction | **fp32 / fp16 / bf16** |
| `dz` / `dbar` / `laplacian_2d` | 3×3 stencil (one AMDGCN lane per output element) | **fp32 / fp16 / bf16** |
| `conformal_jacobian` / `conformal_energy_on_sphere` | Stencil + reduction | **fp32 / fp16 / bf16** |

**Read the dtype column against the entry's status.** fp16/bf16 in
the planned-status rows above describe **target dtypes for the
unbuilt MFMA kernels**, not what runs today. The runnable runtime
dtype is fp32-only via the Python reference path on CPU.

Promotion gated on Phase H. When kernels land, status flips
`planned → fused` and the dtype matrix becomes the live kernel's
contract unchanged.
