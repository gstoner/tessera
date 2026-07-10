---
status: Informative
classification: Reference / Kernel Inventory
authority: Companion to Phase H ROCm backend pre-work
last_updated: 2026-07-10
---

# ROCm 7.2.4 MFMA / WMMA Kernel Inventory

> Reference enumerating every fused kernel Tessera plans to ship on AMD
> CDNA 2/3/4 (MFMA) and RDNA 3 / RDNA 3.5 / RDNA 4 (WMMA) under ROCm 7.2.4 +
> HIP 7.2.4. Companion to `docs/nvidia_cuda13_kernel_inventory.md` (parallel
> coverage tracking) and `docs/apple_gpu_kernel_inventory.md`.
>
> **Execution status (2026-07-10):** far past hardware-free. On the
> **RDNA 3.5 `gfx1151`** (Strix Halo APU — Ryzen AI Max+ 395 / Radeon 8060S)
> **dozens of compiler-generated HIP kernels now execute on real silicon**
> through `runtime.launch()` (`execution_mode="hip_runtime"`), spanning nearly
> every §5 family: matmul/GEMM + the GEMM family, flash-attention + the exotic /
> sparse / linear / delta attention lanes, norm / activation / RoPE / ALiBi,
> optimizers (adam/adamw/lion/muon/lamb), RNG, FFT/spectral, selective-SSM
> (Mamba2, incl. device backward), quantization, the loss families, the
> reduction / scan / sort / scatter / elementwise lanes, and the M7 complex /
> Clifford / EBM families. **`matmul`/`gemm`** and **`flash_attn`** were the
> first two (`hardware_verified` in `backend_manifest`, with shipped C-ABI
> symbols + measured perf ladder for GEMM); the rest execute as correctness-first
> **`compiled`** lanes (execute-vs-reference verified, no perf ladder yet).
>
> **Status truth is the generated matrix, not this prose (Decision #26):**
> [`docs/audit/generated/runtime_execution_matrix.md`](audit/generated/runtime_execution_matrix.md)
> is the drift-gated source for which `(op, target)` rows execute and by which
> lane. This inventory is the *kernel contract* (shapes, dtypes, MFU targets);
> read it against that matrix for live execution status. See §7, §9, and
> `docs/audit/backend/rocm/{ROCM_AUDIT,STRIX_HALO_EXECUTION_PLAN}.md`.

This document is the **authoritative kernel inventory** for the
ROCm backend. It captures:

1. The **toolchain pin** — ROCm 7.2.4 + HIP 7.2.4 (RCCL 2.22,
   rocBLAS 5.0, MIOpen 3.5).
2. The **shipped + planned fused kernel surface** across CDNA 2
   (gfx90a / MI250), CDNA 3 (gfx940 / MI300A, gfx942 / MI300X),
   CDNA 4 (gfx950 / MI325X), RDNA 3 (gfx1100 / RX 7900-series),
   **RDNA 3.5 (gfx1151 / Strix Halo APU — Ryzen AI Max+ 395 / Radeon 8060S)**,
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
| hipcc arch strings | `gfx90a`, `gfx940`, `gfx942`, `gfx950`, `gfx1100`, `gfx1151`, `gfx1200` (+ provisional `gfx1250`/`gfx1251`) |

Pinned in `python/tessera/compiler/rocm_target.py` as
`TESSERA_TARGET_ROCM`, `TESSERA_TARGET_HIP`,
`TESSERA_TARGET_RCCL_MIN`, `TESSERA_TARGET_ROCBLAS_MIN`,
`TESSERA_TARGET_MIOPEN_MIN`.

---

## 2. Per-arch feature matrix

The full matrix lives in `_ROCM_7_2_FEATURES` (`rocm_target.py`).
Summary:

| Feature | gfx90a | gfx940 | gfx942 | gfx950 | gfx1100 | gfx1151 | gfx1200 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `mfma` (baseline) | ✅ | ✅ | ✅ | ✅ | — | — | — |
| `mfma_f8` | — | ✅ | ✅ | ✅ | — | — | — |
| `mfma_xf32` | — | ✅ | ✅ | ✅ | — | — | — |
| `mfma_f4` | — | — | — | ✅ | — | — | — |
| `mfma_f6` | — | — | — | ✅ | — | — | — |
| `wmma_f16` | — | — | — | — | ✅ | ✅ | ✅ |
| `wmma_bf16` | — | — | — | — | ✅ | ✅ | ✅ |
| `wmma_f8` | — | — | — | — | 🟡 | — | ✅ |
| `wmma_i4` | — | — | — | — | — | 🟡 | ✅ |
| `scalar_load_u8_u16_i8_i16` | — | — | — | — | — | — | ✅ |
| `lds_async_copy` | — | ✅ | ✅ | ✅ | — | — | — |
| `buffer_load_lds` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `global_load_lds` | — | ✅ | ✅ | ✅ | — | — | — |
| `cluster_mode` | — | — | — | ✅ | — | — | — |
| `xnack` | ✅ | ✅ | ✅ | ✅ | — | — | — |
| `sram_ecc` | ✅ | ✅ | ✅ | ✅ | — | — | — |

**Wavefront width:** CDNA = 64 lanes; RDNA (incl. RDNA 3.5) = 32 lanes.
`gfx1200` is tracked as an RDNA4 / GFX12 WMMA-class artifact-planning target,
not a CDNA MFMA target.  AMD instruction spelling maps as follows:
`FP8`/`F8` → Tessera `fp8_e4m3`; `BF8` → Tessera `fp8_e5m2`;
`IU4` → planned-gated Tessera `int4` until a distinct unsigned
packed-4 storage policy exists.

**`gfx1151` (RDNA 3.5) — the load-bearing distinction from RDNA 4.** RDNA 3.5
shares RDNA 3's matrix surface: **only 16×16×16 WMMA, `f16`/`bf16` (+ `IU8`/`IU4`
per ISA §7.9 Table 33), with an `fp32` accumulator — and crucially _no FP8 WMMA_**
(that arrives with `gfx1200`/RDNA 4). `wmma_i4` is marked 🟡 because the ISA
exposes the `IU4`/`IU8` combos but Tessera keeps `int4` planned-gated. `xnack`/
`sram_ecc` are left `not_supported` (conservative): the Strix Halo APU has truly
unified LPDDR5x, but managed-memory/XNACK behaviour is not asserted until
validated on shipping silicon. This `gfx1151` WMMA `matmul` path is the one
arch×op that **executes today** (§7). The provisional `gfx1250`/`gfx1251`
(WMMA-v2 "mods/reuse" ABI, K-doubled 16×16×32 + FP8) are `llc`-grounded in the
emitter but otherwise `tba` — not in any execution path.

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
| **gfx1151** (RDNA 3.5) | ∅ — WMMA only, no MFMA |
| **gfx1200** (RDNA 4 / GFX12) | ∅ — WMMA/rocWMMA only, no MFMA |

**WMMA shape table** (`_WMMA_VARIANTS` in `rocm_target.py`, `(M, N, K)`):

| Arch | WMMA shapes |
|---|---|
| **gfx1100 / gfx1151** (RDNA 3 / 3.5) | (16, 16, 16) |
| **gfx1200** (RDNA 4) | (16, 16, 16), (16, 16, 32) |
| **gfx1250 / gfx1251** (WMMA-v2, provisional) | (16, 16, 32), (16, 16, 64), (16, 16, 128) |

---

## 4. Per-arch dtype matrix

| dtype | gfx90a | gfx940 / gfx942 | gfx950 | gfx1100 | gfx1151 | gfx1200 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| `fp64` | ✅ | ✅ | ✅ | — | — | — |
| `fp32` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `bf16` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `fp16` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `fp8_e4m3` | — | ✅ | ✅ | — | — | ✅ |
| `fp8_e5m2` | — | ✅ | ✅ | — | — | ✅ |
| `fp6_e2m3` | — | — | ✅ | — | — | — |
| `fp6_e3m2` | — | — | ✅ | — | — | — |
| `fp4_e2m1` | — | — | ✅ | — | — | — |
| `int8` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `int32` | — | — | — | — | — | ✅ |
| `int4` | — | — | — | — | 🟡 | 🟡 |

`gfx1151` (RDNA 3.5) carries the RDNA-3 matrix dtype set — `fp16`/`bf16`/`int8`
WMMA with `fp32` accumulate — and, like every RDNA arch, no `fp64` matrix path.
The **`fp8` columns are empty** for it: RDNA 3.5 has no FP8 WMMA instruction.
The runnable today is `{fp16, bf16}` (storage) → `fp32` (accumulate); `int8`/
`int4` are ISA-listed but Tessera-gated.

`gfx1200` also exposes scalar load instructions for unsigned/signed
8-bit and 16-bit values (`s_load_u8`, `s_load_u16`, `s_load_i8`,
`s_load_i16`).  Tessera documents those as scalar/storage support, not
as accelerated matrix dtypes.

---

## 5. Fused kernel inventory

Per-kernel MFMA shapes live in `_ROCM_KERNEL_MFMA_SHAPES`
(`backend_manifest.py`). MFU targets in `_ROCM_KERNEL_MFU`.

> **Execution note (2026-07-10).** The tables below are the kernel *contract*
> (MFMA/WMMA shape, dtype variants, MFU target). They are **no longer "planned"**
> on `gfx1151`: as of 2026-07-10 the majority of these families execute on real
> RDNA 3.5 silicon as compiler-generated HIP `compiled` lanes (correctness-first,
> execute-vs-reference verified). The stable lane ids per family:
>
> | §5 family | Executing gfx1151 lane(s) |
> |---|---|
> | 5.1 Matmul / contraction | `rocm_compiled` (WMMA GEMM), `rocm_matmul_family_compiled`, `rocm_dequant_gemm_compiled`, `rocm_conv_compiled` |
> | 5.2 Attention | `rocm_flash_attn_compiled`, `rocm_exotic_attn_compiled`, `rocm_sparse_attn_compiled`, `rocm_linear_attn_compiled`, `rocm_deltanet_compiled` |
> | 5.3 Fused chains | `rocm_silu_mul_compiled`, `rocm_softmax_compiled` (+ SwiGLU via matmul-family) |
> | 5.4 Norm / activation / position | `rocm_norm_compiled`, `rocm_normcompose_compiled`, `rocm_activation_compiled`, `rocm_rope_compiled`, `rocm_alibi_compiled`, `rocm_softcap_compiled` |
> | 5.5 Optimizer | `rocm_optimizer_compiled`, `rocm_muon_compiled`, `rocm_lamb_compiled` |
> | 5.6 KV-cache / paged | `rocm_kv_cache_compiled` (append/read/prune paged-movement core on the scatter+gather kernels); `quantize_kv` via `rocm_intquant_compiled` |
> | 5.7 RNG / sampling | `rocm_rng_compiled` |
> | 5.8 Spectral | `rocm_fft_compiled`, `rocm_spectral_compiled` |
> | 5.9 Recurrent / SSM | `rocm_selective_ssm_compiled` (incl. device backward), `rocm_deltanet_compiled`; single-step `lstm`/`gru`/`simple_rnn` cells via `rocm_structured_compute_compiled` (host-structured) |
>
> Quantization (`rocm_intquant`/`fpquant`/`nvfp4`), the loss families
> (`rocm_loss`/`class_loss`/`binary_loss`/`rl_loss`/`metric_loss`/`ebm_loss`), the
> reduction/scan/sort/scatter/elementwise lanes, and the §10 M7 complex/Clifford/EBM
> families also execute. **`docs/audit/generated/runtime_execution_matrix.md` is the
> authoritative live list; the mapping above is a dated snapshot, not maintained
> truth.**

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

### RDNA 3 / RDNA 3.5 / RDNA 4 WMMA (gfx1100 / gfx1151 / gfx1200)

`gfx1100` and `gfx1151` (RDNA 3.5) share the same 16×16×16 WMMA intrinsics —
`f16`/`bf16` only, **no FP8/large-K** (those `.fp8`/`.bf8`/`16x16x32+` forms are
gfx1200/RDNA 4 and up). On the Strix Halo box these lower via
`llc -mcpu=gfx1151` to real `v_wmma_f32_16x16x16_{f16,bf16}` and execute through
the C-ABI launch bridge (the shipped `tessera_rocm_wmma_gemm_{f16,bf16}` symbol).

```
llvm.amdgcn.wmma.f32.16x16x16.f16            # WMMA fp16  (gfx1100/gfx1151/gfx1200)
llvm.amdgcn.wmma.f32.16x16x16.bf16           # WMMA bf16  (gfx1100/gfx1151/gfx1200)
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

| Gate | What it means | Status |
|---|---|---|
| `artifact_only` | Target IR + AMDGCN intrinsic text are well-formed; lit fixtures pass FileCheck; no execution | CDNA (gfx90a/94x/950) MFMA entries; a few RDNA rows with no execute-compare fixture yet (e.g. KV-cache) |
| `compileable` | `hipcc -S --offload-arch=…` (or `llc -mcpu=…`) accepts the kernel; produces a valid object; **without execution** | reachable now on the box (`rocdl_emit.py` + `llc` proven for gfx1100/gfx1151) |
| `executable` | The kernel loads on a real GPU and produces correct output vs a CPU/numpy reference | ✅ **the majority of §5 on `gfx1151`** — see the generated matrix |
| `fused` | Performance characterized against the MFU targets in §5 | only `matmul`/`gemm` has a *measured perf ladder*; **no MFU-target sign-off anywhere**; CDNA MFU targets need MI300X/MI325X |

**Status truth is the generated matrix, not this table (Decision #26).**
[`docs/audit/generated/runtime_execution_matrix.md`](audit/generated/runtime_execution_matrix.md)
is the drift-gated source for which `(op, target)` rows execute. As of 2026-07-10
the ROCm backend runs **dozens of compiler-generated HIP kernels natively on
`gfx1151`** through `runtime.launch()` (`execution_mode="hip_runtime"`) — see the
§5 lane-id table. These are **`compiled`** lanes: `tessera-opt` generates the
kernel + serializes to `hsaco` in-process, then HIP loads and launches it, with an
execute-vs-reference fixture per family. They are correctness-first (no perf
ladder) and a rung below the two `hardware_verified` rows below.

**The first two — `matmul`/`gemm` + `flash_attn` are `hardware_verified`.** These
were the first ops to run natively on a non-Apple backend and carry shipped C-ABI
symbols (not just an in-process compiled lane):

- `matmul`/`gemm`: shipped `libtessera_rocm_gemm.so` exporting
  `tessera_rocm_wmma_gemm_{f16,bf16}` (HIPRTC-compiled for the device arch at
  load); wired into `runtime.launch()` as the `("rocm", "rocm_wmma")` oracle row;
  execute-compare fixture `tests/unit/test_rocm_wmma_runtime_symbol.py`
  (f16/bf16, ragged + K-looped) vs numpy; **measured GEMM perf ladder** (register
  blocking / LDS staging / software pipelining / APU zero-copy) —
  `STRIX_HALO_EXECUTION_PLAN.md` Stage F. The default `matmul` path is now the
  compiler-generated `rocm_compiled` lane, which degrades to this `rocm_wmma`
  symbol as its oracle.
- `flash_attn`: shipped `libtessera_rocm_flash_attn.so` exporting
  `tessera_rocm_wmma_flash_attn_{f16,bf16}` — FA-2 **forward**, both QK^T and P@V
  on 16×16×16 WMMA, online softmax, LDS-staged scores/accumulator, causal + ragged
  Sq/Sk; execute-compare fixture `tests/unit/test_rocm_flash_attn_runtime_symbol.py`
  (maxerr ~1e-4 f16). The FA-2 **backward** (dQ/dK/dV) also executes on gfx1151
  via the compiler-generated `rocm_flash_attn_bwd_compiled` lane
  (`generate-wmma-flash-attn-bwd-kernel` → `fa_pre`/`fa_dkdv`/`fa_dq`; O
  recomputed via the forward lane, nothing saved from forward), **MHA + GQA/MQA**
  (scale + causal), validated vs autodiff `vjp_flash_attn`. No perf ladder.

Honest scope (Decision #25): everything above is **one arch (RDNA 3.5 `gfx1151`) ×
{fp16, bf16}**, correctness-first. None of it flips the per-primitive
`backend_kernel` axis (that needs *all* targets `hardware_verified`). **CDNA MFMA
entries remain hardware-free** pending MI300-class silicon; Sprint H-4 lit fixtures
validate their IR + MFMA patterns, and `hipcc`/`llc` compile-only validation
promotes them to `compileable`. See §9 for the concrete done / open / blocked split.

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
| WMMA LLVM-IR emitter (AMD analog of `ptx_emit.py`) | `python/tessera/compiler/rocdl_emit.py` |
| Shipped runtime GEMM symbol (HIPRTC at load) | `src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip/tessera_rocm_gemm.cpp` |
| `hardware_verified` row + runtime lane | `python/tessera/compiler/{backend_manifest,execution_matrix}.py`, `python/tessera/runtime.py` |
| Lit fixtures (Sprint H-4) | `tests/tessera-ir/phase8/rocm_7_2/` |
| gfx1151 execute-compare + perf ladder | `tests/unit/test_rocm_wmma_runtime_symbol.py`, `benchmarks/rocm/benchmark_rocm_wmma_gemm.py` |
| Capability tests | `tests/unit/test_target_toolchain_pins.py` |

---

## 9. Roadmap — what's done / hardware-free / blocked

### Done on real silicon (gfx1151 / Strix Halo APU)
- ✅ WMMA `matmul`/`gemm` executes + matches numpy (`{fp16, bf16}`, f32 accum);
  `hardware_verified` row + shipped C-ABI symbol + measured GEMM perf ladder
  (register blocking is the winning lever on this unified-memory APU —
  `STRIX_HALO_EXECUTION_PLAN.md` Stage F)
- ✅ WMMA `flash_attn` FA-2 **forward** executes + matches a numpy attention
  reference (`{fp16, bf16}`, online softmax, causal, ragged); `hardware_verified`,
  no perf ladder
- ✅ WMMA `flash_attn` FA-2 **backward** (dQ/dK/dV) executes via the
  `rocm_flash_attn_bwd_compiled` lane (`generate-wmma-flash-attn-bwd-kernel` →
  `fa_pre`/`fa_dkdv`/`fa_dq`, launched through `runtime.launch()`), matches
  autodiff `vjp_flash_attn` on `{fp16, bf16}`; **MHA + GQA/MQA** (grouped `fa_dkdv`
  atomic-accumulates dK/dV across the group) **+ additive attn_bias**
  (`S = scale*Q@K^T + bias` in the recompute), scale + causal. No perf ladder;
  windowed/softcap backward is the remaining follow-up
- ✅ **Dozens of additional compiler-generated HIP `compiled` lanes execute** and
  match a CPU/numpy reference — nearly all of §5 plus §10: the GEMM/attention
  families, norm / activation / RoPE / ALiBi, optimizers, RNG, FFT/spectral,
  selective-SSM (Mamba2, **incl. device backward**), quantization, the loss
  families, and the reduction / scan / sort / scatter / elementwise lanes. The
  drift-gated live list is `docs/audit/generated/runtime_execution_matrix.md`.

### Hardware-free (lit-validated, no GPU needed)
- ✅ Capability matrix incl. gfx1151 + provisional gfx1250/1251 (`rocm_target.py`)
- ✅ Kernel inventory (this doc)
- ✅ Schema extension (shared with NVIDIA)
- ✅ Lit fixtures with MFMA + WMMA patterns
- ✅ `rocdl_emit.py` WMMA LLVM-IR emitter + `llc -mcpu=gfx1151` object (the AMD
  analog of `ptx_emit.py`)
- 🔜 `mfma_table.inc` C++ refresh to match `_MFMA_VARIANTS`
- 🔜 `hipcc -S --offload-arch=gfx942` compile-only validation (CDNA)
- 🔜 RCCL 2.22 bindings compile + symbol resolution
- 🔜 Register `tessera-to-linalg` so the MLIR `--tessera-emit-rocdl` route works
  (the emitter currently rides the direct LLVM-IR path)

### Open on this box (gfx1151 — workable now, no CDNA needed)
- `flash_attn` **backward** for the sliding-window / logit-softcap variants
  (MHA + GQA/MQA + additive **attn_bias** backward is done and runtime-wired;
  window/softcap still need the masked/capped backward math in the C++ kernel)
- **Fused paged-attention** — the §5.6 movement core (`kv_cache_append/read/prune`)
  now executes via `rocm_kv_cache_compiled` (scatter/gather compose,
  execute-compare vs `KVCacheHandle`); a single fused gather→attention paged
  kernel is the remaining step
- `grad_clip_norm` (§5.5) — global-norm + scale; single-node
- (moe transport is fully native now — `moe_dispatch` device gather, `moe_combine`
  device scatter-add, `grouped_swiglu` on the new `generate-rocm-gemm-f32-kernel`
  f32 GEMM; `rocm_moe_transport_compiled` reports `native_gpu`)
- Plain recurrent cells `lstm_cell` / `gru_cell` / `simple_rnn_cell` (§5.9) — no
  dedicated lane (selective-SSM + deltanet are done)
- **Perf ladders / MFU sign-off** beyond `matmul` — every `compiled` lane is
  correctness-first with no perf characterization

### Still blocked on hardware NOT on this box
- **CDNA execution** on MI300A / MI300X / MI325X (all MFMA entries)
- **FP8 / FP6 / FP4** matrix paths (gfx942/gfx950 MFMA; RDNA 3.5 has no FP8 WMMA)
- MFU sign-off against CDNA §5 targets
- RCCL all-reduce numerical verification across 8× ranks; multi-rank scaling
- Profiler timeline capture (rocprof) at datacenter scale

---

## 10. M7 Visual Complex Analysis (E3 follow-up)

Parallel to the NVIDIA M7 plan (see
`docs/nvidia_cuda13_kernel_inventory.md` §9).

> **Update (2026-07-10).** This family is **no longer reference-only on ROCm.**
> The M7 pointwise complex ops execute on `gfx1151` via `rocm_complex_compiled`
> (9 pointwise ops over interleaved-f32) and `rocm_conformal_compiled`
> (Möbius / stereographic on the complex mul/div lanes); the Clifford / GA
> products run via `rocm_clifford_compiled`, and the EBM family via
> `rocm_ebm_compute_compiled` / `rocm_ebm_langevin_compiled`. See the generated
> `runtime_execution_matrix`. The geometric/certificate ops (`cross_ratio`,
> `dz`/`dbar`, `laplacian_2d`, `conformal_*`, `is_concyclic`,
> `check_cauchy_riemann`) still ride the host-structured reference path. The
> planned dtype matrix below remains the fp16/bf16 *target* contract for a future
> fused MFMA kernel; the runnable dtype today is f32.

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
