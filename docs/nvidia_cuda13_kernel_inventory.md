---
status: Informative
classification: Reference / Kernel Inventory
authority: Companion to Phase G NVIDIA backend pre-work
last_updated: 2026-05-11
---

# NVIDIA CUDA 13.2 Update 1 Kernel Inventory

> Hardware-free reference enumerating every fused kernel Tessera plans
> to ship on NVIDIA SM_90+ under CUDA Toolkit 13.2 Update 1. Companion
> to `docs/apple_gpu_kernel_inventory.md` and
> the backend manifest. Each entry locks the WGMMA tile
> shape, dtype matrix, cluster size, expected MFU, and roofline target
> so the compile path can be validated without hardware.

This document is the **authoritative kernel inventory** for the
NVIDIA backend. It captures:

1. The **toolchain pin** — CUDA Toolkit 13.2 Update 1 (PTX ISA 8.6,
   NCCL 2.22, driver ≥555.85).
2. The **shipped + planned fused kernel surface** across SM_90 Hopper,
   SM_100 Blackwell, and SM_120 Rubin.
3. The **WGMMA tile shape contract** per kernel (M × N × K), the
   **cluster size** for thread-block cluster launch, and the
   **expected MFU** target for each (op, arch) pair.
4. The **PTX assembly patterns** lit fixtures (G-4) validate against.
5. The **execution gates** — `artifact_only` → `compileable` →
   `executable` → `fused`.

---

## 1. Toolchain pin

| Pin | Value |
|---|---|
| CUDA Toolkit | **13.2 Update 1** |
| Driver minimum | **555.85** (shipped with 13.2 U1) |
| PTX ISA | **8.6** |
| NCCL | **2.22** (bundled with 13.2 U1) |
| nvcc arch strings | `sm_80`, `sm_86`, `sm_89`, `sm_90a`, `sm_100a`, `sm_120a` |

Tessera's NVIDIA backend pins these in
`python/tessera/compiler/gpu_target.py` as
`TESSERA_TARGET_CUDA_TOOLKIT`, `TESSERA_TARGET_CUDA_DRIVER_MIN`,
`TESSERA_TARGET_PTX_ISA`, `TESSERA_TARGET_NCCL_MIN`.

---

## 2. Per-SM feature matrix

The full matrix lives in `_CUDA_13_2_FEATURES` (`gpu_target.py`). Summary:

| Feature | SM_80 | SM_86 | SM_89 | SM_90 | SM_100 | SM_120 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| `wmma` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `wgmma` | — | — | — | ✅ | ✅ | ✅ |
| `wgmma_sparse` | — | — | — | ✅ | ✅ | ✅ |
| `tma` | — | — | — | ✅ | ✅ | ✅ |
| `tma_swizzle_128b` | — | — | — | ✅ | ✅ | ✅ |
| `cluster_launch` | — | — | — | ✅ | ✅ | ✅ |
| `mbarrier_arrive_tx` | — | — | — | ✅ | ✅ | ✅ |
| `cp_async_bulk` | — | — | — | ✅ | ✅ | ✅ |
| `async_proxy_fence` | — | — | — | ✅ | ✅ | ✅ |
| `tcgen05` | — | — | — | — | ✅ | ✅ |
| `tcgen05_pair` | — | — | — | — | ✅ | ✅ |
| `tmem` | — | — | — | — | ✅ | ✅ |
| `block_scaled_mma` | — | — | — | — | ✅ | ✅ |

---

## 3. Per-SM dtype matrix

`_TENSOR_CORE_DTYPES` in `gpu_target.py`. TF32 is a **math_mode**, not
a storage dtype (per `docs/reference/tessera_tensor_attributes.md`).

| dtype | SM_80 | SM_90 | SM_100 | SM_120 |
|---|:-:|:-:|:-:|:-:|
| `fp64` | ✅ | ✅ | ✅ | ✅ |
| `fp32` (CUDA-core) | ✅ | ✅ | ✅ | ✅ |
| `fp16` | ✅ | ✅ | ✅ | ✅ |
| `bf16` | ✅ | ✅ | ✅ | ✅ |
| `fp8_e4m3` | — | ✅ | ✅ | ✅ |
| `fp8_e5m2` | — | ✅ | ✅ | ✅ |
| `fp6_e2m3` | — | — | ✅ | ✅ |
| `fp6_e3m2` | — | — | ✅ | ✅ |
| `fp4_e2m1` | — | — | ✅ | ✅ |
| `nvfp4` | — | — | ✅ | ✅ |
| `int8` | ✅ | ✅ | ✅ | ✅ |
| TF32 (math_mode) | ✅ | ✅ | ✅ | ✅ |

---

## 4. Planned fused kernel inventory

Each row lists the WGMMA tile shape (M × N × K), cluster size,
canonical dtype set, expected MFU on Hopper / Blackwell, and the
roofline characterization. Per-kernel WGMMA shapes live in
`_NVIDIA_KERNEL_TILE_SHAPES`.

### 4.1 Matmul / contraction family

| Kernel | WGMMA (M, N, K) | Cluster | Dtypes | MFU (SM_90 / SM_100) | Roofline |
|---|---|---|---|---|---|
| `matmul` / `gemm` | (64, 256, 16) | (1, 1, 1) | bf16, fp16, fp8_e4m3, fp8_e5m2, fp32, fp64 | 80% / 82% | compute-bound at M·N ≥ 8192² |
| `batched_gemm` | (64, 256, 16) | (1, 1, 1) | bf16, fp16 | 78% / 80% | compute-bound; batch dim parallel |
| `einsum` | (64, 256, 16) | (1, 1, 1) | bf16, fp16, fp32 | 80% / 82% | via matmul decomposition |
| `linear_general` | (64, 256, 16) | (1, 1, 1) | bf16, fp16 | 80% / 82% | matmul + bias + activation epilogue |
| `qkv_projection` | (64, 256, 16) | (1, 1, 1) | bf16, fp16 | 78% / 80% | three matmuls fused into one launch |
| `fused_epilogue` | (64, 256, 16) | (1, 1, 1) | bf16, fp16 | 78% / 80% | matmul + bias + GELU/SiLU/ReLU |
| `factorized_matmul` | (64, 128, 16) | (1, 1, 1) | bf16, fp16 | 70% / 72% | low-rank: M × R × N split |

### 4.2 Attention family

All attention kernels use online-softmax + warp-specialized
producer/consumer pipelines. Cluster (2, 1, 1) pairs CTAs for
producer→consumer SMEM forwarding.

| Kernel | WGMMA (M, N, K) | Cluster | Dtypes | MFU (SM_90 / SM_100) | Roofline |
|---|---|---|---|---|---|
| `flash_attn` (FA-4 fwd) | (64, 128, 16) | (2, 1, 1) | bf16, fp16, fp8_e4m3 | 75% / 78% | compute-bound at seq_len ≥ 1024 + head_dim ≥ 64 |
| `flash_attn` (FA-4 bwd) | (64, 128, 16) | (2, 1, 1) | bf16, fp16 | 65% / 68% | compute-bound; recompute fwd activations |
| `multi_head_attention` | (64, 128, 16) | (2, 1, 1) | bf16, fp16 | 75% / 78% | wraps flash_attn |
| `gqa_attention` | (64, 128, 16) | (2, 1, 1) | bf16, fp16 | 75% / 78% | wraps flash_attn with KV-head broadcast |
| `mqa_attention` | (64, 128, 16) | (2, 1, 1) | bf16, fp16 | 75% / 78% | wraps flash_attn with 1-head KV |
| `mla_decode` | (64, 128, 16) | (2, 1, 1) | bf16, fp16 | 55% / 60% | KV-cache-memory-bound (latent KV reduces BW ~4×) |
| `mla_decode_fused` | (64, 128, 16) | (2, 1, 1) | bf16, fp16 | 60% / 65% | fuses latent expand + flash_attn |
| `deepseek_sparse_attention` (NSA) | (64, 128, 16) | (1, 1, 1) | bf16, fp16 | 50% / 55% | gather-bound at top-k ≤ 32 |
| `attn_top_k_blocks` | (64, 128, 16) | (1, 1, 1) | bf16, fp16 | 50% / 55% | top-k block selection on the score matrix |
| `attn_compressed_blocks` | (64, 128, 16) | (1, 1, 1) | bf16, fp16 | 55% / 60% | block-wise pooled KV |
| `attn_sliding_window` | (64, 128, 16) | (1, 1, 1) | bf16, fp16 | 60% / 65% | local-window attention |
| `lightning_attention` | (32, 32, 16) | (1, 1, 1) | bf16, fp16 | 40% / 45% | recurrence-serial; memory-bound on state update |
| `linear_attn` | (32, 32, 16) | (1, 1, 1) | bf16, fp16 | 40% / 45% | linear-attention kernel |
| `gated_deltanet` | (32, 32, 16) | (1, 1, 1) | bf16, fp16 | 35% / 40% | gated delta-rule update + recurrence |
| `kimi_delta_attention` | (32, 32, 16) | (1, 1, 1) | bf16, fp16 | 40% / 45% | Kimi delta-rule attention |
| `modified_delta_attention` | (32, 32, 16) | (1, 1, 1) | bf16, fp16 | 40% / 45% | modified delta-rule attention |
| `gated_attention` | (64, 128, 16) | (1, 1, 1) | bf16, fp16 | 65% / 70% | gated softmax attention |
| `hybrid_attention` | (64, 128, 16) | (2, 1, 1) | bf16, fp16 | 60% / 65% | MLA + Lightning hybrid |

### 4.3 Fused chains (multi-op kernels)

| Kernel | Composition | Cluster | Dtypes | Roofline |
|---|---|---|---|---|
| `matmul_softmax` | matmul → softmax (in-register score matrix) | (1, 1, 1) | bf16, fp16 | saves the score-matrix DRAM round-trip |
| `matmul_softmax_matmul` | full attention block (matmul → softmax → matmul) | (2, 1, 1) | bf16, fp16 | saves both score + softmax round-trips |
| `matmul_gelu` | matmul → gelu | (1, 1, 1) | bf16, fp16, fp32 | bias + activation in epilogue |
| `matmul_rmsnorm` | matmul → rmsnorm | (1, 1, 1) | bf16, fp16, fp32 | feed-forward post-norm |
| `matmul_silu_mul` | matmul → silu·mul (SwiGLU half-block) | (1, 1, 1) | bf16, fp16 | half of SwiGLU MLP |
| `swiglu_mlp` | matmul → silu·mul → matmul (full SwiGLU) | (2, 1, 1) | bf16, fp16 | full SwiGLU feed-forward |

### 4.4 Normalization, activation, position encoding

These don't use WGMMA — they use cooperative warp-shuffle reductions
or single-tile elementwise. Recorded shape is `None`.

| Kernel | Cluster | Dtypes | MFU | Notes |
|---|---|---|---|---|
| `softmax` / `softmax_safe` | (1, 1, 1) | bf16, fp16, fp32 | 45-55% | warp-shuffle row reduce |
| `online_softmax` | (1, 1, 1) | bf16, fp16, fp32 | 45-55% | streaming chunked variant |
| `layer_norm` | (1, 1, 1) | bf16, fp16 | 50-60% | 2× row reductions |
| `rmsnorm` / `rmsnorm_safe` | (1, 1, 1) | bf16, fp16 | 55-65% | 1× row reduction |
| `gelu` | (1, 1, 1) | bf16, fp16, fp32 | 40-50% | elementwise, memory-bound |
| `silu` / `silu_mul` | (1, 1, 1) | bf16, fp16, fp32 | 40-50% | elementwise, memory-bound |
| `rope` | (1, 1, 1) | bf16, fp16, fp32 | 50-60% | rotary embedding |
| `alibi` | (1, 1, 1) | bf16, fp16 | 50-60% | ALiBi position bias |

### 4.5 Optimizer / training-step fused kernels

| Kernel | Cluster | Dtypes (param / accum) | Notes |
|---|---|---|---|
| `adamw_step` | (1, 1, 1) | bf16/fp16 param, fp32 accum + m + v | fused parameter update + LR schedule + decay |
| `lion_step` | (1, 1, 1) | bf16/fp16 param, fp32 accum | fused sign-based update |
| `muon_step` | (1, 1, 1) | bf16/fp16 param, fp32 accum | Newton-Schulz orthogonalization |
| `lamb_step` | (1, 1, 1) | bf16/fp16 param, fp32 accum + m + v | layerwise adaptive moments |
| `grad_clip_norm` | (1, 1, 1) | fp32 | global norm + scale; all-reduce required |

### 4.6 KV-cache / paged-attention kernels

| Kernel | Cluster | Dtypes | Notes |
|---|---|---|---|
| `kv_cache_append` | (1, 1, 1) | bf16, fp16, fp8 | paged write to KV cache |
| `kv_cache_read` | (1, 1, 1) | bf16, fp16, fp8 | paged read for prefill / decode |
| `kv_cache_prune` | (1, 1, 1) | bf16, fp16 | sliding-window eviction |
| `quantize_kv` | (1, 1, 1) | bf16/fp16 → int8/int4 | block-quantized KV |
| `dequantize_kv` | (1, 1, 1) | int8/int4 → bf16/fp16 | inverse |

### 4.7 RNG + sampling kernels

| Kernel | Cluster | Dtypes | Notes |
|---|---|---|---|
| `rng_uniform` / `rng_normal` | (1, 1, 1) | fp32, bf16, fp16 | Philox counter-based |
| `dropout` | (1, 1, 1) | bf16, fp16, fp32 | fused mask + scale |
| `multinomial` / `categorical` | (1, 1, 1) | fp32 | top-k / nucleus sampling |
| `permutation` | (1, 1, 1) | int32, int64 | Philox-shuffled |

### 4.8 Spectral family

| Kernel | Cluster | Dtypes | Notes |
|---|---|---|---|
| `fft` / `ifft` | (1, 1, 1) | fp32, fp64 (complex) | radix-4 Stockham, mixed-radix tail |
| `rfft` / `irfft` | (1, 1, 1) | fp32 | half-spectrum real FFT |
| `stft` / `istft` | (1, 1, 1) | fp32 | windowed FFT |
| `dct` | (1, 1, 1) | fp32 | type-II DCT |
| `spectral_conv` | (1, 1, 1) | fp32 | FFT-based convolution |
| `spectral_filter` | (1, 1, 1) | fp32 | FFT-mask-IFFT |

### 4.9 Recurrent / SSM kernels

| Kernel | Cluster | Dtypes | Notes |
|---|---|---|---|
| `lstm_cell` | (1, 1, 1) | bf16, fp16 | single-step LSTM with state |
| `gru_cell` | (1, 1, 1) | bf16, fp16 | single-step GRU |
| `simple_rnn_cell` | (1, 1, 1) | bf16, fp16 | tanh recurrence |
| `selective_ssm` | (1, 1, 1) | bf16, fp16 | Mamba2 chunked scan + delta rule |
| `depthwise_conv1d` | (1, 1, 1) | bf16, fp16 | causal/streaming conv1d |

---

## 5. PTX assembly patterns (Sprint G-4 lit fixtures validate these)

Each lit fixture under
`tests/tessera-ir/phase3/cuda13/` asserts on the
following PTX text patterns:

### WGMMA (SM_90+)

```
wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16
wgmma.mma_async.sync.aligned.m32n32k16.f32.bf16.bf16
wgmma.mma_async.sync.aligned.m64n256k32.f32.e4m3.e4m3   # FP8
wgmma.mma_async.sync.aligned.m64n256k64.f32.e2m1.e2m1   # FP4 (Blackwell+)
```

### TMA (SM_90+)

```
cp.async.bulk.tensor.2d.shared::cta.global
cp.async.bulk.tensor.5d.shared::cluster.global
fence.proxy.async.shared::cta
```

### mbarrier (SM_90+)

```
mbarrier.init.shared::cta.b64
mbarrier.arrive.expect_tx.shared::cta.b64
mbarrier.try_wait.parity.shared::cta.b64
```

### Cluster launch (SM_90+)

```
griddepcontrol.launch_dependents
griddepcontrol.wait
```

### tcgen05 (SM_100+ Blackwell)

```
tcgen05.mma.cta_group::1.kind::f16
tcgen05.mma.cta_group::2.kind::f16
tcgen05.commit.cta_group::1.shared::cluster
```

### TMEM (SM_100+ Blackwell)

```
tcgen05.alloc.cta_group::1.b32
tcgen05.dealloc.cta_group::1.b32
tcgen05.ld.shared::cta.b32
tcgen05.st.shared::cta.b32
```

### cp.async.bulk + async_proxy_fence (SM_90+)

```
cp.async.bulk.global.shared::cta.bulk_group
cp.async.bulk.commit_group
cp.async.bulk.wait_group
```

---

## 6. Execution gates

| Gate | What it means | When it lifts |
|---|---|---|
| `artifact_only` (current state) | Target IR + PTX text are well-formed; lit fixtures pass FileCheck; no execution | now |
| `compileable` | `ptxas -arch=sm_90a` accepts the emitted PTX; produces a valid cubin; **without execution** | once CUDA Toolkit 13.2 U1 is installed on the dev box (no GPU needed) |
| `executable` | The cubin loads on a real H100/B100 and produces correct output vs CPU reference (fp64) | requires H100/B100/Rubin hardware |
| `fused` | Performance characterized; achieves the MFU targets in §4 | requires hardware + perf tuning sprint |

Today the NVIDIA backend sits at `artifact_only` across every entry.
Sprint G-4 lit fixtures validate the IR + PTX patterns hardware-free.
Sprint G-6/G-7/G-8 will promote entries to `compileable` once
`nvcc 13.2 U1` runs the compile-only validation.

---

## 7. Source map

| Component | Path |
|---|---|
| Toolchain pin + feature matrix | `python/tessera/compiler/gpu_target.py` (CUDA 13.2 U1 block) |
| Per-target capability registry | `python/tessera/compiler/capabilities.py` (`nvidia_sm80`..`nvidia_sm120`) |
| Per-kernel WGMMA shape + MFU + roofline tables | `python/tessera/compiler/backend_manifest.py` (`_NVIDIA_KERNEL_TILE_SHAPES`, `_NVIDIA_KERNEL_MFU`, `_NVIDIA_KERNEL_ROOFLINE`) |
| BackendKernelEntry schema (G-3) | `python/tessera/compiler/backend_manifest.py` |
| MLIR pass library | `src/compiler/codegen/tessera_gpu_backend_NVIDIA/` (WGMMA, TMA, NVTMADescriptorPass, WarpSpecializationPass) |
| Tile IR FA-4 dialect | `src/compiler/tile_opt_fa4/include/tessera/Dialect/Attn/Attn.td` |
| Lit fixtures (Sprint G-4) | `tests/tessera-ir/phase3/cuda13/` |
| Capability tests | `tests/unit/test_target_toolchain_pins.py` (G-1 + H-1) |

---

## 8. Roadmap — what's hardware-free vs. blocked

### Hardware-free (this batch lands now)
- ✅ Capability matrix (Sprint G-1)
- ✅ Kernel inventory (this doc — Sprint G-2)
- ✅ Schema extension (Sprint G-3)
- ✅ Lit fixtures with PTX patterns (Sprint G-4)
- 🔜 `nvcc -ptx -arch=sm_90a` compile-only validation (Sprint G-6/G-7/G-8)
- 🔜 Named pass pipeline `NVIDIATargetPipeline` in `tessera-opt` (Sprint G-5)
- 🔜 NCCL 2.22 bindings compile + symbol resolution (Sprint G-9)

### Blocked on hardware
- End-to-end execution on H100/B100/Rubin
- Numerical correctness vs CPU reference
- TFLOPS / latency / MFU measurement against §4 targets
- NCCL all-reduce numerical verification across 8x ranks
- Profiler timeline capture (NVTX/CUPTI)
- Multi-rank scaling tests

---

## 9. M7 Visual Complex Analysis (E3 follow-up)

The M7 surface (`tessera.complex.*` — 4 fused ops + 16 long-tail) is
the next NVIDIA kernel family after the matmul/attention baseline.
Today only the **Python reference path** runs on the CPU lanes
(`status="reference"`, `dtypes=("fp32",)` on x86/apple_cpu/cpu). The
backend manifest reserves planned slots across every NVIDIA SM with:

| Op family | Lowering target | Planned dtype matrix | Notes |
|---|---|---|---|
| `complex_mul` / `complex_div` / `complex_exp` / `complex_log` / `complex_sqrt` / `complex_pow` | Pointwise PTX (`cuComplex.h`-style intrinsics on packed `(re, im)` pairs) | **fp32 / fp16 / bf16** | fp16/bf16 are storage-side dtypes; complex math typically uses fp32 accum. |
| `complex_conjugate` / `complex_abs` / `complex_arg` | Pointwise PTX | **fp32 / fp16 / bf16** | Same dtype semantics. |
| `mobius` / `mobius_from_three_points` / `stereographic` | Pointwise PTX with broadcast scalar coefficients | **fp32 / fp16 / bf16** | Möbius matrix coefficients broadcast across the batch. |
| `cross_ratio` / `is_concyclic` / `check_cauchy_riemann` | Small fixed-size projective math + reduction | **fp32 / fp16 / bf16** | Output is scalar / certificate; accum stays fp32. |
| `dz` / `dbar` / `laplacian_2d` | WGMMA-irrelevant; 3×3 stencil kernel (one PTX thread per output element) | **fp32 / fp16 / bf16** | Stencil family — no WGMMA path; can share infra with conv2d depthwise. |
| `conformal_jacobian` / `conformal_energy_on_sphere` | Stencil + reduction | **fp32 / fp16 / bf16** | conformal_energy_on_sphere reduces over S² grid. |

**Important — read the dtype column against the entry's status.** All
20 M7 rows are at `status="planned"` on the NVIDIA targets today; the
fp16/bf16 entries describe what the **future native kernel will
support**, not what runs in the runtime now. The fp32 reference path
on CPU is the only execution path for these ops today.

When the kernels actually land (Phase G follow-on), each row's status
flips `planned → fused` and the dtype matrix above becomes the live
kernel's contract unchanged.
