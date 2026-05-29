# Apple GPU kernel inventory

> Last updated: Phase 8.4.7 complete. See [`apple_gpu_overview.md`](apple_gpu_overview.md) for the architecture story.

The runtime dispatches one of these C ABI symbols per Graph IR op or recognized fusion chain. All symbols live in `src/compiler/codegen/Tessera_Apple_Backend/runtime/apple_gpu_runtime.mm` (Darwin) and `apple_gpu_runtime_stub.cpp` (non-Darwin).

## Single-op kernels

| Symbol | Graph IR op | Backend | Phase | Constraints |
|--------|-------------|---------|-------|-------------|
| `tessera_apple_gpu_mps_matmul_f32` | `tessera.matmul` (f32) | MPSMatrixMultiplication | 8.3 | rank-2, static |
| `tessera_apple_gpu_mps_matmul_f16` | `tessera.matmul` (f16) | MPSMatrixMultiplication | 8.4.4 | rank-2, static |
| `tessera_apple_gpu_mps_matmul_bf16` | `tessera.matmul` (bf16) | fp32-conversion + MPS | 8.4.4 | rank-2, static (no native MPS bf16) |
| `tessera_apple_gpu_rope_f32` | `tessera.rope` (f32) | MSL | 8.4.0 | rank-2, K%2==0, X.shape == Theta.shape |
| `tessera_apple_gpu_rope_f16` | `tessera.rope` (f16) | MSL `half` | 8.4.4.1 | rank-2, mixed-precision (fp32 internal) |
| `tessera_apple_gpu_rope_bf16` | `tessera.rope` (bf16) | fp32-conversion | 8.4.4.1 | rank-2, fp32-conversion at boundary |
| `tessera_apple_gpu_softmax_f32` | `tessera.softmax` (f32) | MSL | 8.4.2 | rank-2, axis=-1 |
| `tessera_apple_gpu_softmax_f16` | `tessera.softmax` (f16) | MSL `half` | 8.4.4.1 | rank-2, axis=-1 |
| `tessera_apple_gpu_softmax_bf16` | `tessera.softmax` (bf16) | fp32-conversion | 8.4.4.1 | rank-2, axis=-1 |
| `tessera_apple_gpu_gelu_f32` | `tessera.gelu` (f32) | MSL | 8.4.2 | rank-2, tanh-approximation |
| `tessera_apple_gpu_gelu_f16` | `tessera.gelu` (f16) | MSL `half` | 8.4.4.1 | rank-2 |
| `tessera_apple_gpu_gelu_bf16` | `tessera.gelu` (bf16) | fp32-conversion | 8.4.4.1 | rank-2 |
| `tessera_apple_gpu_flash_attn_f32` | `tessera.flash_attn` (f32) | MSL (online softmax) | 8.4.1 | rank-3, head_dim ≤ 256, optional causal mask |
| `tessera_apple_gpu_flash_attn_f16` | `tessera.flash_attn` (f16) | MSL (mixed precision) | 8.4.4.2 | rank-3, head_dim ≤ 256 |
| `tessera_apple_gpu_flash_attn_bf16` | `tessera.flash_attn` (bf16) | fp32-conversion | 8.4.4.2 | rank-3, head_dim ≤ 256 |

## Fused 2-op kernels

| Symbol | Graph IR chain | Backend | Phase | Constraints |
|--------|----------------|---------|-------|-------------|
| `tessera_apple_gpu_matmul_softmax_f32` | `matmul → softmax` (f32) | MSL fused | 8.4.3 | rank-2, axis=-1, N ≤ 256, single-use intermediate |
| `tessera_apple_gpu_matmul_softmax_f16` | `matmul → softmax` (f16) | MSL fused (mixed precision) | 8.4.4.2 | rank-2, N ≤ 256 |
| `tessera_apple_gpu_matmul_softmax_bf16` | `matmul → softmax` (bf16) | fp32-conversion + MSL | 8.4.4.2 | rank-2, N ≤ 256 |
| **`tessera_apple_gpu_matmul_softmax_tiled_f32`** | `matmul → softmax` (f32, large N) | MSL with threadgroup memory | 8.4.6 | rank-2, axis=-1, N ≤ 8192 |
| `tessera_apple_gpu_matmul_gelu_f32` | `matmul → gelu` (f32) | MSL fused | 8.4.7 | rank-2, N ≤ 256 |
| `tessera_apple_gpu_matmul_rmsnorm_f32` | `matmul → rmsnorm[_safe]` (f32) | MSL fused | 8.4.7 | rank-2, N ≤ 256, eps passed by dispatcher (1e-5 / 1e-6) |

The `matmul_softmax_f32` symbol is a **router** as of Phase 8.4.6: per-thread variant for N ≤ 256, threadgroup-tiled variant for N > 256, reference fallback for N > 8192.

## Fused 3-op kernels

| Symbol | Graph IR chain | Backend | Phase | Constraints |
|--------|----------------|---------|-------|-------------|
| `tessera_apple_gpu_matmul_softmax_matmul_f32` | `matmul → softmax → matmul` (f32) | MSL fused (full attention block) | 8.4.5 | rank-2, N ≤ 256, P ≤ 256, single-use intermediates |
| `tessera_apple_gpu_matmul_softmax_matmul_f16` | `matmul → softmax → matmul` (f16) | MSL fused (mixed precision) | 8.4.5 | rank-2, N ≤ 256, P ≤ 256 |
| `tessera_apple_gpu_matmul_softmax_matmul_bf16` | `matmul → softmax → matmul` (bf16) | fp32-conversion + MSL | 8.4.5 | rank-2, N ≤ 256, P ≤ 256 |

## MetalPerformanceShadersGraph (MPSGraph) lane — Tier-1 + long tail (2026-05-29)

Rather than hand-writing one MSL kernel per pointwise / normalization op, these
symbols route through Apple's **MPSGraph** optimizing graph compiler. One
parametrized runner per shape class covers the Tier-1 activation /
normalization surface (and a broad long tail) and — by composing with the MPS
matmul — completes the f16/bf16 (and large-N) fused MLP / attention chains.
Compute is fp32 internally (inputs cast up, outputs cast down); bf16 upcasts
host-side. None of these carry the per-thread `N ≤ 256` limit.

| Symbol | Graph IR op(s) | Shape class | dtypes |
|--------|----------------|-------------|--------|
| `tessera_apple_gpu_mpsgraph_unary_f32` / `_f16` | `relu`/`sigmoid`(`_safe`)/`tanh`/`softplus`/`silu`/`exp`/`log`/`sqrt`/`rsqrt`/`neg`/`abs` (op-coded) | elementwise, any shape | f32, f16 (bf16 host-upcast) |
| `tessera_apple_gpu_mpsgraph_binary_f32` / `_f16` | `silu_mul` (+ add/sub/mul/div/max/min reserved) | elementwise, any shape | f32, f16 (bf16 host-upcast) |
| `tessera_apple_gpu_layer_norm_f32` / `_f16` | `tessera.layer_norm` | row op over last axis | f32, f16 |
| `tessera_apple_gpu_rmsnorm_gpu_f32` / `_f16` | `tessera.rmsnorm` / `tessera.rmsnorm_safe` | row op over last axis | f32, f16 |
| `tessera_apple_gpu_log_softmax_f32` / `_f16` | `tessera.log_softmax` | row op over last axis | f32, f16 |
| `tessera_apple_gpu_mpsgraph_softmax_f32` / `_f16` | `tessera.softmax` (no N limit) | row op over last axis | f32, f16 |

Op codes (unary / binary) are defined in `apple_gpu_runtime.mm` and mirrored in
`python/tessera/runtime.py` (`_APPLE_GPU_UNARY_OPCODES`). `silu_mul(a, b)` is
`silu(a) * b`; the runtime binary opcode 6 computes `first * silu(second)`, so
the dispatcher passes `(b, a)`.

**Fused-chain dtype completion:** `matmul_gelu` / `matmul_rmsnorm` /
`matmul_softmax` keep their single-kernel f32 fast paths (N ≤ 256, plus the
tiled f32 `matmul_softmax`); outside that envelope (f16/bf16, or large N) the
dispatchers now compose the GPU matmul with an MPSGraph epilogue instead of
falling back to host numpy. The gelu epilogue uses the MPSGraph `gelu` node;
the hand-written MSL gelu kernels were also fixed (2026-05-29) to clamp the
tanh argument to `[-30, 30]` so they no longer overflow to NaN for large
activations (|x| ≳ 16).

**Graph caching:** the MPSGraph lane caches each compiled graph by
`(shape-class, opcode, dtype, shape[, eps, weighted])`, so repeated dispatches
with the same signature reuse one `MPSGraph` and only swap the feed buffers.
`tessera_apple_gpu_mpsgraph_cache_size()` reports the live count (used by tests
to assert reuse).

**Compile-time / MLIR path:** the Tier-1 ops are first-class `tessera` dialect
ops (`silu`/`tanh`/`softplus`/`rmsnorm`/`log_softmax` were registered alongside
the existing `relu`/`sigmoid`/`gelu`/`layer_norm`/`softmax`/`silu_mul`), and
three C++ lowering passes — `tessera-unary-to-apple_gpu`,
`tessera-silu-mul-to-apple_gpu`, `tessera-rowop-to-apple_gpu` — lower them to
the runtime calls inside the `tessera-lower-to-apple_gpu-runtime` pipeline
(the unweighted norms pass a null gamma/beta). Lit-checked by
[tests/tessera-ir/phase8/apple_gpu_tier1_lowering.mlir](../tests/tessera-ir/phase8/apple_gpu_tier1_lowering.mlir).

## Capability + diagnostic symbols

| Symbol | Purpose |
|--------|---------|
| `tessera_apple_gpu_runtime_has_metal` | Returns 1 on Darwin with Metal device available, 0 otherwise |
| `tessera_apple_gpu_runtime_msl_cache_size` | Returns count of cached `MTLComputePipelineState` instances (used by tests to verify cache hits) |

## ABI summary

All kernel symbols share these ABI conventions:

- **Tensor pointers** are `i64` raw pointers at the func.call boundary (extracted via `memref.extract_aligned_pointer_as_index` + `arith.index_cast`)
- **Dimension scalars** are `i32`
- **Scale / eps** are `f32`
- **Boolean flags** (causal) are `i32` (0 or 1)
- For f16/bf16: pointers are `uint16_t*` carrying the bit pattern. `numpy.float16` and `ml_dtypes.bfloat16` are byte-compatible via `.view(np.uint16)`.
- The element type is encoded in the **symbol name only**, not the function signature

## Coverage matrix

|  | f32 | f16 | bf16 |
|---|---|---|---|
| **mps_matmul** | ✅ 8.3 | ✅ 8.4.4 | ✅ 8.4.4 |
| **rope** | ✅ 8.4.0 | ✅ 8.4.4.1 | ✅ 8.4.4.1 |
| **softmax** | ✅ 8.4.2 | ✅ 8.4.4.1 | ✅ 8.4.4.1 |
| **gelu** | ✅ 8.4.2 | ✅ 8.4.4.1 | ✅ 8.4.4.1 |
| **flash_attn** | ✅ 8.4.1 | ✅ 8.4.4.2 | ✅ 8.4.4.2 |
| **matmul_softmax** | ✅ 8.4.3 | ✅ 8.4.4.2 | ✅ 8.4.4.2 |
| **matmul_softmax (tiled, large N)** | ✅ 8.4.6 | — | — |
| **matmul_gelu** | ✅ 8.4.7 | — | — |
| **matmul_rmsnorm** | ✅ 8.4.7 | — | — |
| **matmul_softmax_matmul** | ✅ 8.4.5 | ✅ 8.4.5 | ✅ 8.4.5 |

**9 kernel concepts × dtypes = 26 runtime symbols** sharing one `MetalDeviceContext` and MSL kernel cache.

## Test surface

- **Lit fixtures:** 16 in `tests/tessera-ir/phase8/apple_gpu_*.mlir` — exercise compile-time symbol selection and pipeline composition
- **Python unit tests:** ~80 in `tests/unit/test_apple_backend_roadmap.py` — exercise end-to-end execution, runtime dtype dispatch, ABI shim correctness, fusion gates, MSL cache behavior
- **Benchmark harness:** `benchmarks/apple_gpu/benchmark_fusion.py` — fused vs sequential timing comparison
