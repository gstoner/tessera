---
status: Normative
classification: Normative
authority: Canonical standard operator library guidance; defers exact implemented Python behavior to docs/spec/PYTHON_API_SPEC.md and IR semantics to docs/spec/GRAPH_IR_SPEC.md
last_updated: 2026-04-28
---

# Tessera Standard Operator Library

The Tessera Standard Operator Library, or TSOL, is the curated set of portable
operators that users should call through `tessera.ops` and related standard
namespaces. TSOL is more than a convenience catalog: it defines stable operator
semantics, numerical policy expectations, deterministic-mode behavior,
fusion-friendly epilogues, and the compiler lowering contract from Graph IR to
Schedule IR, Tile IR, and Target IR.

For exact Phase 1-3 public Python behavior, use `docs/spec/PYTHON_API_SPEC.md`.
For Graph IR verifier rules and MLIR examples, use
`docs/spec/GRAPH_IR_SPEC.md`. For GPU lowering behavior, use
`docs/spec/LOWERING_PIPELINE_SPEC.md` and `docs/spec/TARGET_IR_SPEC.md`.

## Design Goals

| Goal | Contract |
|------|----------|
| Portable semantics | Operators keep the same shape, dtype, layout, and error behavior across CPU, GPU, and distributed lowering paths unless a backend is explicitly unsupported. |
| Deterministic when requested | `deterministic=True` fixes reduction trees, collective ordering, RNG streams, dropout masks, and schedule choice where the backend can honor the contract. |
| Numerics-first | Operators accept canonical numeric policies for storage dtype, accumulator dtype, rounding, scaling, quantization axis, and deterministic mode. |
| Fusion-friendly | Common epilogues such as bias, activation, residual add, cast, and dropout are represented as canonical operator attributes or helper objects, not ad hoc graph fragments. |
| Autotuned and reproducible | Schedule choices are cached as schedule artifacts keyed by shape, layout, target architecture, numeric policy, movement plan, and tile knobs. |
| IR-native lowering | Every standard operator has a declared owner in the Graph to Schedule to Tile to Target stack, even if the current backend only implements a subset. |

## API Rule

Use the canonical namespace in public documentation:

```python
import tessera

@tessera.jit
def step(A: tessera.Tensor["M", "K"], B: tessera.Tensor["K", "N"]):
    return tessera.ops.matmul(A, B)
```

Examples may locally alias the namespace:

```python
op = tessera.ops
```

The canonical documentation form remains `tessera.ops.<name>`.

## Phase Status

TSOL uses phase labels so the operator catalog can describe the intended
standard library without pretending every backend path is complete.

| Status | Meaning |
|--------|---------|
| Phase 1-3 implemented | Current public behavior exists in the Python surface, Graph IR, or supported lowering path. |
| Phase 4 planned | Distributed, collective, mesh, and communication behavior is planned or partially modeled. |
| Phase 5 planned | Autodiff, checkpointing, optimizer state, quantization workflows, and autotuning expansion. |
| Phase 6 planned | Production runtime ABI, typed wrappers, observability hooks, and broad backend coverage. |
| Spec anchor | Semantics are reserved now so code, docs, and tests converge on one name. |

## Tensor, Dtype, Layout, And Numeric Policy

Standard operators consume and produce tensors with four compiler-visible
properties:

| Property | Examples |
|----------|----------|
| Shape | Rank-N static or symbolic dimensions such as `["B", "S", "D"]`. |
| Dtype | `fp64`, `fp32`, `tf32`, `bf16`, `fp16`, `fp8_e4m3`, `fp8_e5m2`, `fp6_e2m3`, `fp6_e3m2`, `fp4_e2m1`, `nvfp4`, `int8`, `int32`, `bool`. |
| Layout | `row_major`, `col_major`, `nhwc`, `nchw`, `tile(BM, BN, BK)`, `bsr(block_m, block_n, mask)`, sequence layouts such as `NLD` and `BLH`. |
| Distribution | Optional shard, replicate, mesh, and placement metadata. |

Numerical policy is part of the operator contract. Matmul, convolution,
attention, normalization, softmax, quantized weights, and casts should use the
canonical policy shape:

```text
storage dtype + accumulator dtype + rounding mode + scale policy
  + quantization axis + deterministic mode
```

For example, `bf16 @accum(f32)` means BF16 storage with FP32 accumulation and a
BF16 output cast unless the operator explicitly returns another dtype.

## Determinism Contract

When deterministic mode is active, TSOL operators must choose deterministic
implementations or raise `TS_ERR_NONDETERMINISM` if the backend cannot honor the
request.

Deterministic mode governs:

| Area | Required behavior |
|------|-------------------|
| RNG and dropout | Counter-based streams, fixed seed/subsequence assignment, stable mask generation. |
| Reductions | Fixed reduction tree and stable accumulation order where supported. |
| Collectives | Ordered send/recv and stable collective schedule. |
| Autotuning | Reuse an existing schedule artifact or select from deterministic search order. |
| Numeric fast paths | Disable backend paths that violate the requested numeric contract. |

## Effect Mapping

TSOL operators participate in the compiler effect lattice:

```text
pure < random < movement < state < collective < memory < io < top
```

| Effect | Operator examples |
|--------|-------------------|
| `pure` | `matmul`, `conv2d`, `layer_norm`, `rmsnorm`, `softmax`, `gelu`, `fft`, `transpose`, `cast` |
| `random` | `dropout`, `rng_uniform`, `rng_normal` |
| `movement` | `prefetch`, `async_copy`, `pack`, `unpack`, explicit layout movement |
| `state` | KV cache append/read/prune, optimizer state update, rolling-window cache ops |
| `collective` | `all_reduce`, `reduce_scatter`, `all_gather`, `all_to_all`, MoE dispatch/combine transport |
| `memory` | Mutable tensor writes and alias-visible updates |
| `io` | Host I/O, profiler export, replay capture, unknown external calls |

## Operator Catalog

### Linear Algebra

| Operation | Canonical API | Status | Notes |
|-----------|---------------|--------|-------|
| GEMM | `tessera.ops.gemm(A, B, *, epilogue=None)` | Phase 1-3 implemented | Lowers as canonical matmul where compiled. |
| Matmul | `tessera.ops.matmul(A, B, *, epilogue=None)` | Phase 1-3 implemented | Preferred public name; `gemm` remains accepted. |
| Batched GEMM | `tessera.ops.batched_gemm(A, B)` | Spec anchor | Strided or pointer-array batches. |
| Einsum | `tessera.ops.einsum(spec, *tensors)` | Spec anchor | Lowers to contractions, reductions, and layout transforms. |
| Factorized matmul | `tessera.ops.factorized_matmul(A, B, *, rank)` | Spec anchor | Low-rank trade-off operator. |
| Triangular solve | `tessera.ops.tri_solve(A, b, *, lower=True)` | Spec anchor | Backend may fall back to CPU in early phases. |
| Decompositions | `cholesky`, `qr`, `svd` | Spec anchor | Numeric policy must declare accumulator/result dtype. |

### Neural Network Primitives

| Operation | Canonical API | Status | Notes |
|-----------|---------------|--------|-------|
| Conv2D/3D | `tessera.ops.conv2d`, `tessera.ops.conv3d` | Phase 1-3 implemented for `conv2d` stub/IR path | Supports layout and fused epilogue contracts. |
| LayerNorm | `tessera.ops.layer_norm(x, *, eps=1e-5)` | Phase 1-3 implemented | Deterministic reduction when requested. |
| RMSNorm | `tessera.ops.rmsnorm(x, *, eps=1e-5)` | Spec anchor | Safe normalization primitive. |
| Softmax | `tessera.ops.softmax(x, *, axis=-1)` | Phase 1-3 implemented | Numerically stable baseline; safe variants lower to stable kernels. |
| GELU/ReLU/SiLU | `gelu`, `relu`, `silu` | Phase 1-3 implemented for GELU/ReLU | Activation epilogues should fuse. |
| Dropout | `tessera.ops.dropout(x, p, *, rng=None, training=True)` | Phase 1-3 implemented | Random effect; deterministic mode requires stable RNG policy. |
| QKV projection | `tessera.ops.qkv_projection(x, W_qkv)` | Spec anchor | Tensor-parallel friendly projection. |
| FlashAttention | `tessera.ops.flash_attn(Q, K, V, *, scale=None, causal=False, cache=None, dropout_p=0.0)` | Phase 1-3 implemented for tensor inputs; stateful cache path planned | Uses online softmax and schedule artifacts in optimized lowering. |
| RoPE | `tessera.ops.rope(x, theta, *, axes="qk")` | Spec anchor | Rotation policy is part of op attrs. |
| MoE | `tessera.ops.moe`, `moe_dispatch`, `moe_combine` | Phase 4 planned | Transport hooks cover NCCL, NVSHMEM, and DeepEP-style paths. |

### Spectral Operators

| Operation | Canonical API | Status | Notes |
|-----------|---------------|--------|-------|
| FFT/IFFT | `tessera.ops.fft(x, *, axes=None)`, `ifft` | Spec anchor | Spectral dialect support exists; TSOL fixes public spelling. |
| RFFT/IRFFT | `tessera.ops.rfft`, `irfft` | Spec anchor | Real transform variants. |
| STFT/ISTFT | `tessera.ops.stft`, `istft` | Spec anchor | Windowing included in operator semantics. |
| Spectral filter | `tessera.ops.spectral_filter(Xf, Hf)` | Spec anchor | Complex dtype aware. |

### Sparse, Segment, And Graph Operators

| Operation | Canonical API | Status | Notes |
|-----------|---------------|--------|-------|
| COO/CSR SpMM | `tessera.ops.spmm_coo`, `spmm_csr` | Spec anchor | Sparse format metadata must be explicit. |
| SDDMM | `tessera.ops.sddmm(A, B, mask)` | Spec anchor | Useful for sparse attention. |
| Block-sparse matmul | `tessera.ops.bsmm(X, W_bsr)` | Spec anchor | BSR block sizes should align to tensor-core tiles. |
| Segment reduce | `tessera.ops.segment_reduce(x, seg_ids, *, op="sum")` | Spec anchor | Reduction op must declare deterministic behavior. |

### RNG And Initialization

| Operation | Canonical API | Status | Notes |
|-----------|---------------|--------|-------|
| Uniform RNG | `tessera.ops.rng_uniform(shape, *, dtype, seed, lo, hi)` | Spec anchor | Counter-based Philox-style contract. |
| Normal RNG | `tessera.ops.rng_normal(shape, *, dtype, seed, mean, std)` | Spec anchor | Stable stream assignment under deterministic mode. |
| Dropout | `tessera.ops.dropout(x, p, *, rng=None)` | Phase 1-3 implemented | Shares RNG contract. |

### Collectives

| Operation | Canonical API | Status | Notes |
|-----------|---------------|--------|-------|
| All-reduce | `tessera.ops.all_reduce(x, *, axis="dp", op="sum", deterministic=None)` | Phase 4 planned | Current Python path is single-rank no-op. |
| Reduce-scatter | `tessera.ops.reduce_scatter(x, *, axis="dp", op="sum", deterministic=None)` | Phase 4 planned | Should lower through typed async collectives. |
| All-gather | `tessera.ops.all_gather(x, *, axis="dp", deterministic=None)` | Phase 4 planned | Future-aware value semantics. |
| All-to-all | `tessera.ops.all_to_all(x, *, axis, deterministic=None)` | Phase 4 planned | Required for MoE and sequence sharding. |

### Layout And Packing

| Operation | Canonical API | Status | Notes |
|-----------|---------------|--------|-------|
| Transpose | `tessera.ops.transpose(x, perm=None)` | Phase 1-3 implemented | May fold into producer/consumer attrs. |
| Rearrange | `tessera.ops.rearrange(x, layout)` | Spec anchor | Canonical layout transform. |
| Pack/unpack | `tessera.ops.pack(x, layout)`, `unpack(x)` | Spec anchor | Movement effect when materialized. |
| Tile view | `tessera.ops.tile_view(x, BM, BN, BK=None)` | Spec anchor | Feeds Schedule and Tile IR contracts. |

## Fusion And Epilogues

Fused epilogues are standard TSOL objects, not backend-only tricks. Backends may
lower them into one kernel when legal:

```python
y = tessera.ops.matmul(
    x,
    w,
    epilogue={
        "add_bias": True,
        "bias": b,
        "activation": "silu",
        "add_residual": True,
        "residual": skip,
    },
)
```

Canonical epilogue fields are `add_bias`, `bias`, `activation`,
`add_residual`, `residual`, `dropout_p`, `cast_dtype`, and `numeric_policy`.

## Stateful Cache Operators

Stateful model objects such as KV cache, paged optimizer state, and rolling
windows are Graph IR objects. Attention should accept a typed cache object when
stateful decoding is intended instead of treating cached K/V tensors as anonymous
arrays.

```python
y = tessera.ops.flash_attn(q, k, v, cache=kv_cache, causal=True)
```

The cache path has `state` effect. Its movement plan, such as prefetch into
shared memory and overlap with compute, belongs in Schedule IR and is lowered
into Tile IR.

## Schedule Artifact Contract

Autotuned operators should produce or consume a schedule artifact containing:

| Field | Purpose |
|-------|---------|
| Operator name and version | Stable operator identity. |
| Shape and layout | Guards against stale cache reuse. |
| Numeric policy | Captures dtype, accumulator, rounding, scale, and determinism. |
| Target architecture | Example: `sm90`, `gfx942`, `amx_avx512`. |
| Movement plan | Prefetch, async copy, memory spaces, overlap, and staging. |
| Tile knobs | Blocks, warps, stages, vector width, swizzle. |
| Hash | Reproducible key used by replay and production diagnostics. |

## Error Handling

All TSOL operators raise Tessera errors with stable codes:

| Code | Meaning |
|------|---------|
| `TS_ERR_INVALID_ARG` | Invalid value, option, or malformed metadata. |
| `TS_ERR_SHAPE_MISMATCH` | Shape contract failed. |
| `TS_ERR_UNSUPPORTED_DTYPE` | Backend or operator cannot support requested dtype/policy. |
| `TS_ERR_BACKEND_FAILURE` | Wrapped backend failure such as CUDA, ROCm, NCCL, RCCL, or NVSHMEM. |
| `TS_ERR_OOM` | Allocation failed. |
| `TS_ERR_NONDETERMINISM` | Deterministic mode was requested but cannot be honored. |

## Python Type Stubs

The companion file `python/tessera/ops.pyi` provides IDE and static-checking
stubs for the standard operator surface. The stubs intentionally include
spec-anchor operators before every runtime path is implemented so code examples,
docs, and future implementation work converge on one stable API.

## Authoritative References

| Topic | Document |
|-------|----------|
| Public operation signatures implemented today | `docs/spec/PYTHON_API_SPEC.md` |
| Graph IR op semantics | `docs/spec/GRAPH_IR_SPEC.md` |
| Lowering pipeline | `docs/spec/LOWERING_PIPELINE_SPEC.md` |
| Target/Tile IR dialects | `docs/spec/TARGET_IR_SPEC.md` |
| Language and IR semantics | `docs/spec/LANGUAGE_AND_IR_SPEC.md` |
| Memory model | `docs/spec/MEMORY_MODEL_SPEC.md` |
| Tensor layout and data movement | `docs/guides/Tessera_Tensor_Layout_And_Data_Movement_Guide.md` |
| Canonical public API names | `docs/CANONICAL_API.md` |
| Error handling and diagnostics | `docs/guides/Tessera_Error_Handling_And_Diagnostics_Guide.md` |
| QA and reliability behavior | `docs/guides/Tessera_QA_Reliability_Guide.md` |
| Production replay, chaos, and observability | `docs/guides/Tessera_Production_Reliability_And_Chaos_Guide.md` |
