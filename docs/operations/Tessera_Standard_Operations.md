---
status: Normative
classification: Normative
authority: Canonical standard operator library guidance; defers exact implemented Python behavior to docs/spec/PYTHON_API_SPEC.md and IR semantics to docs/spec/GRAPH_IR_SPEC.md
last_updated: 2026-05-22
generated_dashboard: docs/audit/generated/tsol_coverage.md
---

> **Status note (2026-05-22 refresh):** The per-op tables below were
> rewritten against `primitive_coverage.py` truth on 2026-05-22.
> Numeric per-axis counts live in the generated dashboard at
> [`docs/audit/generated/tsol_coverage.md`](../audit/generated/tsol_coverage.md).
> When that dashboard drifts from the registry, the
> `tests/unit/test_tsol_coverage.py` gate fails — don't hand-edit
> the status labels here without checking the dashboard first.

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

## Status Vocabulary

The per-op catalog below uses six axes from `primitive_coverage.py`
(math / shape / dtype / VJP / JVP / sharding) plus a single
`backend_kernel` summary.  Glyphs:

| Glyph | Meaning |
|-------|---------|
| ✅ `complete` | Contract is implemented + tested.  Math / shape / dtype / VJP / JVP are typically complete across the entire TSOL surface today. |
| ◐ `partial` | Contract has a closed-form rule + tests on the reference path, but the full target × dtype matrix isn't filled in.  Most often appears on `sharding_rule` (Phase G mesh integration pending) and `backend_kernel` (real hardware proof pending). |
| ◯ `planned` | Spec reserves the name; the contract hasn't been written yet.  Zero TSOL ops today carry this status. |
| – `N/A` | Contract doesn't apply for this op (e.g., RNG ops have no VJP; structural reshape ops have no math semantics distinct from shape). |

**Headline summary (2026-05-22 dashboard):**

- 47 / 47 canonical TSOL ops have a registry entry.
- 47 / 47 are `complete` on math / shape / dtype / lowering.
- 41 / 47 have `complete` VJP; the 6 remaining are RNG / pure-layout
  ops where VJP is N/A.
- 40 / 47 have `complete` JVP; the 7 remaining are dropout + RNG +
  pure-layout ops where JVP is N/A.
- 31 / 47 have `complete` sharding; 16 sit at `partial` pending
  Phase G mesh verification.
- **0 / 47** claim `complete` `backend_kernel` — by registry design
  this requires real NVIDIA / ROCm hardware proof
  (see [backend audit](../audit/backend/BACKEND_AUDIT.md)).
  The `◐ partial` status documents which targets have shipping
  kernels today (Apple GPU + x86 paths are real; the rest are
  artifact-only).

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

| Operation | Canonical API | math | shape | dtype | vjp | jvp | sharding | backend |
|-----------|---------------|------|-------|-------|-----|-----|----------|---------|
| GEMM | `tessera.ops.gemm(A, B, *, epilogue=None)` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |
| Matmul | `tessera.ops.matmul(A, B, *, epilogue=None)` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |
| Batched GEMM | `tessera.ops.batched_gemm(A, B)` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |
| Einsum | `tessera.ops.einsum(spec, *tensors)` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |
| Factorized matmul | `tessera.ops.factorized_matmul(A, B, *, rank)` | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ | ◐ |
| Triangular solve | `tessera.ops.tri_solve(A, b, *, lower=True)` | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ | ◐ |
| Cholesky | `tessera.ops.cholesky(A)` | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ | ◐ |
| QR | `tessera.ops.qr(A)` | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ | ◐ |
| SVD | `tessera.ops.svd(A)` | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ | ◐ |

### Neural Network Primitives

| Operation | Canonical API | math | shape | dtype | vjp | jvp | sharding | backend |
|-----------|---------------|------|-------|-------|-----|-----|----------|---------|
| Conv2D | `tessera.ops.conv2d(x, w, ...)` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |
| Conv3D | `tessera.ops.conv3d(x, w, ...)` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |
| LayerNorm | `tessera.ops.layer_norm(x, *, eps=1e-5)` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |
| RMSNorm | `tessera.ops.rmsnorm(x, *, eps=1e-5)` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |
| Softmax | `tessera.ops.softmax(x, *, axis=-1)` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |
| GELU | `tessera.ops.gelu(x)` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |
| ReLU | `tessera.ops.relu(x)` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |
| SiLU | `tessera.ops.silu(x)` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |
| Dropout | `tessera.ops.dropout(x, p, *, rng=None, training=True)` | ✅ | ✅ | ✅ | ✅ | – | ✅ | ◐ |
| QKV projection | `tessera.ops.qkv_projection(x, W_qkv)` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |
| FlashAttention | `tessera.ops.flash_attn(Q, K, V, *, scale=None, causal=False, cache=None, dropout_p=0.0)` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |
| RoPE | `tessera.ops.rope(x, theta, *, axes="qk")` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |
| MoE | `tessera.ops.moe(x, ...)` | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ | ◐ |
| MoE dispatch | `tessera.ops.moe_dispatch(x, route)` | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ | ◐ |
| MoE combine | `tessera.ops.moe_combine(parts, route)` | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ | ◐ |

### Spectral Operators

| Operation | Canonical API | math | shape | dtype | vjp | jvp | sharding | backend |
|-----------|---------------|------|-------|-------|-----|-----|----------|---------|
| FFT | `tessera.ops.fft(x, *, axes=None)` | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ | ◐ |
| IFFT | `tessera.ops.ifft(x, *, axes=None)` | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ | ◐ |
| RFFT | `tessera.ops.rfft(x, *, axes=None)` | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ | ◐ |
| IRFFT | `tessera.ops.irfft(x, *, axes=None)` | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ | ◐ |
| STFT | `tessera.ops.stft(x, *, n_fft, hop, win)` | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ | ◐ |
| ISTFT | `tessera.ops.istft(Xf, *, n_fft, hop, win)` | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ | ◐ |
| Spectral filter | `tessera.ops.spectral_filter(Xf, Hf)` | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ | ◐ |

### Sparse, Segment, And Graph Operators

| Operation | Canonical API | math | shape | dtype | vjp | jvp | sharding | backend |
|-----------|---------------|------|-------|-------|-----|-----|----------|---------|
| COO SpMM | `tessera.ops.spmm_coo(A, B)` | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ | ◐ |
| CSR SpMM | `tessera.ops.spmm_csr(A, B)` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |
| SDDMM | `tessera.ops.sddmm(A, B, mask)` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |
| Block-sparse matmul | `tessera.ops.bsmm(X, W_bsr)` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |
| Segment reduce | `tessera.ops.segment_reduce(x, seg_ids, *, op="sum")` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |

### RNG And Initialization

| Operation | Canonical API | math | shape | dtype | vjp | jvp | sharding | backend |
|-----------|---------------|------|-------|-------|-----|-----|----------|---------|
| Uniform RNG | `tessera.ops.rng_uniform(shape, *, dtype, seed, lo, hi)` | ✅ | ✅ | ✅ | – | – | ✅ | ◐ |
| Normal RNG | `tessera.ops.rng_normal(shape, *, dtype, seed, mean, std)` | ✅ | ✅ | ✅ | – | – | ✅ | ◐ |
| Dropout | `tessera.ops.dropout(x, p, *, rng=None)` | ✅ | ✅ | ✅ | ✅ | – | ✅ | ◐ |

### Collectives

| Operation | Canonical API | math | shape | dtype | vjp | jvp | sharding | backend |
|-----------|---------------|------|-------|-------|-----|-----|----------|---------|
| All-reduce | `tessera.ops.all_reduce(x, *, axis="dp", op="sum", deterministic=None)` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |
| Reduce-scatter | `tessera.ops.reduce_scatter(x, *, axis="dp", op="sum", deterministic=None)` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |
| All-gather | `tessera.ops.all_gather(x, *, axis="dp", deterministic=None)` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |
| All-to-all | `tessera.ops.all_to_all(x, *, axis, deterministic=None)` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |

Collectives run today on the thread-based `MockRankGroup` for tests
and on `NCCLAdapter` / `RCCLAdapter` for production paths (the
adapters are wired but require real GPU hardware for execution
proof — see [backend audit](../audit/backend/BACKEND_AUDIT.md)).

### Layout And Packing

| Operation | Canonical API | math | shape | dtype | vjp | jvp | sharding | backend |
|-----------|---------------|------|-------|-------|-----|-----|----------|---------|
| Transpose | `tessera.ops.transpose(x, perm=None)` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |
| Rearrange | `tessera.ops.rearrange(x, layout)` | ✅ | ✅ | ✅ | – | – | ✅ | ◐ |
| Pack | `tessera.ops.pack(x, layout)` | ✅ | ✅ | ✅ | – | – | ✅ | ◐ |
| Unpack | `tessera.ops.unpack(x)` | ✅ | ✅ | ✅ | – | – | ✅ | ◐ |
| Tile view | `tessera.ops.tile_view(x, BM, BN, BK=None)` | ✅ | ✅ | ✅ | – | – | ✅ | ◐ |

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

All TSOL operators raise Tessera errors with stable codes.  The
canonical TSOL contract codes:

| Code | Meaning |
|------|---------|
| `TS_ERR_INVALID_ARG` | Invalid value, option, or malformed metadata. |
| `TS_ERR_SHAPE_MISMATCH` | Shape contract failed. |
| `TS_ERR_UNSUPPORTED_DTYPE` | Backend or operator cannot support requested dtype/policy. |
| `TS_ERR_BACKEND_FAILURE` | Wrapped backend failure such as CUDA, ROCm, NCCL, RCCL, or NVSHMEM. |
| `TS_ERR_OOM` | Allocation failed. |
| `TS_ERR_NONDETERMINISM` | Deterministic mode was requested but cannot be honored. |

**Implementation note (TSOL-2, 2026-05-22):** the canonical
`TS_ERR_*` family above is a contract.  Today's Python
implementation raises exceptions from
:class:`tessera.diagnostics.TesseraErrorCode` whose values are
prefixed `E_*` (e.g., `E_SHAPE_MISMATCH`, `E_OOM`,
`E_NONDETERMINISTIC`).  The mapping from `TS_ERR_*` to `E_*` is
captured in the unified diagnostic-code registry at
`python/tessera/compiler/diagnostic_codes.py` (Arch-1 + TSOL-2)
with drift gating at
`tests/unit/test_diagnostic_code_registry.py`.

The registry also covers MLIR-level diagnostic codes (`SYMDIM_*`,
`QUEUE_*`, `LAYOUT_LEGALITY_*`) and the JIT-level outcome codes
(`JIT_*`) — one place to discover every code Tessera emits.

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
