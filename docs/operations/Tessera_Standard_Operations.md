---
status: Normative
classification: Normative
authority: TSOL v1 scope, admission, semantics, numerical policy, determinism, effects, and lowering obligations
scope: Curated portable core; not the exhaustive tessera.ops inventory or a backend-status report
tsol_version: 1
last_updated: 2026-07-13
generated_dashboard: docs/audit/generated/tsol_coverage.md
---

# Tessera Standard Operator Library

The Tessera Standard Operator Library (TSOL) defines the curated, portable core
of `tessera.ops`. It standardizes operator semantics and compiler obligations;
it does not duplicate the exhaustive Python API or claim that every target has
a native kernel.

This document answers three questions:

1. Which operations belong to the TSOL v1 core?
2. Which cross-cutting contracts apply to those operations?
3. What is required to admit, change, or remove an operation?

It deliberately delegates volatile facts to generated or implementation-owned
sources:

| Question | Source of truth |
|----------|-----------------|
| Canonical public spelling | [`docs/CANONICAL_API.md`](../CANONICAL_API.md) |
| Exact Python signatures and implemented behavior | [`docs/spec/PYTHON_API_SPEC.md`](../spec/PYTHON_API_SPEC.md) and `python/tessera/ops.pyi` |
| Graph IR semantics | [`docs/spec/GRAPH_IR_SPEC.md`](../spec/GRAPH_IR_SPEC.md) |
| Contract coverage and lowering status | [Generated TSOL coverage dashboard](../audit/generated/tsol_coverage.md) |
| Native execution proof for an exact target | Backend target maps linked from [`docs/audit/backend/BACKEND_AUDIT.md`](../audit/backend/BACKEND_AUDIT.md) |

## TSOL Scope, Admission, And Versioning

TSOL v1 is a curated compatibility surface, not an inventory of every callable
in `tessera.ops`. Variants, backend helpers, experimental operations, and
workload-specific fused forms may be public without becoming TSOL members.

An operation is admitted to TSOL when all of the following are true:

- it has portable, stable semantics useful across more than one workload or
  backend;
- its public name and exact signature have an API owner;
- it has a primitive-coverage registry row with explicit math, shape, dtype,
  autodiff, sharding, backend, and lowering dispositions;
- it has a reference or eager behavior suitable for conformance testing; and
- Graph IR lowering ownership is defined, including an explicit by-design
  disposition when no native kernel is required.

Adding an operation is backward-compatible within TSOL v1 once the catalog,
API specification, registry, tests, and generated dashboard land together.
Changing established semantics or removing a member requires a deprecation
path or a TSOL major-version change. Moving an operation between categories is
editorial but still requires regenerating the catalog and dashboard.

## Design Goals

| Goal | Contract |
|------|----------|
| Portable semantics | Shape, dtype, layout, effect, and error behavior stay stable across supported lowering paths unless a target is explicitly unsupported. |
| Deterministic when requested | The implementation uses deterministic reduction, collective, RNG, and scheduling choices or rejects the request. |
| Numerics-first | Storage dtype, accumulator dtype, rounding, scaling, quantization axis, and deterministic mode are compiler-visible policy. |
| Fusion-friendly | Common epilogues are structured attributes or helper objects rather than undocumented graph fragments. |
| Autotuned and reproducible | Tuned choices are stored as guarded schedule artifacts. |
| IR-native lowering | Every TSOL operation has an explicit owner through Graph IR, Schedule IR, Tile IR, and Target IR, including by-design terminal dispositions. |

## API Rule

Public documentation uses the canonical namespace:

```python
import tessera

@tessera.jit
def step(A: tessera.Tensor["M", "K"], B: tessera.Tensor["K", "N"]):
    return tessera.ops.matmul(A, B)
```

Examples may locally alias `tessera.ops`, but normative text uses
`tessera.ops.<name>`. This catalog intentionally omits parameter lists: exact
signatures belong to the Python API specification and type stub, where they can
be checked against the runtime without maintaining a second handwritten copy.

## Operator Catalog

The following block is generated from
`python/tessera/compiler/tsol_coverage.py`. Each operation has one category;
for example, `dropout` is a Neural Network Primitive with a `random` effect,
not a duplicate RNG catalog entry.

<!-- BEGIN GENERATED TSOL CATALOG -->
| Category | Canonical operations |
|----------|----------------------|
| Linear Algebra | `tessera.ops.gemm`, `tessera.ops.matmul`, `tessera.ops.batched_gemm`, `tessera.ops.einsum`, `tessera.ops.factorized_matmul`, `tessera.ops.tri_solve`, `tessera.ops.cholesky`, `tessera.ops.qr`, `tessera.ops.svd` |
| Neural Network Primitives | `tessera.ops.conv2d`, `tessera.ops.conv3d`, `tessera.ops.layer_norm`, `tessera.ops.rmsnorm`, `tessera.ops.softmax`, `tessera.ops.gelu`, `tessera.ops.relu`, `tessera.ops.silu`, `tessera.ops.dropout`, `tessera.ops.qkv_projection`, `tessera.ops.flash_attn`, `tessera.ops.rope`, `tessera.ops.moe`, `tessera.ops.moe_dispatch`, `tessera.ops.moe_combine` |
| Spectral Operators | `tessera.ops.fft`, `tessera.ops.ifft`, `tessera.ops.rfft`, `tessera.ops.irfft`, `tessera.ops.stft`, `tessera.ops.istft`, `tessera.ops.spectral_filter` |
| Sparse, Segment, And Graph Operators | `tessera.ops.spmm_coo`, `tessera.ops.spmm_csr`, `tessera.ops.sddmm`, `tessera.ops.bsmm`, `tessera.ops.segment_reduce` |
| RNG And Initialization | `tessera.ops.rng_uniform`, `tessera.ops.rng_normal` |
| Collectives | `tessera.ops.all_reduce`, `tessera.ops.reduce_scatter`, `tessera.ops.all_gather`, `tessera.ops.all_to_all` |
| Layout And Packing | `tessera.ops.transpose`, `tessera.ops.rearrange`, `tessera.ops.pack`, `tessera.ops.unpack`, `tessera.ops.tile_view` |
<!-- END GENERATED TSOL CATALOG -->

## Coverage And Status Authority

This normative document does not carry handwritten coverage glyphs, dated
counts, or target claims. The [generated TSOL coverage
dashboard](../audit/generated/tsol_coverage.md) reports the live registry status
for math semantics, shape, dtype/layout, VJP, JVP, lowering, sharding, and the
aggregate backend axis.

These axes are independent. A complete semantic or lowering contract does not
imply native execution on every target. Likewise, `backend_kernel = complete`
is an all-declared-target aggregate: exact-target proof may exist while that
aggregate remains incomplete. Consult the relevant generated target map before
making a target-specific statement. Structural operations such as `tile_view`
may correctly use a terminal `no_kernel_required` disposition.

## Implementation Status

| Layer | Current interpretation | Authority |
|-------|------------------------|-----------|
| Public/eager surface | Catalog membership promises the standard semantic contract, not that every public helper is a TSOL member. | Python API specification, `ops.pyi`, runtime tests |
| Compiler contract | Every catalog member must have registry and Graph IR lowering dispositions. | Generated TSOL dashboard and primitive registry |
| Exact-target execution | Native, JIT, artifact-only, fallback, and by-design states are target-specific evidence. | Backend target maps and execution fixtures |
| Distributed execution | `MockRankGroup` and single-process behavior support tests; native NCCL/RCCL transport is not yet wired and reports `backend_unavailable`. | `python/tessera/collectives.py` and collective tests |

## Open Actions

| ID | Action | Exit criterion |
|----|--------|----------------|
| TSOL-A1 | Review the broader public `tessera.ops` surface for additional TSOL v1 admissions. | Each admitted operation satisfies the admission checklist and lands with registry, API, tests, and regenerated docs. |
| TSOL-A2 | Reconcile exact signature differences between the runtime, Python API specification, and `ops.pyi`, including RNG seed typing and keyword-only behavior. | Signature conformance is mechanically tested for the TSOL catalog. |
| TSOL-A3 | Complete the deterministic-mode enforcement matrix across reductions, collectives, RNG, and autotuning. | Each TSOL operation either proves deterministic execution or emits the documented nondeterminism error. |
| TSOL-A4 | Wire and verify native NCCL/RCCL collective transport. | Multi-rank hardware execution fixtures pass and the adapters no longer return `backend_unavailable`. |
| TSOL-A5 | Clarify or replace the all-target `backend_kernel` aggregate with an exact-target readiness view. | Readers can distinguish per-target native proof from universal target completeness without interpreting registry internals. |

## Tensor, Dtype, Layout, And Numeric Policy

TSOL operators consume and produce tensors whose shape, dtype, layout, and
optional distribution metadata remain compiler-visible. The canonical dtype
and tensor-attribute vocabulary is owned by the Python and IR specifications;
this document defines the cross-cutting numeric-policy requirement.

Numerical policy is part of an operator contract when relevant:

```text
storage dtype + accumulator dtype + rounding mode + scale policy
  + quantization axis + deterministic mode
```

For example, `bf16 @accum(f32)` means BF16 storage with FP32 accumulation and a
BF16 output cast unless the operation explicitly returns another dtype.

## Determinism Contract

When deterministic mode is active, a TSOL implementation must choose a
deterministic path or raise `TS_ERR_NONDETERMINISM` when the target cannot honor
the request.

| Area | Required behavior |
|------|-------------------|
| RNG and dropout | Counter-based streams, fixed seed/subsequence assignment, and stable mask generation. |
| Reductions | Fixed reduction trees and stable accumulation order where supported. |
| Collectives | Ordered communication and a stable collective schedule. |
| Autotuning | Reuse a compatible artifact or search candidates in deterministic order. |
| Numeric fast paths | Disable paths that violate the requested numeric contract. |

The contract is normative even where enforcement remains an open action; a
backend must not silently claim deterministic execution.

## Effect Mapping

TSOL operators participate in the compiler effect lattice:

```text
pure < random < movement < state < collective < memory < io < top
```

| Effect | Representative operations |
|--------|---------------------------|
| `pure` | `matmul`, `conv2d`, `layer_norm`, `softmax`, `fft`, `transpose` |
| `random` | `dropout`, `rng_uniform`, `rng_normal` |
| `movement` | `prefetch`, `async_copy`, `pack`, `unpack`, explicit layout movement |
| `state` | KV-cache and rolling-window mutations |
| `collective` | `all_reduce`, `reduce_scatter`, `all_gather`, `all_to_all`, MoE transport |
| `memory` | Mutable tensor writes and alias-visible updates |
| `io` | Host I/O, profiler export, replay capture, and unknown external calls |

An operation's catalog category does not determine its effect. This is why
`dropout` appears once in the catalog but still has a `random` effect.

## Fusion And Epilogues

Fused epilogues are structured TSOL concepts, not backend-only conventions.
Backends may lower them into one kernel when legal. Canonical fields include
`add_bias`, `bias`, `activation`, `add_residual`, `residual`, `dropout_p`,
`cast_dtype`, and `numeric_policy`; exact accepted forms belong to the Python
API specification.

## Stateful Cache Operators

Stateful model objects such as KV caches, paged optimizer state, and rolling
windows are Graph IR objects. Attention accepts typed cache state for decoding
rather than treating cached K/V tensors as anonymous arrays. Cache mutation has
a `state` effect; prefetch, overlap, staging, and other movement choices belong
to Schedule IR and Tile IR.

## Schedule Artifact Contract

Autotuned TSOL operations produce or consume a guarded schedule artifact:

| Field | Purpose |
|-------|---------|
| Operator name and version | Stable operation identity. |
| Shape and layout | Prevent stale cache reuse. |
| Numeric policy | Capture dtype, accumulator, rounding, scale, and determinism. |
| Target architecture | Bind evidence to an exact architecture such as `sm90`, `gfx1151`, or `amx_avx512`. |
| Movement plan | Record prefetch, async copy, memory spaces, overlap, and staging. |
| Tile knobs | Record blocks, warps, stages, vector width, and swizzle. |
| Hash | Provide a reproducible replay and diagnostic key. |

## Error Handling

The normative TSOL error families are:

| Code | Meaning |
|------|---------|
| `TS_ERR_INVALID_ARG` | Invalid value, option, or malformed metadata. |
| `TS_ERR_SHAPE_MISMATCH` | Shape contract failed. |
| `TS_ERR_UNSUPPORTED_DTYPE` | The operation or target cannot honor the requested dtype or policy. |
| `TS_ERR_BACKEND_FAILURE` | Wrapped target or transport failure. |
| `TS_ERR_OOM` | Allocation failed. |
| `TS_ERR_NONDETERMINISM` | Deterministic mode was requested but cannot be honored. |

The current Python implementation emits `E_*` values from
`tessera.diagnostics.TesseraErrorCode`. The contract-to-runtime mapping is
registered in `python/tessera/compiler/diagnostic_codes.py` and drift-gated by
`tests/unit/test_diagnostic_code_registry.py`. See the [Error Handling and
Diagnostics Guide](../guides/Tessera_Error_Handling_And_Diagnostics_Guide.md)
for exception structure, recovery, and diagnostics.

## Authoritative References

| Topic | Document |
|-------|----------|
| Public names | [`docs/CANONICAL_API.md`](../CANONICAL_API.md) |
| Exact Python signatures | [`docs/spec/PYTHON_API_SPEC.md`](../spec/PYTHON_API_SPEC.md), `python/tessera/ops.pyi` |
| Graph IR semantics | [`docs/spec/GRAPH_IR_SPEC.md`](../spec/GRAPH_IR_SPEC.md) |
| Lowering pipeline | [`docs/spec/LOWERING_PIPELINE_SPEC.md`](../spec/LOWERING_PIPELINE_SPEC.md) |
| Target and Tile IR | [`docs/spec/TARGET_IR_SPEC.md`](../spec/TARGET_IR_SPEC.md) |
| Language and IR semantics | [`docs/spec/LANGUAGE_AND_IR_SPEC.md`](../spec/LANGUAGE_AND_IR_SPEC.md) |
| Memory model | [`docs/spec/MEMORY_MODEL_SPEC.md`](../spec/MEMORY_MODEL_SPEC.md) |
| Tensor layout and movement | [`docs/guides/Tessera_Tensor_Layout_And_Data_Movement_Guide.md`](../guides/Tessera_Tensor_Layout_And_Data_Movement_Guide.md) |
| Error handling and diagnostics | [`docs/guides/Tessera_Error_Handling_And_Diagnostics_Guide.md`](../guides/Tessera_Error_Handling_And_Diagnostics_Guide.md) |
| Contract coverage | [`docs/audit/generated/tsol_coverage.md`](../audit/generated/tsol_coverage.md) |
| Exact-target backend proof | [`docs/audit/backend/BACKEND_AUDIT.md`](../audit/backend/BACKEND_AUDIT.md) and linked generated target maps |
