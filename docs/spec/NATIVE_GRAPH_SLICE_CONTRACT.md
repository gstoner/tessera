---
last_updated: 2026-10-16
---

# Native Graph slice semantics

Owner: FRONTEND-IR-MEDIUM-1 / LAYOUT-ALG-1 / ROCM-NVFP4-INGEST-1 / E2E-REAL-6.
Synchronization key: ROCM-NATIVE-GATE-UP-VIEWS-20261016.

## Semantic foundation

Both existing public operations, `slice` and `dynamic_slice`, mean the eager
Python expression `x[start:start + size]` independently on each axis, with
unit stride. They preserve rank and element type. Their names do not imply
StableHLO dynamic-slice clamping or that requested sizes equal result extents.
Negative coordinates, clipping, empty outputs and negative end offsets retain
the eager semantics. For example a length-four source sliced with start 2 and
size -3 selects `[2:-1]`, whose length is one.

The frontend uses the canonical Graph result-shape rule for abstract and
ordinary shape-only traces. Explicit numerical tracing remains available. Concrete frontend values are
retained as borrowed basic-index views for downstream dtype/shape inference;
this creates no compact copy or native backend buffer.
Positional start/size attributes use the existing canonical binding table.
Eager integer coercion is preserved, and emitted coordinates must fit signed
i64; oversized coordinates are rejected before serialization can wrap.

C++ Graph declarations verify ranked input/output, rank-preserving signed-i64
coordinate arrays, identical element type and normalized static extents.
`start + size` is computed in signed 65-bit arithmetic before clipping.
Unknown source extents retain unknown result extents rather than asserting
unproved static sizes.

Static structured lowering produces upstream `tensor.extract_slice` with
normalized offsets/counts and unit strides. Activity and pure-effect facts are
preserved. Unconsumed layout/numerical policies require their owning consumer.

## Differentiation

JVP applies the identical slice to the input tangent. The initial native VJP
for static source bounds inserts the output cotangent into a zero source;
it does not reshape a smaller result into the parent. Dynamically typed seed
extents are asserted before scatter. Transpose of dynamically sized sources
and native device AD consumers remain follow-up required.

## Gate/up integration and physical evidence boundary

The named integration case is one FP32 parent `[M, 2I]`, its two disjoint
column slices, and `silu_mul(gate, up)`. It must retain the original Graph SSA,
parent pitch, read-only lifetime, a distinct output allocation and native
Schedule/Tile/Target provenance. No compact host uploads may replace the two
views. The existing typed `tile.buffer_view`, `tile.load_element` and
`tile.store_element` can express this addressing contract.

Scratch diagnostics demonstrate padded row/column-major addressing through
the shared Tile memory materializer, LLVM lowering and CPU execution. They
are not ROCm Graph/Schedule/package, SiLU numerical, GPU or timing proof.
The current ROCm native math contract still requires isolated operands;
its real view consumer, checked program-frame ownership, exact gfx1201
numerical proof and separate producer/consumer timings remain in progress.
Column-major diagnostics add no public physical layout admission.

Full NVFP4 MLP closure additionally needs explicit activation quantization,
down projection and original-checkpoint/model-quality gates. The independent
NVIDIA, Apple and x86 physical contracts inherit no ROCm execution evidence.
